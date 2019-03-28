# -*- coding: utf-8 -*-
import time
import random
import copy
import numpy as np
from collections import deque
import wizluk
from wizluk.memory.and_or_graph import OR_Node, AND_Node, AND_OR_Graph
from wizluk.policies._cost_to_go_policies import random_rollout, knuth_alg, stoch_enum_alg
import sys

class One_Step(object) :

    def __init__(self, **kwargs ) :
        self._budget = float(kwargs.get('budget',1.0))
        self.sim_budget = int(kwargs.get('sim_budget',999999))
        self._cost_to_go_est = kwargs.get('cost_to_go_est', 'random_rollout')
        self._num_rollouts = int(kwargs.get('num_rollouts',1))
        self._C = float(kwargs.get('C', 1.0))
        self._gamma = float(kwargs.get('gamma',1.00))
        self._horizon = int(kwargs.get('horizon',2**20))
        self._number_of_paths_to_consider_for_stoch_enum = int(kwargs.get('number_of_paths_to_consider_for_stoch_enum',2))
        self.tabulate_state_visits = bool(kwargs.get('tabulate_state_visits', False))
        self._reset_visits_after_each_step = bool(kwargs.get('reset_visits_after_each_step_for_tabulation', True))
        self._max_base_policy_evaluations = float(kwargs.get('max_base_policy_evaluations',2**20))
        self._atari = kwargs.get('atari','False')
        self._exp_graph = AND_OR_Graph()
        self._epsilon = 1.0e-5
        self.max_depth = 0
        self.num_visited = 0
        self.num_rollouts = 0
        self.mean_rollout_reward = 0
        self.rollout_depth = 0
        self.rollouts_reached_terminal_state = 0
        self.rollouts_reached_horizon = 0
        self.rollouts_reached_discount_value_smaller_than_epsilon = 0
        self.tree_rollout_backup_runtime= 0
        self.root = None
        self.current = None
        self.windowEnv = None
        self.stateCount = {}
        self.sim_calls = 0
        self.stateCountAction = np.array([])
        self.init_sim_calls = 0
        #random rollout stats
        self.num_random_rollouts = 0
        self.random_rollouts_reached_terminal_state = 0
        self.random_rollouts_reached_horizon = 0
        self.random_rollouts_reached_discount_value_smaller_than_epsilon = 0

        self.base_policy = None
        base_policy_type = kwargs.get('base_policy', 'RandomPolicy')
        self.base_policy = wizluk.policies.create(base_policy_type)

    def reset_statistics(self) :
        self.num_visited = 0
        self.max_depth = 0
        self.max_rollout_depth = 0
        self.num_rollouts = 0
        self.mean_rollout_reward = 0
        self.tree_rollout_backup_runtime = 0
        self.rollout_depth = 0
        self.rollouts_reached_terminal_state = 0
        self.rollouts_reached_horizon = 0
        self.rollouts_reached_discount_value_smaller_than_epsilon = 0
        self.sim_calls = 0
        self.init_sim_calls = 0
        #random rollout stats
        self.num_random_rollouts = 0
        self.random_rollouts_reached_terminal_state = 0
        self.random_rollouts_reached_horizon = 0
        self.random_rollouts_reached_discount_value_smaller_than_epsilon = 0

    def init_statistics(self, df) :
        df['visited'] = []
        df['max_depth'] = []
        df['num_rollouts'] = []
        df['mean_rollout_reward'] = []
        df['tree_rollout_backup_runtime'] = []
        df['ave_rollout_depth'] =[]
        df['rollouts_reached_terminal_state'] =[]
        df['rollouts_reached_horizon'] =[]
        df['rollouts_reached_discount_value_smaller_than_epsilon'] =[]
        df['random_rollouts_reached_terminal_state'] =[]
        df['random_rollouts_reached_horizon'] =[]
        df['random_rollouts_reached_discount_value_smaller_than_epsilon'] =[]
        df['num_random_rollouts'] = []

    def collect_statistics(self, df ) :
        df['visited'] += [ self.num_visited ]
        df['max_depth'] += [ self.max_depth ]
        df['num_rollouts'] += [ self.num_rollouts]
        df['mean_rollout_reward'] += [ self.mean_rollout_reward ]
        df['tree_rollout_backup_runtime'] += [ self.tree_rollout_backup_runtime]
        df['ave_rollout_depth'] += [ self.rollout_depth / max(1, self.num_rollouts)]
        df['rollouts_reached_terminal_state'] += [self.rollouts_reached_terminal_state]
        df['rollouts_reached_horizon'] += [self.rollouts_reached_horizon]
        df['rollouts_reached_discount_value_smaller_than_epsilon'] += [self.rollouts_reached_discount_value_smaller_than_epsilon]
        df['num_random_rollouts'] += [self.num_random_rollouts]
        df['random_rollouts_reached_terminal_state'] += [self.random_rollouts_reached_terminal_state]
        df['random_rollouts_reached_horizon'] += [self.random_rollouts_reached_horizon]
        df['random_rollouts_reached_discount_value_smaller_than_epsilon'] += [self.random_rollouts_reached_discount_value_smaller_than_epsilon]

    def get_action(self, env, s0 ) :
        if len(self.stateCountAction) < 1 or self._reset_visits_after_each_step:
            self.stateCountAction = np.array([])
            for a in range(env.action_space.n) :
                self.stateCountAction = np.append(self.stateCountAction, np.array([{}]))

        if self.windowEnv is not None :
            self.windowEnv.close()
        self.make_root_node(s0)
        self.root.r = 0
        budget = self._budget
        n = self.root
        self.init_sim_calls = self.sim_calls
        actionRollout = random.choice(range(env.action_space.n))
        if self._atari == "True":
            envuw = env.unwrapped
            env2 = envuw.clone_full_state()
            episode_started_at = env._episode_started_at
            elapsed_steps = env._elapsed_steps
            env._max_episode_seconds = 2**20
        while not self.root.terminal and budget > 0 and self.sim_calls - self.init_sim_calls < self.sim_budget:
            t0 = time.perf_counter()
            if self._atari == "True":
                action = actionRollout % env.action_space.n
                n, n_history = self.treePolicy(env, self.root, action)
            else :
                env2 = copy.deepcopy(env)
                action = actionRollout % env2.action_space.n
                n, n_history = self.treePolicy(env2, self.root, action)

            actionRollout += 1
            n.visited = True
            if self.sim_calls - self.init_sim_calls < self.sim_budget or self._cost_to_go_est == "heuristic":
                n.num_rollouts += 1
                if self._atari == "True":
                    n.v += (self.cost_to_go_est(env, n) - n.v) / n.num_rollouts #taking ave of rollouts
                else :
                    n.v += (self.cost_to_go_est(env2, n) - n.v) / n.num_rollouts #taking ave of rollouts

            self.backup_iterative(n_history, self._gamma)
            tf = time.perf_counter()
            runtime = tf - t0
            budget -= runtime
            self.tree_rollout_backup_runtime += runtime
            if self._atari == "True":
                envuw.restore_full_state(env2)
                env._episode_started_at = episode_started_at
                env._elapsed_steps = elapsed_steps
        return self.select_best( self.root )

    def select_best( self, n : OR_Node ) :
        wizluk.logger.debug("OneStep: Selecting best action: ")
        best_Q = float('-inf')
        best_action = None
        candidates = []
        for act, child in n.children.items() :
            if child.Q > best_Q :
                candidates = [act]
                best_Q = child.Q
            elif abs(child.Q - best_Q) < 0.0000001 :
                candidates.append(act)

        best_action = random.choice(candidates)
        return best_action

    def backup_iterative(self, n_history, gamma) :
        last_or_node = None
        while len(n_history) > 0 :
            n = n_history[-1] # top of the stack
            n.visited = True
            if isinstance(n, OR_Node):
                if self.tabulate_state_visits:
                    try:
                        tupOfState = tuple(n.state[0].tolist())
                        self.stateCount[tupOfState] += 1
                    except:
                        tupOfState = tuple(n.state[0].tolist())
                        self.stateCount[tupOfState] = 1
                    tupOfState = tuple(n.state[0].tolist())
                    n.num_visits = self.stateCount[tupOfState]
                else:
                    n.num_visits += 1
                n.nv = n.r
                if len(n.children) == 0 and not(n.terminal):
                    n.nv += n.v # Adding cost to go
                elif not(n.terminal) :
                    n.nv += last_or_node.nv * gamma
                last_or_node = n
                n_history.pop()
                continue
            elif isinstance(n, AND_Node):
                #wizluk.logger.debug("back up AND")
                assert last_or_node is not None
                if self.tabulate_state_visits:
                    try:
                        self.stateCountAction[n._action][tuple(n.state[0].tolist())] += 1
                    except:
                        self.stateCountAction[n._action][tuple(n.state[0].tolist())] = 1
                    n.num_visits  = self.stateCountAction[n._action][tuple(n.state[0].tolist())]
                else:
                    n.num_visits += 1

                if n.num_visits == 1:
                    n.Q = last_or_node.nv
                else:
                    n.Q += (last_or_node.nv - n.Q) / n.num_visits
                n_history.pop()
                continue
            else :
                assert False


    def expand(self, env, n : OR_Node ) :
        """
            Construct AND nodes por each of the actions applicable
            on state s(n)
        """
        if len(n.children) == 0 :
            if self.tabulate_state_visits:
                tupOfState = tuple(n.state[0].tolist())
                try:
                    n.num_visits = self.stateCount[tupOfState]
                except:
                    n.num_visits = 0
            else:
                n.num_visits = 0
            #wizluk.logger.debug("expanded out a node")
            for a in range(env.action_space.n) :
                and_node = AND_Node(a,n)
                if self.tabulate_state_visits:
                    try:
                        and_node.num_visits = self.stateCountAction[a][tupOfState]
                    except:
                        and_node.num_visits = 0
                else:
                    and_node.num_visits = 0
                and_node.Q = float('-inf')
                and_node.visited = False

    def pick_action(self, env, n : OR_Node, action, history ) :
        history.append(n.children[action])
        next_state, reward, terminal, _ = env.step(action)
        self.sim_calls += 1
        next_state = np.reshape(next_state, [1, np.prod(env.observation_space.shape)])
        succ = OR_Node(next_state,n.d + 1,terminal)
        if n.children[action].update(reward,succ) :
            if self.tabulate_state_visits:
                tupOfState = tuple(succ.state[0].tolist())
                try:
                    succ.num_visits = self.stateCount[tupOfState]
                except:
                    succ.num_visits = 0
            else:
                succ.num_visits = 0 # if we get a new successor
            succ.r = reward
            self._exp_graph.update(n, action, reward, succ)
            node = succ
            node.v = 0
            node.num_rollouts = 0
        else :
            for child in n.children[action].children :
                succ1, reward1 = child
                if reward1 == reward and succ1 == succ :
                    #wizluk.logger.debug("the state exists")
                    node = succ1
        history.append(node)
        n.children[action].visited = True

        self.max_depth = max( self.max_depth, node.d)
        return node, history

    def isFullyExpanded(self, n : OR_Node) :
        candidates = [ k for k in n.children.keys() if not n.children[k].visited]
        return len(candidates) == 0

    def treePolicy(self, env, n : OR_Node, action) :
        history = deque()
        history.append(n)

        if self.sim_calls - self.init_sim_calls < self.sim_budget:
            self.expand(env,n)
            return self.pick_action(env, n, action, history)
        return n, history

    def cost_to_go_est(self, env, n: OR_Node):
        if self._cost_to_go_est == "random_rollout" :
            return random_rollout(self, env, n)
        elif self._cost_to_go_est == "knuth" :
            return knuth_alg(self, env, n)
        elif self._cost_to_go_est == "stochastic_enum" :
            return stoch_enum_alg(self, env, n)
        elif self._cost_to_go_est == "heuristic" :
            return env.unwrapped.getAdmissibleHeuristic()
        return 0

    def make_root_node(self, s, forget=True) :
        n = OR_Node(s, 0)
        if forget :
            self._exp_graph = AND_OR_Graph()

        try :
            n = self._exp_graph.locate(n)
            self.root = n
            #wizluk.logger.debug("Root node already considered")
        except KeyError :
            n.visited = False
            self._exp_graph.register(n)
            self._exp_graph.add_root(n)
            self.root = n
            #wizluk.logger.debug("New root node ")
        self.current = self.root
