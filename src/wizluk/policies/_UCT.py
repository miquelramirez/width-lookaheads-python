# -*- coding: utf-8 -*-
import time
import random
import copy
import numpy as np
from collections import deque
import wizluk
from wizluk.memory.and_or_graph import OR_Node, AND_Node, AND_OR_Graph
import sys
from wizluk.policies._cost_to_go_policies import random_rollout, knuth_alg, stoch_enum_alg

class UCT(object) :

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
        self._caching = kwargs.get('caching','None')
        self._grayScale = kwargs.get('grayScale','False')
        self._grayScaleSizeX = int(kwargs.get('grayScaleSizeX',30))
        self._grayScaleSizeY = int(kwargs.get('grayScaleSizeY',30))
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
        self.get_action_call = 0
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
        self.get_action_call = 0
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
        self.get_action_call += 1
        if len(self.stateCountAction) < 1 or self._reset_visits_after_each_step:
            self.stateCountAction = np.array([])
            for a in range(env.action_space.n) :
                self.stateCountAction = np.append(self.stateCountAction, np.array([{}]))

        if self.windowEnv is not None :
            self.windowEnv.close()
        no_caching = True
        if self.sim_calls != 0 and (self._caching == "Partial" or self._caching == "Full"):
            no_caching = False
        self.make_root_node(s0, no_caching)
        self.root.r = 0
        budget = self._budget
        n = self.root
        n_history = []
        self.init_sim_calls = self.sim_calls
        if self._atari == "True":
            envuw = env.unwrapped
            env2 = envuw.clone_full_state()
            episode_started_at = env._episode_started_at
            elapsed_steps = env._elapsed_steps
            env._max_episode_seconds = 2**20
        numberOfTimesNoSimCalls = 0
        while not self.root.terminal and budget > 0 and self.sim_calls - self.init_sim_calls < self.sim_budget and n.d <= self._horizon and numberOfTimesNoSimCalls < self.sim_budget * 100:
            # MRJ: this is a critical detail!
            tempSimBudget = self.sim_calls
            t0 = time.perf_counter()
            if self._atari == "True":
                n, n_history = self.treePolicy(env, self.root)
            else :
                env2 = copy.deepcopy(env)
                n, n_history = self.treePolicy(env2, self.root)

            n.visited = True
            n.v = 0
            if self.sim_calls - self.init_sim_calls < self.sim_budget or self._cost_to_go_est == "heuristic" :
                if self._atari == "True":
                    n.v = self.cost_to_go_est(env, n)
                else :
                    n.v = self.cost_to_go_est(env2, n)
            if tempSimBudget == self.sim_calls:
                numberOfTimesNoSimCalls += 1
            else:
                numberOfTimesNoSimCalls = 0

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
        best_Q = float('-inf')
        best_action = None
        candidates = []
        for act, child in n.children.items() :
            for node, reward in child.children: # for caching
                self._exp_graph.register(node)
            if child.Q > best_Q :
                candidates = [act]
                best_Q = child.Q
            elif abs(child.Q - best_Q) < 0.0000001 :
                candidates.append(act)

        best_action = random.choice(candidates)
        return best_action

    def backup_iterative(self, n_history, gamma) :
        last_or_node = None
        firstMember = len(n_history)
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
                if len(n_history) == firstMember and not(n.terminal):
                    n.nv += n.v # Adding cost to go
                elif not(n.terminal) and last_or_node is not None:
                    n.nv += last_or_node.nv * gamma
                last_or_node = n
                n_history.pop()
                continue
            elif isinstance(n, AND_Node):
                assert last_or_node is not None
                assert self.isInChildrenOnce(n, last_or_node)
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

    # Used for assertion tests
    def isInChildrenOnce(self, parentNode: AND_Node, childNode: OR_Node):
        isChildOnce = False
        for child, childReward in parentNode.children:
            if np.array_equal(child._state, childNode._state) and childReward == childNode.r and child.terminal == childNode.terminal:
                if self._atari == "True" and self._caching != "None":
                    if hasattr(childNode, 'restoreStateFrom') and np.array_equal(child.restoreState, childNode.restoreState):
                        isChildOnce = not isChildOnce
                else:
                    isChildOnce = not isChildOnce
        return isChildOnce

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

    def pick_random_unvisited_child(self, env, n : OR_Node, history ) :
        candidates = [ k for k in n.children.keys() if not n.children[k].visited]
        selected = np.random.choice(candidates)
        history.append(n.children[selected])
        if self._atari == "True" and len(n.children[selected].children) != 0 and self._caching != "None":
            elapsed_steps = env._elapsed_steps
            wasRestored = False
            for node, reward in n.children[selected].children:
                if hasattr(node, 'restoreStateFrom') and node.restoreState is not None:
                    break
            if hasattr(node, 'restoreStateFrom') and node.restoreState is not None :
                if node.restoreStateFrom != self.get_action_call and self._caching != "Full": #State is not from this get action call therefore for partial caching don't restore
                    node.restoreState = None
                else:
                    env.unwrapped.restore_full_state(node.restoreState)
                    env._elapsed_steps = elapsed_steps + 1
                    succ = node
                    wasRestored = True

            if not wasRestored:
                next_state, reward, terminal, _ = env.step(selected)
                self.sim_calls += 1
                if np.array_equal(next_state, node._state) and reward == node.r and terminal == node.terminal:
                    succ = node
                else:
                    succ = copy.deepcopy(node)
                    succ.r = reward
                    succ._state = copy.deepcopy(next_state)
                    succ.terminal = terminal
                    n.children[selected].children.add((succ, reward))
                succ.restoreState = env.unwrapped.clone_full_state()
                succ.restoreStateFrom = self.get_action_call
        else:
            next_state, reward, terminal, _ = env.step(selected)
            self.sim_calls += 1
            if self._atari != "True":
                next_state = np.reshape(next_state, [1, np.prod(env.observation_space.shape)])
            succ = OR_Node(next_state,n.d + 1,terminal)
            if n.children[selected].update(reward,succ) :
                if self.tabulate_state_visits:
                    tupOfState = tuple(succ.state[0].tolist())
                    try:
                        succ.num_visits = self.stateCount[tupOfState]
                    except:
                        succ.num_visits = 0
                else:
                    succ.num_visits = 0 # if we get a new successor
                succ.r = reward
            else :
                foundChild = False
                for child in n.children[selected].children :
                    succ1, reward1 = child
                    if reward1 == reward and succ1 == succ :
                        succ = succ1
                        foundChild = True
                assert(foundChild)
            if self._atari == "True" and self._caching != "None":
                succ.restoreState = env.unwrapped.clone_full_state()
                succ.restoreStateFrom = self.get_action_call
        history.append(succ)

        assert self.isInChildrenOnce(n.children[selected], succ)
        n.children[selected].visited = True

        self.max_depth = max( self.max_depth, succ.d)
        return succ, history

    def pick_random_action(self, env) :
        actions = [ a for a in range(env.action_space.n)]
        if len(actions) == 0:
            return None
        return np.random.choice(actions)

    def isFullyExpanded(self, n : OR_Node) :
        candidates = [ k for k in n.children.keys() if not n.children[k].visited]
        return len(candidates) == 0

    def treePolicy(self, env, n : OR_Node) :
        history = deque()
        history.append(n)

        while not n.terminal and self.sim_calls - self.init_sim_calls < self.sim_budget:
            self.expand(env,n)
            if not self.isFullyExpanded(n) :
                # Pick random unsolved child of n
                return self.pick_random_unvisited_child(env, n, history)
            else:
                n, history = self.bestChild(env, n, history)
        return n, history

    def bestChild(self, env, n: OR_Node, history) :
        assert(len(n.children) > 0)
        L = [float('inf') if n.children[k].num_visits == 0 else n.children[k].Q + self._C * np.sqrt(2 * np.log(n.num_visits/n.children[k].num_visits)) for k in n.children.keys()]
        selected = list(n.children.keys())[np.argmax(L)]

        history.append(n.children[selected])
        if self._atari == "True" and len(n.children[selected].children) != 0 and self._caching != "None":
            elapsed_steps = env._elapsed_steps
            envuw = env.unwrapped
            for node, reward in n.children[selected].children:
                if hasattr(node, 'restoreStateFrom') and node.restoreState is not None:
                    break
            wasRestored = False
            if hasattr(node, 'restoreStateFrom') and node.restoreState is not None:
                if node.restoreStateFrom != self.get_action_call and self._caching != "Full": #State is not from this get action call therefore for partial caching don't restore
                    node.restoreState = None
                else:
                    env.unwrapped.restore_full_state(node.restoreState)
                    env._elapsed_steps = elapsed_steps + 1
                    succ = node
                    wasRestored = True

            if not wasRestored:
                assert(False)
        else:
            next_state, reward, terminal, _ = env.step(selected)
            self.sim_calls += 1
            if self._atari != "True":
                next_state = np.reshape(next_state, [1, np.prod(env.observation_space.shape)])
            elif self._caching != "None": # If atari and caching is on
                assert(False)
            succ = OR_Node(next_state,n.d + 1,terminal)
            if n.children[selected].update(reward,succ) :
                if self.tabulate_state_visits:
                    tupOfState = tuple(succ.state[0].tolist())
                    try:
                        succ.num_visits = self.stateCount[tupOfState]
                    except:
                        succ.num_visits = 0
                else:
                    succ.num_visits = 0 # if we get a new successor
                succ.r = reward
            else :
                foundChild = False
                for child in n.children[selected].children :
                    succ1, reward1 = child
                    if reward1 == reward and succ1 == succ :
                        succ = succ1
                        foundChild = True
                assert(foundChild)

        history.append(succ)
        n.children[selected].visited = True

        self.max_depth = max( self.max_depth, succ.d)
        return succ, history

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


    def free_mem(self, old_root: OR_Node , new_root: OR_Node) :
        open = deque()
        open2 = deque()
        open.append(old_root)
        i = 0
        while len(open) > 0 :
            n = open.pop() # top of the stack
            #i += 1
            #wizluk.logger.debug("{}".format(i))
            if isinstance(n, OR_Node):
                n._parents = set()
                if n == new_root :
                    open2.append(n)
                    continue

                for act, child in n.children.items():
                        open.append(child)
                n._children = {}
                del n
                continue
            elif isinstance(n, AND_Node):
                n._parent = None
                for succ, r in n.children :
                        open.append(succ)
                n._children = set()
                del n
                continue
            else :
                assert False

        while len(open2) > 0 :
            n = open2.pop() # top of the stack
            if isinstance(n, OR_Node):

                for act, child in n.children.items():
                        open2.append(child)
                n._d -= 1
                if self._caching != "Full":
                    n.visited = False
                    n.restoreState = None
                continue
            elif isinstance(n, AND_Node):
                for succ, r in n.children :
                        open2.append(succ)
                if self._caching != "Full":
                    n.visited = False
                    n.restoreState = None
                continue
            else :
                assert False

    def make_root_node(self, s, forget=True) :
        n = OR_Node(s, 1)
        if forget :
            self._exp_graph = AND_OR_Graph()

        try :
            n = self._exp_graph.locate(n)
            self.free_mem(self.root, n)
            self.root = n
        except KeyError :
            n.visited = False
            if self.root is not None :
                self.free_mem(self.root, n)
            self._exp_graph.register(n)
            self._exp_graph.add_root(n)
            self.root = n
        self.root._d = 0
        self.current = self.root
