# -*- coding: utf-8 -*-
import time
import random
import copy
import numpy as np
from collections import deque
import wizluk
from wizluk.heuristics.novelty import ClassicTabularNovelty, DepthBasedTabularNovelty, DepthBasedTabularNoveltyOptimised
from wizluk.memory.and_or_graph import OR_Node, AND_Node, AND_OR_Graph
from wizluk.policies._cost_to_go_policies import random_rollout, knuth_alg, stoch_enum_alg
import sys
import cv2
import ctypes
#import gc

class ClassicRollout(object):
    """
        Rollout strategy for IW lookaheads. This class modifies the rollout
        strategy discussed in "Planning with Pixels in (Almost) Real-time" by
        Bandres, Bonet and Geffner, AAAI-18 to use the classical definition of
        novelty as described in "Classical Planning with Simulators: Results on
        the Atari Video Games" by Lipovetzky, Ramirez and Geffner.
    """
    def __init__(self):
        pass

    def initialize_feature_table(self, lookahead, n : OR_Node) :
        if lookahead.emptyFeatureTable is None:
            lookahead.root.feature_table = ClassicTabularNovelty()
            for f in lookahead._features :
                lookahead.root.feature_table.add_feature(f)
            lookahead.emptyFeatureTable = copy.deepcopy(lookahead.root.feature_table)
        else:
            lookahead.root.feature_table = copy.deepcopy(lookahead.emptyFeatureTable)

        if(lookahead._include_root_in_novelty_table == 'True') :
            lookahead.root.feature_table.update_feature_table((n.state[0]))


    def rollout(self, lookahead, env, n : OR_Node):
        while not n.SOLVED and lookahead.sim_calls - lookahead.init_sim_calls < lookahead.sim_budget:
            lookahead.rollout_depth += 1
            t0 = time.perf_counter()
            lookahead.expand(env,n)

            # Pick random unsolved child of n
            t0 = time.perf_counter()
            n = lookahead.pick_random_unsolved_child(env, n)
            tf = time.perf_counter()
            lookahead.rollout_runtime_pick_random_unsolved += tf - t0
            if n.terminal :
                n.visited = True
                lookahead.num_visited += 1
                lookahead.solve_and_propagate_labels(n)
                if lookahead.worst_terminal_accumulated_reward is None or lookahead.worst_terminal_accumulated_reward > n.accumulated_reward :
                    lookahead.worst_terminal_accumulated_reward = n.accumulated_reward
                break
            t0 = time.perf_counter()
            is_novel = lookahead.root.feature_table.is_novel((n.state[0]))
            tf = time.perf_counter()
            lookahead.rollout_runtime_is_novel += tf - t0
            if is_novel :
                n.visited = True
                lookahead.num_visited += 1
                lookahead.root.feature_table.update_feature_table((n.state[0]))
            elif not n.visited :
                #pruned as is not novel
                n.randomV = lookahead.cost_to_go_est(env, n)
                lookahead.solve_and_propagate_labels(n)
                break
        if not n.SOLVED and lookahead._pruned_state_strategy == "heuristic":
            # If didn't finish rollout due to computational budget apply heuistic value
            n.randomV = lookahead.cost_to_go_est(env, n)


class DepthBasedRollout(object):
    """
        Rollout strategy for IW lookaheads. This class encapsulates the rollout
        strategy discussed in "Planning with Pixels in (Almost) Real-time" by
        Bandres, Bonet and Geffner, AAAI-18.
    """
    def __init__(self):
        pass

    def initialize_feature_table(self, lookahead, n : OR_Node) :
        if lookahead.emptyFeatureTable is None:
            if lookahead._useOptimisedDepthTable =='True':
                lookahead.root.feature_table = DepthBasedTabularNoveltyOptimised()
            else:
                lookahead.root.feature_table = DepthBasedTabularNovelty()
            for f in lookahead._features :
                lookahead.root.feature_table.add_feature(f)
            lookahead.emptyFeatureTable = copy.deepcopy(lookahead.root.feature_table)
        else:
            lookahead.root.feature_table = copy.deepcopy(lookahead.emptyFeatureTable)

        if(lookahead._include_root_in_novelty_table == 'True') :
            lookahead.root.feature_table.update_feature_table((n.state[0],n.d))


    def rollout(self, lookahead, env, n : OR_Node):
        while not n.SOLVED and lookahead.sim_calls - lookahead.init_sim_calls < lookahead.sim_budget:
            lookahead.rollout_depth += 1
            lookahead.expand(env,n)
            # Pick random unsolved child of n
            n = lookahead.pick_random_unsolved_child(env, n)
            if n.terminal :
                n.visited = True
                lookahead.num_visited += 1
                lookahead.solve_and_propagate_labels(n)
                if lookahead.worst_terminal_accumulated_reward is None or lookahead.worst_terminal_accumulated_reward > n.accumulated_reward :
                    lookahead.worst_terminal_accumulated_reward = n.accumulated_reward
                break
            f, v, rank, old_depth = lookahead.root.feature_table.get_novel_feature((n.state[0],n.d))
            if n.d < old_depth :
                n.visited = True
                lookahead.num_visited += 1
                lookahead.root.feature_table.update_feature_table((n.state[0],n.d))
            elif not n.visited and n.d >= old_depth :
                n.visited = True
                lookahead.num_visited += 1
                #pruned as is not novel
                n.randomV = lookahead.cost_to_go_est(env, n)

                lookahead.solve_and_propagate_labels(n)
                break
            elif n.visited and old_depth < n.d :
                #pruned as is not novel
                n.randomV = lookahead.cost_to_go_est(env, n)
                n._children = {}
                lookahead.solve_and_propagate_labels(n)
                break

        if not n.SOLVED and lookahead._pruned_state_strategy == "heuristic": #If didn't finish rollout due to computational budget apply heuistic value
            n.randomV = lookahead.cost_to_go_est(env, n)


class IW_Rollout(object) :

    def __init__(self, **kwargs ) :
        self._features = kwargs.get('features')
        self._representation = kwargs.get('representation')
        self._budget = float(kwargs.get('budget',1.0))
        self.sim_budget = int(kwargs.get('sim_budget',999999))
        self._novelty_definition = kwargs.get('novelty_definition','depth')
        self._pruned_state_strategy = kwargs.get('pruned_state_strategy', 'None')
        self._gamma = float(kwargs.get('gamma',1.00))
        self._include_root_in_novelty_table = kwargs.get('include_root_in_novelty_table','False')
        self._caching = kwargs.get('caching','None')
        base_policy_type = kwargs.get('base_policy', 'RandomPolicy')
        self._useOptimisedDepthTable = kwargs.get('use_optimised_table', 'False') #Only for depth novelty and when features are state variables for width = 1

        #for sample trajectories for cost-to-go-approx
        self._num_rollouts = int(kwargs.get('num_rollouts',1))
        self._horizon = int(kwargs.get('horizon',2**20))
        self._number_of_paths_to_consider_for_stoch_enum = int(kwargs.get('number_of_paths_to_consider_for_stoch_enum',2))
        self._max_base_policy_evaluations = float(kwargs.get('max_base_policy_evaluations',2**20))

        #for atari domains
        self._atari = kwargs.get('atari','False')
        self._grayScale = kwargs.get('grayScale','False')
        self._grayScaleSizeX = int(kwargs.get('grayScaleSizeX',30))
        self._grayScaleSizeY = int(kwargs.get('grayScaleSizeY',30))

        self._exp_graph = AND_OR_Graph()
        self._epsilon = 1.0e-5
        self.max_depth = 0
        self.num_visited = 0
        self.num_solved = 0
        self.backup_runtime = 0
        self.num_rollouts = 0
        self.rollout_runtime = 0
        self.rollout_runtime_sim = 0
        self.rollout_runtime_is_novel = 0
        self.rollout_runtime_pick_random_unsolved = 0
        self.new_root_node_runtime = 0
        self.worst_terminal_accumulated_reward = None

        #random rollout stats
        self.num_random_rollouts = 0
        self.random_rollouts_reached_terminal_state = 0
        self.random_rollouts_reached_horizon = 0
        self.random_rollouts_reached_discount_value_smaller_than_epsilon = 0
        self.sim_calls = 0
        self.init_sim_calls = 0
        self.previousScreen = None
        self.emptyFeatureTable = None

        self.base_policy = None
        self.base_policy = wizluk.policies.create(base_policy_type)

        self.rollout_depth = 0
        self._exp_graph = AND_OR_Graph()
        self.root = None
        self.current = None
        if(self._novelty_definition == "classic") :
            self.strategy = ClassicRollout()
        elif(self._novelty_definition == "depth") :
            self.strategy = DepthBasedRollout()
        else :
            # @TODO introduce a new Error class for this
            print("Error: novelty definition {} does not exist".format(self._novelty_definition))
            sys.exit(1)


    def reset_statistics(self) :
        self.num_visited = 0
        self.num_solved = 0
        self.max_depth = 0
        self.backup_runtime = 0
        self.num_rollouts = 0
        self.rollout_runtime = 0
        self.new_root_node_runtime = 0
        self.rollout_depth = 0
        self.sim_calls = 0
        self.init_sim_calls = 0
        self.rollout_runtime_sim = 0
        self.rollout_runtime_is_novel = 0
        self.rollout_runtime_pick_random_unsolved = 0
        self.previousScreen = None # needed for b-post features
        self.emptyFeatureTable = None
        #random rollout stats
        self.num_random_rollouts = 0
        self.random_rollouts_reached_terminal_state = 0
        self.random_rollouts_reached_horizon = 0
        self.random_rollouts_reached_discount_value_smaller_than_epsilon = 0


    def init_statistics(self, df) :
        df['solved'] = []
        df['visited'] = []
        df['max_depth'] = []
        df['backup_runtime'] = []
        df['num_rollouts'] = []
        df['rollout_runtime'] = []
        df['rollout_runtime_sim'] = []
        df['rollout_runtime_is_novel'] = []
        df['rollout_runtime_pick_random_unsolved'] = []
        df['new_root_node_runtime'] = []
        df['ave_rollout_depth'] = []
        df['random_rollouts_reached_terminal_state'] =[]
        df['random_rollouts_reached_horizon'] =[]
        df['random_rollouts_reached_discount_value_smaller_than_epsilon'] =[]
        df['num_random_rollouts'] = []
        df['sim_calls'] = []

    def collect_statistics(self, df ) :
        df['solved'] += [ self.num_solved  ]
        df['visited'] += [ self.num_visited ]
        df['max_depth'] += [ self.max_depth ]
        df['backup_runtime'] += [ self.backup_runtime ]
        df['num_rollouts'] += [ self.num_rollouts]
        df['rollout_runtime'] += [ self.rollout_runtime]
        df['rollout_runtime_sim'] += [self.rollout_runtime_sim]
        df['rollout_runtime_is_novel'] += [self.rollout_runtime_is_novel]
        df['rollout_runtime_pick_random_unsolved'] += [self.rollout_runtime_pick_random_unsolved]
        df['new_root_node_runtime'] += [ self.new_root_node_runtime]
        df['ave_rollout_depth'] += [ self.rollout_depth / max(1, self.num_rollouts)]
        df['num_random_rollouts'] += [self.num_random_rollouts]
        df['random_rollouts_reached_terminal_state'] += [self.random_rollouts_reached_terminal_state]
        df['random_rollouts_reached_horizon'] += [self.random_rollouts_reached_horizon]
        df['random_rollouts_reached_discount_value_smaller_than_epsilon'] += [self.random_rollouts_reached_discount_value_smaller_than_epsilon]
        df['sim_calls'] += [ self.sim_calls ]

    def convertScreenToGrayCompressed(self, state):
        return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (self._grayScaleSizeX, self._grayScaleSizeY)).astype(np.uint8)

    def get_action(self, env, s0 ) :
        t0 = time.perf_counter()

        no_caching = True
        if self.num_visited != 0 and (self._caching == "Partial" or self._caching == "Full"):
            no_caching = False

        if self._representation is not None :
            screen = env.unwrapped.ale.getScreen()
            np.set_printoptions(threshold=np.nan)
            previousScreen = None
            previouslength = 0
            if(self.previousScreen is not None):
                previousScreen = self.previousScreen
                previouslength = previousScreen.size

            state_flat = self._representation.getActiveFeatures(screen, previousScreen, previouslength)


            state_flat = np.reshape(state_flat, [1, len(state_flat)])
            self.previousScreen = state_flat
        elif self._grayScale == "True":
            state_flat = np.reshape(self.convertScreenToGrayCompressed(s0), [1, self._grayScaleSizeX * self._grayScaleSizeY])
        else :
            state_flat = np.reshape(s0, [1, np.prod(env.observation_space.shape)])
        self.make_root_node(state_flat, no_caching)

        self.expand(env, self.root)
        tf = time.perf_counter()
        self.new_root_node_runtime += tf - t0
        budget = self._budget
        self.worst_terminal_accumulated_reward = None
        num_rollouts_for_action = 0
        self.init_sim_calls = self.sim_calls

        if self._atari == "True":
            envuw = env.unwrapped
            env2 = envuw.clone_full_state()
            episode_started_at = env._episode_started_at
            elapsed_steps = env._elapsed_steps
            env._max_episode_seconds = 2**20
        wizluk.logger.debug("ep steps {}".format(env._elapsed_steps))
        numberOfTimesNoSimCalls = 0
        while not self.root.SOLVED and budget > 0 and self.sim_calls - self.init_sim_calls < self.sim_budget and self._max_base_policy_evaluations > num_rollouts_for_action  and numberOfTimesNoSimCalls < self.sim_budget * 100:
            tempSimBudget = self.sim_calls
            t0 = time.perf_counter()
            if self._atari == "True":
                self.rollout(env, self.root)
            else :
                env2 = copy.deepcopy(env)
                self.rollout(env2, self.root)

            tf = time.perf_counter()
            runtime = tf - t0
            budget -= runtime
            self.rollout_runtime += runtime
            num_rollouts_for_action += 1
            self.num_rollouts += 1

            if self._atari == "True":
                envuw.restore_full_state(env2)
                env._episode_started_at = episode_started_at
                env._elapsed_steps = elapsed_steps

            if tempSimBudget == self.sim_calls:
                numberOfTimesNoSimCalls += 1
            else:
                numberOfTimesNoSimCalls = 0

        wizluk.logger.debug("rollouts {}".format(num_rollouts_for_action))
        t0 = time.perf_counter()
        self.backup_iterative(self._gamma)
        tf = time.perf_counter()
        runtime = tf - t0
        self.backup_runtime += runtime

        return self.select_best( self.root )


    def select_best_trajectory(self) :
        trajectory = []
        n = self.root
        while True :
            s1 = n.state
            try :
                a = n.best_action
            except AttributeError :
                wizluk.logger.debug('Trajectory ended abruptly, SOLVED: {} # children: {}'.format(n.SOLVED, len(n.children)))
                break
            best_succ, r = n.children[n.best_action].best_child
            s2 = best_succ.state
            t = best_succ.terminal
            trajectory.append((n.state, n.best_action, r, s2, best_succ.terminal))
            if best_succ.terminal : break
            n = best_succ
        return trajectory

    def backup_iterative(self, gamma) :
        open = deque()
        backed_up_OR = set()
        backed_up_AND = set()
        open.append(self.root)
        while len(open) > 0 :
            n = open[-1] # top of the stack
            if isinstance(n, OR_Node):
                if len(n.children) == 0 or n.terminal :
                    n.V = 0.0
                    if not n.terminal :
                        if self._pruned_state_strategy in ["random_rollout", "knuth", "stochastic_enum", "heuristic"] and hasattr(n,'randomV') :
                            n.V = n.randomV
                        elif self._pruned_state_strategy == "minus_worst_terminal" and not (self.worst_terminal_accumulated_reward is None) :
                            n.V = - np.abs(self.worst_terminal_accumulated_reward)
                        elif self._pruned_state_strategy == "add_large_negative_reward":
                            n.V = -2**20

                    n._d += -1
                    n.SOLVED = False
                    if self._caching != "Full":
                        n.restoreState = None
                    backed_up_OR.add(n)
                    open.pop()
                    continue
                all = True
                for act, child in n.children.items():
                    if not child in backed_up_AND :
                        all = False
                        open.append(child)
                        break
                if not all :
                    continue
                best_child_value = float('-inf')
                for act, child in n.children.items() :
                    if child.Q > best_child_value :
                        n.best_action = act
                        best_child_value = child.Q
                n.V = best_child_value
                if self._caching != "Full":
                    n.restoreState = None
                n.SOLVED = False
                n._d += -1
                backed_up_OR.add(n)
                open.pop()
                continue
            elif isinstance(n, AND_Node):
                n.Q = float('-inf')
                all = True
                for succ, r in n.children :
                    if not succ in backed_up_OR :
                        all = False
                        open.append(succ)
                        break
                if not all : continue
                best_q_value = float('-inf')
                numberOfVistsToChild = 0
                for succ, r in n.children :
                    try :
                        q = r + gamma * succ.V
                    except AttributeError as e :
                        for x in backed_up_OR:
                            if x == succ:
                                succ.V = x.V
                        q = r + gamma * succ.V

                    if q > best_q_value :
                        best_q_value = q
                        n.best_child = (succ, r)
                    numberOfVistsToChild += succ.num_visits
                    n.Q = (q * succ.num_visits)/n.num_visits
                assert(n.num_visits == numberOfVistsToChild)
                n.SOLVED = False
                backed_up_AND.add(n)
                open.pop()
                continue
            else :
                assert False

    def select_best( self, n : OR_Node ) :
        wizluk.logger.debug("IW Rollout: Selecting best action: ")
        best_Q = float('-inf')
        best_action = None
        candidates = []
        for act, child in n.children.items() :
            for node, reward in child.children:
                self._exp_graph.register(node) #register all nodes to tree for caching
            if child.Q > best_Q :
                candidates = [act]
                best_Q = child.Q
            elif abs(child.Q - best_Q) < 0.0000001 :
                candidates.append(act)

        best_action = random.choice(candidates)
        return best_action

    def update_table(self, n: OR_Node):
        open = deque()
        open.append(n)
        i = 0
        while len(open) > 0 :
            n = open.pop() # top of the stack
            if isinstance(n, OR_Node):
                for act, child in n.children.items():
                        open.append(child)
                self.root.feature_table.update_feature_table((n.state[0], n.d))

                n.SOLVED = False
                continue
            elif isinstance(n, AND_Node):
                for succ, r in n.children :
                        open.append(succ)
                n.SOLVED = False
                continue
            else :
                assert False

    def free_mem(self, old_root: OR_Node , new_root: OR_Node) :
        open = deque()
        open2 = deque()
        open.append(old_root)
        i = 0
        while len(open) > 0 :
            n = open.pop() # top of the stack
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
                if self._caching != "Full" and n != new_root:
                    n.visited = False
                continue
            elif isinstance(n, AND_Node):
                for succ, r in n.children :
                        open2.append(succ)
                if self._caching != "Full":
                    n.visited = False
                continue
            else :
                assert False


    def make_root_node(self, s, forget=True) :
        n = OR_Node(s, 0)
        n.accumulated_reward = 0
        n.num_visits = 0
        if forget :
            self._exp_graph = AND_OR_Graph()
            #gc.collect()

        if self.root is not None:
            del self.root.feature_table

        try :
            n = self._exp_graph.locate(n)
            n.SOLVED = False
            self.free_mem(self.root, n)
            #gc.collect()
            self.root = n
            self.strategy.initialize_feature_table(self, n)
            wizluk.logger.debug("Root node already considered")
        except KeyError :
            n.SOLVED = False
            n.visited = False
            if self.root is not None :
                self.free_mem(self.root, n)
                #gc.collect()
            self._exp_graph.register(n)
            self._exp_graph.add_root(n)
            self.root = n
            self.strategy.initialize_feature_table(self, n)
            wizluk.logger.debug("New root node ")
        self.current = self.root

    def is_current_solved(self) :
        return self.current.SOLVED

    def expand_current(self, env ) :
        self.expand(env, self.current)

    def pick_action(self, agent ) :
        candidates = [ k for k in self.current.children.keys() if not self.current.children[k].SOLVED]
        return agent.base_policy.filter_and_select_action(agent, self.current.state, candidates)

    def is_current_terminal(self) :
        self.current.visited = True
        self.num_visited += 1
        self.solve_and_propagate_labels(self.current)
        return self.current.terminal

    def can_continue(self) :
        n = self.current
        f, v, rank, old_depth = self.root.feature_table.get_novel_feature((n.state[0],n.d))
        if n.d < old_depth :
            n.visited = True
            self.num_visited += 1
            self.root.feature_table.update_feature_table((n.state[0],n.d))
        elif not n.visited and n.d >= old_depth :
            n.visited = True
            self.num_visited += 1
            self.solve_and_propagate_labels(n)
            return False
        elif n.visited and old_depth < n.d :
            self.solve_and_propagate_labels(n)
            return False
        return True

    def expand(self, env, n : OR_Node ) :
        """
            Construct AND nodes por each of the actions applicable
            on state s(n)
        """
        if len(n.children) == 0 :
            for a in range(env.action_space.n) :
                and_node = AND_Node(a,n)
                and_node.SOLVED = False
                and_node.Q = 0
                and_node.num_visits = 0
                and_node.accumulated_reward = n.accumulated_reward

    def sample_child(self, n : OR_Node):
        candidates = [ k for k in n.children.keys() if not n.children[k].SOLVED]
        return random.choice(candidates)


    def pick_random_unsolved_child(self, env, n : OR_Node ) :
        selected = self.sample_child(n)
        assert(not n.children[selected].SOLVED)
        if self._atari == "True" and len(n.children[selected].children) != 0 and self._caching != "None":
            elapsed_steps = env._elapsed_steps
            envuw = env.unwrapped
            for node, reward in n.children[selected].children:
                break
            if node.restoreState is not None :
                env.unwrapped.restore_full_state(node.restoreState)
                env._elapsed_steps = elapsed_steps + 1
            else :
                t0 = time.perf_counter()
                next_state, sreward, terminal, _ = env.step(selected)
                tf = time.perf_counter()
                self.rollout_runtime_sim += tf - t0
                reward = sreward
                self.sim_calls += 1
                node.restoreState = env.unwrapped.clone_full_state()
                node.terminal = terminal
        else:
            t0 = time.perf_counter()
            next_state, reward, terminal, _ = env.step(selected)
            tf = time.perf_counter()
            self.rollout_runtime_sim += tf - t0
            self.sim_calls += 1

            if self._representation is not None :
                t0 = time.perf_counter()

                parentLength = int(n.state.size)

                screen =  env.unwrapped.ale.getScreen()
                next_state_flat = self._representation.getActiveFeatures(screen, n.state[0], parentLength)

                tf = time.perf_counter()

                next_state_flat = np.reshape(next_state_flat, [1, len(next_state_flat)])

                self.rollout_runtime_sim += tf - t0
            elif self._grayScale == "True":
                next_state_flat = np.reshape(self.convertScreenToGrayCompressed(next_state), [1, self._grayScaleSizeX * self._grayScaleSizeY])
            else :
                next_state_flat = np.reshape(next_state, [1, np.prod(env.observation_space.shape)])

            succ = OR_Node(next_state_flat,n.d + 1,terminal)
            succ.add_parent(n.children[selected])
            if self._atari == "True" and self._caching != "None":
                succ.restoreState = env.unwrapped.clone_full_state()
            succ.SOLVED = False
            succ.visited = False

            if succ.d == self._horizon:
                succ.terminal = True

            if n.children[selected].update(reward,succ) :
                n.children[selected].SOLVED = False # if we get a new successor, we unsolve the node
                node = succ
            else :
                for child in n.children[selected].children :
                    succ1, reward1 = child
                    if reward1 == reward and succ1 == succ :
                        node = succ1

        try :
            node.accumulated_reward = max(node.accumulated_reward, n.accumulated_reward + reward)
        except AttributeError :
            node.accumulated_reward = n.accumulated_reward + reward
            node.num_visits = 0

        self.max_depth = max( self.max_depth, node.d)
        n.children[selected].num_visits += 1
        node.num_visits += 1
        return node

    def rollout(self, env, n : OR_Node) :
        self.strategy.rollout(self,env,n)

    def check_OR_solved(self, n : OR_Node ) :
        is_solved = True
        for _, a_node in n.children.items() :
            if not a_node.SOLVED :
                is_solved = False
                break
        n.SOLVED = is_solved
        if n.SOLVED :
            self.num_solved +=1
            for p in n.parents :
                self.check_AND_solved(p)


    def check_AND_solved(self, n: AND_Node ) :
        is_solved = True
        for succ, _ in n.children :
            if not succ.SOLVED :
                is_solved = False
                break
        n.SOLVED = is_solved
        if n.SOLVED :
            self.num_solved +=1
            self.check_OR_solved(n.parent)

    def solve_and_propagate_labels(self, n : OR_Node ) :
        n.SOLVED = True
        self.num_solved += 1
        for p in n.parents :
            self.check_AND_solved(p)

    def pick_random_action(self, env) :
        actions = [ a for a in range(env.action_space.n)]
        if len(actions) == 0:
            return None
        return random.choice(actions)

    def cost_to_go_est(self, env, n: OR_Node):
        if self._pruned_state_strategy == "random_rollout" :
            return random_rollout(self, env, n)
        elif self._pruned_state_strategy == "knuth" :
            return knuth_alg(self, env, n)
        elif self._pruned_state_strategy == "stochastic_enum" :
            return stoch_enum_alg(self, env, n)
        elif self._pruned_state_strategy == "heuristic" :
            return env.unwrapped.getAdmissibleHeuristic()
        return 0
