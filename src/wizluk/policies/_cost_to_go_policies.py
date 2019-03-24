from wizluk.memory.and_or_graph import OR_Node
import copy
import time

def random_rollout(lookahead, env, n : OR_Node) :
    sum_reward = 0
    #check if node to rollout is terminal or beyond the horizon
    terminal = n.terminal
    if terminal or n.d > lookahead._horizon  :
        return 0

    if lookahead._atari == "True":
        envuw = env.unwrapped
        env2 = envuw.clone_full_state()
        episode_started_at = env._episode_started_at
        elapsed_steps = env._elapsed_steps
        env._max_episode_seconds = 2**20

    rolloutsComplete = lookahead._num_rollouts
    for k in range(0,lookahead._num_rollouts):
        if lookahead._atari != "True":
            env2 = copy.deepcopy(env)
        num_samples = 0
        r = 0
        state = n.state
        lookahead.num_random_rollouts += 1
        while True:
            # Check for remaining budget
            if lookahead.sim_calls - lookahead.init_sim_calls >= lookahead.sim_budget:
                if k == 0:
                    r =  float('-inf')
                    rolloutsComplete = 1
                else:
                    r = 0.0
                    rolloutsComplete = k
                break
            if lookahead._atari == "True":
                a =lookahead.base_policy.get_action(env, state)
                t0 = time.perf_counter()
                state, reward, terminal, _ = env.step(a)
                tf = time.perf_counter()
                if hasattr(lookahead, 'rollout_runtime_sim'):
                    lookahead.rollout_runtime_sim += tf - t0
            else :
                a =lookahead.base_policy.get_action(env2, state)
                t0 = time.perf_counter()
                state, reward, terminal, _ = env2.step(a)
                tf = time.perf_counter()
                if hasattr(lookahead, 'rollout_runtime_sim'):
                    lookahead.rollout_runtime_sim += tf - t0
            lookahead.sim_calls += 1
            r += reward * lookahead._gamma ** num_samples
            num_samples += 1
            #checking stopping conditions
            if terminal :
                continue_rollout = False
                lookahead.random_rollouts_reached_terminal_state += 1
                break
            elif num_samples + n.d > lookahead._horizon  :
                continue_rollout = False
                lookahead.random_rollouts_reached_horizon += 1
                break
            elif lookahead._gamma ** num_samples < lookahead._epsilon :
                continue_rollout = False
                lookahead.random_rollouts_reached_discount_value_smaller_than_epsilon += 1
                break
        sum_reward += r
        if lookahead._atari == "True":
            envuw.restore_full_state(env2)
            env._episode_started_at = episode_started_at
            env._elapsed_steps = elapsed_steps
    return sum_reward / rolloutsComplete

# Knuth's algorithm for estimating the cost of a tree.
def knuth_alg(lookahead, env, n : OR_Node) :
    # Check if node to rollout is terminal or beyond the horizon.
    terminal = n.terminal
    if terminal or n.d > lookahead._horizon  :
        return 0

    if lookahead._atari == "True":
        envuw = env.unwrapped
        env2 = envuw.clone_full_state()
        episode_started_at = env._episode_started_at
        elapsed_steps = env._elapsed_steps
        env._max_episode_seconds = 2**20
    sum_reward = 0
    rolloutsComplete = lookahead._num_rollouts
    for k in range(0, lookahead._num_rollouts):
        degree = 1 # Number of possible states at the current search depth.
        if lookahead._atari != "True":
            env2 = copy.deepcopy(env)
        num_samples = 0
        r = 0
        state = n.state
        lookahead.num_random_rollouts += 1
        while True:
            # Check for remaining budget
            if lookahead.sim_calls - lookahead.init_sim_calls >= lookahead.sim_budget:
                if k == 0:
                    r =  float('-inf')
                    rolloutsComplete = 1
                else:
                    r = 0.0
                    rolloutsComplete = k
                break
            if lookahead._atari == "True":
                a =lookahead.base_policy.get_action(env, state)
                t0 = time.perf_counter()
                state, reward, terminal, _ = env.step(a)
                tf = time.perf_counter()
                if hasattr(lookahead, 'rollout_runtime_sim'):
                    lookahead.rollout_runtime_sim += tf - t0
            else :
                a =lookahead.base_policy.get_action(env2, state)
                t0 = time.perf_counter()
                state, reward, terminal, _ = env2.step(a)
                tf = time.perf_counter()
                if hasattr(lookahead, 'rollout_runtime_sim'):
                    lookahead.rollout_runtime_sim += tf - t0
            lookahead.sim_calls += 1
            degree = env.action_space.n * degree
            r += degree*(reward * lookahead._gamma ** num_samples)
            num_samples += 1
            #checking stopping conditions
            if terminal :
                continue_rollout = False
                lookahead.random_rollouts_reached_terminal_state += 1
                break
            elif num_samples + n.d > lookahead._horizon  :
                continue_rollout = False
                lookahead.random_rollouts_reached_horizon += 1
                break
            elif lookahead._gamma ** num_samples < lookahead._epsilon :
                continue_rollout = False
                lookahead.random_rollouts_reached_discount_value_smaller_than_epsilon += 1
                break
        sum_reward += r
        if lookahead._atari == "True":
            envuw.restore_full_state(env2)
            env._episode_started_at = episode_started_at
            env._elapsed_steps = elapsed_steps

    return sum_reward / rolloutsComplete

# Stochastic enumeration algorithm for estimating the cost of a tree. TODO: implement Atari verison as well
def stoch_enum_alg(lookahead, env, n : OR_Node) :
    numberOfPathsToConsider = lookahead._number_of_paths_to_consider_for_stoch_enum # often considered as computational budget or 'B'

    # Check if node to rollout is terminal or beyond the horizon.
    terminal = n.terminal
    if terminal or n.d > lookahead._horizon  :
        return 0

    if lookahead._atari == "True":
        envuw = env.unwrapped
        env2 = envuw.clone_full_state()
        episode_started_at = env._episode_started_at
        elapsed_steps = env._elapsed_steps
        env._max_episode_seconds = 2**20

    sum_reward = 0
    rolloutsComplete = lookahead._num_rollouts
    for k in range(0, lookahead._num_rollouts):
        degree = 1 # Number of possible states at the current search depth.
        if lookahead._atari != "True":
            env2 = copy.deepcopy(env)
        depthOfSearch = 0
        hyperChildren = [(env2, [], False)]
        state = n.state
        lookahead.num_random_rollouts += 1
        r = 0
        num_samples = 0
        while True:
            # Check for remaining budget
            if lookahead.sim_calls - lookahead.init_sim_calls >= lookahead.sim_budget:
                if k == 0:
                    r =  float('-inf')
                    rolloutsComplete = 1
                else:
                    r = 0.0
                    rolloutsComplete = k
                break
            rewardForStep = 0
            newHyperChildren = []
            while len(newHyperChildren) < numberOfPathsToConsider and len(hyperChildren) * env.action_space.n  > len(newHyperChildren):
                indxChild = random.choice(range(0, len(hyperChildren)))
                child, actionsConsidered, isTerminal = hyperChildren[indxChild]
                if isTerminal:
                    continue
                childEnv = copy.deepcopy(child)
                if lookahead._atari == "True":
                    a = lookahead.base_policy.get_action(env, state)
                    if a in actionsConsidered:
                        continue #  Already considered successor
                    t0 = time.perf_counter()
                    state, reward, terminal, _ = env.step(a)
                    tf = time.perf_counter()
                    if hasattr(lookahead, 'rollout_runtime_sim'):
                        lookahead.rollout_runtime_sim += tf - t0
                else :
                    a = lookahead.base_policy.get_action(childEnv, state)
                    if a in actionsConsidered:
                        continue #  Already considered successor
                    t0 = time.perf_counter()
                    state, reward, terminal, _ = childEnv.step(a)
                    tf = time.perf_counter()
                    if hasattr(lookahead, 'rollout_runtime_sim'):
                        lookahead.rollout_runtime_sim += tf - t0
                    newHyperChildren.append((childEnv,[], terminal))
                actionsConsidered.append(a)
                rewardForStep += reward
                lookahead.sim_calls += 1
            num_samples += len(newHyperChildren)
            depthOfSearch += 1
            degree = env.action_space.n * degree # simplificaiton of (env.action_space.n * len(hyperChildren)) / len(hyperChildren) * degree
            r += degree * ((rewardForStep * lookahead._gamma ** depthOfSearch) / len(newHyperChildren))
            hyperChildren = newHyperChildren

            allAreTerminal = True
            for dum1, dum2 ,childTerminal in newHyperChildren:
                if not childTerminal:
                    allAreTerminal = False
                    break

            #  Checking stopping conditions.
            if terminal :
                continue_rollout = False
                lookahead.random_rollouts_reached_terminal_state += 1
                break
            elif depthOfSearch + n.d > lookahead._horizon  :
                continue_rollout = False
                lookahead.random_rollouts_reached_horizon += 1
                break
            elif lookahead._gamma ** depthOfSearch < lookahead._epsilon :
                continue_rollout = False
                lookahead.random_rollouts_reached_discount_value_smaller_than_epsilon += 1
                break
        sum_reward += r
        if lookahead._atari == "True":
            envuw.restore_full_state(env2)
            env._episode_started_at = episode_started_at
            env._elapsed_steps = elapsed_steps

    return sum_reward / rolloutsComplete
