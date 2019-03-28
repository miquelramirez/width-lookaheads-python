#To run: xvfb-run -s "-screen 0 1400x900x24" python3 run_dist_script.py |& tee output_dist.log
# -*- coding: utf-8 -*-
import sys, types, time, random, os, copy
import json
import logging
import copy
import time
from queue import PriorityQueue
import pickle

from logging import FileHandler, StreamHandler

import gym

import numpy as np
import gc
import wizluk
import wizluk.agents
import wizluk.policies
import wizluk.heuristics
import wizluk.envs
from wizluk.util import gen_primes
import traceback
import warnings
import sys

import ray

ray.init(num_cpus=32, num_gpus=0)
@ray.remote(num_cpus=1)
def run_single_run(iw_parameters,iw_variants, run_num, Domain, sim_dt, sim_budget, horizon, numberRollouts, runNum, seed) :
    import time
    import numpy as np
    import random
    import json
    import matplotlib.pyplot as plt
    import gc
    import wizluk
    import pandas as pd

    import sys
    import gym
    import wizluk.envs
    import wizluk.policies
    from wizluk.policies import RandomPolicy
    import cv2
    import warnings
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback
    wizluk.setup_logger("iw_gridworld_v1.log")
    env = gym.make(Domain)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)

    state_size = np.prod(env.observation_space.shape)
    vanilla_features = wizluk.heuristics.novelty.SVF.create_features(state_size)
    IW_Rollout = wizluk.policies.IW_Rollout(features = vanilla_features, **(iw_parameters[run_num]))
    IW_Rollout_agent = wizluk.agents.LookaheadAgent(env, IW_Rollout, name='IW_Rollout', domain='GridWorld-16x16-v1')
    IW_Rollout_df = {}
    IW_Rollout_agent.init_evaluation_statistics(IW_Rollout_df)
    S =  np.prod(env.observation_space.shape)

    x = env.reset()
    x0 = copy.deepcopy(x)
    IW_Rollout_agent.start_episode()
    score = 0.0
    for s in range(horizon):
        wizluk.logger.debug("action number: {}".format(s))
        x = np.reshape(x, [1, S])
        u = IW_Rollout_agent.get_action(x)
        x_next, reward, done, info = env.step(u)
        x_next = np.reshape(x, [1, S])
        IW_Rollout_agent.observe_transition(x,u, reward, x_next, done, False)
        x = x_next
        if done:
            break
        score += reward
    IW_Rollout_agent.stop_episode()
    IW_Rollout_agent.collect_evaluation_statistics( IW_Rollout_df, x0 )
    with open('../results/{}_{}_simdt_{}_simBud_{}_Horizon_{}_numRoll_{}_runNum_{}.dat'.format(iw_variants['Name'][run_num],Domain, sim_dt, sim_budget, horizon, numberRollouts, runNum), 'wb') as output:
        pickle.dump(score, output, pickle.HIGHEST_PROTOCOL)

@ray.remote(num_cpus=1)
def run_experiment(Domain, sim_dt, sim_budget, horizon, numberRollouts, N, seeds, seed, includeHeur, includeKuth) :
    import time
    import numpy as np
    import random
    import json
    import matplotlib.pyplot as plt
    import gc
    import wizluk
    import pandas as pd

    import sys
    import gym
    import wizluk.envs
    import wizluk.policies
    from wizluk.policies import RandomPolicy
    import cv2
    import warnings

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback
    wizluk.setup_logger("iw_gridworld_v1.log")

    iw_depth_include_root_parameters = {
    "budget" : sim_dt,
    "sim_budget" : sim_budget,
    "novelty_definition" : "depth",
    "include_root_in_novelty_table": "True",
    "horizon": horizon
    }

    iw_depth_random_rollout_parameters = {
    "budget" : sim_dt,
    "sim_budget" : sim_budget,
    "novelty_definition" : "depth",
    "include_root_in_novelty_table": "True",
    "horizon": horizon,
    "pruned_state_strategy": "random_rollout",
    "num_rollouts": numberRollouts
    }

    iw_depth_knuth_parameters = {
    "budget" : sim_dt,
    "sim_budget" : sim_budget,
    "novelty_definition" : "depth",
    "include_root_in_novelty_table": "True",
    "horizon": horizon,
    "pruned_state_strategy": "knuth",
    "num_rollouts": numberRollouts
    }

    iw_depth_heur_parameters = {
    "budget" : sim_dt,
    "sim_budget" : sim_budget,
    "novelty_definition" : "depth",
    "include_root_in_novelty_table": "True",
    "horizon": horizon,
    "pruned_state_strategy": "heuristic"
    }

    #iw_depth_stochastic_enum_parameters = {
    #"budget" : sim_dt,
    #"sim_budget" : sim_budget,
    #"novelty_definition" : "depth",
    #"include_root_in_novelty_table": "True",
    #"horizon": horizon,
    #"pruned_state_strategy": "stochastic_enum",
    #"number_of_paths_to_consider_for_stoch_enum": 2,
    #"num_rollouts": numberRollouts
    #}

    iw_variants = {'Name': ["iw_depth_include_root", "iw_depth_random_rollout"]}#, "iw_depth_knuth"]}#, "iw_depth_heur"]}#, "iw_depth_stochastic_enum"]}
    iw_parameters = [iw_depth_include_root_parameters, iw_depth_random_rollout_parameters] #, iw_depth_knuth_parameters]#, iw_depth_heur_parameters]#, iw_depth_stochastic_enum_parameters]

    if includeKuth:
        iw_variants['Name'].append("iw_depth_knuth")
        iw_parameters.append(iw_depth_knuth_parameters)

    if includeHeur:
        iw_variants['Name'].append("iw_depth_heur")
        iw_parameters.append(iw_depth_heur_parameters)

    listOfRuns = []
    random.seed(seeds[seed])
    for run_num in range(len(iw_parameters)) :
        seedStart = random.randint(0,50000)
        seedSkip = random.randint(1,2001)
        for k in range(N) :
            listOfRuns.append(run_single_run.remote(iw_parameters, iw_variants, run_num, Domain, sim_dt, sim_budget, horizon, numberRollouts, k, seeds[seedStart + k * seedSkip]))
    ray.get(listOfRuns)
    gc.collect()
    return 0

def main(cmd_line_args) :
    t0 = time.perf_counter()
    listOfRuns = []
    seeds = []
    num_seeds = 100000
    #For reproducibility
    random.seed(1337)
    for p in gen_primes():
        seeds.append(p)
        if len(seeds) == num_seeds:
            break
    for intState in range(10):
        for num_states in [10, 50]:
            for Domain, H in [("Antishape-{}-initS{}-v2".format(num_states, intState), 4*num_states), ("Combolock-{}-initS{}-v2".format(num_states, intState), 4*num_states)]:
            #for Domain, H, numberRollouts in [("GridWorld-4x4-initS{}-v{}".format(intState, gridVersion), 20,4), ("GridWorld-10x10-initS{}-v{}".format(intState, gridVersion), 50, 10), ("GridWorld-20x20-initS{}-v{}".format(intState, gridVersion), 100, 20), ("GridWorld-50x50-initS{}-v{}".format(intState, gridVersion), 250, 50), ("GridWorld-100x100-initS{}-v{}".format(intState, gridVersion), 500, 100)]:
        #for Domain, H, numberRollouts in [("CTP-4x4-initS{}-v{}".format(intState, gridVersion), 20,4), ("CTP-10x10-initS{}-v{}".format(intState, gridVersion), 50, 10), ("CTP-20x20-initS{}-v{}".format(intState, gridVersion), 100, 20), ("CTP-50x50-initS{}-v{}".format(intState, gridVersion), 250, 50)]:
                for lookaheadBudget in [100, 500, 1000]:
                    numberOfRuns = 20
                    seed = random.randint(0,100000)
                    numberRollouts = 1
                    listOfRuns.append(run_experiment.remote(Domain, 99999999, lookaheadBudget, H, numberRollouts, numberOfRuns, seeds, seed, False, False) )

        for gridVersion in [2, 3, 5]:
            for gridDim in [10, 20, 50]:
                for Domain, H, numberRollouts in [("GridWorld-{}x{}-initS{}-v{}".format(gridDim, gridDim, intState, gridVersion), gridDim*5,1)]:
                    #for Domain, H, numberRollouts in [("CTP-4x4-initS{}-v{}".format(intState, gridVersion), 20,4), ("CTP-10x10-initS{}-v{}".format(intState, gridVersion), 50, 10), ("CTP-20x20-initS{}-v{}".format(intState, gridVersion), 100, 20), ("CTP-50x50-initS{}-v{}".format(intState, gridVersion), 250, 50)]:
                    for lookaheadBudget in [100, 1000, 10000]:
                        numberOfRuns = 20
                        seed = random.randint(0,100000)
                        numberRollouts = 1
                        if gridVersion == 5:
                            listOfRuns.append(run_experiment.remote(Domain, 99999999, lookaheadBudget, H, numberRollouts, numberOfRuns, seeds, seed, True, False) )
                        else:
                            listOfRuns.append(run_experiment.remote(Domain, 99999999, lookaheadBudget, H, numberRollouts, numberOfRuns, seeds, seed, False, False) )
        for gridDim in [10, 20]:
            for Domain, H, numberRollouts in [("CTP-{}x{}-initS{}-v1".format(gridDim, gridDim, intState), gridDim*5,1)]:
                for lookaheadBudget in [100, 1000, 10000]:
                    numberOfRuns = 20
                    seed = random.randint(0,100000)
                    numberRollouts = 1
                    listOfRuns.append(run_experiment.remote(Domain, 99999999, lookaheadBudget, H, numberRollouts, numberOfRuns, seeds, seed, True, True) )
        ray.get(listOfRuns)
    ray.error_info()
    tf = time.perf_counter()
    print("Time taken = {}".format(tf-t0))

if __name__ == "__main__":

    # Make sure that the random seed is fixed before running the script, to ensure determinism
    # in the output of the parser.
    if not wizluk.util.fix_seed_and_possibly_rerun():
        main(sys.argv[1:])
