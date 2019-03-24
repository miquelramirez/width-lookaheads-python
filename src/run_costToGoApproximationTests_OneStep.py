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
from wizluk.util import load_agent, gen_primes
import traceback
import warnings
import sys

import ray
ray.init(num_cpus=1, num_gpus=0)

@ray.remote(num_cpus=1)
def run_single_run(UCT_parameters,UCT_variants, run_num, Domain, sim_dt, sim_budget, horizon, numberRollouts, runNum) :
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
    if not os.path.isfile('../results/{}_{}_simdt_{}_simBud_{}_Horizon_{}_numRoll_{}_runNum_{}.dat'.format(UCT_variants['Name'][run_num],Domain, sim_dt, sim_budget, horizon, numberRollouts, runNum)):
        warnings.showwarning = warn_with_traceback
        wizluk.setup_logger("iw_gridworld_v1.log")
        env = gym.make(Domain)
        np.random.seed(1337)
        env.seed(1337)

        UCT = wizluk.policies.One_Step(**(UCT_parameters[run_num]))
        UCT_agent = wizluk.agents.LookaheadAgent(env, UCT, name='UCT', domain=Domain)
        UCT_Rollout_df = {}
        UCT_agent.init_evaluation_statistics(UCT_Rollout_df)
        S =  np.prod(env.observation_space.shape)

        x = env.reset()
        x0 = copy.deepcopy(x)
        UCT_agent.start_episode()
        score = 0.0
        for s in range(horizon):
            #print(s)
            wizluk.logger.debug("action number: {}".format(s))
            x = np.reshape(x, [1, S])
            u = UCT_agent.get_action(x)
            x_next, reward, done, info = env.step(u)
            x_next = np.reshape(x, [1, S])
            UCT_agent.observe_transition(x,u, reward, x_next, done, False)
            x = x_next
            env.render()
            if done:
                break
            score += reward
        env.close()
        UCT_agent.stop_episode()
        UCT_agent.collect_evaluation_statistics( UCT_Rollout_df, x0 )

        with open('../results/{}_{}_simdt_{}_simBud_{}_Horizon_{}_numRoll_{}_runNum_{}.dat'.format(UCT_variants['Name'][run_num],Domain, sim_dt, sim_budget, horizon, numberRollouts, runNum), 'wb') as output:
            pickle.dump(score, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("already done")

@ray.remote(num_cpus=1)
def run_experiment(Domain, sim_dt, sim_budget, horizon, numberRollouts, N) :
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

    uct_random_rollout_parameters = {
    "budget" : sim_dt,
    "sim_budget" : sim_budget,
    "novelty_definition" : "depth",
    "include_root_in_novelty_table": "True",
    "horizon": horizon,
    "cost_to_go_est": "random_rollout",
    "num_rollouts": numberRollouts
    }

    uct_knuth_parameters = {
    "budget" : sim_dt,
    "sim_budget" : sim_budget,
    "novelty_definition" : "depth",
    "include_root_in_novelty_table": "True",
    "horizon": horizon,
    "cost_to_go_est": "knuth",
    "num_rollouts": numberRollouts
    }

    uct_heur_parameters = {
    "budget" : sim_dt,
    "sim_budget" : sim_budget,
    "novelty_definition" : "depth",
    "include_root_in_novelty_table": "True",
    "horizon": horizon,
    "cost_to_go_est": "heuristic",
    }

    #iw_depth_stochastic_enum_parameters = {
    #"budget" : sim_dt,
    #"novelty_definition" : "depth",
    #"include_root_in_novelty_table": "True",
    #"horizon": horizon,
    #"pruned_state_strategy": "stochastic_enum",
    #"number_of_paths_to_consider_for_stoch_enum": 2,
    #"num_rollouts": numberRollouts
    #}

    iw_variants = {'Name': ["OneStep_depth_random_rollout"]}#, "OneStep_depth_knuth", "OneStep_heur"]}#, "iw_depth_stochastic_enum"]}
    iw_parameters = [uct_random_rollout_parameters]#, uct_knuth_parameters, uct_heur_parameters]#, iw_depth_stochastic_enum_parameters]

    listOfRuns = []
    for run_num in range(len(iw_parameters)) :
        for k in range(N) :
            listOfRuns.append(run_single_run.remote(iw_parameters, iw_variants, run_num, Domain, sim_dt, sim_budget, horizon, numberRollouts, k))
    ray.get(listOfRuns)
    gc.collect()
    return 0

def main(cmd_line_args) :
    t0 = time.perf_counter()
    listOfRuns = []
    gridVersion = 1
    for intState in range(1):
        #for num_states in [10, 50]:
            #for Domain, H in [("Antishape-{}-initS{}-v2".format(num_states, intState), 4*num_states), ("Combolock-{}-initS{}-v2".format(num_states, intState), 4*num_states)]:
            #for Domain, H, numberRollouts in [("GridWorld-4x4-initS{}-v{}".format(intState, gridVersion), 20,4), ("GridWorld-10x10-initS{}-v{}".format(intState, gridVersion), 50, 10), ("GridWorld-20x20-initS{}-v{}".format(intState, gridVersion), 100, 20), ("GridWorld-50x50-initS{}-v{}".format(intState, gridVersion), 250, 50), ("GridWorld-100x100-initS{}-v{}".format(intState, gridVersion), 500, 100)]:
        for Domain, H, numberRollouts in [("CTP-10x10-initS{}-v{}".format(intState, gridVersion), 50, 10)]:#, ("CTP-20x20-initS{}-v{}".format(intState, gridVersion), 100, 20)]:
                for lookaheadBudget in [1000]:
                    numberRollouts = 1
                    listOfRuns.append(run_experiment.remote(Domain, 99999999, lookaheadBudget, H, numberRollouts, 20) )
    ray.get(listOfRuns)
    ray.error_info()
    tf = time.perf_counter()
    print("Time taken = {}".format(tf-t0))
    #wizluk.logger.info("Time taken = {}".format(tf-t0))

if __name__ == "__main__":

    # Make sure that the random seed is fixed before running the script, to ensure determinism
    # in the output of the parser.
    if not wizluk.util.fix_seed_and_possibly_rerun():
        main(sys.argv[1:])
