import numpy as np
import pandas as pd
import warnings
import pickle
import bayes_vector_ops as bm
import policy as algo
import sim_wrapper as sw
from itertools import permutations
import copy
from scipy.special import logit
from scipy.special import expit
import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from joblib import dump
# dump(variables, 'variables.joblib')
# loaded_variables = load('variables.joblib')
from tqdm import tqdm

from functools import partial
from typing import Optional, Literal, Dict, Any
import logging
from tqdm import tqdm
from joblib import Parallel, delayed

from simulation_configurator import SimulationConfig
from test_procedure_configurator import TestProcedure, ANOVA, TControl, TConstant, Tukey
import bayes_vector_ops as bayes


"""
Part 1: set sim config
"""
sim_config = SimulationConfig(
    n_rep=2000,
    n_arm=3,
    horizon=1000,  # max horizon to try in simulation
    burn_in_per_arm=5,
    # horizon_check_points=sw.generate_quadratic_schedule(2000), #can ignore for now... TODO: see where I used it (and delete if not)
    # can set tuning_density to make the schedule denser / looser
    n_opt_trials=5,  # TODO: optimize for this in our code
    # arm_mean_reward_dist_loc = [0.7,0.3,0.3],
    # arm_mean_reward_dist_scale = 0.01,
    test_procedure=ANOVA(),
    step_cost=0.1,
    reward_evaluation_method='regret',
    vector_ops=bayes.BackendOpsNP()
)
algo = algo.EpsTS



"""
Part 2: Run sim and extract df result
"""
sim_config.manual_init() #need to run every time before simulation
params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo ,algo_param_list=None)
records = []
for (algo_name, epsilon, setting), metrics in sim_result_keeper.items():
    record = {
        'algo_name': algo_name,
        'epsilon': epsilon,
        'setting': setting,
        **metrics  # unpack obj_score, obj_score_sd, etc.
    }
    records.append(record)

# Create DataFrame
sim_result_df = pd.DataFrame(records)
