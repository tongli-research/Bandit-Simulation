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

logging.getLogger("ax.service.managed_loop").setLevel(logging.CRITICAL)
logging.getLogger('ax.generation_strategy.dispatch_utils').setLevel(logging.CRITICAL)


# for vm?
import sys
import os

# Add project root (where bayes_vector_ops.py is) to sys.path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# TODO: in paper, can define broadly how user can plug in their own test
# TODO: add a custom function (user define a function, input is reward and action hist, output is test. the code help them run it in h0 to get crit, and then evalaute it in objective function


# np.random.seed(0)
sim_config_base = SimulationConfig(
    n_rep=50000,
    n_arm=3,
    horizon=3000,  # max horizon to try in simulation
    burn_in_per_arm=5,
    # horizon_check_points=sw.generate_quadratic_schedule(2000), #can ignore for now... TODO: see where I used it (and delete if not)
    # can set tuning_density to make the schedule denser / looser
    n_opt_trials=15,  # TODO: optimize for this in our code
    # arm_mean_reward_dist_loc = [0.7,0.3,0.3],
    # arm_mean_reward_dist_scale = 0.01,
    test_procedure=ANOVA(),
    step_cost=0.05,
    reward_evaluation_method='regret',
    vector_ops=bayes.BackendOpsTF()
)
# algo_list = ['ts_adapt_explor','ts_postdiff_top','eps_ts', 'ts_postdiff_ur','ts_probclip']
algo_list = [algo.TSProbClip,algo.TSPostDiffTop, algo.TSPostDiffUR, algo.EpsTS,algo.Top2TS] #TODO: import directly
algo_param_list = list(np.arange(0, 1.001, 0.02))

test_list = [
    ANOVA(),
    TControl(),
    TControl(permutation_test=True),
    TControl(test_type='two-sided'),
    TConstant(),
    Tukey(test_type='all-pair-wise'),
    Tukey(test_type='distinct-best-arm'),
]

step_cost_list = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]

loc_list = [0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8]
scale_list = [0.08,0.10,0.12,0.135, 0.15,0.165, 0.18,0.20,0.22]



"""
###  Simulation 1: Optimization Loop
"""
main_task_list = [(i, j, k) for i in range(len(step_cost_list)) for j in range(len(algo_list)) for k in range(len(test_list))]
loc_task_list = [(i, j) for i in range(len(loc_list)) for j in range(len(algo_list))]
scale_task_list = [(i, j) for i in range(len(scale_list)) for j in range(len(algo_list))]

def run_main_task(i, j, k):
    sim_config = copy.deepcopy(sim_config_base)
    sim_config.step_cost = step_cost_list[i]
    algo = algo_list[j]
    sim_config.test_procedure = test_list[k]
    sim_config.manual_init()
    params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo,algo_param_list=algo_param_list)
    # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
    return {'test':sim_config.test_procedure.test_signature,
            'step_cost': sim_config.step_cost,
            'mean_reward_dist_loc':sim_config.arm_mean_reward_dist_loc[0],
            'mean_reward_dist_scale':sim_config.arm_mean_reward_dist_scale[0],
            'algo_name': algo.__name__,
            'algo_param': params['algo_para'],
            **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
            'all_results': sim_result_keeper,
            }

def run_loc_mismatch_task(i, j):
    sim_config = copy.deepcopy(sim_config_base)
    sim_config.arm_mean_reward_dist_loc = loc_list[i]
    algo = algo_list[j]
    sim_config.manual_init()
    params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo,algo_param_list=None)
    # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
    return {'test': sim_config.test_procedure.test_signature,
            'step_cost': sim_config.step_cost,
            'mean_reward_dist_loc': sim_config.arm_mean_reward_dist_loc[0],
            'mean_reward_dist_scale': sim_config.arm_mean_reward_dist_scale[0],
            'algo_name': algo.__name__,
            'algo_param': params['algo_para'],
            **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
            'all_results': sim_result_keeper,
            }

def run_scale_mismatch_task(i, j):
    sim_config = copy.deepcopy(sim_config_base)
    sim_config.horizon = 4000
    sim_config.arm_mean_reward_dist_scale = scale_list[i]
    algo = algo_list[j]
    sim_config.manual_init()
    params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo,algo_param_list=None)
    # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
    return {'test': sim_config.test_procedure.test_signature,
            'step_cost': sim_config.step_cost,
            'mean_reward_dist_loc': sim_config.arm_mean_reward_dist_loc[0],
            'mean_reward_dist_scale': sim_config.arm_mean_reward_dist_scale[0],
            'algo_name': algo.__name__,
            'algo_param': params['algo_para'],
            **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
            'all_results': sim_result_keeper,
            }

def rerun_loc_mismatch_task(j):
    sim_config = copy.deepcopy(sim_config_base)
    algo = algo_list[j]
    sim_config.manual_init()
    algo_param_list = list(loc_best_df.algo_param[loc_best_df.algo_name == algo.__name__])
    params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo,algo_param_list=algo_param_list)

    # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
    return {'test': sim_config.test_procedure.test_signature,
            'step_cost': sim_config.step_cost,
            'mean_reward_dist_loc': sim_config.arm_mean_reward_dist_loc[0],
            'mean_reward_dist_scale': sim_config.arm_mean_reward_dist_scale[0],
            'algo_name': algo.__name__,
            'algo_param': params['algo_para'],
            **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
            'all_results': sim_result_keeper,
            }

def rerun_scale_mismatch_task(j):
    sim_config = copy.deepcopy(sim_config_base)
    algo = algo_list[j]
    sim_config.manual_init()

    algo_param_list = list(scale_best_df.algo_param[scale_best_df.algo_name == algo.__name__])
    params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo, algo_param_list=algo_param_list)
    # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
    return {'test': sim_config.test_procedure.test_signature,
            'step_cost': sim_config.step_cost,
            'mean_reward_dist_loc': sim_config.arm_mean_reward_dist_loc[0],
            'mean_reward_dist_scale': sim_config.arm_mean_reward_dist_scale[0],
            'algo_name': algo.__name__,
            'algo_param': params['algo_para'],
            **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
            'all_results': sim_result_keeper,
            }

def extract_results(results):
    """
    Given a list of results (each row containing 'all_results' as a nested dict),
    returns:
    - best_df: original DataFrame without the 'all_results' column
    - full_df: flattened DataFrame with one row per (algo, param) combination

    Parameters:
        results (list[dict]): List of result dictionaries, each with an 'all_results' key.

    Returns:
        best_df (pd.DataFrame)
        full_df (pd.DataFrame)
    """
    results_df = pd.DataFrame(results)
    best_df = results_df.drop(columns=['all_results'])


    all_rows = []
    for _, row in results_df.iterrows():
        row_info = row.drop('all_results').to_dict()
        all_result_dict = row['all_results']

        for (algo, param, _), result in all_result_dict.items():
            entry = row_info.copy()
            entry['best_algo_param'] = entry['algo_param']
            entry['algo_name'] = algo
            entry['algo_param'] = param
            entry.update({k: float(v) for k, v in result.items()})
            all_rows.append(entry)

    full_df = pd.DataFrame(all_rows)
    return best_df, full_df


main_results = Parallel(n_jobs=-1)(delayed(run_main_task)(i, j, k) for i, j, k in main_task_list)
loc_results = Parallel(n_jobs=-1)(delayed(run_loc_mismatch_task)(i, j) for i, j in loc_task_list)
scale_results = Parallel(n_jobs=-1)(delayed(run_scale_mismatch_task)(i, j) for i, j in scale_task_list)

main_best_df, main_full_df = extract_results(main_results)
loc_best_df, loc_full_df = extract_results(loc_results)
scale_best_df, scale_full_df = extract_results(scale_results)

rerun_loc_mismatch_results = Parallel(n_jobs=-1)(delayed(rerun_loc_mismatch_task)( j) for  j in range(len(algo_list)))
rerun_scale_mismatch_results = Parallel(n_jobs=-1)(delayed(rerun_scale_mismatch_task)( j) for  j in range(len(algo_list)))
rerun_loc_mismatch_best_df, rerun_loc_mismatch_full_df = extract_results(rerun_loc_mismatch_results)
rerun_scale_mismatch_best_df, rerun_scale_mismatch_full_df = extract_results(rerun_scale_mismatch_results)


# === Save all DataFrames to CSV ===
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

dfs = {
    "main_best_df": main_best_df,
    "main_full_df": main_full_df,
    "loc_best_df": loc_best_df,
    "loc_full_df": loc_full_df,
    "scale_best_df": scale_best_df,
    "scale_full_df": scale_full_df,
    "rerun_loc_mismatch_best_df": rerun_loc_mismatch_best_df,
    "rerun_loc_mismatch_full_df": rerun_loc_mismatch_full_df,
    "rerun_scale_mismatch_best_df": rerun_scale_mismatch_best_df,
    "rerun_scale_mismatch_full_df": rerun_scale_mismatch_full_df,
}

for name, df in dfs.items():
    df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

# === Save all variables to a pickle file ===
with open(os.path.join(output_dir, "all_results.pkl"), "wb") as f:
    pickle.dump(dfs, f)