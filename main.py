import numpy as np
import pandas as pd
import warnings
import pickle
import bayes_vector_ops as bm
import policy as pol
import sim_wrapper as sw
from itertools import permutations
import copy
from scipy.special import logit
from scipy.special import expit
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from joblib import dump
#dump(variables, 'variables.joblib')
#loaded_variables = load('variables.joblib')
from tqdm import tqdm
from ax.service.managed_loop import optimize
import optimization_simulation as opt_sim
from functools import partial
from typing import Optional, Literal, Dict, Any
import logging
from tqdm import tqdm
from joblib import Parallel, delayed

from simulation_configurator import SimulationConfig

logging.getLogger("ax.service.managed_loop").setLevel(logging.CRITICAL)
logging.getLogger('ax.generation_strategy.dispatch_utils').setLevel(logging.CRITICAL)

#TODO: change min_effect


#for vm?
import sys
import os

# Add project root (where bayes_vector_ops.py is) to sys.path
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
"""
Process draft
#step 1: get H0 table (algo_para by n_step by mu0)

#step 2: get simulation result (reward_hist and action_hist)
#simulate under H1??

#step 3: process result. test, power, etc.

#step 4: bootstrap objective score
"""

"""
TODO: 1. power bug
2. constraint (0.1), and power calculation function
3. get optimizted parameter and try it under 0.5

for TUKEy, probably need change how we calculate critical region...... (ignore the diagnal)
"""

#TODO: what is the distinction between horizon_check_point and fast batch? can we ... merge them? I htink slightly different...
# Can also just use horizon_check_points = range(max_horizon), which will simulate every single step





def optimize_algorithm(sim_config:SimulationConfig,algo,include_benchmarks=True):
    evaluator = sw.AxObjectiveEvaluator(
        algo=algo,
        config_setting = config_setting,
        hyperparams=hyperparams,
        sim_result_keeper={},
    )

    #get result for TS
    #np.random.seed(0)
    if include_benchmarks:
        hyperparams.horizon = int(hyperparams.horizon * 2)
        evaluator({'algo_para':0})
        hyperparams.horizon = int(hyperparams.horizon /2)
        evaluator({'algo_para':1})


    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "algo_para",
                "type": "range",
                "bounds": [0.01, 0.99],
                "value_type": "float",
            },
        ],
        evaluation_function=evaluator,  # callable class instance
        objective_name="obj_score",  # must match return key
        minimize=False,
        total_trials=hyperparams.n_opt_trials,
    )

    return best_parameters, values, experiment, model, evaluator.sim_result_keeper






#TODO: set parameter for ax.platform

#TODO: in paper, can define broadly how user can plug in their own test
#TODO: make it test objective + other (for easier loop)

#TODO: add a custom function (user define a function, input is reward and action hist, output is test. the code help them run it in h0 to get crit, and then evalaute it in objective function


#ok for now it's make the input for h0_setting and h1_dist  both take standard input that align with 'model' input
#for h0_setting, it is a single setting (we control FPR in that setting)
#for h1_dist, it can be a mix (distribution) of many settings. can also be a single setting.


# np.random.seed(0)
hyperparams = sw.SimulationConfig(
        # User Input
        n_rep=10000, # user_input: Number of simulation replications (for more accurate result)
        n_arm=3,  # TODO: check if this is needed
        burn_in=6,  #* TODO*:make it burnin each arm round rubin
        # Only one of below need to be specified. 'batch_size' or 'fast_batch_epsilon'. Currently, we prioritize using 'fast_batch_epsilon'
        batch_scaling_rate=0.1, # How often the algorithm is updates.
                                # If fast_xxx = 0.1, it means we update the algorithm once we have 10% more data
        horizon=1000,  # max horizon to try in simulation
        #horizon_check_points=sw.generate_quadratic_schedule(2000), #can ignore for now... TODO: see where I used it (and delete if not)
        # can set tuning_density to make the schedule denser / looser
        n_opt_trials = 12 #TODO: optimize for this in our code
    )
algo_list = ['ts_adapt_explor','ts_postdiff_top','eps_ts', 'ts_postdiff_ur','ts_probclip']
algo_list = ['ts_postdiff_top','eps_ts', 'ts_postdiff_ur']
config_setting = [ #TODO: make it a data class
    {'h1_loc': 0.5, 'h1_scale': 0.15,  'test_name': 'anova',    'test_const': 0.8, 'step_cost': -1},
    #{'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 'tukey',    'test_const': 0.8, 'step_cost': -1},
    {'h1_loc': 0.5, 'h1_scale': 0.15,  'test_name': 't_control','test_const': 0.8, 'step_cost': -1, 'test_type': 'greater'},
    {'h1_loc': 0.5, 'h1_scale': 0.15,  'test_name': 't_constant','test_const': 0.8,'step_cost': -1, 'test_type': 'greater'},
    # {'h1_loc': 0.5, 'h1_scale': 0.15,  'test_name': 't_control','test_const': 0.8, 'step_cost': -1, 'test_type': 'two-sided'},
    # {'h1_loc': 0.5, 'h1_scale': 0.15,  'test_name': 't_constant','test_const': 0.8,'step_cost': -1, 'test_type': 'two-sided'},

    {'h1_loc': 0.5, 'h1_scale': 0.2, 'test_name': 'anova',    'test_const': 0.8, 'step_cost': -1},
    # #{'h1_loc': 0.5, 'h1_scale': 0.2,  'test_name': 'tukey',    'test_const': 0.8, 'step_cost': -1},
    {'h1_loc': 0.5, 'h1_scale': 0.2, 'test_name': 't_control', 'test_const': 0.8, 'step_cost': -1,'test_type': 'greater'},
    {'h1_loc': 0.5, 'h1_scale': 0.2, 'test_name': 't_constant', 'test_const': 0.8, 'step_cost': -1,'test_type': 'greater'},
    # {'h1_loc': 0.5, 'h1_scale': 0.2, 'test_name': 't_control', 'test_const': 0.8, 'step_cost': -1,
    #  'test_type': 'two-sided'},
    # {'h1_loc': 0.5, 'h1_scale': 0.2, 'test_name': 't_constant', 'test_const': 0.8, 'step_cost': -1,
    #  'test_type': 'two-sided'},

    # {'h1_loc': 0.5, 'h1_scale': 0.1, 'test_name': 'anova',    'test_const': 0.8, 'step_cost': -1},
    # #{'h1_loc': 0.5, 'h1_scale': 0.1, 'test_name': 'tukey',    'test_const': 0.8, 'step_cost': -1},
    # {'h1_loc': 0.5, 'h1_scale': 0.1, 'test_name': 't_control','test_const': 0.8, 'step_cost': -1},
    # {'h1_loc': 0.5, 'h1_scale': 0.1, 'test_name': 't_constant','test_const': 0.8,'step_cost': -1},
]



task_list = [(i, j) for i in range(len(config_setting)) for j in range(len(algo_list))]
def run_task(i, j):

    algo = algo_list[j]
    params, val, _, _,sim_result_keeper = optimize_algorithm(config_setting[i], hyperparams, algo=algo)
    #return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
    return {'setting_index':i,
            **config_setting[i],
            'algo_name':algo,
            'algo_params':params['algo_para'],
            **sim_result_keeper.get((algo, params['algo_para'],frozenset(config_setting[i].items())), None),
            'all_results':sim_result_keeper
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

        for (algo, param,_), result in all_result_dict.items():
            entry = row_info.copy()
            entry['algo_name'] = algo
            entry['algo_param'] = param
            entry.update({k: float(v) for k, v in result.items()})
            all_rows.append(entry)

    full_df = pd.DataFrame(all_rows)
    return best_df, full_df

results = Parallel(n_jobs=1)(delayed(run_task)(i, j) for i, j in task_list)
# best_df, full_df = extract_results(results)


# best_df.to_csv('~/best_df0703.csv')
# full_df.to_csv('~/full_df0703.csv')
#filtered_df.to_csv('filtered_results3.csv')
#full_df.to_csv('full_results3.csv')
#best_df.to_csv('~/best_results3.csv')
#full_df.to_csv('~/full_results3.csv')


"""
Sim part 2, single 
"""


def get_best_param(best_df, base_setting_dict, algo_name):
    # Copy the base dict and add algo_name for filtering
    query_dict = base_setting_dict.copy()
    query_dict['algo_name'] = algo_name

    # Create the mask for filtering rows
    mask = pd.Series(True, index=best_df.index)
    for key, value in query_dict.items():
        mask &= best_df[key] == value

    # Apply filter and extract the value
    filtered = best_df[mask]
    if not filtered.empty:
        return filtered.iloc[0]['algo_params']
    else:
        return None  # or raise an error if preferred


def mismatch_sim(base_setting_dict,variable,vary_list,algo_name,hyperparams):

    res_list = []
    vary_setting_dict = base_setting_dict.copy()


    #sim the best in base
    params, val, _, _, sim_result_keeper = optimize_algorithm(base_setting_dict,hyperparams, algo=algo_name)
    algo_param = params['algo_para']

    for j in vary_list:
        vary_setting_dict[variable] = j

        evaluator = opt_sim.AxObjectiveEvaluator(
            algo=algo_name,
            config_setting=vary_setting_dict,
            hyperparams=hyperparams,
            sim_result_keeper=sim_result_keeper,
        )
        res = evaluator({'algo_para': algo_param})

        # also sim for the best in this setting
        params, val, _, _, sim_result_keeper_temp = optimize_algorithm(vary_setting_dict, hyperparams, algo=algo_name)


        evaluator = opt_sim.AxObjectiveEvaluator(
            algo=algo_name,
            config_setting=base_setting_dict,
            hyperparams=hyperparams,
            sim_result_keeper=sim_result_keeper,
        )
        current_best_in_base_res = evaluator({**params})

        res_list.append({**base_setting_dict,
                         'algo_name': algo_name,
                         'variable': variable,
                         'var_value': j,
                         'param_base': algo_param,
                         'obj_score(param_base)': res['obj_score'][0],
                         'obj_score_std(param_base)': res['obj_score'][1],
                         'param_current_best': params['algo_para'],
                         'obj_score(param_current_best)': val[0]['obj_score'],
                         'obj_score_std(param_current_best)': np.sqrt(val[1]['obj_score']['obj_score']),
                         'obj_score(param_current_best_in_base_setting)': current_best_in_base_res['obj_score'][0],
                         'obj_score_std(param_current_best_in_base_setting)': current_best_in_base_res['obj_score'][1],
                         })




    return res_list, sim_result_keeper



test_names = ['anova', 't_control', 't_constant']
vary_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
vary_list = [0.2, 0.35, 0.5, 0.65, 0.8]
# Prepare configs
tasks = [
    {
        'base_setting_dict': {
            'h1_loc': 0.5,
            'h1_scale': 0.15,
            'test_name': test_name,
            'test_const': 0.8,
            'step_cost': -1
        },
        'variable': 'h1_loc',
        'vary_list': vary_list,
        'algo_name': algo_name
    }
    for test_name in test_names
    for algo_name in algo_list
]

# Parallel run
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(mismatch_sim)(
        base_setting_dict=task['base_setting_dict'],
        variable=task['variable'],
        vary_list=task['vary_list'],
        algo_name=task['algo_name'],
        hyperparams=hyperparams
    )
    for task in tasks
)


# all_results = []
#
# test_names = ['anova', 't_control', 't_constant']
# vary_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# for test_name in test_names:
#     for algo_name in algo_list:
#         base_setting_dict = {
#             'h1_loc': 0.5,
#             'h1_scale': 0.15,
#             'test_name': test_name,
#             'test_const': 0.8,
#             'step_cost': -1
#         }
#
#         res_list, sim_result_keeper = mismatch_sim(
#             base_setting_dict=base_setting_dict,
#             variable='h1_loc',
#             vary_list=vary_list,
#             algo_name=algo_name,
#             hyperparams=hyperparams
#         )
#
#         all_results.extend(res_list)



with open("results.pkl", "wb") as f:
    pickle.dump(results, f)