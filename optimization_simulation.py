import numpy as np
# import pandas as pd
#
# import bayes_model as bm
import policy as pol
import sim_wrapper as sw
# from itertools import permutations
# import copy
# from scipy.special import logit
# from scipy.special import expit
# import os
# #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import sys
# from joblib import dump
# #dump(variables, 'variables.joblib')
# #loaded_variables = load('variables.joblib')
from tqdm import tqdm
# from ax.service.managed_loop import optimize



#Running iteration 5
#Current best: PostDiff (c=0.23), score = 98.5 (the lower the better)
#Running simualtion on the following:
#PostDiff, c = 0.5, 0.6, 0.7, ...
#Eps TS, eps = xxxxx
def compute_test_quantile(result, test_name: str, horizon_axis: int, q: float):
    """
    Compute the quantile of a test statistic across all axes except the horizon axis.

    Args:
        result: sim_wrapper.SimResult object
        test_name (str): Name of the test (e.g., "anova", "tukey")
        horizon_axis (int): Axis index corresponding to horizon
        q (float): Quantile to compute (e.g., 0.95)

    Returns:
        crit_val: np.ndarray of shape (horizon_len,)
    """
    # Get test statistic array: shape = (n_rep, horizon, ...)
    test_stat = getattr(result, test_name)(slice(None))

    # Move horizon axis to front
    stat_reordered = np.moveaxis(test_stat, horizon_axis, 0)  # shape: (horizon, ...)

    # Reshape all other axes into one
    horizon_len = stat_reordered.shape[0]
    flattened = stat_reordered.reshape(horizon_len, -1)  # shape: (horizon, other_dims)

    # Compute quantile over all dims except horizon
    crit_val = np.quantile(flattened, q=q, axis=1)  # shape: (horizon,)

    return crit_val

def run_h0_simulation(
    algo,#TODO: no need for algo right?
    algo_param,
    model,
    h0_reward_setting,
    objective,
    hyperparams,
):
    """
    TODO: make sure this also work for TUKEY at lest. Better if also work for BAI
    Run a SINGLE H0 simulation for a given algorithm and parameter.

    Returns:
        - h0_critical_values: dict of test_name -> array (mu × para × horizon × ...)
        - sim_record: list of (mu, para) tuples (for interpolation/ref checking)
    """

    reward_model_name = model.__name__
    #scale = h0_setting.get('scale', 1)
    alpha_dict = {k: v['alpha'] for k, v in objective.items()
                  if k not in ['BAI', 'reward', 'step_cost'] and v['include']}

    # Results
    h0_critical_values = {}  # e.g., 'anova': []


    bandit = pol.StoBandit(
        reward_model=sw.RewardModel(
            model=model,
            parameters=h0_reward_setting
        )
    )


    result = sw.run_simulation(
        policy=getattr(bandit, algo),
        algo_para=algo_param,
        hyperparams=hyperparams
    )

    for test_name in alpha_dict:
        # Get test statistic: shape = (n_rep, horizon, ...)
        crit_val = compute_test_quantile(
            result=result,
            test_name=test_name,
            horizon_axis=sw.arr_axis['horizon'],
            q =  alpha_dict[test_name] #TODO: align direction for p-v and test_stat
        ) # shape: (horizon,)

        h0_critical_values[test_name] = crit_val
 # # list of (horizon, ...) per para
 #
 #    # Stack per mu
 #    for test_name in mu_test_stats:
 #        h0_critical_values[test_name].append(mu_test_stats[test_name])  # list of [n_para × (horizon, ...)]
 #
 #    # Convert to np.arrays: shape = (n_mu, n_para, horizon, ...)
 #    for k in h0_critical_values: #TODO: is this neeeded??
 #        h0_critical_values[k] = np.array(h0_critical_values[k])
    return h0_critical_values



# result = np.zeros_like(evaluations, dtype=int)
#
# unique_indices = np.unique(experience_index)
# for idx in unique_indices:
#     mask = experience_index == idx
#     group_evals = evaluations[mask]
#     max_idx_in_group = np.flatnonzero(group_evals == group_evals.max())
#
#     # Set 1 for all maxes (if tie); or just first max if preferred
#     result[np.flatnonzero(mask)[max_idx_in_group]] = 1