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

# import sys
# from joblib import dump
# #dump(variables, 'variables.joblib')
# #loaded_variables = load('variables.joblib')
from tqdm import tqdm
# from ax.service.managed_loop import optimize
import warnings


#Running iteration 5
#Current best: PostDiff (c=0.23), score = 98.5 (the lower the better)
#Running simualtion on the following:
#PostDiff, c = 0.5, 0.6, 0.7, ...
#Eps TS, eps = xxxxx

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

@dataclass
class TestObjective:
    test_name: Literal['anova', 't_control', 't_constant', 'tukey']
    type1_error_constraint: float = 0.05
    power_constraint: float = 0.80
    test_type: Optional[Literal['greater', 'two-sided']] = None
    test_params: Dict[str, Any] = field(default_factory=dict)
    min_effect: float = 0.1

    def run(self, sim_result,  horizon=slice(None)):
        return getattr(sim_result, self.test_name)(horizon=horizon, **self.test_params)

    def compute_power(self, test_stat: np.ndarray, h1_mean_reward_dist: np.ndarray, h0_crit: np.ndarray,
                      sim_result: Any) -> np.ndarray:

        # Correct broadcasting shape
        crit = h0_crit[np.newaxis, :, *([np.newaxis] * (test_stat.ndim - 2))]

        if self.test_name == 'anova':
            return np.mean(test_stat > crit, axis=(0,2)) #here the crit is p_value. can change in future

        elif self.test_name == 't_control' or self.test_name == 't_constant':
            if self.test_name == 't_constant':
                diffs = h1_mean_reward_dist - self.test_params['constant_thres']
            else:
                control = h1_mean_reward_dist[:, [0]]  # shape (n_rep, 1)
                diffs = h1_mean_reward_dist[:, 1:] - control  # shape: (n_rep, n_arm-1)

            if self.test_type == 'greater':
                filter_mask = diffs >= self.min_effect
                masked_stat = np.where( filter_mask[:, np.newaxis, :],  # condition
                    test_stat > crit,  # keep where mask is True
                    np.nan)  # replace False with nan
                power = np.nanmean(masked_stat, axis=(0, 2))  # mean over valid entries only

            elif self.test_type == 'two-sided':
                filter_mask = np.abs(diffs) >= self.min_effect
                masked_stat = np.where(filter_mask[:, np.newaxis, :],  # condition
                                       np.abs(test_stat) > crit,  # keep where mask is True
                                       np.nan)  # replace False with nan
                power = np.nanmean(masked_stat, axis=(0, 2))
            else:
                raise ValueError("Unsupported test_type")

        elif self.test_name == 'tukey':  #TODO: check if code below is correct

            n_rep, horizon_len, n_arm, _ = test_stat.shape
            reward_diffs = h1_mean_reward_dist[:, :, np.newaxis] - h1_mean_reward_dist[:, np.newaxis,:]  # (n_rep, n_arm, n_arm)
            abs_diffs = np.abs(reward_diffs)

            if self.test_type == 'greater':
                best_idx = np.argmax(h1_mean_reward_dist, axis=1)
                stat_masked = []

                for i in range(n_rep):
                    best = best_idx[i]
                    diffs_from_best = h1_mean_reward_dist[i, best] - h1_mean_reward_dist[i, :]
                    valid_targets = (diffs_from_best >= self.min_effect) & (np.arange(n_arm) != best)
                    stat_row = test_stat[i, :, best, :]  # shape: (horizon, n_arm)
                    masked = np.where(valid_targets[np.newaxis, :], stat_row > h0_crit[np.newaxis, :], np.nan)
                    stat_masked.append(masked)

                power = np.nanmean(np.stack(stat_masked), axis=(0, 1))

            elif self.test_type == 'two-sided':
                upper_mask = np.triu(np.ones((n_arm, n_arm), dtype=bool), k=1)
                stat_masked = []

                for i in range(n_rep):
                    valid_pairs = abs_diffs[i] >= self.min_effect
                    stat_sub = test_stat[i, :, upper_mask]  # shape: (horizon, n_pairs)
                    valid_mask = valid_pairs[upper_mask]
                    masked = np.where(valid_mask[np.newaxis, :], np.abs(stat_sub) > h0_crit[np.newaxis, :], np.nan)
                    stat_masked.append(masked)

                power = np.nanmean(np.stack(stat_masked), axis=(0, 1))

            else:
                raise ValueError("Unsupported test_type for tukey")

        else:
            raise NotImplementedError(f"Power not implemented for test: {self.test_name}")

        return power

def compute_test_quantile(result, test_objective, horizon_axis: int, q: float) -> np.ndarray:
    """
    Compute the quantile of a test statistic across all axes except the horizon axis.

    Args:
        result: SimResult object
        test_objective: TestObjective object (with test_name, test_type, test_params, etc.)
        horizon_axis: int – Axis index corresponding to horizon
        q: float – Quantile to compute (e.g., 0.95)

    Returns:
        crit_val: np.ndarray of shape (horizon_len,)
    """
    test_name = test_objective.test_name
    test_type = test_objective.test_type
    test_params = test_objective.test_params

    # Get test statistic: shape = (n_rep, horizon, ...)
    test_stat = getattr(result, test_name)(horizon=slice(None), **test_params)

    # Apply abs if needed (for two-sided t_control or t_constant)
    if test_name in {'t_control', 't_constant'} and test_type == 'two-sided':
        test_stat = np.abs(test_stat)

    # Move horizon axis to front
    stat_reordered = np.moveaxis(test_stat, horizon_axis, 0)  # shape: (horizon, ...)

    # Flatten all axes except horizon
    horizon_len = stat_reordered.shape[0]
    flattened = stat_reordered.reshape(horizon_len, -1)  # shape: (horizon, flattened_dims)

    # Quantile along flattened dims
    crit_val = np.quantile(flattened, q=q, axis=1)  # shape: (horizon,)

    return crit_val

#Archived because 'objective' is re-formatted. Now there's a cleaer test part, no need to decompose
# def decompose_objective_dict_for_test(objective):
#     alpha_dict = {k: v['alpha'] for k, v in objective.items()
#                   if k not in ['BAI', 'reward', 'step_cost'] and v['include']}
#
#     test_param_dict = {
#         k: (
#                 {
#                     pname.removeprefix('test_param_'): pval
#                     for pname, pval in v.items()
#                     if pname.startswith('test_param')
#                 } or {}
#         )
#         for k, v in objective.items()
#         if k not in ['BAI', 'reward', 'step_cost'] and v['include']
#     }
#
#     return alpha_dict, test_param_dict

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


    # Results
    h0_critical_values = {}  # e.g., 'anova': []

    test_objective = objective['test_objective']

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

    crit_val = compute_test_quantile(
        result=result,
        test_objective=test_objective, #TODO: think what if it is a list??
        horizon_axis=sw.arr_axis['horizon'],
        q=1-test_objective.type1_error_constraint,  # TODO: align direction for p-v and test_stat
    )  # shape: (horizon,)

    h0_critical_values[test_objective.test_name] = crit_val

    # for test_name in alpha_dict: #TODO: keep the loop for now but need to revise (in a differnt way)
    #     # Get test statistic: shape = (n_rep, horizon, ...)
    #     crit_val = compute_test_quantile(
    #         result=result,
    #         test_name=test_name,
    #         horizon_axis=sw.arr_axis['horizon'],
    #         q =  alpha_dict[test_name], #TODO: align direction for p-v and test_stat
    #         test_params = test_param_dict[test_name],
    #     ) # shape: (horizon,)
    #
    #     h0_critical_values[test_name] = crit_val
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



def get_objective_score(res_dist, test_name, objective, h0_critical_values, hyperparams, h1_reward_dist):
    """
    Compute the final objective score, score SD, number of steps, and reward at that step.
    #TODO: a function that loop on each test
    #TODO: check and give warning if cost step is too low (and infinity is preferred , or just try max step

    #TODO: add it (flatten reward mean) directly in results

    Args:
        res_dist: result object with .anova(slice) and .mean_reward
        test_name: str, e.g., "anova"
        objective: dict defining constraints and weights
        h0_critical_values: dict[test_name] -> array of shape (mu, horizon)
        hyperparams: object with at least .n_rep and .horizon

    Returns:
        dict with:
            - obj_score: float
            - obj_score_sd: float
            - n_step: float (median)
            - reward: float (mean reward at that step)
    """

    test_objective = objective['test_objective']

    # Step 1: Calculate power under H1
    test_stat = getattr(res_dist, test_name)(horizon=slice(None), **test_objective.test_params)
    power = test_objective.compute_power(test_stat=test_stat, h1_mean_reward_dist=h1_reward_dist['p'], h0_crit=h0_critical_values[test_objective.test_name], sim_result=res_dist)

    # Step 2: Determine minimum step that satisfies power constraint (with noise)
    thres = test_objective.power_constraint
    n_rep = hyperparams.n_rep
    horizon = hyperparams.horizon

    noise = np.random.normal(
        loc=0, scale=np.sqrt(thres * (1 - thres) / n_rep), size=(1,hyperparams.n_rep)
    )

    # steps until constraint is exceeded
    n_step_dist = horizon - np.sum(power[:,np.newaxis] > (thres + noise), axis=0)  # shape: (mu,)

    # Step 3: Compute reward at selected step
    mean_reward = np.mean(res_dist.mean_reward, axis=0).flatten()  # shape: (horizon,)
    reward_at_n_step = mean_reward[n_step_dist-1]

    n_step = np.median(n_step_dist)
    if n_step == horizon:
        # if exceed horizon_max, set reward = 0 as penalty
        warnings.warn("Power threshold may be too hard to achieve: n_step exceeds max horizon. ")
        reward_at_n_step = power[-1] - thres


    # Step 4: Compute objective score
    obj_score_dist = (
        reward_at_n_step * n_step_dist +
        objective['step_cost'] * n_step_dist
    )

    return {
        "obj_score": np.mean(obj_score_dist),
        "obj_score_sd": np.std(obj_score_dist),
        "n_step": np.median(n_step_dist),
        "reward": mean_reward[int(np.median(n_step_dist-1))],
    }