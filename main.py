import numpy as np
import pandas as pd

import bayes_model as bm
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



def calculate_objective_for_single_algorithm(algo_param,algo,model,h0_reward_setting,objective,hyperparams):
    global sim_result_keeper

    h0_critical_values = opt_sim.run_h0_simulation(  # TODO: for now need to edit Tukey functionality
        algo=algo,
        algo_param=algo_param,
        model=model,
        h0_reward_setting=h0_reward_setting,
        objective=objective,
        hyperparams=hyperparams,
    )

    # get reward hist and run h1 simulation
    base_shape = next(iter(h1_reward_dist.values())).shape
    new_shape = (hyperparams.horizon,) + base_shape
    h1_reward_hist = np.random.binomial(**h1_reward_dist, size=new_shape)
    h1_reward_hist = np.moveaxis(h1_reward_hist, 0, 1)

    # H1 simulation
    res_dist = sw.run_simulation(policy=getattr(bandit, algo),
                                 algo_para=algo_param,
                                 hyperparams=hyperparams,
                                 reward_hist=h1_reward_hist)

    result = opt_sim.get_objective_score(
        res_dist=res_dist,
        test_name='anova',
        objective=objective,
        h0_critical_values=h0_critical_values,
        hyperparams=hyperparams
    )

    sim_result_keeper[(algo, algo_param)] = result
    return {'obj_score': (result['obj_score'], result['obj_score_sd'])}

class AxObjectiveEvaluator:
    def __init__(self, algo, model, h0_reward_setting, h1_reward_dist, objective, hyperparams, bandit, sim_result_keeper):
        self.algo = algo
        self.model = model
        self.h0_reward_setting = h0_reward_setting
        self.h1_reward_dist = h1_reward_dist
        self.objective = objective
        self.hyperparams = hyperparams
        self.bandit = bandit
        self.sim_result_keeper = sim_result_keeper

    def __call__(self, params):
        algo_param = params["algo_para"]
        test_objective = objective["test_objective"]
        # H0 simulation
        h0_critical_values = opt_sim.run_h0_simulation(
            algo=self.algo,
            algo_param=algo_param,
            model=self.model,
            h0_reward_setting=self.h0_reward_setting,
            objective=self.objective,
            hyperparams=self.hyperparams,
        )

        # Generate H1 reward histogram
        base_shape = next(iter(self.h1_reward_dist.values())).shape
        new_shape = (self.hyperparams.horizon,) + base_shape
        h1_reward_hist = np.random.binomial(**self.h1_reward_dist, size=new_shape)
        h1_reward_hist = np.moveaxis(h1_reward_hist, 0, 1)

        # Run H1 simulation
        res_dist = sw.run_simulation(
            policy=getattr(self.bandit, self.algo),
            algo_para=algo_param,
            hyperparams=self.hyperparams,
            reward_hist=h1_reward_hist,
        )
        #alpha_dict,_ =opt_sim.decompose_objective_dict_for_test(objective)
        # Evaluate score
        result = opt_sim.get_objective_score(
            res_dist=res_dist,
            test_name=test_objective.test_name,
            objective=self.objective,
            h0_critical_values=h0_critical_values,
            hyperparams=self.hyperparams,
            h1_reward_dist=self.h1_reward_dist,
        )

        # Track result externally
        self.sim_result_keeper[(self.algo, algo_param)] = result

        return {"obj_score": (result["obj_score"], result["obj_score_sd"])}

def optimize_algorithm(model,bandit,h0_reward_setting,h1_reward_dist,objective,hyperparams,algo):
    evaluator = AxObjectiveEvaluator(
        algo=algo,
        model=model,
        h0_reward_setting=h0_reward_setting,
        h1_reward_dist=h1_reward_dist,
        objective=objective,
        hyperparams=hyperparams,
        bandit=bandit,
        sim_result_keeper=sim_result_keeper,
    )

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "algo_para",
                "type": "range",
                "bounds": [0.0, 1.0],
                "value_type": "float",
            },
        ],
        evaluation_function=evaluator,  # callable class instance
        objective_name="obj_score",  # must match return key
        minimize=False,
        total_trials=hyperparams.n_opt_trials,
    )

    return best_parameters, values, experiment, model


def extract_results(config_setting, algo_list, all_results):
    """
    Extract best_df and full_df from simulation results.

    Parameters:
    - config_setting: List of dicts, each dict contains config parameters
    - algo_list: List of algorithm names (indexed by j)
    - all_results: Dict with keys (i, j), each value has 'best_result_log', 'params', and 'result_log'

    Returns:
    - best_df: one row per (i,j) with best result
    - full_df: one row per (i,j,param) with all results
    """
    records_best = []
    records_full = []

    for i, config in enumerate(config_setting):
        for j, algo_name in enumerate(algo_list):
            result_entry = all_results[(i, j)]

            # Best result
            best = result_entry['best_result_log']
            best_param = result_entry['params']
            records_best.append({
                **config,
                'algo_name': algo_name,
                'best_algo_param': best_param,
                'obj_score': best['obj_score'],
                'obj_score_sd': best['obj_score_sd'],
                'n_step': best['n_step'],
                'reward': best['reward']
            })

            # All results
            for (algo_name_inner, param), result in result_entry['result_log'].items():
                records_full.append({
                    **config,
                    'algo_name': algo_name_inner,
                    'algo_param': param,
                    'obj_score': result['obj_score'],
                    'obj_score_sd': result['obj_score_sd'],
                    'n_step': result['n_step'],
                    'reward': result['reward']
                })

    best_df = pd.DataFrame(records_best)
    full_df = pd.DataFrame(records_full)

    return best_df, full_df

def generate_config(h0_loc=0.2, h1_loc=0.5, h1_scale=0.15, test_name:Literal['anova', 't_control', 't_constant', 'tukey'] = 'anova', test_const=0.8, step_cost=-1):

    #Part 1: hyperparameters. This contains ALL BUT reward-distribution and reward_and_cost_objective function related parameters
    hyperparams = sw.HyperParams(
        n_rep=5000, #Number of simulation replications (for more accurate result)
        n_arm=3,  # TODO: check if this is needed
        burn_in=5,  # TODO*:make it burnin each arm round rubin

        # Only one of below need to be specified. 'batch_size' or 'fast_batch_epsilon'. Currently, we prioritize using 'fast_batch_epsilon'
        batch_size=None, # How often the algorithm is updates.
                         # If batch = 3, it means we update the allocation algorithm (based on past data) once we collect 3 more data

        fast_batch_epsilon=0.1, # How often the algorithm is updates.
                                 # If fast_xxx = 0.1, it means we update the algorithm once we have 10% more data

        horizon=2000,  # max horizon to try in simulation

        #horizon_check_points=sw.generate_quadratic_schedule(2000), #can ignore for now... TODO: see where I used it (and delete if not)
        # can set tuning_density to make the schedule denser / looser
        n_opt_trials = 10

    )

    # Part 2: reward_and_cost_objective function related parameters. In GUI, people first specify whether they want to include certain tests,
    #         then they define the weight/cost/constraint on each thing.
    #         I think we want to hide tests that's not been included be the user.
    #
    #         For each test, at least they need to specify type I and II error constraint. In addtion:
    #         [ANOVA]: no need to specify other things.
    #         [TUKEY]: need to define how they want to calculate type II error. (the pairwise test produces many results) Let's discuss later
    #
    #         For the two below, user can choose one-tailed or two-tailed. (just a bool: is_one_tail in python script), and min_effect.
    #         if min_effect = 0.1, if means we don't calculate type II error if the arm is not far enough (>0.1) from the control/constant
    #         [t_control]: compare all arm to the control arm. Need to specify which arm in simulation is the control (by default it is arm 1)
    #         [t_constant]: compare all arm against a fixed constant. Need to specify the constant value (e.g. 0.2, 0.5, etc.).

    test_objective = opt_sim.TestObjective(
        test_name=test_name,
        type1_error_constraint=0.05,
        power_constraint=test_const,
        test_type='greater',
        test_params={} # set 'test_param_constant_thres' for t_constant
    )

    if test_name == 't_constant':
        test_objective = opt_sim.TestObjective(
            test_name=test_name,
            type1_error_constraint=0.05,
            power_constraint=test_const,
            test_type='greater',
            test_params={'constant_thres':h1_loc}  # set 'test_param_constant_thres' for t_constant
        )

    objective = {
        'test_objective': test_objective,
        'step_cost': step_cost,
    }

    # Part 3: distribution.
    #         [reward model] First specify the parametric family of reward distribution: i.e. Bernoulli, Normal
    #
    #         [H0 setting]
    #         Option 1: then they specify (a list) of distributions on H0 (then we simualte in those settings and interpolate critical region
    #         if the truth is in between). The more settings, the more accurate result we get, and the slower the simulation
    #         For H0 we can have a button to let user 'add another setting'
    #
    #         Option 2: we can just let they specify a range. e.g. Normal, mu between [0.03, 0.2], sigma between [0.01, 0.02]
    #         Let's work on Option 2 for now. I think it is better.
    #
    #         [H1 Setting] For H1, people specify the reward distribution for each arm
    model = np.random.binomial

    h0_reward_settings = {
        'p': [h0_loc]*hyperparams.n_arm,
        # mean reward for h0. In the future, can also make it a list of h0 conditions. (if a list): the result will be capped at the boundary and use interpolation between
        'n': [1]*hyperparams.n_arm,
    }

    h1_reward_dist = {  # make sure mean reward doesn't exceed 1. how to do that?
        # 'p': np.tile([0.4, 0.5, 0.6], (hyperparams.n_rep, 1)),
        'p': np.clip(np.random.normal(loc=h1_loc, scale=h1_scale, size=(hyperparams.n_rep, hyperparams.n_arm)), 0.05, 0.95),
        # TODO: edit it later, make it align with hyper.n_rep automatically
        'n': np.tile([1]*hyperparams.n_arm, (hyperparams.n_rep, 1))
    }

    bandit = pol.StoBandit(
        reward_model=sw.RewardModel(
            model=model,
            parameters=h0_reward_settings
            # TODO: this is fine for H1 because it use reward_hist directly. But consider modify the logic flow
        )
    )

    return model,bandit,h0_reward_settings,h1_reward_dist,objective,hyperparams





#TODO: set parameter for ax.platform

#TODO: in paper, can define broadly how user can plug in their own test
#TODO: make it test objective + other (for easier loop)

#TODO: add a custom function (user define a function, input is reward and action hist, output is test. the code help them run it in h0 to get crit, and then evalaute it in objective function


#ok for now it's make the input for h0_setting and h1_dist  both take standard input that align with 'model' input
#for h0_setting, it is a single setting (we control FPR in that setting)
#for h1_dist, it can be a mix (distribution) of many settings. can also be a single setting.


algo = 'eps_ts'
algo = 'ts_postdiff_ur'

np.random.seed(0)
algo_list = ['eps_ts', 'ts_postdiff_ur']
config_setting = [
    {'h0_loc':0.5,'h1_loc':0.5,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
    #{'h0_loc':0.5,'h1_loc':0.5,'h1_scale':0.15,'test_name':'tukey','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.5,'h1_loc':0.5,'h1_scale':0.15,'test_name':'t_control','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.5,'h1_loc':0.5,'h1_scale':0.15,'test_name':'t_constant','test_const':0.8,'step_cost':-1},

    {'h0_loc':0.5,'h1_loc':0.5,'h1_scale':0.2,'test_name':'anova','test_const':0.8,'step_cost':-1},
    #{'h0_loc':0.5,'h1_loc':0.5,'h1_scale':0.2,'test_name':'tukey','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.5,'h1_loc':0.5,'h1_scale':0.2,'test_name':'t_control','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.5,'h1_loc':0.5,'h1_scale':0.2,'test_name':'t_constant','test_const':0.8,'step_cost':-1},

    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 'anova', 'test_const': 0.8, 'step_cost': -0.7},
    #{'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 'tukey', 'test_const': 0.8, 'step_cost': -0.7},
    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 't_control', 'test_const': 0.8, 'step_cost': -0.7},
    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 't_constant', 'test_const': 0.8, 'step_cost': -0.7},

    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 'anova', 'test_const': 0.8, 'step_cost': -1.5},
    #{'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 'tukey', 'test_const': 0.8, 'step_cost': -1.5},
    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 't_control', 'test_const': 0.8, 'step_cost': -1.5},
    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 't_constant', 'test_const': 0.8, 'step_cost': -1.5},

    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 'anova', 'test_const': 0.8, 'step_cost': -3},
    #{'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 'tukey', 'test_const': 0.8, 'step_cost': -1.5},
    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 't_control', 'test_const': 0.8, 'step_cost': -3},
    {'h0_loc': 0.5, 'h1_loc': 0.5, 'h1_scale': 0.15, 'test_name': 't_constant', 'test_const': 0.8, 'step_cost': -3},

    {'h0_loc':0.1,'h1_loc':0.1,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.2,'h1_loc':0.2,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.3,'h1_loc':0.3,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.4,'h1_loc':0.4,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.6,'h1_loc':0.6,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.7,'h1_loc':0.7,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.8,'h1_loc':0.8,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
    {'h0_loc':0.9,'h1_loc':0.9,'h1_scale':0.15,'test_name':'anova','test_const':0.8,'step_cost':-1},
]

all_results = {}
for i in range(len(config_setting)):
    for j in range(len(algo_list)):
        sim_result_keeper = {}
        algo = algo_list[j]
        model, bandit, h0_reward_settings, h1_reward_dist, objective, hyperparams = generate_config(**config_setting[i])
        params, val, _, _ = optimize_algorithm(model, bandit, h0_reward_settings, h1_reward_dist, objective,
                                               hyperparams, algo=algo)
        all_results[(i,j)] = {'params': params, 'val': val, 'result_log': sim_result_keeper, 'best_result_log': sim_result_keeper[(algo,params['algo_para'])]}



best_df, full_df = extract_results(config_setting, algo_list, all_results)

best_df.to_csv('best_results3.csv')
filtered_df.to_csv('filtered_results3.csv')
full_df.to_csv('full_results3.csv')
best_df.to_csv('~/best_results3.csv')
full_df.to_csv('~/full_results3.csv')
