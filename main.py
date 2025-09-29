import numpy as np
import pandas as pd
import pickle
import bandit_algorithm as algorithm
import sim_wrapper as sw
import copy
import logging
from joblib import Parallel, delayed

from simulation_configurator import SimulationConfig
from test_procedure_configurator import TestProcedure, ANOVA, TControl, TConstant, Tukey
import bayes_vector_ops as bayes

import matplotlib.pyplot as plt
logging.getLogger("ax.service.managed_loop").setLevel(logging.CRITICAL)
logging.getLogger('ax.generation_strategy.dispatch_utils').setLevel(logging.CRITICAL)

import os
import itertools

# === Common runner ===
def run_task_common(
    sim_config_base,
    algo,
    algo_param_list=None,
    overrides: dict | None = None,
    test_proc: tuple | None = None,
):
    sim_config = copy.deepcopy(sim_config_base)

    # Apply overrides
    if overrides:
        for k, v in overrides.items():
            setattr(sim_config, k, v)

    # Handle test procedure
    if test_proc:
        sim_config.test_procedure = test_proc[0]
        sim_config.test_procedure.power_constraint = test_proc[1]

    sim_config.manual_init()

    params, val, _, _, sim_result_keeper = sw.optimize_algorithm(
        sim_config, algo=algo, algo_param_list=algo_param_list
    )

    return {
        "test": sim_config.test_procedure.test_signature,
        "step_cost": sim_config.step_cost,
        "mean_reward_dist_loc": sim_config.arm_mean_reward_dist_loc[0],
        "mean_reward_dist_scale": sim_config.arm_mean_reward_dist_scale[0],
        "algo_name": algo.__name__,
        "algo_param": params["algo_para"],
        **sim_result_keeper.get(
            (algo.__name__, params["algo_para"], sim_config.setting_signature), None
        ),
        "all_results": sim_result_keeper,
    }


def sweep_and_run(sweep_specs, base_config):
    """
    sweep_specs: list of dicts, e.g.
        [
            {"horizon": [4000, 8000]},
            {"algo": [Algo1, Algo2]},
            {"algo_param_list": [0.0, 0.2]},
            {"step_cost": [0.05, 0.1]}
        ]
    base_config: SimulationConfig (copied inside run_task_common)

    Returns: DataFrame of results
    """
    import itertools

    sweep_dict = {}
    for d in sweep_specs:
        sweep_dict.update(d)

    keys = list(sweep_dict.keys())
    value_lists = [sweep_dict[k] for k in keys]

    all_results = []
    for combo in itertools.product(*value_lists):
        overrides = {}
        meta = {}

        algo = None
        algo_param_list = None

        for k, v in zip(keys, combo):
            if k == "algo":
                algo = v
                meta[k] = v.__name__
            elif k == "algo_param_list":
                algo_param_list = [v]   # always wrap in list
                meta[k] = v
            elif isinstance(v, (int, float, str)):
                overrides[k] = v
                meta[k] = v
            else:
                overrides[k] = v
                meta[k] = f"option_{combo.index(v)}"

        # call run_task_common directly here
        result = run_task_common(
            base_config,
            algo=algo,
            algo_param_list=algo_param_list,
            overrides=overrides
        )

        all_results.append({**meta, **result})

    return pd.DataFrame(all_results)

def compute_objective(row, w: float):
    n = row["n_step"]
    reward = row["regret_per_step"]
    return n - (w * reward * n / np.log(n))

def compute_baseline(df, w_values=range(1, 16)):
    baseline = {}
    for w in w_values:
        scores = df.apply(lambda r: compute_objective(r, w), axis=1)
        baseline[w] = scores.min()
    return baseline

def select_curves_relative(df, selectors, w_values=range(1, 16)):
    baseline = compute_baseline(df, w_values)
    results = {}

    for sel in selectors:
        algo_name, mode, value = sel
        subset = df[df["algo_name"] == algo_name]

        if mode == "param":
            # fixed param curve
            sub = subset[subset["algo_param"] == value]
            if sub.empty:
                raise ValueError(f"No rows for {algo_name} with param={value}")
            row = sub.iloc[0]
            curves = []
            for w in w_values:
                raw = compute_objective(row, w)
                rel = raw - baseline[w]
                curves.append({"w": w, "obj_rel": rel, "obj_abs": raw})
            label = f"{algo_name} (param={value})"
            results[label] = pd.DataFrame(curves)

        elif mode == "w":
            # step 1: find best param at reference w=value
            w_ref = value
            subset["obj_tmp"] = subset.apply(lambda r: compute_objective(r, w_ref), axis=1)
            best = subset.loc[subset["obj_tmp"].idxmin()]  # <-- minimize objective
            chosen_param = best["algo_param"]

            # step 2: plot curve for this fixed param across all w
            sub = subset[subset["algo_param"] == chosen_param].iloc[0]
            curves = []
            for w in w_values:
                raw = compute_objective(sub, w)
                rel = raw - baseline[w]
                curves.append({"w": w, "obj_rel": rel, "obj_abs": raw})
            label = f"{algo_name} opt for w={w_ref} (param={chosen_param})"
            results[label] = pd.DataFrame(curves)

        else:
            raise ValueError("mode must be 'param' or 'w'")

    return results

def plot_curves(curves):
    plt.figure(figsize=(8, 6))
    for label, curve in curves.items():
        # format: strip old text and only keep param value if present
        if "param=" in label:
            algo, rest = label.split("(", 1)
            param_val = rest.split("=")[1].replace(")", "").strip()
            try:
                param_val = f"{float(param_val):.2f}"
            except ValueError:
                pass
            label = f"{algo.strip()} ({param_val})"
        elif "opt for w" in label and "param=" in label:
            # e.g. "EpsTS opt for w=3 (param=0.25)"
            before, rest = label.split("(param=")
            param_val = rest.replace(")", "").strip()
            try:
                param_val = f"{float(param_val):.2f}"
            except ValueError:
                pass
            label = f"{before.strip()} ({param_val})"

        plt.plot(curve["w"], curve["obj_rel"], marker="o", label=label)

    plt.axhline(0, color="grey", linestyle="--")
    plt.xlabel("w")
    plt.ylabel("Relative objective score (vs. baseline)")
    plt.legend()
    plt.tight_layout()
    plt.show()




"""
Page 1: User input on simualtion setting
(reward distribution, test, horizon etc)
"""
sim_config_base = SimulationConfig(
    n_rep=20000,
    n_arm=3,
    horizon=1500,  # max horizon to try in simulation
    burn_in_per_arm=5,
    # horizon_check_points=sw.generate_quadratic_schedule(2000), #can ignore for now... TODO: see where I used it (and delete if not)
    # can set tuning_density to make the schedule denser / looser
    n_opt_trials=None,  # TODO: optimize for this in our code
    arm_mean_reward_dist_loc = 0.5,
    arm_mean_reward_dist_scale = 0.15,
    test_procedure=ANOVA(),
    step_cost=0.05,
    reward_evaluation_method='reward',
    vector_ops=bayes.BackendOpsNP()
)
# Define sweeps
sweeps = [
    {"algo": [algorithm.EpsTS]},
    {"algo_param_list": list(map(float, np.linspace(0.0, 1.0, 41)))},  # each wrapped in list
    {"arm_mean_reward_dist_loc": [0.2,0.25,0.3,0.325,0.35,0.375,0.4,0.45,0.5]},
]


df = sweep_and_run(sweeps, sim_config_base)
#TODO: change iter = 50000 and re-run..
#df.to_csv('results/bernoulli_misspecification.csv')
"""
Part 2: interactive result page

now we changed it so people don't need to specify 'w' before hand
They can simualte a bunch of algorithm/paramters, and then see their 
impact of objective score under different 'w'
"""

#important columns in df:
#algo_name, algo_param, n_step, regret_per_step (it is actually reward)
#df = df1[df1["arm_mean_reward_dist_loc"] == 0.3].copy()

#the simple setting is people specify one reward distribution setting, and we use the above
#four element to calcualte the perforamnce of any 'w', using formula:
# objective score = n_step  - w*reward*n_step/log(n_step)
#
#since the simulation is looped over a range of parameter for each algortihm (0,0.025,...,0.975,1),
#we let user to plot a subset of those curves.
#they have two ways to specify:
#explict. e.g. Epsilon-TS with epsilon = 0.1
#optimized for an 'w'. e.g. plot the corresponding epsilon-TS that has the best score when 'w' = 8


#we also have a more complex setting, where people can specify multiple reward distribution
#(which means they are not sure about it, and want to do sensitivity check)
#but the way they choose curves are similar. So I think we can focus on the simple case.


selectors = [
    ("EpsTS", "param", 0.2),  # fixed epsilon
    ("EpsTS", "param", 0),  # fixed epsilon
    ("EpsTS", "param", 1),  # fixed epsilon
    #("EpsTS", "w", 5),        # epsilon tuned for w=5
    ("EpsTS", "w", 10),        # epsilon tuned for w=10
]

curves = select_curves_relative(df, selectors, w_values=range(1, 16))
plot_curves(curves)









"""

Old pieces
"""
#
# import numpy as np
# import pandas as pd
# import pickle
# import bandit_algorithm as algorithm
# import sim_wrapper as sw
# import copy
# import logging
# from joblib import Parallel, delayed
#
# from simulation_configurator import SimulationConfig
# from test_procedure_configurator import TestProcedure, ANOVA, TControl, TConstant, Tukey
# import bayes_vector_ops as bayes
#
# logging.getLogger("ax.service.managed_loop").setLevel(logging.CRITICAL)
# logging.getLogger('ax.generation_strategy.dispatch_utils').setLevel(logging.CRITICAL)
#
#
# # for vm?
# import sys
# import os
#
# # Add project root (where bayes_vector_ops.py is) to sys.path
# # sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
#
# # TODO: in paper, can define broadly how user can plug in their own test
# # TODO: add a custom function (user define a function, input is reward and action hist, output is test. the code help them run it in h0 to get crit, and then evalaute it in objective function
#
#
# # np.random.seed(0)
# sim_config_base = SimulationConfig(
#     n_rep=1000,
#     n_arm=3,
#     horizon=500,  # max horizon to try in simulation
#     burn_in_per_arm=5,
#     # horizon_check_points=sw.generate_quadratic_schedule(2000), #can ignore for now... TODO: see where I used it (and delete if not)
#     # can set tuning_density to make the schedule denser / looser
#     n_opt_trials=None,  # TODO: optimize for this in our code
#     # arm_mean_reward_dist_loc = [0.7,0.3,0.3],
#     # arm_mean_reward_dist_scale = 0.01,
#     test_procedure=ANOVA(),
#     step_cost=0.05,
#     reward_evaluation_method='regret',
#     vector_ops=bayes.BackendOpsNP()
# )
# # algo_list = ['ts_adapt_explor','ts_postdiff_top','eps_ts', 'ts_postdiff_ur','ts_probclip']
# algo_list = [ algorithm.EpsTS,algorithm.TSProbClip,algorithm.TSTopUR] #TODO: import directly
# #algo_list =[algo.TSTopUR]
# #algo_list = [algo.EpsTS]
# algo_param_list = list(np.arange(0.1, 1.001, 0.2))
# #algo_param_list = [0]
# test_list = [
#     [ANOVA(),0.8],
#     # [TControl(),0.8],
#     # #TControl(permutation_test=True),
#     # # [TControl(test_type='two-sided'),0.8],
#     # # [TConstant(),0.8], #power you get for UR 300 steps
#     # [Tukey(test_type='all-pair-wise'),0.8],
#     # [Tukey(test_type='distinct-best-arm'),0.8],
# ]
#
# #step_cost_list = [0.001,0.003,0.01,0.02,0.05,0.1,0.3,1]
# loc_list = [0.2,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8]
# scale_list = [0.08,0.10,0.12,0.135, 0.15,0.165, 0.18,0.20,0.22]
#
#
#
# """
# ###  Simulation 1: Optimization Loop
# """
# main_task_list = [(i, j, k) for i in range(len(algo_param_list)) for j in range(len(algo_list)) for k in range(len(test_list))]
# loc_task_list = [(i, j) for i in range(len(loc_list)) for j in range(len(algo_list))]
# scale_task_list = [(i, j) for i in range(len(scale_list)) for j in range(len(algo_list))]
#
#
#
# def run_main_task(i, j, k):
#     sim_config = copy.deepcopy(sim_config_base)
#     sim_config.step_cost = 0.1
#     algo = algo_list[j]
#     sim_config.test_procedure = test_list[k][0]
#     sim_config.test_procedure.power_constraint = test_list[k][1]
#     sim_config.manual_init()
#     params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo,algo_param_list=[float(algo_param_list[i])])
#     # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
#     return {'test':sim_config.test_procedure.test_signature,
#             'step_cost': sim_config.step_cost,
#             'mean_reward_dist_loc':sim_config.arm_mean_reward_dist_loc[0],
#             'mean_reward_dist_scale':sim_config.arm_mean_reward_dist_scale[0],
#             'algo_name': algo.__name__,
#             'algo_param': params['algo_para'],
#             **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
#             'all_results': sim_result_keeper,
#             }
#
# def run_loc_mismatch_task(i, j):
#     sim_config = copy.deepcopy(sim_config_base)
#     sim_config.arm_mean_reward_dist_loc = loc_list[i]
#     algo = algo_list[j]
#     sim_config.manual_init()
#     params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo,algo_param_list=None)
#     # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
#     return {'test': sim_config.test_procedure.test_signature,
#             'step_cost': sim_config.step_cost,
#             'mean_reward_dist_loc': sim_config.arm_mean_reward_dist_loc[0],
#             'mean_reward_dist_scale': sim_config.arm_mean_reward_dist_scale[0],
#             'algo_name': algo.__name__,
#             'algo_param': params['algo_para'],
#             **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
#             'all_results': sim_result_keeper,
#             }
#
# def run_scale_mismatch_task(i, j):
#     sim_config = copy.deepcopy(sim_config_base)
#     sim_config.horizon = 4000
#     sim_config.arm_mean_reward_dist_scale = scale_list[i]
#     algo = algo_list[j]
#     sim_config.manual_init()
#     params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo,algo_param_list=None)
#     # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
#     return {'test': sim_config.test_procedure.test_signature,
#             'step_cost': sim_config.step_cost,
#             'mean_reward_dist_loc': sim_config.arm_mean_reward_dist_loc[0],
#             'mean_reward_dist_scale': sim_config.arm_mean_reward_dist_scale[0],
#             'algo_name': algo.__name__,
#             'algo_param': params['algo_para'],
#             **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
#             'all_results': sim_result_keeper,
#             }
#
# def rerun_loc_mismatch_task(j):
#     sim_config = copy.deepcopy(sim_config_base)
#     algo = algo_list[j]
#     sim_config.manual_init()
#     algo_param_list = list(loc_best_df.algo_param[loc_best_df.algo_name == algo.__name__])
#     params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo,algo_param_list=algo_param_list)
#
#     # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
#     return {'test': sim_config.test_procedure.test_signature,
#             'step_cost': sim_config.step_cost,
#             'mean_reward_dist_loc': sim_config.arm_mean_reward_dist_loc[0],
#             'mean_reward_dist_scale': sim_config.arm_mean_reward_dist_scale[0],
#             'algo_name': algo.__name__,
#             'algo_param': params['algo_para'],
#             **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
#             'all_results': sim_result_keeper,
#             }
#
# def rerun_scale_mismatch_task(j):
#     sim_config = copy.deepcopy(sim_config_base)
#     algo = algo_list[j]
#     sim_config.manual_init()
#
#     algo_param_list = list(scale_best_df.algo_param[scale_best_df.algo_name == algo.__name__])
#     params, val, _, _, sim_result_keeper = sw.optimize_algorithm(sim_config, algo=algo, algo_param_list=algo_param_list)
#     # return (i, j, algo, params, val, sim_result_keeper.get((algo, params['algo_para']), None))
#     return {'test': sim_config.test_procedure.test_signature,
#             'step_cost': sim_config.step_cost,
#             'mean_reward_dist_loc': sim_config.arm_mean_reward_dist_loc[0],
#             'mean_reward_dist_scale': sim_config.arm_mean_reward_dist_scale[0],
#             'algo_name': algo.__name__,
#             'algo_param': params['algo_para'],
#             **sim_result_keeper.get((algo.__name__, params['algo_para'], sim_config.setting_signature), None),
#             'all_results': sim_result_keeper,
#             }
#
# def extract_results(results):
#     """
#     Given a list of results (each row containing 'all_results' as a nested dict),
#     returns:
#     - best_df: original DataFrame without the 'all_results' column
#     - full_df: flattened DataFrame with one row per (algo, param) combination
#
#     Parameters:
#         results (list[dict]): List of result dictionaries, each with an 'all_results' key.
#
#     Returns:
#         best_df (pd.DataFrame)
#         full_df (pd.DataFrame)
#     """
#     results_df = pd.DataFrame(results)
#     best_df = results_df.drop(columns=['all_results'])
#
#
#     all_rows = []
#     for _, row in results_df.iterrows():
#         row_info = row.drop('all_results').to_dict()
#         all_result_dict = row['all_results']
#
#         for (algo, param, _), result in all_result_dict.items():
#             entry = row_info.copy()
#             entry['best_algo_param'] = entry['algo_param']
#             entry['algo_name'] = algo
#             entry['algo_param'] = param
#             entry.update({k: float(v) for k, v in result.items()})
#             all_rows.append(entry)
#
#     full_df = pd.DataFrame(all_rows)
#     return best_df, full_df
#
#
# main_results = Parallel(n_jobs=-1)(delayed(run_main_task)(i, j, k) for i, j, k in main_task_list)
# loc_results = Parallel(n_jobs=-1)(delayed(run_loc_mismatch_task)(i, j) for i, j in loc_task_list)
# scale_results = Parallel(n_jobs=-1)(delayed(run_scale_mismatch_task)(i, j) for i, j in scale_task_list)
#
# main_best_df, main_full_df = extract_results(main_results)
# loc_best_df, loc_full_df = extract_results(loc_results)
# scale_best_df, scale_full_df = extract_results(scale_results)
#
# rerun_loc_mismatch_results = Parallel(n_jobs=-1)(delayed(rerun_loc_mismatch_task)( j) for  j in range(len(algo_list)))
# rerun_scale_mismatch_results = Parallel(n_jobs=-1)(delayed(rerun_scale_mismatch_task)( j) for  j in range(len(algo_list)))
# rerun_loc_mismatch_best_df, rerun_loc_mismatch_full_df = extract_results(rerun_loc_mismatch_results)
# rerun_scale_mismatch_best_df, rerun_scale_mismatch_full_df = extract_results(rerun_scale_mismatch_results)
#
#
# # === Save all DataFrames to CSV ===
# output_dir = "final_results08061"
# os.makedirs(output_dir, exist_ok=True)
#
# dfs = {
#     "main_best_df": main_best_df,
#     "main_full_df": main_full_df,
# }
#
# for name, df in dfs.items():
#     df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
#
#
# df = main_full_df
# #regret_per_step   cum_regret_full n_step
# df['cum_regret_full'] = df['regret_at_end'] * 2500
# df['mean_regret']
#
#
#
#
#
# """
# xxx
# """
# weights = [0.00001,0.0001,0.0003,0.001,0.003,0.01,0.03, 0.1, 1,10,100]
# results = {}
#
# for w in [0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03, 0.1,0.3, 1]:
#     df[f'obj_long_{w}'] = w * df['n_step'] + df['cum_regret_full']
#     df[f'obj_mean_{w}'] = w * df['n_step'] + df['regret_per_step']
#
#     grouped = df.groupby(['test', 'algo_name'])
#
#     summary_rows = []
#
#     for group_name, group_df in grouped:
#         idx_long = group_df[f'obj_long_{w}'].idxmin()
#         idx_mean = group_df[f'obj_mean_{w}'].idxmin()
#
#         summary_rows.append({
#             'test': group_name[0],
#             'algo_name': group_name[1],
#             'weight': w,
#             'best_obj_long': df.loc[idx_long, f'obj_long_{w}'],
#             'best_param_long': df.loc[idx_long, 'algo_param'],
#             'best_obj_mean': df.loc[idx_mean, f'obj_mean_{w}'],
#             'best_param_mean': df.loc[idx_mean, 'algo_param'],
#         })
#
#     results[w] = pd.DataFrame(summary_rows)
#
# # Combine all weights
# final_summary = pd.concat(results.values(), ignore_index=True)
#
#
#
# # === Save all variables to a pickle file ===
# with open(os.path.join(output_dir, "all_results.pkl"), "wb") as f:
#     pickle.dump(dfs, f)