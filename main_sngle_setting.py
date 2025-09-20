import numpy as np
import pandas as pd
import warnings
import pickle
import bayes_vector_ops as bm
import bandit_algorithm as algo
import sim_wrapper as sw
import matplotlib.pyplot as plt
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


arm_diff_list = [0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4]  # extend as needed
#arm_diff_list = [0, 0.05, 0.1, 0.2, 0.3]  # extend as needed
algo_list = [
    algo.TSPostDiffUR(0.11),
    algo.TSProbClip( (1-0.79)*2),
    algo.EpsTS(0.345),
    algo.EpsTS(0),
    algo.EpsTS(1),
]
# --- Step 3. Style mapping (your renamed labels) ---
styles = {
    "TS": {"color": "black", "marker": "*", "linestyle": "--"},
    "UR": {"color": "black", "marker": "D", "linestyle": ":"},
    "TSPostDiffUR (0.110)": {"color": "gold", "marker": "o", "linestyle": "-"},
    "TSProbClip (0.420)": {"color": "purple", "marker": "^", "linestyle": "-"},
    "EpsTS (0.345)": {"color": "green", "marker": "o", "linestyle": "-"},
}
records = []

for algorithm in algo_list:
    # ---- Step 1: Calibrate threshold at diff=0 ----
    arm_means = [0.5, 0.5]   # diff=0

    sim_config = SimulationConfig(
        n_rep=20000,
        n_arm=2,
        horizon=785,
        burn_in_per_arm=1,
        n_opt_trials=5,
        arm_mean_reward_dist_loc=arm_means,
        arm_mean_reward_dist_scale=0.0,
        test_procedure=ANOVA(),
        step_cost=0.1,
        reward_evaluation_method='regret',
        vector_ops=bayes.BackendOpsNP()
    )
    sim_config.manual_init()

    res0 = sw.run_simulation(policy=algorithm, sim_config=sim_config)

    # compute empirical distribution of wald test
    wald0 = np.abs(res0.wald_test())

    # find quantile cutoff so FPR = 0.05
    # (replace 0.05 if you want a different target level)
    threshold = np.quantile(wald0, 0.95)
    print(f"Calibrated threshold for {algorithm.__name__}({algorithm.algo_para}): {threshold:.3f}")

    # ---- Step 2: Run over all diffs with calibrated threshold ----
    for diff in arm_diff_list:
        arm_means = [0.5 + diff/2, 0.5 - diff/2]

        print(f"Simulating {algorithm.__name__}({algorithm.algo_para}) on diff={diff}")

        sim_config = SimulationConfig(
            n_rep=20000,
            n_arm=2,
            horizon=785,
            burn_in_per_arm=1,
            n_opt_trials=5,
            arm_mean_reward_dist_loc=arm_means,
            arm_mean_reward_dist_scale=0.0,
            test_procedure=ANOVA(),
            step_cost=0.1,
            reward_evaluation_method='regret',
            vector_ops=bayes.BackendOpsNP()
        )
        sim_config.manual_init()

        res = sw.run_simulation(policy=algorithm, sim_config=sim_config)

        # power using calibrated threshold
        power = np.mean(np.abs(res.wald_test()) > threshold)
        reward = np.mean(res.combined_means[:, -1, :])

        records.append({
            "arm_diff": diff,
            "algo_name": algorithm.__name__,
            "algo_param": algorithm.algo_para,
            "threshold": threshold,
            "reward": reward,
            "power": power,
        })
# ---- Collect into DataFrame ----
df_results = pd.DataFrame(records)



"""
Plot on power
"""
df = df_results.copy()

# --- Step 1. Format algorithm labels (rounded params) ---
def format_algo(row):
    name = row["algo_name"]
    param = row["algo_param"]

    try:
        param_val = float(param)
        param_str = f"{param_val:.3f}"
    except Exception:
        param_str = str(param)

    if name == "EpsTS" and float(param) == 0:
        return "TS"
    elif name == "EpsTS" and float(param) == 1:
        return "UR"
    else:
        return f"{name} ({param_str})"

df["algo_label"] = df.apply(format_algo, axis=1)

# --- Step 2. Compute relative power loss (vs. UR) ---
ur_power = df[df["algo_label"] == "UR"][["arm_diff", "power"]].rename(columns={"power": "ur_power"})
df = df.merge(ur_power, on="arm_diff", how="left")
df["power_loss"] = df["ur_power"] - df["power"]



# --- Step 4. Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

for label, grp in df.groupby("algo_label"):
    style = styles.get(label, {"color": "gray", "marker": "x", "linestyle": "-"})
    ax.plot(
        grp["arm_diff"],
        grp["power_loss"],
        label=label,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
    )

ax.axhline(0, color="black", linestyle="--")
#ax.axvline(0.2, color="red", linestyle="--")

ax.set_xlabel("Arm Difference")
ax.set_ylabel("Power Loss from UR")
ax.set_title("Relative Power Loss (UR baseline = 0)")
ax.legend(title="Algorithm")

plt.tight_layout()
plt.show()



"""
Plot on reward
"""
df = df_results.copy()

# --- Step 1. Format algorithm labels (rounded params) ---
def format_algo(row):
    name = row["algo_name"]
    param = row["algo_param"]

    try:
        param_val = float(param)
        param_str = f"{param_val:.3f}"
    except Exception:
        param_str = str(param)

    if name == "EpsTS" and float(param) == 0:
        return "TS"
    elif name == "EpsTS" and float(param) == 1:
        return "UR"
    else:
        return f"{name} ({param_str})"

df["algo_label"] = df.apply(format_algo, axis=1)

# --- Step 2. Compute reward loss (vs. TS) ---
ts_reward = df[df["algo_label"] == "TS"][["arm_diff", "reward"]].rename(columns={"reward": "ts_reward"})
df = df.merge(ts_reward, on="arm_diff", how="left")
df["reward_loss"] = df["ts_reward"] - df["reward"]



# --- Step 4. Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

for label, grp in df.groupby("algo_label"):
    style = styles.get(label, {"color": "gray", "marker": "x", "linestyle": "-"})
    ax.plot(
        grp["arm_diff"],
        grp["reward_loss"],
        label=label,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
    )

ax.axhline(0, color="black", linestyle="--")   # TS baseline
#ax.axvline(0.2, color="red", linestyle="--")

ax.set_xlabel("Arm Difference")
ax.set_ylabel("Reward Loss from TS")
ax.set_title("Relative Reward Loss (TS baseline = 0)")
ax.legend(title="Algorithm")

plt.tight_layout()
plt.show()