"""
Table 2: Lack-of-fit detection rate under correct vs misspecified linear model.
F-test: linear model (d=3) vs saturated model (K=6).

This is a simulation script that must be run from mab-simulation:
    cd ~/repos/mab-simulation
    python ~/repos/paper-factorial-bandit/code/tab2_lack_of_fit.py

Output: data/misspec_lof_results.csv (then manually update latex/sections/setup_results.tex)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT.parents[0] / "mab-simulation"))

import copy
import time

import numpy as np
from scipy import stats as sp_stats

from bandit_simulation.bandit_algorithm import (
    AgrawalGoyalLinearTS,
    EpsTS,
    TSPostDiff,
)
from bandit_simulation.sim_wrapper import run_simulation
from bandit_simulation.simulation_configurator import SimulationConfig

F = np.array([
    [1, 0, 0], [1, 1, 0], [1, 0, 1],
    [1, 1, 1], [1, 0, 2], [1, 1, 2],
], dtype=float)
K = 6
d = 3
HORIZON = 200
N_REP = 10_000
SIGMA = 0.5
ALPHA = 0.05

MU_CORRECT = np.array([0.3, 0.5, 0.4, 0.6, 0.5, 0.7])
MU_MISSPEC = np.array([0.3, 0.5, 0.55, 0.75, 0.5, 0.7])

SCENARIOS = [("correct", MU_CORRECT), ("misspec", MU_MISSPEC)]

ALGORITHMS = [
    ("UR",                  EpsTS(1.0),                         False),
    ("TS (flat)",           EpsTS(0.0),                         False),
    ("PostDiff-flat(0.10)", TSPostDiff(0.10),                      False),
    ("AG-TS(R=0.07)",       AgrawalGoyalLinearTS({"R": 0.07}), True),
]


def make_config(mu_true, use_linear):
    cfg = SimulationConfig(
        n_rep=N_REP, n_arm=K, horizon=HORIZON,
        burn_in_per_arm=1, batch_scaling_rate=0, base_batch_size=1,
        reward_model=np.random.normal, reward_std=SIGMA,
        arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": mu_true.tolist(), "scale": 0.0},
        },
        arm_feature_matrix=F if use_linear else None,
    )
    cfg.manual_init()
    return cfg


def lack_of_fit_test(action_hist, reward_hist, F_mat, alpha=0.05):
    n_rep = action_hist.shape[0]
    K_arms = F_mat.shape[0]
    d_params = F_mat.shape[1]
    df_lack = K_arms - d_params

    rejections = 0
    valid = 0

    for rep in range(n_rep):
        n_k = action_hist[rep].sum(axis=0)
        r_k = reward_hist[rep].sum(axis=0)
        r2_k = (reward_hist[rep] ** 2).sum(axis=0)

        if np.any(n_k < 1):
            continue

        valid += 1
        y_bar = r_k / n_k

        W = np.diag(n_k)
        FtWF = F_mat.T @ W @ F_mat
        try:
            theta_hat = np.linalg.solve(FtWF, F_mat.T @ W @ y_bar)
        except np.linalg.LinAlgError:
            continue

        mu_hat = F_mat @ theta_hat
        ss_lack = np.sum(n_k * (y_bar - mu_hat) ** 2)
        ss_pure = np.sum(r2_k - n_k * y_bar ** 2)

        N_total = int(n_k.sum())
        df_pure = N_total - K_arms

        if df_pure <= 0 or ss_pure <= 0:
            continue

        f_stat = (ss_lack / df_lack) / (ss_pure / df_pure)
        p_val = 1 - sp_stats.f.cdf(f_stat, df_lack, df_pure)
        if p_val < alpha:
            rejections += 1

    return rejections / valid if valid > 0 else float("nan"), valid


def main():
    np.random.seed(42)
    import pandas as pd

    rows = []
    for sc_name, mu_true in SCENARIOS:
        for name, policy, use_linear in ALGORITHMS:
            t0 = time.time()
            cfg = make_config(mu_true, use_linear)
            res = run_simulation(policy=copy.deepcopy(policy), sim_config=cfg)
            reward = res.combined_means[:, -1, 0].mean()
            lof_rej, lof_valid = lack_of_fit_test(
                res.action_hist, res.reward_hist, F, alpha=ALPHA
            )
            elapsed = time.time() - t0
            print(f"{sc_name:<10} {name:<22} LoF={lof_rej:.1%} ({elapsed:.0f}s)")
            rows.append({
                "scenario": sc_name,
                "algorithm": name,
                "reward": round(reward, 4),
                "lof_rejection_rate": round(lof_rej * 100, 1),
                "valid_reps": lof_valid,
            })

    df = pd.DataFrame(rows)
    out = ROOT / "data" / "misspec_lof_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
