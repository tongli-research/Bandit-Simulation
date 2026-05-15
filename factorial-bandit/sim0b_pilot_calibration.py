"""
Sim 0b: Pilot calibration -- distribution of beta-hat under TS-linear.
Determines the Approach B threshold (calibrated quantile of beta-hat).

Run from mab-simulation:
    cd ~/repos/mab-simulation
    python ~/repos/paper-factorial-bandit/code/sim0b_pilot_calibration.py

Output: data/sim0b_pilot_results.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT.parents[0] / "mab-simulation"))

import time

import numpy as np
import pandas as pd

from bandit_simulation.analysis import linear_regression_test
from bandit_simulation.bandit_algorithm import EpsTS
from bandit_simulation.sim_wrapper import run_simulation
from bandit_simulation.simulation_configurator import SimulationConfig

F = np.array([
    [1, 0, 0], [1, 1, 0], [1, 0, 1],
    [1, 1, 1], [1, 0, 2], [1, 1, 2],
], dtype=float)
K = 6
HORIZON = 785
SIGMA = 0.5
N_REP = 10_000

SCENARIOS = [
    ("H1_MDE", np.array([0.30, 0.10, 0.10])),
    ("H0",     np.array([0.30, 0.00, 0.00])),
]

QUANTILES = [0.50, 0.75, 0.80, 0.90, 0.95, 0.99]


def make_config(theta, use_linear):
    mu = (F @ theta).tolist()
    cfg = SimulationConfig(
        n_rep=N_REP, n_arm=K, horizon=HORIZON, burn_in_per_arm=1,
        reward_model=np.random.normal, reward_std=SIGMA,
        arm_mean_reward_dist_spec={
            "dist": "normal", "params": {"loc": mu, "scale": 0.0},
        },
        arm_feature_matrix=F if use_linear else None,
    )
    cfg.manual_init()
    return cfg


def compute_design_quality(action_hist):
    arm_counts = action_hist.sum(axis=1)
    n_rep = arm_counts.shape[0]
    inv_diag = np.empty((n_rep, 3))
    min_eig = np.empty(n_rep)
    for r in range(n_rep):
        XtX = np.zeros((3, 3))
        for k in range(K):
            XtX += arm_counts[r, k] * np.outer(F[k], F[k])
        eigvals = np.linalg.eigvalsh(XtX)
        min_eig[r] = eigvals[0]
        try:
            inv_diag[r] = np.diag(np.linalg.inv(XtX))
        except np.linalg.LinAlgError:
            inv_diag[r] = np.nan
    return inv_diag, min_eig


def main():
    np.random.seed(42)
    all_rows = []

    for sc_name, theta in SCENARIOS:
        t0 = time.time()
        print(f"\n{'='*50}")
        print(f"Scenario: {sc_name}, theta = {theta}")

        cfg = make_config(theta, use_linear=True)
        res = run_simulation(policy=EpsTS(0.0), sim_config=cfg)
        lr = linear_regression_test(res.action_hist, res.reward_hist, F, alpha=0.05)
        beta_hat = lr["beta_hat"]
        inv_diag, min_eig = compute_design_quality(res.action_hist)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s")

        for j, name in [(1, "beta_1"), (2, "beta_2")]:
            vals = np.abs(beta_hat[:, j])
            print(f"\n  |{name}_hat| distribution:")
            print(f"    mean={vals.mean():.4f}, std={vals.std():.4f}")
            for q in QUANTILES:
                v = np.quantile(vals, q)
                print(f"    P{int(q*100):02d} = {v:.4f}")
                all_rows.append({
                    "scenario": sc_name,
                    "coefficient": name,
                    "quantile": f"P{int(q*100):02d}",
                    "value": round(v, 4),
                })

        print(f"\n  Design quality:")
        print(f"    mean (X'X)^-1_11 = {inv_diag[:,1].mean():.6f}")
        print(f"    mean (X'X)^-1_22 = {inv_diag[:,2].mean():.6f}")
        print(f"    mean min_eig     = {min_eig.mean():.2f}")

    df = pd.DataFrame(all_rows)
    out = ROOT / "data" / "sim0b_pilot_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
