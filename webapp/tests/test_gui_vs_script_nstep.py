"""
B1 Investigation Part 2: Test with shorter horizon so n_step actually varies.

Uses parameters similar to the user's saved scenario "111":
  n_arm=6, horizon=200, gaussian, reward_std=0.1,
  h1_loc=0.81, h1_scale=0.015, t_control two-sided, min_effect=0.025
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bandit_simulation import EpsTS, TControl, sweep_and_run, compute_objective
from bandit_simulation.simulation_configurator import SimulationConfig


COMMON_PARAMS = dict(
    n_arm=6,
    horizon=200,
    reward_model=np.random.normal,
    reward_std=0.1,
    arm_mean_reward_dist_spec={
        "dist": "normal",
        "params": {"loc": 0.81, "scale": 0.015},
    },
    test_procedure=TControl(
        min_effect=0.025,
        test_type='two-sided',
        type1_error_constraint=0.05,
        power_constraint=0.8,
        family_wise_error_control=True,
    ),
    reward_evaluation_method='reward',
)

N_REP = 1000
PARAM_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 42


def run_config(burn_in, label):
    np.random.seed(SEED)
    sim_config = SimulationConfig(
        n_rep=N_REP,
        burn_in_per_arm=burn_in,
        **COMMON_PARAMS,
    )
    sweeps = [
        {"algo": [EpsTS]},
        {"algo_param_list": PARAM_LIST},
    ]
    t0 = time.time()
    df = sweep_and_run(sweeps, sim_config)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  {label} (burn_in={burn_in})  [{elapsed:.1f}s]")
    print(f"{'='*70}")
    print(f"  {'param':>6}  {'n_step':>10}  {'reward/step':>12}  {'obj(w=0.01)':>12}  {'obj(w=0.03)':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
    for _, row in df.iterrows():
        obj_lo = row['regret_per_step'] - 0.01 * np.log(row['n_step']) if row['n_step'] > 1 else row['n_step']
        obj_hi = row['regret_per_step'] - 0.03 * np.log(row['n_step']) if row['n_step'] > 1 else row['n_step']
        print(f"  {row['algo_param']:>6.2f}  {row['n_step']:>10.1f}  {row['regret_per_step']:>12.6f}  {obj_lo:>12.6f}  {obj_hi:>12.6f}")
    return df


if __name__ == '__main__':
    print("B1 Part 2: n_step variation test (shorter horizon)")
    print(f"n_rep={N_REP}, params={PARAM_LIST}")

    df1 = run_config(1, "Script approach (burn_in=1)")
    df5 = run_config(5, "Webapp approach (burn_in=5)")

    # Compare
    merged = df1.merge(df5, on=['algo_name', 'algo_param'], suffixes=('_b1', '_b5'))
    print(f"\n{'='*70}")
    print(f"  DIFF: burn_in=1 vs burn_in=5")
    print(f"{'='*70}")
    print(f"  {'param':>6}  {'Δ n_step':>10}  {'Δ n_step%':>10}  {'Δ reward':>12}  {'Δ reward%':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    for _, row in merged.iterrows():
        dn = row['n_step_b1'] - row['n_step_b5']
        dn_pct = (dn / row['n_step_b5'] * 100) if row['n_step_b5'] > 0 else 0
        dr = row['regret_per_step_b1'] - row['regret_per_step_b5']
        dr_pct = (dr / row['regret_per_step_b5'] * 100) if row['regret_per_step_b5'] > 0 else 0
        print(f"  {row['algo_param']:>6.2f}  {dn:>+10.1f}  {dn_pct:>+9.2f}%  {dr:>+12.6f}  {dr_pct:>+9.4f}%")

    print("\nDone.")
