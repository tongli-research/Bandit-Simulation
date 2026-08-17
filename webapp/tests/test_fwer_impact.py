"""
B1 Investigation Part 3: Impact of family_wise_error_control.

The scripts (paper) use family_wise_error_control=False (default).
The webapp checkbox is checked by default → family_wise_error_control=True.

This test isolates the FWER impact on n_step and reward.
Uses a longer horizon (5000) and higher n_rep to see meaningful n_step variation.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bandit_simulation import EpsTS, TControl, sweep_and_run
from bandit_simulation.simulation_configurator import SimulationConfig


N_REP = 5000
PARAM_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 42


def run_test(fwer, burn_in, label):
    np.random.seed(SEED)
    sim_config = SimulationConfig(
        n_rep=N_REP,
        n_arm=6,
        horizon=5000,
        burn_in_per_arm=burn_in,
        arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": 0.81, "scale": 0.015},
        },
        test_procedure=TControl(
            min_effect=0.025,
            test_type='two-sided',
            type1_error_constraint=0.05,
            power_constraint=0.8,
            family_wise_error_control=fwer,
        ),
        reward_evaluation_method='reward',
    )
    sweeps = [
        {"algo": [EpsTS]},
        {"algo_param_list": PARAM_LIST},
    ]
    t0 = time.time()
    df = sweep_and_run(sweeps, sim_config)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  {label}  [{elapsed:.1f}s]")
    print(f"{'='*70}")
    print(f"  {'param':>6}  {'n_step':>10}  {'reward/step':>12}  {'power_max':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*10}")
    for _, row in df.iterrows():
        print(f"  {row['algo_param']:>6.2f}  {row['n_step']:>10.1f}  {row['regret_per_step']:>12.6f}  {row['power_max']:>10.4f}")
    return df


if __name__ == '__main__':
    print("B1 Part 3: Family-Wise Error Control Impact")
    print(f"n_rep={N_REP}, n_arm=6, horizon=5000, params={PARAM_LIST}")
    print(f"Paper scripts: FWER=False, burn_in=1")
    print(f"Webapp default: FWER=True, burn_in=5")

    # Paper's approach
    df_paper = run_test(fwer=False, burn_in=1, label="PAPER: FWER=False, burn_in=1")

    # Webapp's approach
    df_webapp = run_test(fwer=True, burn_in=5, label="WEBAPP: FWER=True, burn_in=5")

    # Isolate: FWER only (keep burn_in=1)
    df_fwer_only = run_test(fwer=True, burn_in=1, label="FWER=True only (burn_in=1)")

    # Compare
    merged = df_paper.merge(df_webapp, on=['algo_name', 'algo_param'], suffixes=('_paper', '_webapp'))
    print(f"\n{'='*70}")
    print(f"  DIFF: Paper vs Webapp")
    print(f"{'='*70}")
    print(f"  {'param':>6}  {'n_step_paper':>12}  {'n_step_webapp':>13}  {'Δ n_step':>10}  {'Δ n_step%':>10}  {'Δ reward':>12}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*13}  {'-'*10}  {'-'*10}  {'-'*12}")
    for _, row in merged.iterrows():
        dn = row['n_step_paper'] - row['n_step_webapp']
        dn_pct = (dn / row['n_step_webapp'] * 100) if row['n_step_webapp'] > 0 else 0
        dr = row['regret_per_step_paper'] - row['regret_per_step_webapp']
        print(f"  {row['algo_param']:>6.2f}  {row['n_step_paper']:>12.1f}  {row['n_step_webapp']:>13.1f}  {dn:>+10.1f}  {dn_pct:>+9.2f}%  {dr:>+12.6f}")

    # Isolate FWER effect
    merged2 = df_paper.merge(df_fwer_only, on=['algo_name', 'algo_param'], suffixes=('_nofwer', '_fwer'))
    print(f"\n{'='*70}")
    print(f"  ISOLATED: FWER=False vs FWER=True (both burn_in=1)")
    print(f"{'='*70}")
    print(f"  {'param':>6}  {'n_step_noFW':>12}  {'n_step_FW':>12}  {'Δ n_step':>10}  {'Δ n_step%':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")
    for _, row in merged2.iterrows():
        dn = row['n_step_nofwer'] - row['n_step_fwer']
        dn_pct = (dn / row['n_step_fwer'] * 100) if row['n_step_fwer'] > 0 else 0
        print(f"  {row['algo_param']:>6.2f}  {row['n_step_nofwer']:>12.1f}  {row['n_step_fwer']:>12.1f}  {dn:>+10.1f}  {dn_pct:>+9.2f}%")

    print("\nDone.")
