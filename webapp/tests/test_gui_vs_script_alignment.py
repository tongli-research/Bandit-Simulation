"""
B1 Investigation: Compare GUI (webapp) vs pure code (scripts) simulation outputs.

This test reproduces the Table 3 "Prior (design-time)" scenario from the paper
using BOTH the webapp's approach and the script's approach, to identify where
results diverge.

Key known differences to test:
  1. burn_in_per_arm: scripts=1, webapp=5
  2. sweep approach: scripts=batch sweep_and_run, webapp=per-(algo,param) loop
  3. Other SimulationConfig defaults that might differ
"""
import sys
import os
import time

import numpy as np
import pandas as pd

# Add parent dirs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bandit_simulation import EpsTS, TControl, sweep_and_run, compute_objective
from bandit_simulation.simulation_configurator import SimulationConfig

# ──────────────────────────────────────────────────────────────
# Scenario: Table 3 "Prior (design-time)" from the paper
# ──────────────────────────────────────────────────────────────
COMMON_PARAMS = dict(
    n_arm=6,
    horizon=5000,
    reward_evaluation_method='reward',
    arm_mean_reward_dist_spec={
        "dist": "normal",
        "params": {"loc": 0.81, "scale": 0.015},
    },
    test_procedure=TControl(min_effect=0.025, test_type='two-sided'),
)

# Use lower n_rep for faster testing (still enough to see systematic differences)
N_REP = 2000
GRANULARITY = 5  # Just 5 param values for speed: 0.0, 0.25, 0.5, 0.75, 1.0
PARAM_LIST = list(map(float, np.linspace(0.0, 1.0, GRANULARITY)))

# Fix random seed for reproducibility
SEED = 42


def run_script_approach(burn_in_per_arm=1):
    """Reproduce the script approach: single batch sweep_and_run, burn_in=1."""
    np.random.seed(SEED)
    sim_config = SimulationConfig(
        n_rep=N_REP,
        burn_in_per_arm=burn_in_per_arm,
        **COMMON_PARAMS,
    )
    sweeps = [
        {"algo": [EpsTS]},
        {"algo_param_list": PARAM_LIST},
    ]
    df = sweep_and_run(sweeps, sim_config)
    return df


def run_webapp_approach(burn_in_per_arm=5):
    """Reproduce the webapp approach: per-(algo, param) loop, burn_in=5."""
    all_results = []
    for param in PARAM_LIST:
        np.random.seed(SEED)  # Reset seed each time (webapp doesn't do this, but let's be consistent)
        sim_config = SimulationConfig(
            n_rep=N_REP,
            burn_in_per_arm=burn_in_per_arm,
            **COMMON_PARAMS,
        )
        sweeps = [
            {"algo": [EpsTS]},
            {"algo_param_list": [param]},
        ]
        partial_df = sweep_and_run(sweeps, sim_config)
        all_results.append(partial_df)
    df = pd.concat(all_results, ignore_index=True)
    return df


def run_webapp_approach_no_seed_reset(burn_in_per_arm=5):
    """Reproduce webapp more faithfully: no seed reset between params."""
    np.random.seed(SEED)
    sim_config = SimulationConfig(
        n_rep=N_REP,
        burn_in_per_arm=burn_in_per_arm,
        **COMMON_PARAMS,
    )
    all_results = []
    for param in PARAM_LIST:
        sweeps = [
            {"algo": [EpsTS]},
            {"algo_param_list": [param]},
        ]
        partial_df = sweep_and_run(sweeps, sim_config)
        all_results.append(partial_df)
    df = pd.concat(all_results, ignore_index=True)
    return df


def summarize(df, label):
    """Print summary table for a DataFrame."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'param':>6}  {'n_step':>10}  {'reward/step':>12}  {'obj(w=0.01)':>12}  {'obj(w=0.03)':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
    for _, row in df.iterrows():
        obj_low = compute_objective(row, 0.01)
        obj_high = compute_objective(row, 0.03)
        print(f"  {row['algo_param']:>6.2f}  {row['n_step']:>10.1f}  {row['regret_per_step']:>12.6f}  {obj_low:>12.6f}  {obj_high:>12.6f}")
    return df


def compare(df1, df2, label1, label2):
    """Compare two DataFrames row by row."""
    print(f"\n{'='*60}")
    print(f"  DIFF: {label1} vs {label2}")
    print(f"{'='*60}")
    print(f"  {'param':>6}  {'Δ n_step':>10}  {'Δ reward':>12}  {'Δ obj(w=0.01)':>14}  {'Δ obj(w=0.03)':>14}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*14}")

    merged = df1.merge(df2, on=['algo_name', 'algo_param'], suffixes=('_1', '_2'))
    for _, row in merged.iterrows():
        dn = row['n_step_1'] - row['n_step_2']
        dr = row['regret_per_step_1'] - row['regret_per_step_2']

        # Compute objectives
        class FakeRow1:
            pass
        class FakeRow2:
            pass
        r1, r2 = FakeRow1(), FakeRow2()
        r1.n_step = row['n_step_1']; r1.regret_per_step = row['regret_per_step_1']; r1.__getitem__ = lambda s, k: getattr(s, k)
        r2.n_step = row['n_step_2']; r2.regret_per_step = row['regret_per_step_2']; r2.__getitem__ = lambda s, k: getattr(s, k)

        obj1_lo = row['regret_per_step_1'] - 0.01 * np.log(row['n_step_1']) if row['n_step_1'] > 1 else row['n_step_1']
        obj2_lo = row['regret_per_step_2'] - 0.01 * np.log(row['n_step_2']) if row['n_step_2'] > 1 else row['n_step_2']
        obj1_hi = row['regret_per_step_1'] - 0.03 * np.log(row['n_step_1']) if row['n_step_1'] > 1 else row['n_step_1']
        obj2_hi = row['regret_per_step_2'] - 0.03 * np.log(row['n_step_2']) if row['n_step_2'] > 1 else row['n_step_2']

        do_lo = obj1_lo - obj2_lo
        do_hi = obj1_hi - obj2_hi

        print(f"  {row['algo_param']:>6.2f}  {dn:>+10.1f}  {dr:>+12.6f}  {do_lo:>+14.6f}  {do_hi:>+14.6f}")


if __name__ == '__main__':
    print("B1 Investigation: GUI vs Script Simulation Alignment")
    print(f"n_rep={N_REP}, granularity={GRANULARITY}, params={PARAM_LIST}")
    print()

    # ── Test 1: Script approach (burn_in=1, batch sweep) ──
    print("Running Test 1: Script approach (burn_in=1, batch sweep)...")
    t0 = time.time()
    df_script = run_script_approach(burn_in_per_arm=1)
    print(f"  Done in {time.time()-t0:.1f}s")
    summarize(df_script, "SCRIPT: burn_in=1, batch sweep")

    # ── Test 2: Webapp approach (burn_in=5, per-param loop) ──
    print("\nRunning Test 2: Webapp approach (burn_in=5, per-param loop)...")
    t0 = time.time()
    df_webapp = run_webapp_approach_no_seed_reset(burn_in_per_arm=5)
    print(f"  Done in {time.time()-t0:.1f}s")
    summarize(df_webapp, "WEBAPP: burn_in=5, per-param loop")

    # ── Test 3: Isolate burn_in effect — script approach with burn_in=5 ──
    print("\nRunning Test 3: Script approach with burn_in=5 (isolate burn_in)...")
    t0 = time.time()
    df_burnin5_batch = run_script_approach(burn_in_per_arm=5)
    print(f"  Done in {time.time()-t0:.1f}s")
    summarize(df_burnin5_batch, "SCRIPT w/ burn_in=5: batch sweep")

    # ── Test 4: Isolate loop effect — webapp approach with burn_in=1 ──
    print("\nRunning Test 4: Webapp approach with burn_in=1 (isolate loop)...")
    t0 = time.time()
    df_burnin1_loop = run_webapp_approach_no_seed_reset(burn_in_per_arm=1)
    print(f"  Done in {time.time()-t0:.1f}s")
    summarize(df_burnin1_loop, "WEBAPP w/ burn_in=1: per-param loop")

    # ── Comparisons ──
    print("\n\n" + "#"*60)
    print("# COMPARISONS")
    print("#"*60)

    # Main comparison: script vs webapp (both differences combined)
    compare(df_script, df_webapp, "Script(burn=1,batch)", "Webapp(burn=5,loop)")

    # Isolate burn_in effect: same sweep approach, different burn_in
    compare(df_script, df_burnin5_batch, "Script(burn=1,batch)", "Script(burn=5,batch)")

    # Isolate loop effect: same burn_in, different sweep approach
    compare(df_script, df_burnin1_loop, "Script(burn=1,batch)", "Webapp(burn=1,loop)")

    # Cross-check: batch burn_in=5 vs loop burn_in=5
    compare(df_burnin5_batch, df_webapp, "Script(burn=5,batch)", "Webapp(burn=5,loop)")

    print("\n\nDone. Review diffs above to identify sources of divergence.")
