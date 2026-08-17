"""
B1 Part 4: Use n_arm=2 so power is achievable and n_step actually varies.
This lets us see if FWER and burn_in affect the OPTIMAL algorithm choice.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bandit_simulation import EpsTS, TControl, ANOVA, sweep_and_run, compute_objective
from bandit_simulation.simulation_configurator import SimulationConfig


N_REP = 5000
PARAM_LIST = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEED = 42


def run_test(n_arm, fwer, burn_in, test_proc, label):
    np.random.seed(SEED)
    sim_config = SimulationConfig(
        n_rep=N_REP,
        n_arm=n_arm,
        horizon=2000,
        burn_in_per_arm=burn_in,
        reward_model=np.random.normal,
        reward_std=0.1,
        arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": 0.5, "scale": 0.05},
        },
        test_procedure=test_proc,
        reward_evaluation_method='reward',
    )
    sweeps = [
        {"algo": [EpsTS]},
        {"algo_param_list": PARAM_LIST},
    ]
    t0 = time.time()
    df = sweep_and_run(sweeps, sim_config)
    elapsed = time.time() - t0

    print(f"\n{'='*75}")
    print(f"  {label}  [{elapsed:.1f}s]")
    print(f"{'='*75}")
    print(f"  {'param':>6}  {'n_step':>8}  {'reward':>10}  {'power':>8}  {'obj(w=1)':>10}  {'obj(w=5)':>10}  {'obj(w=10)':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for _, row in df.iterrows():
        n = row['n_step']
        r = row['regret_per_step']
        obj1 = r - 1 * np.log(n) if n > 1 else n
        obj5 = r - 5 * np.log(n) if n > 1 else n
        obj10 = r - 10 * np.log(n) if n > 1 else n
        print(f"  {row['algo_param']:>6.2f}  {n:>8.0f}  {r:>10.4f}  {row['power_max']:>8.4f}  {obj1:>10.4f}  {obj5:>10.4f}  {obj10:>10.4f}")

    # Find best for each w
    for w in [1, 5, 10]:
        best_idx = None
        best_obj = -np.inf
        for idx, row in df.iterrows():
            n = row['n_step']
            r = row['regret_per_step']
            obj = r - w * np.log(n) if n > 1 else n
            if obj > best_obj:
                best_obj = obj
                best_idx = idx
        best = df.loc[best_idx]
        print(f"  Best at w={w}: eps={best['algo_param']:.2f}, n_step={best['n_step']:.0f}, reward={best['regret_per_step']:.4f}")

    return df


if __name__ == '__main__':
    print("B1 Part 4: n_arm=2 and n_arm=3 (power achievable)")
    print(f"n_rep={N_REP}, horizon=2000")

    # ── 2-arm ANOVA (simplest case) ──
    print("\n" + "#"*75)
    print("# 2-ARM ANOVA")
    print("#"*75)

    run_test(2, False, 1,
             ANOVA(min_effect=0.1),
             "PAPER: n_arm=2, ANOVA, FWER=F, burn=1")

    run_test(2, True, 5,
             ANOVA(min_effect=0.1, family_wise_error_control=True),
             "WEBAPP: n_arm=2, ANOVA, FWER=T, burn=5")

    # ── 3-arm TControl (paper Table 4 scenario) ──
    print("\n" + "#"*75)
    print("# 3-ARM T-CONTROL")
    print("#"*75)

    run_test(3, False, 1,
             TControl(min_effect=0.1, test_type='two-sided'),
             "PAPER: n_arm=3, TControl, FWER=F, burn=1")

    run_test(3, True, 5,
             TControl(min_effect=0.1, test_type='two-sided', family_wise_error_control=True),
             "WEBAPP: n_arm=3, TControl, FWER=T, burn=5")

    run_test(3, False, 5,
             TControl(min_effect=0.1, test_type='two-sided'),
             "ISOLATE burn_in: n_arm=3, FWER=F, burn=5")

    run_test(3, True, 1,
             TControl(min_effect=0.1, test_type='two-sided', family_wise_error_control=True),
             "ISOLATE FWER: n_arm=3, FWER=T, burn=1")

    print("\nDone.")
