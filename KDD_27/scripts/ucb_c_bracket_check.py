"""
UCB exploration-constant (c) range check, via pull-allocation.

Not required to reproduce Table 6; this documents how the c grid in
configs/table4_ucb.yaml was chosen. Rather than comparing rewards through
the full AIT/power-analysis pipeline, this looks at the final pull
allocation across arms: one extreme of c should allocate uniformly
(~1/K per arm, UR-like) regardless of arm quality; the other extreme
should allocate almost nothing to the worst arm (TS/greedy-like).

Runs a quick, reduced-n_rep simulation directly (bypassing the H0/power
pipeline entirely) across a broad log-spaced c grid.

Usage (run from this directory, KDD_27/):
    python scripts/ucb_c_bracket_check.py
"""
import copy
import time

import numpy as np

from bandit_simulation.config_loader import load_config
from bandit_simulation.bandit_algorithm import UCB
from bandit_simulation.sim_wrapper import run_simulation

base_config, _sweep_specs = load_config("configs/table4_ucb.yaml")
base_config.n_rep = 1000  # reduced for a quick qualitative check

c_grid = list(np.geomspace(1e-4, 1e4, num=13))

print(f"{'c':>10} | {'best-arm frac':>14} | {'worst-arm frac':>15} | {'uniform=1/K':>12}")
print("-" * 60)

t0 = time.time()
for c in c_grid:
    cfg = copy.deepcopy(base_config)
    cfg.manual_init()

    res = run_simulation(UCB(c), cfg)

    final_counts = res.arm_counts[:, -1, :]
    final_frac = final_counts / final_counts.sum(axis=1, keepdims=True)

    arm_means = cfg.arm_mean_reward_dist
    best_arm_idx = np.argmax(arm_means, axis=1)
    worst_arm_idx = np.argmin(arm_means, axis=1)

    best_frac = np.mean(final_frac[np.arange(cfg.n_rep), best_arm_idx])
    worst_frac = np.mean(final_frac[np.arange(cfg.n_rep), worst_arm_idx])

    print(f"{c:10.4g} | {best_frac:14.3f} | {worst_frac:15.3f} | {1/cfg.n_arm:12.3f}")

print(f"\nTotal time: {time.time() - t0:.1f}s")
