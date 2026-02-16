"""
Factorial / linear-metrics test: run TS vs UR, compute metrics, and plot.

2x3 factorial design with linear encoding [1, x1, x2], K=6 arms, d=3.
"""
import copy
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bandit_simulation.analysis import format_linear_summary
from bandit_simulation.bandit_algorithm import (
    EpsTS,
    TSPostDiffTop,
    TSPostDiffTopLinear,
    TSTopUR,
    TSTopURLinear,
)
from bandit_simulation.plotting import plot_all_factor_effects, plot_gap
from bandit_simulation.sim_wrapper import run_simulation
from bandit_simulation.simulation_configurator import SimulationConfig

np.random.seed(42)
OUT_DIR = os.path.join(os.path.dirname(__file__), "_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Factorial feature matrix (linear encoding) ───────────────────────────
F = np.array([
    [1, 0, 0],  # A: x1=0, x2=0
    [1, 1, 0],  # B: x1=1, x2=0
    [1, 0, 1],  # C: x1=0, x2=1
    [1, 1, 1],  # D: x1=1, x2=1
    [1, 0, 2],  # E: x1=0, x2=2
    [1, 1, 2],  # F: x1=1, x2=2
], dtype=float)
arm_labels = list("ABCDEF")
K = F.shape[0]

# ── 2. Ground truth ─────────────────────────────────────────────────────────
theta_true = np.array([0.3, 0.2, 0.1])
mu_true = F @ theta_true
sigma = 0.5

print("True arm means:")
for k in range(K):
    print(f"  {arm_labels[k]}: {mu_true[k]:.3f}")
print(f"True effects:  Bx1={theta_true[1]:.3f}  Bx2_per_unit={theta_true[2]:.3f}")

# ── 3. Simulation config ────────────────────────────────────────────────────
sim_config = SimulationConfig(
    n_rep=10000,
    n_arm=K,
    horizon=600,
    burn_in_per_arm=1,
    batch_scaling_rate=0,
    base_batch_size=1,
    reward_model=np.random.normal,
    reward_std=sigma,
    arm_mean_reward_dist_spec={
        "dist": "normal",
        "params": {"loc": mu_true.tolist(), "scale": 0.0},
    },
    arm_feature_matrix=F,
)
sim_config.manual_init()
cum_samples = np.cumsum(sim_config.step_schedule)

# ── 4. Run UR and TS ────────────────────────────────────────────────────────
policies = {
    "UR":  EpsTS(1),    # epsilon=1 → pure uniform random
    "TS":  EpsTS(0),    # epsilon=0 → pure Thompson sampling
    'TSTopUR (0.1)': TSTopUR(0.1),
    'TSTopUR (0.2)': TSTopUR(0.2),
    'TSPostDiffTop (0.1)': TSPostDiffTop(0.1),
    'TSPostDiffTop (0.2)': TSPostDiffTop(0.2),
    'TSTopURLinear [.1,.1]': TSTopURLinear([0.1, 0.1]),
    'TSTopURLinear [.1,.2]': TSTopURLinear([0.1, 0.2]),
    'TSTopURLinear [.2,.1]': TSTopURLinear([0.2, 0.1]),
    'TSTopURLinear [.2,.2]': TSTopURLinear([0.2, 0.2]),
    'TSTopURLinear [1.0,.0.2]': TSTopURLinear([1.0, 0.2]),
    'TSPostDiffTopLinear [.1,.1]': TSPostDiffTopLinear([0.1, 0.1]),
    'TSPostDiffTopLinear [.1,.2]': TSPostDiffTopLinear([0.1, 0.2]),
    'TSPostDiffTopLinear [.2,.1]': TSPostDiffTopLinear([0.2, 0.1]),
    'TSPostDiffTopLinear [.2,.2]': TSPostDiffTopLinear([0.2, 0.2]),
    'TSPostDiffTopLinear [1.0,.0.2]': TSPostDiffTopLinear([1.0, 0.2]),
}

import time

results = {}
n_policies = len(policies)
t_start_all = time.time()
for i, (name, policy) in enumerate(policies.items(), 1):
    t0 = time.time()
    print(f"[{i}/{n_policies}] Running {name} ...", end=" ", flush=True)
    cfg = copy.deepcopy(sim_config)
    res = run_simulation(policy=policy, sim_config=cfg)
    metrics = res.compute_linear_factorial_metrics()
    results[name] = metrics
    elapsed = time.time() - t0
    total_elapsed = time.time() - t_start_all
    avg = total_elapsed / i
    eta = avg * (n_policies - i)
    print(f"done in {elapsed:.1f}s  (ETA remaining: {eta:.0f}s)")
    print(f"\n{name}: action_hist shape = {res.action_hist.shape}")

# ── 5. Plots ───────────────────────────────────────────────────────────────
COLORS = {"UR": "#DC3545", "TS": "#007BFF"}


def save_fig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {path}")
    return path


fig = plot_all_factor_effects(results, cum_samples, colors=COLORS)
save_fig(fig, "factor_effects_over_time.png")

fig = plot_gap(results, cum_samples, colors=COLORS)
save_fig(fig, "gap_over_time.png")

# ── 6. Summary table ───────────────────────────────────────────────────────
summary_df = format_linear_summary(results, mu_true)
print("\n" + summary_df.to_string())
summary_df.to_csv(os.path.join(OUT_DIR, "linear_summary.csv"))
print(f"  saved: {os.path.join(OUT_DIR, 'linear_summary.csv')}")

print("\nDone.")
