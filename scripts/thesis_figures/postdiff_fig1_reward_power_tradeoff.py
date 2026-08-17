"""
Recreate PostDiff paper Figure 1 (thesis Chapter 2, Fig 2.1).

Two-panel plot:
  (A) Relative Power Loss from UR (UR baseline = 0)
  (B) Relative Reward Loss from TS (TS baseline = 0)

All algorithms use AIT (Algorithm-Induced Test) to control FPR at 0.05:
  - Simulate each algorithm under H0 to get its own critical value
  - Use that critical value to compute power under H1

Setting:
  - 2-armed Bernoulli bandit, centered at 0.5
  - arm difference w → p1 = 0.5 + w/2, p2 = 0.5 - w/2
  - Sample size (horizon) = 785, burn-in = 1/arm
  - 10,000 reps per setting
  - Algorithms: TS, UR, EpsTS(0.345), TSProbClip(0.420), TSPostDiffURWithResample(0.110)

Output: scripts/thesis_figures/_out/postdiff_fig1_reward_power_tradeoff.pdf
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from bandit_simulation import bandit_algorithm as algo
from bandit_simulation import sim_wrapper as sw
from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.test_procedure_configurator import ANOVA

# === Configuration ============================================================

N_REP = 20_000
HORIZON = 785
BURN_IN = 1

ARM_DIFFS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4]

ALGOS = {
    "TS":                        algo.TSPostDiffURWithResample(0),
    "UR":                        algo.EpsTS(1.0),
    r"$\varepsilon$-TS (0.345)": algo.EpsTS(0.345),
    "TS-PostDiff (0.110)":       algo.TSPostDiffURWithResample(0.110),
    "TS-ProbClip (0.425)":       algo.TSProbClip(0.425),
}

STYLES = {
    "TS":                        dict(color="black",   marker="*", linestyle="--", linewidth=1.5),
    "UR":                        dict(color="gray",    marker="d", linestyle="--", linewidth=1.0),
    r"$\varepsilon$-TS (0.345)": dict(color="green",   marker="o", linestyle="-",  linewidth=1.5),
    "TS-PostDiff (0.110)":       dict(color="#DAA520", marker="s", linestyle="-",  linewidth=1.5),
    "TS-ProbClip (0.425)":       dict(color="purple",  marker="^", linestyle="-",  linewidth=1.5),
}

OUT_DIR = os.path.join(os.path.dirname(__file__), "_out")
os.makedirs(OUT_DIR, exist_ok=True)


# === Simulation ===============================================================

def make_sim_config(arm_means):
    sc = SimulationConfig(
        n_rep=N_REP,
        n_arm=2,
        horizon=HORIZON,
        burn_in_per_arm=BURN_IN,
        arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": arm_means, "scale": 0},
        },
        test_procedure=ANOVA(),
        reward_evaluation_method="regret",
    )
    sc.manual_init()
    return sc


def compute_ait_critical_values():
    """Simulate each algorithm under H0 to get per-algorithm critical values."""
    print("=== Computing AIT critical values under H0 (p1=p2=0.5) ===")
    h0_means = [0.5, 0.5]
    crit_values = {}

    for algo_name, algo_obj in ALGOS.items():
        sim_config = make_sim_config(h0_means)
        res = sw.run_simulation(policy=algo_obj, sim_config=sim_config)
        wald_flat = res.wald_test().flatten()
        # Two-sided 95th percentile (AIT critical value), ignore rare NaNs
        crit = float(np.nanquantile(np.abs(wald_flat), 0.95))
        fpr = float(np.nanmean(np.abs(wald_flat) > crit))
        crit_values[algo_name] = crit
        print(f"  {algo_name:25s}: crit={crit:.4f}  FPR={fpr:.4f}")

    return crit_values


def run_all():
    """Run all algorithms across all arm differences. Returns (metrics, crit_values)."""
    crit_values = compute_ait_critical_values()

    metrics = {name: {"reward": [], "power": [], "prop_opt": []} for name in ALGOS}

    print("\n=== Running simulations ===")
    for w in ARM_DIFFS:
        p1 = 0.5 + w / 2
        p2 = 0.5 - w / 2
        arm_means = [p1, p2]
        print(f"arm_diff={w:.3f}  (p1={p1:.4f}, p2={p2:.4f})")

        for algo_name, algo_obj in ALGOS.items():
            sim_config = make_sim_config(arm_means)
            res = sw.run_simulation(policy=algo_obj, sim_config=sim_config)

            # Mean reward at final step
            mean_reward = float(np.nanmean(res.combined_means[:, -1, :]))
            metrics[algo_name]["reward"].append(mean_reward)

            # Power using AIT critical value
            wald_flat = res.wald_test().flatten()
            crit = crit_values[algo_name]
            power = float(np.nanmean(np.abs(wald_flat) > crit))
            metrics[algo_name]["power"].append(power)

            # Prop Superior: allocation to the arm with highest estimated mean
            action_counts = res.arm_counts[:, -1, :]  # (n_rep, n_arm)
            final_means = res.arm_means[:, -1, :]
            sup_arm = np.argmax(final_means, axis=1)
            prop = action_counts[np.arange(len(sup_arm)), sup_arm] / HORIZON
            metrics[algo_name]["prop_opt"].append(float(np.mean(prop)))

            print(f"  {algo_name:25s}: reward={mean_reward:.4f}  "
                  f"power={power:.4f}  prop_opt={metrics[algo_name]['prop_opt'][-1]:.4f}")

    return metrics, crit_values


# === Plotting =================================================================

def plot_figure(metrics):
    """Create the 3-panel figure and save as PDF."""
    # Map old JSON keys to new display names
    KEY_MAP = {
        "EpsTS (0.345)": r"$\varepsilon$-TS (0.345)",
        "TSPostDiffURWithResample (0.110)": "TS-PostDiff (0.110)",
        "TSProbClip (0.425)": "TS-ProbClip (0.425)",
    }
    metrics = {KEY_MAP.get(k, k): v for k, v in metrics.items()}

    diffs = np.array(ARM_DIFFS)

    ur_power = np.array(metrics["UR"]["power"])
    ts_reward = np.array(metrics["TS"]["reward"])

    fig, axes = plt.subplots(3, 1, figsize=(7, 12), sharex=True)

    # --- (A) Relative Power Loss (UR baseline = 0) ---
    ax = axes[0]
    for name in ALGOS:
        power = np.array(metrics[name]["power"])
        power_loss = ur_power - power
        ax.plot(diffs, power_loss, **STYLES[name], markersize=5, label=name)
    ax.set_ylabel("Power Loss from UR")
    ax.set_title("Relative Power Loss (UR baseline = 0)")
    ax.grid(True, alpha=0.3)

    # --- (B) Relative Reward Loss (TS baseline = 0) ---
    ax = axes[1]
    for name in ALGOS:
        reward = np.array(metrics[name]["reward"])
        reward_loss = ts_reward - reward
        ax.plot(diffs, reward_loss, **STYLES[name], markersize=5, label=name)
    ax.set_ylabel("Reward Loss from TS")
    ax.set_title("Relative Reward Loss (TS baseline = 0)")
    ax.grid(True, alpha=0.3)

    # --- (C) Prop Superior ---
    ax = axes[2]
    for name in ALGOS:
        prop = np.array(metrics[name]["prop_opt"])
        ax.plot(diffs, prop, **STYLES[name], markersize=5, label=name)
    ax.set_ylabel("Prop Superior")
    ax.set_xlabel("Arm Difference")
    ax.set_title("Proportion Allocated to Superior Arm")
    ax.grid(True, alpha=0.3)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out_path = os.path.join(OUT_DIR, "postdiff_fig1_reward_power_tradeoff.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# === Save / Load ==============================================================

RESULTS_PATH = os.path.join(OUT_DIR, "postdiff_fig1_results.json")


def save_results(metrics, crit_values):
    """Save metrics and config to JSON for later re-plotting."""
    data = {
        "config": {
            "n_rep": N_REP,
            "horizon": HORIZON,
            "burn_in": BURN_IN,
            "arm_diffs": ARM_DIFFS,
            "algos": {name: str(obj) for name, obj in ALGOS.items()},
        },
        "crit_values": crit_values,
        "metrics": metrics,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results: {RESULTS_PATH}")


def load_results():
    """Load previously saved metrics."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    print(f"Loaded results: {RESULTS_PATH}")
    print(f"  config: n_rep={data['config']['n_rep']}, horizon={data['config']['horizon']}")
    return data["metrics"]


# === Main =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip simulation, plot from saved results")
    args = parser.parse_args()

    if args.plot_only:
        metrics = load_results()
    else:
        metrics, crit_values = run_all()
        save_results(metrics, crit_values)
    plot_figure(metrics)
