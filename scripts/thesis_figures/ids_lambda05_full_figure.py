"""
IDS (lambda=0.5) full curve at 20k reps + 3-panel figure with Ch.3 baselines.

AIT (per arm-diff setting, same as run_task_common / Ch. 2):
  1. H1 simulation at setting
  2. H0 cores + weights from H1 combined_means (linear interp)
  3. H0 simulation at cores (n_crit_sim_rep per core)
  4. Per-rep interpolated Wald critical value -> power

@ n_rep=20k defaults: 2 groups -> 3 cores, 6667 reps/core, ~20001 H0 reps.

Baselines for plot: postdiff_fig1_results.json (Wald + single-H0 shortcut).
See docstring in main() for why that differs from IDS AIT here.

Usage:
    python scripts/thesis_figures/ids_lambda05_full_figure.py
    python scripts/thesis_figures/ids_lambda05_full_figure.py --plot-only
"""

import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from bandit_simulation import bandit_algorithm as algo
from bandit_simulation import sim_wrapper as sw
from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.test_procedure_configurator import ANOVA

N_REP = 20_000
HORIZON = 785
BURN_IN = 1
IDS_LAMBDA = 0.6
N_MC = 200
WALD_ALPHA = 0.05

ARM_DIFFS = [
    0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175,
    0.2, 0.25, 0.3, 0.35, 0.4,
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "_out")
BASELINE_JSON = os.path.join(OUT_DIR, "postdiff_fig1_3core_ait.json")
BASELINE_JSON_FIG1 = os.path.join(OUT_DIR, "postdiff_fig1_results.json")
IDS_JSON = os.path.join(OUT_DIR, "ids_lambda06_20k_ait.json")
IDS_FIG1_JSON = os.path.join(OUT_DIR, "ids_lambda05_20k.json")
OUT_PDF = os.path.join(OUT_DIR, "ids_lambda05_three_panel.pdf")
OUT_AIT_COMPARE_PDF = os.path.join(OUT_DIR, "ids_lambda05_ait_comparison.pdf")

IDS_LABEL = "IDS (lambda=0.6)"
IDS_FIG1_LABEL = "IDS (lambda=0.5, fig1 AIT)"
IDS_3CORE_LABEL = "IDS (lambda=0.5, 3-core AIT)"

BASELINE_KEYS = {
    "TS": "TS",
    "UR": "UR",
    "EpsTS (0.345)": r"$\varepsilon$-TS (0.345)",
    "TSPostDiffUR (0.110)": "TS-PostDiff (0.110)",
}

STYLES = {
    "TS": dict(color="black", marker="*", linestyle="--", linewidth=1.5),
    "UR": dict(color="gray", marker="d", linestyle="--", linewidth=1.0),
    r"$\varepsilon$-TS (0.345)": dict(color="green", marker="o", linestyle="-", linewidth=1.5),
    "TS-PostDiff (0.110)": dict(color="#DAA520", marker="s", linestyle="-", linewidth=1.5),
    IDS_LABEL: dict(color="#E41A1C", marker="v", linestyle="-.", linewidth=1.5),
    IDS_FIG1_LABEL: dict(color="#E41A1C", marker="o", linestyle="--", linewidth=1.5),
    IDS_3CORE_LABEL: dict(color="#E41A1C", marker="v", linestyle="-", linewidth=1.5),
}

PLOT_ORDER = ["UR", "TS", r"$\varepsilon$-TS (0.345)", "TS-PostDiff (0.110)", IDS_LABEL]
AIT_COMPARE_ORDER = [
    "UR", "TS", r"$\varepsilon$-TS (0.345)", "TS-PostDiff (0.110)",
    IDS_FIG1_LABEL, IDS_3CORE_LABEL,
]


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
        test_procedure=ANOVA(min_effect=0.0),
        reward_evaluation_method="regret",
    )
    sc.manual_init()
    return sc


def print_ait_config(sim_config):
    tp = sim_config.test_procedure
    n_cores = tp.n_crit_sim_groups + 1  # linear
    print(
        f"AIT config: groups={tp.n_crit_sim_groups}, cores={n_cores}, "
        f"reps/core={tp.n_crit_sim_rep}, total_H0~={n_cores * tp.n_crit_sim_rep}"
    )


def wald_ait_power(policy, arm_means):
    """Framework H0 binning + Wald test (Thesis Fig 3.1 test stat)."""
    sim_config = make_sim_config(arm_means)
    tp = sim_config.test_procedure

    h1_res = sw.run_simulation(policy=policy, sim_config=sim_config)

    weight, h0_sim_loc_array = tp.get_h0_cores_and_weights(
        h1_res.combined_means[:, -1, :]
    )
    h1_n_rep = sim_config.n_rep
    sim_config.n_rep = len(h0_sim_loc_array)
    h0_res = sw.run_simulation(
        policy=copy.deepcopy(policy),
        sim_config=sim_config,
        arm_mean_reward_dist=h0_sim_loc_array[:, np.newaxis],
    )
    sim_config.n_rep = h1_n_rep

    wald_h0 = np.abs(h0_res.wald_test()[:, -1, :].reshape(-1))
    n_cores = weight.shape[1]
    rep_per_core = tp.n_crit_sim_rep
    core_crit = np.array([
        np.quantile(wald_h0[i * rep_per_core:(i + 1) * rep_per_core], 1 - WALD_ALPHA)
        for i in range(n_cores)
    ])
    crit_boundary = weight @ core_crit  # (n_rep,)

    wald_h1 = np.abs(h1_res.wald_test()[:, -1, :].reshape(-1))
    power = float(np.mean(wald_h1 > crit_boundary))

    mean_reward = float(np.nanmean(h1_res.combined_means[:, -1, :]))
    action_counts = h1_res.arm_counts[:, -1, :]
    sup_arm = np.argmax(h1_res.arm_means[:, -1, :], axis=1)
    prop = float(np.mean(action_counts[np.arange(len(sup_arm)), sup_arm] / HORIZON))

    return {
        "power": power,
        "reward": mean_reward,
        "prop_opt": prop,
        "n_cores": n_cores,
        "n_crit_sim_rep": rep_per_core,
    }


def run_ids_full():
    policy = algo.IDS({"n_mc": N_MC, "lambda": IDS_LAMBDA})
    sc = make_sim_config([0.5, 0.5])
    print(f"=== IDS lambda={IDS_LAMBDA}, n_rep={N_REP}, Wald AIT ===")
    print_ait_config(sc)

    metrics = {"reward": [], "power": [], "prop_opt": []}
    for w in ARM_DIFFS:
        p1, p2 = 0.5 + w / 2, 0.5 - w / 2
        t0 = time.perf_counter()
        out = wald_ait_power(policy, [p1, p2])
        metrics["reward"].append(out["reward"])
        metrics["power"].append(out["power"])
        metrics["prop_opt"].append(out["prop_opt"])
        print(
            f"  arm_diff={w:.3f}  reward={out['reward']:.4f}  "
            f"power={out['power']:.4f}  prop={out['prop_opt']:.4f}  "
            f"({time.perf_counter()-t0:.1f}s)"
        )

    tp = sc.test_procedure
    payload = {
        "config": {
            "n_rep": N_REP,
            "horizon": HORIZON,
            "lambda": IDS_LAMBDA,
            "n_mc": N_MC,
            "ait": {
                "n_crit_sim_groups": tp.n_crit_sim_groups,
                "n_cores": tp.n_crit_sim_groups + 1,
                "n_crit_sim_rep": tp.n_crit_sim_rep,
                "method": "linear",
                "test": "wald",
            },
        },
        "metrics": metrics,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(IDS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {IDS_JSON}")
    return metrics


def load_all_metrics():
    baseline_path = BASELINE_JSON if os.path.isfile(BASELINE_JSON) else BASELINE_JSON_FIG1
    with open(baseline_path) as f:
        baseline = json.load(f)["metrics"]
    metrics = {BASELINE_KEYS[k]: baseline[k] for k in BASELINE_KEYS if k in baseline}
    if os.path.isfile(IDS_JSON):
        with open(IDS_JSON) as f:
            metrics[IDS_LABEL] = json.load(f)["metrics"]
    return metrics


def load_ait_comparison_metrics():
    metrics = load_all_metrics()
    with open(IDS_FIG1_JSON) as f:
        fig1 = json.load(f)["metrics"]
    with open(IDS_JSON) as f:
        core3 = json.load(f)["metrics"]
    metrics[IDS_FIG1_LABEL] = fig1
    metrics[IDS_3CORE_LABEL] = core3
    return metrics


def plot_three_panel(metrics, out_pdf=OUT_PDF, plot_order=PLOT_ORDER, title_suffix=None):
    diffs = np.array(ARM_DIFFS)
    ur_power = np.array(metrics["UR"]["power"])
    ts_reward = np.array(metrics["TS"]["reward"])

    fig, axes = plt.subplots(3, 1, figsize=(7, 12), sharex=True)

    ax = axes[0]
    for name in plot_order:
        ax.plot(diffs, ur_power - np.array(metrics[name]["power"]),
                **STYLES[name], markersize=5, label=name)
    ax.set_ylabel("Power Loss from UR")
    ax.set_title("Relative Power Loss (UR baseline = 0)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in plot_order:
        ax.plot(diffs, ts_reward - np.array(metrics[name]["reward"]),
                **STYLES[name], markersize=5, label=name)
    ax.set_ylabel("Reward Loss from TS")
    ax.set_title("Relative Reward Loss (TS baseline = 0)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for name in plot_order:
        ax.plot(diffs, metrics[name]["prop_opt"], **STYLES[name], markersize=5, label=name)
    ax.set_ylabel("Prop Superior")
    ax.set_xlabel("Arm Difference")
    ax.set_title("Proportion Allocated to Superior Arm")
    ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    if title_suffix is None:
        title_suffix = "20k reps, default 3-core Wald AIT (all algorithms)"
    fig.suptitle(title_suffix, fontsize=9, y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def plot_ait_comparison(metrics=None):
    if metrics is None:
        metrics = load_ait_comparison_metrics()
    plot_three_panel(
        metrics,
        out_pdf=OUT_AIT_COMPARE_PDF,
        plot_order=AIT_COMPARE_ORDER,
        title_suffix=(
            "IDS lambda=0.5 @ 20k: fig1 single-H0 Wald vs 3-core Wald AIT "
            "(baselines: cached fig1)"
        ),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--ait-comparison", action="store_true",
                        help="Plot fig1 vs 3-core AIT for IDS (no simulation)")
    args = parser.parse_args()

    if args.ait_comparison:
        plot_ait_comparison()
        return

    if not args.plot_only:
        run_ids_full()
    plot_three_panel(load_all_metrics())


if __name__ == "__main__":
    main()
