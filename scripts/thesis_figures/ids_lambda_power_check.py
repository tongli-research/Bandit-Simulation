"""
IDS lambda screening for Thesis Ch.3 Figure 3.1 power panel.

Uses the framework AIT pipeline (run_task_common / verify_table pattern):
  1. Simulate once at the setting of interest (H1)
  2. Derive per-rep null locations from H1 combined_means
  3. Simulate algorithm-induced null at H0 cores
  4. Interpolate critical region per rep -> compute_power

No separate upfront H0 run at p1=p2=0.5. FPR at diff=0 is optional verification.

Screening mode (default): match AIT power at arm_diff=0.1 vs cached mix algorithms.

Usage:
    python scripts/thesis_figures/ids_lambda_power_check.py
    python scripts/thesis_figures/ids_lambda_power_check.py --plot-only

Output:
    scripts/thesis_figures/_out/ids_lambda_power_check.pdf
    scripts/thesis_figures/_out/ids_lambda_power_check.json
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

N_REP = 5_000
HORIZON = 785
BURN_IN = 1
SCREEN_ARM_DIFF = 0.1
IDS_LAMBDAS = [0.5, 1.0, 2.0, 4.0]
N_MC = 200

OUT_DIR = os.path.join(os.path.dirname(__file__), "_out")
BASELINE_JSON = os.path.join(OUT_DIR, "postdiff_fig1_results.json")
OUT_JSON = os.path.join(OUT_DIR, "ids_lambda_power_check.json")
OUT_PDF = os.path.join(OUT_DIR, "ids_lambda_power_check.pdf")

BASELINE_DISPLAY = {
    "EpsTS (0.345)": r"$\varepsilon$-TS (0.345)",
    "TSPostDiffUR (0.110)": "TS-PostDiff (0.110)",
    "TSProbClip (0.425)": "TS-ProbClip (0.425)",
}

MIX_LABELS = list(BASELINE_DISPLAY.values())


def make_sim_config(arm_means, n_rep=N_REP):
    sc = SimulationConfig(
        n_rep=n_rep,
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


def run_ait_at_setting(policy, arm_means):
    """Full AIT pipeline for one (algorithm, arm-diff) setting."""
    sim_config = make_sim_config(arm_means)
    sim_config = copy.deepcopy(sim_config)
    sim_config.manual_init()

    t0 = time.perf_counter()
    h1_res = sw.run_simulation(policy=policy, sim_config=sim_config)

    weight, h0_sim_loc_array = sim_config.test_procedure.get_h0_cores_and_weights(
        h1_res.combined_means[:, -1, :]
    )
    h1_n_rep = sim_config.n_rep
    sim_config.n_rep = len(h0_sim_loc_array)
    h0_res = sw.run_simulation(
        policy=copy.deepcopy(policy),
        sim_config=sim_config,
        arm_mean_reward_dist=h0_sim_loc_array[:, np.newaxis],
    )
    crit_boundary, _se, _core_crit = sim_config.test_procedure.get_adjusted_crit_region(
        weight, h0_res
    )
    sim_config.n_rep = h1_n_rep

    tp = sim_config.test_procedure
    power_curve = tp.compute_power(
        crit_boundary=crit_boundary,
        h1_sim_result=h1_res,
        ground_truth_arm_mean_dist=sim_config.arm_mean_reward_dist,
    )
    power_interp = sw.get_interpolation(power_curve, sim_config.step_schedule)
    power_at_T = float(power_interp[-1])

    elapsed = time.perf_counter() - t0
    return {
        "power_at_T": power_at_T,
        "crit_boundary": crit_boundary,
        "h1_res": h1_res,
        "sim_config": sim_config,
        "elapsed_s": elapsed,
    }


def baseline_power_at_diff(arm_diff):
    """Cached mix-algo power from postdiff_fig1_results.json (not re-run)."""
    arm_diffs = [
        0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175,
        0.2, 0.25, 0.3, 0.35, 0.4,
    ]
    idx = arm_diffs.index(arm_diff)
    with open(BASELINE_JSON) as f:
        data = json.load(f)
    out = {}
    for key, label in BASELINE_DISPLAY.items():
        out[label] = float(data["metrics"][key]["power"][idx])
    ur_power = float(data["metrics"]["UR"]["power"][idx])
    return out, ur_power


def run_screen():
    print(f"IDS lambda screening at arm_diff={SCREEN_ARM_DIFF}, n_rep={N_REP}")
    print("AIT: H0 derived from each H1 run (no separate H0 simulation)\n")

    mix_power, ur_power = baseline_power_at_diff(SCREEN_ARM_DIFF)
    mix_mean = float(np.mean(list(mix_power.values())))
    print(f"Target (cached mix algos @ diff={SCREEN_ARM_DIFF}, fig1 Wald-AIT):")
    for label, p in mix_power.items():
        print(f"  {label:25s}  power={p:.4f}  loss={ur_power - p:.4f}")
    print(f"  {'mix mean':25s}  power={mix_mean:.4f}  loss={ur_power - mix_mean:.4f}\n")

    results = {}
    for lam in IDS_LAMBDAS:
        policy = algo.IDS({"n_mc": N_MC, "lambda": lam})
        p1, p2 = 0.5 + SCREEN_ARM_DIFF / 2, 0.5 - SCREEN_ARM_DIFF / 2
        out = run_ait_at_setting(policy, [p1, p2])
        name = f"IDS (lambda={lam})"
        results[name] = {
            "lambda": lam,
            "power_at_T": out["power_at_T"],
            "power_loss_from_UR": ur_power - out["power_at_T"],
            "elapsed_s": out["elapsed_s"],
        }
        diff_from_mix = out["power_at_T"] - mix_mean
        print(
            f"{name:20s}  AIT power={out['power_at_T']:.4f}  "
            f"loss={ur_power - out['power_at_T']:.4f}  "
            f"vs mix mean: {diff_from_mix:+.4f}  ({out['elapsed_s']:.1f}s)"
        )

    best = min(results, key=lambda k: abs(results[k]["power_at_T"] - mix_mean))
    print(f"\nClosest to mix mean: {best}")

    payload = {
        "mode": "screen",
        "arm_diff": SCREEN_ARM_DIFF,
        "n_rep": N_REP,
        "baseline_mix_power": mix_power,
        "ur_power": ur_power,
        "ids_results": results,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {OUT_JSON}")
    return results, mix_power, ur_power


def plot_screen(results, mix_power, ur_power):
    labels = MIX_LABELS + list(results.keys())
    powers = [mix_power[l] for l in MIX_LABELS] + [r["power_at_T"] for r in results.values()]
    losses = [ur_power - p for p in powers]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#DAA520", "green", "purple"] + ["#E41A1C", "#377EB8", "#984EA3", "#FF7F00"]
    x = np.arange(len(labels))
    ax.bar(x, losses, color=colors[: len(labels)], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Power loss from UR")
    ax.set_title(
        f"AIT power @ arm_diff={SCREEN_ARM_DIFF} (n={HORIZON}, IDS n_rep={N_REP})\n"
        "Gold/green/purple: cached mix algos; bars: IDS lambda sweep"
    )
    ax.axhline(ur_power - np.mean(list(mix_power.values())), color="gray", linestyle=":",
               linewidth=1, label="mix mean loss")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        with open(OUT_JSON) as f:
            data = json.load(f)
        results = data["ids_results"]
        mix_power = data["baseline_mix_power"]
        ur_power = data["ur_power"]
    else:
        results, mix_power, ur_power = run_screen()

    plot_screen(results, mix_power, ur_power)


if __name__ == "__main__":
    main()
