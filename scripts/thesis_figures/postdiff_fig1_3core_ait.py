"""
Rerun Ch.3 fig1 baselines through default 3-core Wald AIT @ 20k.

Same pipeline as ids_lambda05_full_figure.wald_ait_power:
  H1 sim -> H0 cores from combined_means -> per-rep interpolated Wald crit -> power

Usage:
    python scripts/thesis_figures/postdiff_fig1_3core_ait.py
    python scripts/thesis_figures/postdiff_fig1_3core_ait.py --plot-only  # verify @ diff=0.1
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from bandit_simulation import bandit_algorithm as algo

from ids_lambda05_full_figure import (
    ARM_DIFFS,
    BASELINE_KEYS,
    N_REP,
    OUT_DIR,
    make_sim_config,
    print_ait_config,
    wald_ait_power,
)

OUT_JSON = os.path.join(OUT_DIR, "postdiff_fig1_3core_ait.json")
ARM_DIFF_CHECK = 0.1
FIG1_JSON = os.path.join(OUT_DIR, "postdiff_fig1_results.json")

ALGOS = {
    "TS": algo.TSPostDiffURWithResample(0),
    "UR": algo.EpsTS(1.0),
    "EpsTS (0.345)": algo.EpsTS(0.345),
    "TSPostDiffUR (0.110)": algo.TSPostDiffURWithResample(0.110),
    "TSProbClip (0.425)": algo.TSProbClip(0.425),
}


def run_all():
    sc = make_sim_config([0.5, 0.5])
    print(f"=== Fig1 baselines, n_rep={N_REP}, 3-core Wald AIT ===")
    print_ait_config(sc)

    metrics = {}
    for name, policy in ALGOS.items():
        print(f"\n--- {name} ---")
        m = {"reward": [], "power": [], "prop_opt": []}
        for w in ARM_DIFFS:
            p1, p2 = 0.5 + w / 2, 0.5 - w / 2
            t0 = time.perf_counter()
            out = wald_ait_power(copy.deepcopy(policy), [p1, p2])
            m["reward"].append(out["reward"])
            m["power"].append(out["power"])
            m["prop_opt"].append(out["prop_opt"])
            print(
                f"  arm_diff={w:.3f}  reward={out['reward']:.4f}  "
                f"power={out['power']:.4f}  prop={out['prop_opt']:.4f}  "
                f"({time.perf_counter()-t0:.1f}s)"
            )
        metrics[name] = m

    tp = sc.test_procedure
    payload = {
        "config": {
            "n_rep": N_REP,
            "horizon": sc.horizon,
            "arm_diffs": ARM_DIFFS,
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
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
    return metrics


def verify_at_diff():
    """Compare @ arm_diff=0.1 against cached fig1 single-H0 results."""
    idx = ARM_DIFFS.index(ARM_DIFF_CHECK)
    with open(FIG1_JSON) as f:
        fig1 = json.load(f)["metrics"]
    with open(OUT_JSON) as f:
        core3 = json.load(f)["metrics"]

    print(f"\n=== Verification @ arm_diff={ARM_DIFF_CHECK} ===")
    print(f"{'algo':<25s}  {'fig1':>8s}  {'3-core':>8s}  {'delta':>8s}")
    for key in ALGOS:
        p1 = fig1[key]["power"][idx]
        p3 = core3[key]["power"][idx]
        print(f"{key:<25s}  {p1:8.4f}  {p3:8.4f}  {p3-p1:+8.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_at_diff()
        return
    run_all()
    verify_at_diff()


if __name__ == "__main__":
    main()
