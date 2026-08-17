"""
IDS lambda bracket search at arm_diff=0.1.

1. Run lambda in {0.3, 0.4} (+ reuse cached 0.5, 1.0 if present)
2. Linear-interpolate lambda to hit mix-algo mean power
3. Run that interpolated lambda once

Power method matches postdiff_fig1 (Wald + per-algo H0 crit) so comparable
to cached Figure 3.1 baselines.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from bandit_simulation import bandit_algorithm as algo
from bandit_simulation import sim_wrapper as sw
from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.test_procedure_configurator import ANOVA

N_REP = 5_000
HORIZON = 785
BURN_IN = 1
ARM_DIFF = 0.1
N_MC = 200
TARGET_LAMBDAS = [0.3, 0.4]
REUSE_LAMBDAS = [0.5, 1.0]

OUT_DIR = os.path.join(os.path.dirname(__file__), "_out")
BASELINE_JSON = os.path.join(OUT_DIR, "postdiff_fig1_results.json")
CACHE_JSON = os.path.join(OUT_DIR, "ids_lambda_bracket.json")

# From completed 746268 run (5k, fig1-style Wald AIT @ diff=0.1)
PRIOR = {
    0.5: 0.6528,
    1.0: 0.4613,
}


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


def power_at_diff(policy, arm_diff, crit):
    p1, p2 = 0.5 + arm_diff / 2, 0.5 - arm_diff / 2
    res = sw.run_simulation(policy=policy, sim_config=make_sim_config([p1, p2]))
    wald = res.wald_test().flatten()
    return float(np.nanmean(np.abs(wald) > crit))


def eval_lambda(lam):
    policy = algo.IDS({"n_mc": N_MC, "lambda": lam})
    t0 = time.perf_counter()
    h0_res = sw.run_simulation(policy=policy, sim_config=make_sim_config([0.5, 0.5]))
    wald_h0 = h0_res.wald_test().flatten()
    crit = float(np.nanquantile(np.abs(wald_h0), 0.95))
    fpr = float(np.nanmean(np.abs(wald_h0) > crit))
    power = power_at_diff(policy, ARM_DIFF, crit)
    elapsed = time.perf_counter() - t0
    return {"lambda": lam, "crit": crit, "fpr": fpr, "power": power, "elapsed_s": elapsed}


def mix_target_power():
    arm_diffs = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175,
                 0.2, 0.25, 0.3, 0.35, 0.4]
    idx = arm_diffs.index(ARM_DIFF)
    with open(BASELINE_JSON) as f:
        data = json.load(f)["metrics"]
    mix_keys = ["EpsTS (0.345)", "TSPostDiffUR (0.110)", "TSProbClip (0.425)"]
    powers = [float(data[k]["power"][idx]) for k in mix_keys]
    ur = float(data["UR"]["power"][idx])
    return float(np.mean(powers)), ur, powers


def interpolate_lambda(points, target):
    """Linear interp: find lam where power(lam)=target using bracketing pair."""
    pts = sorted((lam, p) for lam, p in points.items())
    for i in range(len(pts) - 1):
        lam0, p0 = pts[i]
        lam1, p1 = pts[i + 1]
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            lam_star = lam0 + (target - p0) * (lam1 - lam0) / (p1 - p0)
            return float(lam_star), (lam0, p0), (lam1, p1)
    # No bracket: extrapolate from two closest points to target
    pts_by_dist = sorted(pts, key=lambda x: abs(x[1] - target))
    (lam0, p0), (lam1, p1) = pts_by_dist[0], pts_by_dist[1]
    if p1 == p0:
        return None, (lam0, p0), (lam1, p1)
    lam_star = lam0 + (target - p0) * (lam1 - lam0) / (p1 - p0)
    return float(lam_star), (lam0, p0), (lam1, p1)


def main():
    target, ur_power, mix_powers = mix_target_power()
    print(f"Target mix mean power @ diff={ARM_DIFF}: {target:.4f}")
    print(f"  eps-TS={mix_powers[0]:.4f}  PostDiff={mix_powers[1]:.4f}  "
          f"ProbClip={mix_powers[2]:.4f}  UR={ur_power:.4f}\n")

    results = {lam: {"lambda": lam, "power": p, "from_cache": True}
               for lam, p in PRIOR.items()}

    for lam in TARGET_LAMBDAS:
        print(f"=== lambda={lam} ===")
        r = eval_lambda(lam)
        results[lam] = r
        print(f"  FPR={r['fpr']:.4f}  power={r['power']:.4f}  "
              f"loss={ur_power - r['power']:.4f}  ({r['elapsed_s']:.1f}s)\n")

    power_map = {lam: r["power"] for lam, r in results.items()}
    lam_star, (la, pa), (lb, pb) = interpolate_lambda(power_map, target)
    print(f"Linear interp between ({la}, {pa:.4f}) and ({lb}, {pb:.4f})")
    print(f"  -> lambda* = {lam_star:.4f} for target power {target:.4f}\n")

    if lam_star is not None and 0.05 <= lam_star <= 10:
        print(f"=== lambda={lam_star:.4f} (interpolated) ===")
        r_star = eval_lambda(lam_star)
        results[lam_star] = r_star
        print(f"  FPR={r_star['fpr']:.4f}  power={r_star['power']:.4f}  "
              f"loss={ur_power - r_star['power']:.4f}  "
              f"err vs target={r_star['power'] - target:+.4f}  "
              f"({r_star['elapsed_s']:.1f}s)\n")

    print("Summary (power @ diff=0.1):")
    for lam in sorted(results):
        r = results[lam]
        p = r["power"]
        tag = " *best*" if abs(p - target) == min(abs(x["power"] - target)
                                                  for x in results.values()) else ""
        src = " (cached)" if r.get("from_cache") else ""
        print(f"  lambda={lam:<6g}  power={p:.4f}  loss={ur_power - p:.4f}  "
              f"delta={p - target:+.4f}{src}{tag}")

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "arm_diff": ARM_DIFF,
        "n_rep": N_REP,
        "target_power": target,
        "ur_power": ur_power,
        "interp_lambda": lam_star,
        "results": {str(k): v for k, v in results.items()},
    }
    with open(CACHE_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {CACHE_JSON}")


if __name__ == "__main__":
    main()
