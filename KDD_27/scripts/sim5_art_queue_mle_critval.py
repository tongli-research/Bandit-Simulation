"""
ART vs Queue vs MLE critical-value comparison.

For each outer rep: draw one genuine TS trial under the true null
H0=(0.5,0.5) (total reward count is random, not fixed), giving the observed
history. Then, per method:
  - ART: fix that exact observed order, resimulate only TS action selection.
  - Queue: reshuffle that outer rep's own observed pool, replay with
    resimulated action selection.
  - MLE: pooled p_hat = mean(that outer rep's observed rewards), simulate
    fresh Bernoulli(p_hat, p_hat) trials.
Each method's critical value for that outer rep is the 95th percentile of its
N_REP_INNER-sample LRT-statistic null distribution. LRT reference:
H0=(0.5,0.5) vs H1=(0.55,0.30) (asymmetric, chosen to avoid the
path-independence masking effect a symmetric H1 can cause). Burn-in =
2 pulls/arm throughout. Settings: N_REP_OUTER=10000, N_REP_INNER=1001.

Output feeds the appendix critical-value figure (plot_critval_distribution.py),
which compares the three methods' critical-value distributions against the
theoretical optimal threshold.

Usage (run from this directory, KDD_27/):
    python scripts/sim5_art_queue_mle_critval.py
"""
import os
import time

import numpy as np
import pandas as pd

import importlib.util

_THIS_DIR = os.path.dirname(__file__)
_SPEC = importlib.util.spec_from_file_location(
    "sim3_base", os.path.join(_THIS_DIR, "sim3_art_order_sensitivity_vectorized.py")
)
sim3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim3)

N_REP_OUTER = 10000
N_REP_INNER = 1001

P_H0 = (0.5, 0.5)
P_H1_REF = (0.55, 0.30)

LOG_RATIO = np.zeros((2, 2))
for _arm in range(2):
    _p1, _p0 = P_H1_REF[_arm], P_H0[_arm]
    for _r in range(2):
        LOG_RATIO[_arm, _r] = _r * np.log(_p1 / _p0) + (1 - _r) * np.log((1 - _p1) / (1 - _p0))


def lrt_stat_batch(arms, rewards):
    return np.sum(LOG_RATIO[arms, rewards], axis=1)


def run_all():
    crit_art = np.empty(N_REP_OUTER, dtype=float)
    crit_queue = np.empty(N_REP_OUTER, dtype=float)
    crit_mle = np.empty(N_REP_OUTER, dtype=float)
    total_counts = np.empty(N_REP_OUTER, dtype=int)

    t_start = time.time()
    for rep in range(N_REP_OUTER):
        rng = np.random.default_rng(sim3.BASE_SEED + rep)

        arms_out, rewards_out = sim3.run_mle_batch(1, rng, p_arm=P_H0)
        reward_stream = rewards_out[0]
        total_counts[rep] = int(reward_stream.sum())

        rng_art = np.random.default_rng(rng.integers(0, 2**63 - 1))
        arms_art, rewards_art = sim3.run_art_batch_fixed_stream(reward_stream, N_REP_INNER, rng_art)
        crit_art[rep] = float(np.percentile(lrt_stat_batch(arms_art, rewards_art), 95))

        rng_q = np.random.default_rng(rng.integers(0, 2**63 - 1))
        sim3.BASE_MULTISET = reward_stream
        arms_q, rewards_q = sim3.run_queue_batch(N_REP_INNER, rng_q)
        crit_queue[rep] = float(np.percentile(lrt_stat_batch(arms_q, rewards_q), 95))

        p_hat = float(np.clip(reward_stream.mean(), 1e-6, 1 - 1e-6))
        rng_mle = np.random.default_rng(rng.integers(0, 2**63 - 1))
        arms_mle, rewards_mle = sim3.run_mle_batch(N_REP_INNER, rng_mle, p_arm=(p_hat, p_hat))
        crit_mle[rep] = float(np.percentile(lrt_stat_batch(arms_mle, rewards_mle), 95))

        if (rep + 1) % 500 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (rep + 1) * (N_REP_OUTER - rep - 1)
            print(f"  {rep + 1}/{N_REP_OUTER}  elapsed={elapsed:.1f}s  ETA={eta:.1f}s", flush=True)

    elapsed = time.time() - t_start
    df = pd.DataFrame({
        "ART_critval": crit_art, "Queue_critval": crit_queue, "MLE_critval": crit_mle,
        "total_reward_count": total_counts,
    })
    print(f"\nDone: {N_REP_OUTER} outer x {N_REP_INNER} inner in {elapsed:.1f}s")
    print(f"Observed total-count range across outer reps: {total_counts.min()}-{total_counts.max()} "
          f"(mean={total_counts.mean():.1f}, std={total_counts.std():.2f}; Binomial(200,0.5) theory std={np.sqrt(200*0.25):.2f})")
    print(df[["ART_critval", "Queue_critval", "MLE_critval"]].describe().loc[["mean", "std", "min", "max"]])

    return df


if __name__ == "__main__":
    df = run_all()
    out_path = "results/sim5_art_queue_mle_critval_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
