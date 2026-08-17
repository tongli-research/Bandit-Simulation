"""
Order-sensitivity of ART vs Queue vs MLE under a fixed reward multiset (vectorized).

Fixed reward pool: exactly 100 zeros and 100 ones (T=200), so counts never
vary, only order. Each of N_REP_OUTER outer reps draws one random permutation
of that multiset as the observed reward history (only ART's inner mechanism
depends on this specific permutation).

For each outer rep, N_REP_INNER inner reps are run fully vectorized:
  - ART: fixed observed reward order, only TS action-selection draws vary.
  - Queue: reward multiset reshuffled independently per inner rep, then
    replayed with resimulated TS action selection.
  - MLE: pooled p_hat = 100/200 = 0.5 exactly, fresh Bernoulli(0.5, 0.5) TS
    trials simulated from scratch.

The LRT statistic (H0=(0.5,0.5) vs H1=(0.6,0.4) reference) is computed per
inner rep; the 95th percentile of the null distribution is that outer rep's
critical value. Queue and MLE critical values are near-constant across outer
reps (variation only from inner Monte Carlo noise), whereas ART's varies with
the observed permutation.

Burn-in = 2 pulls/arm for all three methods.
"""

import time

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

N_REP_OUTER = 100
N_REP_INNER = 5000
HORIZON = 200
BURN_IN_PER_ARM = 2

N_ZEROS = 100
N_ONES = 100
assert N_ZEROS + N_ONES == HORIZON

P_H0 = (0.5, 0.5)
P_H1_REF = (0.6, 0.4)  # reference point for the LRT log-ratio table only

BASE_SEED = 20260717

LOG_RATIO = np.zeros((2, 2))  # [arm][reward]
for _arm in range(2):
    _p1, _p0 = P_H1_REF[_arm], P_H0[_arm]
    for _r in range(2):
        LOG_RATIO[_arm, _r] = _r * np.log(_p1 / _p0) + (1 - _r) * np.log((1 - _p1) / (1 - _p0))

BASE_MULTISET = np.array([0] * N_ZEROS + [1] * N_ONES, dtype=int)


# ── Vectorized ART replay, fixed reward stream, inner reps varying only action selection ──

def run_art_batch_fixed_stream(reward_stream: np.ndarray, n_inner: int, rng: np.random.Generator):
    x = np.zeros((n_inner, 2), dtype=int)
    n = np.zeros((n_inner, 2), dtype=int)
    arms = np.zeros((n_inner, HORIZON), dtype=int)
    idx = np.arange(n_inner)
    t = 0

    for _ in range(BURN_IN_PER_ARM):
        for arm in (0, 1):
            r = int(reward_stream[t])
            arms[:, t] = arm
            x[:, arm] += r
            n[:, arm] += 1
            t += 1

    while t < HORIZON:
        s0 = rng.beta(1 + x[:, 0], 1 + n[:, 0] - x[:, 0])
        s1 = rng.beta(1 + x[:, 1], 1 + n[:, 1] - x[:, 1])
        arm = (s1 > s0).astype(int)  # continuous draws: ties have prob 0
        r = int(reward_stream[t])
        arms[:, t] = arm
        x[idx, arm] += r
        n[idx, arm] += 1
        t += 1

    rewards = np.tile(reward_stream, (n_inner, 1))
    return arms, rewards


def run_queue_batch(n_inner: int, rng: np.random.Generator):
    """Q-based: reshuffle the fixed multiset independently per inner rep, then replay."""
    keys = rng.random((n_inner, HORIZON))
    order = np.argsort(keys, axis=1)
    tiled = np.tile(BASE_MULTISET, (n_inner, 1))
    shuffled = np.take_along_axis(tiled, order, axis=1)

    x = np.zeros((n_inner, 2), dtype=int)
    n = np.zeros((n_inner, 2), dtype=int)
    arms = np.zeros((n_inner, HORIZON), dtype=int)
    idx = np.arange(n_inner)
    t = 0

    for _ in range(BURN_IN_PER_ARM):
        for arm in (0, 1):
            r = shuffled[:, t]
            arms[:, t] = arm
            x[:, arm] += r
            n[:, arm] += 1
            t += 1

    while t < HORIZON:
        s0 = rng.beta(1 + x[:, 0], 1 + n[:, 0] - x[:, 0])
        s1 = rng.beta(1 + x[:, 1], 1 + n[:, 1] - x[:, 1])
        arm = (s1 > s0).astype(int)
        r = shuffled[:, t]
        arms[:, t] = arm
        x[idx, arm] += r
        n[idx, arm] += 1
        t += 1

    return arms, shuffled


def run_mle_batch(n_inner: int, rng: np.random.Generator, p_arm=(0.5, 0.5)):
    """MLE: pooled p_hat under the fixed 100/100 multiset is always 0.5; simulate fresh trials."""
    x = np.zeros((n_inner, 2), dtype=int)
    n = np.zeros((n_inner, 2), dtype=int)
    arms = np.zeros((n_inner, HORIZON), dtype=int)
    rewards = np.zeros((n_inner, HORIZON), dtype=int)
    idx = np.arange(n_inner)
    t = 0

    for _ in range(BURN_IN_PER_ARM):
        for arm in (0, 1):
            r = rng.binomial(1, p_arm[arm], size=n_inner)
            arms[:, t] = arm
            rewards[:, t] = r
            x[:, arm] += r
            n[:, arm] += 1
            t += 1

    while t < HORIZON:
        s0 = rng.beta(1 + x[:, 0], 1 + n[:, 0] - x[:, 0])
        s1 = rng.beta(1 + x[:, 1], 1 + n[:, 1] - x[:, 1])
        arm = (s1 > s0).astype(int)
        p_of_arm = np.where(arm == 0, p_arm[0], p_arm[1])
        r = rng.binomial(1, p_of_arm)
        arms[:, t] = arm
        rewards[:, t] = r
        x[idx, arm] += r
        n[idx, arm] += 1
        t += 1

    return arms, rewards


def lrt_stat_batch(arms: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    return np.sum(LOG_RATIO[arms, rewards], axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_all():
    crit_art = np.empty(N_REP_OUTER, dtype=float)
    crit_queue = np.empty(N_REP_OUTER, dtype=float)
    crit_mle = np.empty(N_REP_OUTER, dtype=float)

    t_start = time.time()
    for rep in range(N_REP_OUTER):
        rng = np.random.default_rng(BASE_SEED + rep)
        reward_stream = rng.permutation(BASE_MULTISET)

        arms, rewards = run_art_batch_fixed_stream(reward_stream, N_REP_INNER, rng)
        crit_art[rep] = float(np.percentile(lrt_stat_batch(arms, rewards), 95))

        rng_q = np.random.default_rng(rng.integers(0, 2**63 - 1))
        arms_q, rewards_q = run_queue_batch(N_REP_INNER, rng_q)
        crit_queue[rep] = float(np.percentile(lrt_stat_batch(arms_q, rewards_q), 95))

        rng_mle = np.random.default_rng(rng.integers(0, 2**63 - 1))
        arms_mle, rewards_mle = run_mle_batch(N_REP_INNER, rng_mle)
        crit_mle[rep] = float(np.percentile(lrt_stat_batch(arms_mle, rewards_mle), 95))

        if (rep + 1) % 20 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (rep + 1) * (N_REP_OUTER - rep - 1)
            print(f"  {rep + 1}/{N_REP_OUTER}  elapsed={elapsed:.1f}s  ETA={eta:.1f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone: {N_REP_OUTER} outer reps x {N_REP_INNER} inner reps in {elapsed:.1f}s")

    df = pd.DataFrame({"ART_critval": crit_art, "Queue_critval": crit_queue, "MLE_critval": crit_mle})
    print(f"\n95% critical value across {N_REP_OUTER} outer reps (fixed 100/100 multiset, order permuted):")
    print(df.describe().loc[["mean", "std", "min", "max"]])

    return df


if __name__ == "__main__":
    df = run_all()
    out_path = "_ai_generated/results/_ai_20260717_sim3_art_order_sensitivity_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
