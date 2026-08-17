"""
Reproduces Table 2 (ART vs AIT vs Queue).

For each outer replication and each adaptive policy (TS, epsilon-greedy(0.1),
UCB1), the script measures a hypothesis test's power and false-positive rate
under three null-calibration methods:

  - POWER: outer reps generated under the test's P_H1 ground truth.
  - FPR:   outer reps generated under the test's P_H0 ground truth.
Only the ground-truth argument differs.

Outer-rep generation and all three calibration methods share one policy
simulator, so the replay matches the process that produced the observed data.

The three calibration methods:
  - ART: fix the outer run's time-indexed reward stream, replay the policy's
    arm-selection, and form the null distribution of the statistic. Depends on
    that rep's own exact stream.
  - Queue: reshuffle the outer run's observed reward pool (same multiset,
    random order per inner rep) and replay against it. Also per-outer-rep.
  - AIT (MLE): the null depends only on the outer run's combined-mean
    estimate, not on the specific stream. For each outer rep, simulate fresh
    null runs at that rep's combined-mean estimate under the same policy, form
    the null distribution of the statistic, and take the alpha critical value.

Supported --test values: t-test (2-arm, T-Control), anova, tukey, and
t-constant (3-arm). See TEST_SPECS.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# Tukey's _filter_stats takes nanmax over NaN-masked rows; an all-NaN row
# (a rep with no distinguishing arm) raises this but leaves the final reject
# decision unaffected, so silence just this message.
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from bandit_simulation.test_procedure_configurator import TControl, ANOVA, Tukey, TConstant
from bandit_simulation.test_statistics import t_test_from_counts, anova_from_counts, tukey_from_counts


# ── Configuration ────────────────────────────────────────────────────────────

N_REP_OUTER = 10000  # outer Monte Carlo reps for POWER and FPR
N_REP_0 = 1000       # inner reps for ART/Queue's per-outer-rep calibration

HORIZON = 300
ALPHA = 0.05
BURN_IN_PER_ARM = 2

# Arm-mean distributions for the outer reps. The 2-arm t-test uses fixed arm
# means. The 3-arm tests draw H1 arm means from Beta(3,3) and set all arms
# equal under H0.
DIST_2ARM_H1 = {"dist": "normal", "params": {"loc": [0.6, 0.4], "scale": 0.0}}
DIST_2ARM_H0 = {"dist": "normal", "params": {"loc": [0.5, 0.5], "scale": 0.0}}
DIST_3ARM_H1 = {"dist": "beta", "params": {"a": 3, "b": 3}}
DIST_3ARM_H0 = {"dist": "normal", "params": {"loc": [0.5, 0.5, 0.5], "scale": 0.0}}
T_CONSTANT_THRESHOLD = 0.5

# Per-test minimum detectable effect, applied to POWER (H1) only: a rep counts
# toward power if at least one arm or comparison has a true effect above this.
# FPR (H0) is never masked. ANOVA and the 2-arm t-test use 0 (no masking).
MIN_EFFECT = {"t-test": 0.0, "anova": 0.0, "tukey": 0.15, "t-constant": 0.05, "t-control": 0.05}

EPS_TS_LIST = [0.0]   # eps=0 is TS
EPS_GREEDY_LIST = [0.1]

BASE_SEED = 12345
MAX_WORKERS = 8   # of 11 cores, leave headroom


# ── Test-type registry ───────────────────────────────────────────────────────
#
# Each entry describes one --test option: arm count, the H1/H0 arm-mean
# distributions, the TestProcedure (used by AIT), and the from-counts
# statistic ART/Queue replay computes.

@dataclass(frozen=True)
class TestSpec:
    n_arm: int
    h1_dist: dict
    h0_dist: dict
    test_procedure_factory: callable
    stat_from_counts: callable


def _t_control_stat(x, n):
    return t_test_from_counts(x, n, arm_index=slice(1, None), control_arm=0, pooled_var=True)


def _t_constant_stat(x, n):
    return t_test_from_counts(x, n, const_thres=T_CONSTANT_THRESHOLD, pooled_var=True)


TEST_SPECS: Dict[str, TestSpec] = {
    "t-test": TestSpec(
        n_arm=2, h1_dist=DIST_2ARM_H1, h0_dist=DIST_2ARM_H0,
        test_procedure_factory=lambda: TControl(type1_error_constraint=ALPHA, test_type="two-sided"),
        stat_from_counts=_t_control_stat,
    ),
    "anova": TestSpec(
        n_arm=3, h1_dist=DIST_3ARM_H1, h0_dist=DIST_3ARM_H0,
        test_procedure_factory=lambda: ANOVA(type1_error_constraint=ALPHA),
        stat_from_counts=anova_from_counts,
    ),
    "tukey": TestSpec(
        n_arm=3, h1_dist=DIST_3ARM_H1, h0_dist=DIST_3ARM_H0,
        test_procedure_factory=lambda: Tukey(type1_error_constraint=ALPHA, test_type="distinct-best-arm",
                                             family_wise_error_control=True),
        stat_from_counts=tukey_from_counts,
    ),
    "t-constant": TestSpec(
        n_arm=3, h1_dist=DIST_3ARM_H1, h0_dist=DIST_3ARM_H0,
        test_procedure_factory=lambda: TConstant(
            type1_error_constraint=ALPHA, test_type="one-sided", constant_threshold=T_CONSTANT_THRESHOLD),
        stat_from_counts=_t_constant_stat,
    ),
    "t-control": TestSpec(
        n_arm=3, h1_dist=DIST_3ARM_H1, h0_dist=DIST_3ARM_H0,
        test_procedure_factory=lambda: TControl(type1_error_constraint=ALPHA, test_type="two-sided"),
        stat_from_counts=_t_control_stat,
    ),
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_null_crit_and_reject(test_procedure, null_stat: np.ndarray, obs_stat: np.ndarray) -> np.ndarray:
    """ART/Queue/AIT calibration for one outer rep. Given the replayed null
    distribution and the observed statistic, apply each TestProcedure's own
    direction/abs/family-wise transform to decide reject/no-reject.

    Returns a 1-D per-cell reject array: length 1 for single-decision tests
    (t-test, ANOVA, Tukey distinct-best-arm); one entry per arm/comparison for
    T-Constant / T-Control (so power can be averaged over meaningful cells,
    matching the framework's per-arm compute_power).
    """
    if isinstance(test_procedure, (ANOVA, Tukey)):
        n_cells = 1
    else:
        n_cells = np.asarray(obs_stat).reshape(-1).size

    # A deterministic policy (UCB) makes ART's fixed-stream replay reproduce the
    # observed run exactly, collapsing the null to a point. The randomized test
    # at the boundary then rejects with probability alpha, so power and FPR both
    # equal alpha.
    spread = np.nanmax(null_stat, axis=0) - np.nanmin(null_stat, axis=0)
    if np.nanmax(spread) < 1e-12:
        return np.full(n_cells, float(test_procedure.type1_error_constraint))

    if isinstance(test_procedure, ANOVA):
        # anova_from_counts returns a p-value directly (smaller = more extreme).
        crit_p = np.quantile(null_stat, test_procedure.type1_error_constraint)
        return np.array([float(np.squeeze(obs_stat) < crit_p)])

    if isinstance(test_procedure, Tukey):
        # Distinct-best-arm statistic = the best arm's min margin (it must beat
        # every other arm, so the binding comparison is its weakest). Null and
        # observed use the same statistic (min) so the test calibrates to the
        # nominal FPR.
        null_reduced = test_procedure._filter_stats(null_stat.copy()[np.newaxis], method="min")[0]
        obs_reduced = test_procedure._filter_stats(obs_stat.copy()[np.newaxis, np.newaxis], method="min")[0, 0]
        crit = np.nanquantile(null_reduced, 1 - test_procedure.type1_error_constraint)
        return np.array([float(np.squeeze(obs_reduced) > crit)])

    # T-Control / T-Constant: abs() if two-sided, one crit pooled across cells;
    # return the per-cell reject vector (averaging happens later, over the cells
    # that meet min_effect).
    two_sided = getattr(test_procedure, "test_type", None) == "two-sided"
    null_t = np.abs(null_stat) if two_sided else null_stat
    obs_t = np.abs(obs_stat) if two_sided else obs_stat
    crit = np.quantile(null_t.reshape(-1), 1 - test_procedure.type1_error_constraint)
    return (obs_t.reshape(-1) > crit).astype(float)


def cell_mask(test_name: str, arm_means: np.ndarray) -> np.ndarray:
    """Per-cell boolean mask (n_rep, n_cells) for POWER: a cell counts if its
    own true effect exceeds the test's min_effect. Matches the per-cell layout
    of get_null_crit_and_reject. Never applied to FPR."""
    me = MIN_EFFECT[test_name]
    n_rep = arm_means.shape[0]
    if test_name == "tukey":
        top2 = np.sort(arm_means, axis=1)[:, ::-1][:, :2]
        return ((top2[:, 0] - top2[:, 1]) > me)[:, np.newaxis]
    if test_name == "t-constant":
        return (arm_means - T_CONSTANT_THRESHOLD) > me            # one-sided, per arm
    if test_name == "t-control":
        return np.abs(arm_means[:, 1:] - arm_means[:, [0]]) > me  # per comparison
    # anova, t-test: single cell, no min_effect
    return np.ones((n_rep, 1), dtype=bool)


# ── Policy spec ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicySpec:
    name: str  # "ucb1", "eps_ts", "eps_greedy"
    eps: Optional[float] = None

    def key(self) -> str:
        # paper-style labels
        if self.name == "ucb1":
            return "UCB"
        if self.name == "eps_greedy":
            return rf"eps-greedy({self.eps})"
        if self.name == "eps_ts":
            if float(self.eps) == 0.0:
                return "TS"
            return rf"eps-TS({self.eps})"
        if self.eps is None:
            return self.name
        return f"{self.name}(eps={self.eps})"


# ── Vectorized batch policy: one simulator for generation, replay, and AIT ───

# UCB exploration constant. UCB index = mean_k + sqrt(UCB_C * log(t) / N_k),
# t = total pulls so far. Matches the framework UCB default.
UCB_C = 2.0


def _select_arm_batch(policy: PolicySpec, x: np.ndarray, n: np.ndarray,
                       n_inner: int, n_arm: int, rng: np.random.Generator) -> np.ndarray:
    """x, n: shape (n_inner, n_arm). Returns arm indices, shape (n_inner,).

    Ties are broken deterministically (lowest arm index) so UCB, which is a
    deterministic algorithm, carries no hidden randomness. Its exploration is
    driven only by the UCB index, not by tie-breaking."""
    if policy.name == "ucb1":
        total = np.maximum(np.sum(n, axis=1, keepdims=True), 1)  # pulls so far, per rep
        mean = x / np.maximum(n, 1)
        bonus = np.sqrt(UCB_C * np.log(total) / np.maximum(n, 1))
        return np.argmax(mean + bonus, axis=1)

    if policy.name == "eps_greedy":
        eps = float(policy.eps)
        explore = rng.random(n_inner) < eps
        greedy_arm = np.argmax(x / np.maximum(n, 1), axis=1)
        random_arm = rng.integers(0, n_arm, size=n_inner)
        return np.where(explore, random_arm, greedy_arm)

    if policy.name == "eps_ts":
        eps = float(policy.eps)
        explore = rng.random(n_inner) < eps
        samples = rng.beta(1 + x, 1 + (n - x))
        ts_arm = np.argmax(samples, axis=1)
        random_arm = rng.integers(0, n_arm, size=n_inner)
        return np.where(explore, random_arm, ts_arm)

    raise ValueError(f"Unknown policy: {policy}")


def draw_arm_means(dist_spec: dict, n_rep: int, n_arm: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Per-rep Bernoulli arm means. Beta draws one triple per rep; normal with
    scale 0 gives the fixed loc vector to every rep."""
    dist = dist_spec["dist"]
    params = dist_spec["params"]
    if dist == "beta":
        return rng.beta(params["a"], params["b"], size=(n_rep, n_arm))
    if dist == "normal":
        loc = np.asarray(params["loc"], dtype=float)
        scale = params.get("scale", 0.0)
        if scale == 0.0:
            return np.tile(loc, (n_rep, 1))
        return rng.normal(loc, scale, size=(n_rep, n_arm))
    raise ValueError(dist)


def run_batch(policy: PolicySpec, arm_means: np.ndarray, n_arm: int,
               rng: np.random.Generator, record_stream: bool = False):
    """Online policy simulation, vectorized over the batch. arm_means is the
    per-rep, per-arm Bernoulli success probability. Returns terminal counts
    x, n (shape (n_batch, n_arm)), the per-timestep reward stream (shape
    (n_batch, HORIZON) when record_stream, else None), and the combined mean.
    Burn-in pulls each arm BURN_IN_PER_ARM times before the adaptive phase."""
    n_batch = arm_means.shape[0]
    x = np.zeros((n_batch, n_arm), dtype=int)
    n = np.zeros((n_batch, n_arm), dtype=int)
    reward_stream = np.zeros((n_batch, HORIZON), dtype=int) if record_stream else None
    idx = np.arange(n_batch)
    t = 0

    for _k in range(BURN_IN_PER_ARM):
        for arm in range(n_arm):
            r = (rng.random(n_batch) < arm_means[:, arm]).astype(int)
            x[:, arm] += r
            n[:, arm] += 1
            if record_stream:
                reward_stream[:, t] = r
            t += 1

    while t < HORIZON:
        arm = _select_arm_batch(policy, x, n, n_batch, n_arm, rng)
        r = (rng.random(n_batch) < arm_means[idx, arm]).astype(int)
        x[idx, arm] += r
        n[idx, arm] += 1
        if record_stream:
            reward_stream[:, t] = r
        t += 1

    combined_mean = x.sum(axis=1) / n.sum(axis=1)
    return x, n, reward_stream, combined_mean


def run_batch_replay_fixed(policy: PolicySpec, reward_stream_batch: np.ndarray,
                            n_inner: int, n_arm: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized replay. reward_stream_batch: shape (n_inner, HORIZON), each
    inner rep's own reward stream (ART tiles one stream n_inner times; Queue
    passes n_inner independent shuffles).
    """
    x = np.zeros((n_inner, n_arm), dtype=int)
    n = np.zeros((n_inner, n_arm), dtype=int)
    idx = np.arange(n_inner)
    t = 0

    for _k in range(BURN_IN_PER_ARM):
        for arm in range(n_arm):
            r = reward_stream_batch[:, t]
            x[:, arm] += r
            n[:, arm] += 1
            t += 1

    while t < HORIZON:
        arm = _select_arm_batch(policy, x, n, n_inner, n_arm, rng)
        r = reward_stream_batch[:, t]
        x[idx, arm] += r
        n[idx, arm] += 1
        t += 1

    return x, n


# ── ART / Queue calibration (per-outer-rep) ─────────────────────────────────

def calibrate_art_queue_for_one_outer_run(
    policy: PolicySpec,
    test_procedure,
    stat_from_counts,
    x_outer: np.ndarray,
    n_outer: np.ndarray,
    reward_stream: np.ndarray,
    n_arm: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (reject_art, reject_q) per-cell arrays for this outer rep. Both
    depend on the rep's own reward stream, so each needs its own inner sim."""
    obs_stat = stat_from_counts(x_outer, n_outer)
    reward_arr = np.asarray(reward_stream)

    # ART null: replay the fixed reward stream in its observed order.
    rngB = np.random.default_rng(rng.integers(0, 2**63 - 1))
    art_stream_batch = np.tile(reward_arr, (N_REP_0, 1))
    x_art, n_art = run_batch_replay_fixed(policy, art_stream_batch, N_REP_0, n_arm, rngB)
    null_art = stat_from_counts(x_art, n_art)
    reject_art = get_null_crit_and_reject(test_procedure, null_art, obs_stat)

    # Queue null: reshuffle the observed reward pool per inner rep.
    rngC = np.random.default_rng(rng.integers(0, 2**63 - 1))
    keys = rngC.random((N_REP_0, HORIZON))
    order = np.argsort(keys, axis=1)
    q_stream_batch = np.take_along_axis(np.tile(reward_arr, (N_REP_0, 1)), order, axis=1)
    x_q, n_q = run_batch_replay_fixed(policy, q_stream_batch, N_REP_0, n_arm, rngC)
    null_q = stat_from_counts(x_q, n_q)
    reject_q = get_null_crit_and_reject(test_procedure, null_q, obs_stat)

    return reject_art, reject_q


def _one_outer_rep_art_queue(job):
    """job = (policy, test_procedure, stat_from_counts, rep_id, x_outer, n_outer, reward_stream, n_arm, seed)."""
    policy, test_procedure, stat_from_counts, rep_id, x_outer, n_outer, reward_stream, n_arm, seed = job
    rng = np.random.default_rng(seed)
    reject_art, reject_q = calibrate_art_queue_for_one_outer_run(
        policy=policy, test_procedure=test_procedure, stat_from_counts=stat_from_counts,
        x_outer=x_outer, n_outer=n_outer, reward_stream=reward_stream, n_arm=n_arm, rng=rng,
    )
    return rep_id, reject_art, reject_q


def _stable_pol_key(policy: PolicySpec) -> int:
    if policy.name == "ucb1":
        return 111_111
    if policy.name == "eps_greedy":
        return 300_000 + int(round(1000 * float(policy.eps)))
    if policy.name == "eps_ts":
        return 200_000 + int(round(1000 * float(policy.eps)))
    raise ValueError(policy)


# ── AIT (MLE) calibration ────────────────────────────────────────────────────

AIT_CHUNK = 500  # outer reps whose null sims run in one vectorized batch


def compute_ait_perrep(policy: PolicySpec, test_procedure, stat_from_counts,
                        combined_mean: np.ndarray, obs_stat_all: np.ndarray,
                        n_arm: int, n_null: int, rng: np.random.Generator,
                        threshold: Optional[float] = None) -> np.ndarray:
    """AIT reject decision for every outer rep. Each rep's null is centered on
    its own combined-mean estimate (all arms equal to it), or on a fixed
    threshold for T-Constant. Simulate n_null fresh null runs under the same
    policy, form the null distribution of the statistic, and apply the test's
    own critical-value rule. Reps are processed in chunks so many nulls run in
    one vectorized batch.

    Returns shape (n_rep, n_cells): per-cell reject indicators.
    """
    n_rep = len(combined_mean)
    reject = [None] * n_rep
    p_per_rep = np.full(n_rep, threshold) if threshold is not None else combined_mean

    for start in range(0, n_rep, AIT_CHUNK):
        end = min(start + AIT_CHUNK, n_rep)
        block = end - start
        p_null = np.repeat(p_per_rep[start:end], n_null)  # (block*n_null,)
        arm_means = np.repeat(p_null[:, np.newaxis], n_arm, axis=1)
        x_h0, n_h0, _, _ = run_batch(policy, arm_means, n_arm, rng)
        null_stat = stat_from_counts(x_h0, n_h0)
        null_stat = null_stat.reshape(block, n_null, *null_stat.shape[1:])
        for j in range(block):
            reject[start + j] = get_null_crit_and_reject(
                test_procedure, null_stat[j], obs_stat_all[start + j])

    return np.stack(reject)  # (n_rep, n_cells)


# ── Per-policy driver ────────────────────────────────────────────────────────

def run_for_policy(test_name: str, policy: PolicySpec, which: str, n_rep: int,
                    max_workers=MAX_WORKERS):
    spec = TEST_SPECS[test_name]
    dist_spec = spec.h1_dist if which == "H1" else spec.h0_dist
    n_arm = spec.n_arm
    test_procedure = spec.test_procedure_factory()

    # Outer reps: one online run of the hand-written simulator. POWER and FPR
    # differ only in the arm-mean ground truth.
    gen_seed = BASE_SEED + 1_000_000 * _stable_pol_key(policy) + (0 if which == "H1" else 50_000_000)
    rng_gen = np.random.default_rng(gen_seed)
    arm_means = draw_arm_means(dist_spec, n_rep, n_arm, rng_gen)
    x_arr, n_arr, reward_full, combined_mean = run_batch(
        policy, arm_means, n_arm, rng_gen, record_stream=True)

    # AIT (MLE): per-rep null at each rep's combined-mean estimate.
    obs_stat_all = spec.stat_from_counts(x_arr, n_arr)
    threshold = T_CONSTANT_THRESHOLD if test_name == "t-constant" else None
    rng_ait = np.random.default_rng(gen_seed + 900_000)
    reject_ait_arr = compute_ait_perrep(
        policy, test_procedure, spec.stat_from_counts, combined_mean,
        obs_stat_all, n_arm, N_REP_0, rng_ait, threshold)

    # ART / Queue: per-outer-rep replay of that rep's own reward stream.
    aq_base = gen_seed + 500_000
    jobs = [
        (policy, test_procedure, spec.stat_from_counts, rep_id,
         x_arr[rep_id], n_arr[rep_id], reward_full[rep_id], n_arm, aq_base + rep_id)
        for rep_id in range(n_rep)
    ]

    reject_art = [None] * n_rep
    reject_q = [None] * n_rep
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one_outer_rep_art_queue, job) for job in jobs]
        for fut in as_completed(futs):
            rep_id, r_art, r_q = fut.result()
            reject_art[rep_id] = r_art
            reject_q[rep_id] = r_q

    return reject_ait_arr, np.stack(reject_art), np.stack(reject_q), arm_means


# ── Main: policy sweep + table output ───────────────────────────────────────

def build_policy_sweep() -> List[PolicySpec]:
    pols: List[PolicySpec] = []
    for eps in EPS_TS_LIST:
        pols.append(PolicySpec(name="eps_ts", eps=float(eps)))
    for eps in EPS_GREEDY_LIST:
        pols.append(PolicySpec(name="eps_greedy", eps=float(eps)))
    pols.append(PolicySpec(name="ucb1", eps=None))
    return pols


def run_all(test_name: str = "t-test", n_rep_outer: int = N_REP_OUTER, raw_path: str = None):
    policies = build_policy_sweep()
    rows: List[Dict[str, float]] = []
    raw = {}  # per-policy raw reject arrays + true means, for later re-analysis

    for policy in policies:
        alg = policy.key()
        print(f"\n=== test={test_name}  Policy={alg} ===")

        r_ait_H1, r_art_H1, r_q_H1, arm_means_H1 = run_for_policy(
            test_name, policy, which="H1", n_rep=n_rep_outer)
        # POWER: averaged over the cells (arms/comparisons) that meet min_effect,
        # per-cell like the framework's compute_power; also unconditional for ref.
        mask = cell_mask(test_name, arm_means_H1)  # (n_rep, n_cells)
        power_ait, power_art, power_q = (float(r[mask].mean()) for r in (r_ait_H1, r_art_H1, r_q_H1))
        unc_ait, unc_art, unc_q = (float(r.mean()) for r in (r_ait_H1, r_art_H1, r_q_H1))

        r_ait_H0, r_art_H0, r_q_H0, _ = run_for_policy(
            test_name, policy, which="H0", n_rep=n_rep_outer)
        fpr_ait, fpr_art, fpr_q = (float(r.mean()) for r in (r_ait_H0, r_art_H0, r_q_H0))

        rows.append({
            "Algorithm": alg, "ART": power_art, "Queue": power_q, "AIT": power_ait,
            "ART_unc": unc_art, "Queue_unc": unc_q, "AIT_unc": unc_ait,
            "ART_FPR": fpr_art, "Queue_FPR": fpr_q, "AIT_FPR": fpr_ait,
        })
        raw[alg] = dict(
            r_ait_H1=r_ait_H1, r_art_H1=r_art_H1, r_q_H1=r_q_H1, arm_means_H1=arm_means_H1,
            r_ait_H0=r_ait_H0, r_art_H0=r_art_H0, r_q_H0=r_q_H0, mask=mask,
        )

        print(f"power[cond] ART={power_art:.3f} Queue={power_q:.3f} AIT={power_ait:.3f} "
              f"(frac kept={mask.mean():.2f})")
        print(f"power[unc]  ART={unc_art:.3f} Queue={unc_q:.3f} AIT={unc_ait:.3f}")
        print(f"FPR         ART={fpr_art:.3f} Queue={fpr_q:.3f} AIT={fpr_ait:.3f}")

    cols = ["ART", "Queue", "AIT", "ART_unc", "Queue_unc", "AIT_unc", "ART_FPR", "Queue_FPR", "AIT_FPR"]
    df = pd.DataFrame(rows).set_index("Algorithm")[cols]
    print(f"\n=== Table ({test_name}): power[cond] / power[unc] / FPR ===\n")
    print(df)

    if raw_path:
        flat = {f"{alg}__{k}": v for alg, d in raw.items() for k, v in d.items()}
        np.savez(raw_path, **flat)
        print(f"Saved raw arrays: {raw_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=list(TEST_SPECS.keys()), default="t-test")
    parser.add_argument("--n-rep-outer", type=int, default=N_REP_OUTER)
    parser.add_argument("--n-rep-inner", type=int, default=N_REP_0)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    N_REP_0 = args.n_rep_inner
    HORIZON = args.horizon

    output = args.output or f"results/table2_ait_art_queue_{args.test}.csv"
    raw_path = output.rsplit(".", 1)[0] + "_raw.npz"
    df = run_all(test_name=args.test, n_rep_outer=args.n_rep_outer, raw_path=raw_path)
    df.to_csv(output)
    print(f"\nSaved: {output}")
