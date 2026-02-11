"""
Reproduces Table 2 (ART vs AIT for t-tests).

What this script does:
  1. Fixes a 2-armed Bernoulli bandit with horizon T=200.
  2. Sweeps adaptive policies:
       - epsilon-TS for eps in {0.0, 0.1, 0.2, 0.4, 0.8}   (eps>0 used in appendix)
       - epsilon-greedy (0.1)
       - UCB1 (Auer 2002)
  3. Computes both:
       - POWER under H1: (p1, p2) = (0.6, 0.4)
       - FPR   under H0: (p1, p2) = (0.5, 0.5)
  4. For each outer replication, calibrates the (two-sided) t-test via:
       - AIT: simulate the null using p_hat = mean reward from the outer run
              (i.e., (p_hat, p_hat)), and re-run the SAME policy to form the null
              distribution of the test statistic.
       - ART: fix the outer run's time-indexed reward stream, replay the policy's
              arm-selection, and form the null distribution of the statistic.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


# ── Configuration ────────────────────────────────────────────────────────────

# Outer Monte Carlo reps for POWER and FPR
N_REP_OUTER = 20000

# Inner Monte Carlo reps used to calibrate p-values / critical regions
N_REP_0 = 401

HORIZON = 200
ALPHA = 0.05

# Data-generating processes
P_H1 = (0.6, 0.4)  # POWER setting
P_H0 = (0.5, 0.5)  # FPR setting

# Policy sweep
EPS_TS_LIST = [0.0, 0.1, 0.2, 0.4, 0.8]   # eps=0 is TS
EPS_GREEDY_LIST = [0.1]                  # main paper only uses 0.1

BASE_SEED = 12345
MAX_WORKERS = None  # e.g. 8


# ── Helpers ─────────────────────────────────────────────────────────────────

def mc_pvalue_right_tail(null_stats: np.ndarray, obs_stat: float) -> float:
    null_stats = np.asarray(null_stats, dtype=float)
    return float(np.mean(null_stats >= obs_stat))


def _argmax_random_tie(values: np.ndarray, rng: np.random.Generator) -> int:
    values = np.asarray(values, dtype=float)
    m = np.max(values)
    idx = np.flatnonzero(values == m)
    return int(rng.choice(idx))


def t_stat_from_counts(x0: int, n0: int, x1: int, n1: int) -> float:
    """
    Two-sample pooled-proportion z/t-style statistic.
    Two-sided test implemented as right-tail on abs(t).

    Returns: abs(t)
    """
    if n0 <= 0 or n1 <= 0:
        return 0.0

    p0 = x0 / n0
    p1 = x1 / n1
    p_pool = (x0 + x1) / (n0 + n1)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1))
    if not np.isfinite(se) or se <= 0:
        return 0.0

    t = (p0 - p1) / se
    return float(abs(t))


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


# ── Bandit simulators ───────────────────────────────────────────────────────

def run_mab_generate_rewards(policy: PolicySpec, p_arm: Tuple[float, float], rng: np.random.Generator):
    """
    Fresh-reward bandit run for HORIZON steps.
    Returns:
      history: list[(arm, reward)] length HORIZON
      x: successes per arm (2,)
      n: pulls per arm (2,)
    """
    x = np.zeros(2, dtype=int)
    n = np.zeros(2, dtype=int)
    history: List[Tuple[int, int]] = []

    # classical init: play each arm once
    for arm in (0, 1):
        if len(history) >= HORIZON:
            break
        r = int(rng.binomial(1, p_arm[arm]))
        history.append((arm, r))
        x[arm] += r
        n[arm] += 1

    while len(history) < HORIZON:
        t = len(history) + 1  # 1-indexed time

        if policy.name == "ucb1":
            # Classical UCB1 (no parameter)
            mean = x / np.maximum(n, 1)
            bonus = np.sqrt(2.0 * np.log(max(t, 2)) / np.maximum(n, 1))
            score = mean + bonus
            arm = _argmax_random_tie(score, rng)

        elif policy.name == "eps_greedy":
            eps = float(policy.eps)
            if rng.random() < eps:
                arm = int(rng.integers(0, 2))
            else:
                mean = x / np.maximum(n, 1)
                arm = _argmax_random_tie(mean, rng)

        elif policy.name == "eps_ts":
            eps = float(policy.eps)
            if rng.random() < eps:
                arm = int(rng.integers(0, 2))
            else:
                # Bernoulli TS with Beta(1,1) prior
                samples = rng.beta(1 + x, 1 + (n - x))
                arm = _argmax_random_tie(samples, rng)

        else:
            raise ValueError(f"Unknown policy: {policy}")

        r = int(rng.binomial(1, p_arm[arm]))
        history.append((arm, r))
        x[arm] += r
        n[arm] += 1

    return history, x, n


def run_mab_replay_fixed_rewards(policy: PolicySpec, reward_stream: List[int], rng: np.random.Generator):
    """
    ART replay:
      at time t, reward is reward_stream[t] regardless of chosen arm.
    """
    assert len(reward_stream) >= HORIZON
    x = np.zeros(2, dtype=int)
    n = np.zeros(2, dtype=int)

    t_idx = 0

    # classical init: play each arm once (consumes stream)
    for arm in (0, 1):
        if t_idx >= HORIZON:
            break
        r = int(reward_stream[t_idx])
        t_idx += 1
        x[arm] += r
        n[arm] += 1

    while t_idx < HORIZON:
        t = t_idx + 1

        if policy.name == "ucb1":
            mean = x / np.maximum(n, 1)
            bonus = np.sqrt(2.0 * np.log(max(t, 2)) / np.maximum(n, 1))
            score = mean + bonus
            arm = _argmax_random_tie(score, rng)

        elif policy.name == "eps_greedy":
            eps = float(policy.eps)
            if rng.random() < eps:
                arm = int(rng.integers(0, 2))
            else:
                mean = x / np.maximum(n, 1)
                arm = _argmax_random_tie(mean, rng)

        elif policy.name == "eps_ts":
            eps = float(policy.eps)
            if rng.random() < eps:
                arm = int(rng.integers(0, 2))
            else:
                samples = rng.beta(1 + x, 1 + (n - x))
                arm = _argmax_random_tie(samples, rng)

        else:
            raise ValueError(f"Unknown policy: {policy}")

        r = int(reward_stream[t_idx])
        t_idx += 1
        x[arm] += r
        n[arm] += 1

    return x, n


# ── Inner calibration for AIT / ART (per outer run) ─────────────────────────

def calibrate_pvalues_for_one_outer_run(
    policy: PolicySpec,
    x_outer: np.ndarray,
    n_outer: np.ndarray,
    reward_stream: List[int],
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Returns (p_ait, p_art) for the observed outer statistic.
    Null for both methods is defined via p_hat estimated from the OUTER RUN ONLY.
    """
    t_obs = t_stat_from_counts(int(x_outer[0]), int(n_outer[0]), int(x_outer[1]), int(n_outer[1]))

    # Estimate null mean from outer run only (algorithm does not know true p)
    p_hat = float(np.mean(reward_stream))
    p_hat = float(np.clip(p_hat, 1e-6, 1 - 1e-6))

    # ---- AIT null: re-simulate bandit under (p_hat, p_hat) ----
    t_null = np.empty(N_REP_0, dtype=float)
    rngA = np.random.default_rng(rng.integers(0, 2**63 - 1))
    for ii in range(N_REP_0):
        _, x_sim, n_sim = run_mab_generate_rewards(policy, (p_hat, p_hat), rngA)
        t_null[ii] = t_stat_from_counts(int(x_sim[0]), int(n_sim[0]), int(x_sim[1]), int(n_sim[1]))
    p_ait = mc_pvalue_right_tail(t_null, t_obs)

    # ---- ART null: replay fixed reward stream ----
    t_art_null = np.empty(N_REP_0, dtype=float)
    rngB = np.random.default_rng(rng.integers(0, 2**63 - 1))
    for ii in range(N_REP_0):
        xB, nB = run_mab_replay_fixed_rewards(policy, reward_stream, rngB)
        t_art_null[ii] = t_stat_from_counts(int(xB[0]), int(nB[0]), int(xB[1]), int(nB[1]))
    p_art = mc_pvalue_right_tail(t_art_null, t_obs)

    return float(p_ait), float(p_art)


# ── Parallel worker ─────────────────────────────────────────────────────────

def _stable_pol_key(policy: PolicySpec) -> int:
    if policy.name == "ucb1":
        return 111_111
    if policy.name == "eps_greedy":
        return 300_000 + int(round(1000 * float(policy.eps)))
    if policy.name == "eps_ts":
        return 200_000 + int(round(1000 * float(policy.eps)))
    raise ValueError(policy)


def one_outer_rep(job):
    """
    job = (policy, rep_id, which)
      which in {"H1", "H0"}
    Returns:
      rep_id, p_ait, p_art
    """
    policy, rep_id, which = job
    pol_key = _stable_pol_key(policy)

    rng = np.random.default_rng(
        BASE_SEED + 1_000_000 * pol_key + rep_id + (0 if which == "H1" else 50_000_000)
    )

    p_arm = P_H1 if which == "H1" else P_H0

    history, x, n = run_mab_generate_rewards(policy, p_arm, rng)
    reward_stream = [r for _, r in history]

    p_ait, p_art = calibrate_pvalues_for_one_outer_run(
        policy=policy,
        x_outer=x,
        n_outer=n,
        reward_stream=reward_stream,
        rng=rng,
    )

    return rep_id, p_ait, p_art


def run_for_policy(policy: PolicySpec, which: str, max_workers=MAX_WORKERS):
    n_rep = N_REP_OUTER if which == "H1" else N_REP_OUTER

    p_ait = np.empty(n_rep, dtype=float)
    p_art = np.empty(n_rep, dtype=float)

    jobs = [(policy, rep_id, which) for rep_id in range(n_rep)]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(one_outer_rep, job) for job in jobs]
        for fut in as_completed(futs):
            rep_id, pait, part = fut.result()
            p_ait[rep_id] = pait
            p_art[rep_id] = part

    return p_ait, p_art


# ── Main: policy sweep + table output ───────────────────────────────────────

def build_policy_sweep() -> List[PolicySpec]:
    pols: List[PolicySpec] = []

    # epsilon-TS sweep (eps=0 is TS; eps>0 are appendix)
    for eps in EPS_TS_LIST:
        pols.append(PolicySpec(name="eps_ts", eps=float(eps)))

    # epsilon-greedy (0.1)
    for eps in EPS_GREEDY_LIST:
        pols.append(PolicySpec(name="eps_greedy", eps=float(eps)))

    # UCB1
    pols.append(PolicySpec(name="ucb1", eps=None))

    return pols


def run_all():
    policies = build_policy_sweep()
    rows: List[Dict[str, float]] = []

    for policy in policies:
        alg = policy.key()
        print(f"\n=== Policy={alg} ===")

        # POWER under H1
        p_ait_H1, p_art_H1 = run_for_policy(policy, which="H1")
        power_ait = float(np.mean(p_ait_H1 < ALPHA))
        power_art = float(np.mean(p_art_H1 < ALPHA))

        # FPR under H0
        p_ait_H0, p_art_H0 = run_for_policy(policy, which="H0")
        fpr_ait = float(np.mean(p_ait_H0 < ALPHA))
        fpr_art = float(np.mean(p_art_H0 < ALPHA))

        rows.append(
            {
                "Algorithm": alg,
                "ART": power_art,
                "AIT": power_ait,
                "ART_FPR": fpr_art,
                "AIT_FPR": fpr_ait,
            }
        )

        print(f"power (ART) = {power_art:.3f} | power (AIT) = {power_ait:.3f}")
        print(f"FPR   (ART) = {fpr_art:.3f} | FPR   (AIT) = {fpr_ait:.3f}")

    df = pd.DataFrame(rows).set_index("Algorithm")[["ART", "AIT", "ART_FPR", "AIT_FPR"]]
    print("\n=== Table: Power / FPR (rows=Algorithm) ===\n")
    print(df)

    return df


if __name__ == "__main__":
    run_all()
