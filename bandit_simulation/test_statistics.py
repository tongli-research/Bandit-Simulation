"""
Test-statistic functions with a uniform calling interface.

Every function's standard input is the raw simulation history
(action_hist, reward_hist, reward2_hist, ad) -- the same arrays SimResult
itself is constructed from -- not pre-derived quantities like arm_means or
combined_vars. Test-specific extras (arm indices, thresholds, dist) are
plain keyword args, so all functions share one calling shape:

    stat = test_fn(action_hist, reward_hist, reward2_hist, ad, horizon=..., **extra)

This lets SimResult collapse its 8 separate test methods into a single
dispatcher (TEST_REGISTRY below) and lets other code -- e.g.
ait_art_comparison.py's ART/Queue inner replay loop, which works with raw
count arrays for performance and previously re-derived this math by hand --
call the identical statistic without constructing a SimResult.

Design decided with Tong, 2026-07-25:
  - t_control's permutation_test=True branch is commented off (not migrated
    this pass -- see _t_control_permutation_stat below).
  - The old wald_test, t_control (non-permutation branch), t_constant,
    wald_test_normal, and t_test are all "compare arm mean(s) against a
    reference, with pooled or unpooled variance" -- merged into one
    t_test_stat, switched via const_thres (vs. a fixed value) or
    control_arm (vs. another arm) and pooled_var.
"""

import numpy as np
from scipy.stats import bernoulli, f


def _derive_arm_stats(action_hist, reward_hist, reward2_hist, ad):
    """Shared first step for every test function below -- mirrors the
    cumulative-stats computation in SimResult.__init__/arm_vars/combined_vars
    exactly. Returns arm_means, arm_vars, arm_counts, combined_means,
    combined_vars, total_counts."""
    horizon_axis = ad.arr_axis['horizon']
    arm_axis = ad.arr_axis['n_arm']

    with np.errstate(divide='ignore', invalid='ignore'):
        arm_counts = np.cumsum(action_hist, axis=horizon_axis)
        total_counts = np.sum(arm_counts, axis=arm_axis, keepdims=True)

        arm_means = np.cumsum(reward_hist, axis=horizon_axis) / arm_counts
        combined_means = np.cumsum(np.sum(reward_hist, axis=arm_axis, keepdims=True),
                                    axis=horizon_axis) / total_counts

        arm_square_means = np.cumsum(reward2_hist, axis=horizon_axis) / arm_counts
        arm_vars = (arm_square_means - arm_means ** 2) * (1 / (arm_counts - 1))

        combined_square_means = np.cumsum(np.sum(reward2_hist, axis=arm_axis, keepdims=True),
                                           axis=horizon_axis) / total_counts
        combined_vars = combined_square_means - combined_means ** 2

    return arm_means, arm_vars, arm_counts, combined_means, combined_vars, total_counts


def _as_arm_slice(index):
    """int -> length-1 slice (keeps the arm axis for broadcasting); slice -> unchanged."""
    return slice(index, index + 1) if isinstance(index, int) else index


def t_test_stat(action_hist, reward_hist, reward2_hist, ad, horizon=slice(-1, None),
                 arm_index=slice(None), pooled_var=True, const_thres=None, control_arm=None):
    """Unifies the old wald_test, t_control's non-permutation branch,
    t_constant, wald_test_normal, and t_test: compares arm_index's mean(s)
    against either a fixed constant (const_thres) or another arm's mean
    (control_arm), with a choice of pooled or unpooled variance.

    Exactly one of const_thres / control_arm should be set:
      - const_thres:  one-sample test, arm_index vs a fixed value
                       (old t_constant [pooled_var=True] / t_test [False])
      - control_arm:  two-sample test, arm_index vs another arm
                       (old wald_test/t_control [pooled_var=True] /
                       wald_test_normal [pooled_var=False])
    arm_index / control_arm: int (single arm) or slice (e.g. slice(1, None)
    for t_control's "all non-control arms").
    """
    arm_means, arm_vars, arm_counts, _, combined_vars, _ = _derive_arm_stats(
        action_hist, reward_hist, reward2_hist, ad)
    arm_slice = ad.slicing(n_arm=_as_arm_slice(arm_index), horizon=horizon)

    with np.errstate(divide='ignore', invalid='ignore'):
        if const_thres is not None:
            ref = const_thres
            var = combined_vars / arm_counts[arm_slice] if pooled_var else arm_vars[arm_slice]
        else:
            control_slice = ad.slicing(n_arm=_as_arm_slice(control_arm), horizon=horizon)
            ref = arm_means[control_slice]
            if pooled_var:
                var = combined_vars * (1 / arm_counts[arm_slice] + 1 / arm_counts[control_slice])
            else:
                var = arm_vars[arm_slice] + arm_vars[control_slice]

        return (arm_means[arm_slice] - ref) / np.sqrt(var)


# t_control's permutation_test=True branch (hypergeometric permutation p-value,
# ignores the horizon argument in the original code) -- commented off per
# Tong 2026-07-25, not part of this pass's unification.
#
# def _t_control_permutation_stat(action_hist, reward_hist, reward2_hist, ad,
#                                  permutation_rep=100, n_iter=10):
#     arm_means, _, arm_counts, _, _, _ = _derive_arm_stats(action_hist, reward_hist, reward2_hist, ad)
#     arm_cum_reward = arm_counts * arm_means
#     n_good = (arm_cum_reward[:, :, 0:1] + arm_cum_reward[:, :, 1:]).astype(int)
#     n_bad = (arm_counts[:, :, 0:1] + arm_counts[:, :, 1:] - n_good).astype(int)
#     count = np.zeros_like(arm_cum_reward[..., 1:], dtype=float)
#     for _ in range(n_iter):
#         permutation_samples = np.random.hypergeometric(
#             ngood=n_good, nbad=n_bad, nsample=arm_counts[:, :, 0:1],
#             size=(permutation_rep,) + n_good.shape
#         )
#         count += np.mean(permutation_samples > arm_cum_reward[np.newaxis, :, :, 0:1], axis=0)
#     return count / n_iter


def anova_stat(action_hist, reward_hist, reward2_hist, ad, horizon=slice(-1, None)):
    """SimResult.anova(): one-way F-test p-value. Computed at every horizon
    step, THEN sliced -- SSB/SSW need the full per-step arm aggregation, so
    (unlike the pooled two-sample tests) slicing can't happen beforehand."""
    arm_means, arm_vars, arm_counts, combined_means, _, total_counts = _derive_arm_stats(
        action_hist, reward_hist, reward2_hist, ad)
    arm_axis = ad.arr_axis['n_arm']
    K = action_hist.shape[arm_axis]

    with np.errstate(divide='ignore', invalid='ignore'):
        variances = arm_vars * (arm_counts - 1)

        ssb = np.sum(arm_counts * (arm_means - combined_means) ** 2, axis=arm_axis, keepdims=True)
        ssw = np.sum((arm_counts - 1) * variances, axis=arm_axis, keepdims=True)

        msb = ssb / (K - 1)
        msw = ssw / (total_counts - K)
        F_stat = msb / msw

        p_value = 1 - f.cdf(F_stat, K - 1, total_counts - K)

    # negative-direction p-value, so all tests share a right-side critical region
    return p_value[ad.slicing(horizon=horizon)]


def tukey_stat(action_hist, reward_hist, reward2_hist, ad, horizon=slice(-1, None)):
    """SimResult.tukey(): pairwise Tukey HSD statistic."""
    arm_means, _, arm_counts, _, combined_vars, _ = _derive_arm_stats(
        action_hist, reward_hist, reward2_hist, ad)

    group_means = arm_means[:, horizon, :]
    pooled_var_arr = combined_vars[:, horizon, :]
    arm_weights = 1 / (arm_counts[:, horizon, :] - 1)
    pooled_std = np.sqrt(pooled_var_arr)[..., :, np.newaxis]

    mean_diffs = group_means[..., :, np.newaxis] - group_means[..., np.newaxis, :]
    sum_arm_weights = arm_weights[..., :, np.newaxis] + arm_weights[..., np.newaxis, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        # multiplied by sqrt(2): https://en.wikipedia.org/wiki/Tukey%27s_range_test
        return mean_diffs / (pooled_std * np.sqrt(sum_arm_weights)) * np.sqrt(2)


def lrt_stat(action_hist, reward_hist, reward2_hist, ad, horizon=slice(-1, None), dist=bernoulli):
    """SimResult.LRT(): likelihood-ratio statistic, Bernoulli rewards.
    reward_hist/action_hist are used FULL (unsliced) in the likelihood sums --
    only the hypothesized parameter values (p_hat_h0/p_hat_h1) are
    horizon-sliced. Preserved from the original method exactly."""
    arm_means, _, _, combined_means, _, _ = _derive_arm_stats(action_hist, reward_hist, reward2_hist, ad)
    arm_axis = ad.arr_axis['n_arm']
    horizon_axis = ad.arr_axis['horizon']

    sli = ad.slicing(horizon=horizon)
    p_hat_h0 = combined_means[sli[0:-1]]  # assumes arm is the last dim
    p_hat_h1 = arm_means[sli]

    L0 = np.sum(np.log(dist.pmf(np.sum(reward_hist, axis=arm_axis), p_hat_h0)), axis=-1)
    L1 = np.sum(np.log(dist.pmf(reward_hist, p_hat_h1)) * action_hist, axis=(arm_axis, horizon_axis))
    return -2 * (L0 - L1)


# ── "From counts" variants ───────────────────────────────────────────────────
#
# Same formulas as t_test_stat/anova_stat/tukey_stat, but for callers that
# already have terminal (successes, pulls) counts and don't need/want
# SimResult-style cumulative history or horizon slicing -- e.g.
# ait_art_comparison.py's ART/Queue inner replay loop, which produces
# terminal counts directly. Bernoulli-only (assumes reward^2 == reward).
# x, n: shape (..., K), arm axis last, no horizon axis.

def _derive_arm_stats_from_counts(x, n):
    """Same six quantities as _derive_arm_stats, computed from terminal
    Bernoulli counts instead of a full action_hist/reward_hist trajectory."""
    with np.errstate(divide='ignore', invalid='ignore'):
        arm_means = x / n
        arm_vars = arm_means * (1 - arm_means) / (n - 1)
        total_counts = np.sum(n, axis=-1, keepdims=True)
        combined_means = np.sum(x, axis=-1, keepdims=True) / total_counts
        combined_vars = combined_means * (1 - combined_means)
    return arm_means, arm_vars, n, combined_means, combined_vars, total_counts


def t_test_from_counts(x, n, arm_index=slice(None), pooled_var=True, const_thres=None, control_arm=None):
    """t_test_stat's formula, from terminal counts.

    arm_index / control_arm are converted via _as_arm_slice (int -> length-1
    slice) so the arm axis is never dropped -- combined_vars/arm_vars keep a
    trailing keepdims=True axis (from _derive_arm_stats_from_counts), and an
    int index would drop that axis on arm_means/arm_counts only, silently
    broadcasting the two into the wrong shape instead of erroring.
    """
    arm_means, arm_vars, arm_counts, _, combined_vars, _ = _derive_arm_stats_from_counts(x, n)
    arm_index = _as_arm_slice(arm_index)

    with np.errstate(divide='ignore', invalid='ignore'):
        if const_thres is not None:
            ref = const_thres
            var = combined_vars / arm_counts[..., arm_index] if pooled_var else arm_vars[..., arm_index]
        else:
            control_arm = _as_arm_slice(control_arm)
            ref = arm_means[..., control_arm]
            if pooled_var:
                var = combined_vars * (1 / arm_counts[..., arm_index] + 1 / arm_counts[..., control_arm])
            else:
                var = arm_vars[..., arm_index] + arm_vars[..., control_arm]

        return (arm_means[..., arm_index] - ref) / np.sqrt(var)


def anova_from_counts(x, n):
    """anova_stat's formula, from terminal counts."""
    arm_means, arm_vars, arm_counts, combined_means, _, total_counts = _derive_arm_stats_from_counts(x, n)
    K = x.shape[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        variances = arm_vars * (arm_counts - 1)
        ssb = np.sum(arm_counts * (arm_means - combined_means) ** 2, axis=-1, keepdims=True)
        ssw = np.sum((arm_counts - 1) * variances, axis=-1, keepdims=True)
        F_stat = (ssb / (K - 1)) / (ssw / (total_counts - K))
        return 1 - f.cdf(F_stat, K - 1, total_counts - K)


def tukey_from_counts(x, n):
    """tukey_stat's formula, from terminal counts."""
    arm_means, _, arm_counts, _, combined_vars, _ = _derive_arm_stats_from_counts(x, n)

    pooled_std = np.sqrt(combined_vars)[..., np.newaxis]
    arm_weights = 1 / (arm_counts - 1)
    mean_diffs = arm_means[..., :, np.newaxis] - arm_means[..., np.newaxis, :]
    sum_arm_weights = arm_weights[..., :, np.newaxis] + arm_weights[..., np.newaxis, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        return mean_diffs / (pooled_std * np.sqrt(sum_arm_weights)) * np.sqrt(2)


TEST_REGISTRY = {
    't-test': t_test_stat,
    'anova': anova_stat,
    'tukey': tukey_stat,
    'lrt': lrt_stat,
}
