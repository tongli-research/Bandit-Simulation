from abc import ABC, abstractmethod

import numpy as np


class BanditAlgorithm(ABC):
    def __init__(self, algo_para):
        self.algo_para = algo_para
        self.__name__ = f"{self.__class__.__name__}" # ({algo_para})

    @abstractmethod
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        pass

class EpsGreedy(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_arm = sim_config.n_arm


        samples = (
            np.sum(reward_hist, axis=ad.arr_axis['horizon'], keepdims=True)
            / np.sum(action_hist, axis=ad.arr_axis['horizon'], keepdims=True)
        )

        ur_size = np.delete(np.array(samples.shape), ad.arr_axis['n_arm'])
        ur_ind = ad.tile(arr=(np.random.binomial(n=1, p=self.algo_para, size=ur_size) == 1),
                         axis_name='n_arm')

        actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(1, np.ones(n_arm) / n_arm, size=ur_size)[ur_ind]
        return actions



class UCB(BanditAlgorithm):
    """Upper Confidence Bound (UCB1) algorithm.

    algo_para: exploration constant c (default 2).
        UCB index = mean_k + sqrt(c * log(t) / N_k)
        where t = total pulls so far, N_k = pulls of arm k.

    Does not require a Bayesian posterior model.
    Arms with zero pulls get UCB index = +inf (pulled first in round-robin order).
    """
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        c = self.algo_para if self.algo_para is not None else 2.0

        # Arm counts and means from history: sum over horizon axis
        N_k = np.sum(action_hist, axis=ad.arr_axis['horizon'], keepdims=True)  # (n_rep, 1, n_arm)
        reward_sum = np.sum(reward_hist, axis=ad.arr_axis['horizon'], keepdims=True)

        # Total pulls so far
        t_total = np.sum(N_k, axis=ad.arr_axis['n_arm'], keepdims=True)  # (n_rep, 1, 1)
        t_total = np.maximum(t_total, 1)  # avoid log(0)

        # Arm means (0 where unpulled)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_k = np.where(N_k > 0, reward_sum / N_k, 0.0)

        # UCB index: +inf for unpulled arms to force initial exploration
        with np.errstate(divide='ignore', invalid='ignore'):
            bonus = np.where(N_k > 0, np.sqrt(c * np.log(t_total) / N_k), np.inf)

        ucb_index = mean_k + bonus

        # Tiny noise for tie-breaking (especially among +inf arms → round-robin-like)
        ucb_index += np.random.uniform(0, 1e-10, size=ucb_index.shape)

        # Argmax → one-hot, replicate for batch_size
        actions = (ucb_index == np.max(ucb_index, axis=ad.arr_axis['n_arm'], keepdims=True))

        # Replicate the same action for all batch elements (UCB is deterministic given history)
        if batch_size > 1:
            actions = np.repeat(actions, batch_size, axis=ad.arr_axis['horizon'])

        return actions


class FixedAllocation(BanditAlgorithm):
    """Fixed allocation: each arm is sampled with a fixed probability.

    algo_para: list/array of length n_arm, sums to 1.
               e.g. [0.45, 0.45, 0.10] allocates 10% to arm 2.
    """
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_rep = action_hist.shape[ad.arr_axis['n_rep']]
        n_arm = sim_config.n_arm
        probs = np.asarray(self.algo_para, dtype=float)
        choices = np.random.choice(n_arm, size=(n_rep, batch_size), p=probs)
        actions = np.eye(n_arm, dtype=int)[choices]  # (n_rep, batch_size, n_arm)
        return actions


class RoundRobin(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_arm = sim_config.n_arm
        shape_list = list(action_hist.shape)
        time_step = shape_list[ad.arr_axis['horizon']]
        shape_list[ad.arr_axis['horizon']] = batch_size

        actions = np.zeros(shape_list)
        slice_list = [slice(None)] * len(shape_list)
        slice_list[ad.arr_axis['horizon']] = np.arange(batch_size)
        slice_list[ad.arr_axis['n_arm']] = (np.arange(batch_size) + time_step) % n_arm
        actions[tuple(slice_list)] = 1
        return actions

class TSProbClip(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist,
                      batch_size=1, approx_rep=101):
        ad = sim_config.ad
        n_arm = sim_config.n_arm
        bayes_model = sim_config.bayes_model

        if batch_size > approx_rep:
            approx_rep = batch_size
        min_prob = self.algo_para / n_arm
        uniform_prob = 1.0 / n_arm
        n_rep = action_hist.shape[ad.arr_axis['n_rep']]

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=approx_rep)['mean'],
                              source=0,
                              destination=ad.arr_axis['horizon'])
        ap_actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        ap_estimate = np.mean(ap_actions, axis=ad.arr_axis['horizon'], keepdims=True)

        min_est_prob = np.min(ap_estimate, axis=ad.arr_axis['n_arm'], keepdims=True)
        min_est_prob[min_est_prob == uniform_prob] = uniform_prob - 0.000001

        x = (min_prob - uniform_prob) / (min_est_prob - uniform_prob)
        x = np.clip(x, 0.0, 1.0)

        ts_actions = ap_actions[:, :batch_size, :]

        ur_indices = np.random.randint(0, n_arm, size=(n_rep, batch_size))
        ur_actions = np.eye(n_arm)[ur_indices].astype(bool)
        mix_mask = (np.random.rand(n_rep, batch_size, 1) < x)
        actions = mix_mask * ts_actions + (1 - mix_mask) * ur_actions
        return actions

class EpsTS(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_arm = sim_config.n_arm
        bayes_model = sim_config.bayes_model

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0,
                              destination=ad.arr_axis['horizon'])

        ur_size = np.delete(np.array(samples.shape), ad.arr_axis['n_arm'])
        ur_ind = ad.tile(arr=(np.random.binomial(n=1, p=self.algo_para, size=ur_size) == 1),
                         axis_name='n_arm')

        actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(1, np.ones(n_arm) / n_arm, size=ur_size)[ur_ind]
        return actions

class Top2TS(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        bayes_model = sim_config.bayes_model

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0,
                              destination=ad.arr_axis['horizon'])

        ur_size = np.delete(np.array(samples.shape), ad.arr_axis['n_arm'])
        ur_ind = ad.tile(
            arr=(np.random.binomial(n=1, p=1 - self.algo_para, size=ur_size) == 1),
            axis_name='n_arm',
        )

        actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        samples[actions] = -np.inf
        sec_actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))

        if np.max(ur_ind) == 1:
            actions[ur_ind] = sec_actions[ur_ind]
        return actions


def _solve_info_ratio_analytical(Delta, g, lam=2.0):
    """Min of pair-mix generalized info ratio. Vectorized across n_rep.

    For each arm pair (a, b), the mixed policy pi = q*e_a + (1-q)*e_b has
        psi(q) = (q*Delta_a + (1-q)*Delta_b)^lam / (q*g_a + (1-q)*g_b).
    lam=2 is standard V-IDS; other lam values follow the generalized ratio
    in Lattimore & Gyorgy (2020).

    Parameters
    ----------
    Delta : ndarray (n_rep, K) -- non-negative expected regret per arm.
    g     : ndarray (n_rep, K) -- strictly positive info gain per arm.
    lam   : float -- exponent on expected regret in the info ratio.

    Returns
    -------
    best_a, best_b : ndarray (n_rep,) int -- chosen arm pair.
    best_q         : ndarray (n_rep,)     -- probability of playing best_a.
    """
    n_rep, K = Delta.shape
    D_a = Delta[:, :, None]
    D_b = Delta[:, None, :]
    g_a = g[:, :, None]
    g_b = g[:, None, :]

    q_grid = np.linspace(0.0, 1.0, 51)[:, np.newaxis, np.newaxis, np.newaxis]
    num = q_grid * D_a + (1.0 - q_grid) * D_b
    den = np.maximum(q_grid * g_a + (1.0 - q_grid) * g_b, 1e-12)
    psi = np.power(np.maximum(num, 0.0), lam) / den

    best_q_idx = np.argmin(psi, axis=0)
    pair_psi = np.min(psi, axis=0)
    pair_q = q_grid[best_q_idx, 0, 0, 0]

    flat = pair_psi.reshape(n_rep, K * K)
    idx = np.argmin(flat, axis=-1)
    best_a, best_b = np.divmod(idx, K)
    best_q = pair_q[np.arange(n_rep), best_a, best_b]
    return best_a, best_b, best_q


class IDS(BanditAlgorithm):
    """Information-Directed Sampling, variance-based (V-IDS).

    Russo & Van Roy 2018, "Learning to Optimize via Information-Directed
    Sampling". Plays the distribution over arms minimizing the information
    ratio
        psi(pi) = (pi . Delta)^lam / (pi . g)
    where Delta_a is the expected regret of arm a (against the unknown
    optimal arm) and g_a is the variance-based info gain. lam=2 is the
    standard V-IDS objective; other lam values give a tunable exploration
    tradeoff (generalized info ratio). The optimal distribution has support
    on at most 2 arms (RvR Prop 6), so we search over arm pairs and the mix
    weight q via `_solve_info_ratio_analytical`.

    All three joint-posterior summaries (p_best, mu_hat, M_cond) are
    estimated from a single batch of M Monte Carlo posterior samples per
    step.

    algo_para : dict with optional keys
        'n_mc'   : int (default 200) -- Monte Carlo samples per step.
        'lambda' : float (default 2) -- regret exponent in the info ratio.
    """

    def sample_action(self, sim_config, action_hist, reward_hist,
                      reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_arm = sim_config.n_arm
        bayes_model = sim_config.bayes_model

        para = self.algo_para if isinstance(self.algo_para, dict) else {}
        M = para.get('n_mc', 200)
        lam = para.get('lambda', 2.0)

        bayes_model.update_posterior(
            action_hist, reward_hist, reward2_hist, ad.arr_axis
        )

        # Raw posterior samples: (M, n_rep, n_arm)
        samples = bayes_model.get_posterior_sample(size=M)['mean']
        samples = samples.reshape(M, -1, n_arm)
        n_rep = samples.shape[1]

        # Joint-posterior summaries
        best_idx = np.argmax(samples, axis=-1)
        one_hot = np.eye(n_arm, dtype=samples.dtype)[best_idx]

        p_best = one_hot.mean(axis=0)                            # (n_rep, K)
        mu_hat = samples.mean(axis=0)                            # (n_rep, K)

        numerator = np.einsum('mra,mrb->rab', samples, one_hot)
        denom = (p_best * M)[:, np.newaxis, :]
        M_cond = np.divide(numerator, denom,
                           out=np.zeros_like(numerator),
                           where=denom > 0)                      # (n_rep, K, K)

        diag_M = np.einsum('rbb->rb', M_cond)
        rho_star = np.sum(p_best * diag_M, axis=-1)              # (n_rep,)

        Delta = np.maximum(rho_star[:, np.newaxis] - mu_hat, 0)  # (n_rep, K)
        diff_sq = (M_cond - mu_hat[..., np.newaxis]) ** 2
        g = np.einsum('rb,rab->ra', p_best, diff_sq)             # (n_rep, K)
        g = np.maximum(g, 1e-12)

        best_a, best_b, best_q = _solve_info_ratio_analytical(Delta, g, lam=lam)

        # Sample one action per batch element from the chosen 2-arm mix
        rng = np.random.random(size=(n_rep, batch_size))
        chosen = np.where(rng < best_q[:, np.newaxis],
                          best_a[:, np.newaxis],
                          best_b[:, np.newaxis])
        actions = np.eye(n_arm, dtype=bool)[chosen]              # (n_rep, batch_size, K)
        return actions


class TSPostDiffURWithResample(BanditAlgorithm):
    """Archived. Uses two posterior draws (resample). Current version is TSPostDiff."""
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_arm = sim_config.n_arm
        bayes_model = sim_config.bayes_model

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0, destination=ad.arr_axis['horizon'])

        diff = (np.max(samples, axis=ad.arr_axis['n_arm'])
                - np.min(samples, axis=ad.arr_axis['n_arm']))
        ur_ind = ad.tile(arr=(diff < self.algo_para), axis_name='n_arm')

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0, destination=ad.arr_axis['horizon'])

        actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(
                1, np.ones(n_arm) / n_arm, size=diff.shape
            )[ur_ind]
        return actions


class TSPostDiffTopWithResample(BanditAlgorithm):
    """Archived. Uses two posterior draws (resample). Current version is TSPostDiff."""
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        bayes_model = sim_config.bayes_model

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0, destination=ad.arr_axis['horizon'])

        diff = np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True) - samples
        ur_ind = (diff <= self.algo_para)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0, destination=ad.arr_axis['horizon'])

        actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        top_ur_samples = np.random.random(size=ur_ind.shape) * ur_ind
        ur_actions = (top_ur_samples == np.max(
            top_ur_samples, axis=ad.arr_axis['n_arm'], keepdims=True
        ))

        ur_bool = ad.tile(
            arr=np.max(actions * ur_ind, axis=ad.arr_axis['n_arm']),
            axis_name='n_arm',
        )
        if np.max(ur_bool) == 1:
            actions[ur_bool] = ur_actions[ur_bool]
        return actions

class TSPostDiff(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        bayes_model = sim_config.bayes_model

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0, destination=ad.arr_axis['horizon'])

        diff = np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True) - samples
        ur_ind = (diff <= self.algo_para)

        top_ur_samples = np.random.random(size=ur_ind.shape) * ur_ind
        actions = (top_ur_samples == np.max(
            top_ur_samples, axis=ad.arr_axis['n_arm'], keepdims=True
        ))

        return actions


class TSPostDiffLinear(BanditAlgorithm):
    """Winner-circle TS with per-factor posterior thresholding.

    algo_para : dict with keys:
        'delta_vec' : list of float, length (d-1)
            One threshold per factor column (excluding intercept).
        'resample'  : bool (default False)
            If True, draw a second posterior sample for the arm value
            when |theta_j| > delta_j (the factor is "significant").
            If False, use the same draw that determined significance.

    Logic per factor j = 1..d-1:
        - Draw theta from posterior (output_theta=True).
        - If |theta_j| > delta_j  → keep theta_j (or redraw if resample).
        - If |theta_j| <= delta_j → set theta_j = 0  (factor "in the circle").
    Arm values = F @ theta_modified.  Pick argmax (tiny noise for ties).
    """

    def sample_action(self, sim_config, action_hist, reward_hist,
                      reward2_hist, batch_size=1):
        ad = sim_config.ad
        bayes_model = sim_config.bayes_model
        F = bayes_model.F  # (K, d)
        d = bayes_model.d
        para = self.algo_para
        delta_vec = np.asarray(para['delta_vec'])  # (d-1,)
        resample = para.get('resample', False)

        bayes_model.update_posterior(
            action_hist, reward_hist, reward2_hist, ad.arr_axis
        )

        # First draw — used for thresholding (and arm selection if no resample)
        result1 = bayes_model.get_posterior_sample(
            size=batch_size, output_theta=True
        )
        theta1 = np.moveaxis(
            result1['theta'], source=0, destination=ad.arr_axis['horizon']
        )  # (n_rep, batch_size, d)

        if resample:
            # Second draw for arm selection among significant factors
            result2 = bayes_model.get_posterior_sample(
                size=batch_size, output_theta=True
            )
            theta2 = np.moveaxis(
                result2['theta'], source=0, destination=ad.arr_axis['horizon']
            )
        else:
            theta2 = theta1

        # Build modified theta: zero out insignificant factors
        theta_mod = theta2.copy()
        for j in range(1, d):
            insignificant = np.abs(theta1[..., j]) <= delta_vec[j - 1]
            theta_mod[..., j][insignificant] = 0.0

        # Arm values = theta_mod @ F^T  →  (n_rep, batch_size, K)
        arm_values = np.einsum('...d,kd->...k', theta_mod, F)

        # Argmax with tiny noise for tie-breaking
        arm_values += np.random.uniform(0, 1e-10, size=arm_values.shape)
        actions = (
            arm_values
            == np.max(arm_values, axis=ad.arr_axis['n_arm'], keepdims=True)
        )
        return actions



class AgrawalGoyalLinearTS(BanditAlgorithm):
    """Agrawal & Goyal (2013) Linear TS with inflated posterior.

    Uses their exact formulation:
        B(t) = I + X^T X           (no noise scaling)
        mu_hat = B^{-1} X^T y      (ridge regression)
        theta ~ N(mu_hat, v^2 B^{-1})

    where v = R * sqrt(9 * d * ln(T / delta)).

    We achieve this by setting sigma2=1, prior precision=I in the shared
    LinearNormalKnownVar model, then passing v as scale_override to sampling.

    algo_para : dict with keys:
        'R'     : float — sub-Gaussian parameter (default 0.5)
        'delta' : float — confidence parameter (default 1/T)
    """

    def sample_action(self, sim_config, action_hist, reward_hist,
                      reward2_hist, batch_size=1):
        ad = sim_config.ad
        bayes_model = sim_config.bayes_model
        d = bayes_model.d

        # Override model to match A&G: B = I + X^T X
        bayes_model.sigma2 = 1.0
        bayes_model.prior['Sigma0_inv'] = np.eye(d)
        bayes_model.prior['mu0'] = np.zeros(d)

        bayes_model.update_posterior(
            action_hist, reward_hist, reward2_hist, ad.arr_axis
        )

        # Compute inflation factor v
        R = self.algo_para.get('R', 0.5)
        T = sim_config.horizon
        delta = self.algo_para.get('delta', 1.0 / T)
        v = R * np.sqrt(9 * d * np.log(T / delta))

        result = bayes_model.get_posterior_sample(
            size=batch_size, scale_override=v
        )
        samples = np.moveaxis(
            result['mean'], source=0, destination=ad.arr_axis['horizon']
        )

        actions = (
            samples
            == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True)
        )
        return actions


class LinearUCB(BanditAlgorithm):
    """LinUCB using the shared linear posterior from LinearNormalKnownVar.

    UCB index for arm k:
        ucb_k = f_k^T theta_hat + alpha * sqrt(sigma2 * f_k^T A^{-1} f_k)

    where theta_hat = A^{-1} b is the posterior mean and A is the precision.

    algo_para : float
        Exploration parameter alpha (e.g., 0.5, 1.0, 2.0).
        Larger alpha → more exploration → lower reward, better estimation.
    """

    def sample_action(self, sim_config, action_hist, reward_hist,
                      reward2_hist, batch_size=1):
        ad = sim_config.ad
        bayes_model = sim_config.bayes_model
        F = bayes_model.F                       # (K, d)
        alpha = self.algo_para

        bayes_model.update_posterior(
            action_hist, reward_hist, reward2_hist, ad.arr_axis
        )

        A = bayes_model.posterior['A']           # (n_rep, d, d)
        b = bayes_model.posterior['b']           # (n_rep, d)

        # Posterior mean: theta_hat = A^{-1} b  →  (n_rep, d)
        theta_hat = np.linalg.solve(A, b[..., np.newaxis]).squeeze(-1)

        # Posterior covariance: A^{-1}  →  (n_rep, d, d)
        # Note: A already includes 1/sigma2 scaling, so posterior cov = A^{-1}
        A_inv = np.linalg.inv(A)

        # Mean per arm: mu_k = f_k^T theta_hat  →  (n_rep, K)
        mu_k = np.einsum('rd,kd->rk', theta_hat, F)

        # Bonus per arm: alpha * sqrt(f_k^T A^{-1} f_k)
        # A already has 1/sigma2 baked in, so no extra sigma2 factor needed
        FA_inv = np.einsum('kd,rde->rke', F, A_inv)  # (n_rep, K, d)
        var_k = np.einsum('rkd,kd->rk', FA_inv, F)   # (n_rep, K)
        bonus = alpha * np.sqrt(np.maximum(var_k, 0))

        ucb = mu_k + bonus                      # (n_rep, K)

        # Reshape to (n_rep, 1, K) and argmax
        ucb = ucb[:, np.newaxis, :]
        ucb += np.random.uniform(0, 1e-10, size=ucb.shape)  # tie-breaking
        actions = (ucb == np.max(ucb, axis=ad.arr_axis['n_arm'], keepdims=True))

        if batch_size > 1:
            actions = np.repeat(actions, batch_size, axis=ad.arr_axis['horizon'])

        return actions


# ── Commented-out linear factorial algorithms (replaced by TSPostDiffLinear) ─
#
# class TSTopURLinear(BanditAlgorithm):
#     """Per-factor Thompson Sampling with uncertainty region.
#
#     algo_para : array-like of shape (d-1,)
#         One threshold per factor column (excluding intercept).
#     """
#     def sample_action(self, sim_config, action_hist, reward_hist,
#                       reward2_hist, batch_size=1):
#         ad = sim_config.ad
#         bayes_model = sim_config.bayes_model
#         F = bayes_model.F
#         d = bayes_model.d
#         thresholds = np.asarray(self.algo_para)
#         bayes_model.update_posterior(
#             action_hist, reward_hist, reward2_hist, ad.arr_axis
#         )
#         result = bayes_model.get_posterior_sample(
#             size=batch_size, output_theta=True
#         )
#         theta = np.moveaxis(
#             result['theta'], source=0, destination=ad.arr_axis['horizon']
#         )
#         candidate_mask = _linear_factor_mask(theta, F, d, thresholds)
#         no_cand = ~np.any(
#             candidate_mask, axis=ad.arr_axis['n_arm'], keepdims=True
#         )
#         candidate_mask |= no_cand
#         rand = np.random.random(size=candidate_mask.shape) * candidate_mask
#         actions = (
#             rand == np.max(rand, axis=ad.arr_axis['n_arm'], keepdims=True)
#         )
#         return actions
#
#
# def _linear_factor_mask(theta, F, d, thresholds):
#     K = F.shape[0]
#     mask = np.ones(theta.shape[:-1] + (K,), dtype=bool)
#     for j in range(1, d):
#         col_vals = F[:, j]
#         levels = np.sort(np.unique(col_vals))
#         theta_j = theta[..., j]
#         thresh_j = thresholds[j - 1]
#         pos_mask = (col_vals == levels[-1])
#         neg_mask = (col_vals == levels[0])
#         go_pos = theta_j >= thresh_j
#         go_neg = theta_j <= -thresh_j
#         uncertain = ~go_pos & ~go_neg
#         factor_mask = (
#             go_pos[..., np.newaxis] * pos_mask
#             + go_neg[..., np.newaxis] * neg_mask
#             + uncertain[..., np.newaxis]
#         ).astype(bool)
#         mask &= factor_mask
#     return mask
#
#
# class TSPostDiffTopLinear(BanditAlgorithm):
#     """Per-factor two-sample TS with uncertainty-region override."""
#     def sample_action(self, sim_config, action_hist, reward_hist,
#                       reward2_hist, batch_size=1):
#         ad = sim_config.ad
#         bayes_model = sim_config.bayes_model
#         F = bayes_model.F
#         d = bayes_model.d
#         thresholds = np.asarray(self.algo_para)
#         bayes_model.update_posterior(
#             action_hist, reward_hist, reward2_hist, ad.arr_axis
#         )
#         result1 = bayes_model.get_posterior_sample(
#             size=batch_size, output_theta=True
#         )
#         theta1 = np.moveaxis(
#             result1['theta'], source=0, destination=ad.arr_axis['horizon']
#         )
#         ur_ind = _linear_factor_mask(theta1, F, d, thresholds)
#         no_cand = ~np.any(
#             ur_ind, axis=ad.arr_axis['n_arm'], keepdims=True
#         )
#         ur_ind |= no_cand
#         samples2 = np.moveaxis(
#             bayes_model.get_posterior_sample(size=batch_size)['mean'],
#             source=0, destination=ad.arr_axis['horizon']
#         )
#         actions = (
#             samples2
#             == np.max(samples2, axis=ad.arr_axis['n_arm'], keepdims=True)
#         )
#         top_ur = np.random.random(size=ur_ind.shape) * ur_ind
#         ur_actions = (
#             top_ur
#             == np.max(top_ur, axis=ad.arr_axis['n_arm'], keepdims=True)
#         )
#         ur_bool = ad.tile(
#             arr=np.max(actions * ur_ind, axis=ad.arr_axis['n_arm']),
#             axis_name='n_arm'
#         )
#         if np.max(ur_bool) == 1:
#             actions[ur_bool] = ur_actions[ur_bool]
#         return actions
