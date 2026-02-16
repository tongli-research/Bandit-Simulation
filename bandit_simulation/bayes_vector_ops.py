"""
Bayesian Posterior Updating Models (Vectorized)

This module defines Bayesian updating models designed for efficient **vectorized** simulation.
Unlike typical Bayesian packages, these models are built to support simultaneous updates across
multiple simulations or experimental replications (i.e., batch processing).

Key Features:
- Vectorized posterior updates for parallel simulations.
- Simple, modular design for easy extension to other Bayesian models.

Primary Models:
- Beta-Bernoulli (for binary reward models, e.g., Bernoulli bandits).
- Normal-Inverse-Gamma (for normal rewards with unknown mean and variance).

Typical Use Case:
Used in large-scale simulation studies, such as bandit algorithm evaluations, where posterior
distributions are updated repeatedly in a high-dimensional setting.

"""
from abc import ABC, abstractmethod
import numpy as np


def _beta_sample(a, b, size):
    """Sample from Beta distribution with proper broadcasting."""
    a = np.asarray(a)
    b = np.asarray(b)
    shape = np.broadcast(a, b).shape
    total_shape = (size,) + shape if isinstance(size, int) else size + shape
    return np.random.beta(np.broadcast_to(a, shape), np.broadcast_to(b, shape), size=total_shape)


def _inverse_gamma_sample(alpha, beta, size):
    """Sample from Inverse-Gamma distribution via Gamma."""
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    shape = np.broadcast(alpha, beta).shape
    total_shape = (size,) + shape if isinstance(size, int) else size + shape
    gamma_sample = np.random.gamma(np.broadcast_to(alpha, shape), 1 / np.broadcast_to(beta, shape),
                                   size=total_shape)
    return 1 / gamma_sample


class BayesianPosteriorModel(ABC):
    """
    Abstract base class for Bayesian posterior models.
    All models must implement posterior update and sampling.
    """

    def __init__(self, number_of_arms, prior=None):
        self.number_of_arms = number_of_arms

    @abstractmethod
    def update_posterior(self, action_hist, reward_hist, reward2_hist, arr_axis):
        pass

    @abstractmethod
    def get_posterior_sample(self, size=1):
        pass


class BetaBernoulli(BayesianPosteriorModel):
    """Bayesian model for Bernoulli rewards with Beta prior."""

    def __init__(self, number_of_arms, prior=None):
        super().__init__(number_of_arms, prior)
        if prior is None:
            self.prior = {
                'a': np.ones([number_of_arms]),
                'b': np.ones([number_of_arms])
            }
        self.posterior = self.prior.copy()

    def update_posterior(self, action_hist, reward_hist, reward2_hist, arr_axis):
        reward_counts = np.sum(reward_hist, axis=arr_axis['horizon'])
        action_counts = np.sum(action_hist, axis=arr_axis['horizon'])
        self.posterior['a'] = self.prior['a'] + reward_counts
        self.posterior['b'] = self.prior['b'] + action_counts - reward_counts

    def get_posterior_sample(self, size=1):
        samples = _beta_sample(self.posterior['a'], self.posterior['b'], size)
        return {
            'mean': samples,
            'var': samples * (1 - samples)
        }


class LinearNormalKnownVar(BayesianPosteriorModel):
    """Bayesian linear model for Normal rewards with known variance.

    Arms share coefficients through a feature matrix F (K x d).
    mu_k = F[k,:] @ theta for each arm k.

    Prior:      theta ~ N(mu0, sigma2 * Sigma0_inv^{-1})
    Likelihood: reward | arm k ~ N(F[k,:] @ theta, sigma2)

    Posterior stored as precision form: A = Sigma0_inv + (1/sigma2) * F^T diag(n) F,
    b = Sigma0_inv @ mu0 + (1/sigma2) * F^T @ reward_sums.
    """

    def __init__(self, number_of_arms, arm_feature_matrix, prior=None, sigma2=1.0):
        super().__init__(number_of_arms, prior)
        F = np.asarray(arm_feature_matrix, dtype=float)
        if F.shape[0] != number_of_arms:
            raise ValueError(
                f"arm_feature_matrix has {F.shape[0]} rows, expected {number_of_arms}"
            )
        self.F = F
        self.d = F.shape[1]
        self.sigma2 = sigma2

        if prior is None:
            lambda0 = 1e-2
            self.prior = {
                'mu0': np.zeros(self.d),
                'Sigma0_inv': lambda0 * np.eye(self.d),
            }
        self.posterior = {
            'A': self.prior['Sigma0_inv'].copy(),
            'b': self.prior['Sigma0_inv'] @ self.prior['mu0'],
        }

    def update_posterior(self, action_hist, reward_hist, reward2_hist, arr_axis):
        # Sum over horizon first — same pattern as BetaBernoulli / NormalFull
        action_counts = np.sum(action_hist, axis=arr_axis['horizon'])  # (n_rep, K)
        reward_sums = np.sum(reward_hist, axis=arr_axis['horizon'])    # (n_rep, K)

        # Sufficient statistics via per-arm totals
        # Sxx[r] = sum_k n[r,k] * F[k,:] F[k,:]^T  -> (n_rep, d, d)
        Sxx = np.einsum('rk,kd,ke->rde', action_counts, self.F, self.F)
        # Sxy[r] = sum_k reward_sum[r,k] * F[k,:]   -> (n_rep, d)
        Sxy = np.einsum('rk,kd->rd', reward_sums, self.F)

        Sigma0_inv = self.prior['Sigma0_inv']
        mu0 = self.prior['mu0']

        self.posterior['A'] = Sigma0_inv + (1.0 / self.sigma2) * Sxx
        self.posterior['b'] = Sigma0_inv @ mu0 + (1.0 / self.sigma2) * Sxy

    def get_posterior_sample(self, size=1, output_theta=False):
        A = self.posterior['A']   # (n_rep, d, d)
        b = self.posterior['b']   # (n_rep, d)

        # Posterior mean: m = A^{-1} b  -> (n_rep, d)
        m = np.linalg.solve(A, b[..., np.newaxis]).squeeze(-1)

        # Cholesky: L L^T = A,  cov = sigma2 * A^{-1}
        # theta = m + sqrt(sigma2) * L^{-T} z,  z ~ N(0, I)
        L = np.linalg.cholesky(A)
        LT = np.swapaxes(L, -2, -1)

        z = np.random.standard_normal((size,) + m.shape)  # (size, n_rep, d)
        x = np.linalg.solve(LT, z[..., np.newaxis]).squeeze(-1)
        theta_samples = m + np.sqrt(self.sigma2) * x      # (size, n_rep, d)

        # Map to arm means: (size, n_rep, K)
        samples = np.einsum('...d,kd->...k', theta_samples, self.F)
        result = {'mean': samples}
        if output_theta:
            result['theta'] = theta_samples
        return result


class NormalFull(BayesianPosteriorModel):
    """Bayesian model for Normal rewards with unknown mean and variance."""

    def __init__(self, number_of_arms, prior=None):
        super().__init__(number_of_arms, prior)
        if prior is None:
            self.prior = {
                'mu': np.zeros([number_of_arms]),
                'lambda': np.ones([number_of_arms]) * 0.01,
                'alpha': np.ones([number_of_arms]) * 0.1,
                'beta': np.ones([number_of_arms]) * 0.1
            }
        self.posterior = self.prior.copy()

    def update_posterior(self, action_hist, reward_hist, reward2_hist, arr_axis):
        n = np.sum(action_hist, axis=arr_axis['horizon'])
        sum_rewards = np.sum(reward_hist, axis=arr_axis['horizon'])
        sum_squared_rewards = np.sum(reward2_hist, axis=arr_axis['horizon'])

        valid_arms = n > 0
        n0 = np.where(valid_arms, n, np.ones_like(n))
        sample_mean = sum_rewards / n0
        sum_square = sum_squared_rewards - n0 * sample_mean ** 2

        self.posterior['mu'] = (self.prior['lambda'] * self.prior['mu'] + n * sample_mean) / (self.prior['lambda'] + n)
        self.posterior['lambda'] = self.prior['lambda'] + n
        self.posterior['alpha'] = self.prior['alpha'] + n / 2
        self.posterior['beta'] = self.prior['beta'] + 0.5 * sum_square + (
            (self.prior['lambda'] * n * (sample_mean - self.prior['mu']) ** 2) / (2 * (self.prior['lambda'] + n))
        )

    def get_posterior_sample(self, size=1):
        sigma_sq_samples = _inverse_gamma_sample(
            self.posterior['alpha'], self.posterior['beta'], size
        )
        mean_samples = np.random.normal(
            loc=self.posterior['mu'],
            scale=np.sqrt(sigma_sq_samples / self.posterior['lambda']),
            size=None
        )
        return {
            'mean': mean_samples,
            'variance': sigma_sq_samples
        }
