"""
Bayesian Posterior Updating Models (Vectorized)

This module defines Bayesian updating models designed for efficient **vectorized** simulation.
Unlike typical Bayesian packages, these models are built to support simultaneous updates across
multiple simulations or experimental replications (i.e., batch processing).

Key Features:
- Vectorized posterior updates for parallel simulations.
- Backend-agnostic design: can switch between NumPy and TensorFlow (or other frameworks) via backend_ops.
- Simple, modular design for easy extension to other Bayesian models.

Primary Models:
- Beta-Bernoulli (for binary reward models, e.g., Bernoulli bandits).
- Normal-Inverse-Gamma (for normal rewards with unknown mean and variance).

Typical Use Case:
Used in large-scale simulation studies, such as bandit algorithm evaluations, where posterior
distributions are updated repeatedly in a high-dimensional setting.

"""
from abc import ABC, abstractmethod


class BackendOps(ABC):
    """Abstract base class for backend operations (NumPy / TensorFlow)."""
    sqrt = None
    normal = None
    beta = None
    inverse_gamma = None
    sum = None
    ones = None
    zeros = None
    ones_like = None
    where = None


class BackendOpsNP(BackendOps):
    """Backend operations using NumPy."""

    def __init__(self):
        super().__init__()

    def manual_init(self):
        import numpy as np

        def beta_wrapper(a, b, size):
            # Broadcast parameters properly for np.random.beta
            a = np.asarray(a)
            b = np.asarray(b)
            shape = np.broadcast(a, b).shape
            total_shape = (size,) + shape if isinstance(size, int) else size + shape
            return np.random.beta(np.broadcast_to(a, shape), np.broadcast_to(b, shape), size=total_shape)

        def inverse_gamma_wrapper(alpha, beta, size):
            alpha = np.asarray(alpha)
            beta = np.asarray(beta)
            shape = np.broadcast(alpha, beta).shape
            total_shape = (size,) + shape if isinstance(size, int) else size + shape
            gamma_sample = np.random.gamma(np.broadcast_to(alpha, shape), 1 / np.broadcast_to(beta, shape),
                                           size=total_shape)
            return 1 / gamma_sample

        self.sqrt = np.sqrt
        self.normal = np.random.normal
        self.beta = beta_wrapper
        self.inverse_gamma = inverse_gamma_wrapper
        self.sum = lambda x, axis: np.sum(x, axis=axis)
        self.ones = lambda shape: np.ones(shape)
        self.zeros = lambda shape: np.zeros(shape)
        self.ones_like = np.ones_like
        self.where = np.where

class BackendOpsTF(BackendOps):
    """Backend operations using TensorFlow Probability (TFP)."""

    def __init__(self):
        super().__init__()  # Inherit the placeholders


    def manual_init(self):
        import tensorflow as tf
        import tensorflow_probability as tfp
        self.sqrt = tf.sqrt
        self.normal = lambda loc, scale, size: tfp.distributions.Normal(loc, scale).sample(sample_shape=size)
        self.beta = lambda a, b, size: tfp.distributions.Beta(a, b).sample(sample_shape=size)
        self.inverse_gamma = lambda alpha, beta, size: tfp.distributions.InverseGamma(alpha, beta).sample(
            sample_shape=size)
        self.sum = lambda x, axis: tf.reduce_sum(tf.cast(x, tf.float32), axis=axis)
        self.ones = lambda shape: tf.ones(shape, dtype=tf.float32)
        self.zeros = lambda shape: tf.zeros(shape, dtype=tf.float32)
        self.ones_like = tf.ones_like
        self.where = tf.where

class BayesianPosteriorModel(ABC):
    """
    Abstract base class for Bayesian posterior models.
    All models must implement posterior update and sampling.
    """

    def __init__(self, number_of_arms, backend_ops, prior=None):
        self.number_of_arms = number_of_arms
        self.backend_ops = backend_ops

    @abstractmethod
    def update_posterior(self, action_hist, reward_hist, reward2_hist, arr_axis):
        pass

    @abstractmethod
    def get_posterior_sample(self, size=1):
        pass


class BetaBernoulli(BayesianPosteriorModel):
    """Bayesian model for Bernoulli rewards with Beta prior."""

    def __init__(self, number_of_arms, backend_ops, prior=None):
        super().__init__(number_of_arms, backend_ops, prior)
        if prior is None:
            self.prior = {
                'a': self.backend_ops.ones([number_of_arms]),
                'b': self.backend_ops.ones([number_of_arms])
            }
        self.posterior = self.prior.copy()

    def update_posterior(self, action_hist, reward_hist, reward2_hist, arr_axis):
        reward_counts = self.backend_ops.sum(reward_hist, axis=arr_axis['horizon'])
        action_counts = self.backend_ops.sum(action_hist, axis=arr_axis['horizon'])
        self.posterior['a'] = self.prior['a'] + reward_counts
        self.posterior['b'] = self.prior['b'] + action_counts - reward_counts

    def get_posterior_sample(self, size=1):
        samples = self.backend_ops.beta(self.posterior['a'], self.posterior['b'], size)
        return {
            'mean': samples,
            'var': samples * (1 - samples)
        }


class NormalFull(BayesianPosteriorModel):
    """Bayesian model for Normal rewards with unknown mean and variance."""

    def __init__(self, number_of_arms, backend_ops, prior=None):
        super().__init__(number_of_arms, backend_ops, prior)
        if prior is None:
            self.prior = {
                'mu': self.backend_ops.zeros([number_of_arms]),
                'lambda': self.backend_ops.ones([number_of_arms]) * 0.01,
                'alpha': self.backend_ops.ones([number_of_arms]) * 0.1,
                'beta': self.backend_ops.ones([number_of_arms]) * 0.1
            }
        self.posterior = self.prior.copy()

    def update_posterior(self, action_hist, reward_hist, reward2_hist, arr_axis):
        n = self.backend_ops.sum(action_hist, axis=arr_axis['horizon'])
        sum_rewards = self.backend_ops.sum(reward_hist, axis=arr_axis['horizon'])
        sum_squared_rewards = self.backend_ops.sum(reward2_hist, axis=arr_axis['horizon'])

        valid_arms = n > 0
        n0 = self.backend_ops.where(valid_arms, n, self.backend_ops.ones_like(n))
        sample_mean = sum_rewards / n0
        sum_square = sum_squared_rewards - n0 * sample_mean ** 2

        self.posterior['mu'] = (self.prior['lambda'] * self.prior['mu'] + n * sample_mean) / (self.prior['lambda'] + n)
        self.posterior['lambda'] = self.prior['lambda'] + n
        self.posterior['alpha'] = self.prior['alpha'] + n / 2
        self.posterior['beta'] = self.prior['beta'] + 0.5 * sum_square + (
            (self.prior['lambda'] * n * (sample_mean - self.prior['mu']) ** 2) / (2 * (self.prior['lambda'] + n))
        )

    def get_posterior_sample(self, size=1):
        sigma_sq_samples = self.backend_ops.inverse_gamma(
            self.posterior['alpha'], self.posterior['beta'], size
        )
        mean_samples = self.backend_ops.normal(
            loc=self.posterior['mu'],
            scale=self.backend_ops.sqrt(sigma_sq_samples / self.posterior['lambda']),
            size=None
        )
        return {
            'mean': mean_samples,
            'variance': sigma_sq_samples
        }
