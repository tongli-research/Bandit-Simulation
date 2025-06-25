"""
This file contains all Bayes updating models.
The main purpose (reason for not using existing packages) is to support vectorized calculation
I.e. we want to update a batch of (representing multiple simulations in parallel) Bayes models simulteniously
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import invgamma, norm


class BetaBernoulli:
    def __init__(self, number_of_arms, prior=None):
        self.number_of_arms = number_of_arms

        if prior is None:
            self.prior = {
                'a': np.ones(number_of_arms, dtype=np.float32),
                'b': np.ones(number_of_arms, dtype=np.float32)
            }
        else:
            self.prior = {
                'a': np.array(prior['a'], dtype=np.float32),
                'b': np.array(prior['b'], dtype=np.float32)
            }

        self.posterior = self.prior.copy()

    def update_posterior(self, action_hist, reward_hist, arr_axis):
        reward_counts = np.sum(reward_hist, axis=arr_axis['horizon']).astype(np.float32)
        action_counts = np.sum(action_hist.astype(int), axis=arr_axis['horizon']).astype(np.float32)

        self.posterior['a'] = self.prior['a'] + reward_counts
        self.posterior['b'] = self.prior['b'] + action_counts - reward_counts

    def get_posterior_sample(self, size=1):
        samples = beta_dist.rvs(
            a=self.posterior['a'],
            b=self.posterior['b'],
            size=(size, *self.posterior['a'].shape)
        )

        return {
            'mean': samples,
            'var': samples * (1 - samples)
        }


class NormalFull:
    def __init__(self, number_of_arms, prior=None):
        self.number_of_arms = number_of_arms

        if prior is None:
            self.prior = {
                'mu': np.zeros(number_of_arms, dtype=np.float32),
                'lambda': np.ones(number_of_arms, dtype=np.float32) * 0.01,
                'alpha': np.ones(number_of_arms, dtype=np.float32) * 0.1,
                'beta': np.ones(number_of_arms, dtype=np.float32) * 0.1
            }
        else:
            self.prior = {key: np.array(val, dtype=np.float32) for key, val in prior.items()}

        self.posterior = self.prior.copy()

    def update_posterior(self, action_hist, reward_hist, arr_axis):
        action_hist = np.array(action_hist, dtype=np.float32)
        reward_hist = np.array(reward_hist, dtype=np.float32)

        n = np.sum(action_hist, axis=arr_axis['horizon'])
        sum_rewards = np.sum(reward_hist, axis=arr_axis['horizon'])
        sum_squared_rewards = np.sum(reward_hist ** 2, axis=arr_axis['horizon'])

        valid_arms = n > 0
        n0 = np.where(valid_arms, n, 1.0)

        sample_mean = sum_rewards / n0
        sum_square = sum_squared_rewards - n0 * sample_mean ** 2

        self.posterior['mu'] = (self.prior['lambda'] * self.prior['mu'] + n * sample_mean) / (self.prior['lambda'] + n)
        self.posterior['lambda'] = self.prior['lambda'] + n
        self.posterior['alpha'] = self.prior['alpha'] + n / 2
        self.posterior['beta'] = self.prior['beta'] + 0.5 * sum_square + (
            (self.prior['lambda'] * n * (sample_mean - self.prior['mu']) ** 2) / (2 * (self.prior['lambda'] + n))
        )

    def get_posterior_sample(self, size=1):
        sigma_sq_samples = invgamma.rvs(
            a=self.posterior['alpha'],
            scale=self.posterior['beta'],
            size=(size, *self.posterior['alpha'].shape)
        )

        mean_samples = norm.rvs(
            loc=self.posterior['mu'],
            scale=np.sqrt(sigma_sq_samples / self.posterior['lambda'])
        )

        return {
            'mean': mean_samples,
            'variance': sigma_sq_samples
        }