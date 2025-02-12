"""
this file contains all Bayes updating models.
in the future can check if there's packages available
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class BetaBernoulli:
    def __init__(self, number_of_arms, prior=None):
        self.number_of_arms = number_of_arms

        # Define prior as TensorFlow tensors
        if prior is None:
            self.prior = {
                'a': tf.ones([number_of_arms], dtype=tf.float32),
                'b': tf.ones([number_of_arms], dtype=tf.float32)
            }
        else:
            self.prior = {
                'a': tf.convert_to_tensor(prior['a'], dtype=tf.float32),
                'b': tf.convert_to_tensor(prior['b'], dtype=tf.float32)
            }

        # Initialize posterior as a copy of the prior
        self.posterior = self.prior.copy()

    def update_posterior(self, action_hist, reward_hist, arr_axis):
        """
        Updates the posterior based on action and reward history.
        `action_hist` and `reward_hist` are assumed to be TensorFlow tensors.
        the dimension of self.posterior['a'] should be n_rep by 1 by n_arm
        """
        # Sum over horizon axis to calculate counts
        reward_counts = tf.cast(tf.reduce_sum(reward_hist, axis=arr_axis['horizon']), dtype=tf.float32)
        action_counts = tf.cast(tf.reduce_sum(action_hist.astype(int), axis=arr_axis['horizon']), dtype=tf.float32)

        # Update posterior parameters
        self.posterior['a'] = self.prior['a'] + reward_counts # Beta(alpha, beta) alpha = 1+ successes , beta 1+ N_total_played_for that arm - success n_rep,horizon,n_arm dim n_rep,1,n_arm
        self.posterior['b'] = self.prior['b'] + action_counts - reward_counts

    def get_posterior_sample(self, size=1):
        """
        Draws samples from the posterior distribution and returns as a NumPy array.
        `size` specifies the number of samples to draw.
        """
        beta_dist = tfp.distributions.Beta(
            concentration1=self.posterior['a'],
            concentration0=self.posterior['b']
        )
        samples = beta_dist.sample(size).numpy() #dim = n_rep by 1 by n_arm

        return {'mean':samples,
                'var':samples*(1-samples)}  # Convert to NumPy array


class NormalFull:
    """
    Normal with unknown mean and variance
    """
    def __init__(self, number_of_arms, prior=None):
        self.number_of_arms = number_of_arms

        # Define prior as TensorFlow tensors
        if prior is None:
            self.prior = {
                'mu': tf.zeros([number_of_arms], dtype=tf.float32),  # Prior mean
                'lambda': tf.ones([number_of_arms], dtype=tf.float32)*0.01,  # Precision (1/variance)
                'alpha': tf.ones([number_of_arms], dtype=tf.float32) * 0.1,  # Shape of inverse-gamma.
                # Want to choose a small value to be non-informative. But e.g. 0.01 can make variance samples goto infinity. So changed to 0.1
                'beta': tf.ones([number_of_arms], dtype=tf.float32) * 0.1   # Scale of inverse-gamma
            }
        else:
            self.prior = {key: tf.convert_to_tensor(val, dtype=tf.float32) for key, val in prior.items()}

        # Initialize posterior as a copy of the prior
        self.posterior = self.prior.copy()

    def update_posterior(self, action_hist, reward_hist, arr_axis):
        """
        Updates the posterior for all arms based on action and reward histories.
        `action_hist` is a 2D array where 1 indicates the arm was sampled.
        `reward_hist` is a 2D array where rewards are recorded for sampled arms.
        `arr_axis` specifies the axes for time and replication.
        """
        # Convert inputs to tensors
        action_hist = tf.convert_to_tensor(action_hist, dtype=tf.float32)
        reward_hist = tf.convert_to_tensor(reward_hist, dtype=tf.float32)

        # Compute sufficient statistics
        n = tf.reduce_sum(action_hist, axis=arr_axis['horizon'])  # Total samples per arm
        sum_rewards = tf.reduce_sum(reward_hist, axis=arr_axis['horizon'])  # Total rewards per arm
        sum_squared_rewards = tf.reduce_sum(reward_hist ** 2, axis=arr_axis['horizon'])  # Total squared rewards

        # Avoid divide-by-zero issues for arms with no data
        valid_arms = n > 0  # Mask for arms with valid data
        n0 = tf.where(valid_arms, n, tf.ones_like(n))  # Replace zero counts temporarily

        # Compute sample mean and variance
        sample_mean = sum_rewards / n0
        sum_square = (sum_squared_rewards - n0 * sample_mean ** 2)
        #sum_square = tf.where(valid_arms, sum_square, tf.zeros_like(sum_square))  # Handle zero samples

        # Update posterior parameters for all arms
        self.posterior['mu'] = (self.prior['lambda'] * self.prior['mu'] + n * sample_mean) / (self.prior['lambda'] + n)
        self.posterior['lambda'] = self.prior['lambda'] + n
        self.posterior['alpha'] = self.prior['alpha'] + n / 2
        self.posterior['beta'] = self.prior['beta'] + 0.5 * sum_square + (
            (self.prior['lambda'] * n * (sample_mean - self.prior['mu']) ** 2) / (2 * (self.prior['lambda'] + n))
        )

    def get_posterior_sample(self, size=1):
        """
        Draws samples from the posterior distribution of the mean and variance for all arms.
        """
        # Sample variance (sigma^2) from inverse gamma
        sigma_sq_samples = tfp.distributions.InverseGamma(
            concentration=self.posterior['alpha'], scale=self.posterior['beta']).sample(size)

        # Sample mean given variance
        mean_samples = tfp.distributions.Normal(
            loc=self.posterior['mu'], scale=tf.sqrt(sigma_sq_samples / self.posterior['lambda'])).sample() #no need to define size, as it will follow the var sample size

        return {
            'mean': mean_samples.numpy(),  # Convert to NumPy
            'variance': sigma_sq_samples.numpy()  # Convert to NumPy
        }