import numpy as np
from abc import ABC, abstractmethod


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


        samples = np.sum(reward_hist,axis=ad.arr_axis['horizon'], keepdims=True)/np.sum(action_hist,axis=ad.arr_axis['horizon'], keepdims=True)

        ur_size = np.delete(np.array(samples.shape), ad.arr_axis['n_arm'])
        ur_ind = ad.tile(arr=(np.random.binomial(n=1, p=self.algo_para, size=ur_size) == 1),
                         axis_name='n_arm')

        actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(1, np.ones(n_arm) / n_arm, size=ur_size)[ur_ind]
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
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1, approx_rep=101):
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
        n_arm = sim_config.n_arm
        bayes_model = sim_config.bayes_model

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0,
                              destination=ad.arr_axis['horizon'])

        ur_size = np.delete(np.array(samples.shape), ad.arr_axis['n_arm'])
        ur_ind = ad.tile(arr=(np.random.binomial(n=1, p=1 - self.algo_para, size=ur_size) == 1), axis_name='n_arm')

        actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        samples[actions] = -np.inf
        sec_actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))

        if np.max(ur_ind) == 1:
            actions[ur_ind] = sec_actions[ur_ind]
        return actions

class TSPostDiffUR(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_arm = sim_config.n_arm
        bayes_model = sim_config.bayes_model

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0, destination=ad.arr_axis['horizon'])

        diff = np.max(samples, axis=ad.arr_axis['n_arm']) - np.min(samples, axis=ad.arr_axis['n_arm'])
        ur_ind = ad.tile(arr=(diff < self.algo_para), axis_name='n_arm')

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0, destination=ad.arr_axis['horizon'])

        actions = (samples == np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True))
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(1, np.ones(n_arm) / n_arm, size=diff.shape)[ur_ind]
        return actions


class TSPostDiffTop(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_arm = sim_config.n_arm
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
        ur_actions = (top_ur_samples == np.max(top_ur_samples, axis=ad.arr_axis['n_arm'], keepdims=True))

        ur_bool = ad.tile(arr=np.max(actions * ur_ind, axis=ad.arr_axis['n_arm']), axis_name='n_arm')
        if np.max(ur_bool) == 1:
            actions[ur_bool] = ur_actions[ur_bool]
        return actions

class TSTopUR(BanditAlgorithm):
    def sample_action(self, sim_config, action_hist, reward_hist, reward2_hist, batch_size=1):
        ad = sim_config.ad
        n_arm = sim_config.n_arm
        bayes_model = sim_config.bayes_model

        bayes_model.update_posterior(action_hist, reward_hist, reward2_hist, ad.arr_axis)

        samples = np.moveaxis(bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0, destination=ad.arr_axis['horizon'])

        diff = np.max(samples, axis=ad.arr_axis['n_arm'], keepdims=True) - samples
        ur_ind = (diff <= self.algo_para)

        top_ur_samples = np.random.random(size=ur_ind.shape) * ur_ind
        actions = (top_ur_samples == np.max(top_ur_samples, axis=ad.arr_axis['n_arm'], keepdims=True))

        return actions
