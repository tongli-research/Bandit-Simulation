import numpy as np
import bayes_model as bm
import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class StoBandit:
    def __init__(self,  reward_model, bayes_model=None):
        #self.ad = arr_dim


        self.bayes_model = bayes_model  # allow user input prior. Otherwise, use default prior
        self.n_arm = reward_model.n_arm
        self.reward_model = reward_model

        if self.bayes_model is None:  # for now, default is beta-bernoulli. Will modify it (the default should depend on reward distribution)
            self.bayes_model = bm.BetaBernoulli(number_of_arms=self.n_arm)

    def round_rubin(self, algo_para, action_hist, reward_hist, batch_size=1):
        shape_list = list(action_hist.shape)
        time_step = shape_list[self.ad.arr_axis['horizon']]
        shape_list[self.ad.arr_axis['horizon']] = batch_size

        # advance index:  (:, range(batch), range(batch)+time_step mod num_arm)
        actions = np.zeros(shape_list)
        slice_list = [slice(None)] * len(shape_list)
        slice_list[self.ad.arr_axis['horizon']] = np.arange(batch_size)
        slice_list[self.ad.arr_axis['n_arm']] = (np.arange(batch_size) + time_step) % self.n_arm
        actions[tuple(slice_list)]=1
        return actions



    def eps_ts(self, algo_para, action_hist, reward_hist, batch_size=1):
        # update posterior
        self.bayes_model.update_posterior(action_hist, reward_hist, self.ad.arr_axis)

        # sample posterior
        samples = np.moveaxis(self.bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0,
                              destination=self.ad.arr_axis['horizon'])

        ur_size = np.delete(np.array(samples.shape), self.ad.arr_axis['n_arm']) #the array shape for the following ur_ind
        ur_ind =self.ad.tile(arr= (np.random.binomial(n=1, p=algo_para, size= ur_size  ) == 1), #detarmine places (with probability = eps) where we want to do UR
                             axis_name = 'n_arm')

        #dim of action is: n_rep by 1 by n_arm
        actions = (samples == np.max(samples, axis=  self.ad.arr_axis['n_arm'] , keepdims=True))  # TS actions
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(1, np.ones(self.n_arm) / self.n_arm, size= ur_size )[ur_ind]
        return actions


    def eps_top2_ts(self, algo_para, action_hist, reward_hist, batch_size=1):
        """
        algo_para: epsilon percent time, UR. Otherwise, play top 2 with 50/50
        """

        # update posterior
        self.bayes_model.update_posterior(action_hist, reward_hist, self.ad.arr_axis)

        # sample posterior
        samples = np.moveaxis(self.bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0,
                              destination=self.ad.arr_axis['horizon'])

        ur_size = np.delete(np.array(samples.shape),
                            self.ad.arr_axis['n_arm'])  # the array shape for the following ur_ind
        ur_ind = self.ad.tile(arr=(np.random.binomial(n=1, p=algo_para, size=ur_size) == 1),# detarmine places (with probability = eps) where we want to do UR
                              axis_name='n_arm')
        sec_best_ind = self.ad.tile(arr=(np.random.binomial(n=1, p=0.5, size=ur_size) == 1),# detarmine places (with probability = eps) where we want to do UR
                                    axis_name='n_arm')


        actions = (samples == np.max(samples, axis=self.ad.arr_axis['n_arm'], keepdims=True))  # TS actions
        samples[actions] = -np.inf  # remove max samples (so 2nd best become the best)
        sec_actions = (samples == np.max(samples, axis=self.ad.arr_axis['n_arm'],
                                         keepdims=True))  # candidate actions (2nd best)

        if np.max(sec_best_ind) == 1:
            actions[sec_best_ind] = sec_actions[sec_best_ind]
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(1, np.ones(self.n_arm) / self.n_arm, size=ur_size)[ur_ind]
        return actions

    def top2_ts(self, algo_para, action_hist, reward_hist, batch_size=1):
        """
        algo_para: refer to beta in Top-2 TS.
        beta is the probability we KEEP the best arm, and with probability 1-beta, choose the 2nd best arm
        """
        # update posterior
        self.bayes_model.update_posterior(action_hist, reward_hist, self.ad.arr_axis)

        # sample posterior
        samples = np.moveaxis(self.bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0,
                              destination=self.ad.arr_axis['horizon'])

        ur_size = np.delete(np.array(samples.shape), self.ad.arr_axis['n_arm']) #the array shape for the following ur_ind
        ur_ind =self.ad.tile(arr= (np.random.binomial(n=1, p=1-algo_para, size= ur_size  ) == 1), #detarmine places (with probability = eps) where we want to do UR
                             axis_name = 'n_arm')

        actions = (samples == np.max(samples, axis=  self.ad.arr_axis['n_arm'] , keepdims=True))  # TS actions
        samples[actions] = -np.inf #remove max samples (so 2nd best become the best)
        sec_actions = (samples == np.max(samples, axis=  self.ad.arr_axis['n_arm'] , keepdims=True)) #candidate actions (2nd best)

        if np.max(ur_ind) == 1:
            actions[ur_ind] = sec_actions[ur_ind]
        return actions

    def ts_postdiff_ur(self, algo_para, action_hist, reward_hist, batch_size=1):
        # update posterior
        self.bayes_model.update_posterior(action_hist, reward_hist, self.ad.arr_axis)

        # sample posterior
        samples = np.moveaxis(self.bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0,
                              destination=self.ad.arr_axis['horizon'])

        diff = np.max(samples, axis=self.ad.arr_axis['n_arm']) - np.min(samples, axis=self.ad.arr_axis['n_arm'])
        ur_ind = self.ad.tile(arr = (diff < algo_para),
                              axis_name = 'n_arm')

        samples = np.moveaxis(self.bayes_model.get_posterior_sample(size=batch_size)['mean'], #re-sample (needed for PostDiff)
                              source=0,
                              destination=self.ad.arr_axis['horizon'])


        actions = (samples == np.max(samples, axis=  self.ad.arr_axis['n_arm'] , keepdims=True))  # TS actions
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(1, np.ones(self.n_arm) / self.n_arm, size= diff.shape )[ur_ind]
        return actions

    def ts_postdiff_top(self, algo_para, action_hist, reward_hist, batch_size=1):
        # update posterior
        self.bayes_model.update_posterior(action_hist, reward_hist, self.ad.arr_axis)

        # sample posterior
        samples = np.moveaxis(self.bayes_model.get_posterior_sample(size=batch_size)['mean'],
                              source=0,
                              destination=self.ad.arr_axis['horizon'])

        diff = np.max(samples, axis=self.ad.arr_axis['n_arm'], keepdims=True) - samples
        ur_ind = (diff <= algo_para)

        samples = np.moveaxis(self.bayes_model.get_posterior_sample(size=batch_size)['mean'], #re-sample (needed for PostDiff)
                              source=0,
                              destination=self.ad.arr_axis['horizon'])

        actions = (samples == np.max(samples, axis=  self.ad.arr_axis['n_arm'] , keepdims=True))  # TS actions
        top_ur_samples = np.random.random(size = ur_ind.shape)*ur_ind
        ur_actions = (top_ur_samples == np.max(top_ur_samples, axis=  self.ad.arr_axis['n_arm'] , keepdims=True))

        ur_bool = self.ad.tile(arr=np.max(actions*ur_ind,axis=  self.ad.arr_axis['n_arm'] ),axis_name='n_arm')

        if np.max(ur_bool) == 1:
            actions[ur_bool] = ur_actions[ur_bool]
        return actions

    def ts_adapt_explor(self, algo_para, action_hist, reward_hist, batch_size=1):
        # update posterior
        self.bayes_model.update_posterior(action_hist, reward_hist, self.ad.arr_axis)

        # sample posterior
        samples = self.bayes_model.get_posterior_sample(size=batch_size)
        sample_mean = np.moveaxis(samples['mean'],
                                  source=0,
                                  destination=self.ad.arr_axis['horizon'])

        sample_var = np.moveaxis(samples['var'],
                                  source=0,
                                  destination=self.ad.arr_axis['horizon'])

        pulled_var = np.var(np.sum(reward_hist, axis=self.ad.arr_axis['n_arm'],keepdims=True),axis=self.ad.arr_axis['horizon'],keepdims=True)

        sample_n = np.sum(action_hist,axis = self.ad.arr_axis['horizon'],keepdims=True)

        total_mean = np.sum(sample_n * sample_mean,axis=self.ad.arr_axis['n_arm'],keepdims=True)/np.sum(sample_n,axis=self.ad.arr_axis['n_arm'],keepdims=True)

        ss_effect = np.sum(sample_n*(sample_mean - total_mean)**2,axis=self.ad.arr_axis['n_arm'])

        ss_total = np.sum((sample_n-1)*pulled_var,axis=self.ad.arr_axis['n_arm']) + ss_effect

        eta2 = ss_effect/ss_total

        cohen_f = np.sqrt(eta2 / (1-eta2))

        ur_ind = self.ad.tile(arr=cohen_f<algo_para,
                              axis_name='n_arm')

        samples = np.moveaxis(self.bayes_model.get_posterior_sample(size=batch_size)['mean'], # re-sample (needed for PostDiff)
                              source=0,
                              destination=self.ad.arr_axis['horizon'])

        actions = (samples == np.max(samples, axis=self.ad.arr_axis['n_arm'], keepdims=True))  # TS actions
        if np.max(ur_ind) == 1:
            actions[ur_ind] = np.random.multinomial(1, np.ones(self.n_arm) / self.n_arm, size=cohen_f.shape)[ur_ind]
        return actions





class UCB:
    def __init__(self, delta, reward_model, bayes_model=None):
        self.delta = delta
        # self.bayes_model = bayes_model  # allow user input prior. Otherwise, use default prior
        self.reward_model = reward_model
        self.num_of_arms = reward_model.n_arms

    def get_action(self, action_hist, reward_hist,  batch_size=1,):
        rep_number = action_hist.shape[0]

        # sample arm
        action_count = np.sum(action_hist,axis=0, keepdims=True)
        reward_sum = np.sum(reward_hist,axis=0, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            stepwise_ucb = reward_sum/action_count + np.sqrt( 2*np.log(1/self.delta) / action_count )
        stepwise_ucb[action_count==0]=np.inf
        initial_actions = stepwise_ucb == np.max(stepwise_ucb, axis=2, keepdims=True) #there might be ties, so need to 'randomize' it a bit

        actions_break_tie = initial_actions + np.random.random(initial_actions.shape)

        return  actions_break_tie == np.max(actions_break_tie, axis=2, keepdims=True)