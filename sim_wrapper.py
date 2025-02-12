import numpy as np
import policy as pol
from scipy.stats import norm
from scipy.stats import bernoulli
from scipy.stats import f_oneway
from scipy.stats import f, studentized_range
from scipy.stats import t, distributions
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#from joblib import Parallel, delayed
import pandas as pd
import os
import math
from multiprocessing import Pool
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import copy

class RewardModel:
    # Define reward model, and build function that draw reward based on action index/indices input
    def __init__(self, model, parameters):
        # input: model and parameters
        # e.g. model = np.random.binomial
        #      parameters = {'n': [1,1], 'p': [0.6,0.4]}
        # 'number of arms' is extracted from parameters input
        self.model = model
        self.parameters = parameters
        self.n_arm = len(next(iter(parameters.values())))

    def sample(self, size=1):
        # simply draw reward from all arms. not commonly used.
        return self.model(**self.parameters, size=size)



class ArrDim:
    # example: ad = ArrDim(arr_axis={'n_rep':0,'horizon':-2,'n_arm':-1},n_arm=6,n_rep=2,horizon=11)
    def __init__(self, arr_axis, hyperparams):

        self.total_dims = len(arr_axis)
        self.arr_axis = arr_axis
        shape_list = [None] * self.total_dims
        order_list = [None] * self.total_dims
        for key, pos in arr_axis.items():
            value = hyperparams[key]
            shape_list[pos] = value
            order_list[pos] = key
        self.shape_arr = np.array(shape_list)
        self.order_arr = np.array(order_list)

    def slicing(self, **dims):
        """
        dim_name = slice
        """
        # example: ad.slicing(n_arm=slice(6),horizon=slice(2,4))
        slice_list = [slice(None)] * self.total_dims
        for key, sli in dims.items():
            pos = self.arr_axis[key]
            slice_list[pos] = sli
        return tuple(slice_list)

    def match_dim(self, arr, arr_order):
        order_map = {dim: idx for idx, dim in enumerate(arr_order)}

        # Determine the new axes order
        new_axes = [order_map[dim] for dim in self.order_list]
        return np.transpose(arr, axes=new_axes)

    def tile(self, arr, axis_name, repeats = None):
        axis_ind = self.arr_axis[axis_name]
        expanded_arr = np.expand_dims(arr, axis=axis_ind)
        if repeats is None:
            stacked_arr = np.repeat(expanded_arr, repeats=self.shape_arr[axis_ind], axis=axis_ind)
        else:
            stacked_arr = np.repeat(expanded_arr, repeats=repeats, axis=axis_ind)
        return stacked_arr

    def sub_shape(self, exclude_list):
        indices = ~np.isin(self.order_arr, exclude_list)
        return self.shape_arr[indices]






def run_simulation(policy, algo_para, hyperparams,reward_hist=None):
    burn_in = hyperparams['burn_in']
    batch_size = hyperparams['batch_size']
    n_ap_rep = hyperparams['n_ap_rep']
    record_ap = hyperparams['record_ap']
    n_rep = hyperparams['n_rep']
    fast_batch_epsilon = hyperparams['fast_batch_epsilon']

    time_step = 0

    bandit = policy.__self__
    n_arm = bandit.n_arm

    if 'horizon_per_arm' in hyperparams.keys():
        horizon = hyperparams['horizon_per_arm'] * bandit.n_arm
        hyperparams['horizon'] = horizon
    else:
        horizon = hyperparams['horizon']




    hyperparams['n_arm'] = n_arm
    ad = ArrDim({'n_rep': 0, 'horizon': 1, 'n_arm': -1}, hyperparams)
    bandit.ad = ad


    action_hist = np.zeros(ad.shape_arr, dtype=bool)
    if reward_hist is None:
        reward_hist = bandit.reward_model.sample(size = ad.shape_arr)
    ap_hist = np.zeros(ad.shape_arr)

    # burn_in
    if burn_in > 0:
        #ad.slicing(horizon = slice(burn_in))
        slice_index = ad.slicing(horizon = slice(burn_in))
        size = np.delete(np.array(action_hist[slice_index].shape),ad.arr_axis['n_arm'])
        arm_ind=-1
        for bt in range(burn_in):
            """
            need modification to fit axis framework
            """
            arm_ind+=1
            action_hist[:,bt,np.mod(arm_ind,n_arm)] = 1
        #action_hist[slice_index] = np.random.multinomial(1,np.ones(n_arm)/n_arm,size=size)

        reward_hist[slice_index] = reward_hist[slice_index] * action_hist[slice_index]
        if record_ap:
            ap_hist[slice_index] = 1 / n_arm
        time_step = burn_in

    while time_step < horizon:
        if fast_batch_epsilon>0: #set it t be 0 if wants exactly step-by-step update
            batch_size = math.ceil(time_step*fast_batch_epsilon+0.01) #+0.01 to avoid batch = 0
        if time_step + batch_size > horizon:
            batch_size = horizon - time_step

        slice_cur = ad.slicing(horizon=slice(time_step))
        slice_nex = ad.slicing(horizon=slice(time_step, time_step + batch_size))

        if record_ap:
            actions = policy(algo_para,
                             action_hist[slice_cur],
                             reward_hist[slice_cur],
                             batch_size=n_ap_rep)
            ap = np.mean(actions,axis = ad.arr_axis['horizon'])  # dim = num_arm, rep
            ap_hist[slice_nex] = ad.tile(arr = ap, axis_name='horizon',repeats=batch_size)

        action_hist[slice_nex] = policy(algo_para, action_hist[slice_cur], reward_hist[slice_cur],batch_size = batch_size)
        reward_hist[slice_nex] = reward_hist[slice_nex]*action_hist[slice_nex]

        time_step = time_step + batch_size

    return SimResult(action_hist, reward_hist, hyperparams, ad, ap_hist=ap_hist)


def art_replication(policy, algo_para, hyperparams, reward_hist):
    burn_in = hyperparams['burn_in']
    horizon = hyperparams['horizon']
    batch_size = hyperparams['batch_size']
    #n_ap_rep = hyperparams['n_ap_rep']
    record_ap = hyperparams['record_ap']
    n_rep = hyperparams['n_rep']
    n_art_rep = hyperparams['n_art_rep']

    time_step = 0

    bandit = policy.__self__
    n_arm = bandit.n_arm
    old_ad = bandit.ad
    bandit.ad = ArrDim({'n_rep':1,'horizon':2,'n_arm':-1,'n_art_rep':0},hyperparams)
    ad = bandit.ad

    action_hist = np.zeros(ad.shape_arr, dtype=int)

    #reward_hist = ad.tile(reward_hist, axis_name='n_art_rep')
    arr = np.sum(reward_hist,axis = ad.arr_axis['n_arm'],keepdims=True)
    arr = ad.tile(arr, axis_name = 'n_art_rep' )
    reward_hist = np.repeat(arr, axis = ad.arr_axis['n_arm'], repeats = n_arm) #reward hist is the same for all arms in ART
    #reward_hist = bandit.reward_model.sample(size=ad.shape_arr)
    #ap_hist = np.zeros(ad.shape_arr)

    # burn_in
    if burn_in > 0:
        # ad.slicing(horizon = slice(burn_in))
        slice_index = ad.slicing(horizon=slice(burn_in))
        size = np.delete(np.array(action_hist[slice_index].shape), ad.arr_axis['n_arm']) #everything but n_arm axis
        action_hist[slice_index] = np.random.multinomial(1, np.ones(n_arm) / n_arm, size=size)
        reward_hist[slice_index] = reward_hist[slice_index] * action_hist[slice_index]

        time_step = burn_in

    while time_step < horizon:
        if time_step + batch_size > horizon:
            batch_size = horizon - time_step
            policy.batch_size = batch_size

        slice_cur = ad.slicing(horizon=slice(time_step))
        slice_nex = ad.slicing(horizon=slice(time_step, time_step + batch_size))

        action_hist[slice_nex] = policy(algo_para, action_hist[slice_cur], reward_hist[slice_cur])
        reward_hist[slice_nex] = reward_hist[slice_nex] * action_hist[slice_nex]

        time_step = time_step + batch_size

    bandit.ad = old_ad

    return SimResult(action_hist, reward_hist, hyperparams, ad)


"""
                        Part 2  
               process simulation results      
"""

class SimResult:
    def __init__(self, action_hist, reward_hist, hyperparams, ad, ap_hist=None):
        self.ad = ad
        self.n_arm = action_hist.shape[self.ad.arr_axis['n_arm']]
        self.n_rep = action_hist.shape[self.ad.arr_axis['n_rep']]

        self.tukey_matrix = None

        self.action_hist = action_hist
        self.reward_hist = reward_hist
        self.ap_hist = ap_hist
        self.horizon = hyperparams['horizon']
        self.total_counts = np.sum(np.cumsum(self.action_hist, axis=self.ad.arr_axis['horizon']), axis=self.ad.arr_axis['n_arm'], keepdims=True)

        self.reward_hist_flat = np.sum(self.reward_hist, axis=self.ad.arr_axis['n_arm'])
        self.action_hist_flat = np.argmax(self.action_hist, axis=self.ad.arr_axis['n_arm'])


        if len(ad.arr_axis) ==3:
            self.mean_reward = np.cumsum(
                np.mean(
                    np.sum(reward_hist, axis=ad.arr_axis['n_arm'], keepdims=True),
                    axis=ad.arr_axis['n_rep'], keepdims=True), axis=ad.arr_axis['horizon']
            ) / self.total_counts

        with np.errstate(divide='ignore', invalid='ignore'):
            self.arm_counts = np.cumsum(action_hist, axis=ad.arr_axis['horizon'])
            self.arm_cum_rewards = np.cumsum(reward_hist, axis=ad.arr_axis['horizon'])
            self.arm_means = self.arm_cum_rewards / self.arm_counts

            self.arm_square_cum_rewards = np.cumsum(reward_hist ** 2,
                                                    axis=ad.arr_axis['horizon'])  # for variance calculation
            self.arm_square_means = self.arm_square_cum_rewards / self.arm_counts
            self.arm_vars = ( (self.arm_square_means - self.arm_means ** 2) * (1/ (self.arm_counts - 1))) #var for arm mean! not arm reward!

            self.combined_means = np.cumsum(np.sum(reward_hist, axis=ad.arr_axis['n_arm'], keepdims=True),
                                            axis=ad.arr_axis['horizon']) / self.total_counts

            self.combined_square_cum_rewards = np.cumsum(np.sum(reward_hist, axis=ad.arr_axis['n_arm'], keepdims=True) ** 2, axis=ad.arr_axis['horizon'])  # for variance calculation
            self.combined_square_means = self.combined_square_cum_rewards / self.total_counts
            self.combined_vars = ((self.combined_square_means - self.combined_means ** 2) * (1 / (self.total_counts - 1)))  # var for arm mean! not arm reward!
            self.combined_reward_vars = (self.combined_square_means - self.combined_means ** 2)

    def wald_test(self, arm1_index=0, arm2_index=1,horizon = slice(-1,None)):
        arm1_slice = self.ad.slicing(n_arm=slice(arm1_index,arm1_index+1), horizon=horizon)
        arm2_slice = self.ad.slicing(n_arm=slice(arm2_index,arm2_index+1), horizon=horizon)

        cm_slice = self.ad.slicing(horizon=horizon)[0:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            walds = (self.arm_means[arm1_slice] - self.arm_means[arm2_slice]) / np.sqrt(
                self.combined_means[cm_slice] * (1 - self.combined_means[cm_slice]) * (1 / (self.arm_counts[arm1_slice]) + 1/ (self.arm_counts[arm2_slice]))
            )
        return walds

    def anova(self, horizon = slice(-1,None)):

        variances = self.arm_vars *  (self.arm_counts - 1)

        # Number of groups
        K = self.n_arm

        # Total number of samples
        total_n = self.total_counts

        # Grand mean
        grand_mean = self.combined_means

        # Between-group sum of squares (SSB)
        ssb = np.sum(self.arm_counts * (self.arm_means - grand_mean) ** 2, axis = self.ad.arr_axis['n_arm'],keepdims=True)

        # Within-group sum of squares (SSW)
        ssw = np.sum((self.arm_counts - 1) * variances, axis = self.ad.arr_axis['n_arm'],keepdims=True)

        # Between-group mean square (MSB)
        msb = ssb / (K - 1)

        # Within-group mean square (MSW)
        msw = ssw / (total_n - K)

        # F-statistic
        F_stat = msb / msw

        # Degrees of freedom
        df_between = K - 1
        df_within = total_n - K

        # p-value
        p_value = 1 - f.cdf(F_stat, df_between, df_within)
        return p_value[self.ad.slicing(horizon=horizon)]

    def tukey_single(self,rep_slice,horizon_slice):

        """
        archived

        :param rep_slice:
        :param horizon_slice:
        :return:
        """
        sli = np.array(self.ad.slicing(n_rep=rep_slice,horizon=horizon_slice))
        sli = tuple(sli[np.arange(self.ad.total_dims)[self.ad.order_arr != 'n_arm']])
        if np.var(self.reward_hist_flat[sli])==0:
            return {'arm_decision': np.random.random(self.n_arm),
                    'reject':0} #return a random action

        else:
            tukey = pairwise_tukeyhsd(endog=self.reward_hist_flat[sli],
                                      groups=self.action_hist_flat[sli],
                                      alpha=0.05)

            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

            if self.tukey_matrix is None:
                self.tukey_matrix = np.zeros((self.n_arm, tukey_df.shape[0]))
                for i in range(self.n_arm):
                    self.tukey_matrix[i, :] = ((tukey_df['group2'] == i) * 1 - (tukey_df['group1'] == i)) * 1
            test_df = self.tukey_matrix * np.array(np.sign(tukey_df['meandiff']) * (tukey_df['reject']))

            return {'arm_decision': np.argmax(np.sum(test_df, axis=1) + np.random.random(self.n_arm)),
                    'reject':np.mean(tukey_df['reject'])}  # add random to break tie randomly


    def tukey(self, horizon = slice(200,-1,100)):

        """
        archived chode below
        self.tukey_single(1,slice(0,100))
        np.random.seed(1)
        with Parallel(n_jobs=-1) as parallel:
            results_parallel = parallel(delayed(self.tukey_single)(rep, slice(0,step)) for rep in range(self.n_rep) for step in range(self.horizon)[horizon_index])

        :param horizon_index:
        :return:
        """
        #
        horizon_steps = np.arange(self.horizon)[horizon]


        """
        also need modification for arr_axis
        """

        group_means = self.arm_means[:,horizon,:]  # Shape: (n_groups, n_replications)


        # Step 2: Calculate pooled standard deviation
        group_variances = self.arm_vars[:,horizon,:]  # Variance for each group
        pooled_var = (self.combined_vars * self.total_counts)[:,horizon,:]
        arm_weights = 1 / (self.arm_counts[:, horizon, :]-1)
        pooled_std = np.sqrt(pooled_var)[..., :, np.newaxis]  # Shape: (n_replications,)

        # Step 3: Compute pairwise mean differences and standard errors

        mean_diffs = group_means[..., :, np.newaxis] - group_means[..., np.newaxis, :]  # Shape: (n_groups, n_groups, n_replications)
        sum_arm_weights = arm_weights[..., :, np.newaxis] + arm_weights[..., np.newaxis, :]

        #triu_indices = np.triu_indices(self.n_arm, k=1)
        #mean_diffs = mean_diffs[..., triu_indices[0], triu_indices[1]]

        # Step 4: Compute Tukey HSD statistic
        #note: the statistic need to be multiplied by sqrt(2). See https://en.wikipedia.org/wiki/Tukey%27s_range_test
        hsd_stat = np.abs(mean_diffs) / (pooled_std*np.sqrt(sum_arm_weights))*np.sqrt(2)  # Shape: (n_groups, n_groups, n_replications)

        # Step 5: Calculate the critical value from the Studentized range distribution
        #upper_critical = studentized_range.interval(0.9, self.n_arm, (horizon_steps - self.n_arm - 1))[1]  # Scalar critical value

        # Step 6: Determine significant differences
        #significant_pairs = hsd_stat > upper_critical[np.newaxis,:,np.newaxis, np.newaxis]

        #return {'arm_decision': np.argmax(np.sum(significant_pairs*(mean_diffs>0),axis = -1)+
        #                                  np.random.random(arm_weights.shape),axis=-1), # add random to break tie randomly
        #        'reject_rate': np.sum(significant_pairs,axis=(-1,-2))/self.n_arm/(self.n_arm-1)}

        return hsd_stat



    def wald_test_normal(self, arm1_index=0, arm2_index=1, horizon = slice(-1,None)):
        arm1_slice = self.ad.slicing(n_arm=arm1_index, horizon=horizon)
        arm2_slice = self.ad.slicing(n_arm=arm2_index, horizon=horizon)
        cm_slice = self.ad.slicing(horizon=horizon)[0:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            walds = (self.arm_means[arm1_slice] - self.arm_means[arm2_slice]) / np.sqrt(self.arm_vars[arm1_slice]+self.arm_vars[arm2_slice])
        return walds

    def t_test(self, test_bar, horizon = slice(-1,None)):
        slice_arr = self.ad.slicing(horizon=horizon)
        return (self.arm_means[slice_arr] - test_bar) / np.sqrt(self.arm_vars[slice_arr])

    def LRT(self,horizon = slice(-1,None), dist = bernoulli):
        sli = self.ad.slicing(horizon=horizon)

        p_hat_H0 = self.combined_means[sli[0:-1]] #assume arm is the last dim
        p_hat_H1 = self.arm_means[sli]

        L0 = np.sum(np.log(dist.pmf(np.sum(self.reward_hist,axis = self.ad.arr_axis['n_arm']), p_hat_H0)), axis=-1)
        L1 = np.sum(np.log(dist.pmf(self.reward_hist, p_hat_H1))*self.action_hist,
                    axis = (self.ad.arr_axis['n_arm'],self.ad.arr_axis['horizon']) )

        return -2*(L0-L1)








def run_simulation_ts(reward_model, policy, hyperparams, n_rep):
    # stochastic_bandit_simulation(reward_model = construct_reward(model = np.random.binomial, parameters = {'n': [1,1], 'p': [0.6,0.4]}),policy = eps_ts(epsilon = 0.3))

    policy.setup(reward_model=reward_model)
    ts_policy = pol.EpsTS(0)
    ts_policy.setup(reward_model=reward_model)

    time_step = 0

    burn_in = hyperparams['burn_in']
    horizon = hyperparams['horizon']
    batch_size = hyperparams['batch_size']
    n_ap_rep = hyperparams['n_ap_rep']
    record_ap = hyperparams['record_ap']

    n_arm = reward_model.n_arms

    action_hist = np.zeros((horizon, n_rep, n_arm), dtype=int)
    reward_hist = reward_model.sample(size=(horizon, n_rep, n_arm))
    AP_hist = np.zeros((horizon, n_rep, n_arm))
    ts_AP_hist = np.zeros((horizon, n_rep, n_arm))
    # burn_in
    if burn_in > 0:
        action_hist[0:burn_in, :, :] = np.random.multinomial(1, np.ones(n_arm) / n_arm, size=(burn_in, n_rep))
        reward_hist[0:burn_in, :, :] = reward_hist[0:burn_in, :, :] * action_hist[0:burn_in, :, :]
        if record_ap:
            AP_hist[0:burn_in, :, :] = 1 / n_arm
        time_step = burn_in

    while time_step < horizon:
        if time_step + batch_size > horizon:
            batch_size = horizon - time_step
            policy.batch_size = batch_size
        if record_ap:
            actions = policy.get_action(action_hist[0:time_step, :, :],
                                        reward_hist[0:time_step, :, :],
                                        reward_model,
                                        batch_size=n_ap_rep)
            AP = np.mean(actions, axis=0)  # dim = num_arm, rep
            AP_hist[time_step:(time_step + batch_size), :, :] = np.tile(AP, (batch_size, 1, 1))

            ts_actions = ts_policy.get_action(action_hist[0:time_step, :, :],
                                        reward_hist[0:time_step, :, :],
                                        reward_model,
                                        batch_size=n_ap_rep)
            ts_AP = np.mean(ts_actions, axis=0)  # dim = num_arm, rep
            ts_AP_hist[time_step:(time_step + batch_size), :, :] = np.tile(ts_AP, (batch_size, 1, 1))


        action_hist[time_step:(time_step + batch_size), :, :] = policy.get_action(action_hist[0:time_step, :, :],
                                                                                  reward_hist[0:time_step, :, :],
                                                                                  reward_model)
        reward_hist[time_step:(time_step + batch_size), :, :] = reward_hist[time_step:(time_step + batch_size), :,
                                                                :] * action_hist[time_step:(time_step + batch_size), :,
                                                                     :]

        time_step = time_step + batch_size

    return (AP_hist, ts_AP_hist)


