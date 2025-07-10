import numpy as np
import policy as pol

from scipy.stats import bernoulli, f
from scipy.stats import norm, f_oneway, studentized_range, t, distributions

from statsmodels.stats.multicomp import pairwise_tukeyhsd
#from joblib import Parallel, delayed
import pandas as pd
import os
import math
from multiprocessing import Pool
from policy import BanditAlgorithm

# import bayes_vector_ops as bayes
# from test_procedure_configurator import TestProcedure
from simulation_configurator import SimulationConfig

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import copy

from typing import Optional, Callable, Union, Literal, Type
import warnings


def generate_quadratic_schedule(max_horizon, tuning_density=1.0):
    """
    Generate a sequence of increasing integers up to (max_horizon - 1),
    with increasing step sizes but decreasing relative increments.

    Always includes max_horizon - 1.

    Args:
        max_horizon (int): Maximum horizon (exclusive upper limit).
        tuning_density (float): Controls density. Higher = denser.

    Returns:
        List[int]: The schedule.
    """
    schedule = []
    n = 1
    while True:
        x = int((n * tuning_density) ** 2)
        if x >= max_horizon - 1:
            break
        if len(schedule) == 0 or x > schedule[-1]:
            schedule.append(x)
        n += 1

    if (max_horizon - 1) not in schedule:
        schedule.append(max_horizon - 1)

    return schedule

def run_simulation(
    policy: BanditAlgorithm,
    sim_config: SimulationConfig,
) -> "SimResult":
    """
    The main function for running simulation.
    :param policy:
    :param algo_para: the parameter of the algorithm. For different algorithms, please check parameter definition in their comment
    :param sim_config:
    :param full_reward_trajectory:
    :return:
    """
    burn_in = sim_config.burn_in
    base_batch_size = sim_config.base_batch_size
    batch_scaling_rate = sim_config.batch_scaling_rate
    n_ap_rep = sim_config.n_ap_rep
    record_ap = sim_config.record_ap
    horizon = sim_config.horizon
    n_arm = sim_config.n_arm
    ad = sim_config.ad #TODO: remove...

    action_hist = np.zeros(ad.shape_arr, dtype=bool)
    reward_hist = np.zeros(ad.shape_arr)
    full_reward_trajectory = sim_config.full_reward_trajectory
    ap_hist = np.zeros(ad.shape_arr)

    time_step = 0

    # burn_in
    if burn_in > 0:
        #ad.slicing(horizon = slice(burn_in))
        slice_index = ad.slicing(horizon = slice(burn_in))
        #size = np.delete(np.array(action_hist[slice_index].shape),ad.arr_axis['n_arm'])
        arm_ind=-1
        for bt in range(burn_in):
            """
            TODO: negate: 'need modification to fit axis framework'
            """
            arm_ind+=1
            action_hist[:,bt,np.mod(arm_ind,n_arm)] = 1
        #action_hist[slice_index] = np.random.multinomial(1,np.ones(n_arm)/n_arm,size=size)

        reward_hist[slice_index] = full_reward_trajectory[slice_index] * action_hist[slice_index]
        if record_ap:
            ap_hist[slice_index] = 1 / n_arm
        time_step = burn_in

    while time_step < horizon:
        batch_size = math.floor(base_batch_size + time_step*batch_scaling_rate)
        if time_step + batch_size > horizon:
            batch_size = horizon - time_step

        slice_current = ad.slicing(horizon=slice(time_step))
        slice_next = ad.slicing(horizon=slice(time_step, time_step + batch_size))

        if record_ap:
            actions = policy.sample_action(sim_config,
                             action_hist[slice_current],
                             reward_hist[slice_current],
                             batch_size=n_ap_rep)
            ap = np.mean(actions,axis = ad.arr_axis['horizon'])  # dim = num_arm, rep
            ap_hist[slice_next] = ad.tile(arr = ap, axis_name='horizon',repeats=batch_size)

        action_hist[slice_next] = policy.sample_action(sim_config, action_hist[slice_current], reward_hist[slice_current],batch_size = batch_size)
        reward_hist[slice_next] = full_reward_trajectory[slice_next]*action_hist[slice_next]

        time_step = time_step + batch_size

    return SimResult(action_hist, reward_hist, sim_config, ap_hist=ap_hist)


# def art_replication(policy, algo_para, hyperparams, reward_hist):
#     burn_in = hyperparams.burn_in
#     horizon = hyperparams.horizon
#     batch_size = hyperparams.base_batch_size
#     #n_ap_rep = hyperparams.n_ap_rep
#     record_ap = hyperparams.record_ap
#     n_rep = hyperparams.n_rep
#     n_art_rep = hyperparams.n_art_rep
#
#     time_step = 0
#
#     bandit = policy.__self__
#     n_arm = bandit.n_arm
#     old_ad = bandit.ad
#     bandit.ad = ArrDim({'n_rep':1,'horizon':2,'n_arm':-1,'n_art_rep':0},hyperparams)
#     ad = bandit.ad
#
#     action_hist = np.zeros(ad.shape_arr, dtype=int)
#
#     #reward_hist = ad.tile(reward_hist, axis_name='n_art_rep')
#     arr = np.sum(reward_hist,axis = ad.arr_axis['n_arm'],keepdims=True)
#     arr = ad.tile(arr, axis_name = 'n_art_rep' )
#     reward_hist = np.repeat(arr, axis = ad.arr_axis['n_arm'], repeats = n_arm) #reward hist is the same for all arms in ART
#     #reward_hist = bandit.reward_model.sample(size=ad.shape_arr)
#     #ap_hist = np.zeros(ad.shape_arr)
#
#     # burn_in
#     if burn_in > 0:
#         # ad.slicing(horizon = slice(burn_in))
#         slice_index = ad.slicing(horizon=slice(burn_in))
#         size = np.delete(np.array(action_hist[slice_index].shape), ad.arr_axis['n_arm']) #everything but n_arm axis
#         action_hist[slice_index] = np.random.multinomial(1, np.ones(n_arm) / n_arm, size=size)
#         reward_hist[slice_index] = reward_hist[slice_index] * action_hist[slice_index]
#
#         time_step = burn_in
#
#     while time_step < horizon:
#         if time_step + batch_size > horizon:
#             batch_size = horizon - time_step
#             policy.base_batch_size = batch_size
#
#         slice_cur = ad.slicing(horizon=slice(time_step))
#         slice_nex = ad.slicing(horizon=slice(time_step, time_step + batch_size))
#
#         action_hist[slice_nex] = policy(algo_para, action_hist[slice_cur], reward_hist[slice_cur])
#         reward_hist[slice_nex] = reward_hist[slice_nex] * action_hist[slice_nex]
#
#         time_step = time_step + batch_size
#
#     bandit.ad = old_ad
#
#     return SimResult(action_hist, reward_hist, hyperparams, ad)


"""
                        Part 2  
               process simulation results      
"""

class SimResult:
    def __init__(self, action_hist, reward_hist, sim_config:SimulationConfig, ap_hist=None):
        """
        A class for storing and analyzing the results of bandit simulations.

        Upon initialization, this class computes a range of cumulative statistics
        (e.g., means, variances, counts) from the action and reward histories.

        It provides built-in methods to conduct various hypothesis tests
        (e.g., ANOVA, t-tests against a control or constant) to support downstream inference.

        Parameters:
        -----------
        action_hist : np.ndarray
            A multidimensional array indicating which arm was selected at each timestep
            (typically one-hot encoded).

        reward_hist : np.ndarray
            An array of observed binary rewards for each arm selection.

        hyperparams : Namespace or custom config object
            Configuration object containing parameters like the simulation horizon.

        ad : AxisDescriptor
            An object that maps named axis roles (e.g., 'n_arm', 'horizon') to integer axis indices
            for flexible array manipulation.

        ap_hist : np.ndarray, optional
            Array of posterior parameters or additional statistics recorded during the simulation.

        Attributes:
        -----------
        arm_means : np.ndarray
            Cumulative mean reward for each arm across time and repetitions.

        arm_vars : np.ndarray
            Estimated variance of the mean reward for each arm.

        combined_means : np.ndarray
            Pooled (across arms) mean reward over time.

        Methods:
        --------
        wald_test(arm1_index, arm2_index, horizon)
            Compare two arms using a Wald-type statistic.

        t_control(horizon)
            Compare all arms against a fixed control arm (arm 0).

        t_constant(constant_thres, horizon)
            Compare each arm's mean reward against a constant threshold.
        """
        self.ad = sim_config.ad
        self.n_arm = action_hist.shape[self.ad.arr_axis['n_arm']]
        self.n_rep = action_hist.shape[self.ad.arr_axis['n_rep']]

        self.tukey_matrix = None

        self.action_hist = action_hist
        self.reward_hist = reward_hist
        self.ap_hist = ap_hist
        self.horizon = sim_config.horizon
        self.total_counts = np.sum(np.cumsum(self.action_hist, axis=self.ad.arr_axis['horizon']), axis=self.ad.arr_axis['n_arm'], keepdims=True)

        self.reward_hist_flat = np.sum(self.reward_hist, axis=self.ad.arr_axis['n_arm'])
        self.action_hist_flat = np.argmax(self.action_hist, axis=self.ad.arr_axis['n_arm'])


        if len(self.ad.arr_axis) ==3:
            self.mean_reward = np.cumsum(
                np.mean(
                    np.sum(reward_hist, axis=self.ad.arr_axis['n_arm'], keepdims=True),
                    axis=self.ad.arr_axis['n_rep'], keepdims=True), axis=self.ad.arr_axis['horizon']
            ) / self.total_counts

        with np.errstate(divide='ignore', invalid='ignore'):
            self.arm_counts = np.cumsum(action_hist, axis=self.ad.arr_axis['horizon'])
            self.arm_cum_rewards = np.cumsum(reward_hist, axis=self.ad.arr_axis['horizon'])
            self.arm_means = self.arm_cum_rewards / self.arm_counts

            self.arm_square_cum_rewards = np.cumsum(reward_hist ** 2,
                                                    axis=self.ad.arr_axis['horizon'])  # for variance calculation
            self.arm_square_means = self.arm_square_cum_rewards / self.arm_counts
            self.arm_vars = ( (self.arm_square_means - self.arm_means ** 2) * (1/ (self.arm_counts - 1))) #var for arm mean! not arm reward!

            self.combined_means = np.cumsum(np.sum(reward_hist, axis=self.ad.arr_axis['n_arm'], keepdims=True),
                                            axis=self.ad.arr_axis['horizon']) / self.total_counts

            self.combined_square_cum_rewards = np.cumsum(np.sum(reward_hist, axis=self.ad.arr_axis['n_arm'], keepdims=True) ** 2, axis=self.ad.arr_axis['horizon'])  # for variance calculation
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

    def t_control(self, horizon = slice(-1,None)):
        """
        Compare all arms against the first arm (now we hard coded it, so the control must be the first arm)
        :param horizon:
        :return:
        """
        control_slice = self.ad.slicing(n_arm=slice(0,1), horizon=horizon)
        other_arm_slice = self.ad.slicing(n_arm=slice(1,None), horizon=horizon)

        cm_slice = self.ad.slicing(horizon=horizon)[0:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            walds = (self.arm_means[other_arm_slice] - self.arm_means[control_slice]) / np.sqrt(
                self.combined_means[cm_slice] * (1 - self.combined_means[cm_slice]) * (
                        1 / self.arm_counts[other_arm_slice] + 1 / self.arm_counts[control_slice])
            )
        return walds

    def t_constant(self, constant_threshold, horizon=slice(-1, None)):
        """
        Compare all arms against a user-specified constant threshold using a Wald-type statistic.

        :param constant_threshold: The constant value to compare each arm's estimated mean against.
        :param horizon: The time slice for evaluation (default is the last step only).
        :return: An array of Wald-type statistics for each arm.
        """
        arm_slice = self.ad.slicing(n_arm=slice(None), horizon=horizon)  # all arms
        cm_slice = self.ad.slicing(horizon=horizon)[0:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            walds = (self.arm_means[arm_slice] - constant_threshold) / np.sqrt(
                self.combined_means[cm_slice] * (1 - self.combined_means[cm_slice]) / self.arm_counts[arm_slice]
            )
        return walds

    def anova(self, horizon = slice(-1,None)):
        with np.errstate(divide='ignore', invalid='ignore'):
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
            p_value = f.cdf(F_stat, df_between, df_within)
        return -p_value[self.ad.slicing(horizon=horizon)] #return negative p-value so all test has right side critical region (easy to generalize)

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


def get_objective_score(h0_res:SimResult, h1_res:SimResult, sim_config:SimulationConfig):
    """
    Compute the final objective score, score SD, number of steps, and reward at that step.
    #TODO: a function that loop on each test
    #TODO: check and give warning if cost step is too low (and infinity is preferred , or just try max step

    #TODO: add it (flatten reward mean) directly in results

    Args:
        res_dist: result object with .anova(slice) and .mean_reward
        test_name: str, e.g., "anova"
        objective: dict defining constraints and weights
        h0_critical_values: dict[test_name] -> array of shape (mu, horizon)
        hyperparams: object with at least .n_rep and .horizon

    Returns:
        dict with:
            - obj_score: float
            - obj_score_sd: float
            - n_step: float (median)
            - reward: float (mean reward at that step)
    """

    # Step 1: Calculate power under H1
    power =  sim_config.test_procedure.compute_power(h0_sim_result=h0_res, h1_sim_result=h1_res)

    # Step 2: Determine minimum step that satisfies power constraint (with noise)
    power_constraint = sim_config.test_procedure.power_constraint
    n_rep = sim_config.n_rep
    horizon = sim_config.horizon

    noise = np.random.normal(
        loc=0, scale=np.sqrt(power_constraint * (1 - power_constraint) / n_rep), size=(1,n_rep)
    )

    # steps until constraint is exceeded
    n_step_dist = horizon - np.sum(power[:,np.newaxis] > (power_constraint + noise), axis=0)  # shape: (mu,)

    # Step 3: Compute reward at selected step
    mean_reward = np.mean(h1_res.mean_reward, axis=0).flatten()  # shape: (horizon,)
    reward_at_n_step = mean_reward[n_step_dist-1]

    n_step = np.median(n_step_dist)
    if n_step == horizon:
        # if exceed horizon_max, set reward = 0 as penalty
        warnings.warn("Power threshold may be too hard to achieve: n_step exceeds max horizon. ")
        reward_at_n_step = power[-1] - power_constraint

    # Step 4: Compute objective score
    obj_score_dist = (
        reward_at_n_step * n_step_dist +
        sim_config.step_cost * n_step_dist
    )

    return {
        "obj_score": np.mean(obj_score_dist),
        "obj_score_sd": np.std(obj_score_dist),
        "n_step": np.median(n_step_dist),
        "reward": mean_reward[int(np.median(n_step_dist-1))],
    }


# def run_simulation_ts(reward_model, policy, hyperparams, n_rep):
#     # stochastic_bandit_simulation(reward_model = construct_reward(model = np.random.binomial, parameters = {'n': [1,1], 'p': [0.6,0.4]}),policy = eps_ts(epsilon = 0.3))
#
#     policy.setup(reward_model=reward_model)
#     ts_policy = pol.EpsTS(0)
#     ts_policy.setup(reward_model=reward_model)
#
#     time_step = 0
#
#     burn_in = hyperparams.burn_in
#     horizon = hyperparams.horizon
#     batch_size = hyperparams.base_batch_size
#     n_ap_rep = hyperparams.n_ap_rep
#     record_ap = hyperparams.record_ap
#
#     n_arm = reward_model.n_arms
#
#     action_hist = np.zeros((horizon, n_rep, n_arm), dtype=int)
#     reward_hist = reward_model.sample(size=(horizon, n_rep, n_arm))
#     AP_hist = np.zeros((horizon, n_rep, n_arm))
#     ts_AP_hist = np.zeros((horizon, n_rep, n_arm))
#     # burn_in
#     if burn_in > 0:
#         action_hist[0:burn_in, :, :] = np.random.multinomial(1, np.ones(n_arm) / n_arm, size=(burn_in, n_rep))
#         reward_hist[0:burn_in, :, :] = reward_hist[0:burn_in, :, :] * action_hist[0:burn_in, :, :]
#         if record_ap:
#             AP_hist[0:burn_in, :, :] = 1 / n_arm
#         time_step = burn_in
#
#     while time_step < horizon:
#         if time_step + batch_size > horizon:
#             batch_size = horizon - time_step
#             policy.base_batch_size = batch_size
#         if record_ap:
#             actions = policy.get_action(action_hist[0:time_step, :, :],
#                                         reward_hist[0:time_step, :, :],
#                                         reward_model,
#                                         batch_size=n_ap_rep)
#             AP = np.mean(actions, axis=0)  # dim = num_arm, rep
#             AP_hist[time_step:(time_step + batch_size), :, :] = np.tile(AP, (batch_size, 1, 1))
#
#             ts_actions = ts_policy.get_action(action_hist[0:time_step, :, :],
#                                         reward_hist[0:time_step, :, :],
#                                         reward_model,
#                                         batch_size=n_ap_rep)
#             ts_AP = np.mean(ts_actions, axis=0)  # dim = num_arm, rep
#             ts_AP_hist[time_step:(time_step + batch_size), :, :] = np.tile(ts_AP, (batch_size, 1, 1))
#
#
#         action_hist[time_step:(time_step + batch_size), :, :] = policy.get_action(action_hist[0:time_step, :, :],
#                                                                                   reward_hist[0:time_step, :, :],
#                                                                                   reward_model)
#         reward_hist[time_step:(time_step + batch_size), :, :] = reward_hist[time_step:(time_step + batch_size), :,
#                                                                 :] * action_hist[time_step:(time_step + batch_size), :,
#                                                                      :]
#
#         time_step = time_step + batch_size
#
#     return (AP_hist, ts_AP_hist)



class AxObjectiveEvaluator:
    """
    wrapper for ax-optimization. result holder.
    input: all parameters except algorithm parameter (for which we run optimization loop)
    output: 1. obj score as direct output (for ax-opt); 2. hold other info in self.sim_result_keeper

    """

    def __init__(self, algo_class:Type[BanditAlgorithm], sim_config:SimulationConfig, sim_result_keeper):
        self.algo_class = algo_class
        self.sim_config = sim_config
        self.sim_result_keeper = sim_result_keeper

    def __call__(self, algo_param:float):
        self.sim_config.test_procedure.compute_power()
        h1_res = run_simulation(
            policy=self.algo_class(algo_param),
            sim_config=self.sim_config,
        )
        h0_sim_config = copy.deepcopy(self.sim_config)
        h0_sim_config.arm_mean_reward_dist_loc = h1_res.



        # H0 simulation
        h0_res = run_simulation()

        # Generate H1 reward histogram
        # base_shape = next(iter(self.h1_reward_dist.values())).shape
        # new_shape = (self.hyperparams.horizon,) + base_shape
        # h1_reward_hist = np.random.binomial(**self.h1_reward_dist, size=new_shape)
        # h1_reward_hist = np.moveaxis(h1_reward_hist, 0, 1)


        # alpha_dict,_ =opt_sim.decompose_objective_dict_for_test(objective)
        # Evaluate score
        result = opt_sim.get_objective_score(
            res_dist=res_dist,
            test_name=test_objective.test_name,
            objective=self.objective,
            h0_critical_values=h0_critical_values,
            hyperparams=self.hyperparams,
            h1_reward_dist=self.h1_reward_dist,
        )

        # Track result externally
        self.sim_result_keeper[(self.algo, algo_param, frozenset(self.config_setting.items()))] = result

        return {"obj_score": (result["obj_score"], result["obj_score_sd"])}

