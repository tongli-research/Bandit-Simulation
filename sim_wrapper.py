from functools import cached_property

import numpy as np
from sympy.codegen.ast import Raise

from scipy.stats import bernoulli, f

import os
from ax.service.managed_loop import optimize
from bandit_algorithm import BanditAlgorithm

from simulation_configurator import SimulationConfig

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from typing import Optional, Callable, Union, Literal, Type, Dict, Any
import warnings

"""
Table of Content:
run_simulation 
    main function
    
    
Note:
currently have schedule. But also can use 'get_interpolation' to scale back.

need to better design AxObjectiveEvaluator (what is the sandard input?
what should be one or multiple default input
can it take optimization iter?
I think we have differnt mode:
(is schedule a mode? can we have a standard process to transform it back? 
can it be within ResSim?)
no-test mode
with test, we have:
no iteration (single sim
loop of sim (without iter?
loop of sim (deterministic on each iter)
loop on a optimization (adaptive deciiding next point,used for ???
do we need the optimization?
even in reality??



what do we do now:
use input their arm mean and var, and test etc
what we do?
run a loop 
(can they change test later?? or like can we do multiple test in a single loop??
    


Edit Sept 14th:
comment off 'generate_quadratic_schedule' (seems it is not in used. we have other
ways to generate such schedule? where? confirm that this is true


"""

# def generate_quadratic_schedule(max_horizon, tuning_density=1.0):
#     """
#     Generate a sequence of increasing integers up to (max_horizon - 1),
#     with increasing step sizes but decreasing relative increments.
#
#     Always includes max_horizon - 1.
#
#     Args:
#         max_horizon (int): Maximum horizon (exclusive upper limit).
#         tuning_density (float): Controls density. Higher = denser.
#
#     Returns:
#         List[int]: The schedule.
#     """
#     schedule = []
#     n = 1
#     while True:
#         x = int((n * tuning_density) ** 2)
#         if x >= max_horizon - 1:
#             break
#         if len(schedule) == 0 or x > schedule[-1]:
#             schedule.append(x)
#         n += 1
#
#     if (max_horizon - 1) not in schedule:
#         schedule.append(max_horizon - 1)
#
#     return schedule

def run_simulation(
    policy: BanditAlgorithm,
    sim_config: SimulationConfig,
    arm_mean_reward_dist = None,
) -> "SimResult":
    """
    The main function for running simulation.
    :param policy:
    :param algo_para: the parameter of the algorithm. For different algorithms, please check parameter definition in their comment
    :param sim_config:
    :param full_reward_trajectory:
    :return:
    """
    burn_in_per_arm = sim_config.burn_in_per_arm #15 per arm?
    base_batch_size = sim_config.base_batch_size
    batch_scaling_rate = sim_config.batch_scaling_rate
    n_ap_rep = sim_config.n_ap_rep
    record_ap = sim_config.record_ap
    horizon = sim_config.horizon
    n_arm = sim_config.n_arm
    ad = sim_config.ad #TODO: remove...

    sample_batch_schedule = sim_config.sample_batch_schedule
    step_schedule = sim_config.step_schedule

    action_hist = np.zeros(ad.shape_arr).astype(int)
    reward_hist = np.zeros(ad.shape_arr)
    reward2_hist = np.zeros(ad.shape_arr)
    ap_hist = np.zeros(ad.shape_arr)

    if arm_mean_reward_dist is None:
        full_reward_trajectory, full_reward2_trajectory = sim_config.generate_full_reward_trajectory()
    else:
        full_reward_trajectory, full_reward2_trajectory = sim_config.generate_full_reward_trajectory(arm_mean_reward_dist)


    time_step = 0


    # burn_in_per_arm
    if burn_in_per_arm > 0:
        #ad.slicing(horizon = slice(burn_in_per_arm))
        #slice_index = ad.slicing(horizon = slice(burn_in_per_arm))
        #size = np.delete(np.array(action_hist[slice_index].shape),ad.arr_axis['n_arm'])
        if sim_config.compact_array:
            action_hist[:,0,:] = sample_batch_schedule[time_step]
            time_step = 1
            total_action_samples = n_arm
            reward_hist[:,0:1,:] = full_reward_trajectory[:,0:1,:]
            reward2_hist[:, 0:1, :] = full_reward2_trajectory[:, 0:1, :]

        else:
            Raise(NotImplementedError)
        # arm_ind=-1
        # burn_in_time_step = 0
        # while burn_in_time_step < burn_in_per_arm*n_arm:
        #
        # for bt in range(burn_in_per_arm):
        #     """
        #     TODO: negate: 'need modification to fit axis framework'
        #     """
        #     arm_ind+=1
        #     action_hist[:,bt,np.mod(arm_ind,n_arm)] = 1
        # #action_hist[slice_index] = np.random.multinomial(1,np.ones(n_arm)/n_arm,size=size)
        #
        # reward_hist[slice_index] = full_reward_trajectory[slice_index] * action_hist[slice_index]
        # if record_ap:
        #     ap_hist[slice_index] = 1 / n_arm
        # time_step = burn_in_per_arm

    while time_step < len(step_schedule):
        np.random.seed(time_step+1)
        # if time_step > 47:
        #     x=1
        batch_size = sample_batch_schedule[time_step]

        slice_current = ad.slicing(horizon=slice(time_step))
        slice_next = ad.slicing(horizon=slice(time_step, time_step + batch_size))

        if record_ap:
            actions = policy.sample_action(sim_config,
                             action_hist[slice_current],
                             reward_hist[slice_current],
                             batch_size=n_ap_rep)
            ap = np.mean(actions,axis = ad.arr_axis['horizon'])  # dim = num_arm, rep
            ap_hist[slice_next] = ad.tile(arr = ap, axis_name='horizon',repeats=batch_size)

        if sim_config.compact_array:
            number_of_action_samples = round(step_schedule[time_step]/sample_batch_schedule[time_step])

            action_sample = policy.sample_action(
                sim_config,
                action_hist[:,:time_step,:],
                reward_hist[:,:time_step,:],
                batch_size = number_of_action_samples,)
            reward_sample = action_sample * full_reward_trajectory[:,total_action_samples:(total_action_samples+number_of_action_samples),:]
            reward2_sample = action_sample * full_reward2_trajectory[:,total_action_samples:(total_action_samples + number_of_action_samples), :]

            action_hist[:, time_step:(time_step + 1), :] = sample_batch_schedule[time_step] * np.sum(action_sample, axis=1, keepdims=True, )
            reward_hist[:,time_step:(time_step+1),:] = np.sum(reward_sample,axis=1,keepdims=True)
            reward2_hist[:, time_step:(time_step + 1), :] = np.sum(reward2_sample, axis=1, keepdims=True)

            total_action_samples += number_of_action_samples
        else:
            action_hist[slice_next] = policy.sample_action(sim_config, action_hist[slice_current], reward_hist[slice_current],batch_size = batch_size)
            reward_hist[slice_next] = full_reward_trajectory[slice_next]*action_hist[slice_next]

        time_step += 1


    return SimResult(action_hist.astype(int), reward_hist, reward2_hist, sim_config, ap_hist=ap_hist)


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
    def __init__(self, action_hist, reward_hist, reward2_hist, sim_config:SimulationConfig, ap_hist=None):
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
        #self.tukey_matrix = None
        self.action_hist = action_hist
        self.reward_hist = reward_hist
        self.reward2_hist = reward2_hist
        self.ap_hist = ap_hist
        self.horizon = sim_config.horizon


        #TODO: check here. seems total count is 1,2,...,N and duplicated. Also check mean_reward. document them...
        self.total_counts = np.sum(np.cumsum(self.action_hist, axis=self.ad.arr_axis['horizon']), axis=self.ad.arr_axis['n_arm'], keepdims=True)

        # self.reward_hist_flat = np.sum(self.reward_hist, axis=self.ad.arr_axis['n_arm'])
        # self.action_hist_flat = np.argmax(self.action_hist, axis=self.ad.arr_axis['n_arm'])
        # if len(self.ad.arr_axis) ==3:
        #     self.mean_reward = np.cumsum(
        #         np.mean(
        #             np.sum(reward_hist, axis=self.ad.arr_axis['n_arm'], keepdims=True),
        #             axis=self.ad.arr_axis['n_rep'], keepdims=True), axis=self.ad.arr_axis['horizon']
        #     ) / self.total_counts

        with np.errstate(divide='ignore', invalid='ignore'):
            self.arm_counts = np.cumsum(action_hist, axis=self.ad.arr_axis['horizon'])
            self.arm_means = np.cumsum(reward_hist, axis=self.ad.arr_axis['horizon']) / self.arm_counts
            self.combined_means = np.cumsum(np.sum(reward_hist, axis=self.ad.arr_axis['n_arm'], keepdims=True),
                                            axis=self.ad.arr_axis['horizon']) / self.total_counts
            #self.combined_reward_vars = (self.combined_square_means - self.combined_means ** 2)

    @cached_property
    def combined_vars(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            combined_square_cum_rewards = np.cumsum(np.sum(self.reward2_hist, axis=self.ad.arr_axis['n_arm'], keepdims=True),
                                                    axis=self.ad.arr_axis['horizon'])  # for variance calculation
            combined_square_means = combined_square_cum_rewards / self.total_counts
            combined_vars = ((combined_square_means - self.combined_means ** 2) * (
                        1 / (self.total_counts - 1)))  # var for arm mean! not arm reward!
        return combined_vars

    @cached_property
    def arm_vars(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            arm_square_cum_rewards = np.cumsum(self.reward2_hist, axis=self.ad.arr_axis['horizon'])  # for variance calculation
            arm_square_means = arm_square_cum_rewards / self.arm_counts
            arm_vars = ((arm_square_means - self.arm_means ** 2) * (
                        1 / (self.arm_counts - 1)))  # var for arm mean! not arm reward!
        return arm_vars
    def wald_test(self, arm1_index=0, arm2_index=1,horizon = slice(-1,None)):
        arm1_slice = self.ad.slicing(n_arm=slice(arm1_index,arm1_index+1), horizon=horizon)
        arm2_slice = self.ad.slicing(n_arm=slice(arm2_index,arm2_index+1), horizon=horizon)

        cm_slice = self.ad.slicing(horizon=horizon)[0:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            walds = (self.arm_means[arm1_slice] - self.arm_means[arm2_slice]) / np.sqrt(
                self.combined_means[cm_slice] * (1 - self.combined_means[cm_slice]) * (1 / (self.arm_counts[arm1_slice]) + 1/ (self.arm_counts[arm2_slice]))
            )
        return walds

    def t_control(self, horizon = slice(-1,None),permutation_test=False,permutation_rep=100):
        """
        Compare all arms against the first arm (now we hard coded it, so the control must be the first arm)
        :param horizon:
        :return:
        """

        if permutation_test:
            arm_cum_reward = self.arm_counts * self.arm_means
            n_good = (arm_cum_reward[:,:,0:1] + arm_cum_reward[:,:,1:]).astype(int)
            n_bad = (self.arm_counts[:,:,0:1] + self.arm_counts[:,:,1:] - n_good).astype(int)

            count = np.zeros_like(arm_cum_reward[..., 1:], dtype=float)
            for i in range(10):
                permutation_samples = np.random.hypergeometric(
                    ngood=n_good,
                    nbad=n_bad,
                    nsample=self.arm_counts[:,:,0:1],
                    size=(permutation_rep,)+n_good.shape
                )
                count += np.mean(permutation_samples > arm_cum_reward[np.newaxis,:,:,0:1],axis = 0)

            test_stats = count/10

        else:
            control_slice = self.ad.slicing(n_arm=slice(0, 1), horizon=horizon)
            other_arm_slice = self.ad.slicing(n_arm=slice(1, None), horizon=horizon)

            cm_slice = self.ad.slicing(horizon=horizon)[0:-1]

            with np.errstate(divide='ignore', invalid='ignore'):
                test_stats = (self.arm_means[other_arm_slice] - self.arm_means[control_slice]) / np.sqrt(
                    self.combined_means[cm_slice] * (1 - self.combined_means[cm_slice]) * (
                            1 / self.arm_counts[other_arm_slice] + 1 / self.arm_counts[control_slice])
                )


        return test_stats

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
            p_value = 1 - f.cdf(F_stat, df_between, df_within)
        return p_value[self.ad.slicing(horizon=horizon)] #return negative p-value so all test has right side critical region (easy to generalize)

    # def tukey_single(self,rep_slice,horizon_slice):
    #
    #     """
    #     archived
    #
    #     :param rep_slice:
    #     :param horizon_slice:
    #     :return:
    #     """
    #     sli = np.array(self.ad.slicing(n_rep=rep_slice,horizon=horizon_slice))
    #     sli = tuple(sli[np.arange(self.ad.total_dims)[self.ad.order_arr != 'n_arm']])
    #     if np.var(self.reward_hist_flat[sli])==0:
    #         return {'arm_decision': np.random.random(self.n_arm),
    #                 'reject':0} #return a random action
    #
    #     else:
    #         tukey = pairwise_tukeyhsd(endog=self.reward_hist_flat[sli],
    #                                   groups=self.action_hist_flat[sli],
    #                                   alpha=0.05)
    #
    #         tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    #
    #         if self.tukey_matrix is None:
    #             self.tukey_matrix = np.zeros((self.n_arm, tukey_df.shape[0]))
    #             for i in range(self.n_arm):
    #                 self.tukey_matrix[i, :] = ((tukey_df['group2'] == i) * 1 - (tukey_df['group1'] == i)) * 1
    #         test_df = self.tukey_matrix * np.array(np.sign(tukey_df['meandiff']) * (tukey_df['reject']))
    #
    #         return {'arm_decision': np.argmax(np.sum(test_df, axis=1) + np.random.random(self.n_arm)),
    #                 'reject':np.mean(tukey_df['reject'])}  # add random to break tie randomly

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
        #horizon_steps = np.arange(self.horizon)[horizon]


        """
        also need modification for arr_axis
        """

        group_means = self.arm_means[:,horizon,:]  # Shape: (n_groups, n_replications)


        # Step 2: Calculate pooled standard deviation
        #group_variances = self.arm_vars[:,horizon,:]  # Variance for each group
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
        with np.errstate(divide='ignore', invalid='ignore'):
            hsd_stat = mean_diffs / (pooled_std*np.sqrt(sum_arm_weights))*np.sqrt(2)  # Shape: (n_groups, n_groups, n_replications)

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


def get_interpolation(arr: np.ndarray, step_schedule: np.ndarray) -> np.ndarray:
    """
    Linearly interpolates values in `arr` across sample counts defined by `step_schedule`.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)
        Values to interpolate between (e.g., power at each step).
    step_schedule : np.ndarray of shape (n,)
        Number of samples added at each step (defines the spacing for interpolation).

    Returns
    -------
    interpolated : np.ndarray of shape (sum(step_schedule),)
        Interpolated values, assuming linear trend between arr[i] and arr[i+1].
    """
    total_samples = np.sum(step_schedule)
    interpolated = np.empty(total_samples, dtype=float)

    cursor = 0

    # First segment: flat (constant) at arr[0]
    interpolated[:step_schedule[0]] = arr[0]
    cursor += step_schedule[0]

    # Remaining: interpolate from arr[i] to arr[i+1]
    for i in range(1, len(arr)):
        n = step_schedule[i]
        start = arr[i-1]
        end = arr[i]
        interpolated[cursor:cursor + n] = np.linspace(start, end, n, endpoint=False)
        cursor += n

    return interpolated

def get_objective_score(crit_boundary:np.ndarray, h1_res:SimResult, sim_config:SimulationConfig):
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
    power =  sim_config.test_procedure.compute_power(
        crit_boundary=crit_boundary,
        h1_sim_result=h1_res,
        ground_truth_arm_mean_dist=sim_config.arm_mean_reward_dist
    ) #TODO: check if h0 sim is correct
    power = get_interpolation(power,sim_config.step_schedule)
    # Step 2: Determine minimum step that satisfies power constraint (with noise)
    power_constraint = sim_config.test_procedure.power_constraint
    n_rep = sim_config.n_rep
    horizon = sim_config.horizon

    noise = np.random.normal(
        loc=0, scale=np.sqrt(power_constraint * (1 - power_constraint) / n_rep), size=(1,n_rep)
    )

    # steps until constraint is exceeded
    n_step_dist = horizon - np.sum(power[:,np.newaxis] > (power_constraint + noise), axis=0)  # shape: (mu,)

    true_means = sim_config.arm_mean_reward_dist
    best_mean = np.max(true_means, axis=1)
    # Step 3: Compute reward at selected step
    if sim_config.reward_evaluation_method == 'reward':
        mean_reward = np.mean(h1_res.combined_means, axis=0).flatten()
        mean_reward =  get_interpolation(mean_reward, sim_config.step_schedule)# shape: (horizon,)
    elif sim_config.reward_evaluation_method == 'regret':
        # selected_means = np.sum( (h1_res.action_hist>0) * true_means[:, np.newaxis, :], axis=2)
        # regret = np.mean(best_mean[:, np.newaxis] - selected_means,axis=0)
        step_wise_regret = np.mean(np.sum((best_mean[:, np.newaxis] - true_means)[:,np.newaxis,:]*h1_res.action_hist,axis=2),axis=0)/sim_config.step_schedule
        step_wise_regret = get_interpolation(step_wise_regret, sim_config.step_schedule)
        cumulative_regret = np.cumsum(step_wise_regret)
        mean_reward = (cumulative_regret / np.arange(1, horizon + 1)).flatten() #TODO: change reward name to regret
    else:
        raise ValueError(f'Unsupported reward evaluation method: {sim_config.reward_evaluation_method}')
    reward_at_n_step = mean_reward[n_step_dist-1]

    n_step = np.median(n_step_dist)
    if n_step == horizon:
        # if exceed horizon_max, set reward = 0 as penalty
        warnings.warn("Power threshold may be too hard to achieve: n_step exceeds max horizon. ")
        reward_at_n_step = best_mean.mean() + power[-1] - power_constraint

    # Step 4: Compute objective score
    obj_score_dist = (
        reward_at_n_step * n_step_dist +
        sim_config.step_cost * n_step_dist
    )

    #Step 5: get posterior rewrad (deployment phse)
    best_estimated_arm_indices = np.argmax(h1_res.arm_means, axis=2)
    rows = np.arange(true_means.shape[0])[:, None]
    selected_means = np.mean(true_means[rows, best_estimated_arm_indices],axis=0)
    selected_means = get_interpolation(selected_means, sim_config.step_schedule)

    return {
        "obj_score": np.mean(obj_score_dist),
        "obj_score_sd": np.std(obj_score_dist),
        "n_step": np.median(n_step_dist),
        "regret_per_step": mean_reward[int(np.median(n_step_dist-1))],
        "deployment_regret":best_mean.mean() - selected_means[int(np.median(n_step_dist-1))],
        "power_max": np.max(power),
        "mean_regret_at_horizon":mean_reward[-1]
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

    def __call__(self, algo_param_dict:Dict):
        algo_param = algo_param_dict['algo_para']

        algo = self.algo_class(algo_param)
        h1_res = run_simulation(
            policy=algo,
            sim_config=self.sim_config,
        )

        #Based on h1 result, create H0 simulation setting
        weight, h0_sim_loc_array = self.sim_config.test_procedure.get_h0_cores_and_weights(h1_res.combined_means[:,-1,:])
        if 'T-Constant' in self.sim_config.test_procedure.test_signature: #TODO: maybe put in test procedure class... but how to involve 'run_sim'?
            h0_sim_loc_array = h0_sim_loc_array * 0 + self.sim_config.test_procedure.constant_threshold
        h1_n_rep = self.sim_config.n_rep
        self.sim_config.n_rep = len(h0_sim_loc_array)
        h0_res = run_simulation(
            policy=algo,
            sim_config=self.sim_config,
            arm_mean_reward_dist=h0_sim_loc_array[:,np.newaxis], #is this the right var?
        )
        crit_boundary = self.sim_config.test_procedure.get_adjusted_crit_region(weight, h0_res)
        self.sim_config.n_rep = h1_n_rep


        result = get_objective_score(
            crit_boundary=crit_boundary,
            h1_res=h1_res,
            sim_config=self.sim_config
        )

        # Track result externally
        self.sim_result_keeper[(algo.__name__, algo_param, self.sim_config.setting_signature)] = result #TODO: define config.setting_signiture / dict?

        return {"obj_score": (result["obj_score"], result["obj_score_sd"])}

def optimize_algorithm(sim_config:SimulationConfig,algo,algo_param_list):
    evaluator = AxObjectiveEvaluator(
        algo_class=algo,
        sim_config=sim_config,
        sim_result_keeper={}
    )

    if algo_param_list is not None:
        best_parameters, values, experiment, model = optimize(
            parameters=[
                {
                    "name": "algo_para",
                    "type": "choice",
                    "values": algo_param_list,
                    "value_type": "float",
                },
            ],
            evaluation_function=evaluator,  # callable class instance
            objective_name="obj_score",  # must match return key
            minimize=True,
            total_trials=len(algo_param_list),
        )


    if sim_config.n_opt_trials is not None:
        best_parameters, values, experiment, model = optimize(
            parameters=[
                {
                    "name": "algo_para",
                    "type": "range",
                    "bounds": [0, 1],
                    "value_type": "float",
                },
            ],
            evaluation_function=evaluator,  # callable class instance
            objective_name="obj_score",  # must match return key
            minimize=True,
            total_trials=sim_config.n_opt_trials,
        )

    return best_parameters, values, experiment, model, evaluator.sim_result_keeper