import math

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Union, Literal, Type
import warnings

import bayes_vector_ops as bayes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from test_procedure_configurator import TestProcedure


class ArrDim:
    #TODO: remove this... not useful
    # example: ad = ArrDim(arr_axis={'n_rep':0,'horizon':-2,'n_arm':-1},n_arm=6,n_rep=2,horizon=11)
    def __init__(self, arr_axis, n_arm,n_rep,horizon,step_schedule,sample_batch_schedule):

        self.total_dims = len(arr_axis)
        self.arr_axis = arr_axis
        # shape_list = [None] * self.total_dims
        # self.order_list = [None] * self.total_dims
        # for key, pos in arr_axis.items():
        #     value = args[key]
        #     shape_list[pos] = value
        #     self.order_list[pos] = key
        self.shape_arr = np.array([n_rep,len(step_schedule),n_arm])
        # self.order_arr = np.array(self.order_list)

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

    # def sub_shape(self, exclude_list):
    #     indices = ~np.isin(self.order_arr, exclude_list)
    #     return self.shape_arr[indices]


@dataclass
class SimulationConfig:
    """
    Configuration for bandit simulation.

    ----------------------------------------
    Simulation Schedule Parameters
    ----------------------------------------
    horizon (int): Total number of rounds (time-steps).
    burn_in (int or None): Number of initial burn-in rounds before the main simulation begins.

    Batching Setting Parameters:
        base_batch_size (int): Base number of samples drawn per batch at each round.
        batch_scaling_rate (float): Scaling factor that adjusts batch size according to the cumulative number of samples collected so far.

        These parameters jointly determine how many samples (e.g., customers) are drawn per simulation round.

        At each round t, the actual batch size is computed as:
            floor(base_batch_size + n_t * batch_scaling_rate)
        where n_t is the total number of samples collected before round t.

        Larger batch_scaling_rate values can significantly speed up simulations by reducing update frequency over time (up to exponentially).

        Note:
            Setting base_batch_size = 1 and batch_scaling_rate = 0 yields the classic per-step update.

    ----------------------------------------
    Reward & Arm Distribution Settings
    ----------------------------------------
    n_arm (int): Number of arms (actions) available in the bandit environment.

    reward_model (Callable): Function used to sample rewards from the reward distribution (e.g., Bernoulli or Normal).

    arm_mean_reward_dist_model (Callable): Distribution used to generate the expected (mean) rewards for each arm across replications.
    arm_mean_reward_dist_loc (float or list of float): Mean(s)/location parameter of the distribution for the true expected rewards of the arms (i.e., the "ground truth").
                                                       Can be a single float (applied to all arms) or a list specifying the mean for each arm.
    arm_mean_reward_dist_scale (float or list of float): Standard deviation(s)/scale parameter of the distribution for the true expected rewards of the arms.
                                                       Can be a single float or a list specifying each arm’s standard deviation.
                                                       A value of 0 implies a fixed simulation scenario across replications (but still unknown to the algorithm).

    reward_std (float or None): Standard deviation of the reward distribution itself.
                                Ignored for Bernoulli rewards, where the variance is automatically determined by the mean.

    arm_mean_reward_cap (list of float): Lower and upper bounds on the expected reward means for each arm.
                                         Used to prevent extreme values that may cause numerical instability or unrealistic simulations.

    ----------------------------------------
    Objective & Evaluation Parameters
    ----------------------------------------
    test_procedure (TestProcedure or None): Hypothesis test procedure defining test of interest and constraints. See 'TestProcedure' in 'test_procedure_configurator.py' for details.
    reward_evaluation_method (Literal): Method used to evaluate simulation performance.
        Options: 'reward' — Use raw reward;
                 'scaled_reward' — Scale reward by subtracting 'arm_dist_mean';
                 'regret' — Regret is defined as the difference in expected reward between the optimal arm and the selected arm.
    step_cost (float): Penalty applied to the score for taking more steps to meet a predefined power constraint.

    ----------------------------------------
    General Simulation Settings
    ----------------------------------------
    n_rep (int): Number of independent simulation replications (for better accuracy in results).
    record_ap (bool): Whether to record allocation probabilities for each algorithm during the simulation (mostly for diagnostic purposes).
    n_ap_rep (int): Number of replications used to approximate allocation probabilities (only relevant if 'record_ap' is True).
    n_opt_trials (int): Number of optimization trials (only eligible for optimization simulations).
    horizon_check_points (np.ndarray or None): Specific checkpoints (time-steps) at which results should be saved. Useful for reducing result file size while still capturing key points in the simulation.
    """

    # Bandit: Schedule parameters
    horizon: int = 1000  # [GUI_INPUT]
    burn_in_per_arm: int = 10
    base_batch_size: int = 1
    batch_scaling_rate: float = 0.0
    compact_array: bool = True
    step_schedule:np.ndarray = None
    sample_batch_schedule: np.ndarray = None
    sample_batch_extended_schedule: np.ndarray = None

    # Bandit: Reward/Arm distribution parameters
    n_arm: int = 2  # [GUI_INPUT]
    reward_model: Callable = np.random.binomial  # [GUI_INPUT]
    arm_mean_reward_dist_model: Callable = np.random.normal
    arm_mean_reward_dist_loc: Union[float, list[float]] = 0.5  # [GUI_INPUT]
    arm_mean_reward_dist_scale: Union[float, list[float]] = 0.15  # [GUI_INPUT]
    reward_std: Optional[float] = None # [GUI_INPUT]
    arm_mean_reward_cap: list[float] = field(default_factory=lambda: [0.05, 0.95])


    # Bayes model class
    bayes_model: Optional[bayes.BayesianPosteriorModel] = None


    # Objective function and score-evaluation parameters
    test_procedure: Optional['TestProcedure'] = None # [GUI_INPUT]

    reward_evaluation_method: Literal['reward', 'scaled_reward', 'regret'] = 'reward'
    step_cost: float = 0  # [GUI_INPUT]

    # General simulation parameters
    n_rep: int = 10000  # [GUI_INPUT]
    record_ap: bool = False
    n_ap_rep: int = 100
    n_opt_trials: int = 10  # [GUI_INPUT]
    horizon_check_points: Optional[np.ndarray] = None

    #backend params
    vector_ops: bayes.BackendOps = field(default_factory=bayes.BackendOpsNP)

    def __post_init__(self):
        self.vector_ops.manual_init()

    def manual_init(self): #TODO: can we remove manual init? Maybe not..
        # Check type for arm distributions
        mean_is_list = isinstance(self.arm_mean_reward_dist_loc, list)
        std_is_list = isinstance(self.arm_mean_reward_dist_scale, list)

        # Case 1: Both lists → check length
        if mean_is_list and std_is_list:
            if len(self.arm_mean_reward_dist_loc) != len(self.arm_mean_reward_dist_scale):
                raise ValueError("arm_distribution_mean and arm_distribution_std must have the same length.")
            if len(self.arm_mean_reward_dist_loc) != self.n_arm:
                warnings.warn(
                    f"Length of arm_distribution_mean and arm_distribution_std ({len(self.arm_mean_reward_dist_loc)}) "
                    f"differs from n_arm ({self.n_arm}). Updating n_arm automatically."
                )
                self.n_arm = len(self.arm_mean_reward_dist_loc)

        # Case 2: Any scalar → broadcast both to length n_arm
        else:
            if not mean_is_list:
                self.arm_mean_reward_dist_loc = [self.arm_mean_reward_dist_loc] * self.n_arm
            if not std_is_list:
                self.arm_mean_reward_dist_scale = [self.arm_mean_reward_dist_scale] * self.n_arm

        # set default bayes model
        if self.bayes_model is None:
            if self.reward_model.__name__ == 'binomial':
                self.bayes_model = bayes.BetaBernoulli(number_of_arms=self.n_arm, backend_ops=self.vector_ops)
            elif self.reward_model.__name__ == 'normal':
                self.bayes_model = bayes.NormalFull(number_of_arms=self.n_arm, backend_ops=self.vector_ops)
            else:
                raise ValueError(f'{self.reward_model.__name__} is not implemented.')
        if self.test_procedure.n_crit_sim_groups is None:
            self.test_procedure.n_crit_sim_groups = max(int(np.floor(np.sqrt(self.n_rep/500))) , 1)

        if self.test_procedure.n_crit_sim_rep == -1:
            if self.test_procedure.n_crit_approx_method == 'bin':
                n_base = self.test_procedure.n_crit_sim_groups
            elif self.test_procedure.n_crit_approx_method == 'linear':
                n_base = self.test_procedure.n_crit_sim_groups + 1
            else:
                raise ValueError(f"Unknown method {self.test_procedure.n_crit_approx_method}")

            self.test_procedure.n_crit_sim_rep = int(np.ceil(self.n_rep / n_base))

        if 'T-Constant' in self.test_procedure.test_signature:
            if self.test_procedure.constant_threshold is None:
                self.test_procedure.constant_threshold = np.mean(self.arm_mean_reward_dist_loc)

        if self.step_schedule is None:
            if self.compact_array:
                self.step_schedule, self.sample_batch_schedule = self._get_step_schedule()
                rep_per_step = (self.step_schedule/self.sample_batch_schedule).astype(int)
                self.sample_batch_extended_schedule = np.repeat(self.sample_batch_schedule, rep_per_step)
            else:
                self.step_schedule = np.ones(self.horizon)
                self.sample_batch_schedule = np.ones(self.horizon)
                self.sample_batch_extended_schedule = np.ones(self.horizon)

    def _get_step_schedule(self):
        t=self.burn_in_per_arm * self.n_arm
        step_schedule_list = [self.burn_in_per_arm* self.n_arm]
        sample_batch_schedule_list = [self.burn_in_per_arm]
        while t<self.horizon:
            sample_batch_size = math.floor(1 + t**(1/3)/3 )
            step_size = math.floor(self.base_batch_size + t*0.05)
            if t + sample_batch_size > self.horizon:
                sample_batch_size = self.horizon - t
                step_size = self.horizon - t
            else:
                if t + step_size > self.horizon:
                    step_size = self.horizon - t
                if sample_batch_size > step_size:
                    sample_batch_size = step_size
                step_size = round(np.floor(step_size/sample_batch_size)*sample_batch_size)
            t = t + step_size
            step_schedule_list.append(step_size)
            sample_batch_schedule_list.append(sample_batch_size)
        return np.array(step_schedule_list), np.array(sample_batch_schedule_list)

    @property
    def ad(self):
        return ArrDim(arr_axis={'n_rep': 0, 'horizon': -2, 'n_arm': -1},
                      n_arm=self.n_arm, n_rep=self.n_rep, horizon=self.horizon,
                      step_schedule=self.step_schedule,sample_batch_schedule=self.sample_batch_schedule)

    @property
    def setting_signature(self):
        return f'step cost: {self.step_cost}; test: {self.test_procedure.test_signature}; reward metric: {self.reward_evaluation_method}'

    @property
    def arm_mean_reward_dist(self):
        np.random.seed(0) #TODO: can deal with model without scale / other parameters later.
        samples = np.random.normal(loc = self.arm_mean_reward_dist_loc, scale = self.arm_mean_reward_dist_scale, size =(self.n_rep,self.n_arm)) #TODO: here i use hard code to solve randomness issue. but doesn't work if use sself.model...
        return np.clip(samples, self.arm_mean_reward_cap[0], self.arm_mean_reward_cap[1])

    def generate_full_reward_trajectory(self, arm_mean_reward_dist = None):
        """

        :param arm_mean_reward_dist: Modification to arm_mean_reward_dist (for H0 sim). If None, use self.arm_mean_reward_dist
        :return:
        """
        np.random.seed(0)
        if arm_mean_reward_dist is None:
            arm_mean_reward_dist = self.arm_mean_reward_dist
        # base_shape = next(iter(self.h1_reward_dist.values())).shape
        # new_shape = (self.hyperparams.horizon,) + base_shape
        if self.reward_model.__name__ == 'binomial':
            params = {'n': 1, 'p': arm_mean_reward_dist[np.newaxis,:,:], 'size': (self.horizon,self.n_rep,self.n_arm)}
        elif self.reward_model.__name__ == 'normal':
            if self.reward_std is None:
                raise ValueError("reward_std must be provided for normal reward_model.")
            params = {'loc': arm_mean_reward_dist, 'scale': self.reward_std, 'size': self.n_rep}
        else:
            raise NotImplementedError(f"Reward model '{self.reward_model.__name__}' not supported.")

        reward_trajectory = self.reward_model(**params)
        reward2_trajectory = reward_trajectory**2

        if self.compact_array:
            reward_trajectory = self._get_compact_array(reward_trajectory)
            reward2_trajectory = self._get_compact_array(reward2_trajectory)


        reward_trajectory = np.moveaxis(reward_trajectory, 0, 1)
        reward2_trajectory = np.moveaxis(reward2_trajectory, 0, 1)


        return reward_trajectory, reward2_trajectory

    def _get_compact_array(self,arr):
        result = []
        start = 0
        for batch_size in self.sample_batch_extended_schedule:
            end = start + batch_size
            group_sum = np.sum(arr[start:end], axis=0)  # sum across axis=0 (rows)
            result.append(group_sum)
            start = end

        # Convert to array: shape will be (len(schedule), 10, 3)
        arr = np.stack(result)
        return arr





