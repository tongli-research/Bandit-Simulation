import warnings
from typing import Optional, Literal, Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from sim_wrapper import SimResult


@dataclass
class TestProcedure(ABC):
    """
    Abstract base class for hypothesis test procedures.

    Defines the common interface and parameters for all tests, including test name,
    Type I/II error constraints, minimum detectable effect, and other test-specific settings.

    Attributes:
        type1_error_constraint (float): The maximum allowed Type I error rate (default: 0.05).
        power_constraint (float): The desired statistical power (default: 0.80).
        min_effect (float): Minimum detectable effect size for power calculations.
                            Defines the smallest difference between arms to consider meaningful in power evaluation.

        crit_region_direction (int): Determines the direction of the critical region.
                                     Use 1 for tests where larger test statistics indicate significance (e.g., Wald-type tests),
                                     and -1 for tests based on p-values where smaller values indicate significance.
    """
    type1_error_constraint: float = 0.05 # [GUI_INPUT]
    family_wise_error_control: bool = False
    power_constraint: float = 0.80 # [GUI_INPUT]
    min_effect: float = 0.1 # [GUI_INPUT]
    crit_region_direction: int = 1
    n_crit_sim_groups: int = 9
    n_crit_sim_rep: int = -1 #-1 to auto match total rep in H1
    n_crit_approx_method: Literal['bin','linear'] = "linear"


    def get_h0_cores_and_weights(self,samples: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            samples: (n_rep,) array of sample values (e.g., combined_means)
            n_crit_sim_groups: number of groups to partition samples into
            n_crit_sim_rep: number of times to replicate the core locations (set -1 to auto)
            n_crit_approx_method: 'bin' for hard group assignment, 'linear' for interpolation

        Returns:
            group_index: (n_rep,) group index per sample (0-based)
            weight: (n_rep, n_core) weight matrix
            h0_sim_loc_array: (n_final_core,) replicated loc array
        """

        samples = np.asarray(samples).flatten()
        n_rep = len(samples)

        # Sort samples and get group index by rank
        rank = samples.argsort().argsort()
        group_index = (rank / n_rep * self.n_crit_sim_groups).astype(int)
        group_index = np.clip(group_index, 0, self.n_crit_sim_groups - 1)

        if self.n_crit_approx_method == 'bin':
            # Median per group
            h0_sim_loc_array = np.array([
                np.median(samples[group_index == g]) for g in range(self.n_crit_sim_groups)
            ])
            # Weight is 1 for corresponding group, 0 otherwise
            weight = np.zeros((n_rep, self.n_crit_sim_groups))
            weight[np.arange(n_rep), group_index] = 1.0

        elif self.n_crit_approx_method == 'linear':
            group_mins = np.array([
                np.min(samples[group_index == g]) for g in range(self.n_crit_sim_groups)
            ])
            global_max = np.max(samples)
            h0_sim_loc_array = np.append(group_mins, global_max)  # (n_groups + 1,)

            # Compute linear interpolation weights
            weight = np.zeros((n_rep, self.n_crit_sim_groups + 1))
            for i in range(n_rep):
                s = samples[i]
                # Find where to insert sample s in h0_sim_loc_array
                idx = np.searchsorted(h0_sim_loc_array, s, side='right') - 1
                idx = np.clip(idx, 0, len(h0_sim_loc_array) - 2)
                l, r = h0_sim_loc_array[idx], h0_sim_loc_array[idx + 1]
                if r > l:
                    weight[i, idx] = (r - s) / (r - l)
                    weight[i, idx + 1] = (s - l) / (r - l)
                else:
                    weight[i, idx] = 1.0

        else:
            raise ValueError(f"Unknown method {self.n_crit_approx_method}")

        h0_sim_loc_array = np.repeat(h0_sim_loc_array, self.n_crit_sim_rep)

        return  weight, h0_sim_loc_array

    def get_adjusted_crit_region(self,weight: np.ndarray,h0_sim_result:SimResult):

        test_stat = self.get_test_statistics(h0_sim_result)  # shape = (n_rep_total, horizon, ...)
        n_cores = weight.shape[1]
        rep_per_core = self.n_crit_sim_rep

        # Step 2: Slice test_stat into n_cores blocks and get critical region for each
        core_crit_list = []
        for i in range(n_cores):
            start = i * rep_per_core
            end = (i + 1) * rep_per_core
            stat_slice = test_stat[start:end]  # shape = (rep_per_core, horizon, ...)
            crit = self.get_critical_region(stat_slice)  # shape = (horizon, ...) or whatever
            core_crit_list.append(crit)

        # Step 3: Stack into array of shape (n_cores, horizon, ...)
        core_crit_array = np.stack(core_crit_list, axis=0)  # (n_cores, horizon, ...)

        # Step 4: Interpolate using weight — broadcasting weight @ core_crit_array
        # weight: (n_rep, n_cores)
        # core_crit_array: (n_cores, horizon, ...) -> need to broadcast multiply
        # result: (n_rep, horizon, ...)
        adjusted_crit_region = np.tensordot(weight, core_crit_array, axes=(1, 0))  # shape (n_rep, horizon, ...)

        return adjusted_crit_region

    @property
    @abstractmethod
    def test_signature(self) -> str:
        pass

    @abstractmethod
    def get_test_statistics(self, sim_result:SimResult, horizon=slice(None)):
        pass

    @abstractmethod
    def get_critical_region(self, test_stat:np.ndarray): #output lower and upper

        pass

    @abstractmethod
    def create_min_effect_filter(self, ground_truth_arm_mean_dist: np.ndarray):  # output lower and upper

        pass

    @abstractmethod
    def compute_power(self,
                      h1_sim_result:SimResult,
                      crit_boundary: np.ndarray,
                      ground_truth_arm_mean_dist: np.ndarray,):
        """

        :param h1_sim_result:
        :param crit_boundary:
        :param ground_truth_arm_mean_dist: for filter min_effect
        :return:
        """

        pass


@dataclass
class ANOVA(TestProcedure):
    crit_region_direction: int = -1

    @property
    def test_signature(self) -> str:
        return "ANOVA"

    def get_test_statistics(self, sim_result, horizon=slice(None)):
        return sim_result.anova(horizon=horizon)

    def get_critical_region(self, test_stat):
        horizon_axis = 1

        # Move horizon axis to front
        stat_reordered = np.moveaxis(test_stat, horizon_axis, 0)  # shape: (horizon, ...)

        # Flatten all axes except horizon
        horizon_len = stat_reordered.shape[0]
        flattened = stat_reordered.reshape(horizon_len, -1)  # shape: (horizon, flattened_dims)

        # Quantile along flattened dims
        crit_boundary = np.quantile(flattened, q=self.type1_error_constraint, axis=1,keepdims=True)  # shape: (horizon,)

        return crit_boundary

    def create_min_effect_filter(self, ground_truth_arm_mean_dist: np.ndarray):  # output lower and upper
        diff = np.max(ground_truth_arm_mean_dist,axis=1) - np.min(ground_truth_arm_mean_dist,axis=1)

        return  diff > self.min_effect

    def compute_power(self,
                      h1_sim_result:SimResult,
                      crit_boundary,
                      ground_truth_arm_mean_dist: np.ndarray,):

        h1_test_stat = self.get_test_statistics(h1_sim_result) #TODO: can change horizon?

        if self.crit_region_direction >0:
            test_result = h1_test_stat  > crit_boundary
        else:
            test_result = h1_test_stat < crit_boundary
        min_effect_filter = self.create_min_effect_filter(ground_truth_arm_mean_dist)
        test_result = test_result * 1.0 #make it float, otherwise nan asignment cannot succeed
        test_result[~min_effect_filter] = np.nan
        power = np.nanmean(test_result, axis=(0, 2))
        return power




@dataclass
class TControl(TestProcedure):
    """
    Test procedure comparing all arms against the control arm.

    Currently, the first arm (index 0) is always treated as the control arm (this is hard-coded and cannot be changed).

    Attributes:
        test_type (str): Specifies whether the test is 'one-sided' (only detects arms better than control) or 'two-sided'.
        control_group_index (int): Index of the control arm (fixed at 0, cannot be changed).
    """
    test_type: Literal['one-sided', 'two-sided'] = 'one-sided' # [GUI_INPUT]
    control_group_index: int = 0

    @property
    def test_signature(self) -> str:
        signature = f"T-Control ({self.test_type})"
        if self.min_effect != 0.1:
            signature += f", min_effect={self.min_effect}"
        return signature

    def get_test_statistics(self, sim_result, horizon=slice(None)):
        return sim_result.t_control(horizon=horizon)

    def get_critical_region(self, test_stat):
        horizon_axis = 1

        # Move horizon axis to front
        stat_reordered = np.moveaxis(test_stat, horizon_axis, 0)  # shape: (horizon, ...)

        # Flatten all axes except horizon
        horizon_len = stat_reordered.shape[0]
        flattened = stat_reordered.reshape(horizon_len, -1)  # shape: (horizon, flattened_dims)

        # Quantile along flattened dims
        if self.test_type == 'two-sided':
            crit_boundary = np.quantile(np.abs(flattened), q=1-self.type1_error_constraint, axis=1,keepdims=True)  # shape: (horizon,)
        elif self.test_type == 'one-sided':
            crit_boundary = np.quantile(flattened, q=1-self.type1_error_constraint, axis=1, keepdims=True)
        else:
            raise NotImplementedError
        return crit_boundary

    def create_min_effect_filter(self, ground_truth_arm_mean_dist: np.ndarray):  # output lower and upper
        diff = ground_truth_arm_mean_dist[:,1:] - ground_truth_arm_mean_dist[:,0:1]
        if self.test_type == 'two-sided':
            mask = np.abs(diff)>self.min_effect
        elif self.test_type == 'one-sided':
            mask = diff>self.min_effect
        else:
            raise NotImplementedError

        return  mask

    def compute_power(self,
                      h1_sim_result:SimResult,
                      crit_boundary,
                      ground_truth_arm_mean_dist: np.ndarray,):

        h1_test_stat = self.get_test_statistics(h1_sim_result) #TODO: can change horizon?

        if self.crit_region_direction >0:
            test_result = h1_test_stat  > crit_boundary
        else:
            test_result = h1_test_stat < crit_boundary
        min_effect_filter = self.create_min_effect_filter(ground_truth_arm_mean_dist)
        test_result = test_result * 1.0  # make it float, otherwise nan asignment cannot succeed
        expanded_mask = np.broadcast_to(min_effect_filter[:, np.newaxis, :], test_result.shape)
        test_result[~expanded_mask] = np.nan
        power = np.nanmean(test_result, axis=(0, 2))
        return power

@dataclass
class TConstant(TestProcedure):
    """
    Compare all arm against a fixed constant.
    """
    test_type: Literal['one-sided', 'two-sided'] = 'one-sided' # one-sided: only test arms that's better than control
    constant_threshold: Optional[float] = None #TODO: check and assign it in hyper

    def __post_init__(self):
        if self.n_crit_sim_groups != 1:
            warnings.warn("Set 'n_crit_sim_group' back to 1 for t-constant test ")
            self.n_crit_sim_groups = 1
        if self.n_crit_approx_method != 'bin':
            warnings.warn("Set 'n_crit_approx_method' back to 'bin' for t-constant test ")
            self.n_crit_approx_method = 'bin'

    @property
    def test_signature(self) -> str:
        signature = f"T-Constant ({self.test_type})"
        if self.min_effect != 0.1:
            signature += f", min_effect={self.min_effect}"
        return signature

    def get_test_statistics(self, sim_result:SimResult, horizon=slice(None)):
        return sim_result.t_constant(constant_threshold=self.constant_threshold,horizon=horizon)

    def get_critical_region(self, test_stat):
        horizon_axis = 1

        # Move horizon axis to front
        stat_reordered = np.moveaxis(test_stat, horizon_axis, 0)  # shape: (horizon, ...)

        # Flatten all axes except horizon
        horizon_len = stat_reordered.shape[0]
        if self.family_wise_error_control:
            if self.test_type == 'two-sided':
                flattened = np.max(np.abs(stat_reordered),axis=-1)
            elif self.test_type == 'one-sided':
                flattened = np.max(stat_reordered,axis=-1)
            else:
                raise NotImplementedError
        else:
            if self.test_type == 'two-sided':
                flattened = np.abs(stat_reordered.reshape(horizon_len, -1))  # shape: (horizon, flattened_dims)
            elif self.test_type == 'one-sided':
                flattened = stat_reordered.reshape(horizon_len, -1)
            else:
                raise NotImplementedError


        # Quantile along flattened dims
        if self.test_type == 'two-sided':
            crit_boundary = np.quantile(flattened, q=1-self.type1_error_constraint, axis=1,keepdims=True)  # shape: (horizon,)
        elif self.test_type == 'one-sided':
            crit_boundary = np.quantile(flattened, q=1-self.type1_error_constraint, axis=1, keepdims=True)
        else:
            raise NotImplementedError
        return crit_boundary

    def create_min_effect_filter(self, ground_truth_arm_mean_dist: np.ndarray):  # output lower and upper
        diff = ground_truth_arm_mean_dist - self.constant_threshold
        if self.test_type == 'two-sided':
            mask = np.abs(diff)>self.min_effect
        elif self.test_type == 'one-sided':
            mask = diff>self.min_effect
        else:
            raise NotImplementedError
        return  mask

    def compute_power(self,
                      h1_sim_result:SimResult,
                      crit_boundary,
                      ground_truth_arm_mean_dist: np.ndarray,):

        h1_test_stat = self.get_test_statistics(h1_sim_result) #TODO: can change horizon?

        if self.crit_region_direction >0:
            test_result = h1_test_stat  > crit_boundary
        else:
            test_result = h1_test_stat < crit_boundary
        min_effect_filter = self.create_min_effect_filter(ground_truth_arm_mean_dist)
        test_result = test_result * 1.0 # make it float, otherwise nan asignment cannot succeed
        expanded_mask = np.broadcast_to(min_effect_filter[:, np.newaxis, :], test_result.shape)
        test_result[~expanded_mask] = np.nan
        power = np.nanmean(test_result, axis=(0, 2))
        return power




@dataclass
class Tukey(TestProcedure):

    test_type: Literal['all-pair-wise', 'distinct-best-arm'] = 'distinct-best-arm' # [GUI_INPUT]
    control_group_index: int = 0

    @property
    def test_signature(self) -> str:
        signature = f"Tukey ({self.test_type})"
        if self.min_effect != 0.1:
            signature += f", min_effect={self.min_effect}"
        return signature

    def get_test_statistics(self, sim_result, horizon=slice(None)):
        return sim_result.tukey(horizon=horizon)

    def _filter_stats(self,stat_reordered):
        stat_reordered[stat_reordered == 0] = np.nan
        horizon, n_rep, n_arm, _ = stat_reordered.shape
        h_idx = np.arange(horizon)[:, None]  # shape (horizon, 1)
        r_idx = np.arange(n_rep)[None, :]  # shape (1, n_rep)
        i_idx = np.argmax(np.nansum(stat_reordered, axis=3), axis=2)
        flattened = np.nanmin(stat_reordered[h_idx, r_idx, i_idx, :], axis=-1)
        return flattened

    def get_critical_region(self, test_stat):
        horizon_axis = 1

        # Move horizon axis to front
        stat_reordered = np.moveaxis(test_stat, horizon_axis, 0)  # shape: (horizon, ...)

        if self.test_type == 'distinct-best-arm':
            # Create broadcastable indices
            flattened = self._filter_stats(stat_reordered)
            crit_boundary = np.quantile(flattened, q=1 - self.type1_error_constraint, axis=1, keepdims=True) #TODO: see why it gets smaller as T increase
        elif self.test_type == 'all-pair-wise':
            horizon, n_rep, n_arm, _ = stat_reordered.shape
            flattened = stat_reordered.reshape(horizon, -1)
            crit_boundary = np.nanquantile(flattened, q=1 - self.type1_error_constraint, axis=1, keepdims=True)
        else:
            raise NotImplementedError

        return crit_boundary

    def create_min_effect_filter(self, ground_truth_arm_mean_dist: np.ndarray):  # output lower and upper
        diff = ground_truth_arm_mean_dist[:,1:] - ground_truth_arm_mean_dist[:,0:1]
        if self.test_type == 'two-sided':
            mask = np.abs(diff)>self.min_effect
        elif self.test_type == 'one-sided':
            mask = diff>self.min_effect
        else:
            raise NotImplementedError

        return  mask

    def compute_power(self,
                      h1_sim_result:SimResult,
                      crit_boundary,
                      ground_truth_arm_mean_dist: np.ndarray,):

        h1_test_stat = self.get_test_statistics(h1_sim_result) #TODO: can change horizon?

        if self.test_type == 'distinct-best-arm':
            stat_reordered = np.moveaxis(h1_test_stat, 1, 0)
            sorted_h1_test_stat = self._filter_stats(stat_reordered)

            sorted_vals = np.sort(ground_truth_arm_mean_dist, axis=1)[:, ::-1]
            diff = sorted_vals[:,0] - sorted_vals[:,1]
            mask = diff > self.min_effect

            test_result = sorted_h1_test_stat.T[:,:,np.newaxis] > crit_boundary
            test_result = test_result * 1.0  # make it float, otherwise nan asignment cannot succeed
            test_result[~mask] = np.nan
            power = np.nanmean(test_result, axis=(0, 2))
        elif self.test_type == 'all-pair-wise':
            mask = abs(ground_truth_arm_mean_dist[:,:,np.newaxis] - ground_truth_arm_mean_dist[:,np.newaxis,:]) > self.min_effect
            test_result = np.abs(h1_test_stat) >  crit_boundary[:,:,:,np.newaxis]
            test_result = test_result * 1.0  # make it float, otherwise nan asignment cannot succeed
            expanded_mask = np.broadcast_to(mask[:, np.newaxis, :, :], test_result.shape)
            test_result[~expanded_mask] = np.nan
            power = np.nanmean(test_result, axis=(0, 2,3))
        else:
            raise NotImplementedError

        return power