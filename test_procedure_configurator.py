from typing import Optional, Literal, Dict, Any
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
    power_constraint: float = 0.80 # [GUI_INPUT]
    min_effect: float = 0.1 # [GUI_INPUT]
    crit_region_direction: int = 1

    @property
    @abstractmethod
    def test_signature(self) -> str:
        pass

    @abstractmethod
    def get_test_statistics(self, sim_result:SimResult, horizon=slice(-1, None)):
        pass

    @abstractmethod
    def get_critical_region(self, h0_sim_result:SimResult): #output lower and upper

        pass

    @abstractmethod
    def create_min_effect_filter(self, ground_truth_arm_mean_dist: np.ndarray):  # output lower and upper

        pass

    @abstractmethod
    def compute_power(self,
                      h1_sim_result:SimResult,
                      h0_sim_result:SimResult,
                      ground_truth_arm_mean_dist: np.ndarray,):
        """

        :param h1_sim_result:
        :param h0_sim_result:
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

    def get_test_statistics(self, sim_result, horizon=slice(-1, None)):
        return sim_result.anova(horizon=horizon)

    def get_critical_region(self, h0_sim_result: SimResult):
        horizon_axis = 1
        # Get test statistic: shape = (n_rep, horizon, ...)
        test_stat = self.get_test_statistics(h0_sim_result)

        # Move horizon axis to front
        stat_reordered = np.moveaxis(test_stat, horizon_axis, 0)  # shape: (horizon, ...)

        # Flatten all axes except horizon
        horizon_len = stat_reordered.shape[0]
        flattened = stat_reordered.reshape(horizon_len, -1)  # shape: (horizon, flattened_dims)

        # Quantile along flattened dims
        crit_boundary = np.quantile(flattened, q=self.type1_error_constraint, axis=1)  # shape: (horizon,)

        return crit_boundary

    def create_min_effect_filter(self, ground_truth_arm_mean_dist: np.ndarray):  # output lower and upper
        diff = np.max(ground_truth_arm_mean_dist,axis=1) - np.min(ground_truth_arm_mean_dist,axis=1)

        return  diff > self.min_effect

    def compute_power(self,
                      h1_sim_result:SimResult,
                      h0_sim_result:SimResult,
                      ground_truth_arm_mean_dist: np.ndarray,):

        h1_test_stat = self.get_test_statistics(h1_sim_result)
        crit_boundary = self.get_critical_region(h0_sim_result)
        test_result = h1_test_stat * self.crit_region_direction > crit_boundary * self.crit_region_direction
        min_effect_filter = self.create_min_effect_filter(ground_truth_arm_mean_dist)
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

    def get_test_statistics(self, sim_result, horizon=slice(-1, None)):
        return sim_result.t_control(horizon=horizon)

@dataclass
class TConstant(TestProcedure):
    """
    Compare all arm against a fixed constant.
    """
    test_type: Literal['one-sided', 'two-sided'] = 'one-sided' # one-sided: only test arms that's better than control
    constant_threshold: Optional[float] = None #TODO: check and assign it in hyper

    @property
    def test_signature(self) -> str:
        signature = f"T-Constant ({self.test_type})"
        if self.min_effect != 0.1:
            signature += f", min_effect={self.min_effect}"
        return signature

    def get_test_statistics(self, sim_result:SimResult, horizon=slice(-1, None)):
        return sim_result.t_constant(constant_threshold=self.constant_threshold,horizon=horizon)