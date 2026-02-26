import copy
import os
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from scipy.stats import bernoulli, f
from .bandit_algorithm import BanditAlgorithm
from .simulation_configurator import SimulationConfig

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import pandas as pd
import warnings

"""
Table of Content:
run_simulation - main simulation function
sweep_and_run - parameter sweep runner
run_task_common - single config evaluation (H1 sim + H0 crit + objective score)
get_objective_score - compute objective score from simulation results
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

# === Common runner ===
def run_task_common(
    sim_config_base,
    algo,
    algo_param_list=None,
    overrides: dict | None = None,
    test_proc: tuple | None = None,
):
    sim_config = copy.deepcopy(sim_config_base)

    # Apply overrides
    if overrides:
        for k, v in overrides.items():
            setattr(sim_config, k, v)

    # Handle test procedure
    if test_proc:
        sim_config.test_procedure = test_proc[0]
        sim_config.test_procedure.power_constraint = test_proc[1]

    sim_config.manual_init()

    sim_result_keeper = {}
    best_param = None
    best_score = None

    for algo_param in algo_param_list:
        policy = algo(algo_param)
        h1_res = run_simulation(policy=policy, sim_config=sim_config)

        # Based on h1 result, create H0 simulation setting
        weight, h0_sim_loc_array = sim_config.test_procedure.get_h0_cores_and_weights(
            h1_res.combined_means[:, -1, :]
        )
        if 'T-Constant' in sim_config.test_procedure.test_signature:
            h0_sim_loc_array = h0_sim_loc_array * 0 + sim_config.test_procedure.constant_threshold
        h1_n_rep = sim_config.n_rep
        sim_config.n_rep = len(h0_sim_loc_array)
        h0_res = run_simulation(
            policy=policy,
            sim_config=sim_config,
            arm_mean_reward_dist=h0_sim_loc_array[:, np.newaxis],
        )
        crit_boundary, _se_crit_adjusted, core_crit_array = sim_config.test_procedure.get_adjusted_crit_region(weight, h0_res)
        sim_config.n_rep = h1_n_rep

        # Extract unique H0 grid locations (before np.repeat in get_h0_cores_and_weights)
        h0_locations = h0_sim_loc_array[::sim_config.test_procedure.n_crit_sim_rep]

        result = get_objective_score(
            crit_boundary=crit_boundary,
            h1_res=h1_res,
            sim_config=sim_config,
            core_crit_array=core_crit_array,
            weight=weight,
            h0_locations=h0_locations,
            h0_res=h0_res,
        )

        key = (algo.__name__, algo_param, sim_config.setting_signature)
        sim_result_keeper[key] = result

        score = result["obj_score"]
        if best_score is None or score < best_score:
            best_score = score
            best_param = algo_param

    return {
        "test": sim_config.test_procedure.test_signature,
        "algo_name": algo.__name__,
        "algo_param": best_param,
        **sim_result_keeper.get(
            (algo.__name__, best_param, sim_config.setting_signature), None
        ),
        "all_results": sim_result_keeper,
    }

def sweep_and_run(sweep_specs, base_config):
    """
    sweep_specs: list of dicts, e.g.
        [
            {"horizon": [4000, 8000]},
            {"algo": [Algo1, Algo2]},
            {"algo_param_list": [0.0, 0.2]},
        ]
    base_config: SimulationConfig (copied inside run_task_common)

    Returns: DataFrame of results
    """
    import itertools

    sweep_dict = {}
    for d in sweep_specs:
        sweep_dict.update(d)

    keys = list(sweep_dict.keys())
    value_lists = [sweep_dict[k] for k in keys]

    all_results = []
    for combo in itertools.product(*value_lists):
        overrides = {}
        meta = {}

        algo = None
        algo_param_list = None

        for k, v in zip(keys, combo):
            if k == "algo":
                algo = v
                meta[k] = v.__name__
            elif k == "algo_param_list":
                algo_param_list = [v]   # always wrap in list
                meta[k] = v
            elif k == "test_proc":
                # Map to actual SimulationConfig field name
                overrides["test_procedure"] = v
                meta[k] = getattr(v, 'test_signature', str(v))
            elif isinstance(v, (int, float, str)):
                overrides[k] = v
                meta[k] = v
            else:
                overrides[k] = v
                meta[k] = f"option_{combo.index(v)}"

        # call run_task_common directly here
        result = run_task_common(
            base_config,
            algo=algo,
            algo_param_list=algo_param_list,
            overrides=overrides
        )

        all_results.append({**meta, **result})

    return pd.DataFrame(all_results)


# =============================================================================
# Simulation building blocks
# =============================================================================

@dataclass
class SimRunState:
    """Mutable state for an incrementally-advancing simulation.

    Holds the pre-allocated history arrays and pre-generated reward trajectories.
    run_one_batch_step() mutates this in-place, advancing time_step each call.

    Used for both H1 and H0 simulations. The only difference:
      - H1: theta is None, rewards come from prior-generated arm means
      - H0: theta is the null parameter value, all arms have equal mean = theta
    """
    action_hist: np.ndarray       # (M, T_batch, K) — pre-allocated to full schedule length
    reward_hist: np.ndarray       # (M, T_batch, K)
    reward2_hist: np.ndarray      # (M, T_batch, K)
    ap_hist: np.ndarray           # (M, T_batch, K) — only used if record_ap
    full_reward_traj: np.ndarray  # (M, n_action_groups, K) — pre-generated
    full_reward2_traj: np.ndarray # (M, n_action_groups, K) — pre-generated
    time_step: int                # next batch index to simulate
    total_action_samples: int     # cursor into reward trajectory (compact mode)
    theta: float = None           # None for H1, set for H0 (null parameter value)


def init_simulation(sim_config, arm_mean_reward_dist=None):
    """
    Allocate arrays, generate rewards, and apply burn-in.

    Returns a SimRunState ready for the main simulation loop.
    Both run_simulation() and adaptive_power_search use this.
    """
    ad = sim_config.ad
    n_arm = sim_config.n_arm
    sample_batch_schedule = sim_config.sample_batch_schedule

    action_hist = np.zeros(ad.shape_arr).astype(int)
    reward_hist = np.zeros(ad.shape_arr)
    reward2_hist = np.zeros(ad.shape_arr)
    ap_hist = np.zeros(ad.shape_arr)

    # Generate reward trajectories
    if sim_config.determined_reward_trajectory is not None:
        full_reward_traj, full_reward2_traj = sim_config.determined_reward_trajectory
    elif arm_mean_reward_dist is None:
        full_reward_traj, full_reward2_traj = sim_config.generate_full_reward_trajectory()
    else:
        full_reward_traj, full_reward2_traj = sim_config.generate_full_reward_trajectory(arm_mean_reward_dist)

    time_step = 0
    total_action_samples = 0

    # Burn-in
    if sim_config.burn_in_per_arm > 0:
        if sim_config.compact_array:
            action_hist[:, 0, :] = sample_batch_schedule[0]
            reward_hist[:, 0:1, :] = full_reward_traj[:, 0:1, :]
            reward2_hist[:, 0:1, :] = full_reward2_traj[:, 0:1, :]
            time_step = 1
            total_action_samples = n_arm
        else:
            raise NotImplementedError

    return SimRunState(
        action_hist=action_hist,
        reward_hist=reward_hist,
        reward2_hist=reward2_hist,
        ap_hist=ap_hist,
        full_reward_traj=full_reward_traj,
        full_reward2_traj=full_reward2_traj,
        time_step=time_step,
        total_action_samples=total_action_samples,
    )


def run_one_batch_step(policy, sim_config, state: SimRunState):
    """
    Execute one batch step of the simulation. Mutates state in-place.

    This is the extracted inner loop from run_simulation(). Both the original
    run_simulation and adaptive_power_search call this.
    """
    t = state.time_step
    step_schedule = sim_config.step_schedule
    sample_batch_schedule = sim_config.sample_batch_schedule
    batch_size = sample_batch_schedule[t]
    ad = sim_config.ad

    if sim_config.record_ap:
        slice_current = ad.slicing(horizon=slice(t))
        slice_next = ad.slicing(horizon=slice(t, t + batch_size))
        actions = policy.sample_action(
            sim_config,
            state.action_hist[slice_current],
            state.reward_hist[slice_current],
            state.reward2_hist[slice_current],
            batch_size=sim_config.n_ap_rep,
        )
        ap = np.mean(actions, axis=ad.arr_axis['horizon'])
        state.ap_hist[slice_next] = ad.tile(arr=ap, axis_name='horizon', repeats=batch_size)

    if sim_config.compact_array:
        n_action_samples = round(step_schedule[t] / sample_batch_schedule[t])

        action_sample = policy.sample_action(
            sim_config,
            state.action_hist[:, :t, :],
            state.reward_hist[:, :t, :],
            state.reward2_hist[:, :t, :],
            batch_size=n_action_samples,
        )

        tas = state.total_action_samples
        reward_sample = action_sample * state.full_reward_traj[:, tas:tas + n_action_samples, :]
        reward2_sample = action_sample * state.full_reward2_traj[:, tas:tas + n_action_samples, :]

        state.action_hist[:, t:t + 1, :] = batch_size * np.sum(action_sample, axis=1, keepdims=True)
        state.reward_hist[:, t:t + 1, :] = np.sum(reward_sample, axis=1, keepdims=True)
        state.reward2_hist[:, t:t + 1, :] = np.sum(reward2_sample, axis=1, keepdims=True)

        state.total_action_samples += n_action_samples
    else:
        slice_current = ad.slicing(horizon=slice(t))
        slice_next = ad.slicing(horizon=slice(t, t + batch_size))
        state.action_hist[slice_next] = policy.sample_action(
            sim_config, state.action_hist[slice_current],
            state.reward_hist[slice_current], batch_size=batch_size
        )
        state.reward_hist[slice_next] = state.full_reward_traj[slice_next] * state.action_hist[slice_next]

    state.time_step += 1


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
    state = init_simulation(sim_config, arm_mean_reward_dist)

    while state.time_step < len(sim_config.step_schedule):
        run_one_batch_step(policy, sim_config, state)

    return SimResult(
        state.action_hist.astype(int), state.reward_hist, state.reward2_hist,
        sim_config, ap_hist=state.ap_hist,
    )

def _art_batch(policy, sim_config, reward_hist_single, n_boot):
    """
    Vectorized ART bootstrap.
    Generates n_boot bootstrap resamples in ONE call.
    reward_hist_single: shape (1, T, K)
    Returns SimResult containing n_boot trajectories.
    """
    cfg = copy.deepcopy(sim_config)
    cfg.n_rep = n_boot

    # Expand reward history n_boot times
    reward_rep = np.repeat(reward_hist_single, n_boot, axis=0)
    reward2_rep = reward_rep  # Bernoulli

    cfg.determined_reward_trajectory = (reward_rep, reward2_rep)
    cfg.manual_init()

    return run_simulation(policy, cfg)

def _induced_batch(policy, sim_config, p_equal, n_boot):
    """
    Vectorized parametric bootstrap under H0.
    Produces n_boot bootstrap samples in one shot.
    """
    cfg = copy.deepcopy(sim_config)
    cfg.n_rep = n_boot

    cfg.arm_mean_reward_dist_spec = {
        "dist": "normal",
        "params": {
            "loc": p_equal,
            "scale": 0.0,
        }
    }
    cfg.determined_reward_trajectory = None
    cfg.manual_init()

    return run_simulation(policy, cfg)

# def art_replication(policy, sim_config, reward_hist_single, n_art_rep=200):
#     """
#     ART: Fix the observed H1 reward trajectory, and re-run policy multiple times
#     to compute distribution of test statistics under algorithm randomness.
#
#     reward_hist_single: shape (1, horizon, n_arm) from one H1 simulation result.
#     """
#     horizon = sim_config.horizon
#     n_arm = sim_config.n_arm
#
#     # 1) Expand reward trajectory
#     reward_hist = np.tile(reward_hist_single, (n_art_rep, 1, 1))
#
#     # 2) Initialize action history
#     action_hist = np.zeros_like(reward_hist, dtype=int)
#
#     # 3) Burn-in
#     burn = sim_config.burn_in_per_arm
#     if burn > 0:
#         for t in range(burn):
#             a = np.random.choice(n_arm, size=n_art_rep)
#             action_hist[:, t, :] = np.eye(n_arm)[a]
#         reward_hist[:, :burn, :] *= action_hist[:, :burn, :]
#
#     # 4) Forward simulation under fixed rewards
#     for t in range(burn, horizon):
#         a_prev = action_hist[:, :t, :]
#         r_prev = reward_hist[:, :t, :]
#
#         acts = policy.sample_action(sim_config, a_prev, r_prev, batch_size=1)
#         action_hist[:, t:t+1, :] = acts
#         reward_hist[:, t:t+1, :] *= acts
#
#     return SimResult(action_hist, reward_hist, reward_hist, sim_config)


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
        self.F = sim_config.arm_feature_matrix
        self.arm_mean_reward_dist_spec = sim_config.arm_mean_reward_dist_spec
        self.step_schedule = sim_config.step_schedule


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

            # # === NEW: combined variance (works for Normal rewards) ===
            # combined_square_cum = np.cumsum(
            #     np.sum(reward2_hist, axis=self.ad.arr_axis['n_arm'], keepdims=True),
            #     axis=self.ad.arr_axis['horizon']
            # )
            #
            # combined_square_means = combined_square_cum / self.total_counts
            #
            # # unbiased pooled variance estimate: Var = E[X²] − (E[X])²
            # self.combined_vars = (combined_square_means - self.combined_means ** 2) * (
            #         1 / (self.total_counts - 1)
            # )
            #self.combined_reward_vars = (self.combined_square_means - self.combined_means ** 2)

    @cached_property
    def combined_vars(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            combined_square_cum_rewards = np.cumsum(np.sum(self.reward2_hist, axis=self.ad.arr_axis['n_arm'], keepdims=True),
                                                    axis=self.ad.arr_axis['horizon'])  # for variance calculation
            combined_square_means = combined_square_cum_rewards / self.total_counts
            combined_vars = (combined_square_means - self.combined_means ** 2)  # var for arm mean! not arm reward!
        return combined_vars

    @cached_property
    def arm_vars(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            arm_square_cum_rewards = np.cumsum(self.reward2_hist, axis=self.ad.arr_axis['horizon'])  # for variance calculation
            arm_square_means = arm_square_cum_rewards / self.arm_counts
            arm_vars = ((arm_square_means - self.arm_means ** 2) * (
                        1 / (self.arm_counts - 1)))  # var for arm mean! not arm reward!
        return arm_vars

    def allocation_by_rank(self, ground_truth_arm_means):
        """Mean allocation proportion per arm, ordered by rank (best first).

        Parameters:
            ground_truth_arm_means: (n_rep, n_arm) true arm means.
                Can be identical across reps or different per rep.

        Returns:
            (n_arm,) array — index 0 = best arm's avg allocation, 1 = second best, etc.
        """
        # Total counts per arm across horizon: (n_rep, n_arm)
        total_per_arm = np.sum(self.action_hist, axis=self.ad.arr_axis['horizon'])
        total_all = np.sum(total_per_arm, axis=-1, keepdims=True)
        prop = total_per_arm / total_all  # (n_rep, n_arm)

        # Rank order per rep: descending by true mean
        rank_order = np.argsort(-ground_truth_arm_means, axis=1)  # (n_rep, n_arm)
        prop_ranked = np.take_along_axis(prop, rank_order, axis=1)

        return np.mean(prop_ranked, axis=0)  # (n_arm,)

    def compute_linear_factorial_metrics(self, horizon=slice(None)):
        """Compute gap and factorial-effect time series from cumulative arm means.

        Requires fixed arm means (scale=0) and an arm_feature_matrix (F).

        Returns dict with:
          gap_mean, gap_var        — (T, K) best-arm gap stats across reps
          x{col}_{hi}v{lo}_mean/var/true — per-factor successive-difference contrasts
          mu_true, factor_groups   — for debugging
        """
        spec = self.arm_mean_reward_dist_spec
        if spec is None:
            raise ValueError("arm_mean_reward_dist_spec not set")
        if spec["dist"] != "normal":
            raise ValueError(f"Expected normal dist, got {spec['dist']}")

        mu_true = np.array(spec["params"]["loc"])
        scale = spec["params"]["scale"]
        if isinstance(scale, list):
            scale = max(scale)
        if float(scale) != 0.0:
            raise ValueError(
                "Only fixed arm means supported: scale must be 0.0"
            )

        F = self.F
        if F is None:
            raise ValueError("arm_feature_matrix (F) not set in sim_config")

        K, d = F.shape
        arm_means = self.arm_means[:, horizon, :]  # (n_rep, T_sel, K)

        result = {"mu_true": mu_true}

        with np.errstate(divide='ignore', invalid='ignore'):
            # ── Mean reward per step ─────────────────────────────────
            arm_counts = self.arm_counts[:, horizon, :]  # (n_rep, T_sel, K)
            total_reward = np.nansum(arm_means * arm_counts, axis=-1)  # (n_rep, T)
            total_counts = np.nansum(arm_counts, axis=-1)              # (n_rep, T)
            mean_reward = total_reward / total_counts                  # (n_rep, T)
            result["reward_mean"] = np.nanmean(mean_reward, axis=0)    # (T,)
            result["reward_var"] = np.nanvar(mean_reward, axis=0)

            # ── Proportion of samples per arm ────────────────────────
            prop = arm_counts / total_counts[..., np.newaxis]  # (n_rep, T, K)
            result["prop_mean"] = np.nanmean(prop, axis=0)     # (T, K)
            result["prop_var"] = np.nanvar(prop, axis=0)

            # ── Best-arm gaps ────────────────────────────────────────
            best = np.nanmax(arm_means, axis=-1, keepdims=True)
            gaps = best - arm_means  # (n_rep, T, K)
            result["gap_mean"] = np.nanmean(gaps, axis=0)  # (T, K)
            result["gap_var"] = np.nanvar(gaps, axis=0)

            # ── Factorial effects from F columns ─────────────────────
            factor_groups = {}
            for col in range(1, d):
                col_vals = F[:, col]
                levels = np.sort(np.unique(col_vals))
                groups = {}
                for lev in levels:
                    groups[lev] = np.where(col_vals == lev)[0].tolist()
                factor_groups[col] = groups

                for i in range(len(levels) - 1):
                    lo, hi = levels[i], levels[i + 1]
                    lo_idx = groups[lo]
                    hi_idx = groups[hi]

                    effect = (
                        np.nanmean(arm_means[:, :, hi_idx], axis=-1)
                        - np.nanmean(arm_means[:, :, lo_idx], axis=-1)
                    )  # (n_rep, T)

                    lo_s = str(int(lo)) if lo == int(lo) else str(lo)
                    hi_s = str(int(hi)) if hi == int(hi) else str(hi)
                    prefix = f"x{col}_{hi_s}v{lo_s}"

                    result[f"{prefix}_mean"] = np.nanmean(effect, axis=0)
                    result[f"{prefix}_var"] = np.nanvar(effect, axis=0)
                    result[f"{prefix}_true"] = float(
                        np.mean(mu_true[hi_idx]) - np.mean(mu_true[lo_idx])
                    )

            result["factor_groups"] = factor_groups

        return result

    def wald_test(self, arm1_index=0, arm2_index=1,horizon = slice(-1,None)):
        arm1_slice = self.ad.slicing(n_arm=slice(arm1_index,arm1_index+1), horizon=horizon)
        arm2_slice = self.ad.slicing(n_arm=slice(arm2_index,arm2_index+1), horizon=horizon)

        cm_slice = self.ad.slicing(horizon=horizon)[0:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            walds = (self.arm_means[arm1_slice] - self.arm_means[arm2_slice]) / np.sqrt(
                self.combined_vars * (1 / (self.arm_counts[arm1_slice]) + 1/ (self.arm_counts[arm2_slice]))
            )

            # var1 = self.arm_vars[arm1_slice]
            # var2 = self.arm_vars[arm2_slice]
            # #
            # walds = (self.arm_means[arm1_slice] - self.arm_means[arm2_slice]) / np.sqrt(var1 + var2)
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

            # cm_slice = self.ad.slicing(horizon=horizon)[0:-1]

            with np.errstate(divide='ignore', invalid='ignore'):
                # test_stats = (self.arm_means[other_arm_slice] - self.arm_means[control_slice]) / np.sqrt(
                #     self.combined_means[cm_slice] * (1 - self.combined_means[cm_slice]) * (
                #             1 / self.arm_counts[other_arm_slice] + 1 / self.arm_counts[control_slice])
                # )

                test_stats = (self.arm_means[other_arm_slice] - self.arm_means[control_slice]) / np.sqrt(
                    self.combined_vars * (1 / self.arm_counts[other_arm_slice] + 1 / self.arm_counts[control_slice])
                )

                # var_c = self.arm_vars[control_slice] / self.arm_counts[control_slice]
                # var_o = self.arm_vars[other_arm_slice] / self.arm_counts[other_arm_slice]
                #
                # test_stats = (self.arm_means[other_arm_slice] - self.arm_means[control_slice]) / np.sqrt(var_c + var_o)


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
                self.combined_vars/ self.arm_counts[arm_slice]
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
        pooled_var = self.combined_vars[:,horizon,:]
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

    def bootstrap_test(self,policy,sim_config, rep_id, n_boot=200,mode="art",test="wald_test",):
        """
        Vectorized bootstrap test.
        Returns n_boot test statistics.
        """
        reward_hist_single = self.reward_hist[rep_id:rep_id + 1]

        # ART reward construction
        r_sum = np.sum(reward_hist_single, axis=2, keepdims=True)
        reward_hist_art = np.repeat(r_sum, sim_config.n_arm, axis=2)   # shape (1,T,K)

        # shape (K,) – means for each arm at final step for THIS replication
        p_equal = float(self.combined_means[rep_id, -1, :])

        # Vectorized sim_result
        if mode == "art":
            sim_res = _art_batch(policy, sim_config, reward_hist_art, n_boot)

        elif mode == "induced":
            sim_res = _induced_batch(policy, sim_config, p_equal, n_boot)

        else:
            raise ValueError("mode must be 'art' or 'induced'")

        # Lookup test function (no if/else)
        test_fn = getattr(SimResult, test)

        # Compute stats for all n_boot in one vector op
        stats = test_fn(sim_res, horizon=slice(-1, None)).flatten()

        return stats




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

def _gp_smooth_cores(core_crit_array, h0_locations):
    """GP-smooth per-core critical values across H0 locations to remove H0 variance noise.

    For each horizon step, fits a GP across H0 locations and returns smoothed predictions.
    This isolates the systematic trend in the critical boundary from sampling noise,
    giving a cleaner bias estimate.

    Args:
        core_crit_array: (n_cores, horizon_batch, ...) raw per-core critical values
        h0_locations: (n_cores,) H0 grid locations (theta values)

    Returns:
        smoothed: same shape as core_crit_array, GP-smoothed values
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    n_cores = core_crit_array.shape[0]
    if n_cores < 3:
        return core_crit_array  # not enough points for GP

    smoothed = np.empty_like(core_crit_array)
    X = h0_locations.reshape(-1, 1)
    kernel = Matern(nu=2.5) + WhiteKernel()

    # Flatten trailing dims (e.g., for TControl with multiple comparisons)
    orig_shape = core_crit_array.shape
    horizon_steps = orig_shape[1]
    trailing = orig_shape[2:]
    n_trailing = int(np.prod(trailing)) if trailing else 1
    flat = core_crit_array.reshape(n_cores, horizon_steps, n_trailing)
    smooth_flat = np.empty_like(flat)

    # Suppress ConvergenceWarnings: adjacent H0 locations have very similar
    # critical values, so the WhiteKernel noise level converges to its lower
    # bound. This is expected and harmless.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning, message=".*lbfgs.*")
        for col in range(n_trailing):
            for t in range(horizon_steps):
                y = flat[:, t, col]
                if np.any(np.isnan(y)):
                    smooth_flat[:, t, col] = y
                    continue
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=0)
                gp.fit(X, y)
                smooth_flat[:, t, col] = gp.predict(X)

    return smooth_flat.reshape(orig_shape)


def bootstrap_steps_se(
    tp,
    h1_test_stat,
    h0_test_stat,
    weight,
    min_effect_filter,
    power_curve,
    n_step,
    step_schedule,
    horizon,
    power_target,
    n_boot=200,
    seed=None,
):
    """
    Bootstrap SE of n_step using Single-Point Resample method.

    At T=n_step, for each bootstrap replicate:
      1. Resample H1 reps (and their weights + filter) with replacement
      2. For each H0 core, resample H0 stats and recompute critical value
      3. Interpolate through resampled weight matrix
      4. Compute power_b at that step
      5. Map power deviation to steps: shifted_target = 2*target - power_b

    Args:
        tp: TestProcedure instance
        h1_test_stat: (M1, H_batch, D) H1 test statistics at batch resolution
        h0_test_stat: (B*M0, H_batch, D_h0) H0 test statistics at batch resolution
        weight: (M1, B) interpolation weights from get_h0_cores_and_weights
        min_effect_filter: (M1,) or (M1, D) boolean mask
        power_curve: (horizon,) interpolated power curve (per-sample resolution)
        n_step: int, minimum sample size (per-sample units)
        step_schedule: list[int], samples per batch step
        horizon: int, total samples
        power_target: float, power constraint (e.g. 0.80)
        n_boot: int, number of bootstrap replicates
        seed: optional RNG seed

    Returns:
        dict with 'se_total', 'se_h1', 'se_h0'
    """
    from .test_procedure_configurator import Tukey

    # Skip bootstrap for Tukey (complex reject logic not supported yet)
    if isinstance(tp, Tukey):
        return {'se_total': 0.0, 'se_h1': 0.0, 'se_h0': 0.0}

    # Edge case: power never reaches target
    if n_step <= 0 or n_step >= horizon:
        return {'se_total': 0.0, 'se_h1': 0.0, 'se_h0': 0.0}

    rng = np.random.default_rng(seed)

    M1 = h1_test_stat.shape[0]
    n_cores = weight.shape[1]
    rep_per_core = tp.n_crit_sim_rep

    # Map n_step (per-sample) to batch step index
    cumulative = np.cumsum(step_schedule)
    t_idx = int(np.searchsorted(cumulative, n_step, side='left'))
    t_idx = min(t_idx, h1_test_stat.shape[1] - 1)

    # Pre-extract data at batch step t_idx
    h1_at_t = h1_test_stat[:, t_idx, :]  # (M1, D)

    # Pre-extract per-core H0 stats as single-step 3D slices
    h0_core_slices = []
    for b in range(n_cores):
        start = b * rep_per_core
        end = (b + 1) * rep_per_core
        h0_core_slices.append(
            h0_test_stat[start:end, t_idx:t_idx+1, :]  # (M0, 1, D_h0)
        )

    # Original core critical values at this step
    orig_core_crits = np.empty(n_cores)
    for b in range(n_cores):
        crit_val, _ = tp.get_critical_region(h0_core_slices[b])
        orig_core_crits[b] = crit_val[0, 0]

    # Reject parameters
    crit_direction = tp.crit_region_direction
    two_sided = getattr(tp, 'test_type', None) == 'two-sided'

    def _power_at_step(h1_b, crit_b, filter_b):
        """Compute power (reject fraction) at a single step."""
        if two_sided:
            reject = np.abs(h1_b) > crit_b
        elif crit_direction > 0:
            reject = h1_b > crit_b
        else:
            reject = h1_b < crit_b
        reject = reject * 1.0
        if filter_b.ndim == 1:
            reject[~filter_b] = np.nan
        else:
            reject[~filter_b] = np.nan
        return np.nanmean(reject)

    def _find_step(shifted_target):
        """Find n_step for a shifted power target on the original curve."""
        if shifted_target <= 0:
            return 0
        if shifted_target >= 1:
            return horizon
        return horizon - np.sum(power_curve > shifted_target)

    def _resample_core_crits(rng_local):
        """Resample H0 stats per core and recompute critical values."""
        core_crits_b = np.empty(n_cores)
        for b in range(n_cores):
            idx_h0 = rng_local.integers(0, rep_per_core, size=rep_per_core)
            h0_b = h0_core_slices[b][idx_h0]  # (M0, 1, D_h0)
            crit_val, _ = tp.get_critical_region(h0_b)
            core_crits_b[b] = crit_val[0, 0]
        return core_crits_b

    # === Total bootstrap: resample H1 + resample H0 ===
    boot_total = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, M1, size=M1)
        h1_b = h1_at_t[idx]
        weight_b = weight[idx]
        filter_b = min_effect_filter[idx]

        core_crits_b = _resample_core_crits(rng)
        crit_b = (weight_b @ core_crits_b)[:, np.newaxis]  # (M1, 1)

        power_b = _power_at_step(h1_b, crit_b, filter_b)
        boot_total[i] = _find_step(2 * power_target - power_b)

    # === H1-only: resample H1, keep original H0 ===
    boot_h1 = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, M1, size=M1)
        h1_b = h1_at_t[idx]
        weight_b = weight[idx]
        filter_b = min_effect_filter[idx]

        crit_b = (weight_b @ orig_core_crits)[:, np.newaxis]
        power_b = _power_at_step(h1_b, crit_b, filter_b)
        boot_h1[i] = _find_step(2 * power_target - power_b)

    # === H0-only: original H1, resample H0 ===
    boot_h0 = np.empty(n_boot)
    for i in range(n_boot):
        core_crits_b = _resample_core_crits(rng)
        crit_b = (weight @ core_crits_b)[:, np.newaxis]

        power_b = _power_at_step(h1_at_t, crit_b, min_effect_filter)
        boot_h0[i] = _find_step(2 * power_target - power_b)

    return {
        'se_total': float(np.std(boot_total)),
        'se_h1': float(np.std(boot_h1)),
        'se_h0': float(np.std(boot_h0)),
    }


def get_objective_score(crit_boundary:np.ndarray, h1_res:SimResult, sim_config:SimulationConfig,
                        core_crit_array=None, weight=None,
                        h0_locations=None, h0_res=None):
    """
    Compute the final objective score, score SD, number of steps, and reward at that step.
    Also computes error decomposition metrics (H1 variance, H0 variance, H0 bias) expressed as steps.

    Args:
        crit_boundary: interpolated critical boundary, shape (n_rep, horizon, ...)
        h1_res: SimResult from H1 simulation
        sim_config: SimulationConfig
        core_crit_array: per-core critical values, shape (n_cores, horizon, ...)
        weight: interpolation weight matrix, shape (n_rep, n_cores)
        h0_locations: H0 grid locations, shape (n_cores,)
        h0_res: SimResult from H0 simulation (for bootstrap SE)

    Returns:
        dict with objective score, error metrics, and diagnostic info
    """

    tp = sim_config.test_procedure

    # Step 1: Calculate power under H1
    power = tp.compute_power(
        crit_boundary=crit_boundary,
        h1_sim_result=h1_res,
        ground_truth_arm_mean_dist=sim_config.arm_mean_reward_dist
    )
    power = get_interpolation(power, sim_config.step_schedule)
    # Step 2: Determine minimum step that satisfies power constraint (with noise)
    power_constraint = tp.power_constraint
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
        step_wise_regret = np.mean(np.sum((best_mean[:, np.newaxis] - true_means)[:,np.newaxis,:]*h1_res.action_hist,axis=2),axis=0)/sim_config.step_schedule
        step_wise_regret = get_interpolation(step_wise_regret, sim_config.step_schedule)
        cumulative_regret = np.cumsum(step_wise_regret)
        mean_reward = (cumulative_regret / np.arange(1, horizon + 1)).flatten()
    else:
        raise ValueError(f'Unsupported reward evaluation method: {sim_config.reward_evaluation_method}')
    reward_at_n_step = mean_reward[n_step_dist-1]

    n_step = np.median(n_step_dist)
    if n_step == horizon:
        warnings.warn("Power threshold may be too hard to achieve: n_step exceeds max horizon. ")
        reward_at_n_step = best_mean.mean() + power[-1] - power_constraint

    # Step 4: Compute objective score
    obj_score_dist = reward_at_n_step * n_step_dist

    # Step 5: get posterior reward (deployment phase)
    best_estimated_arm_indices = np.argmax(h1_res.arm_means, axis=2)
    rows = np.arange(true_means.shape[0])[:, None]
    selected_means = np.mean(true_means[rows, best_estimated_arm_indices],axis=0)
    selected_means = get_interpolation(selected_means, sim_config.step_schedule)

    # ── Error decomposition: Bootstrap SE + Bias ──
    n_step_int = int(n_step)

    # Helper: find n_step for a shifted power target
    def _steps_for_target(target):
        if target <= 0:
            return 0
        return horizon - np.sum(power > target)

    # --- Bootstrap SE (H1 + H0 variance) ---
    se_steps_h1 = 0.0
    se_steps_h0 = 0.0
    se_steps_total = 0.0
    if h0_res is not None and weight is not None:
        try:
            h1_test_stat = tp.get_test_statistics(h1_res)
            h0_test_stat = tp.get_test_statistics(h0_res)
            min_effect_filter = tp.create_min_effect_filter(sim_config.arm_mean_reward_dist)

            boot_se = bootstrap_steps_se(
                tp=tp,
                h1_test_stat=h1_test_stat,
                h0_test_stat=h0_test_stat,
                weight=weight,
                min_effect_filter=min_effect_filter,
                power_curve=power,
                n_step=n_step_int,
                step_schedule=sim_config.step_schedule,
                horizon=horizon,
                power_target=power_constraint,
                n_boot=200,
            )
            se_steps_h1 = boot_se['se_h1']
            se_steps_h0 = boot_se['se_h0']
            se_steps_total = boot_se['se_total']
        except Exception:
            pass

    # --- H0 Bias (Worst Case) → Steps ---
    bias_steps_h0 = 0
    if core_crit_array is not None and weight is not None:
        try:
            n_cores = core_crit_array.shape[0]

            # GP-smooth core crits to remove H0 variance noise from bias estimate
            if h0_locations is not None and n_cores >= 3:
                try:
                    smoothed_cores = _gp_smooth_cores(core_crit_array, h0_locations)
                except Exception:
                    smoothed_cores = core_crit_array
            else:
                smoothed_cores = core_crit_array

            # For each rep, find left and right core indices from weight matrix
            left_core_idx = np.argmax(weight > 0, axis=1)  # (n_rep_h1,)
            right_core_idx = np.minimum(left_core_idx + 1, n_cores - 1)

            # Gather left and right endpoint crits for each rep (using smoothed values)
            crit_left = smoothed_cores[left_core_idx]   # (n_rep_h1, horizon, ...)
            crit_right = smoothed_cores[right_core_idx]  # (n_rep_h1, horizon, ...)

            power_left = tp.compute_power(
                crit_boundary=crit_left,
                h1_sim_result=h1_res,
                ground_truth_arm_mean_dist=sim_config.arm_mean_reward_dist
            )
            power_left = get_interpolation(power_left, sim_config.step_schedule)

            power_right = tp.compute_power(
                crit_boundary=crit_right,
                h1_sim_result=h1_res,
                ground_truth_arm_mean_dist=sim_config.arm_mean_reward_dist
            )
            power_right = get_interpolation(power_right, sim_config.step_schedule)

            # Worst-case power shift at each horizon step
            bias_power = np.maximum(
                np.abs(power_left - power),
                np.abs(power_right - power)
            )
            bias_at_nstep = bias_power[max(n_step_int - 1, 0)]

            shifted_bias = power_constraint - bias_at_nstep
            n_step_shifted_bias = _steps_for_target(shifted_bias)
            bias_steps_h0 = abs(n_step_shifted_bias - n_step_int)
        except Exception:
            bias_steps_h0 = 0

    # Per-rep reward at the fixed median stopping step
    n_step_batch_idx = int(np.searchsorted(np.cumsum(sim_config.step_schedule), n_step_int, side='left'))
    n_step_batch_idx = min(n_step_batch_idx, h1_res.combined_means.shape[1] - 1)
    per_rep_reward = h1_res.combined_means[:, n_step_batch_idx, :].flatten()
    reward_se = np.std(per_rep_reward) / np.sqrt(n_rep)

    return {
        "obj_score":0,
        "obj_score_sd": 0,
        "log_n_step_sd": np.std(np.log(n_step_dist)),
        "obj_score_sd": np.std(obj_score_dist),
        "reward_se": reward_se,
        "n_step": np.median(n_step_dist),
        "regret_per_step": mean_reward[int(np.median(n_step_dist-1))],
        "deployment_regret":best_mean.mean() - selected_means[int(np.median(n_step_dist-1))],
        "power_max": np.max(power),
        "mean_regret_at_horizon":mean_reward[-1],
        # Error decomposition
        "se_steps_h1": se_steps_h1,
        "se_steps_h0": se_steps_h0,
        "se_steps_total": se_steps_total,
        "bias_steps_h0": int(bias_steps_h0),
        "n_h0_cores": int(tp.n_crit_sim_groups + 1) if tp.n_crit_approx_method == 'linear' else int(tp.n_crit_sim_groups),
        "n_h0_reps_per_core": int(tp.n_crit_sim_rep),
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



