"""
Adaptive Power-Constrained Bandit Simulation
=============================================

This module implements a power-aware simulation workflow with incremental
expansion — simulations advance one batch step at a time, checking power
at each checkpoint. This avoids simulating to max_horizon when power is
reached earlier.

1. UR Baseline (run separately, outside this module):
   - UR has fixed allocation (1/K), simple H0 (one location), fast.
   - Produces T_UR = horizon needed for UR to reach power target.

2. Adaptive Search (this module), 3 phases:
   - Phase 1 (burn-in): handled by init_simulation().
   - Phase 2 (run to T_UR): advance H1 + H0 batch-by-batch, no power check.
     Adaptive algorithms need at most T_UR samples, so checking earlier is wasteful.
   - Phase 3 (power-checking): from T_UR onward, advance one step, refine H0,
     check power. Stops early when power target is met.

Building blocks reused from sim_wrapper:
   - SimRunState: mutable simulation state (theta=None for H1, float for H0)
   - init_simulation(): allocate arrays, generate rewards, apply burn-in
   - run_one_batch_step(): advance one simulation by one batch step
   - SimResult: snapshot for test statistics (constructed from raw arrays)

All configuration reuses existing classes:
   - sim_config (SimulationConfig): horizon, n_rep, n_arm, reward settings
   - sim_config.test_procedure (TestProcedure): power_constraint, type1_error_constraint,
     power_error_threshold, max_h0_total_reps

See ideas-incremental-sim.md in memory for design rationale.
"""

import numpy as np
from typing import List, Tuple

from .sim_wrapper import init_simulation, run_one_batch_step, SimRunState, SimResult
from .simulation_configurator import SimulationConfig


# =============================================================================
# H0 Manager
# =============================================================================

class H0Manager:
    """
    Manages H0 simulation states and tracks per-bin power error.

    Each H0 is a SimRunState with theta set to the null parameter value.
    Locations are kept sorted by theta. Bins are the intervals between
    adjacent locations. The manager handles:
    - Adding new locations (midpoint splitting)
    - Computing per-bin and global power error
    - Checking if budget is exceeded
    """

    def __init__(self):
        self.locations: List[SimRunState] = []
        self.bin_errors: List[float] = []
        self.global_error: float = float('inf')

    @property
    def sorted_locations(self) -> List[SimRunState]:
        """Locations sorted by theta (ascending)."""
        return sorted(self.locations, key=lambda s: s.theta)

    @property
    def n_locations(self) -> int:
        return len(self.locations)

    @property
    def bins(self) -> List[Tuple[SimRunState, SimRunState]]:
        """Return list of (left, right) pairs for adjacent locations."""
        s = self.sorted_locations
        return [(s[i], s[i + 1]) for i in range(len(s) - 1)]

    def total_h0_reps(self, M: int) -> int:
        """Total H0 replications = number_of_locations × M."""
        return self.n_locations * M

    def add_location(self, state: SimRunState):
        """Add a new H0 location (state.theta must be set)."""
        self.locations.append(state)


# =============================================================================
# SimResult snapshot helper
# =============================================================================

def _snapshot(state: SimRunState, sim_config: SimulationConfig) -> SimResult:
    """
    Create a SimResult from the current state of a running simulation.

    SimResult computes cumulative stats (arm_means, combined_means, etc.)
    from the raw arrays. Since arrays are pre-allocated to full schedule
    length, entries beyond state.time_step are zeros — but cumulative stats
    at indices <= time_step are correct (cumsum of zeros doesn't affect
    earlier values).
    """
    return SimResult(
        state.action_hist.astype(int),
        state.reward_hist,
        state.reward2_hist,
        sim_config,
    )


# =============================================================================
# Linear interpolation weight matrix (mirrors test_procedure_configurator)
# =============================================================================

def _build_linear_weight_matrix(theta_hat_m, sorted_thetas):
    """Build linear interpolation weight matrix (same logic as standard's 'linear' mode)."""
    M = len(theta_hat_m)
    n_locs = len(sorted_thetas)
    weight = np.zeros((M, n_locs))
    for i in range(M):
        s = theta_hat_m[i]
        idx = np.searchsorted(sorted_thetas, s, side='right') - 1
        idx = np.clip(idx, 0, n_locs - 2)
        l, r = sorted_thetas[idx], sorted_thetas[idx + 1]
        if r > l:
            weight[i, idx] = (r - s) / (r - l)
            weight[i, idx + 1] = (s - l) / (r - l)
        else:
            weight[i, idx] = 1.0
    return weight


# =============================================================================
# Main Entry Point
# =============================================================================

def run_adaptive_power_search(
    sim_config: SimulationConfig,
    algo,            # Algorithm class (e.g., EpsTS)
    algo_param,      # Single algorithm parameter to evaluate
    T_UR: int,       # Horizon from UR baseline
    progress_callback=None,  # callable(batch_idx, total_batches, t_actual)
):
    """
    Run the adaptive power-constrained simulation for one algorithm parameter.

    Strategy: initialize simulations, then advance one batch step at a time.
    At each step, check power. Stop early when power target is reached.

    Returns:
        dict with T, reward, power, power_error, h0_locations, n_h0_locations
    """
    tp = sim_config.test_procedure
    M = sim_config.n_rep
    K = sim_config.n_arm
    max_horizon = sim_config.horizon

    policy = algo(algo_param)

    # =================================================================
    # STEP 0: ONE-TIME SETUP
    # =================================================================

    # 0a. Arm means for H1 (fixed across the entire process)
    arm_means = sim_config.arm_mean_reward_dist  # shape (M, K)

    # 0b. Two fixed H0 locations
    theta_A = float(np.mean(arm_means))
    theta_B = float(np.mean(np.max(arm_means, axis=1)))

    # =================================================================
    # STEP 1: INITIALIZE SIMULATIONS (allocate + burn-in only)
    # =================================================================

    h1_state = init_simulation(sim_config)

    h0_mgr = H0Manager()
    for theta in [theta_A, theta_B]:
        h0_arm_means = np.full((M, K), theta)
        h0_state = init_simulation(sim_config, arm_mean_reward_dist=h0_arm_means)
        h0_state.theta = theta
        h0_mgr.add_location(h0_state)

    # =================================================================
    # STEP 2: CONVERT T_UR TO BATCH INDEX
    # =================================================================

    step_schedule = sim_config.step_schedule
    power = 0.0
    power_error = float('inf')
    t_actual = 0  # cumulative horizon in samples

    T_UR_capped = min(T_UR, max_horizon)
    cumulative = np.cumsum(step_schedule)
    # batch index where cumulative samples first reach T_UR_capped
    t_ur_batch_idx = int(np.searchsorted(cumulative, T_UR_capped, side='left'))
    t_ur_batch_idx = min(t_ur_batch_idx, len(step_schedule) - 1)

    # =================================================================
    # PHASE 2: ADVANCE TO T_UR (no power check, no H0 refinement)
    # =================================================================

    for batch_idx in range(h1_state.time_step, t_ur_batch_idx + 1):
        run_one_batch_step(policy, sim_config, h1_state)
        for h0 in h0_mgr.locations:
            run_one_batch_step(policy, sim_config, h0)
        t_actual += step_schedule[batch_idx]
        if progress_callback:
            progress_callback(batch_idx + 1, len(step_schedule), t_actual, None)

    # =================================================================
    # CHECK POWER AT T_UR, THEN PHASE 3: POWER-CHECKING LOOP
    # =================================================================

    h1_result = _snapshot(h1_state, sim_config)
    power, power_error, _se_regions = _refine_h0_and_check_power(
        h1_result=h1_result,
        h0_mgr=h0_mgr,
        batch_idx=t_ur_batch_idx,
        sim_config=sim_config,
        policy=policy,
    )

    # Report power at T_UR immediately
    if progress_callback:
        progress_callback(t_ur_batch_idx + 1, len(step_schedule), t_actual, power)

    if power >= tp.power_constraint:
        # Power already met at T_UR — search for crossing between burn-in and T_UR
        T_phi = _find_crossing_step(
            h1_state=h1_state,
            h0_mgr=h0_mgr,
            batch_idx_low=0,
            batch_idx_high=t_ur_batch_idx,
            sim_config=sim_config,
        )
    elif t_ur_batch_idx >= len(step_schedule) - 1:
        # T_UR >= max_horizon: no room for Phase 3, power not met
        T_phi = max_horizon
    else:
        # Phase 3: advance one step at a time, check power each step
        prev_batch_idx = t_ur_batch_idx

        for batch_idx in range(t_ur_batch_idx + 1, len(step_schedule)):
            run_one_batch_step(policy, sim_config, h1_state)
            for h0 in h0_mgr.locations:
                run_one_batch_step(policy, sim_config, h0)
            t_actual += step_schedule[batch_idx]

            h1_result = _snapshot(h1_state, sim_config)
            power, power_error, _se_regions = _refine_h0_and_check_power(
                h1_result=h1_result,
                h0_mgr=h0_mgr,
                batch_idx=batch_idx,
                sim_config=sim_config,
                policy=policy,
            )

            if progress_callback:
                progress_callback(batch_idx + 1, len(step_schedule), t_actual, power)

            if power >= tp.power_constraint:
                T_phi = _find_crossing_step(
                    h1_state=h1_state,
                    h0_mgr=h0_mgr,
                    batch_idx_low=prev_batch_idx,
                    batch_idx_high=batch_idx,
                    sim_config=sim_config,
                )
                break

            prev_batch_idx = batch_idx

        else:
            # Reached max_horizon without hitting power target
            T_phi = max_horizon

    # =================================================================
    # STEP 4: COMPUTE RESULTS
    # =================================================================

    avg_reward = _compute_mean_reward(h1_state, T_phi, sim_config)

    return {
        "algo_param": algo_param,
        "T": T_phi,
        "reward": avg_reward,
        "power": power,
        "power_error": power_error,
        "h0_locations": [round(h0.theta, 3) for h0 in h0_mgr.sorted_locations],
        "n_h0_locations": h0_mgr.n_locations,
    }


# =============================================================================
# STEP 3: H0 REFINEMENT + POWER CHECK (the core logic)
# =============================================================================

def _refine_h0_and_check_power(
    h1_result: SimResult,
    h0_mgr: H0Manager,
    batch_idx: int,
    sim_config: SimulationConfig,
    policy,
) -> Tuple[float, float, dict]:
    """
    Two-part process at each scheduled horizon:

    Part A: Check if we need more H0 locations (midpoint splitting).
            Keep splitting until global power error < threshold or budget exhausted.
    Part B: Estimate power using all current H0 locations via per-rep interpolation.

    Returns:
        (power_estimate, global_power_error)
    """
    tp = sim_config.test_procedure
    M = sim_config.n_rep
    K = sim_config.n_arm
    horizon_slice = slice(batch_idx, batch_idx + 1)

    # ------------------------------------------------------------------
    # H1: test statistics and null parameter estimates at this batch step
    # ------------------------------------------------------------------

    S_m = tp.get_test_statistics(h1_result, horizon=horizon_slice)  # (M, 1, ...)
    S_m = S_m.reshape(M, -1)  # flatten to (M, n_comparisons)

    # theta_hat_m = pooled mean reward per rep = MLE of null parameter
    theta_hat_m = h1_result.combined_means[:, batch_idx, 0]  # (M,)

    # ------------------------------------------------------------------
    # H0: critical region at each location
    # ------------------------------------------------------------------

    crit_regions = {}
    se_regions = {}
    for h0 in h0_mgr.sorted_locations:
        h0_result = _snapshot(h0, sim_config)
        h0_stats = tp.get_test_statistics(h0_result, horizon=horizon_slice)
        crit, se = tp.get_critical_region(h0_stats)
        crit_regions[h0.theta] = crit
        se_regions[h0.theta] = se

    # ==================================================================
    # PART A: H0 REFINEMENT — midpoint splitting
    # ==================================================================

    global_error = _compute_global_power_error(
        S_m, theta_hat_m, h0_mgr, crit_regions, tp.crit_region_direction
    )

    while global_error > tp.power_error_threshold:
        # Check H0 budget
        max_budget = tp.max_h0_total_reps if tp.max_h0_total_reps else float('inf')
        if h0_mgr.total_h0_reps(M) + M > max_budget:
            break

        # Find worst bin
        bin_errors = h0_mgr.bin_errors
        worst_idx = int(np.argmax(bin_errors))
        left_h0, right_h0 = h0_mgr.bins[worst_idx]
        theta_mid = (left_h0.theta + right_h0.theta) / 2.0

        # Create new H0 and catch up to current batch step
        h0_arm_means = np.full((M, K), theta_mid)
        new_state = init_simulation(sim_config, arm_mean_reward_dist=h0_arm_means)
        new_state.theta = theta_mid
        for idx in range(new_state.time_step, batch_idx + 1):
            run_one_batch_step(policy, sim_config, new_state)
        h0_mgr.add_location(new_state)

        # Critical region for new location
        new_result = _snapshot(new_state, sim_config)
        h0_stats = tp.get_test_statistics(new_result, horizon=horizon_slice)
        crit, se = tp.get_critical_region(h0_stats)
        crit_regions[theta_mid] = crit
        se_regions[theta_mid] = se

        # Recompute global error
        global_error = _compute_global_power_error(
            S_m, theta_hat_m, h0_mgr, crit_regions, tp.crit_region_direction
        )

    # ==================================================================
    # PART B: ESTIMATE POWER via weight matrix + tensordot + compute_power
    # ==================================================================

    sorted_thetas = np.array([h0.theta for h0 in h0_mgr.sorted_locations])
    weight = _build_linear_weight_matrix(theta_hat_m, sorted_thetas)

    # Stack crit regions and interpolate via tensordot
    core_crit_array = np.stack([crit_regions[t] for t in sorted_thetas], axis=0)  # (n_locs, 1, ...)
    crit_boundary = np.tensordot(weight, core_crit_array, axes=(1, 0))  # (M, 1, ...)

    # Compute power using standard compute_power (handles two-sided, min_effect, aggregation)
    power_all = tp.compute_power(h1_result, crit_boundary, sim_config.arm_mean_reward_dist)
    power = float(power_all[batch_idx])

    return power, global_error, se_regions


def _compute_global_power_error(
    S_m: np.ndarray,           # (M, n_comparisons) test statistics from H1
    theta_hat_m: np.ndarray,   # (M,) null parameter estimates from H1
    h0_mgr: H0Manager,
    crit_regions: dict,        # theta -> critical_value
    crit_direction: int,       # +1 or -1
) -> float:
    """
    Compute per-bin power error and aggregate into global power error.

    For each bin [theta_left, theta_right]:
      - power_left  = fraction of in-bin H1 reps rejected using c_left
      - power_right = fraction of in-bin H1 reps rejected using c_right
      - theta_bar   = mean of theta_hat_m for in-bin reps
      - drift       = distance from theta_bar to the nearest boundary
      - bin_error   = (drift / span) * |power_left - power_right|

    Global error = weighted average of bin errors (weighted by # reps in bin).
    """
    bins = h0_mgr.bins
    bin_errors = []
    bin_weights = []

    if crit_direction > 0:
        S_scalar = np.max(S_m, axis=1) if S_m.ndim > 1 else S_m
    else:
        S_scalar = np.min(S_m, axis=1) if S_m.ndim > 1 else S_m

    for left_h0, right_h0 in bins:
        c_left = float(np.asarray(crit_regions[left_h0.theta]).flatten()[0])
        c_right = float(np.asarray(crit_regions[right_h0.theta]).flatten()[0])

        in_bin = (theta_hat_m >= left_h0.theta) & (theta_hat_m < right_h0.theta)
        if right_h0.theta == h0_mgr.sorted_locations[-1].theta:
            in_bin = in_bin | (theta_hat_m == right_h0.theta)

        n_in_bin = np.sum(in_bin)
        if n_in_bin == 0:
            bin_errors.append(0.0)
            bin_weights.append(0)
            continue

        S_in_bin = S_scalar[in_bin]
        if crit_direction > 0:
            power_left = np.mean(S_in_bin > c_left)
            power_right = np.mean(S_in_bin > c_right)
        else:
            power_left = np.mean(S_in_bin < c_left)
            power_right = np.mean(S_in_bin < c_right)

        theta_bar_bin = np.mean(theta_hat_m[in_bin])
        drift = min(
            abs(theta_bar_bin - left_h0.theta),
            abs(theta_bar_bin - right_h0.theta),
        )
        span = right_h0.theta - left_h0.theta
        bin_error = (drift / span) * abs(power_left - power_right) if span > 0 else 0.0

        bin_errors.append(bin_error)
        bin_weights.append(n_in_bin)

    h0_mgr.bin_errors = bin_errors

    total = sum(bin_weights)
    if total == 0:
        return 0.0
    return sum(e * w for e, w in zip(bin_errors, bin_weights)) / total


# =============================================================================
# STEP 4 HELPER: Find exact crossing step
# =============================================================================

def _find_crossing_step(
    h1_state: SimRunState,
    h0_mgr: H0Manager,
    batch_idx_low: int,
    batch_idx_high: int,
    sim_config: SimulationConfig,
) -> int:
    """
    Find the earliest batch step between low and high where power >= target.

    Since all simulations have been advanced to batch_idx_high, their arrays
    contain valid data at all earlier indices. We create SimResult snapshots
    and evaluate power at each intermediate step without re-running.
    """
    tp = sim_config.test_procedure
    step_schedule = sim_config.step_schedule
    M = sim_config.n_rep

    h1_result = _snapshot(h1_state, sim_config)

    for batch_idx in range(batch_idx_low, batch_idx_high + 1):
        horizon_slice = slice(batch_idx, batch_idx + 1)
        theta_hat_m = h1_result.combined_means[:, batch_idx, 0]

        sorted_h0s = h0_mgr.sorted_locations
        sorted_thetas = np.array([h0.theta for h0 in sorted_h0s])
        weight = _build_linear_weight_matrix(theta_hat_m, sorted_thetas)

        crit_list = []
        for h0 in sorted_h0s:
            h0_result = _snapshot(h0, sim_config)
            h0_stats = tp.get_test_statistics(h0_result, horizon=horizon_slice)
            crit, _se = tp.get_critical_region(h0_stats)
            crit_list.append(crit)
        core_crit_array = np.stack(crit_list, axis=0)  # (n_locs, 1, ...)
        crit_boundary = np.tensordot(weight, core_crit_array, axes=(1, 0))  # (M, 1, ...)

        power_all = tp.compute_power(h1_result, crit_boundary, sim_config.arm_mean_reward_dist)
        power = float(power_all[batch_idx])

        if power >= tp.power_constraint:
            return int(np.sum(step_schedule[:batch_idx + 1]))

    return int(np.sum(step_schedule[:batch_idx_high + 1]))


# =============================================================================
# Reward helper
# =============================================================================

def _compute_mean_reward(h1_state: SimRunState, T_phi: int, sim_config: SimulationConfig):
    """Compute mean reward across replications at the batch step corresponding to T_phi."""
    step_schedule = sim_config.step_schedule
    cumulative = np.cumsum(step_schedule)
    batch_idx = int(np.searchsorted(cumulative, T_phi, side='left'))
    batch_idx = min(batch_idx, len(step_schedule) - 1)

    h1_result = _snapshot(h1_state, sim_config)
    return float(np.mean(h1_result.combined_means[:, batch_idx, 0]))


# =============================================================================
# Parameter sweep wrapper
# =============================================================================

def run_parameter_sweep(
    sim_config: SimulationConfig,
    algo,
    algo_param_list: list,
    T_UR: int,
    progress_callback=None,  # callable(param_idx, total_params, algo_name, param)
):
    """
    Run adaptive power search across multiple algorithm parameters.

    Returns:
        dict with best_param, best_result, and all_results
    """
    all_results = {}
    best_param = None
    best_score = None

    for i, param in enumerate(algo_param_list):
        if progress_callback:
            progress_callback(i, len(algo_param_list), algo.__name__, param)

        result = run_adaptive_power_search(
            sim_config=sim_config,
            algo=algo,
            algo_param=param,
            T_UR=T_UR,
        )
        all_results[param] = result

        # TODO: Compute objective score from result["T"] and result["reward"]
        # using experiment_cost_w.
        score = result.get("reward", 0)
        if best_score is None or score > best_score:
            best_score = score
            best_param = param

    return {
        "best_param": best_param,
        "best_result": all_results[best_param],
        "all_results": all_results,
    }
