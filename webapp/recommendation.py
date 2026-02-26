"""
recommendation.py - Wraps simulation logic and provides algorithm recommendations.
"""
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from bandit_simulation import (
    SimulationConfig,
    compute_objective,
    select_curves_relative,
    sweep_and_run,
)


def create_simulation_config(n_arm, horizon, n_rep, reward_model,
                             arm_mean_reward_dist_loc, arm_mean_reward_dist_scale,
                             reward_std, test_procedure):
    """Creates a SimulationConfig from user inputs."""
    return SimulationConfig(
        n_arm=n_arm,
        horizon=horizon,
        n_rep=n_rep,
        burn_in_per_arm=1,
        reward_model=reward_model,
        arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {
                "loc": arm_mean_reward_dist_loc,
                "scale": arm_mean_reward_dist_scale,
            }
        },
        reward_std=reward_std,
        test_procedure=test_procedure,
        reward_evaluation_method='reward',
    )


def run_simulation_sweep(sim_config, algo_list, granularity=21):
    """
    Runs parameter sweep for all specified algorithms.

    Returns:
        DataFrame with all simulation results.
    """
    sweeps = [
        {"algo": algo_list},
        {"algo_param_list": list(map(float, np.linspace(0.0, 1.0, granularity)))}
    ]
    return sweep_and_run(sweeps, sim_config)


def find_best_algorithm(df, w_value=10):
    """
    Finds the best performing algorithm and parameter for a given weight value.

    Returns:
        dict with best algorithm name, parameter, and performance metrics.
    """
    df = df.copy()
    df['objective'] = df.apply(lambda r: compute_objective(r, w_value), axis=1)
    best_idx = df['objective'].idxmax()
    best_row = df.loc[best_idx]

    return {
        'algorithm': best_row['algo_name'],
        'parameter': best_row['algo_param'],
        'n_steps': best_row['n_step'],
        'reward_per_step': best_row.get('regret_per_step', None),
        'objective_score': best_row['objective'],
        'w_value': w_value
    }


def fit_gp_curves(df, power_constraint):
    """
    Fit GP to each algorithm's (param -> n_step, reward) curve.

    Only fits on rows that achieved power. Returns:
      - gp_grid: fine-grid predictions (200 points per algo)
      - gp_at_raw: GP predictions at raw param values (for comparison table)
    """
    gp_grid = []
    gp_at_raw = []

    for algo_name in df['algo_name'].unique():
        algo_df = df[df['algo_name'] == algo_name].copy()
        powered = algo_df[algo_df['power_max'] >= power_constraint]

        if len(powered) < 2:
            # Not enough points for GP — return raw data as-is
            for _, row in algo_df.iterrows():
                gp_at_raw.append({
                    'algo_name': algo_name,
                    'algo_param': float(row['algo_param']),
                    'n_step': float(row['n_step']),
                    'regret_per_step': float(row['regret_per_step']),
                    'n_step_raw': float(row['n_step']),
                    'regret_per_step_raw': float(row['regret_per_step']),
                    'power_max': float(row['power_max']),
                })
            continue

        X = powered['algo_param'].values.reshape(-1, 1)
        y_log_step = np.log(powered['n_step'].values)
        y_reward = powered['regret_per_step'].values

        kernel = Matern(nu=2.5) + WhiteKernel()

        gp_step = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                           random_state=42)
        gp_step.fit(X, y_log_step)

        gp_reward = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                             random_state=42)
        gp_reward.fit(X, y_reward)

        # Fine grid predictions (200 points)
        p_min, p_max = float(X.min()), float(X.max())
        X_fine = np.linspace(p_min, p_max, 200).reshape(-1, 1)

        step_pred = np.exp(gp_step.predict(X_fine))
        reward_pred = gp_reward.predict(X_fine)

        for j in range(len(X_fine)):
            gp_grid.append({
                'algo_name': algo_name,
                'algo_param': float(X_fine[j, 0]),
                'n_step': float(step_pred[j]),
                'regret_per_step': float(reward_pred[j]),
            })

        # Predictions at raw param values (for comparison table)
        X_raw = algo_df['algo_param'].values.reshape(-1, 1)
        step_at_raw = np.exp(gp_step.predict(X_raw))
        reward_at_raw = gp_reward.predict(X_raw)

        for k, (_, row) in enumerate(algo_df.iterrows()):
            gp_at_raw.append({
                'algo_name': algo_name,
                'algo_param': float(row['algo_param']),
                'n_step': float(step_at_raw[k]),
                'regret_per_step': float(reward_at_raw[k]),
                'n_step_raw': float(row['n_step']),
                'regret_per_step_raw': float(row['regret_per_step']),
                'power_max': float(row['power_max']),
            })

    return {'gp_grid': gp_grid, 'gp_at_raw': gp_at_raw}


def generate_performance_plot(df, selectors=None, output_path='static/performance_plot.png'):
    """
    Generates performance comparison plot.

    Returns:
        Path to saved plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if selectors is None:
        unique_algos = df['algo_name'].unique()
        selectors = []
        for algo in unique_algos[:3]:
            selectors.append((algo, "param", 0.0))
            selectors.append((algo, "param", 1.0))
            selectors.append((algo, "w", 10))

    curves = select_curves_relative(df, selectors, w_values=range(1, 16))

    plt.figure(figsize=(10, 6))
    for label, curve in curves.items():
        if "param=" in label:
            algo, rest = label.split("(", 1)
            param_val = rest.split("=")[1].replace(")", "").strip()
            try:
                param_val = f"{float(param_val):.2f}"
            except ValueError:
                pass
            label = f"{algo.strip()} (param={param_val})"

        plt.plot(curve["w"], curve["obj_rel"], marker="o", label=label)

    plt.xlabel("Experiment Extension Cost ('w')", fontsize=11)
    plt.ylabel("Relative ECP-reward\n(lower is better, 0 = optimal)", fontsize=11)
    plt.title("Algorithm Performance Comparison", fontsize=13)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def get_recommendation(n_arm, horizon, n_rep, reward_model, h1_loc, h1_scale,
                       reward_std, test_procedure, algo_list,
                       granularity=21, progress_callback=None, result_callback=None,
                       cancel_check=None, algo_param_config=None):
    """
    Main recommendation function - orchestrates the full pipeline.

    Args:
        algo_param_config: Optional dict mapping algo class name to
            {"min": float, "max": float, "granularity": int}.
            When provided, each algorithm uses its own parameter range and granularity.
            When None, falls back to global [0.0, 1.0] with `granularity` points.

    Returns:
        tuple: (recommendations_list, plot_path, results_summary)
    """
    sim_config = create_simulation_config(
        n_arm=n_arm,
        horizon=horizon,
        n_rep=n_rep,
        reward_model=reward_model,
        arm_mean_reward_dist_loc=h1_loc,
        arm_mean_reward_dist_scale=h1_scale,
        reward_std=reward_std,
        test_procedure=test_procedure,
    )

    default_cfg = {"min": 0.0, "max": 1.0, "granularity": granularity}

    # Compute total steps across all algorithms
    total_steps = 0
    for algo_class in algo_list:
        cfg = (algo_param_config or {}).get(algo_class.__name__, default_cfg)
        total_steps += cfg["granularity"]

    all_results = []
    step = 0
    cancelled = False

    for algo_class in algo_list:
        name = algo_class.__name__
        cfg = (algo_param_config or {}).get(name, default_cfg)
        param_list = list(map(float, np.linspace(cfg["min"], cfg["max"], cfg["granularity"])))
        for param in param_list:
            if cancel_check and cancel_check():
                cancelled = True
                break
            if progress_callback:
                progress_callback(step, total_steps, algo_class.__name__, param)
            sweeps = [
                {"algo": [algo_class]},
                {"algo_param_list": [param]},
            ]
            partial_df = sweep_and_run(sweeps, sim_config)
            all_results.append(partial_df)
            step += 1
            if result_callback:
                row = partial_df.iloc[0]
                result_callback({
                    'algo_name': row['algo_name'],
                    'algo_param': float(row['algo_param']),
                    'regret_per_step': float(row['regret_per_step']),
                    'n_step': float(row['n_step']),
                    'power_max': float(row['power_max']),
                    'se_steps_h1': float(row.get('se_steps_h1', 0)),
                    'se_steps_h0': float(row.get('se_steps_h0', 0)),
                    'se_steps_total': float(row.get('se_steps_total', 0)),
                    'bias_steps_h0': int(row.get('bias_steps_h0', 0)),
                    'reward_se': float(row.get('reward_se', 0)),
                })
        if cancelled:
            break

    if progress_callback:
        progress_callback(total_steps, total_steps, '', 0)

    df = pd.concat(all_results, ignore_index=True)

    best_low_w = find_best_algorithm(df, w_value=3)
    best_mid_w = find_best_algorithm(df, w_value=10)
    best_high_w = find_best_algorithm(df, w_value=15)

    plot_path = generate_performance_plot(df)

    recommendations = [
        f"Best Overall (Balanced): {best_mid_w['algorithm']} "
        f"with parameter = {best_mid_w['parameter']:.3f}",
        f"  → Expected steps to reach power: {best_mid_w['n_steps']:.0f}",
        "",
        f"If you prioritize efficiency (fewer steps): "
        f"{best_low_w['algorithm']} with parameter = {best_low_w['parameter']:.3f}",
        f"If you prioritize reward maximization: "
        f"{best_high_w['algorithm']} with parameter = {best_high_w['parameter']:.3f}",
    ]

    results_summary = {
        'dataframe': df,
        'best_balanced': best_mid_w,
        'best_efficient': best_low_w,
        'best_reward': best_high_w,
        'test_procedure': test_procedure.test_signature,
        'n_algorithms_tested': len(algo_list),
        'n_parameter_values': total_steps,
        'power_constraint': test_procedure.power_constraint,
        'chart_data_json': df[
            ['algo_name', 'algo_param', 'n_step', 'regret_per_step', 'power_max',
             'se_steps_h1', 'se_steps_h0', 'se_steps_total', 'bias_steps_h0',
             'log_n_step_sd', 'obj_score_sd', 'reward_se']
        ].to_json(orient='records'),
        'n_h0_cores': int(df['n_h0_cores'].iloc[0]) if 'n_h0_cores' in df.columns else None,
        'n_h0_reps_per_core': int(df['n_h0_reps_per_core'].iloc[0]) if 'n_h0_reps_per_core' in df.columns else None,
    }

    return recommendations, plot_path, results_summary


def get_recommendation_adaptive(n_arm, horizon, n_rep, reward_model, h1_loc, h1_scale,
                                reward_std, test_procedure, algo_list,
                                granularity=21, progress_callback=None,
                                step_callback=None, algo_param_config=None):
    """
    Alternative recommendation function using adaptive power search.

    Uses incremental simulation: advances one batch step at a time and stops
    early when the power target is reached. Same return interface as
    get_recommendation() so the chart and UI work unchanged.

    Args:
        algo_param_config: Optional dict mapping algo class name to
            {"min": float, "max": float, "granularity": int}.

    Returns:
        tuple: (recommendations_list, plot_path, results_summary)
    """
    from bandit_simulation.adaptive_power_search import run_adaptive_power_search

    sim_config = create_simulation_config(
        n_arm=n_arm,
        horizon=horizon,
        n_rep=n_rep,
        reward_model=reward_model,
        arm_mean_reward_dist_loc=h1_loc,
        arm_mean_reward_dist_scale=h1_scale,
        reward_std=reward_std,
        test_procedure=test_procedure,
    )
    sim_config.manual_init()

    default_cfg = {"min": 0.0, "max": 1.0, "granularity": granularity}
    T_UR = horizon  # adaptive search accepts T_UR but doesn't currently use it

    total_steps = 0
    for algo_class in algo_list:
        cfg = (algo_param_config or {}).get(algo_class.__name__, default_cfg)
        total_steps += cfg["granularity"]

    all_results = []
    step = 0

    for algo_class in algo_list:
        name = algo_class.__name__
        cfg = (algo_param_config or {}).get(name, default_cfg)
        param_list = list(map(float, np.linspace(cfg["min"], cfg["max"], cfg["granularity"])))
        for param in param_list:
            if progress_callback:
                progress_callback(step, total_steps, algo_class.__name__, param)

            result = run_adaptive_power_search(
                sim_config=sim_config,
                algo=algo_class,
                algo_param=param,
                T_UR=T_UR,
                progress_callback=step_callback,
            )

            # Map adaptive result to standard DataFrame columns
            # Error metrics not available in adaptive mode
            all_results.append({
                'algo_name': algo_class.__name__,
                'algo_param': param,
                'n_step': result['T'],
                'regret_per_step': result['reward'],
                'power_max': result['power'],
                'se_steps_h1': 0.0,
                'se_steps_h0': 0.0,
                'se_steps_total': 0.0,
                'bias_steps_h0': 0,
                'log_n_step_sd': 0,
                'obj_score_sd': 0,
                'reward_se': 0,
            })
            step += 1

    if progress_callback:
        progress_callback(total_steps, total_steps, '', 0)

    df = pd.DataFrame(all_results)

    best_low_w = find_best_algorithm(df, w_value=3)
    best_mid_w = find_best_algorithm(df, w_value=10)
    best_high_w = find_best_algorithm(df, w_value=15)

    plot_path = generate_performance_plot(df)

    recommendations = [
        f"Best Overall (Balanced): {best_mid_w['algorithm']} "
        f"with parameter = {best_mid_w['parameter']:.3f}",
        f"  → Expected steps to reach power: {best_mid_w['n_steps']:.0f}",
        "",
        f"If you prioritize efficiency (fewer steps): "
        f"{best_low_w['algorithm']} with parameter = {best_low_w['parameter']:.3f}",
        f"If you prioritize reward maximization: "
        f"{best_high_w['algorithm']} with parameter = {best_high_w['parameter']:.3f}",
    ]

    results_summary = {
        'dataframe': df,
        'best_balanced': best_mid_w,
        'best_efficient': best_low_w,
        'best_reward': best_high_w,
        'test_procedure': test_procedure.test_signature,
        'n_algorithms_tested': len(algo_list),
        'n_parameter_values': total_steps,
        'power_constraint': test_procedure.power_constraint,
        'chart_data_json': df[
            ['algo_name', 'algo_param', 'n_step', 'regret_per_step', 'power_max',
             'se_steps_h1', 'se_steps_h0', 'se_steps_total', 'bias_steps_h0',
             'log_n_step_sd', 'obj_score_sd', 'reward_se']
        ].to_json(orient='records'),
        'n_h0_cores': None,
        'n_h0_reps_per_core': None,
    }

    return recommendations, plot_path, results_summary


def run_single_check(sim_config, algo_class, algo_param):
    """Run simulation for a single algorithm at one specific parameter value."""
    sweeps = [
        {"algo": [algo_class]},
        {"algo_param_list": [float(algo_param)]}
    ]
    df = sweep_and_run(sweeps, sim_config)
    row = df.iloc[0]
    result = {
        'algo_name': row['algo_name'],
        'algo_param': float(algo_param),
        'n_step': float(row['n_step']),
        'regret_per_step': float(row['regret_per_step']),
        'power_max': float(row['power_max']),
    }
    # Include error metrics if available
    for field in ['se_steps_h1', 'se_steps_h0', 'se_steps_total', 'bias_steps_h0',
                  'reward_se', 'log_n_step_sd', 'obj_score_sd']:
        result[field] = float(row.get(field, 0))
    return result
