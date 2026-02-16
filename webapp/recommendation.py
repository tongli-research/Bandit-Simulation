"""
recommendation.py - Wraps simulation logic and provides algorithm recommendations.
"""
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np

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
        burn_in_per_arm=5,
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

    plt.axhline(0, color="grey", linestyle="--", label="Baseline (best possible)")
    plt.xlabel("Weight (w) - Higher = More Reward Focus", fontsize=11)
    plt.ylabel("Relative Objective Score\n(lower is better, 0 = optimal)", fontsize=11)
    plt.title("Algorithm Performance Comparison", fontsize=13)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def get_recommendation(n_arm, horizon, n_rep, reward_model, h1_loc, h1_scale,
                       reward_std, test_procedure, algo_list,
                       granularity=21):
    """
    Main recommendation function - orchestrates the full pipeline.

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

    df = run_simulation_sweep(sim_config, algo_list, granularity)

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
        'n_parameter_values': granularity,
        'chart_data_json': df[
            ['algo_name', 'algo_param', 'n_step', 'regret_per_step']
        ].to_json(orient='records'),
    }

    return recommendations, plot_path, results_summary
