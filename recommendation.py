"""
recommendation.py - Wraps simulation logic and provides recommendations
"""
import os
import numpy as np
# import pandas as pd
# import copy

from simulation_configurator import SimulationConfig
# from test_procedure_configurator import ANOVA, TControl, TConstant, Tukey
import bayes_vector_ops as bayes
# import sim_wrapper as sw

# Import plotting functions from main
from main import sweep_and_run, select_curves_relative, compute_objective
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt


# def create_test_procedure(test_name, type1_error, power_constraint, min_effect, 
#                           family_wise_error_control, **kwargs):
#     """
#     Factory function to create the appropriate TestProcedure object.
    
#     Why a factory function? It centralizes test creation logic, making main_app.py
#     cleaner and ensuring consistent parameter handling across the app.
#     """
#     base_params = {
#         'type1_error_constraint': type1_error,
#         'power_constraint': power_constraint,
#         'min_effect': min_effect,
#         'family_wise_error_control': family_wise_error_control
#     }
    
#     if test_name == 'anova':
#         return ANOVA(**base_params)
    
#     elif test_name == 't_control':
#         test_type = kwargs.get('test_type', 'one-sided')
#         control_idx = kwargs.get('control_group_index', 0)
#         return TControl(**base_params, test_type=test_type, control_group_index=control_idx)
    
#     elif test_name == 't_constant':
#         test_type = kwargs.get('test_type', 'one-sided')
#         constant = kwargs.get('constant_threshold', None)
#         return TConstant(**base_params, test_type=test_type, constant_threshold=constant)
    
#     elif test_name == 'tukey':
#         tukey_type = kwargs.get('tukey_test_type', 'distinct-best-arm')
#         return Tukey(**base_params, test_type=tukey_type)
    
#     else:
#         raise ValueError(f"Unknown test type: {test_name}")


def create_simulation_config(n_arm, horizon, n_rep, reward_model, 
                             arm_mean_reward_dist_loc, arm_mean_reward_dist_scale,
                             reward_std, test_procedure, step_cost, n_opt_trials=None):
    """
    Creates a SimulationConfig from user inputs.
    
    Why separate function? Keeps configuration creation isolated from business logic,
    making it easier to test and modify.
    """
    return SimulationConfig(
        n_arm=n_arm,
        horizon=horizon,
        n_rep=n_rep,
        burn_in_per_arm=5,  # Reasonable default for most cases
        n_opt_trials=n_opt_trials,
        reward_model=reward_model,
        arm_mean_reward_dist_loc=arm_mean_reward_dist_loc,
        arm_mean_reward_dist_scale=arm_mean_reward_dist_scale,
        reward_std=reward_std,
        test_procedure=test_procedure,
        step_cost=step_cost,
        reward_evaluation_method='reward',
        vector_ops=bayes.BackendOpsNP()
    )


def run_simulation_sweep(sim_config, algo_list, granularity=21):
    """
    Runs parameter sweep for all specified algorithms.
    
    Args:
        sim_config: Base SimulationConfig
        algo_list: List of algorithm classes (e.g., [EpsTS, TSProbClip])
        granularity: Number of parameter values to sweep (default 21)
    
    Returns:
        DataFrame with all simulation results
    """
    sweeps = [
        {"algo": algo_list},
        {"algo_param_list": list(map(float, np.linspace(0.0, 1.0, granularity)))}
    ]
    
    df = sweep_and_run(sweeps, sim_config)
    return df


def find_best_algorithm(df, w_value=10):
    """
    Finds the best performing algorithm and parameter for a given weight value.
    
    Args:
        df: Results DataFrame from simulation sweep
        w_value: Weight for objective function (higher = more emphasis on reward)
    
    Returns:
        dict with best algorithm name, parameter, and performance metrics
    """
    # Compute objective for each row
    df = df.copy()
    df['objective'] = df.apply(lambda r: compute_objective(r, w_value), axis=1)
    
    # Find minimum objective (best performance)
    best_idx = df['objective'].idxmin()
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
    
    Args:
        df: Results DataFrame
        selectors: List of (algo_name, mode, value) tuples for curves to plot
        output_path: Where to save the plot
    
    Returns:
        Path to saved plot
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Default selectors if none provided
    if selectors is None:
        # Get unique algorithms and create sensible defaults
        unique_algos = df['algo_name'].unique()
        selectors = []
        for algo in unique_algos[:3]:  # Limit to 3 algorithms
            selectors.append((algo, "param", 0.0))   # Pure exploitation
            selectors.append((algo, "param", 1.0))   # Maximum exploration
            selectors.append((algo, "w", 10))        # Optimized for w=10
    
    curves = select_curves_relative(df, selectors, w_values=range(1, 16))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    for label, curve in curves.items():
        # Simplify label for readability
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
                       reward_std, test_procedure, step_cost, algo_list, 
                       granularity=21, n_opt_trials=None):
    """
    Main recommendation function - orchestrates the full pipeline.
    
    This is the primary interface called by main_app.py.
    
    Args:
        n_arm: Number of arms
        horizon: Maximum time steps
        n_rep: Number of simulation replications
        reward_model: numpy function (np.random.binomial or np.random.normal)
        h1_loc: Mean of arm reward distribution
        h1_scale: Std dev of arm reward distribution
        reward_std: Observation noise (for Gaussian rewards)
        test_procedure: TestProcedure object
        step_cost: Cost per step
        algo_list: List of algorithm classes to compare
        granularity: Parameter sweep density
        n_opt_trials: Bayesian optimization trials (optional)
    
    Returns:
        tuple: (recommendations_list, plot_path, results_summary)
    """
    # Step 1: Create simulation configuration
    sim_config = create_simulation_config(
        n_arm=n_arm,
        horizon=horizon,
        n_rep=n_rep,
        reward_model=reward_model,
        arm_mean_reward_dist_loc=h1_loc,
        arm_mean_reward_dist_scale=h1_scale,
        reward_std=reward_std,
        test_procedure=test_procedure,
        step_cost=step_cost,
        n_opt_trials=n_opt_trials
    )
    
    # Step 2: Run simulation sweep
    df = run_simulation_sweep(sim_config, algo_list, granularity)
    
    # Step 3: Find best algorithm for different w values
    best_low_w = find_best_algorithm(df, w_value=3)   # Efficiency-focused
    best_mid_w = find_best_algorithm(df, w_value=10)  # Balanced
    best_high_w = find_best_algorithm(df, w_value=15) # Reward-focused
    
    # Step 4: Generate plot
    plot_path = generate_performance_plot(df)
    
    # Step 5: Format recommendations for display
    recommendations = [
        f"Best Overall (Balanced): {best_mid_w['algorithm']} with parameter = {best_mid_w['parameter']:.3f}",
        f"  → Expected steps to reach power: {best_mid_w['n_steps']:.0f}",
        "",
        f"If you prioritize efficiency (fewer steps): {best_low_w['algorithm']} with parameter = {best_low_w['parameter']:.3f}",
        f"If you prioritize reward maximization: {best_high_w['algorithm']} with parameter = {best_high_w['parameter']:.3f}",
    ]
    
    # Step 6: Create detailed summary for advanced display
    results_summary = {
        'dataframe': df,
        'best_balanced': best_mid_w,
        'best_efficient': best_low_w,
        'best_reward': best_high_w,
        'test_procedure': test_procedure.test_signature,
        'n_algorithms_tested': len(algo_list),
        'n_parameter_values': granularity
    }
    
    return recommendations, plot_path, results_summary


#OLD CODE BELOW
# # from Bandit_Simulation.main import sweep_and_run, select_curves_relative, plot_curves
# import pandas as pd
# import numpy as np
# #import requests

# #define function to generate the recommendation based on user input
# #will have to be some type of wrapper around all the code (main.py?)
# # def get_recommendation(n_arms=2):
# #TODO: add in all the other parameters needed for the recommendation logic and edit the logic accordingly
# def get_recommendation(n_arm, horizon, algo):
#     # Placeholder for the recommendation logic
#     # This function would implement the logic to recommend an algorithm based on the number of arms and horizon
#     # For now, it just returns a placeholder string
#     if n_arm >= 3 and horizon > 10:
#         algo = algo
#         hyperparameters = {"epsilon": 0.1}
#     else:
#         algo = "Thompson Sampling"
#         hyperparameters = {"alpha": 1, "beta": 1}

#     recommendations = [f"Recommended Algorithm: {algo}", f"Recommended Hyperparameters: {hyperparameters}"]
#     plot_path = "static/performance_plot.png" #path to save the plot image - ensure this matches the path used in main.py
    
#     # return recommendations, plot_path
#     return [f"Recommended Algorithm: {algo}", f"Recommended Hyperparameters: {hyperparameters}"]
    
# # def get_recommendation(n_arm, horizon, n_rep, reward_distribution, h1_loc, h1_scale, test_name, type1_error, test_const, sweeps, selectors=None):
# #     """
# #     Runs the simulation, generates results, and creates a plot based on selectors.

# #     Args:
# #         n_arm (int): Number of arms for the simulation.
# #         horizon (int): Horizon for the simulation.
# #         selectors (list of tuples): List of selectors for filtering and plotting. 
# #                                     Each selector is a tuple (algo_name, mode, value).

# #     Returns:
# #         tuple: A list of recommendations and the path to the generated plot.
# #     """
# #     # Step 1: Define the simulation configuration based on user inputs
# #     sim_config = {
# #     "n_arm": n_arm,
# #     "horizon": horizon,
# #     "n_rep": n_rep,
# #     "burn_in_per_arm": 5,
# #     "n_opt_trials": None,
# #     "reward_model": reward_distribution,  # e.g., "bernoulli" or "gaussian" - No longer needed?
# #     "arm_mean_reward_dist_loc": h1_loc,
# #     "arm_mean_reward_dist_scale": h1_scale,
# #     "reward_std": None,  #float for Gaussian rewards (none for Bernoulli)
# #     "test_procedure": test_name,  
# #     "step_cost": 0, #float
# #     # "type1_error_constraint": type1_error, #what should these be?
# #     # "test_const": test_const,
# #     "reward_evaluation_method": 'reward'
# #     "family_wise_error_control": bool = False
# #     "vector_ops": bayes.BackendOpsNP()
# #     }

# #     # Step 2: Run the simulation with the specified configuration and sweeps
# #     df = sweep_and_run(sweeps, sim_config)  # This function generates the simulation results
# #     df.to_csv("simulation_results.csv", index=False)

# #     # Step 3: Define default selectors if none are provided
# #     if selectors is None:
# #             selectors = [
# #                 ("EpsTS", "param", 0), #UR
# #                 ("EpsTS", "param", 1), #TS
# #                 ("EpsTS", "w", 10), #default
# #             ]
    
# #     # Step 4: Generate curves based on selectors
# #     curves = select_curves_relative(df, selectors, w_values=range(1, 16))

# #     # Step 5: Save the plot
# #     plot_path = "static/performance_plot.png"
# #     plot_curves(curves, output_path=plot_path)

# #     # Step 7: Generate recommendations (placeholder logic)
# #     #TODO: replace with the algorithm and hyperparameter selection logic based on the simulation results - one with best objective score
# #     #TODO: for prob clipping one, convert any probability output to actual probability values by dividing parameter by number of arms

# #     recommendations = [
# #         "Recommendation : Use EpsTS with epsilon = 0.2"
# #     ]

# #     return recommendations, plot_path

# if __name__ == "__main__": #this is to test the function locally in terminal/console - ensures that the code below here (this code block) only runs when the script is executed directly, not when it is imported.
#     print('\n*** Welcome to the MAB Recommendation System! ***\n')

#     #n_arm = input ('\nPlease enter the number of arms - test: ')

#     recommendation = "Epsilon-Greedy: epsilon = 0.7"  # Placeholder for the recommendation response - got to include parameters etc
#     print(f'\nYour recommended algorithm is: {recommendation}\n')