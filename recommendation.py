#import any libraries needed
# from Bandit_Simulation.main import sweep_and_run, select_curves_relative, plot_curves
import pandas as pd
import numpy as np
#import requests

#define function to generate the recommendation based on user input
#will have to be some type of wrapper around all the code (main.py?)
# def get_recommendation(n_arms=2):
#TODO: add in all the other parameters needed for the recommendation logic and edit the logic accordingly
def get_recommendation(n_arm, horizon, algo):
    # Placeholder for the recommendation logic
    # This function would implement the logic to recommend an algorithm based on the number of arms and horizon
    # For now, it just returns a placeholder string
    if n_arm >= 3 and horizon > 10:
        algo = algo
        hyperparameters = {"epsilon": 0.1}
    else:
        algo = "Thompson Sampling"
        hyperparameters = {"alpha": 1, "beta": 1}

    recommendations = [f"Recommended Algorithm: {algo}", f"Recommended Hyperparameters: {hyperparameters}"]
    plot_path = "static/performance_plot.png" #path to save the plot image - ensure this matches the path used in main.py
    
    # return recommendations, plot_path
    return [f"Recommended Algorithm: {algo}", f"Recommended Hyperparameters: {hyperparameters}"]
    
# def get_recommendation(n_arm, horizon, n_rep, reward_distribution, h1_loc, h1_scale, test_name, type1_error, test_const, sweeps, selectors=None):
#     """
#     Runs the simulation, generates results, and creates a plot based on selectors.

#     Args:
#         n_arm (int): Number of arms for the simulation.
#         horizon (int): Horizon for the simulation.
#         selectors (list of tuples): List of selectors for filtering and plotting. 
#                                     Each selector is a tuple (algo_name, mode, value).

#     Returns:
#         tuple: A list of recommendations and the path to the generated plot.
#     """
#     # Step 1: Define the simulation configuration based on user inputs
#     sim_config = {
#     "n_arm": n_arm,
#     "horizon": horizon,
#     "n_rep": n_rep,
#     "burn_in_per_arm": 5,
#     "n_opt_trials": None,
#     "reward_model": reward_distribution,  # e.g., "bernoulli" or "gaussian" - No longer needed?
#     "arm_mean_reward_dist_loc": h1_loc,
#     "arm_mean_reward_dist_scale": h1_scale,
#     "reward_std": None,  #float for Gaussian rewards (none for Bernoulli)
#     "test_procedure": test_name,  
#     "step_cost": 0, #float
#     # "type1_error_constraint": type1_error, #what should these be?
#     # "test_const": test_const,
#     "reward_evaluation_method": 'reward'
#     "family_wise_error_control": bool = False
#     "vector_ops": bayes.BackendOpsNP()
#     }

#     # Step 2: Run the simulation with the specified configuration and sweeps
#     df = sweep_and_run(sweeps, sim_config)  # This function generates the simulation results
#     df.to_csv("simulation_results.csv", index=False)

#     # Step 3: Define default selectors if none are provided
#     if selectors is None:
#             selectors = [
#                 ("EpsTS", "param", 0), #UR
#                 ("EpsTS", "param", 1), #TS
#                 ("EpsTS", "w", 10), #default
#             ]
    
#     # Step 4: Generate curves based on selectors
#     curves = select_curves_relative(df, selectors, w_values=range(1, 16))

#     # Step 5: Save the plot
#     plot_path = "static/performance_plot.png"
#     plot_curves(curves, output_path=plot_path)

#     # Step 7: Generate recommendations (placeholder logic)
#     #TODO: replace with the algorithm and hyperparameter selection logic based on the simulation results - one with best objective score
#     #TODO: for prob clipping one, convert any probability output to actual probability values by dividing parameter by number of arms

#     recommendations = [
#         "Recommendation : Use EpsTS with epsilon = 0.2"
#     ]

#     return recommendations, plot_path

if __name__ == "__main__": #this is to test the function locally in terminal/console - ensures that the code below here (this code block) only runs when the script is executed directly, not when it is imported.
    print('\n*** Welcome to the MAB Recommendation System! ***\n')

    #n_arm = input ('\nPlease enter the number of arms - test: ')

    recommendation = "Epsilon-Greedy: epsilon = 0.7"  # Placeholder for the recommendation response - got to include parameters etc
    print(f'\nYour recommended algorithm is: {recommendation}\n')