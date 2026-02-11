"""
Reproduces Table 3 (Post / realized columns).

What this script does:
  1. Fixes the empirically observed 6-arm reward means (realized setting).
  2. Evaluates the selected experiment designs using the fixed sample sizes
     determined during the prior optimization stage.
  3. Computes the realized Reward (and ECP, if applicable) for:
       - UR (naive)
       - TS (AIT-corrected design)
       - epsilon-TS (AIT + optimized epsilon)

This corresponds to:
  - "Post (realized)" columns in Table 3.
"""


from src import bandit_algorithm as algorithm
from src import sim_wrapper as sw

from src.simulation_configurator import SimulationConfig
from src.test_procedure_configurator import TControl
from src import bayes_vector_ops as bayes


#UR reward can be calculated directly from mean of 6 arms
reward_UR = (0.81 + 0.805 + 0.801 + 0.777 + 0.827 + 0.812)/6

#For TS
sim_config_base = SimulationConfig(
    n_rep=20000,
    n_arm=6,
    horizon=4186, 
    burn_in_per_arm=1,
    n_opt_trials=None,  
    arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": [0.81, 0.805, 0.801, 0.777, 0.827, 0.812], "scale": 0.0} #fixed ground truth from empirical data
        },
    test_procedure=TControl(min_effect=0.025, test_type='two-sided'),
    step_cost=0.05,
    reward_evaluation_method='reward',
    vector_ops=bayes.BackendOpsNP()
)

sweeps = [
    {"algo": [algorithm.EpsTS]},
    {"algo_param_list": [0]},
]

res_df = sw.sweep_and_run(sweeps, sim_config_base)
reward_TS = res_df['reward']

#For eps-TS
sim_config_base = SimulationConfig(
    n_rep=20000,
    n_arm=6,
    horizon=1338, 
    burn_in_per_arm=1,
    n_opt_trials=None,  
    arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": [0.81, 0.805, 0.801, 0.777, 0.827, 0.812], "scale": 0.0} #fixed ground truth from empirical data
        },
    test_procedure=TControl(min_effect=0.025, test_type='two-sided'),
    step_cost=0.05,
    reward_evaluation_method='reward',
    vector_ops=bayes.BackendOpsNP()
)

sweeps = [
    {"algo": [algorithm.EpsTS]},
    {"algo_param_list": [0.3]},
]

res_df = sw.sweep_and_run(sweeps, sim_config_base)
reward_eps_TS = res_df['reward']

print(f"UR reward: {reward_UR}, TS reward: {reward_TS}, eps-TS reward: {reward_eps_TS}")