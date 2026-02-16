"""
template for running a simulation (how to set up configurations etc)
"""

import numpy as np

from bandit_simulation import bandit_algorithm as algorithm
from bandit_simulation import bayes_vector_ops as bayes
from bandit_simulation import sim_wrapper as sw
from bandit_simulation.analysis import select_curves_relative
from bandit_simulation.plotting import plot_curves
from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.test_procedure_configurator import TControl

sim_config_base = SimulationConfig(
    n_rep=2000,
    n_arm=3,
    horizon=1000,  # max horizon to try in simulation
    burn_in_per_arm=1,
    arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": 0.81, "scale": 0.015}
        },
    test_procedure=TControl(min_effect=0.025, test_type='two-sided'),
    reward_evaluation_method='reward',
    vector_ops=bayes.BackendOpsNP()
)

# Define sweeps
sweeps = [
    {"algo": [algorithm.EpsTS]},
    {"algo_param_list": list(map(float, np.linspace(0.0, 1.0, 2)))},
]

res_df = sw.sweep_and_run(sweeps, sim_config_base)


"""
Part 2: interactive result page
"""
selectors = [
    ("EpsTS", "param", 0.0, "Pure-TS", "#28A745", "--"),
    ("EpsTS", "param", 1.0, "Pure-UR", "#DC3545", "--"),
    ("EpsTS", "w", 0.01, None, "#007BFF", "-"),
]

w_values = np.arange(0.00, 0.06, 0.001)
curves = select_curves_relative(res_df, selectors, w_values=w_values)
plot_curves(curves, -0.03, df=res_df, w_values=w_values)
