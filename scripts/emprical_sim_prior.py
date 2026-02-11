"""
Reproduces Figure 1 (GUI optimization visualization) and Table 3 (partially).

What this script does:
  1. Sets up the empirically inspired 6-arm simulation under the PRIOR specification
     described in Section 4 (design-time setting).
  2. Sweeps epsilon-TS across a grid of epsilon values.
  3. Computes and plots relative ECP-reward curves across a range of extension costs w.
  4. Displays TS (epsilon=0) and UR (epsilon=1) as benchmark baselines.

This corresponds to:
  - Figure: Relative ECP-reward performance in the GUI (Figure EpsTS_ANOVA_objective_score).
  - "Prior (design-time)" columns in Table 3.
"""

import numpy as np
from src import bandit_algorithm as algorithm
from src import sim_wrapper as sw

from src.simulation_configurator import SimulationConfig
from src.test_procedure_configurator import TControl
from src import bayes_vector_ops as bayes
from src.analysis import select_curves_relative
from src.plotting import plot_curves







sim_config_base = SimulationConfig(
    n_rep=20000,
    n_arm=6,
    horizon=5000,  # max horizon to try in simulation
    burn_in_per_arm=1,
    n_opt_trials=None,  
    arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": 0.81, "scale": 0.015}
        },
    test_procedure=TControl(min_effect=0.025, test_type='two-sided'),
    step_cost=0.05,
    reward_evaluation_method='reward',
    vector_ops=bayes.BackendOpsNP()
)

# Define sweeps
sweeps = [
    {"algo": [algorithm.EpsTS]},
    {"algo_param_list": list(map(float, np.linspace(0.0, 1.0, 21)))},
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
