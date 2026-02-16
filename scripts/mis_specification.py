"""
Reproduces Table 5 (loc / scale mis-specification).
"""


import numpy as np

from bandit_simulation import bandit_algorithm as algorithm
from bandit_simulation.sim_wrapper import sweep_and_run
from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.test_procedure_configurator import ANOVA

sim_config_base = SimulationConfig(
    n_rep=20000,
    n_arm=3,
    horizon=2000,
    burn_in_per_arm=1,
    arm_mean_reward_dist_spec={
            "dist": "beta",
            "params": {"a": 3.2, "b": 5.9}
        },
    test_procedure=ANOVA(),
    reward_evaluation_method='reward',
)



beta_sweep = [ #Use this for location mis-specification analysis
    {"dist": "beta", "params": {"a": 1.2, "b": 4.9}},
    {"dist": "beta", "params": {"a": 1.8, "b": 5.5}},
    {"dist": "beta", "params": {"a": 2.5, "b": 5.8}},
    {"dist": "beta", "params": {"a": 3.2, "b": 5.9}},
    {"dist": "beta", "params": {"a": 3.9, "b": 5.8}},
    {"dist": "beta", "params": {"a": 4.5, "b": 5.5}},
    {"dist": "beta", "params": {"a": 5.1, "b": 5.1}},
]

# beta_sweep = [ #Use this for scale mis-specification analysis
#     {"dist": "beta", "params": {"a": 9.5, "b": 17.7}},   # std ≈ 0.09
#     {"dist": "beta", "params": {"a": 6.0, "b": 11.2}},   # std ≈ 0.11
#     {"dist": "beta", "params": {"a": 4.0, "b": 7.4}},    # std ≈ 0.13
#     {"dist": "beta", "params": {"a": 2.8, "b": 5.2}},    # std ≈ 0.15
#     {"dist": "beta", "params": {"a": 2.0, "b": 3.7}},    # std ≈ 0.17
#     {"dist": "beta", "params": {"a": 1.5, "b": 2.8}},    # std ≈ 0.19
#     {"dist": "beta", "params": {"a": 1.2, "b": 2.2}},    # std ≈ 0.21
# ]


sweeps = [
    {"algo": [algorithm.EpsTS]},
    {"algo_param_list": list(map(float, np.linspace(0.0, 1.0, 41)))},
    {"arm_mean_reward_dist_spec": beta_sweep},
]

df = sweep_and_run(sweeps, sim_config_base)
