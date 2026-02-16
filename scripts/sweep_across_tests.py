"""
Reproduces Table 4 (optimize eps-TS across different tests).
"""


import numpy as np

from bandit_simulation import bandit_algorithm as algorithm
from bandit_simulation import bayes_vector_ops as bayes
from bandit_simulation.sim_wrapper import sweep_and_run
from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.test_procedure_configurator import ANOVA, TConstant, TControl, Tukey

sim_config_base = SimulationConfig(
    n_rep=20000,
    n_arm=3,
    horizon=5000,
    burn_in_per_arm=1,
    arm_mean_reward_dist_spec={
            "dist": "beta",
            "params": {"a": 5, "b": 5}
        },
    test_procedure=ANOVA(),
    reward_evaluation_method='reward',
    vector_ops=bayes.BackendOpsNP()
)


algo_list = [algorithm.EpsTS]
test_list = [
    ANOVA(),
    TConstant(test_type='one-sided',min_effect=0.1),
    TControl(test_type='two-sided',min_effect=0.1),
    Tukey(test_type='distinct-best-arm',min_effect=0.1,family_wise_error_control=True),
]


sweeps = [
    {"algo": algo_list},
    {"algo_param_list": list(map(float, np.linspace(0.0, 1.0, 421)))},
    {"test_proc": test_list},   # ← key point
]

df = sweep_and_run(sweeps, sim_config_base)
