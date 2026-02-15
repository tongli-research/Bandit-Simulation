#TODO: run it again and save result....

import numpy as np
import bandit_algorithm as algo
import sim_wrapper as sw

from simulation_configurator import SimulationConfig
from test_procedure_configurator import TestProcedure, ANOVA, TControl, TConstant, Tukey
import bayes_vector_ops as bayes
arm_means = [0.5, 0.5]   # diff=0

sim_config = SimulationConfig(
    n_rep=1,
    n_arm=2,
    horizon=100,
    burn_in_per_arm=1,
    n_opt_trials=5,
    arm_mean_reward_dist_loc=arm_means,
    arm_mean_reward_dist_scale=0.0,
    test_procedure=ANOVA(),
    step_cost=0.1,
    reward_evaluation_method='regret',
    vector_ops=bayes.BackendOpsNP()
)
sim_config.manual_init()

res0 = sw.run_simulation(policy=algo.TSPostDiffUR(0.1), sim_config=sim_config)


import matplotlib.pyplot as plt
samples = samples.squeeze()
x = samples[:, 0]
y = samples[:, 1]

fig, ax = plt.subplots(figsize=(6,6))

# scatter plot of samples
ax.scatter(x, y, alpha=0.65, s=20, marker="x", color="black")

# reference lines y = x+0.1 and y = x-0.1
xx = np.linspace(0, 1, 200)
ax.plot(xx, xx + 0.1, color="red", linestyle="--")
ax.plot(xx, xx - 0.1, color="red", linestyle="--")

# shade the exploration band (between the two lines)
ax.fill_between(xx, xx-0.1, xx+0.1, where=((xx-0.1)>=0) & ((xx+0.1)<=1),
                color="lightgrey", alpha=0.5)

# annotate regions
ax.text(0.15, 0.4, "PostDiff exploration",
        ha="left", va="top", fontsize=12, color="black",
        rotation=45,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

ax.text(0.8, 0.2, "PostDiff exploitation",rotation=45,
        ha="center", va="center", fontsize=12, color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

ax.text(0.2, 0.8, "PostDiff exploitation",rotation=45,
        ha="center", va="center", fontsize=12, color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

# limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Arm 1 sample")
ax.set_ylabel("Arm 2 sample")
ax.set_title("PostDiff posterior samples (p1=p2=0.5)")

plt.show()