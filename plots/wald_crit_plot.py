import numpy as np
import matplotlib.pyplot as plt
import bandit_algorithm as algo
import sim_wrapper as sw
from simulation_configurator import SimulationConfig
from test_procedure_configurator import ANOVA
import bayes_vector_ops as bayes


# def run_wald_distribution(arm_means, policy, n_rep=20000, horizon=500):
#     """Run simulation and return flattened Wald statistics."""
#     sim_config = SimulationConfig(
#         n_rep=n_rep,
#         n_arm=2,
#         horizon=horizon,
#         burn_in_per_arm=1,
#         n_opt_trials=5,
#         arm_mean_reward_dist_loc=arm_means,
#         arm_mean_reward_dist_scale=0.0,
#         test_procedure=ANOVA(),
#         step_cost=0.1,
#         reward_evaluation_method="regret",
#         vector_ops=bayes.BackendOpsNP()
#     )
#     sim_config.manual_init()
#     res = sw.run_simulation(policy=policy, sim_config=sim_config)
#     wald_stats = np.array(res.wald_test()).ravel()
#     return wald_stats
#
#
# # sweep over symmetric arm means
arm_means_list = [[p, p] for p in np.linspace(0.1, 0.9, 9)]
#
# # algorithms to test
# algos = {
#     "TS": algo.TSPostDiffUR(0),
#     "ER": algo.TSPostDiffUR(1)
# }
#
# # horizons to test
# horizons = [300, 1000]
#
# # store results: dict[(algo_name, horizon)] = list of critical values
# results = {}
#
# for algo_name, policy in algos.items():
#     for horizon in horizons:
#         crit_vals = []
#         for arm_means in arm_means_list:
#             stats = run_wald_distribution(arm_means, policy, n_rep=100000, horizon=horizon)
#             crit_val = np.quantile(stats, 0.95)
#             crit_vals.append(crit_val)
#             print(f"{algo_name}, horizon={horizon}, arm means {arm_means}: "
#                   f"95% critical value = {crit_val:.3f}")
#         results[(algo_name, horizon)] = np.array(crit_vals)


#np.save("plots/wald_crit_value.npy", results)

results = np.load("plots/wald_crit_value.npy", allow_pickle=True).item()


# ---- Plot ----
plt.figure(figsize=(8, 6))
x_vals = [m[0] for m in arm_means_list]

# shapes per algorithm (TS vs ER), colors per horizon
algo_markers = {"TS": "o", "ER": "x"}   # TS → circle, ER → cross
horizon_colors = {300: "grey", 1000: "royalblue"}

for (algo_name, horizon), crit_vals in results.items():
    if algo_name == "TS":
        label = f"Thompson Sampling, Horizon = {horizon}"
    elif algo_name == "ER":
        label = f"Equal Randomization, Horizon = {horizon}"
    else:
        label = f"{algo_name}, H={horizon}"

    plt.plot(
        x_vals,
        crit_vals,
        marker=algo_markers[algo_name],
        linestyle="--",
        color=horizon_colors[horizon],
        label=label
    )

# reference line at 1.645
plt.axhline(
    1.645,
    color="lightcoral",
    linestyle="--",
    linewidth=1,
    label="Standard Normal 95% Quantile"
)

# labels + title
plt.xlabel("Arm mean for H0: p1=p2", fontsize=14)
plt.ylabel("95% critical value for Wald test (Type I error = 0.05)", fontsize=14)
plt.title("Critical values across H0 locations", fontsize=15)

# ticks for x (0.1, 0.2, …, 0.9) and bigger font for both axes
plt.xticks(np.arange(0.1, 1.0, 0.1), fontsize=12)
plt.yticks(fontsize=12)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
