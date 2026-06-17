import numpy as np
import matplotlib.pyplot as plt

import bandit_algorithm as algo
import sim_wrapper as sw
from simulation_configurator import SimulationConfig
from test_procedure_configurator import ANOVA
import bayes_vector_ops as bayes

#
# def run_wald_distribution(arm_means, policy, n_rep=20000, horizon=1000):
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
#
#     res = sw.run_simulation(policy=policy, sim_config=sim_config)
#
#     # Wald statistics: shape (n_rep, ) or (n_rep, k)? Flatten into 1-D
#     wald_stats = np.array(res.wald_test()).ravel()
#     return wald_stats
#
#
# ts_null = run_wald_distribution([0.5, 0.5], algo.TSPostDiffUR(0))
# ts_alt = run_wald_distribution([0.3, 0.3], algo.TSPostDiffUR(0))
#
# # UR (gamma=1)
# ur_null = run_wald_distribution([0.5, 0.5], algo.TSPostDiffUR(0))
# ur_alt = run_wald_distribution([0.7, 0.7], algo.TSPostDiffUR(0))

#data from Sept 15th 2025, simulated 200,000 with horizon = 1000
# np.savez_compressed("plots/wald_test_distribution_plot_results.npz",
#     ts_null=ts_null, ts_alt=ts_alt,
#     ur_null=ur_null, ur_alt=ur_alt)



# --- Load saved results ---
data = np.load("plots/wald_test_distribution_plot_results.npz")
ts_null, ts_alt, ur_null, ur_alt = data["ts_null"], data["ts_alt"], data["ur_null"], data["ur_alt"]

# --- Shared config ---
colors = {"null": "#d62728" , "alt":"#7f7f7f" }
xlim, ylim = (-5, 7), (0, 0.75)
crit_val_std = 1.645
crit_color_std = "#1f3c88"   # navy blue
crit_color_corr = "#1b9e77"  # teal / green-blue


# --- Calculate errors ---
def compute_errors(null_samples, alt_samples, crit_val):
    type_I  = np.mean(null_samples > crit_val)
    type_II = np.mean(alt_samples <= crit_val)
    return type_I, type_II

# TS errors
ts_typeI_std, ts_typeII_std = compute_errors(ts_null, ts_alt, crit_val_std)
crit_val_corr = np.quantile(ts_null, 0.95)  # empirical 95% quantile
ts_typeI_corr, ts_typeII_corr = compute_errors(ts_null, ts_alt, crit_val_corr)

# ER errors
ur_typeI_std, ur_typeII_std = compute_errors(ur_null, ur_alt, crit_val_std)

# --- Create side-by-side subplots ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

# --- TS subplot ---
axes[0].hist(ts_null, bins=50, density=True, alpha=0.6, label=r"$H_0$: $\mu_1=\mu_2=0.5$", color=colors["null"])
axes[0].hist(ts_alt, bins=50, density=True, alpha=0.6, label=r"$H_1$: $\mu_1 = 0.6, \mu_2 = 0.4$", color=colors["alt"])

# Standard critical value
axes[0].axvline(crit_val_std, color=crit_color_std, linestyle="--", linewidth=1.2)
axes[0].text(crit_val_std - 4.15, ylim[1]*0.65, "Standard 95% critical\nvalue (one-sided)",
             fontsize=10, va="center", color=crit_color_std)

# Corrected critical value
axes[0].axvline(crit_val_corr, color=crit_color_corr, linestyle="--", linewidth=1.4)
axes[0].text(
    crit_val_corr + 0.1, ylim[1]*0.3,
    "Corrected critical value",
    fontsize=10, va="center", color=crit_color_corr
)
# --- TS subplot error box ---
axes[0].text(
    0.98, 0.95,
    f"[Standard test]\n"
    f"Type I error: {ts_typeI_std:.3f}\n"
    f"Type II error: {ts_typeII_std:.3f}\n\n"
    f"[Corrected test]\n"
    f"Type I error: " + r"$\mathbf{0.050}$" + f"\n"
    f"Type II error: {ts_typeII_corr:.3f}",
    transform=axes[0].transAxes, ha="right", va="top",
    fontsize=9.5,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)
axes[0].set_title("Thompson Sampling", fontsize=13, weight="bold")
axes[0].set_xlabel("Wald statistic", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].set_xlim(xlim)
axes[0].set_ylim(ylim)
axes[0].legend(frameon=False, fontsize=11, loc="upper left")

# --- ER subplot ---
axes[1].hist(ur_null, bins=50, density=True, alpha=0.6, label=r"$H_0$: $\mu_1=\mu_2=0.5$", color=colors["null"])
axes[1].hist(ur_alt, bins=50, density=True, alpha=0.6, label=r"$H_1$: $\mu_1 = 0.6, \mu_2 = 0.4$", color=colors["alt"])

# Standard critical value
axes[1].axvline(crit_val_std, color=crit_color_std, linestyle="--", linewidth=1.2)
axes[1].text(crit_val_std - 4.15, ylim[1]*0.65, "Standard 95% critical\nvalue (one-sided)",
             fontsize=10, va="center", color=crit_color_std)
# --- ER subplot error box ---
axes[1].text(
    0.98, 0.95,
    f"[Standard test]\n"
    f"Type I error: " + r"$\mathbf{0.050}$" + f"\n"
    f"Type II error: {ur_typeII_std:.3f}",
    transform=axes[1].transAxes, ha="right", va="top",
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)
axes[1].set_title("Equal Randomization", fontsize=13, weight="bold")
axes[1].set_xlabel("Wald statistic", fontsize=12)
axes[1].set_xlim(xlim)
axes[1].set_ylim(ylim)
axes[1].legend(frameon=False, fontsize=11, loc="upper left")

# --- Final layout ---
plt.tight_layout()
plt.show()