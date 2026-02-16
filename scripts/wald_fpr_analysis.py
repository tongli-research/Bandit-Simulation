"""
Reproduces Table 1.

What this script does:
  1. Sweeps symmetric null hypotheses (p1 = p2 = p) across a grid of mean rewards.
  2. For each null setting, runs bandit simulations and collects Wald statistics.
  3. Produces three figures:
     (a) Wald statistic distribution at two representative null means vs N(0,1).
     (b) Simulated 95% critical threshold as a function of null mean.
     (c) FPR comparison: fixed classical threshold vs algorithm-induced correction.
"""

import matplotlib.pyplot as plt
import numpy as np

from bandit_simulation import bandit_algorithm as algo
from bandit_simulation import bayes_vector_ops as bayes
from bandit_simulation import sim_wrapper as sw
from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.test_procedure_configurator import ANOVA

# ── Configuration ────────────────────────────────────────────────────────────

N_REP = 50000
HORIZON = 200
P_GRID = np.linspace(0.1, 0.9, 9)
ARM_MEANS_LIST = [[p, p] for p in P_GRID]
THRESHOLD_FIXED = 1.645

ALGOS = {
    "TS": algo.TSPostDiffUR(0),
}


# ── Simulation ───────────────────────────────────────────────────────────────

results = {}

for algo_name, algo_obj in ALGOS.items():
    n_settings = len(ARM_MEANS_LIST)
    stats_mat = None
    mean_mat = None

    for i, arm_means in enumerate(ARM_MEANS_LIST):
        sim_config = SimulationConfig(
            n_rep=N_REP,
            n_arm=2,
            horizon=HORIZON,
            burn_in_per_arm=1,
            arm_mean_reward_dist_spec={
                "dist": "normal",
                "params": {"loc": arm_means, "scale": 0},
            },
            test_procedure=ANOVA(),
            reward_evaluation_method="regret",
            vector_ops=bayes.BackendOpsNP(),
        )
        sim_config.manual_init()

        res = sw.run_simulation(policy=algo_obj, sim_config=sim_config)
        wald_stats = np.array(res.wald_test())[:, -1, :].flatten()
        mean_rewards = res.combined_means[:, -1, :].flatten()

        if stats_mat is None:
            n_rep_actual = wald_stats.shape[0]
            stats_mat = np.empty((n_settings, n_rep_actual), dtype=float)
            mean_mat = np.empty((n_settings, n_rep_actual), dtype=float)

        stats_mat[i, :] = wald_stats
        mean_mat[i, :] = mean_rewards

    # ── Figure (a): Wald distribution at two null means vs N(0,1) ────────

    def plot_hist_with_normal(ax, x, label, bins=60):
        ax.hist(x, bins=bins, density=True, alpha=0.4, label=label)
        grid = np.linspace(-5, 5, 800)
        normal_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * grid**2)
        ax.plot(grid, normal_pdf, linewidth=2, label="N(0,1)")

    idx_05 = int(np.argmin(np.abs(P_GRID - 0.50)))
    idx_02 = int(np.argmin(np.abs(P_GRID - 0.20)))

    fig, ax = plt.subplots()
    plot_hist_with_normal(ax, stats_mat[idx_05], label=f"Wald @ p={P_GRID[idx_05]:.2f}")
    plot_hist_with_normal(ax, stats_mat[idx_02], label=f"Wald @ p={P_GRID[idx_02]:.2f}")
    ax.axvline(THRESHOLD_FIXED, linestyle="--", linewidth=2, label="1.645 (fixed)")
    ax.set_title(f"Wald statistic distribution comparison ({algo_name}, T={HORIZON})")
    ax.set_xlabel("Wald statistic")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figures/wald_distribution_{algo_name}_T{HORIZON}.pdf")
    plt.show()

    # ── Figure (b): Simulated 95% critical threshold vs null mean ────────

    true_thres = np.quantile(stats_mat, 0.95, axis=1)

    fig, ax = plt.subplots()
    ax.plot(P_GRID, true_thres, marker="o", label="Simulated 95% threshold")
    ax.axhline(THRESHOLD_FIXED, linestyle="--", linewidth=1, label="1.645 (fixed)")
    ax.set_title(f"Simulated 95% critical threshold vs null p ({algo_name}, T={HORIZON})")
    ax.set_xlabel("Null p (p1=p2=p)")
    ax.set_ylabel("Critical value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figures/wald_threshold_{algo_name}_T{HORIZON}.pdf")
    plt.show()

    # ── Figure (c): FPR comparison ───────────────────────────────────────

    p_min, p_max = float(P_GRID.min()), float(P_GRID.max())
    mean_clamped = np.clip(mean_mat, p_min, p_max)

    interp_thres_mat = np.empty_like(stats_mat)
    for i in range(n_settings):
        interp_thres_mat[i, :] = np.interp(mean_clamped[i, :], P_GRID, true_thres)

    fpr_interpolated = np.mean(stats_mat > interp_thres_mat, axis=1)
    fpr_regular = np.mean(stats_mat > THRESHOLD_FIXED, axis=1)

    fig, ax = plt.subplots()
    ax.plot(P_GRID, fpr_interpolated, marker="o", label="FPR (algorithm-induced test correction)")
    ax.plot(P_GRID, fpr_regular, marker="x", label="FPR (classical threshold, 1.645)")
    ax.axhline(0.05, linestyle="--", linewidth=0.8, color="black", label="Target FPR (0.05)")
    ax.set_title("FPR: classical critical region vs AIT correction")
    ax.set_xlabel(r"Null hypothesis mean reward $\mu$ ($\mu_1 = \mu_2 = \mu$)")
    ax.set_ylabel("False positive rate")
    ax.set_ylim(0.0, 0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figures/wald_fpr_{algo_name}_T{HORIZON}.pdf")
    plt.show()

    results[(algo_name, HORIZON)] = {
        "p_grid": P_GRID,
        "stats_mat": stats_mat,
        "mean_mat": mean_mat,
        "true_thres": true_thres,
        "interp_thres_mat": interp_thres_mat,
        "fpr_interpolated": fpr_interpolated,
        "fpr_regular": fpr_regular,
    }
