"""
Fig 1: PostDiff two-arm posterior illustration.
Scatter plot showing winner circle (uniform exploration) vs direct exploitation.

Each dot = one independent replication of TSPostDiff(delta=0.1) for T=100 steps,
then one posterior draw per arm. Two layers of randomness: observation noise + posterior sampling.

Output: latex/figures/fig1_postdiff_illustration.pdf
"""
import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt

from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.bandit_algorithm import TSPostDiff
from bandit_simulation.sim_wrapper import run_simulation

DELTA = 0.1
N_REP_PLOT = 50
N_REP_PROB = 1000
HORIZON = 200
SEED = 42

SCENARIOS = [
    {
        "arms": [0.5, 0.5],
        "label": "(a) No arm difference: $p_1 = p_2 = 0.5$",
    },
    {
        "arms": [0.6, 0.4],
        "label": "(b) Large arm difference: $p_1 = 0.6$, $p_2 = 0.4$",
    },
]

ROOT = __import__("pathlib").Path(__file__).resolve().parents[0].parent
FIG_DIR = ROOT / "latex" / "figures"

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle(
    f"PostDiff ($\\delta = {DELTA}$) posterior samples after {HORIZON} steps",
    fontsize=13, y=0.98,
)

for ax, sc in zip(axes, SCENARIOS):
    # Large run for accurate winner circle probability
    np.random.seed(SEED)
    cfg_prob = SimulationConfig(
        n_rep=N_REP_PROB, n_arm=2, horizon=HORIZON, burn_in_per_arm=1,
        reward_model=np.random.binomial,
        arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": sc["arms"], "scale": 0.0},
        },
    )
    cfg_prob.manual_init()
    run_simulation(policy=TSPostDiff(DELTA), sim_config=cfg_prob)
    bm_prob = cfg_prob.bayes_model
    rng_prob = np.random.default_rng(SEED)
    s1_prob = beta_dist.rvs(bm_prob.posterior['a'][:, 0], bm_prob.posterior['b'][:, 0], random_state=rng_prob)
    s2_prob = beta_dist.rvs(bm_prob.posterior['a'][:, 1], bm_prob.posterior['b'][:, 1], random_state=rng_prob)
    wc_prob = (np.abs(s1_prob - s2_prob) < DELTA).mean()

    # Small run for plotting
    np.random.seed(SEED + 1)
    cfg_plot = SimulationConfig(
        n_rep=N_REP_PLOT, n_arm=2, horizon=HORIZON, burn_in_per_arm=1,
        reward_model=np.random.binomial,
        arm_mean_reward_dist_spec={
            "dist": "normal",
            "params": {"loc": sc["arms"], "scale": 0.0},
        },
    )
    cfg_plot.manual_init()
    run_simulation(policy=TSPostDiff(DELTA), sim_config=cfg_plot)
    bm = cfg_plot.bayes_model
    rng = np.random.default_rng(SEED + 1)
    s1 = beta_dist.rvs(bm.posterior['a'][:, 0], bm.posterior['b'][:, 0], random_state=rng)
    s2 = beta_dist.rvs(bm.posterior['a'][:, 1], bm.posterior['b'][:, 1], random_state=rng)

    ax.fill_between([0, 1], [0, 0], [1, 1], color="0.88", zorder=0)
    xs = np.linspace(0, 1, 300)
    ax.fill_between(xs, np.clip(xs - DELTA, 0, 1), np.clip(xs + DELTA, 0, 1),
                    color="white", zorder=1)
    ax.plot([0, 1 - DELTA], [DELTA, 1], "r--", lw=1.5, zorder=2)
    ax.plot([DELTA, 1], [0, 1 - DELTA], "r--", lw=1.5, zorder=2)
    ax.scatter(s1, s2, marker="x", c="black", s=30, linewidths=0.8, zorder=3)

    label_kw = dict(fontsize=8.5, rotation=45, ha="center", va="center",
                    fontweight="bold", zorder=4)
    ax.text(0.20, 0.16, "Winner circle\n(uniform exploration)", **label_kw)
    ax.text(0.13, 0.80, "Direct\nexploitation", **label_kw)
    ax.text(0.80, 0.13, "Direct\nexploitation", **label_kw)
    ax.text(0.03, 0.97, f"Winner circle probability = {wc_prob:.2f}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.7"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Arm 1 posterior sample", fontsize=10)
    ax.set_ylabel("Arm 2 posterior sample", fontsize=10)
    ax.set_title(sc["label"], fontsize=10)
    ax.set_aspect("equal")

plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig(FIG_DIR / "fig1_postdiff_illustration.pdf", bbox_inches="tight")
print(f"Saved to {FIG_DIR / 'fig1_postdiff_illustration.pdf'}")
