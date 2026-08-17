"""
Appendix critical-value figure: a box plot of the 95% critical value calibrated
by ART / Queue / MLE against the theoretical optimal LRT threshold, from the
output of sim5_art_queue_mle_critval.py.

Usage (run from KDD_27/, after sim5):
    python scripts/plot_critval_distribution.py
"""
import os
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "sim5", os.path.join(_HERE, "sim5_art_queue_mle_critval.py"))
sim5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim5)

CSV = "results/sim5_art_queue_mle_critval_results.csv"
OUT = "results/critval_distribution"

# theoretical optimal = 95th percentile of the LRT statistic under the true null
rng = np.random.default_rng(0)
arms0, rew0 = sim5.sim3.run_mle_batch(500000, rng, p_arm=sim5.P_H0)
optimal = float(np.percentile(sim5.lrt_stat_batch(arms0, rew0), 95))

df = pd.read_csv(CSV)
data = {"ART": df["ART_critval"].values, "Queue": df["Queue_critval"].values,
        "MLE (AIT)": df["MLE_critval"].values}
colors = {"ART": "#1f77b4", "Queue": "#2ca02c", "MLE (AIT)": "#ff7f0e"}

fig, ax = plt.subplots(figsize=(7, 5))
labels = list(data.keys())
for i, k in enumerate(labels):
    v = data[k]
    sub = np.random.default_rng(1).choice(v, size=min(300, len(v)), replace=False)
    x = np.random.default_rng(2 + i).normal(i, 0.06, size=len(sub))
    ax.scatter(x, sub, s=8, alpha=0.25, color=colors[k], zorder=1)
    bp = ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True, zorder=2)
    for box in bp["boxes"]:
        box.set(facecolor="none", edgecolor=colors[k], linewidth=2)
    for elem in ("whiskers", "caps", "medians"):
        for art in bp[elem]:
            art.set(color=colors[k], linewidth=2)

ax.axhline(optimal, ls="--", color="black", lw=1.5, zorder=3,
           label=f"theoretical optimal $\\approx$ {optimal:.3f}")
ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("distribution of critical threshold induced by H0 data", fontsize=12)
ax.set_ylim(-4, 4.2)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=160, bbox_inches="tight")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print(f"saved {OUT}.png / .pdf")
