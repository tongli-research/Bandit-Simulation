"""
Plot Power-Reward Pareto frontiers from CSV data.

Usage:
    python scripts/thesis_figures/postdiff_fig2_pareto_plot.py

Input:  scripts/thesis_figures/_out/postdiff_fig2_pareto_50k.csv
Output: scripts/thesis_figures/_out/postdiff_fig2_pareto.pdf
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# === Configuration ============================================================

CSV_PATH = os.path.join(os.path.dirname(__file__), "_out/postdiff_fig2_pareto_50k.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "_out/postdiff_fig2_pareto.pdf")

FAMILIES = {
    "EpsTS":    {"marker": "o", "color": "green",   "label": r"$\varepsilon$/TT-TS"},
    "PostDiff": {"marker": "s", "color": "#DAA520", "label": "TS-PostDiff"},
    "ProbClip": {"marker": "^", "color": "purple",  "label": "TS-ProbClip"},
    # Top2TS omitted: equivalent to EpsTS in 2-armed case
}

BASELINES = {
    "TS": {"marker": "*", "color": "black", "size": 150},
    "UR": {"marker": "*", "color": "gray",  "size": 150},
}

ARM_DIFFS = [0.2, 0.3]

# === Load data ================================================================

rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "arm_diff": float(r["arm_diff"]),
            "family": r["family"],
            "param": float(r["param"]),
            "reward": float(r["reward"]),
            "power": float(r["power"]),
        })

# === Plot =====================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for idx, d in enumerate(ARM_DIFFS):
    ax = axes[idx]

    # Plot each algorithm family
    for fam, style in FAMILIES.items():
        pts = [(r["reward"], r["power"]) for r in rows
               if r["arm_diff"] == d and r["family"] == fam]
        if not pts:
            continue
        rews, pows = zip(*pts)
        order = np.argsort(rews)
        ax.plot(
            np.array(rews)[order], np.array(pows)[order],
            marker=style["marker"], color=style["color"],
            markersize=4, linewidth=1.5, label=style["label"], alpha=0.85,
        )

    # Plot TS and UR as special points
    for r in rows:
        if r["arm_diff"] == d and r["family"] in BASELINES:
            s = BASELINES[r["family"]]
            ax.scatter(
                r["reward"], r["power"],
                marker=s["marker"], color=s["color"], s=s["size"],
                label=r["family"], zorder=4,
            )

    # Dashed lines from TS and UR to guide the eye
    ts_pt = next((r for r in rows if r["arm_diff"] == d and r["family"] == "TS"), None)
    ur_pt = next((r for r in rows if r["arm_diff"] == d and r["family"] == "UR"), None)
    if ts_pt and ur_pt:
        # Horizontal dashed from UR (power reference)
        ax.axhline(ur_pt["power"], color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        # Vertical dashed from TS (reward reference)
        ax.axvline(ts_pt["reward"], color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Mean Reward", fontsize=12)
    if idx == 0:
        ax.set_ylabel("Power (AIT-corrected, FPR = 0.05)", fontsize=12)
    ax.set_title(f"Arm difference = {d}", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

fig.suptitle("Power-Reward Pareto Frontier (n = 197, Wald test)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
