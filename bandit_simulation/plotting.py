import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _downsample_xy(x, y, n_points=20):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= n_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, n_points).round().astype(int)
    idx = np.unique(idx)
    return x[idx], y[idx]

def plot_curves(curves, ymin, df=None, w_values=None):
    fig, ax = plt.subplots(figsize=(9.9, 6))  # 20% wider

    LABEL_FONTSIZE = 11
    TITLE_FONTSIZE = 15
    LEGEND_FONTSIZE = 11
    TICK_FONTSIZE = 10

    # -----------------------------
    # Plot curves
    # -----------------------------
    for label, curve in curves.items():
        color = curve.attrs.get("color", None)
        linestyle = curve.attrs.get("linestyle", None)

        x, y = _downsample_xy(curve["w"], curve["obj_rel"], n_points=20)

        plot_kwargs = dict(
            marker="o",
            markersize=7,
            linewidth=1.8,
            color=color,
        )
        if linestyle is not None:
            plot_kwargs["linestyle"] = linestyle

        ax.plot(x, y, **plot_kwargs)

    # -----------------------------
    # Axis labels & limits (reverted)
    # -----------------------------
    ax.set_xlabel(
        "Experiment Extension Cost ('w')",
        fontsize=LABEL_FONTSIZE,
        color="#666666",
    )
    ax.set_ylabel(
        "Relative ECP-Reward",
        fontsize=LABEL_FONTSIZE,
        color="#666666",
    )
    ax.set_ylim(ymin, 0.0001)

    # -----------------------------
    # Double x ticks
    # -----------------------------
    if w_values is not None:
        w_min, w_max = min(w_values), max(w_values)
        ax.set_xticks(np.arange(w_min, w_max, 0.005))

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors="#777777")

    # -----------------------------
    # Spines & grid
    # -----------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#D0D0D0")
        ax.spines[side].set_linewidth(1.0)

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.25)

    # -----------------------------
    # Title (reverted)
    # -----------------------------
    ax.set_title(
        "Relative ECP-Reward Compared to Per-'w' Optimal",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        color="#555555",
    )

    # -----------------------------
    # Legend: big dots only
    # -----------------------------
    legend_handles = []
    legend_labels = []

    for label, curve in curves.items():
        color = curve.attrs.get("color", None)
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=14,
                markerfacecolor=color,
                markeredgecolor=color,
            )
        )
        legend_labels.append(label)

    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(curves),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        handletextpad=0.0,
        columnspacing=0.8,
    )

    # Set legend text color
    for text in legend.get_texts():
        text.set_color("#666666")

    # ax.legend(
    #     legend_handles,
    #     legend_labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.12),
    #     ncol=len(curves),
    #     fontsize=LEGEND_FONTSIZE,
    #     frameon=False,
    #     handletextpad=0.0,  # <-- bring text closer to dot
    #     columnspacing=0.8,  # <-- reduce gap between legend entries
    # )

    plt.tight_layout()
    plt.show()
