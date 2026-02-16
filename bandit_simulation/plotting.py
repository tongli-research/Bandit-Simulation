import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ── Factorial / linear-metrics plotting ──────────────────────────────────────

def plot_factor_effect(results, cum_samples, key, title,
                       colors=None, ax=None):
    """Plot a single factorial effect (mean +/- std) over cumulative samples.

    Parameters
    ----------
    results : dict[str, dict]
        Algo name -> metrics dict from compute_linear_factorial_metrics().
    cum_samples : array-like
        Cumulative sample counts (x-axis).
    key : str
        Metric key prefix, e.g. "x1_1v0".
    title : str
        Subplot title.
    colors : dict[str, str] or None
        Algo name -> colour hex string.
    ax : matplotlib.axes.Axes or None
        If None a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    colors = colors or {}

    for name, m in results.items():
        mean = m[f"{key}_mean"]
        std = np.sqrt(m[f"{key}_var"])
        c = colors.get(name)
        ax.plot(cum_samples, mean, label=name, color=c)
        ax.fill_between(cum_samples, mean - std, mean + std,
                        alpha=0.15, color=c)

    # true value line (same across algorithms)
    true_val = next(iter(results.values()))[f"{key}_true"]
    ax.axhline(true_val, ls="--", color="grey",
               label=f"true = {true_val:.3f}")

    ax.set_xlabel("Cumulative samples")
    ax.set_ylabel("Estimated effect")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_all_factor_effects(results, cum_samples, colors=None):
    """Plot all factorial effects found in the metrics dicts.

    Detects keys of the form ``x{col}_{hi}v{lo}_mean`` and creates one
    subplot per contrast.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Discover contrast keys from first algorithm's metrics
    sample_m = next(iter(results.values()))
    keys = sorted(
        k.replace("_mean", "")
        for k in sample_m
        if k.startswith("x") and k.endswith("_mean")
    )

    n = len(keys)
    if n == 0:
        raise ValueError("No factor-effect keys found in metrics")

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), sharey=True, squeeze=False)
    axes = axes.ravel()

    for ax, key in zip(axes, keys):
        title = f"Factor effect: {key.replace('_', ' ')}"
        plot_factor_effect(results, cum_samples, key, title,
                           colors=colors, ax=ax)

    axes[0].set_ylabel("Estimated effect")
    fig.tight_layout()
    return fig


def plot_gap(results, cum_samples, colors=None):
    """Plot average best-arm gap over cumulative samples.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = colors or {}

    for name, m in results.items():
        avg_gap = np.nanmean(m["gap_mean"], axis=-1)
        ax.plot(cum_samples, avg_gap, label=name,
                color=colors.get(name))

    ax.set_xlabel("Cumulative samples")
    ax.set_ylabel("Average gap (best - arm mean)")
    ax.set_title("Average best-arm gap over time")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


# ── ECP-Reward curve plotting ────────────────────────────────────────────────

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
