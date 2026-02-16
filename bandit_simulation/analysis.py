import numpy as np
import pandas as pd


# ── Linear / factorial summary table ─────────────────────────────────────────

def format_linear_summary(results, mu_true, step_index=-1):
    """Build a tidy summary DataFrame for factorial/linear bandit results.

    Algorithms are columns; metrics are rows.

    Parameters
    ----------
    results : dict[str, dict]
        Algorithm name -> metrics dict returned by
        ``SimResult.compute_linear_factorial_metrics()``.
    mu_true : array-like, shape (K,)
        True arm means.
    step_index : int
        Which time-step to evaluate (default ``-1`` = final step).

    Returns
    -------
    pd.DataFrame
        Columns: ``true``, ``{algo}_mean``, ``{algo}_std`` for each algorithm.
        Rows: ``mean_reward``, ``prop_arm_{k}`` for each arm,
              ``gap_arm_{k}`` for non-best arms, and factorial contrasts.
    """
    mu_true = np.asarray(mu_true)
    K = len(mu_true)
    best_arm = int(np.argmax(mu_true))

    # Discover contrast keys from first algorithm
    sample_m = next(iter(results.values()))
    contrast_keys = sorted(
        k.replace("_mean", "")
        for k in sample_m
        if k.startswith("x") and k.endswith("_mean")
    )

    # Non-best arm indices (in original order)
    non_best = [k for k in range(K) if k != best_arm]

    # ── Row names ────────────────────────────────────────────────────
    row_names = (
        ["mean_reward"]
        + [f"prop_arm_{k}" for k in range(K)]
        + [f"gap_arm_{k}" for k in non_best]
        + list(contrast_keys)
    )

    # ── True column ──────────────────────────────────────────────────
    true_vals = (
        [np.nan]                                                    # mean_reward
        + [np.nan] * K                                              # proportions
        + [mu_true[best_arm] - mu_true[k] for k in non_best]       # gaps
        + [sample_m[f"{ck}_true"] for ck in contrast_keys]          # effects
    )

    data = {"true": true_vals}

    # ── Per-algorithm columns ────────────────────────────────────────
    for name, m in results.items():
        mean_vals = (
            [m["reward_mean"][step_index]]
            + [m["prop_mean"][step_index, k] for k in range(K)]
            + [m["gap_mean"][step_index, k] for k in non_best]
            + [m[f"{ck}_mean"][step_index] for ck in contrast_keys]
        )
        std_vals = (
            [np.sqrt(m["reward_var"][step_index])]
            + [np.sqrt(m["prop_var"][step_index, k]) for k in range(K)]
            + [np.sqrt(m["gap_var"][step_index, k]) for k in non_best]
            + [np.sqrt(m[f"{ck}_var"][step_index]) for ck in contrast_keys]
        )
        data[f"{name}_mean"] = mean_vals
        data[f"{name}_std"] = std_vals

    df = pd.DataFrame(data, index=row_names)
    return df


# ── ECP-Reward analysis ─────────────────────────────────────────────────────

def compute_objective(row, w: float):
    n = row["n_step"]
    reward = row["regret_per_step"]
    score = reward - w * np.log(n)
    return score

def compute_baseline(df, w_values=range(1, 16)):
    baseline = {}
    for w in w_values:
        scores = df.apply(lambda r: compute_objective(r, w), axis=1)
        baseline[w] = scores.max()
    return baseline






def select_curves_relative(df, selectors, w_values):
    baseline = compute_baseline(df, w_values)
    results = {}

    for sel in selectors:
        if len(sel) == 3:
            algo_name, mode, value = sel
            custom_label = None
            color = None
            linestyle = "-"
        elif len(sel) == 4:
            algo_name, mode, value, custom_label = sel
            color = None
            linestyle = "-"
        elif len(sel) == 5:
            algo_name, mode, value, custom_label, color = sel
            linestyle = "-"
        elif len(sel) == 6:
            algo_name, mode, value, custom_label, color, linestyle = sel
        else:
            raise ValueError("Each selector must have 3–6 elements")

        subset = df[df["algo_name"] == algo_name].copy()

        if mode == "param":
            sub = subset[subset["algo_param"] == value]
            if sub.empty:
                raise ValueError(f"No rows for {algo_name} with param={value}")
            row = sub.iloc[0]

            curves = []
            for w in w_values:
                raw = compute_objective(row, w)
                rel = raw - baseline[w]
                curves.append({"w": w, "obj_rel": rel, "obj_abs": raw})

            label = custom_label if custom_label is not None else f"{algo_name} (param={value})"
            df_curve = pd.DataFrame(curves)
            df_curve.attrs = {"algo_name": algo_name, "color": color, "linestyle": linestyle}
            results[label] = df_curve

        elif mode == "w":
            w_ref = value
            subset["obj_tmp"] = subset.apply(lambda r: compute_objective(r, w_ref), axis=1)
            best = subset.loc[subset["obj_tmp"].idxmax()]
            chosen_param = best["algo_param"]

            sub = subset[subset["algo_param"] == chosen_param].iloc[0]

            curves = []
            for w in w_values:
                raw = compute_objective(sub, w)
                rel = raw - baseline[w]
                curves.append({"w": w, "obj_rel": rel, "obj_abs": raw})

            label = (
                custom_label
                if custom_label is not None
                else f"{algo_name} opt for w={w_ref} (param={chosen_param})"
            )
            df_curve = pd.DataFrame(curves)
            df_curve.attrs = {"algo_name": algo_name, "color": color, "linestyle": linestyle}
            results[label] = df_curve

        else:
            raise ValueError("mode must be 'param' or 'w'")

    return results




def compute_best_param_per_w(df, algo_name, w_values):
    rows = []
    subset = df[df["algo_name"] == algo_name].copy()

    for w in w_values:
        subset["obj_tmp"] = subset.apply(lambda r: compute_objective(r, w), axis=1)
        best = subset.loc[subset["obj_tmp"].idxmax()]
        rows.append({"w": w, "best_param": best["algo_param"]})

    return pd.DataFrame(rows)
