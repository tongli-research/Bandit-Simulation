import numpy as np
import pandas as pd


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
