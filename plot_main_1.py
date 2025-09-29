"""
plot latex label: EpsTS_ANOVA_objective_score
Use data: "main_df0724.csv"

main plot in paper and GUI (give people decision reference)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
TEST_NAME = "T-Constant (one-sided)"
ALGO_FAMILY = "EpsTS"   # after remapping PostDiffTop(param=1) -> EpsTS(1)
PARAM_CHOICES = [0, 0.1, 0.2, 0.5, 1]  # sub algo_param choices

# --- Step 1: compute opt_reward ---
def compute_opt_reward(n_samples=1_000_000, mu=0.5, sigma=0.15, cap=0.95):
    samples = np.random.normal(mu, sigma, size=(n_samples, 3))
    samples = np.minimum(samples, cap)  # cap values
    return np.mean(np.max(samples, axis=1))

opt_reward = compute_opt_reward()
print("Estimated opt_reward:", opt_reward)

# --- Step 2: load CSV ---
df = pd.read_csv("plots/main_df0724.csv")

# --- Step 3: add reward_per_step ---
df["reward_per_step"] = opt_reward - df["regret_per_step"]

# --- Step 4: filter and remap ---
df = df[df["test"] == TEST_NAME].copy()

# round algo_param to nearest 0.025
df["algo_param"] = (df["algo_param"] / 0.025).round() * 0.025

if not ((df["algo_name"] == ALGO_FAMILY) & (df["algo_param"] == 1)).any():
    mask = (df["algo_name"] == "TSPostDiffTop") & (df["algo_param"] == 1)
    df.loc[mask, "algo_name"] = ALGO_FAMILY

# keep only the chosen family
df = df[df["algo_name"] == ALGO_FAMILY].copy()

# --- Step 5: compute obj_score for each w ---
w_list = np.linspace(0.0, 15, num=20).round(3).tolist()
records = []

for w in w_list:
    df_w = df.copy()
    df_w["w"] = w
    df_w["obj_score"] = (
        - df_w["reward_per_step"] * df_w["n_step"] / np.log(df_w["n_step"]) * w
        + df_w["n_step"]
    )
    records.append(df_w)

df_all = pd.concat(records, ignore_index=True)

# --- Step 6: add group labels ---
def assign_group(row):
    if row["algo_param"] in PARAM_CHOICES:
        val = int(row["algo_param"]) if float(row["algo_param"]).is_integer() else row["algo_param"]
        return f"{ALGO_FAMILY} ({val})"
    elif row["algo_param"] == "opt":
        return f"{ALGO_FAMILY} optimal"
    else:
        return None

df_all["group"] = df_all.apply(assign_group, axis=1)
df_all = df_all.dropna(subset=["group"])

# --- Step 7: add 'opt' row ---
opt_rows = []
for w in w_list:
    df_w = df_all[df_all["w"] == w]
    best = df_w.loc[df_w["obj_score"].idxmin()]
    row = best.copy()
    row["algo_param"] = "opt"
    row["group"] = f"{ALGO_FAMILY} optimal"
    opt_rows.append(row)

df_all = pd.concat([df_all, pd.DataFrame(opt_rows)], ignore_index=True)

# --- Step 7.5: normalize by optimal (so optimal = 0 everywhere) ---
opt_lookup = df_all[df_all["group"] == f"{ALGO_FAMILY} optimal"].set_index("w")["obj_score"]
df_all["obj_score"] = df_all.apply(lambda r: r["obj_score"] - opt_lookup.loc[r["w"]], axis=1)

# --- Step 8: plot ---
plot_groups = [f"{ALGO_FAMILY} ({p if not float(p).is_integer() else int(p)})" for p in PARAM_CHOICES] \
              + [f"{ALGO_FAMILY} optimal"]

# rename certain groups for legend
name_map = {
    f"{ALGO_FAMILY} (0)": "TS",
    f"{ALGO_FAMILY} (1)": "UR",
}

plt.figure(figsize=(8, 6))
for key, grp in df_all[df_all["group"].isin(plot_groups)].groupby("group"):
    display_name = name_map.get(key, key)
    plt.plot(
        grp["w"],
        grp["obj_score"],
        marker="o",
        markersize=4,
        linewidth=1,
        alpha=0.7,
        label=display_name
    )

plt.xlabel("w (1 unit reward improvement is worth w steps)")
plt.ylabel("Extra equivalent steps vs. optimal (lower is better)")
plt.title(f"Equivalent Steps Gap vs Reward-Value Weight\nFamily: {ALGO_FAMILY}, Test: {TEST_NAME}")
plt.legend()
plt.ylim(-5, 500)
plt.tight_layout()
plt.show()
