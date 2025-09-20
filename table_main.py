import numpy as np
import pandas as pd

# --- Step 1: compute opt_reward ---
def compute_opt_reward(n_samples=1_000_000, mu=0.5, sigma=0.15, cap=0.95):
    samples = np.random.normal(mu, sigma, size=(n_samples, 3))
    samples = np.minimum(samples, cap)  # cap values
    return np.mean(np.max(samples, axis=1))

opt_reward = compute_opt_reward()
print("Estimated opt_reward:", opt_reward)

# --- Step 2: load CSV ---
df = pd.read_csv("main_df0724.csv")

# --- Step 3: add reward_per_step ---
df["reward_per_step"] = opt_reward - df["regret_per_step"]

# round algo_param to nearest 0.025
df["algo_param"] = (df["algo_param"] / 0.025).round() * 0.025
# --- Step 4: remap algorithms ---
# PostDiffTop(param=1) -> UR
mask_ur = (df["algo_name"] == "TSPostDiffTop") & (df["algo_param"] == 1)
df.loc[mask_ur, "algo_name"] = "UR"

# EpsTS(param=0) -> TS
mask_ts = (df["algo_name"] == "EpsTS") & (df["algo_param"] == 0)
df.loc[mask_ts, "algo_name"] = "TS"

# --- Step 5: prepare test list ---
test_list = df["test"].unique().tolist()
print("Tests found:", test_list)

# --- Step 6: set w = 10 ---
w = 10

# compute obj_score at this w
df["obj_score_w10"] = (
    - df["reward_per_step"] * df["n_step"] / np.log(df["n_step"]) * w
    + df["n_step"]
)

# --- Step 7: get minimal score per (algo_name, test) ---
summary = (
    df.groupby(["algo_name", "test"])["obj_score_w10"]
    .min()
    .reset_index()
)

# --- Step 8: pivot into wide format (algorithms as rows, tests as columns) ---
table = summary.pivot(index="algo_name", columns="test", values="obj_score_w10")

# --- Step 9: save to CSV ---
table.to_csv("objective_score_table_w10_1.csv")

print("Saved table to objective_score_table_w10_2.csv")
print(table)
