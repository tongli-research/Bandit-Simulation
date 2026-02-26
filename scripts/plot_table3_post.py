"""
Table 3 — Post (realized) comparison.

Reads pre-computed simulation results for TS and epsilon-TS under the
empirically observed 6-arm reward means, and prints the comparison table.

Usage:
    python scripts/run_from_config.py configs/table3_post_ts.yaml
    python scripts/run_from_config.py configs/table3_post_epsts.yaml
    python scripts/plot_table3_post.py
"""

import pandas as pd

TS_PATH = "results/table3_post_ts.csv"
EPSTS_PATH = "results/table3_post_epsts.csv"

# UR reward: arithmetic mean of the 6 empirical arm means
arm_means = [0.81, 0.805, 0.801, 0.777, 0.827, 0.812]
reward_UR = sum(arm_means) / len(arm_means)

df_ts = pd.read_csv(TS_PATH)
df_epsts = pd.read_csv(EPSTS_PATH)

reward_TS = df_ts.iloc[0]["regret_per_step"]
reward_epsTS = df_epsts.iloc[0]["regret_per_step"]

print("Table 3 — Post (Realized) Performance")
print("=" * 50)
print(f"  UR (naive):           reward = {reward_UR:.6f}")
print(f"  TS (eps=0, T=4186):   reward = {reward_TS:.6f}")
print(f"  eps-TS (eps=0.3, T=1338): reward = {reward_epsTS:.6f}")
