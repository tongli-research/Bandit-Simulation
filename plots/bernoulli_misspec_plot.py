import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
w=10

# 1. Load data
df = pd.read_csv("results/bernoulli_misspecification.csv")

# 2. Compute new objective score
df["obj_score_custom"] = df["n_step"] - w* (df["n_step"] / np.log(df["n_step"])) * df["regret_per_step"]

# 3. Round algo_param to nearest 0.025
df["algo_param_rounded"] = (df["algo_param"] / 0.025).round() * 0.025

# 4. For loc = 0.35, find opt_param (the minimizing param)
df_loc035 = df[df["arm_mean_reward_dist_loc"] == 0.35]
opt_param = df_loc035.loc[df_loc035["obj_score_custom"].idxmin(), "algo_param_rounded"]

print("opt_param for loc=0.35:", opt_param)

# 5. Compute scores for all loc
results = []
for loc, g in df.groupby("arm_mean_reward_dist_loc"):
    # score for opt_param
    opt_score = g.loc[g["algo_param_rounded"] == opt_param, "obj_score_custom"].min()

    # min score + corresponding param
    idxmin = g["obj_score_custom"].idxmin()
    min_score = g.loc[idxmin, "obj_score_custom"]
    min_param = g.loc[idxmin, "algo_param_rounded"]

    results.append({
        "loc": loc,
        "opt_score": opt_score,
        "min_score": min_score,
        "min_param": min_param
    })

res_df = pd.DataFrame(results)

# 6. Plot
plt.figure(figsize=(8,6))
plt.plot(res_df["loc"], res_df["opt_score"], marker="o", label=f"ε={opt_param:.3f}")
plt.plot(res_df["loc"], res_df["min_score"], marker="x", label="Best per location")

# annotate min param values BELOW points
for _, row in res_df.iterrows():
    plt.annotate(
        f"(ε = {row['min_param']:.3f})",
        (row["loc"], row["min_score"]),
        textcoords="offset points",
        xytext=(0,-15),
        ha="center",
        fontsize=9
    )

plt.xlabel(r"$E(\mu_i)$, the location of the arm mean distribution", fontsize=14)
plt.ylabel("Objective Score", fontsize=14)
plt.title(r"Mis-specification performance for $\varepsilon$-TS optimized at $E(\mu_i)=0.35$", fontsize=16)
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()