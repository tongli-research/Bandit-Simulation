
import pandas as pd
import numpy as np


import pickle

# Load results.pkl
with open("results.pkl", "rb") as f:
    results = pickle.load(f)

import pandas as pd



flattened_rows = []

for result in results:
    for sublist in result:
        for row_dict in sublist:
            if isinstance(row_dict, dict):
                row = {k: float(v) if isinstance(v, (np.floating, float, int)) else v for k, v in row_dict.items()}
                flattened_rows.append(row)
            else:
                print("Warning: Non-dict found!", type(row_dict))

df_results = pd.DataFrame(flattened_rows)
df_results.to_csv("results0703.csv", index=False)


df1= pd.read_csv("results0703.csv")
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Compute performance drop and std
df1['performance_drop'] = df1['obj_score(param_current_best)'] - df1['obj_score(param_base)']
df1['performance_std'] = (
    df1['obj_score_std(param_base)'] + df1['obj_score_std(param_current_best)']
) / np.sqrt(2)

# Unique tests
test_names = df1['test_name'].unique()

for test in test_names:
    plt.figure(figsize=(10, 6))
    df_test = df1[df1['test_name'] == test]

    # Prepare bar positions
    var_values = sorted(df_test['var_value'].unique())
    n_algos = df_test['algo_name'].nunique()
    bar_width = 0.8 / n_algos  # Total bar group width ~0.8
    x = np.arange(len(var_values))

    for idx, algo in enumerate(df_test['algo_name'].unique()):
        df_algo = df_test[df_test['algo_name'] == algo].sort_values('var_value')
        plt.bar(
            x + idx * bar_width,
            df_algo['performance_drop'],
            yerr=df_algo['performance_std'],
            width=bar_width,
            capsize=4,
            label=algo
        )

    plt.xticks(x + bar_width * (n_algos - 1) / 2, var_values)
    plt.xlabel("true setting location")
    plt.ylabel("Performance Drop")
    plt.title(f"Performance Drop for different algorithms ({test})")
    plt.legend(title="Algorithm")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
