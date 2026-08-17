"""
Build the naive-vs-optimized, multi-w summary table for Table 6's
main-text T-Control panel.

Filters the full 4-test sweep results (from table4.yaml / table4_ucb.yaml)
down to T-Control, then for each w in --w-list computes ECP-reward
(= reward - w * log(n_step)) for every naive baseline and finds the
best-scoring parameter for each optimized family (eps-TS, UCB).

Usage (run from this directory, KDD_27/, so `results/` resolves here):
    python scripts/build_table6_w_sweep.py
    python scripts/build_table6_w_sweep.py --w-list 0.03 0.1 0.3
"""
import argparse

import numpy as np
import pandas as pd


def score(n_step, reward, w):
    return reward - w * np.log(n_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsts-csv", default="results/table4.csv")
    parser.add_argument("--ucb-csv", default="results/table4_ucb.csv")
    parser.add_argument("--w-list", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    parser.add_argument("--output", default="results/table6_w_sweep_summary.csv")
    args = parser.parse_args()

    eps = pd.read_csv(args.epsts_csv)
    eps = eps[eps["test_proc"].str.contains("T-Control")].copy()
    ucb = pd.read_csv(args.ucb_csv)
    ucb = ucb[ucb["test_proc"].str.contains("T-Control")].copy()

    naive_rows = {
        "Naive UR (eps=1)": eps[eps["algo_param"] == 1.0].iloc[0],
        "Naive TS (eps=0)": eps[eps["algo_param"] == 0.0].iloc[0],
        "Naive eps-TS(0.5)": eps[eps["algo_param"] == 0.5].iloc[0],
        "Naive UCB (c=2)": ucb[ucb["algo_param"] == 2.0].iloc[0],
    }

    records = []
    for name, row in naive_rows.items():
        rec = {"row": name}
        for w in args.w_list:
            rec[f"w={w}"] = round(score(row["n_step"], row["regret_per_step"], w), 4)
        records.append(rec)

    for name, df in [("Optimized eps-TS", eps), ("Optimized UCB", ucb)]:
        rec = {"row": name}
        for w in args.w_list:
            df["score"] = score(df["n_step"], df["regret_per_step"], w)
            best = df.loc[df["score"].idxmax()]
            rec[f"w={w}"] = f"{best['score']:.4f} (param={best['algo_param']:g})"
        records.append(rec)

    summary = pd.DataFrame(records)
    print(summary.to_string(index=False))
    summary.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
