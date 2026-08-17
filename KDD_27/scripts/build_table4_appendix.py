"""
Build the appendix ECP-reward panels (ANOVA / T-Constant / Tukey) across w,
the counterpart of the main-text T-Control panel (build_table6_w_sweep.py).

Reads the full 4-test sweep results (table4.csv / table4_ucb.csv), and for each
test and each w in --w-list computes ECP-reward (= reward - w*log(n_step)) for
every naive baseline and the best-scoring parameter of each optimized family.

Usage (from KDD_27/): python scripts/build_table4_appendix.py
"""
import argparse
import numpy as np
import pandas as pd

TESTS = [("ANOVA", "ANOVA"), ("T-Constant", "T-Constant"), ("Tukey", "Tukey")]


def score(n_step, reward, w):
    return reward - w * np.log(n_step)


def panel_for_test(eps, ucb, pattern, w_list):
    e = eps[eps["test_proc"].str.contains(pattern)].copy()
    u = ucb[ucb["test_proc"].str.contains(pattern)].copy()
    naive = {
        "Naive UR (eps=1)": e[e["algo_param"] == 1.0].iloc[0],
        "Naive TS (eps=0)": e[e["algo_param"] == 0.0].iloc[0],
        "Naive eps-TS(0.5)": e[e["algo_param"] == 0.5].iloc[0],
        "Naive UCB (c=2)": u[u["algo_param"] == 2.0].iloc[0],
    }
    records = []
    for name, row in naive.items():
        rec = {"method": name}
        for w in w_list:
            rec[f"w={w}"] = round(score(row["n_step"], row["regret_per_step"], w), 4)
        records.append(rec)
    for name, df in [("Optimized eps-TS", e), ("Optimized UCB", u)]:
        rec = {"method": name}
        for w in w_list:
            df["score"] = score(df["n_step"], df["regret_per_step"], w)
            best = df.loc[df["score"].idxmax()]
            rec[f"w={w}"] = f"{best['score']:.4f} (param={best['algo_param']:g})"
        records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsts-csv", default="results/table4.csv")
    parser.add_argument("--ucb-csv", default="results/table4_ucb.csv")
    parser.add_argument("--w-list", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    parser.add_argument("--output", default="results/table4_appendix_w_sweep.csv")
    args = parser.parse_args()

    eps = pd.read_csv(args.epsts_csv)
    ucb = pd.read_csv(args.ucb_csv)

    all_rows = []
    for test_name, pattern in TESTS:
        for rec in panel_for_test(eps, ucb, pattern, args.w_list):
            all_rows.append({"test": test_name, **rec})

    summary = pd.DataFrame(all_rows)
    print(summary.to_string(index=False))
    summary.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
