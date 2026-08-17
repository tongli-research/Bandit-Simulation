"""
Run a bandit simulation experiment from a YAML config file.

Usage:
    python scripts/run_from_config.py configs/table4.yaml
    python scripts/run_from_config.py configs/table4.yaml -o results/table4.csv
"""

import argparse
import os
import time

from bandit_simulation.config_loader import load_config
from bandit_simulation.sim_wrapper import sweep_and_run


def main():
    parser = argparse.ArgumentParser(
        description="Run bandit simulation from YAML config"
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--output", "-o",
        help="Output CSV path (default: results/<config_name>.csv)",
    )
    args = parser.parse_args()

    # Default output path: results/<config_stem>.csv
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.config))[0]
        args.output = f"results/{stem}.csv"

    print(f"Loading config: {args.config}")
    base_config, sweep_specs = load_config(args.config)

    # Show what we're about to run
    total_combos = 1
    for spec in sweep_specs:
        for k, v in spec.items():
            n = len(v) if isinstance(v, list) else 1
            print(f"  {k}: {n} values")
            total_combos *= n
    print(f"  Total combinations: {total_combos}")
    print()

    t_start = time.time()
    df = sweep_and_run(sweep_specs, base_config)
    elapsed = time.time() - t_start

    print(f"Completed in {elapsed:.1f}s")
    print(f"Results shape: {df.shape}")
    print()

    # Display key columns
    display_cols = [
        c for c in ["algo_name", "algo_param", "test_proc", "n_step",
                     "regret_per_step", "power_max", "deployment_regret"]
        if c in df.columns
    ]
    print(df[display_cols].to_string(index=False))

    # Save CSV (drop nested dict columns that don't serialize)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    drop_cols = [c for c in df.columns if c == "all_results"]
    df.drop(columns=drop_cols, errors="ignore").to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
