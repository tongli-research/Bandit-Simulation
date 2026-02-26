"""
Figure 1 — Relative ECP-reward curves for Table 3 (prior / design-time).

Reads pre-computed simulation results and generates the ECP-reward plot
comparing Pure-TS, Pure-UR, and the optimized epsilon-TS.

Usage:
    python scripts/run_from_config.py configs/table3_prior.yaml
    python scripts/plot_table3_prior.py
"""

import numpy as np
import pandas as pd

from bandit_simulation.analysis import select_curves_relative
from bandit_simulation.plotting import plot_curves

RESULTS_PATH = "results/table3_prior.csv"

df = pd.read_csv(RESULTS_PATH)

selectors = [
    ("EpsTS", "param", 0.0, "Pure-TS", "#28A745", "--"),
    ("EpsTS", "param", 1.0, "Pure-UR", "#DC3545", "--"),
    ("EpsTS", "w", 0.01, None, "#007BFF", "-"),
]

w_values = np.arange(0.00, 0.06, 0.001)
curves = select_curves_relative(df, selectors, w_values=w_values)
plot_curves(curves, -0.03, df=df, w_values=w_values)
