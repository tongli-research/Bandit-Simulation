"""
Table 1's "Power analysis (AIT)" row: the sample size Thompson Sampling and
Uniform Random need to reach 80% power, at a fixed Cohen's h, across null
locations mu (tab:t_test_classical_vs_AIT_fpr).

Effect size: raw delta=p1-p2 is NOT location-invariant for Bernoulli
outcomes (variance p(1-p) depends on p), so a fixed raw delta needs a
location-dependent sample size even for UR. Uses Cohen's h instead (the
arcsine-transform effect size for proportions, Cohen 1988): delta at each
mu is chosen to match h=0.3444, the reference from mu=0.9, delta=0.1
(arms 0.95/0.85).

Reuses run_task_common (bandit_simulation.sim_wrapper) directly for the
H1-sim + H0-core-binning + interpolated-critical-region pipeline and its
n_step estimator (the same one Table 6 uses) rather than a separate
ad hoc crossing-search, for full methodological consistency and to avoid
duplicating that pipeline.

Usage (run from this directory, KDD_27/):
    python scripts/table1_ts_ur_power_analysis.py
"""
import time

import numpy as np
import pandas as pd

from bandit_simulation.bandit_algorithm import EpsTS
from bandit_simulation.simulation_configurator import SimulationConfig
from bandit_simulation.test_procedure_configurator import ANOVA
from bandit_simulation.sim_wrapper import run_task_common

N_REP = 20000
HORIZON = 4000
TARGET_POWER = 0.8

# delta per location, matched to Cohen's h=0.3444 (mu=0.9, delta=0.1 reference)
DELTA_BY_MU = {0.1: 0.1000, 0.3: 0.1566, 0.5: 0.1713, 0.7: 0.1566, 0.9: 0.1000}


def run_one(mu, delta, eps):
    arm_means = [mu + delta / 2, mu - delta / 2]
    base_config = SimulationConfig(
        n_rep=N_REP, n_arm=2, horizon=HORIZON, burn_in_per_arm=1,
        reward_model=np.random.binomial,
        arm_mean_reward_dist_spec={"dist": "normal", "params": {"loc": arm_means, "scale": 0}},
        test_procedure=ANOVA(type1_error_constraint=0.05, power_constraint=TARGET_POWER, min_effect=0.0),
        reward_evaluation_method="reward",
    )
    return run_task_common(base_config, algo=EpsTS, algo_param_list=[eps])


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    for algo_name, eps in [("TS", 0.0), ("UR", 1.0)]:
        for mu, delta in DELTA_BY_MU.items():
            t_combo = time.time()
            result = run_one(mu, delta, eps)
            n_step = result["n_step"]
            print(f"{algo_name} mu={mu}: n_step={n_step:.0f} ({time.time()-t_combo:.1f}s)", flush=True)
            rows.append({"algo": algo_name, "mu": mu, "delta": delta, "n_step": n_step})

    print(f"\nTotal: {time.time()-t0:.1f}s")
    pd.DataFrame(rows).to_csv("results/table1_ts_ur_power_analysis_results.csv", index=False)
    print("Saved: results/table1_ts_ur_power_analysis_results.csv")
