# ECP-Reward: Simulation Code

Code to reproduce the simulation tables and figures in *A Statistically
Reliable Optimization Framework for Bandit Experiments in Scientific Discovery*.
It uses the `bandit_simulation/` core library in the parent directory.

Run every command from this directory so `results/` resolves here:

```bash
cd KDD_27
export PYTHONPATH=..
```

Outputs are written to `results/`. Each section below reproduces one paper
table or figure.

---

## Table 1: FPR inflation and power-analysis sample size

`table1_ts_ur_power_analysis.py` reports the sample size Thompson Sampling (TS)
and Uniform Random (UR) each need to reach 80% power for a two-sample t-test, at
a fixed **Cohen's h** effect size, across null locations `mu`. Cohen's h (the
arcsine-transform effect size for proportions) is used instead of the raw
difference `p1 - p2`, because Bernoulli variance `p(1-p)` depends on `p`, so a
fixed raw difference would need a location-dependent sample size even for UR.

```bash
python scripts/table1_ts_ur_power_analysis.py
```

Writes `results/table1_ts_ur_power_analysis_results.csv` (10 combos:
TS/UR x 5 null locations; n_rep=20000, horizon=4000).

---

## Table 2: ART vs Queue vs AIT (power and FPR)

`ait_art_comparison.py` compares three null-calibration methods (ART, Queue,
AIT) for a hypothesis test under adaptive sampling, over three policies
(TS, eps-greedy(0.1), UCB with c=2). Each `--test` option writes its own CSV
plus a `_raw.npz` (per-rep, per-cell reject arrays and true arm means, so power
can be recomputed at any `min_effect`).

Main text (two-arm t-test, means 0.6/0.4):

```bash
python scripts/ait_art_comparison.py --test t-test --n-rep-outer 10000 --n-rep-inner 1000 \
    --output results/table2_ttest.csv
```

Appendix (three-arm, H1 means from Beta(3,3)):

```bash
for t in anova tukey t-constant t-control; do
  python scripts/ait_art_comparison.py --test $t --n-rep-outer 5000 --n-rep-inner 500 \
      --output results/table2_$t.csv
done
```

Each CSV has power (conditional on `min_effect`), unconditional power (`*_unc`),
and FPR, per method. UCB + ART is degenerate and reported as power = FPR = alpha.

---

## Table 6 (main text and appendix): ECP-reward across tests and w

First run the two objective-score sweeps (all four tests x three arms, over the
eps-TS and UCB families):

```bash
python scripts/run_from_config.py configs/table4.yaml     -o results/table4.csv
python scripts/run_from_config.py configs/table4_ucb.yaml -o results/table4_ucb.csv
```

Then build the panels. Each reports, per `w`, every naive baseline's ECP-reward
plus the best-scoring parameter of each optimized family:

```bash
python scripts/build_table6_w_sweep.py      # main text: T-Control x 3 w
python scripts/build_table4_appendix.py     # appendix: ANOVA / T-Constant / Tukey x 3 w
```

`w = 0.03 / 0.1 / 0.3` are chosen so the optimized design's best parameter
visibly shifts across columns (smaller w favors more TS-like exploitation,
larger w favors more UR-like exploration) while still beating the naive
baselines. Writes `results/table6_w_sweep_summary.csv` and
`results/table4_appendix_w_sweep.csv`.

---

## Critical-value figure: ART vs Queue vs AIT calibration

`sim5_art_queue_mle_critval.py` computes, for each outer rep, each method's 95%
critical value for the LRT statistic, calibrated from that rep's own observed
history under the true null. `plot_critval_distribution.py` then plots the three
distributions against the theoretically optimal threshold.

```bash
python scripts/sim5_art_queue_mle_critval.py       # -> results/sim5_art_queue_mle_critval_results.csv
python scripts/plot_critval_distribution.py        # -> results/critval_distribution.{png,pdf}
```

Queue and MLE concentrate near the optimal threshold; ART is biased and
high-variance.

---

## Supporting: UCB c-range check

```bash
python scripts/ucb_c_bracket_check.py
```

Not required to reproduce any table; it documents the pull-allocation check used
to pick the `c` grid in `configs/table4_ucb.yaml` (one extreme allocates almost
uniformly across arms, the other collapses the worst arm's allocation toward 0).

---

## Other tables and figures

Tables 3 and 5 and the GUI figure are unchanged from the original submission;
see `../README.md` at the repository root.
