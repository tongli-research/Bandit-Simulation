# factorial-bandit

Code and data for the factorial PostDiff paper (joint with Eric Schwartz).

The paper itself (LaTeX, notes, related-work synthesis) lives in a separate private repo:
[github.com/tongli-research/paper-factorial-bandit](https://github.com/tongli-research/paper-factorial-bandit).

This folder collects only the parts of `mab-simulation` that generate the paper's
figures and tables. Everything else in `mab-simulation` is used by other projects
and isn't paper-specific.

## Setup

Scripts here import the `bandit_simulation` library at the repo root. From the
top of `mab-simulation/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run any script from the repo root, e.g.:

```bash
python factorial-bandit/fig1_postdiff_illustration.py
```

## Scripts

| Script | Output | What it shows |
|---|---|---|
| `fig1_postdiff_illustration.py` | `latex/figures/fig1_postdiff_illustration.pdf` | Two-arm posterior scatter: winner-circle (uniform exploration) vs direct exploitation under TSPostDiff |
| `fig2_mab_tradeoff_sweep.py` | `latex/figures/fig2_mab_tradeoff_sweep.pdf` | Reward/power trade-off across algorithms and sample budgets (MAB setting) |
| `fig3_factorial_sweep.py` | `latex/figures/fig3_factorial_sweep.pdf` | Same trade-off in the 2x3 factorial setting (the paper's main result) |
| `sim0a_power_analysis.py` | console / CSV in `data/` | Pre-experiment power analysis for the factorial design |
| `sim0b_pilot_calibration.py` | console / CSV in `data/` | Pilot run that calibrates sigma and effect sizes |
| `tab2_lack_of_fit.py` | text for `latex/sections/setup_results.tex` | Lack-of-fit table — model misspecification check |

`data/` holds simulation result files that some scripts read or write:
`March_12_full_results.csv`, `agts_sweep_R_results.csv`, `mab_tradeoff_results.json`,
`sweep_matched_results.csv`.

## Where the rest of the factorial-relevant material lives in `mab-simulation`

### Library code
- `bandit_simulation/` — algorithms and simulation engine. Key entry points for
  this paper:
  - `bandit_algorithm.py` — `TSPostDiff`, flat / linear variants
  - `sim_wrapper.py` — `run_simulation` driver
  - `simulation_configurator.py` — `SimulationConfig` builder

### Related scripts (outside this folder)
- `scripts/factorial_power_fpr.py` — factorial power and FPR sweep
- `scripts/factorial_ait_experiment.py` — algorithm-induced test on factorial designs
- `scripts/compare_flat_vs_linear_ts.py`, `scripts/full_flat_vs_linear_comparison.py`
  — flat vs linear TS comparison runs
- `scripts/wald_density_2arm_binary.py`, `scripts/wald_fpr_analysis.py`,
  `scripts/wald_gap_b_test.py` — Wald-test diagnostics

### Notes / specs
- `notes/factorial-bandit-eric.md` — factorial bandit project notes
- `notes/factorial-algorithms-and-metrics.md` — algorithms and metrics spec
  (2x3 factorial, K=6, d=3)
- `notes/flat-vs-linear-comparison-results.md` — flat vs linear posterior comparison

### Working notes (paper-specific Claude runs)
- `claude_scripts/2026-03-26_factorial-t400-sigma1/` — factorial T=400, sigma=1 run
- `claude_scripts/2026-04-02_power_analysis/` — analytical power-analysis verification
- `claude_scripts/2026-04-16_misspec/` — misspecification study (PostDiff-flat vs AG-TS)

## A note on duplication

The figure-output files in this folder's `data/` also exist in the paper repo as
`code/` and `data/` (kept locally there but not pushed). When updating either,
prefer this copy as the canonical source.
