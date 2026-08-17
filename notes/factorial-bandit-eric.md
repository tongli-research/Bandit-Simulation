# Factorial Bandit Project (Joint with Eric)

> Reference doc: `~/Documents/tong-notes/tong-phd-research/factorial_bandit_paper.md`
> Task IDs: R5 (research), B2 (code)

## What This Is

Extension of PostDiff to **factorial bandit** settings where arms have multi-factor structure
(e.g., 2x3 marketing experiment: headline style x visual intensity). Joint paper with Eric.

**Core contribution:** Standard TS in factorial experiments causes poor estimation of
factor-level treatment effects (beyond the known FPR/power issues). The winner circle
mechanism addresses this by spreading allocation across near-optimal arms.

## The 4 Algorithms (2x2 family)

|  | Flat winner circle | Factorial winner circle |
|--|---|---|
| **Uniform from W** | Alg 1: TSPostDiff | Alg 3: TSPostDiffLinear (resample=False) |
| **PostDiff two-draw (archived)** | TSPostDiffTopWithResample | TSPostDiffLinear (resample=True) |

- **Flat** winner circle: arms within delta of sampled best (arm-level gap)
- **Factorial** winner circle: per-factor threshold on linear coefficients (factor-level)
- **Uniform**: select uniformly from winner circle
- **PostDiff two-draw**: second independent TS draw; override with uniform only if TS agrees with winner circle

## Code Mapping

| Paper concept | Code location |
|---|---|
| Alg 1: TS-WC (flat, uniform) | `bandit_algorithm.py` — `TSPostDiff` |
| Archived: flat two-draw | `bandit_algorithm.py` — `TSPostDiffTopWithResample` |
| Alg 2: TS-WC-Factorial (linear) | `bandit_algorithm.py` — `TSPostDiffLinear` (resample=False) |
| Archived: linear two-draw | `bandit_algorithm.py` — `TSPostDiffLinear` (resample=True) |
| Factorial metrics computation | `sim_wrapper.py` — `compute_linear_factorial_metrics()` |
| Factor effect plots | `plotting.py` — `plot_all_factor_effects()` |
| Gap plots | `plotting.py` — `plot_gap()` |
| Summary table | `analysis.py` — `format_linear_summary()` |
| Main simulation script | `scripts/test_factorial_metrics_plot.py` |
| Output directory | `scripts/_out/` |

## Simulation Setup

- **Design:** 2x3 factorial, 6 arms (A-F), feature matrix `[1, x1, x2]`
- **Ground truth:** theta = [0.3, 0.2, 0.1] (intercept, X1 effect, X2 effect per unit)
- **Arm means:** A=0.3, B=0.5, C=0.4, D=0.6, E=0.5, F=0.7
- **Config:** 10k reps, horizon=600, sigma=0.5, burn_in=1 per arm

## Metrics

1. **Mean reward** — cumulative reward performance
2. **Treatment effect estimates** — X1 (1v0), X2 (1v0), X2 (2v1) via difference of means
3. **Gap** — distance from best arm over time
4. Factor effects plotted over time with variance bands

## Key Results So Far (from paper doc)

- UR: unbiased effects, lowest variance, but reward = 0.500
- TS: highest reward (0.675), but biased effects (X2 2v1: 0.004 vs true 0.100)
- PostDiff-WC(0.1): reward 0.656, reduced variance vs TS
- PostDiff-WC-Factorial([0.1,0.1]): reward 0.643, lowest variance on X2 effects

## TODOs (from paper doc)

### Simulation
- [ ] Sweep delta: 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5
- [ ] Add allocation plots (per-factor balance over time)
- [ ] Add t-test/z-test p-values
- [ ] Add posterior summaries of beta_1, beta_2
- [ ] Add ANOVA results
- [ ] Add FPR and power for factor-level contrasts
- [ ] Test different ground-truth mu configs
- [ ] Test different horizons

### Paper
- [ ] Clean up Overleaf algorithms (unified winner circle format)
- [ ] Unify notation (delta vs epsilon)
- [ ] Write Introduction, Discussion, Conclusion
