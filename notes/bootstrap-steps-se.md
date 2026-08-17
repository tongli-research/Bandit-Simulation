# Bootstrap Steps SE Estimation

**Created:** 2026-02-25
**Status:** Design verified with standalone tests, not yet integrated into main code.

## Purpose

Estimate the standard error of `n_step` (the minimum sample size to achieve target power) by bootstrapping. Replaces the current discrete shift-on-fixed-curve method with a joint resampling approach that produces a continuous SE estimate.

## Notation

| Symbol | Meaning | Example value |
|--------|---------|---------------|
| `M1` | Number of H1 simulation reps (`sim_config.n_rep`) | 2000 |
| `B` | Number of H0 grid cores (`n_cores`) | 3 |
| `M0` | Number of H0 simulation reps per core (`tp.n_crit_sim_rep`) | 667 |
| `K` | Number of arms (`sim_config.n_arm`) | 3 |
| `H` | Number of horizon batch steps (length of `step_schedule`) | 100 |
| `horizon` | Total sample budget (`sim_config.horizon` = `sum(step_schedule)`) | 500 |
| `n_boot` | Number of bootstrap replicates | 1000 |

## Inputs (available after the standard pipeline completes)

All variables below use the ANOVA example: `M1=2000`, `K=3`, `B=3`, `M0=667`, `H=100` (with `step_schedule=[5]*100`, total `horizon=500`).

### From H1 simulation

| Variable | Code source | Shape | Description |
|----------|-------------|-------|-------------|
| `h1_test_stat` | `tp.get_test_statistics(h1_res)` | `(M1, H, D)` = `(2000, 100, 1)` | Test statistic for each H1 rep at each horizon batch step. `D` = trailing dim (ANOVA: 1, TControl: K-1 arms). |
| `weight` | `tp.get_h0_cores_and_weights(...)` — 1st return | `(M1, B)` = `(2000, 3)` | Interpolation weights. Each row sums to 1. Maps each H1 rep to its position between H0 grid cores. Most rows have only 2 nonzero entries (left and right neighbor). |
| `min_effect_filter` | `tp.create_min_effect_filter(...)` | `(M1,)` or `(M1, K-1)` | Bool mask. `True` = this rep participates in power calculation. `False` = excluded (set to NaN). Shape depends on test type: ANOVA → `(M1,)`, TControl → `(M1, K-1)`. |

### From H0 simulation

| Variable | Code source | Shape | Description |
|----------|-------------|-------|-------------|
| `core_crit_array` | 3rd return of `tp.get_adjusted_crit_region(weight, h0_res)` | `(B, H, D)` = `(3, 100, 1)` | Critical value at each H0 core, each horizon batch step. Estimated as a quantile of the null test statistics at that core. |
| `core_se_array` | Computed inside `get_adjusted_crit_region` but **not currently returned**. Extract via `compute_core_se_array()` or modify return signature. | `(B, H, D)` = `(3, 100, 1)` | SE of the critical value at each core. From order-statistic density estimation: `SE(c) = sqrt(q*(1-q)/M0) / f_hat(c)`. |

### Pipeline parameters

| Variable | Value (example) | Description |
|----------|----------------|-------------|
| `step_schedule` | `[5, 5, ..., 5]` (length `H`=100) | Samples added per batch step. |
| `horizon` | `500` | Total sample budget = `sum(step_schedule)`. |
| `power_target` | `0.80` | Required power level. |
| `crit_direction` | `-1` for ANOVA (left tail: reject when p-value < crit), `+1` for TControl (right tail: reject when stat > crit) | Determines the comparison direction in the reject step. |

## Algorithm

### For each bootstrap replicate b = 1, ..., n_boot:

#### Step 1: Resample H1 reps (captures H1 variance)

Draw `M1` indices with replacement from {0, 1, ..., M1-1}:

```
idx = rng.integers(0, M1, size=M1)       # (M1,) = (2000,)

h1_stat_b = h1_test_stat[idx]            # (M1, H, D) = (2000, 100, 1)
weight_b  = weight[idx]                  # (M1, B)    = (2000, 3)
filter_b  = min_effect_filter[idx]       # (M1,) or (M1, K-1)
```

**Why resample weight too:** Each H1 rep has its own estimated null parameter, which determines its interpolation weights. Resampling H1 reps changes the distribution of null parameters across reps.

#### Step 2: Perturb H0 critical values at core level (captures H0 variance)

Draw one independent standard normal per core, per horizon step, per trailing dimension:

```
z = rng.standard_normal(core_crit_array.shape)   # (B, H, D) = (3, 100, 1)

core_crit_b = core_crit_array + z * core_se_array  # (B, H, D) = (3, 100, 1)
```

**Critical design choice:** The noise is per-CORE (`B`=3 independent draws), not per-H1-rep (not `M1`=2000 draws). All H1 reps that share the same H0 core receive the same perturbation. This correctly models the uncertainty structure: `SE(c_hat)` reflects how much the critical value at that core would change if we re-ran the H0 simulation with different random seeds. All reps using that core are affected identically.

**What goes wrong with per-rep noise:** If you draw independent z for each of the `M1` H1 reps, the perturbations average out in the power mean (law of large numbers), giving a severely underestimated SE_H0. With `B`=3 cores, you have only 3 independent noise sources, not 2000.

#### Step 3: Interpolate through weight matrix

```
crit_b = tensordot(weight_b, core_crit_b, axes=(1, 0))
# weight_b:     (M1, B)    = (2000, 3)
# core_crit_b:  (B, H, D)  = (3, 100, 1)
# contraction:  axis 1 of weight_b (B) with axis 0 of core_crit_b (B)
# result:       (M1, H, D) = (2000, 100, 1)
```

Each H1 rep i gets: `crit_b[i] = sum_j weight_b[i,j] * core_crit_b[j]`. If rep i has weight `[0, 0.7, 0.3]`, its perturbed critical boundary = `0.7 * core_crit_b[1] + 0.3 * core_crit_b[2]`.

#### Step 4: Reject decision

```
# ANOVA (crit_direction = -1): reject when test stat < crit (p-value is small)
reject_b = h1_stat_b < crit_b            # (M1, H, D) = (2000, 100, 1), bool

# TControl (crit_direction = +1): reject when test stat > crit
reject_b = h1_stat_b > crit_b            # (M1, H, D) = (2000, 100, 1), bool

# Convert to float for NaN support
reject_b = reject_b * 1.0                # (M1, H, D) = (2000, 100, 1), float {0.0, 1.0}
```

#### Step 5: Apply min_effect_filter

```
# ANOVA: filter_b is (M1,) bool — 1D indexing sets entire row to NaN
reject_b[~filter_b] = np.nan

# TControl: filter_b is (M1, K-1) bool — needs broadcasting to (M1, H, K-1)
expanded = np.broadcast_to(filter_b[:, np.newaxis, :], reject_b.shape)
reject_b[~expanded] = np.nan
```

#### Step 6: Compute power curve

```
power_b = np.nanmean(reject_b, axis=(0, 2))
# axis 0: average over M1 reps (ignoring NaN from filter)
# axis 2: average over D trailing dims (ANOVA: D=1; TControl: D=K-1)
# result: (H,) = (100,) — power at each horizon batch step
```

#### Step 7: Interpolate to per-sample steps

```
power_interp_b = get_interpolation(power_b, step_schedule)
# power_b:      (H,)       = (100,)  — one value per batch (every 5 samples)
# step_schedule: (H,)       = (100,)  — [5, 5, ..., 5]
# result:       (horizon,)  = (500,)  — linearly interpolated to every sample
```

#### Step 8: Find n_step

```
n_step_b = horizon - np.sum(power_interp_b > power_target)
# power_interp_b: (500,)
# Count how many of the 500 steps exceed 0.80 power
# n_step = first step where power > target
# Example: if 345 steps exceed target → n_step = 500 - 345 = 155
```

### Aggregate across bootstrap replicates

```
boot_n_steps = [n_step_1, n_step_2, ..., n_step_n_boot]   # shape: (n_boot,)

SE_total = np.std(boot_n_steps)         # e.g., 2.34
mean_n_step = np.mean(boot_n_steps)     # e.g., 154.1
CI_95 = np.percentile(boot_n_steps, [2.5, 97.5])  # e.g., [149, 158]
```

## Decomposition (optional)

To attribute SE to H1 vs H0 sources separately:

| Variant | Step 1 (H1) | Step 2 (H0) | Result |
|---------|-------------|-------------|--------|
| **Full bootstrap** | Resample idx | Perturb z | `SE_total` |
| **H1 only** | Resample idx | Skip (use original `core_crit_array`) | `SE_H1` |
| **H0 only** | Skip (use all original reps) | Perturb z | `SE_H0` |

**Additivity check:** `sqrt(SE_H1^2 + SE_H0^2) ≈ SE_total` (holds because H1 and H0 simulations are independent).

## Required code change

`get_adjusted_crit_region()` in `test_procedure_configurator.py` currently returns 3-tuple:
```python
return adjusted_crit_region, se_adjusted, core_crit_array
```

Need to also return `core_se_array` (already computed internally at line 174):
```python
return adjusted_crit_region, se_adjusted, core_crit_array, core_se_array
```

All callers of `get_adjusted_crit_region` must be updated to unpack the 4th value.

## Validation results (2026-02-25)

### Standalone test: Bernoulli + UR + t-test (`tests/test_normal_steps_se.py`)

Compares current discrete method vs bootstrap (B=1000):

| Config | Method | SE_H1 | SE_H0 | SE_total | H0 share |
|--------|--------|-------|-------|----------|----------|
| NH0=1k, NH1=10k | Current | 1 | 6.0 | 6.1 | 97.3% |
| | Bootstrap | 1.49 | 1.92 | 2.36 | 62.6% |
| NH0=3k, NH1=10k | Current | 3 | 5.0 | 5.8 | 73.5% |
| | Bootstrap | 1.51 | 1.72 | 2.22 | 56.3% |
| NH0=10k, NH1=10k | Current | 3 | 3.5 | 4.6 | 57.6% |
| | Bootstrap | 2.28 | 1.35 | 2.34 | 25.9% |

Current method systematically overestimates SE (~1.5-2.5x) because it shifts the target on a fixed, noisy power curve. Bootstrap recomputes the entire power curve each time.

### Full pipeline test: ANOVA + EpsTS (`tests/test_bootstrap_steps_se.py`)

Bootstrap with core-level perturbation, B=500:
- Bootstrap SE_total ~1.7x lower than current method
- n_boot=200 adds only ~18% to pipeline runtime (H1+H0 simulations are 88% of cost)
- Additivity check: `sqrt(SE_H1^2 + SE_H0^2) ≈ SE_total` confirmed

## Key insight: density at critical quantile

The relative importance of H0 vs H1 depends on the density f0(c) at the critical quantile:
- **ANOVA p-values:** f0 ≈ 1.0 (nearly uniform under H0) → small SE(c_hat) → H0 share ~25%
- **Normal/t-test statistics:** f0 ≈ 0.10 (thin tail) → large SE(c_hat) → H0 share ~60-97%

Formula: `SE(c_hat) = sqrt(q*(1-q)/N) / f0(c)`. Lower density = larger SE = more H0 variance.
