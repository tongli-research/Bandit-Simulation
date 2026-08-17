# Error Decomposition — Implementation & Caveats

## Overview

Three error metrics are computed in `sim_wrapper.py::get_objective_score()` (line ~1238), all expressed as **steps**:

| Metric | Method | Source |
|--------|--------|--------|
| `se_steps_h1` | Bootstrap (H1 only) | Variance from finite H1 reps |
| `se_steps_h0` | Bootstrap (H0 only) | Variance from finite H0 reps per core |
| `se_steps_total` | Bootstrap (H1 + H0) | Combined variance |
| `bias_steps_h0` | Analytical (worst-case) | Interpolation bias from finite H0 grid |

Additionally, `reward_se` is computed separately (not via bootstrap).

---

## 1. Bootstrap SE — `bootstrap_steps_se()` (sim_wrapper.py:1009)

### Idea
At the fixed stopping step T = `n_step`, resample the simulation data and recompute power. Map the power deviation back to steps via the original power curve.

### Inputs
- `h1_test_stat`: (M1, H_batch, D) — H1 test statistics at batch resolution
- `h0_test_stat`: (B*M0, H_batch, D_h0) — H0 test statistics, B cores × M0 reps each
- `weight`: (M1, B) — interpolation weights from `get_h0_cores_and_weights`
- `min_effect_filter`: (M1,) or (M1, D) — boolean mask for min-effect filtering
- `power_curve`: (horizon,) — interpolated power curve at per-sample resolution
- `n_step`: median stopping time (per-sample units)

### Setup
1. Map `n_step` to batch index `t_idx` via `searchsorted(cumsum(step_schedule), n_step)`
2. Extract single-step slices: `h1_at_t = h1_test_stat[:, t_idx, :]` shape (M1, D)
3. Extract per-core H0 slices: `h0_core_slices[b] = h0_test_stat[b*M0:(b+1)*M0, t_idx:t_idx+1, :]`
4. Compute original per-core critical values: `orig_core_crits[b] = quantile(h0_core_slices[b])`

### Three Bootstrap Loops (each n_boot=200 replicates)

**Total (H1 + H0 resampled):**
```
for each replicate:
    1. Resample M1 H1 indices → h1_b, weight_b, filter_b
    2. Resample M0 H0 indices per core → recompute core_crits_b
    3. Interpolate: crit_b = weight_b @ core_crits_b
    4. power_b = reject_fraction(h1_b, crit_b, filter_b)
    5. shifted_target = 2 * power_target - power_b
    6. boot_total[i] = find_step(shifted_target) on original power curve
```

**H1-only (H0 fixed):**
```
Same as total, but step 2 uses orig_core_crits (no H0 resampling)
```

**H0-only (H1 fixed):**
```
Same as total, but step 1 uses original h1_at_t, weight, filter (no H1 resampling)
```

### Output
```python
se_total = std(boot_total)    # combined SE in steps
se_h1    = std(boot_h1)       # H1-only SE in steps
se_h0    = std(boot_h0)       # H0-only SE in steps
```

### Key detail: power-to-steps mapping
`shifted_target = 2 * power_target - power_b` reflects the bootstrap replicate: if the bootstrap power is higher than target, the shifted target is lower (fewer steps needed), and vice versa. `find_step()` walks the original power curve to convert this shifted target to a step count.

### Skipped cases
- **Tukey test:** returns all zeros (complex reject logic not supported)
- **n_step ≤ 0 or ≥ horizon:** returns all zeros (power never reaches target)

---

## 2. Worst-Case Bias — `get_objective_score()` (sim_wrapper.py:1276)

### Idea
The critical boundary is linearly interpolated between B grid points. The bias bounds how much the power could differ if we used the exact endpoint critical values instead.

### Steps
1. **GP-smooth** core crits to remove H0 variance noise: `_gp_smooth_cores()` (Matern 2.5 + WhiteKernel, only when n_cores ≥ 3)
2. For each H1 rep, find its **left and right** H0 grid points from the weight matrix
3. Compute full power curves at both endpoints: `power_left`, `power_right`
4. Worst-case power deviation at the stopping step:
   ```
   bias_power = max(|power_left - power|, |power_right - power|)  at step n_step
   ```
5. Translate to steps: `shifted = power_target - bias_power` → `find_step(shifted)` → `bias = |shifted_step - n_step|`

---

## 3. Reward SE — `get_objective_score()` (sim_wrapper.py:1326)

### Idea
Standard error of the per-rep reward at the fixed median stopping step. Measures how precisely the mean reward is estimated across H1 replications.

### Steps
1. Map `n_step_int` to batch index: `searchsorted(cumsum(step_schedule), n_step_int)`
2. Read each rep's cumulative mean reward: `per_rep_reward = h1_res.combined_means[:, batch_idx, :].flatten()`
3. `reward_se = std(per_rep_reward) / sqrt(n_rep)`

This is NOT computed via bootstrap — it's a direct sample SE.

---

## 4. How Errors Are Combined on the Frontend

### Loading page (index.html) — accuracy panel
```
Relative Steps SE = avg(se_steps_total) / avg(n_step) × 100  (as %)
H1 contribution   = avg(se_steps_h1²) / avg(se_steps_h1² + se_steps_h0²) × 100
H0 contribution   = 100 - H1 contribution
Worst-Case Bias   = avg(bias_steps_h0) / avg(n_step) × 100  (as %)
```

### Output page (recommend.html) — summary table
```
stepsSE  = round(sqrt(se_steps_h1² + se_steps_h0²) + bias_steps_h0)
rewardSE = reward_se   (displayed as ±value)
```
Note: the summary table uses the quadrature formula, not `se_steps_total` directly.

### Output page — full data table
Shows `se_steps_h1`, `se_steps_h0`, `bias_steps_h0` as separate columns.

---

## Caveats

### Caveat 1: Critical Boundary Variance Introduces Bias
Power = P(S > c). When c is estimated with variance, E[P(S > ĉ)] ≠ P(S > c) — biased toward 0.5 (Jensen's inequality). The bootstrap captures the variance effect but not this systematic bias. With few H0 reps, this bias can be significant.

### Caveat 2: Interpolation Bias Bound Assumes Smoothness
The bias estimate assumes `c(θ)` is bounded by endpoint values within each bin. If the true critical boundary has local extrema between grid points, actual bias could exceed the bound.

### Caveat 3: Bootstrap Is Single-Step
The bootstrap operates at one fixed batch step (`t_idx` corresponding to `n_step`). It does not resample the full power curve — it only measures variability of power at the stopping step and maps that to step uncertainty via the original (non-resampled) curve.
