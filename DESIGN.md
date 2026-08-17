# Bandit Simulation Framework — Design Document

This document describes the key architectural decisions and data structures in the simulation framework. **Read this before modifying any core code.**

---

## 1. Vectorized Simulation (The Core Design Principle)

All simulations are **fully vectorized across n_rep** (independent replications). There is no loop over replications — everything uses numpy broadcasting.

**Array shapes (compact mode, the default):**

| Array | Shape | Description |
|-------|-------|-------------|
| `action_hist` | `(n_rep, T_batch, n_arm)` | Cumulative pull counts per arm per batch step |
| `reward_hist` | `(n_rep, T_batch, n_arm)` | Cumulative reward sums per arm per batch step |
| `reward2_hist` | `(n_rep, T_batch, n_arm)` | Cumulative squared reward sums (for variance) |
| `arm_means` | `(n_rep, T_batch, n_arm)` | Cumulative mean reward per arm (cached) |
| `combined_means` | `(n_rep, T_batch, 1)` | Pooled mean across all arms |

Where `T_batch` is the length of the step schedule (typically 10–30 steps depending on horizon), NOT the raw horizon. Batch scheduling trades update frequency for speed.

**Pre-generated reward trajectory:** All rewards are sampled once at init and stored as `(n_rep, n_action_groups, n_arm)`. The policy only controls arm selection — it indexes into this fixed trajectory. This guarantees reproducibility and enables ART (algorithm-randomness testing).

---

## 2. Step Scheduling & Batching

Instead of updating the policy at every sample, the framework batches samples before updating:

- **`step_schedule`**: cumulative samples at each batch step. Example: `[60, 120, 180, 360, ...]`
- **`sample_batch_schedule`**: new samples per step. Example: `[60, 60, 60, 180, ...]`

Both grow as ~t^(1/3), so early steps are frequent (high exploration), late steps are sparse (lower cost). Burn-in always pulls each arm `burn_in_per_arm` times first.

When computing per-sample results (e.g., power at every horizon step), use `get_interpolation(batch_result, step_schedule)` to linearly interpolate from batch-level to sample-level.

---

## 3. SimResult: The Statistics Engine

`SimResult` (in `sim_wrapper.py`) wraps the simulation output and provides all statistical computations. Key methods:

**Test statistics:**
- `wald_test(arm1, arm2)` → `(n_rep, T_batch)` — pairwise z-test
- `t_control()` → `(n_rep, T_batch, K-1)` — all arms vs control
- `anova()` → `(n_rep, T_batch)` — F-test p-values
- `tukey()` → `(n_rep, T_batch, K, K)` — pairwise HSD
- `linear_regression_pvalues(F)` → `(n_rep, d-1)` — WLS on factorial coefficients

**Bootstrap:**
- `bootstrap_test(mode='art')` — fix rewards, re-randomize algorithm
- `bootstrap_test(mode='induced')` — re-simulate under estimated null

All statistics are cumulative over time — `arm_means[:, t, :]` gives the running mean up to step t.

---

## 4. Algorithm-Induced Test (AIT) — Location-Dependent Null Distributions

**This is the central insight of the Objective Function paper (Paper 2).**

For adaptively collected data (bandit algorithms), the null distribution of any test statistic **cannot be expressed in closed form**. It depends on:
1. Which algorithm was used (different allocation patterns → different null distributions)
2. Where in the parameter space H0 sits (the null distribution varies by location)

Therefore the null must be **simulated**, and simulated **at multiple H0 locations**.

### The AIT Procedure (Per-Rep Correction)

For each outer replication:
1. Run the algorithm → get observed data
2. **Estimate the null from observed data**: compute theta_hat, construct theta_null (e.g., keep intercept, zero out treatment effects)
3. **Re-simulate the same algorithm** under theta_null for N_inner reps → simulated null distribution of the test statistic
4. **Corrected p-value** = fraction of null stats ≥ observed stat

### Avoiding the Simulation Explosion (H0 Binning)

Naively, this requires N_outer × N_inner simulations (e.g., 10K × 200 = 2M). The framework avoids this via **H0 binning and interpolation**:

1. From the outer reps, estimate where each rep's null sits on the parameter space
2. **Bin** these into `n_crit_sim_groups` groups (typically 3–10)
3. At each **bin center** (called an "H0 core"), run one batch of null simulations
4. **Interpolate** the null distribution between cores using a weight matrix

**Two interpolation methods:**
- **`bin`**: Hard assignment — each rep uses its group's null distribution
- **`linear`** (default): Smooth interpolation between two nearest cores

**Implementation in the framework:**
```
get_h0_cores_and_weights(combined_means)
  → weight: (n_rep, n_cores)        # interpolation weights
  → h0_locations: (n_cores,)         # bin center values

# Run H0 simulation at each core location
run_simulation(H0_at_core_k) for each core k
  → core_crit_array: (n_cores, T_batch, ...)

# Interpolated critical boundary per rep
crit_boundary = tensordot(weight, core_crit_array)
  → (n_rep, T_batch, ...)
```

This reduces the cost to N_outer + n_cores × N_per_core, which is O(N_outer) — no explosion.

### Error Decomposition

The framework quantifies three sources of error (all in units of **steps**):
- **H1 variance** (`se_steps_h1`): finite H1 reps → noisy power estimate. Decreases as 1/√n_rep.
- **H0 variance** (`se_steps_h0`): finite H0 reps per core → noisy critical values. Decreases as 1/√n_crit_sim_rep.
- **H0 bias** (`bias_steps_h0`): finite grid → interpolation bias. Decreases with more cores. Uses GP smoothing (Matérn 2.5) to separate trend from noise.

---

## 5. Bayesian Posterior Models

All vectorized — update all n_rep simultaneously.

| Model | File location | Prior | Used when |
|-------|---------------|-------|-----------|
| `BetaBernoulli` | `bayes_vector_ops.py` | Beta(1,1) per arm | Bernoulli rewards |
| `NormalFull` | `bayes_vector_ops.py` | Normal-Inverse-Gamma per arm | Normal rewards, flat (independent arms) |
| `LinearNormalKnownVar` | `bayes_vector_ops.py` | N(mu0, σ²Σ0⁻¹) on theta | Normal rewards with feature matrix F |

**Linear model update** (the key one for factorial designs):
```python
# Precision-form posterior: A @ theta ~ N(b, σ² A⁻¹)
Sxx = einsum('rk,kd,ke->rde', counts, F, F)   # (n_rep, d, d)
Sxy = einsum('rk,kd->rd', rewards, F)          # (n_rep, d)
A = Σ0_inv + (1/σ²) Sxx                        # Posterior precision
b = Σ0_inv @ μ0 + (1/σ²) Sxy                   # Precision-weighted mean
```

**Sampling:** solve A, add noise, map to arm means via `F @ theta`.

**Config controls which model is used:**
- `arm_feature_matrix=None` → flat model (NormalFull or BetaBernoulli)
- `arm_feature_matrix=F` → linear model (LinearNormalKnownVar)

---

## 6. Algorithm Classes

All inherit from `BanditAlgorithm`. The `sample_action()` method is vectorized across n_rep.

| Class | algo_para | Posterior | Description |
|-------|-----------|-----------|-------------|
| `EpsTS` | float (epsilon) | any | ε-greedy Thompson Sampling. eps=0 → pure TS, eps=1 → uniform random |
| `TSPostDiffTopWithResample` | float (delta) | flat | Archived. Winner circle with two posterior draws (resample) |
| `TSPostDiff` | float (delta) | flat | Winner circle: arms within delta → uniform random from circle |
| `TSPostDiffLinear` | dict: `{delta_vec, resample}` | linear | Per-factor thresholding on posterior coefficients |

**TSPostDiffLinear** is the key factorial algorithm:
- Draw theta from posterior
- For each factor j: if |theta_j| ≤ delta_j → zero out (uncertain), else keep
- Arm values = F @ theta_modified → pick argmax
- `resample=True`: use a second posterior draw for arm selection (PostDiff two-draw)

---

## 7. Test Procedures

Abstract base `TestProcedure` in `test_procedure_configurator.py`. Each implements:
1. `get_test_statistics()` — compute the test stat from SimResult
2. `get_critical_region()` — compute critical values from H0 simulations
3. `compute_power()` — compare H1 stats against critical boundary

| Class | Test | Stat shape | Direction |
|-------|------|-----------|-----------|
| `ANOVA` | F-test | `(n_rep, T_batch)` p-values | -1 (small p → reject) |
| `TControl` | t-test vs control | `(n_rep, T_batch, K-1)` | +1 (large t → reject) |
| `TConstant` | t-test vs threshold | `(n_rep, T_batch, K)` | +1 |
| `Tukey` | Pairwise HSD | `(n_rep, T_batch, K, K)` | +1 |

---

## 8. File Map

| File | Role |
|------|------|
| `sim_wrapper.py` | Simulation engine: `run_simulation()`, `SimResult`, objective score |
| `simulation_configurator.py` | `SimulationConfig`, step scheduling, reward trajectory generation |
| `test_procedure_configurator.py` | `TestProcedure` ABC, ANOVA/TControl/TConstant/Tukey, H0 binning |
| `bayes_vector_ops.py` | Bayesian models: BetaBernoulli, NormalFull, LinearNormalKnownVar |
| `bandit_algorithm.py` | All bandit algorithms: EpsTS, UCB, TSPostDiffTop, TSPostDiffLinear, etc. |
| `analysis.py` | Post-hoc analysis: `linear_regression_test()`, summary tables, objective curves |
| `plotting.py` | Factor effect plots, gap plots |
