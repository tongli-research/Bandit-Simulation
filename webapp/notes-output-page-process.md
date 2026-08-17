# Output Page (recommend.html) — Full Process Documentation

## 1. Data Sources

Three JSON datasets are injected from Flask into the template via Jinja2:

### 1a. `rawData` (from `chart_data_json`)
- **Source:** `recommendation.py` → `df[columns].to_json(orient='records')`
- **One row per (algo, param) configuration** that was simulated
- **Fields per row:**

| Field | Source (sim_wrapper.py) | Description |
|-------|------------------------|-------------|
| `algo_name` | sweep_and_run | Algorithm class name (e.g. "EpsTS") |
| `algo_param` | sweep_and_run | Parameter value (e.g. 0.333) |
| `n_step` | `get_objective_score` → `np.median(n_step_dist)` | Median stopping time (samples) |
| `regret_per_step` | `get_objective_score` → `mean_reward[median(n_step_dist-1)]` | Average reward at median stopping step |
| `power_max` | `get_objective_score` → `np.max(power)` | Maximum achieved power across all steps |
| `se_steps_h1` | Bootstrap SE (H1 component) | Steps SE from H1 rep variance |
| `se_steps_h0` | Bootstrap SE (H0 component) | Steps SE from H0 rep variance |
| `se_steps_total` | Bootstrap SE (total) | Combined steps SE |
| `bias_steps_h0` | Worst-case bias translation | Interpolation bias in steps |
| `log_n_step_sd` | `np.std(np.log(n_step_dist))` | SD of log stopping times |
| `obj_score_sd` | `np.std(reward_at_n_step * n_step_dist)` | SD of total cumulative reward |
| `reward_se` | `np.std(per_rep_reward) / sqrt(n_rep)` | SE of per-rep reward at fixed median step |

### 1b. `gpAtRaw` (from `gp_at_raw_json`)
- **Source:** `fit_gp_curves()` → GP predictions evaluated at raw (simulated) param values
- **One row per (algo, param)** — same params as rawData
- **Fields per row:**

| Field | Description |
|-------|-------------|
| `algo_name` | Algorithm name |
| `algo_param` | Same param values as raw |
| `n_step` | GP-predicted steps (fitted on log scale: exp(GP(log(steps)))) |
| `regret_per_step` | GP-predicted reward |
| `n_step_raw` | Copy of raw n_step (for comparison display) |
| `regret_per_step_raw` | Copy of raw reward (for comparison display) |
| `power_max` | Copy of raw power_max |

**No SE fields** — GP smooths away individual error metrics.

### 1c. `gpGrid` (from `gp_grid_json`)
- **Source:** `fit_gp_curves()` → GP predictions on a fine 200-point grid per algorithm
- **Only generated between min and max powered param values**
- **Fields per row:** `algo_name`, `algo_param`, `n_step`, `regret_per_step`
- **No power_max, no SE** — these are interpolated points that don't correspond to actual simulations

---

## 2. GP Fitting Process (`fit_gp_curves` in recommendation.py)

For each algorithm:

1. **Filter** to powered rows: `algo_df[power_max >= power_constraint]`
2. **If < 2 powered rows:** skip GP, copy raw data as-is into `gp_at_raw` (no `gp_grid`)
3. **Fit two GPs** (Matern(2.5) + WhiteKernel, 3 restarts):
   - Steps GP: `X = param`, `y = log(n_step)` → predict → `exp()` to get steps
   - Reward GP: `X = param`, `y = regret_per_step` → predict directly
4. **Generate outputs:**
   - `gp_grid`: 200 evenly spaced points between `min(powered_param)` and `max(powered_param)`
   - `gp_at_raw`: predictions at ALL raw param values (including unpowered ones — the GP extrapolates)

### Known Issue (2026-02-26)
With few powered points (e.g., 3), the WhiteKernel can absorb all signal as noise. The GP then predicts a near-constant value (the prior mean), causing wildly wrong step predictions (e.g., 2 instead of 1000). The reward GP is less affected because reward values are naturally less variable.

---

## 3. Frontend Data Structures

After the 3 JSON datasets are loaded, the JS builds these derived structures:

| Variable | Built from | Description |
|----------|-----------|-------------|
| `poweredData` | `rawData.filter(achievedPower)` | Raw rows that achieved power |
| `algorithms` | `poweredData` grouped by `algo_name` | Dict: algo_name → [rows] |
| `algoNames` | `Object.keys(algorithms)` | List of algo names |
| `gpGridByAlgo` | `gpGrid` grouped by `algo_name` | Dict: algo_name → [fine-grid points] |
| `gpAtRawByAlgo` | `gpAtRaw` grouped by `algo_name` | Dict: algo_name → [GP-at-raw points] |
| `baseline` | Recomputed per `w` | Dict: w → best objective score across all sources |

---

## 4. Tables on the Page

### 4a. Power Warning Table (always visible if any config failed)
- **Data:** `rawData` rows where `power_max < POWER_CONSTRAINT`
- **Columns:** Algorithm, Parameter, Power Achieved, Required, Shortfall
- **Built once** on page load by `buildPowerWarnings()`

### 4b. Summary Table (dynamic — updates when slider moves)
- **Built by:** `updateForW(w)` → `renderRow()`
- **Row order:**
  1. Pure-TS (param=0) — from `getRefRow(0.0)` (gpAtRaw first, raw fallback)
  2. Pure-UR (param=1) — from `getRefRow(1.0)`
  3. Best algorithm for current w (highlighted blue with star)
  4. Remaining algorithms sorted by objective score descending
- **Columns:** Algorithm, Param, Reward, ±Rew, Steps, ±Steps, Score(w=X)
- **Data source for values:** GP-fitted data (n_step, regret_per_step come from `bestForW()` which returns GP data)
- **Data source for SE:** `findRawSE()` which looks up `rawData` by (algo_name, algo_param)
  - `rewardSE` = `raw.reward_se`
  - `stepsSE` = `round(sqrt(se_steps_h1² + se_steps_h0²) + bias_steps_h0)`
- **Power check:** Looks up `rawData` for the matching row to check `achievedPower(rawMatch)`

### 4c. Full Data Table (collapsible, under "Full Simulation Results")
- **Data:** All `rawData` rows, sorted by algo then param
- **Built once** on page load by `buildFullTable()`
- **Columns:** Algorithm, Parameter, Reward, Steps, H1 Var, H0 Var, WC Bias, Power Max, Status
- **Steps = "NA"** if unpowered; SE columns blank if unpowered

### 4d. GP-Fitted Results Table (collapsible, under "GP-Fitted Results")
- **Data:** All `gpAtRaw` rows, sorted by algo then param
- **Built once** on page load by `buildGPTable()`
- **Columns:** Algorithm, Parameter, Reward(GP), Steps(GP), Reward(Raw), Steps(Raw), Power
- **No SE columns** — this table shows GP vs raw comparison

---

## 5. Chart

### Type
Line chart (Chart.js), X-axis = w, Y-axis = Relative ECP-Reward

### Datasets (lines on chart)
- **Fixed datasets (built once, refreshed on GP toggle):**
  - Pure-TS (param=0): green dashed line
  - Pure-UR (param=1): red dashed line
  - Data source: `getRefRow()` → gpAtRaw, then raw fallback
- **Dynamic datasets (updated every slider move):**
  - One blue bold line per algorithm: the optimal (algo, param) at current w
  - Data source: `bestForW()` → gpGrid (if interpolation on) or gpAtRawByAlgo (if off), raw fallback

### Baseline Computation (`recomputeBaseline`)
For each w value in `[W_MIN, W_MAX]` (21 points, 0 to 0.06):
```
baseline[w] = max objective across:
  1. gpGrid (if interpolation on)
  2. gpAtRaw (always — since GP always on)
  3. poweredData (raw, as safety net)
```

### Relative Curve
For a row: `y = computeObjective(row.n_step, row.regret_per_step, w) - baseline[w]`
- 0 = optimal at this w
- Negative = suboptimal

### Objective Function
```javascript
function computeObjective(nStep, reward, w) {
    if (nStep <= 1) return nStep;
    return reward - w * Math.log(nStep);
}
```

### Vertical Line Plugin
Dashed black vertical line at current w, with rotated label.

---

## 6. Slider and Controls

### w Slider
- Range: `W_MIN=0` to `W_MAX=0.06`, 20 discrete steps
- Moving the slider calls `updateForW(w)` which:
  1. Finds best (algo, param) per algorithm at this w
  2. Updates dynamic chart lines
  3. Updates recommendation card (best algo, param, steps)
  4. Rebuilds summary table with new optimal rows

### Interpolation Toggle ("Interpolate")
- **Checked (on):** `bestForW()` searches `gpGrid` (200-point fine grid) — can find optimal params between simulated values
- **Unchecked (off):** `bestForW()` searches `gpAtRawByAlgo` — only considers GP-fitted values at actually simulated params
- Toggling calls `recomputeBaseline()` + `refreshFixedCurves()` + `updateForW(currentW)`
- **GP is always on** in both modes — the toggle only controls search resolution, not whether GP is used

### Y-axis Min Toggle
Controls whether chart Y-axis starts at 0 or auto-scales.

---

## 7. Data Flow Diagram

```
sim_wrapper.py::get_objective_score()
  → returns dict per (algo, param) config
  → fields: n_step, regret_per_step, power_max, se_*, bias_*, reward_se, ...

recommendation.py::get_recommendation()
  → collects all configs into DataFrame (df)
  → calls fit_gp_curves(df, power_constraint)
      → filters to powered rows
      → fits 2 GPs per algo (log-steps, reward)
      → produces gp_grid (200 fine points) + gp_at_raw (at raw params)
  → builds results_summary:
      chart_data_json  = df[columns].to_json()   → rawData
      gp_grid_json     = json.dumps(gp_grid)      → gpGrid
      gp_at_raw_json   = json.dumps(gp_at_raw)    → gpAtRaw

app.py::/recommend route
  → passes results_summary to template

recommend.html:
  → Parses 3 JSON datasets
  → Builds 4 tables:
      Power Warnings (rawData, filtered to unpowered)
      Summary Table  (GP-fitted values + raw SE, dynamic with slider)
      Full Data      (rawData, all rows)
      GP-Fitted      (gpAtRaw, GP vs raw comparison)
  → Builds chart:
      Fixed lines: Pure-TS, Pure-UR (from gpAtRaw)
      Dynamic lines: per-algo optimal (from gpGrid or gpAtRawByAlgo)
      Baseline: best across all sources per w
      Y-axis: relative ECP-reward = objective - baseline
```
