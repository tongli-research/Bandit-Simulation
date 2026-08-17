# Bandit Simulation — Spec File

> Shared contract between Tong and the Engineer agent. Read at session start.

## Code Structure

### Core Library (`bandit_simulation/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `sim_wrapper.py` | Simulation engine | `run_simulation()`, `run_task_common()`, `sweep_and_run()`, `SimRunState`, `SimResult`, `get_objective_score()`, `_gp_smooth_cores()` |
| `simulation_configurator.py` | Config & scheduling | `SimulationConfig` (must call `.manual_init()`), `ArrDim` |
| `test_procedure_configurator.py` | Hypothesis tests | `TestProcedure` (ABC), `ANOVA`, `TControl`, `TConstant`, `Tukey` |
| `bandit_algorithm.py` | Bandit algorithms | `EpsTS`, `TSProbClip`, `TSPostDiff`, `TSPostDiffTopWithResample` (archived), and linear variants |
| `bayes_vector_ops.py` | Posterior models | `BetaBernoulli`, `NormalFull`, `LinearNormalKnownVar` |
| `adaptive_power_search.py` | Incremental power search | `run_adaptive_power_search()`, `H0Manager`, `_refine_h0_and_check_power()` |
| `config_loader.py` | YAML config parser | `load_config()`, `build_sim_config()` |
| `analysis.py` | Result analysis | `compute_objective()`, `select_curves_relative()`, `format_linear_summary()` |

### Web App (`webapp/`)

| File | Purpose |
|------|---------|
| `app.py` | Flask routes, progress tracking, form parsing |
| `recommendation.py` | `get_recommendation()` (standard), `get_recommendation_adaptive()` (adaptive), GP curve fitting |

### Data Flow (standard path)

```
User Input → SimulationConfig.manual_init()
  → run_simulation(H1) → SimResult
  → get_h0_cores_and_weights() → weight matrix + H0 locations
  → run_simulation(H0) → SimResult
  → get_adjusted_crit_region(weight, H0) → (crit_boundary, se_adjusted, core_crit_array)
  → get_objective_score(crit_boundary, h1_res, sim_config,
        se_crit_adjusted, core_crit_array, weight, h0_locations)
      → power curve + error decomposition (H1 var, H0 var, WC bias)
      → T_opt, reward, objective, se_steps_h1, se_steps_h0, bias_steps_h0
```

**Key:** `get_adjusted_crit_region()` returns a **3-tuple** `(crit_boundary, se_adjusted, core_crit_array)`. All callers must unpack all three values.

### Data Flow (adaptive path)

```
User Input → SimulationConfig.manual_init()
  → Phase 1: init_simulation() (burn-in)
  → Phase 2: run_one_batch_step() loop to T_UR (no power check)
  → Phase 3: at each step:
      → _refine_h0_and_check_power():
          Part A: midpoint splitting until power_error < threshold
          Part B: weight matrix + tensordot + tp.compute_power()
      → if power >= target: _find_crossing_step() → T_phi
```

### Array Shape Conventions

| Array | Shape | Notes |
|-------|-------|-------|
| `action_hist` | `(n_rep, T_batch, n_arm)` | T_batch = len(step_schedule) |
| `reward_hist` | `(n_rep, T_batch, n_arm)` | |
| `arm_means` | `(n_rep, T_batch, n_arm)` | Cumulative mean per arm |
| `combined_means` | `(n_rep, T_batch, 1)` | Pooled across arms |
| Test stat (ANOVA) | `(n_rep, T_batch)` | p-values |
| Test stat (TControl) | `(n_rep, T_batch, K-1)` | vs control arm 0 |
| Test stat (Tukey) | `(n_rep, T_batch, K, K)` | Pairwise |
| `weight` | `(n_rep, n_cores)` | H0 interpolation weights |
| `crit_boundary` | `(n_rep, T_batch, ...)` | Interpolated critical values |
| `se_crit_adjusted` | `(n_rep, T_batch, ...)` | SE of crit, propagated through weights: `sqrt(w² · se²)` |
| `core_crit_array` | `(n_cores, T_batch, ...)` | Per-core raw critical values (before interpolation) |
| `power` | `(T_batch,)` | One value per horizon step |

## Error Decomposition (all expressed as steps)

Three error metrics are computed in `get_objective_score()` and displayed in the web UI.

### 1. H1 Variance → `se_steps_h1`
- **Source:** Finite H1 reps (M). Power is a sample mean of {0,1}.
- **Calculation:** `SE_power = sqrt(p̂(1-p̂) / M_eff)` where M_eff = non-NaN reps.
- **Translation:** Find where power curve crosses `power_target - SE_power`, subtract from n_step.
- **Scaling:** Decreases as ~1/sqrt(n_rep).

### 2. H0 Variance → `se_steps_h0`
- **Source:** Finite H0 reps per bin. Critical boundary is a quantile estimate with SE.
- **Calculation:** `_quantile_with_se()` computes SE per core. Propagated through weight matrix: `se_adjusted = sqrt(w² · se²)`.
- **Translation:** Compute power with `crit ± se_adjusted`, find n_step for each, half-range = SE_steps.
- **Scaling:** Decreases as ~1/sqrt(reps_per_core).

### 3. H0 Worst-Case Bias → `bias_steps_h0`
- **Source:** Finite H0 grid points (B). Interpolation between grid points introduces bias.
- **Calculation:** For each bin, compute power using only left endpoint crit vs only right endpoint crit. Worst deviation from interpolated power across all bins (weighted by rep count).
- **GP smoothing:** Before bias calculation, per-core crit values are GP-smoothed across H0 locations (`_gp_smooth_cores()`, Matern(2.5) + WhiteKernel) to isolate systematic trend from H0 sampling noise. Only when n_cores >= 3.
- **Translation:** `n_step(power_target - worst_bias) - n_step(power_target)`.
- **Scaling:** Decreases with more H0 grid points (more cores).
- **Caveat:** This is worst-case; actual bias is usually much smaller.

### Combined error (displayed in Summary table)
```
±Steps = round(sqrt(H1_Var² + H0_Var²) + WC_Bias)
```

### Key design decisions
- All metrics are in **steps** (not power or probability) for user interpretability.
- GP smoothing uses `warnings.catch_warnings()` to suppress expected ConvergenceWarnings from sklearn.
- Adaptive path (`get_recommendation_adaptive`) sets all error fields to 0 (not computed).

## Invariants

> Things that MUST always be true. Never violate without Tong's explicit approval.

1. **Standard and adaptive paths produce matching power estimates** at the same horizon (within simulation noise). Both must use `tp.compute_power()` for the final power calculation.

2. **`SimulationConfig.manual_init()` must be called** before any simulation runs. Forgetting this breaks step schedules and Bayesian model setup.

3. **H0 null distributions are location-dependent.** You cannot replace multi-location H0 simulation with a single null or analytical formula.

4. **`compute_power()` handles test-specific logic** — two-sided `np.abs()`, min_effect NaN-masking, per-comparison `nanmean` aggregation. All power calculations must go through this method, not hand-rolled comparisons.

5. **Critical regions are interpolated, not binned** (default `n_crit_approx_method='linear'`). Weight matrix + tensordot pattern must be used for combining H0 locations.

6. **`get_adjusted_crit_region()` returns a 3-tuple:** `(crit_boundary, se_adjusted, core_crit_array)`. All callers must unpack all three. The SE is propagated through the weight matrix via `sqrt(w² · se²)` tensordot.

## Locked Behaviors

> Features confirmed working. Cannot be changed without asking Tong first.

| Date | Behavior | Verified By |
|------|----------|-------------|
| 2026-02-24 | Adaptive power search produces same power as standard path for ANOVA, TControl one-sided, TControl two-sided (tested at n_rep=5000, diff < 0.006) | `tests/test_adaptive_vs_standard.py` |
| 2026-02-24 | Incremental simulation (`init_simulation` + `run_one_batch_step` loop) produces identical results to `run_simulation()` | `tests/test_incremental_design.py` |
| 2026-02-24 | Web app progress timer uses running average of completed configs (not locked to first config time) | `webapp/app.py:195-200` |

## Output Page Design (`webapp/templates/recommend.html`)

> Canonical reference for the results page layout. Any change to the output page must preserve
> elements listed here unless Tong explicitly approves removal.

### Page-Level Layout

- **Container:** `max-width: 1500px`, centered. Background `#f4f7f8`.
- **Stylesheet:** `webapp/static/styles/style2.css`
- **Chart library:** Chart.js v4 (CDN)
- **All sections stack vertically**, full width within container.

### 1. Header (`<header class="results-header">`) — centered

| Element | Tag/Class | Alignment | Style |
|---------|-----------|-----------|-------|
| Back link | `<a class="back-link">` "← Run New Analysis" | center (inline-block) | blue `#007BFF`, no underline, underline on hover |
| Page title | `<h1>` "Algorithm Performance Explorer" | center | color `#2c3e50`, margin 8px top |
| Subtitle | `<p class="subtitle">` | center | gray `#666`, 1.05em |
| Elapsed time | `<p class="elapsed-time">` "Simulation completed in Xs" | center | green `#28a745`, 0.9em, bold 500. Conditional (only shown if `elapsed_time` is set). |

### 2. Save Scenario Bar (`<div class="save-bar">`) — centered row

| Element | Style | Notes |
|---------|-------|-------|
| Text input `#scenarioName` | 220px wide, rounded 6px | Placeholder: "Name this scenario (optional)" |
| Save button | blue `btn-primary`, tooltip on hover | Tooltip explains "Save...reload instantly from home page" |
| Status span `#saveStatus` | green `#28a745` (or red `.error`) 0.85em | Shows save result |

### 3. Power Warning Section (`#powerWarningSection`) — conditional

- **Default:** `display: none`. Shown by JS if any algorithm param failed power.
- Collapsible `<details open>` with **red left border** (4px `#dc3545`).
- **Summary:** "Power Constraint Warning"
- **Content:** explanation paragraph + table.
- **Table columns:** Algorithm | Parameter | Power Achieved | Power Required | Shortfall

### 4. Chart Section (`<section class="chart-section">`) — white card

This is the main interactive area. White bg, rounded 12px, shadow, padding 24px.

#### 4a. Chart + Summary Table (`<div class="chart-and-table">`) — side-by-side flex

| Sub-element | Width | Notes |
|-------------|-------|-------|
| **Chart canvas** (`#performanceChart`) inside `.chart-wrapper` | flex `68%` | Aspect ratio 1.7. Responsive width 100%. |
| **Summary table** (`.summary-table-wrapper`) | flex `1`, min-width 220px | Stacks below chart on screens < 800px |

**Chart specifications:**
- **Title:** "Relative ECP-Reward (Experiment-Cost-Penalized) vs Per-'w' Optimal" (bold 16px)
- **X-axis:** "Experiment Extension Cost ('w')" (font 13px)
- **Y-axis:** "Relative ECP-Reward (Experiment-Cost-Penalized)" (font 13px). Auto-scaled by Chart.js, or user-controlled via Y Range controls (4b2).
- **Grid:** very light `rgba(0,0,0,0.05)`
- **Datasets:**
  - Pure-TS: green `#28a745`, dashed `[6,4]`, width 2.5, point radius 4
  - Pure-UR: red `#dc3545`, dashed `[6,4]`, width 2.5, point radius 4
  - Optimal (dynamic): blue `#007BFF`, solid, width 3.5, point radius 5
- **Vertical line plugin:** dashed black line at current `w`, with rotated annotation
- **Legend:** bottom, point-style icons, padding 12, font 12px
- **Tooltips:** mode "index", non-intersecting. Shows `w`, label, relECP, reward, steps.

**Summary table:**
- **Title line 1:** "Summary at w = [value]" (0.95em, `#2c3e50`, centered)
- **Title line 2:** "Best: [AlgoName] (ε = [value])" (`#tableBest`, blue `#007BFF`, 0.88em, bold 600). Updates dynamically.
- **Columns:** Algorithm | Reward | Steps | ±Steps
- **Rows:** Pure-TS (gray italic `.ref-row`), Pure-UR (gray italic `.ref-row`), then per-algorithm optimal (`.best-row` = blue bold for winner). Power-failed refs get `[!]` marker.
- **±Steps formula:** `round(sqrt(H1_Var² + H0_Var²) + WC_Bias)` — tooltip on header explains this.
- Updates dynamically when slider moves.

#### 4b. W Range Controls (`#wRangeBar`) — aligned with chart plot edges

| Element | Position | Style |
|---------|----------|-------|
| "Min w:" + `<input type="number" #wMin>` | left | 72px, step 0.01 |
| "Update Range" button | center | secondary btn |
| "Max w:" + `<input type="number" #wMax>` | right | 72px, step 0.01 |

Padding-left/right set dynamically by JS `alignSlider()` to match chart plot area edges.

#### 4a2. Y-Axis Min Control (`#yMinRow`) — above W range bar, slightly left

Horizontal row above the W range bar (4b). Aligned by JS with a 20px left offset from the chart plot area edge (closer to the Y-axis than the W controls).

| Element | Style |
|---------|-------|
| "Y min:" label + `<input type="number" #yMin>` | 68px input, 0.82em gray `#555` |
| "Set" button | small secondary btn (0.8em) |

- Max Y is always 0 (hardcoded — 0 = optimal, all relative values are ≤ 0).
- Empty input = Chart.js auto-scale.
- Visually grouped with 4b (W range) controls but offset slightly left toward Y-axis.

#### 4c. Slider Strip (`<div class="slider-strip">`)

All sub-elements horizontally aligned to chart plot area via JS.

| Element | Details |
|---------|---------|
| **Slider** (`#wSlider`) | `<input type="range">`. Min/max/step set dynamically. Custom thumb: 22px circle, white + blue border. Track: 6px height, gray `#ddd`. |
| **Labels row** (`.slider-labels`) | Left: "← More Reward". Center: **Check Setting** button (small secondary) + status span + **GP Smoothing** checkbox toggle. Right: "Fewer Steps →" |
| **Readout** (`.slider-readout`) | "Adding one extra step...costs (**w = [inline input]**) cumulative reward." Inline input: 5em, bottom-border only (blue `#007BFF`), editable — Enter/blur applies value. |

#### 4d. Slider Explanation (`#sliderExplanation`)

Gray text (0.85em, `#666`), left AND right aligned with chart plot area (same as 4b/4c). Content: "Choosing w: A good starting point is w ≈ 5-10% of max achievable reward..."

#### 4e. Chart Help (collapsible `<details class="chart-help">`)

- Transparent bg, no shadow. **Open by default** (`<details open>`).
- **Summary:** "How to read this chart" (0.95em, gray `#555`)
- **Content:** bullet list explaining Y-axis (with ECP = Experiment-Cost-Penalized Reward expansion + formula), X-axis, dashed lines, bold lines, slider.

### 5. Recommendation Card (`<section class="recommendation-card">`)

White bg, rounded 12px, blue left border (4px `#007BFF`), shadow.

| Element | Style |
|---------|-------|
| Header: "Recommended Configuration" + badge "for w = [value]" | flex row, space-between. Badge: gray bg `#e9ecef`, rounded 20px pill. |
| Algorithm name `#rec-algo` | centered, 1.5em bold |
| "parameter = [value]" `#rec-param` | centered, 1.05em gray |
| Metrics row | centered flex, gap 40px. Two items: **Expected Steps** (blue `#007BFF` 1.4em bold + gray label) and **Statistical Test** (same style, static value from server). |

### 6. Your Input Parameters (collapsible `<details>`)

- **Summary:** "Your Input Parameters"
- **Content:** CSS grid, auto-fill columns min 200px. Each item: label (gray 0.85em bold 500) + value.

### 7. Technical Details (collapsible `<details>`)

- **Summary:** "Technical Details"
- **Subsections** (h4 headings):
  1. "What the Parameters Mean" — EpsTS definition (currently only EpsTS; TSProbClip and TSPostDiff are commented out)
  2. "What is w?" — explanation paragraph
  3. "What is ECP-Reward?" — multi-paragraph explanation
  4. "Interpreting Results" — paragraph + analysis summary (n_algorithms, n_parameter_values)
  5. "Error Decomposition" — definition list: H1 Var, H0 Var, WC Bias
  6. "Caveats" — ordered list: Jensen's inequality bias, within-bin monotonicity assumption

### 8. Full Simulation Results (collapsible `<details>`)

- **Summary:** "Full Simulation Results"
- **Table** (`#fullDataTable`): scrollable max-height 500px, sticky header.
- **Columns:** Algorithm | Parameter | Reward | Steps | H1 Var (steps) | H0 Var (steps) | WC Bias (steps) | Power Max | Status
- Failed-power rows highlighted red (`.power-fail-row`).
- Sorted by algo name then parameter.

### 9. GP-Fitted Results (collapsible `<details>`)

- **Summary:** "GP-Fitted Results (Raw vs Smoothed)"
- **Table** (`#gpDataTable`): same styling as full data table.
- **Columns:** Algorithm | Parameter | Reward (GP) | Steps (GP) | Reward (Raw) | Steps (Raw) | Power
- Sorted by algo name then parameter.

### 10. Action Buttons (`<div class="action-buttons">`) — centered

| Button | Style |
|--------|-------|
| "Run New Analysis" | blue `btn-primary`, links to `/` |
| "Print / Save Results" | gray `btn-secondary`, calls `window.print()` |

Print media query hides: action buttons, back link, slider strip. Chart section loses shadow.

### JS Behavior Summary

| Feature | How it works |
|---------|-------------|
| **W auto-range** | On load, detects max reward from data (or user-supplied `max_reward`), sets wMax = 10% of max reward. |
| **Slider ↔ chart** | `updateForW(w)`: recomputes optimal row per algo at given w (GP or discrete), updates dynamic datasets, vertical line, summary table, recommendation card. |
| **GP Smoothing toggle** | `#interpToggle` checkbox. Checked = use GP grid for optimal param search. Unchecked = discrete simulated points only. |
| **Check Setting** | POSTs to `/check_setting` with sim_params + best algo/param at current w. Response adds new data point to chart via `addDataPoint()`. |
| **Save Scenario** | POSTs to `/save_scenario` with all data (inputs, chart_data, GP data, sim_params). |
| **Align slider** | `alignSlider()` reads `chart.chartArea` pixel coords, sets padding-left AND padding-right on slider row, labels, w-range bar, readout, AND explanation to match chart plot area edges. All 4b/4c/4d elements share identical alignment. Runs on load + resize. |
| **W precision** | `setWPrecision(wMin, wMax)`: digits = ceil(-log10(increment)). All w display uses `formatW()`. |
| **W inline input** | Editable text field in readout. Enter/blur → clamp to slider range → apply. |

## Current Focus

- Error decomposition implemented and tested (H1 var, H0 var, WC bias — all in steps)
- GP smoothing of per-core crit values for cleaner bias estimation
- Live results table during simulation (rows appear as configs complete)
- H0 config inputs (grid points + reps per location) with tooltips
- Adaptive mode hidden from UI (code retained)

## Pending UI Adjustments (from Tong's testing 2026-02-24)

### 1. Simulation Setup field layout
- **n_arm, horizon** on one row (they are the "experiment design" pair)
- **n_rep (H1)** on its own row (it's the "simulation accuracy" control)
- **H0 Grid Points, H0 Reps** on their own row(s) below n_rep
- Visually group related fields so the user sees the relationship (e.g., n_rep drives H0 defaults)

### 2. H0 defaults: show computed values, warn if too low
- Don't show complex formulas in the hint text. Just say "default" that updates with H1 reps.
- **Placeholder** should show the actual computed default (e.g., `placeholder="3"` not `"Auto"`), updated live when H1 reps changes.
- **Hint text** shows "Recommended minimum: X" (also live-updated).
- **Do NOT overwrite** user's manual input when H1 reps changes (too annoying).
- **Orange warning** if user enters a value below the recommended minimum, with note: "Not recommended below default — may unbalance H1/H0 standard errors."

### 3. Max Reward Per Step hint text
- Remove "Leave blank to auto-detect." (there is no auto-detect)
- Replace with: "If known, helps auto-set the w range on the results page. Doesn't affect simulation results."

### 4. Live results table during simulation (loading overlay)

#### 4a. Power status column
- Current "Power" column label is misleading — it shows a number but doesn't convey pass/fail.
- **Rename to "Max Power Attained (at horizon = X)"** where X = user's maximum horizon input.
- **Failed rows**: highlight red background so user immediately sees which configs didn't reach the power constraint.

#### 4b. Hierarchical table with error decomposition
Replace the current flat columns with a hierarchical (multi-row header) table:

```
| Algorithm | ε | Reward       | Steps                                                                    | Power  |
|           |   | Est. | SE   | Est. | SE    | H1 % | H0 % | Worst-Case Bias |        |
|-----------|---|------|------|------|-------|------|------|-----------------|--------|
| EpsTS     |0.5| 0.82 |±0.03 | 1200 | ±85   |  48% |  52% | ≤57             | 0.81   |
```

- **Reward**: Est. (point estimate) + SE (reward_sd)
- **Steps Est.**: n_step (point estimate)
- **Steps SE**: Total SE = `sqrt(H1_SE² + H0_SE²)` — combined sampling uncertainty, shown first
- **H1 %**: `round(H1_SE² / Total_SE² × 100)%` — proportion of variance from H1 sampling
- **H0 %**: `round(H0_SE² / Total_SE² × 100)%` — proportion of variance from H0 sampling
- **Worst-Case Bias**: `bias_steps_h0`, shown with ≤ prefix (upper bound). Full spelling, no abbreviation.
- Rationale: showing H1/H0 as percentages avoids the misconception that H1_SE + H0_SE = Total_SE (they combine as root-sum-of-squares, not additive)
- Use `<thead>` with two `<tr>` rows + `colspan` for hierarchical headers

### 5. Cancel simulation button
- Add a "Stop Simulation" button visible during the loading overlay (next to or below the progress bar).
- Backend: need a mechanism to signal the simulation thread to stop (e.g., a global `cancel_requested` flag checked between each (algo, param) config).
- Frontend: POST to `/cancel` endpoint, which sets the flag. Progress polling detects cancellation and closes the overlay.
- After cancel: show partial results if any configs completed, or return to form if none completed.
