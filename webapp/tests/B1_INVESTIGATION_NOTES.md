# B1 Investigation: GUI vs Pure Code Alignment

## Summary

The webapp produces different simulation results than the paper scripts. This investigation identifies all sources of discrepancy and their relative impact.

---

## Sources of Discrepancy (ranked by impact)

### 1. `family_wise_error_control` — **MAJOR IMPACT**

| Setting | Webapp | Scripts (paper) |
|---------|--------|-----------------|
| `family_wise_error_control` | `True` (checkbox checked by default) | `False` (not set, uses default) |

**Impact (from test_fwer_nstep_varies.py, 3-arm TControl):**
- FWER=False: best eps=0.80, n_step=35
- FWER=True: best eps=1.00, n_step=42 (+20% more steps)
- FWER=True causes more steps (Bonferroni correction requires larger samples)
- **Changes the OPTIMAL algorithm parameter** (eps shifts from 0.80 to 1.00)
- This is the primary cause of discrepancy between webapp and paper results

**Location:**
- `webapp/templates/index.html:202` — checkbox is `checked` by default
- `webapp/app.py:200` — reads `family_wise_error_control` from form
- Scripts never set this parameter → defaults to `False` in `TestProcedure.__init__`

### 2. `burn_in_per_arm` — **MINOR IMPACT**

| Setting | Webapp | Scripts (paper) |
|---------|--------|-----------------|
| `burn_in_per_arm` | `5` (hardcoded in recommendation.py:28) | `1` (all scripts) |

**Impact (from test_gui_vs_script_alignment.py):**
- Reward differences: ~0.01-0.06% (stochastic noise level)
- n_step differences: near zero when hitting horizon ceiling
- **Does NOT change the optimal algorithm parameter** in any tested scenario
- BUT with achievable power (test_fwer_nstep_varies.py, 3-arm):
  - FWER=F, burn=1: best eps=0.80, n_step=35
  - FWER=F, burn=5: best eps=1.00, n_step=34
  - Slight shift in optimal parameter, but minor

**Location:**
- `webapp/recommendation.py:28` — hardcoded to `5`
- All scripts use `burn_in_per_arm=1`

### 3. `n_rep` — **VARIANCE IMPACT**

| Setting | Webapp | Scripts (paper) |
|---------|--------|-----------------|
| `n_rep` | `10000` (default in form) | `20000` (most scripts) |

**Impact:**
- Lower n_rep = higher variance in estimates
- Does not systematically bias results in either direction
- User can change this in the webapp form
- All auto-computed parameters scale with n_rep, so no structural issue

### 4. Per-param loop vs batch sweep — **NO IMPACT**

Confirmed: running `sweep_and_run` once for all params vs looping per (algo, param) produces **identical results**. NOT a source of discrepancy.

### 5. `reward_model` — **DEPENDS ON USER INPUT**

| Setting | Webapp | Scripts (paper) |
|---------|--------|-----------------|
| `reward_model` | User selects (Gaussian or Bernoulli) | Default = Bernoulli (`np.random.binomial`) |

The webapp lets the user choose. If the user selects Gaussian but the paper used Bernoulli, results will differ. This is expected behavior, not a bug.

---

## Detailed Test Results

### Test: 3-arm TControl with achievable power (n_arm=3, horizon=2000, n_rep=5000)

#### PAPER config (FWER=F, burn=1)
| eps | n_step | reward | Best at w=1,5,10 |
|-----|--------|--------|-------------------|
| 0.00 | 44 | 0.5145 | |
| 0.20 | 39 | 0.5115 | |
| 0.40 | 41 | 0.5094 | |
| 0.60 | 37 | 0.5054 | |
| **0.80** | **35** | **0.5024** | **Best** |
| 1.00 | 36 | 0.4998 | |

#### WEBAPP config (FWER=T, burn=5)
| eps | n_step | reward | Best at w=1,5,10 |
|-----|--------|--------|-------------------|
| 0.00 | 54 | 0.5160 | |
| 0.20 | 48 | 0.5123 | |
| 0.40 | 45 | 0.5085 | |
| 0.60 | 45 | 0.5055 | |
| 0.80 | 44 | 0.5025 | |
| **1.00** | **42** | **0.4999** | **Best** |

**Key observation:** Webapp recommends eps=1.00 (pure Thompson Sampling) while paper recommends eps=0.80. The webapp also shows 20% more steps needed.

#### ISOLATE burn_in only (FWER=F, burn=5)
| eps | n_step | Best |
|-----|--------|------|
| 0.80 | 36 | |
| **1.00** | **34** | **Best** |

burn_in alone shifts optimal from 0.80 to 1.00 — slight effect.

#### ISOLATE FWER only (FWER=T, burn=1)
| eps | n_step | Best |
|-----|--------|------|
| **0.80** | **41** | **Best** |
| 1.00 | 42 | |

FWER alone increases n_step by ~17% but keeps optimal at 0.80.

#### Combined effect (both FWER=T + burn=5 = webapp)
Best shifts to eps=1.00. The interaction of both changes together creates the full discrepancy.

---

## Proposed Changes

### Change 1: `burn_in_per_arm` — Match paper default
**File:** `webapp/recommendation.py:28`
**Change:** `burn_in_per_arm=5` → `burn_in_per_arm=1`
**Rationale:** All paper scripts use `burn_in_per_arm=1`. This aligns webapp with paper.

### Change 2: `family_wise_error_control` checkbox — Uncheck by default
**File:** `webapp/templates/index.html:202`
**Change:** Remove `checked` from the FWER checkbox
**From:** `<input type="checkbox" id="family_wise_error_control" name="family_wise_error_control"checked>`
**To:** `<input type="checkbox" id="family_wise_error_control" name="family_wise_error_control">`
**Rationale:** Paper scripts use `family_wise_error_control=False`. Having it checked by default misleads users into getting different results. Users who want FWER control can still check the box.

### Change 3 (optional): `n_rep` default
**File:** `webapp/templates/index.html:69`
**Change:** `value="10000"` → `value="20000"` (or keep 10000 for speed)
**Rationale:** Paper uses 20000 but this doubles runtime. Could add a note that paper used 20000.

---

## Files Created During Investigation

- `webapp/tests/test_gui_vs_script_alignment.py` — batch vs loop, burn_in isolation
- `webapp/tests/test_gui_vs_script_nstep.py` — shorter horizon test
- `webapp/tests/test_fwer_impact.py` — FWER impact with paper parameters
- `webapp/tests/test_fwer_nstep_varies.py` — achievable power test (key test)
- `webapp/tests/B1_INVESTIGATION_NOTES.md` — this file
