# Engine Audit Findings

## Summary
- **Audited Commit**: 897b9da8a460ca5874eb232985128e2c239fc53e
- **Branch**: audit/engine-forensic
- **Files Audited**: engine.py (6591 lines), app.py (2517 lines), demo_data.py, tests/test_app.py

---

## E-001 — Duplicate ScenarioResult Dataclass

**Severity**: High  
**Status**: Confirmed  
**Location**: engine.py:242 (v1, dead) and engine.py:1738 (v2, active)  
**Evidence**: Two `@dataclass` definitions with same name `ScenarioResult`. v1 has fields `df, price_change, nd_change, summary`. v2 has 15 fields including per-draw arrays. Only v2 is referenced (lines 1890, 1910, 2057, 2185, 2313, 2324).  
**Impact**: Maintenance risk, confusion, potential runtime errors if wrong class imported.  
**Recommendation**: **Remove v1 (line 242-246)**. It's dead code.

---

## E-002 — Duplicate get_posterior_predictive_check Functions

**Severity**: High  
**Status**: Confirmed  
**Location**: engine.py:2874 (v1) and engine.py:4264 (v2)  
**Evidence**: Two functions with identical name but different signatures. v1: `get_posterior_predictive_check(model, idata, prepared_df, random_seed)`. v2: `get_posterior_predictive_check(trace, df, max_points)`. Both for legacy model.  
**Impact**: Which one is called? Name collision. v2 has broad exception handler.  
**Recommendation**: **Consolidate into one**. Determine which signature is needed by callers. Check app.py usage.

---

## E-003 — Broad Exception Handlers (4 instances)

**Severity**: Critical  
**Status**: Confirmed  
**Location**: 
1. engine.py:260-261 in `_posterior_group()`
2. engine.py:3125 in `summarize_convergence_diagnostics()`
3. engine.py:3137 in `summarize_convergence_diagnostics()`
4. engine.py:4285 in `get_posterior_predictive_check` (v2)  
**Evidence**: All use `except Exception:` or `except Exception: pass`  
**Impact**: Silent failures, hidden bugs, incorrect diagnostics, wrong scenario results.  
**Recommendation**: **Replace all with specific exception types** (KeyError, ValueError, AttributeError, IndexError) or raise informative domain exceptions.

---

## E-004 — Incomplete run_rolling_backtest Implementation

**Severity**: Critical  
**Status**: Confirmed  
**Location**: engine.py:3327-3380  
**Evidence**: Function has `results = []` that is never populated. Contains comment "Placeholder for full implementation". Returns empty DataFrame. No tests call this function.  
**Impact**: False appearance of validation capability. Users may think backtesting works.  
**Recommendation**: **Raise explicit NotImplementedError** with message directing to roadmap, OR implement fully with tests, OR remove from production code.

---

## E-005 — Cross-File create_scenario_action_table Schema Inconsistency

**Severity**: High  
**Status**: Confirmed  
**Location**: engine.py:6062 vs app.py:1114  
**Evidence**: Two functions with same name, different schemas:
- engine.py: returns `apply, retailer, category, brand, sku, pack_size, baseline_price_std, baseline_nd, new_price_std, new_nd, price_change, nd_change`
- app.py: returns `month, retailer, category, brand, sku, standard_volume, price_std, nd, velocity_std, relative_price_std, include, price_change_pct, nd_change_pp, nd_mode, baseline_revenue, baseline_volume`  
**Impact**: UI uses app.py version, joint model scenario engine uses engine.py version. Schema drift causes bugs.  
**Recommendation**: **Unify into single canonical function in engine.py**. app.py should import and use it. Define canonical schema in dataclass.

---

## E-006 — Scenario Action Table Rejects Raw Data with KeyError (Bad API)

**Severity**: High  
**Status**: Confirmed  
**Location**: tests/test_app.py:165-175, engine.py:6062  
**Evidence**: Test expects `KeyError` when raw data passed. Function fails on `df["price_std"]` access.  
**Impact**: Uninformative error for users. Public API should validate explicitly.  
**Recommendation**: **Add explicit validation** at start of `create_scenario_action_table`:
```python
required = {"price_std", "nd", "standard_volume", ...}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Scenario action table requires enriched data. Missing: {missing}")
```

---

## E-007 — SCHEMA_ALIASES Dict Defined But Never Used

**Severity**: Medium  
**Status**: Confirmed  
**Location**: engine.py:115-119  
**Evidence**: Dict maps old names to new names but no code references it.  
**Impact**: Dead code, false sense of schema migration support.  
**Recommendation**: **Remove** or **implement actual alias resolution** in prepare_data/validate_input_data.

---

## E-008 — Stale Schema Names in Aggregation Aliases

**Severity**: Medium  
**Status**: Confirmed  
**Location**: engine.py:4324, 4341, 4358, 4375, 4395  
**Evidence**: `price_per_standard_unit=("price_std", "mean")` used in `create_market_summary`, `create_category_summary`, `create_retailer_summary`, `create_brand_summary`, `create_sku_summary`.  
**Impact**: Output DataFrames have column `price_per_standard_unit` instead of `price_std`. Downstream consumers may break.  
**Recommendation**: **Change alias to `price_std=("price_std", "mean")`** in all 5 functions.

---

## E-009 — Legacy vs Joint Model Diagnostic Variable Name Mismatch

**Severity**: High  
**Status**: Confirmed  
**Location**: engine.py:3093-3094 (defaults), app.py:1040 (call site)  
**Evidence**: `summarize_convergence_diagnostics` defaults to `observed_var="log_velocity_std_z_obs", predictive_var="log_velocity_std_z_obs"` (legacy). Joint model uses `observed_units` / `sku_units_obs`. app.py line 1040 calls without args for legacy but joint model path may call incorrectly.  
**Impact**: Wrong PPC coverage calculation for joint model.  
**Recommendation**: **Centralize variable name resolution** via `get_model_variable_names(model_mode)` returning typed dataclass.

---

## E-010 — cross_category_sensitivity Chart Function Exists But Should Be Removed

**Severity**: Medium  
**Status**: Confirmed  
**Location**: engine.py:6023, app.py:2354  
**Evidence**: Function `create_cross_category_sensitivity_chart` exists and is called from app.py. But cross-effect multipliers were removed from joint model UI (commit bdbc91c).  
**Impact**: Chart may produce misleading output based on removed assumption-led logic.  
**Recommendation**: **Remove function and call site** since cross-category sensitivity is assumption-led, not posterior-estimated.

---

## E-011 — Duplicate Aggregation Logic (Market/Category/Retailer/Brand/SKU)

**Severity**: Medium  
**Status**: Likely  
**Location**: Multiple functions:
- `create_market_summary` (4290), `create_category_summary` (4310), `create_retailer_summary` (4330), `create_brand_summary` (4350), `create_sku_summary` (4370)
- `aggregate_market_impact` (5410), `compute_reallocation_breakdown` (5450)
- `run_scenario_suite` (5250) -> `aggregate_scenario` (4080)
- `aggregate_scenario_result` (2160) for joint model  
**Evidence**: Similar groupby/agg patterns repeated across 10+ functions.  
**Impact**: Maintenance burden, inconsistency risk, bug duplication.  
**Recommendation**: **Extract common aggregation helper** with configurable groupby columns and metrics.

---

## E-012 — Legacy Model Code Retained Without Clear Boundaries

**Severity**: Medium  
**Status**: Confirmed  
**Location**: engine.py sections 5-6 (lines 850-1380, 2850-4250, 3570-4150, 5250-5720)  
**Evidence**: Legacy model (build_pymc_model, build_pymc_model_v2, fit_model, extract_elasticities, run_scenario_suite, etc.) coexist with joint model. No clear deprecation markers.  
**Impact**: Cognitive load, dual maintenance, potential for accidental legacy use in joint model paths.  
**Recommendation**: **Mark legacy functions with `_legacy` suffix or move to `legacy.py`**. Add deprecation warnings. Keep until joint model passes all validation gates.

---

## E-013 — Test Expects KeyError Instead of Explicit ValueError

**Severity**: High  
**Status**: Confirmed  
**Location**: tests/test_app.py:165-175  
**Evidence**: `test_scenario_action_table_rejects_raw_data` expects `KeyError` match "price_std".  
**Impact**: Test encodes bad API contract.  
**Recommendation**: **Update test to expect ValueError** after fixing E-006.

---

## E-014 — Test Uses Stale Column Aliases (expected_units_median)

**Severity**: Medium  
**Status**: Confirmed  
**Location**: tests/test_app.py:177-178  
**Evidence**: Test sets `scope["expected_units_median"] = scope["units_p50"]` and `expected_revenue_median = revenue_p50`. These aliases don't exist in canonical schema.  
**Impact**: Test validates wrong schema.  
**Recommendation**: **Update test to use canonical column names** (units_p50, revenue_p50).

---

## E-015 — Undefined Variable compare_scenario in app.py

**Severity**: Critical  
**Status**: Confirmed  
**Location**: app.py:2323  
**Evidence**: `scenarios[compare_scenario]` but `compare_scenario` is not defined in scope. Only check is `"compare_scenario" in locals()`.  
**Impact**: Runtime NameError when that code path executes.  
**Recommendation**: **Fix variable reference** - likely should be a string key or session state variable.

---

## E-016 — conjure Variable Names in Joint Model Path

**Severity**: High  
**Status**: Confirmed  
**Location**: engine.py:3093-3094 (defaults in summarize_convergence_diagnostics)  
**Evidence**: Default args use legacy variable names. Joint model call sites must override explicitly.  
**Impact**: Easy to forget override, leading to wrong diagnostics.  
**Recommendation**: **Remove defaults**, require explicit args, or add model_mode parameter that resolves names.

---

## E-017 — Legacy Chart Functions Unused/Untested

**Severity**: Low  
**Status**: Needs Validation  
**Location**: engine.py:5570-5720 (create_parameter_waterfall, create_reallocation_waterfall, create_dumbbell_chart, create_category_bubble_map, create_brand_pack_heatmap, create_cross_category_sensitivity_chart)  
**Evidence**: Not obviously called from app.py (except create_cross_category_sensitivity_chart).  
**Impact**: Dead code bloat.  
**Recommendation**: **Verify usage**, remove if truly dead.

---

## E-018 — _posterior_group Exception Swallows AttributeError

**Severity**: Medium  
**Status**: Confirmed  
**Location**: engine.py:255-265  
**Evidence**: 
```python
if hasattr(trace, "groups"):
    try:
        return trace["posterior"]
    except Exception:
        pass
```
If trace has `groups` but `__getitem__` raises `KeyError`, it's swallowed. Then falls through to final try/except which raises ValueError.  
**Impact**: Correct error eventually raised but with extra noise.  
**Recommendation**: **Catch specific exceptions** (KeyError, TypeError).

---

## E-019 — _find_posterior_variable Returns First Match Without Validation

**Severity**: Low  
**Status**: Confirmed  
**Location**: engine.py:290-305  
**Evidence**: Returns first candidate found in `trace.posterior.data_vars`. No validation that it's the right variable.  
**Impact**: Could pick wrong variable if multiple candidates exist.  
**Recommendation**: **Add validation** or require exact match.

---

## E-020 — Mixed slots=True / Regular Dataclasses

**Severity**: Low  
**Status**: Confirmed  
**Location**: engine.py various  
**Evidence**: Some dataclasses use `@dataclass(frozen=True, slots=True)` (JointModelConfig, ScenarioResult v2, ConvergenceDiagnostics, NestModelConfig) while others use plain `@dataclass` (ModelConfig, ValidationReport, PreparedData, ModelOutput, ScenarioResult v1, ChoiceSetData, ScenarioAction, BacktestConfig, ActionTableRow).  
**Impact**: Inconsistent performance and mutability guarantees.  
**Recommendation**: **Standardize on `@dataclass(frozen=True, slots=True)`** for all public dataclasses.

---

## Phase 2: Issue Register Complete

### Severity Distribution
| Severity | Count |
|----------|-------|
| Critical | 4 (E-003, E-004, E-015, plus E-001/E-002 as High) |
| High     | 7 (E-001, E-002, E-005, E-006, E-009, E-013, E-016) |
| Medium   | 6 (E-007, E-008, E-010, E-011, E-014, E-018) |
| Low      | 3 (E-012, E-017, E-019, E-020) |

### Priority Fix Order
1. **E-015** - Undefined variable (runtime crash)
2. **E-003** - Broad exception handlers (silent corruption)
3. **E-004** - Incomplete backtest (false capability)
4. **E-001** - Duplicate ScenarioResult (confusion)
5. **E-002** - Duplicate get_posterior_predictive_check (confusion)
6. **E-006** + **E-013** - Scenario action table validation + test fix
7. **E-005** - Cross-file schema inconsistency
8. **E-009** + **E-016** - Diagnostic variable name centralization
9. **E-008** - Stale aggregation aliases
10. **E-010** - Remove cross-category sensitivity chart
11. **E-007** - Remove unused SCHEMA_ALIASES
12. **E-011** - Consolidate aggregation logic
13. **E-012** - Mark legacy code boundaries
14. **E-014** - Fix test aliases
15. **E-020** - Standardize dataclass slots
16. **E-017** - Remove dead chart functions
17. **E-018** - Fix _posterior_group exception
18. **E-019** - Improve _find_posterior_variable

---

## Dependencies

- E-005 depends on E-006 (validation first, then unify)
- E-009 depends on E-016 (centralize variable names)
- E-013 depends on E-006 (test fix follows implementation fix)
- E-012 (legacy boundaries) should be done after joint model validation passes
- E-011 (aggregation consolidation) can be done incrementally

---

## Quality Gate Baseline (Current)

```
python -m compileall app.py engine.py demo_data.py  → PASS
ruff check .  → 6 errors (1 F821 undefined name, 2 C402, 1 PLR5501, 1 I001, 1 E712)
mypy app.py engine.py demo_data.py  → ERROR (google package missing)
pytest -q  → 35 tests collected, 35 passed (with PYTHONPATH=.)
```

