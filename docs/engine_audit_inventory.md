# Engine.py Audit Inventory

Generated from AST analysis of engine.py (6591 lines)

## Classes (13)

1. **ModelConfig** (line ~89) - Legacy model configuration
2. **ValidationReport** (line ~130) - Input validation result
3. **PreparedData** (line ~150) - Prepared data container
4. **ModelOutput** (line ~170) - Legacy model output
5. **ScenarioResult** (line ~242) - **DUPLICATE/DEAD** - Legacy scenario result, never used
6. **ChoiceSetData** (line ~270) - Joint model choice set data
7. **JointModelConfig** (line ~1345) - Joint model configuration
8. **ScenarioAction** (line ~1430) - Joint model scenario action
9. **ScenarioResult** (line ~1738) - **ACTIVE** - Joint model scenario result (frozen, slots=True)
10. **NestModelConfig** (line ~2587) - Nest model configuration
11. **ConvergenceDiagnostics** (line ~3080) - Diagnostic results (frozen, slots=True)
12. **BacktestConfig** (line ~3310) - Backtest configuration
13. **ActionTableRow** (line ~6050) - Action table row type

## Functions (115 total, 21 private)

### Public API Functions (94)

| Function | Line | Returns | Notes |
|----------|------|---------|-------|
| make_pack_group | 35 | pd.Series | |
| validate_input_data | 200 | ValidationReport | |
| prepare_data | 280 | tuple[pd.DataFrame, dict] | |
| add_indices | 380 | tuple[pd.DataFrame, dict] | |
| fit_spline | 420 | tuple[SplineTransformer, np.ndarray] | |
| create_data_quality_report | 550 | dict | |
| build_choice_set_data | 610 | ChoiceSetData | |
| validate_choice_set | 720 | dict | |
| apply_scenario_to_choice_set | 790 | tuple[np.ndarray, np.ndarray, np.ndarray] | |
| build_pymc_model | 850 | pm.Model | Legacy |
| build_pymc_model_v2 | 1030 | pm.Model | Legacy v2 |
| build_joint_sku_share_model | 1390 | pm.Model | Joint model |
| sample_joint_posterior_predictive | 1640 | az.InferenceData | Joint model |
| fit_joint_model | 1680 | az.InferenceData | Joint model |
| extract_joint_posterior | 1720 | dict | Joint model |
| run_joint_scenario_draws | 1880 | ScenarioResult | Joint model |
| aggregate_scenario_result | 2160 | pd.DataFrame | Joint model |
| check_scenario_reconciliation | 2230 | dict | Joint model |
| build_source_destination_summary | 2300 | pd.DataFrame | Joint model |
| create_reallocation_sankey | 2380 | go.Figure | Joint model |
| build_nest_allocation_model | 2607 | pm.Model | Nest model |
| fit_nest_model | 2768 | az.InferenceData | Nest model |
| extract_nest_posterior | 2830 | dict | Nest model |
| fit_model | 2850 | az.InferenceData | Legacy |
| get_model_diagnostics | 2900 | dict | Legacy |
| get_posterior_predictive_check (v1) | 2874 | pd.DataFrame | Legacy - **DUPLICATE** |
| make_trace_figure | 2940 | Figure | |
| make_rank_figure | 2980 | Figure | |
| make_energy_figure | 3010 | Figure | |
| make_ppc_figure | 3030 | Figure | |
| make_elasticity_forest_figure | 3050 | Figure | |
| summarize_convergence_diagnostics | 3091 | ConvergenceDiagnostics | Both models |
| build_elasticity_forest_figure | 3160 | go.Figure | |
| calculate_wape | 3230 | float | |
| calculate_bias | 3240 | float | |
| calculate_directional_accuracy | 3250 | float | |
| run_rolling_backtest | 3327 | pd.DataFrame | **INCOMPLETE** |
| build_nd_response_curve | 3410 | pd.DataFrame | |
| add_posterior_predictive | 3470 | None | Legacy |
| apply_target_scope | 3490 | pd.Series | |
| calculate_market_shares | 3510 | pd.DataFrame | |
| summarize_period_shares | 3540 | pd.DataFrame | |
| extract_elasticities | 3570 | pd.DataFrame | Legacy |
| extract_scenario_posterior | 3630 | dict | Legacy |
| price_delta_log_volume | 3680 | np.ndarray | Legacy |
| resolve_target_nd | 3710 | np.ndarray | Legacy |
| nd_delta_log_volume | 3740 | np.ndarray | Legacy |
| combined_delta_log_volume | 3780 | np.ndarray | Legacy |
| summarise_scenario_draws | 3800 | pd.DataFrame | Legacy |
| get_nd_response_curve | 3830 | pd.DataFrame | Legacy |
| get_growth_opportunities | 3880 | pd.DataFrame | |
| create_price_distribution_scenario | 3920 | pd.DataFrame | Legacy |
| calculate_shares | 4020 | dict | |
| calculate_period_market_share | 4050 | pd.DataFrame | |
| aggregate_scenario | 4080 | pd.DataFrame | Legacy |
| get_posterior_predictive_check (v2) | 4264 | pd.DataFrame | Legacy - **DUPLICATE** |
| create_market_summary | 4290 | pd.DataFrame | |
| create_category_summary | 4310 | pd.DataFrame | |
| create_retailer_summary | 4330 | pd.DataFrame | |
| create_brand_summary | 4350 | pd.DataFrame | |
| create_sku_summary | 4370 | pd.DataFrame | |
| classify_distribution_velocity | 4400 | pd.DataFrame | |
| get_expansion_candidates | 4430 | pd.DataFrame | |
| create_growth_decomposition | 4460 | pd.DataFrame | |
| prepare_market | 4510 | PreparedData | |
| run_fast_share_scenario | 4580 | tuple | Legacy |
| decompose_sales_growth | 4660 | pd.DataFrame | |
| compute_price_architecture | 4710 | pd.DataFrame | |
| compute_distribution_opportunity | 4750 | pd.DataFrame | |
| compute_market_share_analytics | 4790 | pd.DataFrame | |
| compute_contribution_to_growth | 4830 | pd.DataFrame | |
| generate_data_health_report | 4870 | dict | |
| compute_model_validation_metrics | 5053 | dict | Legacy |
| build_observed_vs_predicted_data | 5110 | pd.DataFrame | Legacy |
| build_price_ladder_data | 5140 | pd.DataFrame | |
| build_nd_velocity_quadrant_data | 5170 | pd.DataFrame | |
| build_retail_features | 5170 | pd.DataFrame | |
| run_scenario_suite | 5250 | dict | Legacy |
| classify_segment_relationship | 5350 | str | Legacy |
| add_segment_classification | 5380 | pd.DataFrame | Legacy |
| aggregate_market_impact | 5410 | pd.DataFrame | Legacy |
| compute_reallocation_breakdown | 5450 | pd.DataFrame | Legacy |
| compute_parameter_attribution | 5510 | pd.DataFrame | Legacy |
| create_parameter_waterfall | 5570 | go.Figure | Legacy |
| create_reallocation_waterfall | 5600 | go.Figure | Legacy |
| create_dumbbell_chart | 5630 | go.Figure | Legacy |
| create_category_bubble_map | 5660 | go.Figure | Legacy |
| create_brand_pack_heatmap | 5690 | go.Figure | Legacy |
| create_cross_category_sensitivity_chart | 5720 | go.Figure | Legacy |
| create_scenario_action_table (engine) | 6062 | pd.DataFrame | Joint model |
| update_action_table_changes | 6095 | pd.DataFrame | Joint model |
| run_scenario_actions | 6110 | pd.DataFrame | Joint model |
| run_scenario_suite_actions | 6140 | dict | Joint model |
| create_driver_waterfall | 6180 | pd.DataFrame | Joint model |
| create_market_impact_table | 6220 | pd.DataFrame | Joint model |
| create_winner_loser_dumbbell | 6260 | go.Figure | Joint model |
| check_scenario_guardrails | 6300 | dict | Joint model |

### Private Functions (21)

| Function | Line | Returns | Notes |
|----------|------|---------|-------|
| _add_category_relative_pack_groups | 52 | pd.Series | |
| _posterior_group | 250 | Any | |
| _posterior_array | 270 | np.ndarray | |
| _find_posterior_variable | 290 | str | None |
| _coerce_month | 310 | pd.Series | |
| _get_category_pack_peer_price | 470 | pd.Series | |
| _compute_peer_price_matrix | 1780 | np.ndarray | |
| _compute_utility | 1800 | np.ndarray | |
| _softmax | 1820 | np.ndarray | |
| _dirichlet_multinomial_draw | 1830 | np.ndarray | |
| _classify_source_relationship | 2310 | str | |
| _thin_posterior | 4110 | dict | |
| _posterior_delta_log_velocity | 4130 | tuple | Legacy |
| _simulate_scenario_core | 5230 | pd.DataFrame | Legacy |
| _qcut_category | 5100 | pd.Series | |
| compute_quantiles | 2000 | tuple | In run_joint_scenario_draws |

### Constants (22)

| Constant | Line |
|----------|------|
| REQUIRED_COLUMNS | 89 |
| VALIDATION_RULES | 120 |
| SPLINE_DEFAULTS | 140 |
| MODEL_DEFAULTS | 160 |
| FAST_CONFIG | 180 |
| DEFAULT_CONFIG | 190 |
| ADVANCED_CONFIG | 200 |
| JOINT_MODEL_DEFAULT_CONFIG | 1380 |
| NEST_MODEL_DEFAULT_CONFIG | 2604 |
| SCHEMA_ALIASES | 115 |
| etc. | |

## Duplicate Functions

### get_posterior_predictive_check (2 versions)
- **v1** (line 2874): Takes `model: pm.Model, idata, prepared_df` - uses PyMC's sample_posterior_predictive
- **v2** (line 4264): Takes `trace, df, max_points` - uses internal posterior extraction

Both are for the legacy model but have different signatures and implementations.

### ScenarioResult (2 classes)
- **v1** (line 242): Simple dataclass with `df, price_change, nd_change, summary` - **NEVER USED**
- **v2** (line 1738): Full joint model result with per-draw arrays - **ACTIVE**

### create_scenario_action_table (2 functions - cross-file)
- **engine.py** (line 6062): Produces simplified table with `baseline_price_std, baseline_nd, new_price_std, new_nd, price_change, nd_change`
- **app.py** (line 1114): Produces detailed table with `month, standard_volume, velocity_std, relative_price_std, include, price_change_pct, nd_change_pp, nd_mode`

## Schema Drift Findings

### Active Old Names Still in Code

| Old Name | New Name | Locations |
|----------|----------|-----------|
| price_per_standard_unit | price_std | engine.py:115 (SCHEMA_ALIASES), engine.py:4324,4341,4358,4375,4395 (agg aliases), blueprint.md |
| brand_units | brand_std_volume | engine.py:5175 (docstring only) |
| log_velocity_z_obs | log_velocity_std_z_obs | 15+ locations in engine.py |
| velocity_std | (keep) | Used throughout - standard volume metric |
| expected_units_median | units_p50 | tests/test_app.py:177 (test alias) |
| expected_revenue_median | revenue_p50 | tests/test_app.py:178 (test alias) |
| cross_category_sensitivity | (removed from UI) | engine.py:6023 (function exists), app.py:2354 (calls it) |
| own_brand / rival_brand | (never in code) | blueprint.md only |

### SCHEMA_ALIASES Dict (line 115)
```python
SCHEMA_ALIASES = {
    "price_per_standard_unit": "price_std",
    "velocity_std": "velocity_std",
    "log_velocity_std": "log_velocity_std",
}
```
This alias mapping is defined but **never used** in the codebase.

## Dangerous Coding Patterns

### Broad Exception Handlers (4)

1. **Line 260-261** in `_posterior_group()`:
   ```python
   try:
       return trace["posterior"]
   except Exception:
       pass
   ```

2. **Line 3125** in `summarize_convergence_diagnostics()`:
   ```python
   try:
       summary = az.summary(...)
   except Exception:
       pass
   ```

3. **Line 3137** in `summarize_convergence_diagnostics()`:
   ```python
   try:
       y_obs = ...
       coverage_90 = ...
   except Exception:
       pass
   ```

4. **Line 4285** in `get_posterior_predictive_check` (v2):
   ```python
   try:
       pred = _posterior_array(...)
   except Exception:
       return pd.DataFrame()
   ```

### Incomplete Implementation

**run_rolling_backtest** (line 3327):
- Has placeholder comment: "Placeholder for full implementation"
- `results = []` list is created but **never populated**
- Returns empty DataFrame
- **Not tested**

### Dead Code

1. **ScenarioResult v1** (line 242) - Never instantiated or referenced
2. **SCHEMA_ALIASES** dict - Never used
3. **get_posterior_predictive_check v1** (line 2874) - Need to verify if called
4. **_compute_peer_price_matrix** - Used only in run_joint_scenario_draws
5. **Various legacy chart functions** - create_cross_category_sensitivity_chart, create_category_bubble_map, etc. - Need to verify usage

## Cross-File References

### Functions defined in engine.py, called from app.py
- validate_input_data
- prepare_data
- build_retail_features
- build_choice_set_data
- build_joint_sku_share_model
- fit_joint_model
- sample_joint_posterior_predictive
- extract_joint_posterior
- run_joint_scenario_draws
- aggregate_scenario_result
- check_scenario_reconciliation
- build_source_destination_summary
- create_reallocation_sankey
- get_model_diagnostics
- summarize_convergence_diagnostics
- build_elasticity_forest_figure
- create_scenario_action_table (engine version)
- update_action_table_changes
- run_scenario_actions
- run_scenario_suite_actions
- create_driver_waterfall
- create_market_impact_table
- create_winner_loser_dumbbell
- check_scenario_guardrails
- create_cross_category_sensitivity_chart
- build_price_ladder_data
- build_nd_velocity_quadrant_data
- compute_price_architecture
- compute_distribution_opportunity
- compute_market_share_analytics
- classify_distribution_velocity
- get_expansion_candidates
- create_growth_decomposition
- generate_data_health_report
- compute_model_validation_metrics
- build_observed_vs_predicted_data
- make_ppc_figure
- make_trace_figure
- make_rank_figure
- make_energy_figure
- extract_elasticities

### Functions defined in app.py
- create_scenario_action_table (app version - different schema!)
- render_scenario_action_editor

## Test Coverage (from test_app.py)

### Tested Functions
- test_demo_data_schema
- test_prepare_data
- test_build_retail_features
- test_model_config_presets
- test_scenario_action_table_rejects_raw_data (expects KeyError - **BAD PATTERN**)
- test_scenario_action_table_accepts_enriched_data
- test_joint_model_builds
- test_joint_scenario_reconciliation
- test_mutual_exclusivity
- test_gain_loss_balance
- test_sankey_hidden_when_unreconciled
- test_winner_loser_interval_ordering
- test_unavailable_sku_zero_share
- test_market_delta_consistency

### Not Tested (High Priority)
- build_pymc_model / build_pymc_model_v2 (legacy)
- fit_model (legacy)
- get_model_diagnostics (legacy)
- get_posterior_predictive_check (both versions)
- run_rolling_backtest (incomplete)
- build_nest_allocation_model
- fit_nest_model
- All legacy chart functions
- All legacy scenario functions (run_scenario_suite, etc.)

