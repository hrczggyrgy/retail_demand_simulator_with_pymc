"""
Engine modules package - modularized engine components.

This package splits the monolithic engine.py into focused modules:
- contracts: Single source of truth for all schemas and dataclasses
- validation: Input validation and data preparation
- features: Feature engineering utilities
- choice_sets: Choice-set data layer for joint model
- model: PyMC model building (v1, v2, joint)
- fitting: Model fitting and sampling
- diagnostics: Convergence diagnostics, elasticity, PPC
- scenarios: Counterfactual scenario engine, market impact
- reporting: Aggregation, market overview, summaries
"""

# Re-export contracts (single source of truth)
from contracts import (
    RAW_COLUMNS,
    REQUIRED_COLUMNS,
    NUMERIC_RAW_COLUMNS as NUMERIC_COLUMNS,
    KEY_COLUMNS,
    DERIVED_COLUMNS,
    SCENARIO_RESULT_COLUMNS,
    SCENARIO_AGG_LEVELS,
    DEFAULT_TARGET_ACCEPT,
    DEFAULT_N_SPLINE_KNOTS,
    DEFAULT_SPLINE_DEGREE,
    FAST_DRAWS,
    FAST_TUNE,
    FAST_CHAINS,
    DEFAULT_DRAWS,
    DEFAULT_TUNE,
    DEFAULT_CHAINS,
    ADVANCED_DRAWS,
    ADVANCED_TUNE,
    ADVANCED_CHAINS,
    DEFAULT_SCENARIO_DRAWS,
    MAX_RHAT_THRESHOLD,
    MIN_ESS_BULK_THRESHOLD,
    MIN_ESS_TAIL_THRESHOLD,
    MIN_BFMI_THRESHOLD,
    MAX_TREE_DEPTH_THRESHOLD,
    ModelConfig,
    FAST_CONFIG,
    DEFAULT_CONFIG,
    ADVANCED_CONFIG,
    JOINT_CONFIG,
    ScenarioAction,
    ScenarioPlan,
    ChoiceSetData,
    ScenarioResult,
    PreprocessState,
    ModelVariableNames,
    ConvergenceDiagnostics,
    ModelHealth,
    ValidationReport,
    validate_raw_columns,
    validate_scenario_result,
)

# Re-export validation
from .validation import (
    make_pack_group,
    validate_input_data,
    prepare_data,
    build_retail_features,
    PreparedData,
)

# Re-export features
from .features import (
    add_core_features,
    add_pack_group,
    add_time_features,
    add_log_features,
    add_standardized_features,
    build_spline_basis,
    apply_standardization,
    build_all_features,
)

# Re-export choice_sets
from .choice_sets import (
    _get_category_pack_peer_price,
    build_choice_set_data,
    _compute_peer_price_matrix,
    _recompute_relative_prices,
)

# Re-export model
from .model import (
    build_pymc_model,
    build_pymc_model_v2,
    JointModelConfig,
    build_joint_model,
    extract_joint_posterior,
)

# Re-export fitting
from .fitting import (
    fit_model,
    fit_joint_model,
    run_prior_predictive,
    run_posterior_predictive,
    get_model_config,
)

# Re-export diagnostics
from .diagnostics import (
    get_model_diagnostics,
    summarize_convergence_diagnostics,
    compute_sku_elasticities,
    build_elasticity_forest_figure,
    compute_posterior_predictive_check,
)

# Re-export scenarios
from .scenarios import (
    run_joint_scenario_draws,
    _compute_utility,
    _softmax,
    compute_source_destination_flows,
    SOURCE_DESTINATION_COLUMNS,
    MARKET_LEVELS,
    aggregate_scenario_result,
    check_scenario_reconciliation,
    create_driver_waterfall,
)

# Re-export reporting
from .reporting import (
    aggregate_scenario,
    calculate_shares,
    decompose_sales_growth,
    prepare_market,
    build_market_overview,
    build_price_pack_architecture,
    build_nd_velocity_quadrant,
    _SUMMARY_METRICS,
)

# Legacy ValidationReport from validation.py
from .validation import ValidationReport as LegacyValidationReport

__all__ = [
    # Contracts
    "RAW_COLUMNS",
    "REQUIRED_COLUMNS",
    "NUMERIC_COLUMNS",
    "KEY_COLUMNS",
    "DERIVED_COLUMNS",
    "SCENARIO_RESULT_COLUMNS",
    "SCENARIO_AGG_LEVELS",
    "DEFAULT_TARGET_ACCEPT",
    "DEFAULT_N_SPLINE_KNOTS",
    "DEFAULT_SPLINE_DEGREE",
    "FAST_DRAWS",
    "FAST_TUNE",
    "FAST_CHAINS",
    "DEFAULT_DRAWS",
    "DEFAULT_TUNE",
    "DEFAULT_CHAINS",
    "ADVANCED_DRAWS",
    "ADVANCED_TUNE",
    "ADVANCED_CHAINS",
    "DEFAULT_SCENARIO_DRAWS",
    "MAX_RHAT_THRESHOLD",
    "MIN_ESS_BULK_THRESHOLD",
    "MIN_ESS_TAIL_THRESHOLD",
    "MIN_BFMI_THRESHOLD",
    "MAX_TREE_DEPTH_THRESHOLD",
    "ModelConfig",
    "FAST_CONFIG",
    "DEFAULT_CONFIG",
    "ADVANCED_CONFIG",
    "JOINT_CONFIG",
    "ScenarioAction",
    "ScenarioPlan",
    "ChoiceSetData",
    "ScenarioResult",
    "PreprocessState",
    "ModelVariableNames",
    "ConvergenceDiagnostics",
    "ModelHealth",
    "ValidationReport",
    "validate_raw_columns",
    "validate_scenario_result",
    # Validation
    "make_pack_group",
    "validate_input_data",
    "prepare_data",
    "build_retail_features",
    "PreparedData",
    # Features
    "add_core_features",
    "add_pack_group",
    "add_time_features",
    "add_log_features",
    "add_standardized_features",
    "build_spline_basis",
    "apply_standardization",
    "build_all_features",
    # Choice sets
    "_get_category_pack_peer_price",
    "build_choice_set_data",
    "_compute_peer_price_matrix",
    "_recompute_relative_prices",
    # Model
    "build_pymc_model",
    "build_pymc_model_v2",
    "JointModelConfig",
    "build_joint_model",
    "extract_joint_posterior",
    # Fitting
    "fit_model",
    "fit_joint_model",
    "run_prior_predictive",
    "run_posterior_predictive",
    "get_model_config",
    # Diagnostics
    "get_model_diagnostics",
    "summarize_convergence_diagnostics",
    "compute_sku_elasticities",
    "build_elasticity_forest_figure",
    "compute_posterior_predictive_check",
    # Scenarios
    "run_joint_scenario_draws",
    "_compute_utility",
    "_softmax",
    "compute_source_destination_flows",
    "SOURCE_DESTINATION_COLUMNS",
    "MARKET_LEVELS",
    "aggregate_scenario_result",
    "check_scenario_reconciliation",
    "create_driver_waterfall",
    # Reporting
    "aggregate_scenario",
    "calculate_shares",
    "decompose_sales_growth",
    "prepare_market",
    "build_market_overview",
    "build_price_pack_architecture",
    "build_nd_velocity_quadrant",
    "_SUMMARY_METRICS",
    # Legacy
    "LegacyValidationReport",
]