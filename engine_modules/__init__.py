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
    ADVANCED_CHAINS,
    ADVANCED_CONFIG,
    ADVANCED_DRAWS,
    ADVANCED_TUNE,
    DEFAULT_CHAINS,
    DEFAULT_CONFIG,
    DEFAULT_DRAWS,
    DEFAULT_N_SPLINE_KNOTS,
    DEFAULT_SCENARIO_DRAWS,
    DEFAULT_SPLINE_DEGREE,
    DEFAULT_TARGET_ACCEPT,
    DEFAULT_TUNE,
    DERIVED_COLUMNS,
    DiagnosticStatus,
    FAST_CHAINS,
    FAST_CONFIG,
    FAST_DRAWS,
    FAST_TUNE,
    JOINT_CONFIG,
    KEY_COLUMNS,
    MAX_RHAT_THRESHOLD,
    MAX_TREE_DEPTH_THRESHOLD,
    MIN_BFMI_THRESHOLD,
    MIN_ESS_BULK_THRESHOLD,
    MIN_ESS_TAIL_THRESHOLD,
    RAW_COLUMNS,
    REQUIRED_COLUMNS,
    SCENARIO_AGG_LEVELS,
    SCENARIO_RESULT_COLUMNS,
    ChoiceSetData,
    ConvergenceDiagnostics,
    ModelConfig,
    ModelHealth,
    ModelVariableNames,
    PreprocessState,
    PreviewModelConfig,
    ScenarioAction,
    ScenarioPlan,
    ScenarioResult,
    ValidatedModelConfig,
    ValidationReport,
    validate_raw_columns,
    validate_scenario_result,
)
from contracts import (
    NUMERIC_RAW_COLUMNS as NUMERIC_COLUMNS,
)

# Re-export choice_sets (public API only)
from .choice_sets import (
    build_choice_set_data,
)

# Re-export diagnostics (public API only)
from .diagnostics import (
    build_elasticity_forest_figure,
    compute_posterior_predictive_check,
    compute_sku_elasticities,
    get_model_diagnostics,
    summarize_convergence_diagnostics,
)

# Re-export features (public API only)
from .features import (
    add_core_features,
    add_log_features,
    add_pack_group,
    add_standardized_features,
    add_time_features,
    apply_standardization,
    build_all_features,
    build_spline_basis,
)

# Re-export fitting (public API only)
from .fitting import (
    fit_joint_model,
    fit_model,
    get_model_config,
    run_posterior_predictive,
    run_prior_predictive,
)

# Re-export model (public API only)
from .model import (
    JointModelConfig,
    build_joint_model,
    build_pymc_model,
    build_pymc_model_v2,
    extract_joint_posterior,
    UNAVAILABLE_UTILITY,
    UNAVAILABLE_SHARE_TOLERANCE,
)

# Re-export reporting (public API only)
from .reporting import (
    aggregate_scenario,
    build_market_overview,
    build_nd_velocity_quadrant,
    build_price_pack_architecture,
    calculate_shares,
    decompose_sales_growth,
    prepare_market,
)

# Re-export scenarios (public API only)
from .scenarios import (
    MARKET_LEVELS,
    SOURCE_DESTINATION_COLUMNS,
    aggregate_scenario_result,
    check_scenario_reconciliation,
    compute_source_destination_flows,
    create_driver_waterfall,
    run_joint_scenario_draws,
    run_scenario_plan,
)

# Re-export validation (public API only)
from .validation import (
    PreparedData,
    build_retail_features,
    make_pack_group,
    prepare_data,
    validate_input_data,
)

__all__ = [
    # Contracts (from contracts.py)
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
    "PreviewModelConfig",
    "ValidatedModelConfig",
    "DiagnosticStatus",
    "UNAVAILABLE_UTILITY",
    "UNAVAILABLE_SHARE_TOLERANCE",
    "PreviewModelConfig",
    "ValidatedModelConfig",
    "DiagnosticStatus",
    "UNAVAILABLE_UTILITY",
    "UNAVAILABLE_SHARE_TOLERANCE",
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
    "build_choice_set_data",
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
    "run_scenario_plan",
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
]
