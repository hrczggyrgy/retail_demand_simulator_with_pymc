"""
Single source of truth for all shared contracts, schemas, and dataclasses.

This module eliminates schema drift between app.py, engine.py, and demo_data.py.
All raw/derived column names, scenario/action/result types, model configs,
and preprocessing state are defined here exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# =============================================================================
# RAW DATA CONTRACT (exactly 10 columns)
# =============================================================================

RAW_COLUMNS: tuple[str, ...] = (
    "month",
    "retailer",
    "category",
    "brand",
    "sku",
    "units",
    "revenue",
    "sku_stores",
    "retailer_stores",
    "pack_size",
)

NUMERIC_RAW_COLUMNS: list[str] = [
    "units",
    "revenue",
    "sku_stores",
    "retailer_stores",
    "pack_size",
]

KEY_COLUMNS: list[str] = ["month", "retailer", "category", "brand", "sku"]

# Backward compatibility alias (list for pandas DataFrame constructor)
REQUIRED_COLUMNS = list(RAW_COLUMNS)

# =============================================================================
# DERIVED / FEATURE COLUMN NAMES (canonical)
# =============================================================================

# Core derived measures
STANDARD_VOLUME = "standard_volume"
PRICE_STD = "price_std"              # revenue / units / pack_size
ND = "nd"                            # sku_stores / retailer_stores
VELOCITY_STD = "velocity_std"        # standard_volume / sku_stores
RELATIVE_PRICE = "relative_price"    # price_std / peer_price (within retailer×category×pack_group×month)
LOG_ND = "log_nd"
LOG_RELATIVE_PRICE = "log_relative_price"
LOG_VELOCITY_STD = "log_velocity_std"

# Z-scored (standardized) variants for modeling
LOG_ND_Z = "log_nd_z"
LOG_RELATIVE_PRICE_Z = "log_relative_price_z"
LOG_VELOCITY_STD_Z = "log_velocity_std_z"

# Hierarchy / grouping columns
PACK_GROUP = "pack_group"            # Derived from pack_size (e.g., "small", "medium", "large")
MONTH_NUM = "month_num"              # Numeric month for seasonality

# All derived columns expected after feature engineering
DERIVED_COLUMNS: tuple[str, ...] = (
    STANDARD_VOLUME,
    PRICE_STD,
    ND,
    VELOCITY_STD,
    RELATIVE_PRICE,
    LOG_ND,
    LOG_RELATIVE_PRICE,
    LOG_VELOCITY_STD,
    LOG_ND_Z,
    LOG_RELATIVE_PRICE_Z,
    LOG_VELOCITY_STD_Z,
    PACK_GROUP,
    MONTH_NUM,
)

# =============================================================================
# SCENARIO OUTPUT CONTRACT (canonical column names)
# =============================================================================

SCENARIO_RESULT_COLUMNS: tuple[str, ...] = (
    "units_p05",
    "units_p50",
    "units_p95",
    "revenue_p05",
    "revenue_p50",
    "revenue_p95",
    "scenario_unit_price",
    "scenario_nd",
    "scenario_price_change",
    "scenario_nd_change",
)

SCENARIO_AGG_LEVELS: tuple[str, ...] = (
    "market",
    "retailer",
    "category",
    "brand",
    "brand_pack",
    "sku",
)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

DEFAULT_TARGET_ACCEPT = 0.90
DEFAULT_N_SPLINE_KNOTS = 4
DEFAULT_SPLINE_DEGREE = 3

FAST_DRAWS = 500
FAST_TUNE = 500
FAST_CHAINS = 2

DEFAULT_DRAWS = 800
DEFAULT_TUNE = 800
DEFAULT_CHAINS = 4

ADVANCED_DRAWS = 1000
ADVANCED_TUNE = 1000
ADVANCED_CHAINS = 4

DEFAULT_SCENARIO_DRAWS = 400

# Model health thresholds
MAX_RHAT_THRESHOLD = 1.01
MIN_ESS_BULK_THRESHOLD = 400
MIN_ESS_TAIL_THRESHOLD = 400
MIN_BFMI_THRESHOLD = 0.3
MAX_TREE_DEPTH_THRESHOLD = 10


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for PyMC model fitting."""
    draws: int = DEFAULT_DRAWS
    tune: int = DEFAULT_TUNE
    chains: int = DEFAULT_CHAINS
    target_accept: float = 0.95
    random_seed: int = 42

    likelihood: str = "student_t"       # "normal" or "student_t"
    use_category_price_pooling: bool = True
    use_sku_nd_effects: bool = False    # Turn on after base model validates
    n_spline_knots: int = DEFAULT_N_SPLINE_KNOTS
    spline_degree: int = DEFAULT_SPLINE_DEGREE
    scenario_draws: int = DEFAULT_SCENARIO_DRAWS


# Predefined configs
FAST_CONFIG = ModelConfig(
    draws=FAST_DRAWS,
    tune=FAST_TUNE,
    chains=FAST_CHAINS,
    target_accept=0.92,
    likelihood="normal",
    use_category_price_pooling=False,
    use_sku_nd_effects=False,
)

DEFAULT_CONFIG = ModelConfig()

ADVANCED_CONFIG = ModelConfig(
    draws=ADVANCED_DRAWS,
    tune=ADVANCED_TUNE,
    chains=ADVANCED_CHAINS,
    target_accept=0.95,
    likelihood="student_t",
    use_category_price_pooling=True,
    use_sku_nd_effects=False,
)

JOINT_CONFIG = ModelConfig(
    draws=1200,
    tune=1500,
    chains=4,
    target_accept=0.98,
    likelihood="student_t",
    use_category_price_pooling=True,
    use_sku_nd_effects=False,
    n_spline_knots=4,
    spline_degree=3,
    scenario_draws=500,
)


# =============================================================================
# SCENARIO TYPES
# =============================================================================

@dataclass(frozen=True, slots=True)
class ScenarioAction:
    """A single scenario action for a specific retailer × SKU × month."""
    retailer: str
    category: str
    brand: str
    sku: str
    month: str
    new_price_std: float | None = None      # Absolute price per standard unit
    new_nd: float | None = None             # Numeric distribution (0-1)
    nd_mode: str = "pp"                     # "pp" (percentage points), "relative", "absolute"


@dataclass(frozen=True, slots=True)
class ScenarioPlan:
    """
    High-level commercial scenario plan.
    
    This is the user-facing input: a set of focal SKU interventions
    with explicit price and ND changes. The engine translates this
    into counterfactual evaluations.
    """
    actions: tuple[ScenarioAction, ...]
    name: str = "Custom Scenario"
    description: str = ""
    
    def __post_init__(self):
        if not self.actions:
            raise ValueError("ScenarioPlan requires at least one action")
        # Validate unique (month, retailer, sku) keys
        keys = [(a.month, a.retailer, a.sku) for a in self.actions]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate (month, retailer, sku) in scenario actions")

    @classmethod
    def from_action_table(cls, action_df: pd.DataFrame, name: str = "Custom Scenario") -> ScenarioPlan:
        """Build a ScenarioPlan from the editable action table DataFrame."""
        checked = action_df[action_df.get("include", False)].copy()
        if checked.empty:
            raise ValueError("No actions selected (check 'Include' column)")
        
        actions = []
        for _, row in checked.iterrows():
            price_change_pct = row.get("price_change_pct", 0.0) / 100.0
            baseline_price_std = row.get("price_std", 0.0)
            new_price_std = baseline_price_std * (1 + price_change_pct) if price_change_pct != 0 else None
            
            nd_change_pp = row.get("nd_change_pp", 0.0) / 100.0
            nd_mode = row.get("nd_mode", "pp")
            if nd_change_pp != 0:
                if nd_mode == "pp":
                    new_nd = row["nd"] + nd_change_pp
                elif nd_mode == "relative":
                    new_nd = row["nd"] * (1 + nd_change_pp)
                elif nd_mode == "absolute":
                    new_nd = nd_change_pp
                else:
                    new_nd = None
            else:
                new_nd = None
            
            action = ScenarioAction(
                retailer=row["retailer"],
                category=row["category"],
                brand=row["brand"],
                sku=row["sku"],
                month=str(row["month"]),
                new_price_std=new_price_std,
                new_nd=new_nd,
                nd_mode=nd_mode,
            )
            actions.append(action)
        
        return cls(actions=tuple(actions), name=name)


# =============================================================================
# ENGINE CORE TYPES
# =============================================================================

@dataclass(frozen=True, slots=True)
class ChoiceSetData:
    """Container for all choice-set arrays and metadata for the SKU allocation model."""
    # Core arrays
    observed_units: np.ndarray              # (n_markets, n_skus)
    market_total_units: np.ndarray          # (n_markets,)
    price_std: np.ndarray                   # (n_markets, n_skus) - absolute price per standard unit
    relative_price: np.ndarray              # (n_markets, n_skus) - price_std / peer_price
    nd: np.ndarray                          # (n_markets, n_skus)
    available_mask: np.ndarray              # (n_markets, n_skus)
    # Mapping arrays
    market_retailer_idx: np.ndarray         # (n_markets,)
    market_category_idx: np.ndarray         # (n_markets,)
    market_month_idx: np.ndarray            # (n_markets,)
    sku_brand_idx: np.ndarray               # (n_skus,)
    sku_category_idx: np.ndarray            # (n_skus,)
    sku_pack_group_idx: np.ndarray          # (n_skus,)
    brand_category_idx: np.ndarray          # (n_brands,)
    # Metadata
    market_ids: list[str]                   # Length n_markets
    sku_ids: list[str]                      # Length n_skus
    retailer_levels: np.ndarray
    category_levels: np.ndarray
    brand_levels: np.ndarray
    month_levels: np.ndarray
    pack_group_levels: np.ndarray
    sku_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_sku: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Results from a draw-by-draw counterfactual scenario."""
    # Per-draw results (n_draws, n_markets, n_skus)
    baseline_units: np.ndarray
    scenario_units: np.ndarray
    baseline_shares: np.ndarray
    scenario_shares: np.ndarray
    # Summaries (n_markets, n_skus)
    baseline_units_p50: np.ndarray
    scenario_units_p50: np.ndarray
    baseline_units_p05: np.ndarray
    scenario_units_p05: np.ndarray
    baseline_units_p95: np.ndarray
    scenario_units_p95: np.ndarray
    delta_units_p50: np.ndarray
    delta_units_p05: np.ndarray
    delta_units_p95: np.ndarray
    # Metadata
    choice_data: ChoiceSetData
    actions: tuple[ScenarioAction, ...]


# =============================================================================
# MARKET-FIRST OUTCOME CONTRACT (market-level scenario results)
# =============================================================================

@dataclass(frozen=True, slots=True)
class MarketOutcome:
    """
    Market-level scenario outcome - the primary commercial answer.
    
    Aggregated to retailer×category level (the decision unit).
    Shows total market volume change, selected SKU/brand impact,
    and competitor reallocation sources.
    """
    market_id: str                          # "retailer|category"
    retailer: str
    category: str
    # Baseline vs scenario
    baseline_volume: float
    scenario_volume: float
    delta_volume: float
    delta_volume_pct: float
    # Share impact
    baseline_share: float
    scenario_share: float
    delta_share_pp: float
    # Revenue
    baseline_revenue: float
    scenario_revenue: float
    delta_revenue: float
    # Selected focal SKU impact (if single action)
    focal_sku: str | None = None
    focal_baseline_volume: float | None = None
    focal_scenario_volume: float | None = None
    focal_delta_volume: float | None = None
    # Competitor reallocation sources (per-draw median)
    competitor_sources: dict[str, float] | None = None  # {source_label: delta_volume}
    # Confidence
    probability_positive: float = 0.5
    is_decision_ready: bool = False


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    """
    Complete scenario outcome bundle.
    
    Contains market-level outcomes (primary) plus SKU-level detail (secondary).
    """
    plan: ScenarioPlan
    market_outcomes: tuple[MarketOutcome, ...]
    sku_outcomes: pd.DataFrame              # Per-SKU deltas with intervals
    driver_waterfall: pd.DataFrame | None = None  # Price vs ND vs Interaction
    source_destination: pd.DataFrame | None = None  # Reallocation flows
    reconciliation_passed: bool = True
    diagnostics: ConvergenceDiagnostics | None = None


# =============================================================================
# PREPROCESSING STATE (fitted artifact)
# =============================================================================

@dataclass(frozen=True, slots=True)
class PreprocessState:
    """
    Serialized preprocessing state for exact scenario transform reuse.
    
    Contains all encodings, mappings, and standardization parameters
    needed to transform new scenario inputs identically to training data.
    """
    # Category encodings
    retailer_levels: np.ndarray
    category_levels: np.ndarray
    brand_levels: np.ndarray
    month_levels: np.ndarray
    pack_group_levels: np.ndarray
    sku_ids: np.ndarray
    
    # Mappings
    sku_to_retailer: dict[str, str]
    sku_to_category: dict[str, str]
    sku_to_brand: dict[str, str]
    sku_to_pack_group: dict[str, str]
    sku_to_pack_size: dict[str, float]
    
    # Standardization parameters (z-score)
    log_nd_mean: float
    log_nd_std: float
    log_relative_price_mean: float
    log_relative_price_std: float
    log_velocity_std_mean: float
    log_velocity_std_std: float
    
    # Spline basis
    spline_knots: np.ndarray
    spline_degree: int
    
    # Peer price computation
    peer_price_baseline: dict[str, float]  # Key: market_id, Value: peer_price
    
    # Model variable names (for PPC/diagnostics)
    model_variable_names: ModelVariableNames


@dataclass(frozen=True, slots=True)
class ModelVariableNames:
    """Model-specific variable names for observed and predictive data."""
    observed_name: str
    predictive_name: str
    metric_name: str


def get_model_variable_names(model_mode: str) -> ModelVariableNames:
    """
    Get the correct variable names for a given model mode.
    
    Args:
        model_mode: One of "Fast (2 chains)", "Default (4 chains)", "Advanced (4 chains, Student-t)",
                   "Joint SKU Share Model (Dirichlet-Multinomial)"
    
    Returns:
        ModelVariableNames with observed_name, predictive_name, and metric_name
    """
    if model_mode == "Joint SKU Share Model (Dirichlet-Multinomial)":
        return ModelVariableNames(
            observed_name="observed_units",
            predictive_name="sku_units_obs",
            metric_name="units"
        )
    else:
        return ModelVariableNames(
            observed_name="log_velocity_std_z_obs",
            predictive_name="log_velocity_std_z_obs",
            metric_name="log_velocity_std_z"
        )


# =============================================================================
# DIAGNOSTICS & HEALTH
# =============================================================================

@dataclass(frozen=True, slots=True)
class ConvergenceDiagnostics:
    divergences: int
    max_rhat: float | None
    min_ess_bulk: float | None
    min_ess_tail: float | None
    coverage_90: float | None
    min_bfmi: float | None
    max_tree_depth: float | None
    is_acceptable: bool


@dataclass(frozen=True, slots=True)
class ModelHealth:
    """Overall model health assessment."""
    status: str  # "PASS", "WARNING", "FAIL"
    diagnostics: ConvergenceDiagnostics
    messages: tuple[str, ...]
    
    @property
    def is_decision_grade(self) -> bool:
        return self.status == "PASS"


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Report from raw data validation."""
    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    row_count: int
    column_count: int
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]


# =============================================================================
# UTILITIES
# =============================================================================

def validate_raw_columns(df: pd.DataFrame) -> ValidationReport:
    """Validate that DataFrame has exactly the required raw columns."""
    import pandas as pd
    
    missing = set(RAW_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(RAW_COLUMNS)
    
    errors = []
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    if extra:
        errors.append(f"Unexpected columns: {sorted(extra)}")
    
    # Check numeric columns are actually numeric
    for col in NUMERIC_RAW_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric")
    
    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(),
        row_count=len(df),
        column_count=len(df.columns),
        missing_columns=tuple(sorted(missing)),
        extra_columns=tuple(sorted(extra)),
    )


def validate_scenario_result(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a scenario result DataFrame contains all required columns."""
    missing = set(SCENARIO_RESULT_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Scenario result missing columns: {sorted(missing)}")
    return df


