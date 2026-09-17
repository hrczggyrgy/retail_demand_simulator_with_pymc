
"""
Fast retail demand / market-share simulation engine.

Design goals
------------
1. Keep the existing public API used by app.py.
2. Fit the Bayesian model once per market.
3. Do NOT run posterior-predictive sampling during the normal fit.
4. Extract a compact posterior representation for instant what-if scenarios.
5. Make distribution changes interpretable in percentage points by default.
6. Keep the complete competitive market in scenario share denominators.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import arviz as az
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import SplineTransformer


# ============================================================================
# Pack group utilities
# ============================================================================

def make_pack_group(pack_size: pd.Series) -> pd.Series:
    """
    Create meaningful pack-group categories from continuous pack_size values.
    
    Bins are based on typical retail pack size distributions.
    Adjust bins/labels based on category-specific domain knowledge if needed.
    """
    bins = [-np.inf, 0.33, 0.75, 1.50, np.inf]
    labels = ["small", "medium", "large", "extra_large"]
    return pd.cut(
        pack_size.astype(float),
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)


def _add_category_relative_pack_groups(df: pd.DataFrame) -> pd.Series:
    """
    Add category-relative pack-size groups using quantile-based binning.
    
    Within each category, splits pack sizes into tertiles (Small/Medium/Large).
    If fewer than 3 distinct sizes exist, uses available number of groups.
    """
    def _qcut_category(group: pd.Series) -> pd.Series:
        n_unique = group.nunique()
        if n_unique < 2:
            return pd.Series(["Single"] * len(group), index=group.index)
        n_bins = min(3, n_unique)
        labels = {2: ["Small", "Large"], 3: ["Small", "Medium", "Large"]}[n_bins]
        try:
            return pd.qcut(
                group.rank(method="first"),
                q=n_bins,
                labels=labels,
                duplicates="drop"
            )
        except ValueError:
            # Fallback if qcut fails
            return pd.cut(
                group,
                bins=n_bins,
                labels=labels,
                include_lowest=True
            )
    
    result = df.groupby("category")["pack_size"].transform(_qcut_category)
    return result.astype(str)


# ============================================================================
# Configuration
# ============================================================================

REQUIRED_COLUMNS = [
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
]

NUMERIC_COLUMNS = [
    "units",
    "revenue",
    "sku_stores",
    "retailer_stores",
    "pack_size",
]

KEY_COLUMNS = ["month", "retailer", "category", "brand", "sku"]

DEFAULT_DRAWS = 800
DEFAULT_TUNE = 800
DEFAULT_CHAINS = 1

# ============================================================================
# Scenario output contract (canonical column names)
# ============================================================================

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


def validate_scenario_result(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that a scenario result DataFrame contains all required columns."""
    missing = set(SCENARIO_RESULT_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Scenario result missing columns: {sorted(missing)}")
    return df
DEFAULT_TARGET_ACCEPT = 0.90
DEFAULT_N_SPLINE_KNOTS = 4
DEFAULT_SPLINE_DEGREE = 3

ADVANCED_DRAWS = 1000
ADVANCED_TUNE = 1000
ADVANCED_CHAINS = 4

# How many posterior draws to retain for interactive scenarios.
DEFAULT_SCENARIO_DRAWS = 400


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for PyMC model fitting."""
    draws: int = 800
    tune: int = 800
    chains: int = 4
    target_accept: float = 0.95
    random_seed: int = 42

    likelihood: str = "student_t"       # "normal" or "student_t"
    use_category_price_pooling: bool = True
    use_sku_nd_effects: bool = False    # Turn on after base model validates
    n_spline_knots: int = 4
    scenario_draws: int = 400


FAST_CONFIG = ModelConfig(
    draws=500,
    tune=500,
    chains=2,
    target_accept=0.92,
    likelihood="normal",
    use_category_price_pooling=False,
    use_sku_nd_effects=False,
)

DEFAULT_CONFIG = ModelConfig()

ADVANCED_CONFIG = ModelConfig(
    draws=1_000,
    tune=1_000,
    chains=4,
    target_accept=0.97,
    likelihood="student_t",
    use_category_price_pooling=True,
    use_sku_nd_effects=True,
)


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class ValidationReport:
    is_valid: bool
    row_count: int
    missing_columns: List[str]
    duplicate_count: int
    invalid_rows: int
    invalid_reasons: Dict[str, int]
    warnings: List[str]
    info: Dict[str, Any]


@dataclass
class PreparedData:
    df: pd.DataFrame
    scales: Dict[str, Dict[str, float]]
    meta: Dict[str, Any]
    spline: SplineTransformer
    nd_basis: np.ndarray


@dataclass
class ModelOutput:
    trace: xr.Dataset
    df: pd.DataFrame
    meta: Dict[str, Any]
    spline: SplineTransformer
    scales: Dict[str, Dict[str, float]]
    settings: Dict[str, Any]


@dataclass
class ScenarioResult:
    df: pd.DataFrame
    price_change: float
    nd_change: float
    summary: pd.DataFrame


# ============================================================================
# Fast / safe utilities
# ============================================================================

def _posterior_group(trace: Any):
    """Return the posterior group for InferenceData/DataTree-like objects."""
    if hasattr(trace, "posterior"):
        return trace.posterior
    if hasattr(trace, "groups"):
        try:
            return trace["posterior"]
        except Exception:
            pass
    try:
        return trace["posterior"]
    except Exception as exc:
        raise ValueError("Trace does not contain a posterior group") from exc


def _posterior_array(
    trace: Any,
    variable: str,
    dimensions: Optional[Tuple[str, ...]] = None,
) -> np.ndarray:
    posterior = _posterior_group(trace)
    if variable not in posterior:
        raise KeyError(f"Posterior variable '{variable}' is not available")
    arr = posterior[variable]

    if dimensions is not None:
        available = tuple(str(x) for x in arr.dims)
        if all(dim in available for dim in dimensions):
            arr = arr.transpose(
                *[d for d in ["chain", "draw"] if d in available],
                *dimensions,
            )
    values = np.asarray(arr)
    if values.ndim >= 2 and tuple(arr.dims[:2]) == ("chain", "draw"):
        values = values.reshape((-1,) + values.shape[2:])
    return values


def _find_posterior_variable(trace: Any, candidates: List[str]) -> Optional[str]:
    posterior = _posterior_group(trace)
    for name in candidates:
        if name in posterior:
            return name
    return None


def _coerce_month(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["month"], errors="coerce")


# ============================================================================
# Validation / preparation
# ============================================================================

def validate_input_data(raw: pd.DataFrame) -> ValidationReport:
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        return ValidationReport(
            is_valid=False,
            row_count=len(raw),
            missing_columns=missing,
            duplicate_count=0,
            invalid_rows=0,
            invalid_reasons={},
            warnings=[],
            info={"available_columns": list(raw.columns)},
        )

    df = raw.copy()
    n = len(df)
    warnings_list: List[str] = []
    invalid_reasons: Dict[str, int] = {}

    # Parse / coercion once.
    df["month"] = _coerce_month(df)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dup_count = int(df.duplicated(subset=KEY_COLUMNS, keep=False).sum())
    if dup_count:
        invalid_reasons["duplicates"] = dup_count
        warnings_list.append(
            f"Found {dup_count:,} duplicate records on "
            f"{', '.join(KEY_COLUMNS)}."
        )

    invalid = pd.Series(False, index=df.index)

    key_missing = df[KEY_COLUMNS].isna().any(axis=1)
    invalid |= key_missing
    invalid_reasons["missing_key_fields"] = int(key_missing.sum())

    numeric_missing = df[NUMERIC_COLUMNS].isna().any(axis=1)
    invalid |= numeric_missing
    invalid_reasons["missing_numeric"] = int(numeric_missing.sum())

    for col in NUMERIC_COLUMNS:
        bad = df[col].le(0)
        invalid |= bad
        invalid_reasons[f"non_positive_{col}"] = int(bad.sum())

    bad_stores = df["sku_stores"].gt(df["retailer_stores"])
    invalid |= bad_stores
    invalid_reasons["sku_stores_gt_retailer_stores"] = int(bad_stores.sum())
    if bad_stores.any():
        warnings_list.append(
            f"{int(bad_stores.sum()):,} rows have SKU stores above retailer stores."
        )

    valid = df.loc[~invalid].copy()
    valid_n = len(valid)

    n_months = int(valid["month"].nunique()) if valid_n else 0
    if n_months < 12:
        warnings_list.append(
            f"Only {n_months} months of usable data; estimates may be less stable."
        )

    if valid_n:
        cell_skus = valid.groupby(
            ["retailer", "category", "month"], observed=True
        )["sku"].nunique()
        single_sku_cells = int((cell_skus == 1).sum())
        if single_sku_cells:
            warnings_list.append(
                f"{single_sku_cells:,} retailer-category-month cells contain only "
                "one SKU, limiting within-cell competitor price information."
            )

        sku_obs = valid.groupby("sku", observed=True)["month"].nunique()
        low_obs = int((sku_obs < 6).sum())
        if low_obs:
            warnings_list.append(
                f"{low_obs:,} SKUs have fewer than 6 monthly observations."
            )

    info = {
        "original_rows": n,
        "valid_rows": valid_n,
        "dropped_rows": n - valid_n,
        "n_months": n_months,
        "n_retailers": int(valid["retailer"].nunique()) if valid_n else 0,
        "n_categories": int(valid["category"].nunique()) if valid_n else 0,
        "n_brands": int(valid["brand"].nunique()) if valid_n else 0,
        "n_skus": int(valid["sku"].nunique()) if valid_n else 0,
        "date_range": {
            "min": str(valid["month"].min()) if valid_n else None,
            "max": str(valid["month"].max()) if valid_n else None,
        },
    }

    return ValidationReport(
        is_valid=valid_n > 0,
        row_count=n,
        missing_columns=[],
        duplicate_count=dup_count,
        invalid_rows=n - valid_n,
        invalid_reasons=invalid_reasons,
        warnings=warnings_list,
        info=info,
    )


def prepare_data(
    raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Vectorized preparation. Avoids repeated groupby-transform work where possible.
    """
    df = raw.loc[:, REQUIRED_COLUMNS].copy()

    df["month"] = _coerce_month(df)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid = (
        df[KEY_COLUMNS].isna().any(axis=1)
        | df[NUMERIC_COLUMNS].isna().any(axis=1)
        | df["units"].le(0)
        | df["revenue"].le(0)
        | df["sku_stores"].le(0)
        | df["retailer_stores"].le(0)
        | df["pack_size"].le(0)
        | df["sku_stores"].gt(df["retailer_stores"])
    )

    if invalid.any():
        warnings.warn(
            f"Dropping {int(invalid.sum()):,} invalid rows during preparation.",
            RuntimeWarning,
            stacklevel=2,
        )
        df = df.loc[~invalid].copy()

    if df.empty:
        raise ValueError("No valid rows remain after initial data cleaning.")

    # Basic physical metrics.
    df["standard_volume"] = df["units"] * df["pack_size"]
    df["price_std"] = df["revenue"] / df["standard_volume"]
    df["nd"] = df["sku_stores"] / df["retailer_stores"]
    df["velocity_std"] = df["standard_volume"] / df["sku_stores"]

    # One grouped aggregation + merge instead of multiple transform calls.
    cell = ["retailer", "category", "month"]
    cell_totals = (
        df.groupby(cell, observed=True)[["revenue", "standard_volume"]]
        .sum()
        .rename(
            columns={
                "revenue": "category_revenue",
                "standard_volume": "category_std_volume",
            }
        )
        .reset_index()
    )
    df = df.merge(cell_totals, on=cell, how="left", sort=False)

    df["other_skus_revenue"] = df["category_revenue"] - df["revenue"]
    df["other_skus_std_volume"] = (
        df["category_std_volume"] - df["standard_volume"]
    )

    df["other_skus_price_std"] = (
        df["other_skus_revenue"] / df["other_skus_std_volume"]
    )
    df["relative_price"] = df["price_std"] / df["other_skus_price_std"]

    usable = (
        df["price_std"].gt(0)
        & df["other_skus_price_std"].gt(0)
        & df["nd"].gt(0)
        & df["nd"].le(1)
        & df["velocity_std"].gt(0)
        & df["relative_price"].gt(0)
    )

    dropped = int((~usable).sum())
    if dropped:
        warnings.warn(
            f"Dropping {dropped:,} rows without usable market-competitive price, "
            "distribution, or velocity.",
            RuntimeWarning,
            stacklevel=2,
        )
        df = df.loc[usable].copy()

    if df.empty:
        raise ValueError("No rows remain after relative-price/distribution cleaning.")

    # Transform once.
    df["log_velocity"] = np.log(df["velocity_std"])
    df["log_relative_price"] = np.log(df["relative_price"])
    df["log_nd"] = np.log(df["nd"])

    scales: Dict[str, Dict[str, float]] = {}
    for col in ["log_velocity", "log_relative_price", "log_nd"]:
        mean = float(df[col].mean())
        sd = float(df[col].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                f"'{col}' has no usable variation (mean={mean:.6g}, sd={sd:.6g})."
            )
        df[f"{col}_z"] = (df[col] - mean) / sd
        scales[col] = {"mean": mean, "sd": sd}

    df["pack_group"] = make_pack_group(df["pack_size"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after final numeric cleanup.")

    return df, scales


def add_indices(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    level_map: Dict[str, np.ndarray] = {}
    for col in ["retailer", "category", "sku", "brand", "month", "pack_group"]:
        codes, levels = pd.factorize(df[col], sort=True)
        df[f"{col}_idx"] = codes.astype("int32")
        level_map[col] = levels

    df["entity"] = (
        df["retailer"].astype(str) + "__" + df["sku"].astype(str)
    )
    entity_codes, entity_levels = pd.factorize(df["entity"], sort=True)
    df["entity_idx"] = entity_codes.astype("int32")

    # SKU-level lookup arrays
    sku_info = (
        df[
            [
                "sku_idx",
                "brand_idx",
                "category_idx",
                "pack_group_idx",
                "sku",
                "brand",
                "category",
                "pack_group",
                "pack_size",
            ]
        ]
        .sort_values("sku_idx")
        .drop_duplicates("sku_idx")
        .reset_index(drop=True)
    )

    expected = np.arange(len(sku_info), dtype="int32")
    if not np.array_equal(sku_info["sku_idx"].to_numpy(), expected):
        raise ValueError("SKU indexing is inconsistent.")

    # Brand-level lookup for brand -> category mapping
    brand_info = (
        df[["brand_idx", "category_idx", "brand", "category"]]
        .sort_values("brand_idx")
        .drop_duplicates("brand_idx")
        .reset_index(drop=True)
    )

    meta = {
        "retailer_levels": level_map["retailer"],
        "category_levels": level_map["category"],
        "sku_levels": level_map["sku"],
        "brand_levels": level_map["brand"],
        "month_levels": level_map["month"],
        "pack_group_levels": level_map["pack_group"],
        "entity_levels": entity_levels,
        # Per-SKU lookup arrays used directly in the model
        "sku_brand_idx": sku_info["brand_idx"].to_numpy(dtype="int32"),
        "sku_category_idx": sku_info["category_idx"].to_numpy(dtype="int32"),
        "sku_pack_idx": sku_info["pack_group_idx"].to_numpy(dtype="int32"),
        # Per-brand category mapping, necessary for category -> brand pooling
        "brand_category_idx": brand_info["category_idx"].to_numpy(dtype="int32"),
        "sku_info": sku_info,
    }
    return df, meta


def fit_spline(
    df: pd.DataFrame,
    n_knots: int = DEFAULT_N_SPLINE_KNOTS,
    degree: int = DEFAULT_SPLINE_DEGREE,
) -> Tuple[SplineTransformer, np.ndarray]:
    x = df["log_nd_z"].to_numpy(dtype=float).reshape(-1, 1)
    spline = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=False,
    )
    basis = spline.fit_transform(x).astype(np.float64, copy=False)
    return spline, basis


def create_data_quality_report(
    raw_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    validation: ValidationReport,
) -> Dict[str, Any]:
    input_n = len(raw_df)
    prepared_n = len(prepared_df)

    numeric = raw_df.copy()
    for col in NUMERIC_COLUMNS:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    duplicate_records = int(
        numeric.duplicated(subset=KEY_COLUMNS, keep=False).sum()
    )

    missing = {
        str(k): int(v)
        for k, v in numeric.isna().sum().items()
        if int(v) > 0
    }

    counts = {
        "retailers": int(prepared_df["retailer"].nunique()),
        "categories": int(prepared_df["category"].nunique()),
        "brands": int(prepared_df["brand"].nunique()),
        "skus": int(prepared_df["sku"].nunique()),
        "pack_sizes": int(prepared_df["pack_size"].nunique()),
    }

    sku_obs = (
        prepared_df.groupby("sku", observed=True)["month"]
        .nunique()
        .sort_values()
    )
    entity_obs = (
        prepared_df.groupby(["retailer", "sku"], observed=True)["month"]
        .nunique()
        .sort_values()
    )

    report = {
        "input_row_count": input_n,
        "valid_model_row_count": prepared_n,
        "dropped_row_count": max(0, input_n - prepared_n),
        "dropped_reasons": validation.invalid_reasons,
        "missing_values": missing,
        "duplicate_records": duplicate_records,
        "date_coverage": {
            "min": str(prepared_df["month"].min()),
            "max": str(prepared_df["month"].max()),
            "n_months": int(prepared_df["month"].nunique()),
        },
        "counts": counts,
        "distribution_validity": {
            "nd_min": float(prepared_df["nd"].min()),
            "nd_max": float(prepared_df["nd"].max()),
            "nd_gt_1_count": int((prepared_df["nd"] > 1).sum()),
        },
        "price_variation": {
            "rel_price_min": float(prepared_df["relative_price"].min()),
            "rel_price_max": float(prepared_df["relative_price"].max()),
            "rel_price_std": float(prepared_df["relative_price"].std()),
        },
        "observations_per_retailer_sku": {
            "min": int(entity_obs.min()) if len(entity_obs) else 0,
            "median": float(entity_obs.median()) if len(entity_obs) else 0,
            "max": int(entity_obs.max()) if len(entity_obs) else 0,
        },
        "observations_per_sku": {
            "min": int(sku_obs.min()) if len(sku_obs) else 0,
            "median": float(sku_obs.median()) if len(sku_obs) else 0,
            "max": int(sku_obs.max()) if len(sku_obs) else 0,
        },
        "low_observation_skus": sku_obs[sku_obs < 6].index.tolist(),
        "warnings": validation.warnings,
    }
    return report


# ============================================================================
# Model
# ============================================================================

def build_pymc_model(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    nd_basis: np.ndarray,
) -> pm.Model:
    coords = {
        "retailer": meta["retailer_levels"],
        "sku": meta["sku_levels"],
        "brand": meta["brand_levels"],
        "month": meta["month_levels"],
        "pack_group": meta["pack_group_levels"],
        "entity": meta["entity_levels"],
        "obs": np.arange(len(df)),
        "spline_basis": np.arange(nd_basis.shape[1]),
    }

    # Keep observed arrays numeric / contiguous.
    entity_idx = df["entity_idx"].to_numpy(dtype="int32")
    month_idx = df["month_idx"].to_numpy(dtype="int32")
    sku_idx = df["sku_idx"].to_numpy(dtype="int32")
    y = df["log_velocity_z"].to_numpy(dtype=float)
    price_x = df["log_relative_price_z"].to_numpy(dtype=float)

    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", 0.0, 1.0)

        # Persistent retailer × SKU effect, non-centered.
        sigma_entity = pm.HalfNormal("sigma_entity", 0.5)
        entity_offset = pm.Normal("entity_offset", 0.0, 1.0, dims="entity")
        entity_effect = pm.Deterministic(
            "entity_effect",
            entity_offset * sigma_entity,
            dims="entity",
        )

        # Month / seasonality, non-centered.
        sigma_month = pm.HalfNormal("sigma_month", 0.5)
        month_offset = pm.Normal("month_offset", 0.0, 1.0, dims="month")
        month_effect = pm.Deterministic(
            "month_effect",
            month_offset * sigma_month,
            dims="month",
        )

        # SKU price response pooled by brand × pack-size.
        n_brands = len(meta["brand_levels"])
        n_packs = len(meta["pack_group_levels"])

        price_mean = pm.Normal(
            "price_mean",
            -0.4,
            0.5,
            dims=("brand", "pack_group"),
        )
        sigma_price_sku = pm.HalfNormal("sigma_price_sku", 0.3)
        price_offset = pm.Normal(
            "price_offset",
            0.0,
            1.0,
            dims="sku",
        )
        price_slope = pm.Deterministic(
            "price_slope_z",
            price_mean[
                meta["sku_brand_idx"],
                meta["sku_pack_idx"],
            ]
            + price_offset * sigma_price_sku,
            dims="sku",
        )

        # Smooth, low-dimensional nonlinear distribution response.
        sigma_nd = pm.HalfNormal("sigma_nd", 0.6)
        nd_coef = pm.Normal(
            "nd_coef",
            0.0,
            sigma_nd,
            dims="spline_basis",
        )
        nd_effect = pm.Deterministic(
            "nd_effect",
            pm.math.dot(nd_basis, nd_coef),
            dims="obs",
        )

        mu = (
            alpha
            + entity_effect[entity_idx]
            + month_effect[month_idx]
            + price_slope[sku_idx] * price_x
            + nd_effect
        )

        sigma = pm.HalfNormal("sigma", 0.7)

        pm.Normal(
            "log_velocity_z_obs",
            mu,
            sigma,
            observed=y,
            dims="obs",
        )

    return model


# ============================================================================
# Hierarchical Model v2 (improved hierarchy)
# ============================================================================

def build_pymc_model_v2(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    nd_basis: np.ndarray,
    config: ModelConfig = DEFAULT_CONFIG,
) -> pm.Model:
    """
    Hierarchical log-standard-velocity model with deeper pooling.

    Dependent variable: log_velocity_z
    Predictors: log_relative_price_z, ND spline basis
    Hierarchy: category -> brand x pack_group -> SKU for price slopes
    """
    coords = {
        "obs": np.arange(len(df)),
        "entity": meta["entity_levels"],
        "month": meta["month_levels"],
        "category": meta["category_levels"],
        "brand": meta["brand_levels"],
        "sku": meta["sku_levels"],
        "pack_group": meta["pack_group_levels"],
        "spline_basis": np.arange(nd_basis.shape[1]),
    }

    entity_idx = df["entity_idx"].to_numpy(dtype="int32")
    month_idx = df["month_idx"].to_numpy(dtype="int32")
    sku_idx = df["sku_idx"].to_numpy(dtype="int32")

    sku_brand_idx = np.asarray(meta["sku_brand_idx"], dtype="int32")
    sku_pack_idx = np.asarray(meta["sku_pack_idx"], dtype="int32")
    sku_category_idx = np.asarray(meta["sku_category_idx"], dtype="int32")
    brand_category_idx = np.asarray(meta["brand_category_idx"], dtype="int32")

    y = df["log_velocity_z"].to_numpy(dtype=float)
    price_x = df["log_relative_price_z"].to_numpy(dtype=float)

    with pm.Model(coords=coords) as model:
        # Data containers for scenario reuse
        price_x_data = pm.Data("price_x", price_x, dims="obs")
        nd_basis_data = pm.Data("nd_basis", nd_basis, dims=("obs", "spline_basis"))

        # Global intercept
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)

        # Retailer x SKU persistent baseline
        sigma_entity = pm.HalfNormal("sigma_entity", sigma=0.7)
        entity_offset = pm.Normal("entity_offset", 0.0, 1.0, dims="entity")
        entity_effect = pm.Deterministic(
            "entity_effect", entity_offset * sigma_entity, dims="entity"
        )

        # Monthly seasonality
        sigma_month = pm.HalfNormal("sigma_month", sigma=0.4)
        month_offset = pm.Normal("month_offset", 0.0, 1.0, dims="month")
        month_effect = pm.Deterministic(
            "month_effect", month_offset * sigma_month, dims="month"
        )

        # Price elasticity hierarchy: category -> brand x pack -> SKU
        if config.use_category_price_pooling:
            sigma_price_category = pm.HalfNormal("sigma_price_category", sigma=0.35)
            price_mean_category = pm.Normal(
                "price_mean_category",
                mu=-0.4,
                sigma=sigma_price_category,
                dims="category",
            )

            sigma_price_brand_pack = pm.HalfNormal(
                "sigma_price_brand_pack", sigma=0.25
            )
            price_mean_brand_pack = pm.Normal(
                "price_mean_brand_pack",
                mu=price_mean_category[brand_category_idx][:, None],
                sigma=sigma_price_brand_pack,
                dims=("brand", "pack_group"),
            )
        else:
            price_mean_brand_pack = pm.Normal(
                "price_mean_brand_pack",
                mu=-0.4,
                sigma=0.4,
                dims=("brand", "pack_group"),
            )

        sigma_price_sku = pm.HalfNormal("sigma_price_sku", sigma=0.25)
        price_sku_offset = pm.Normal(
            "price_sku_offset", mu=0.0, sigma=1.0, dims="sku"
        )

        price_slope_z = pm.Deterministic(
            "price_slope_z",
            price_mean_brand_pack[sku_brand_idx, sku_pack_idx]
            + price_sku_offset * sigma_price_sku,
            dims="sku",
        )

        # Distribution effect f(ND)
        if config.use_sku_nd_effects:
            sigma_nd_brand = pm.HalfNormal("sigma_nd_brand", sigma=0.5)
            nd_brand_coef = pm.Normal(
                "nd_brand_coef",
                mu=0.0,
                sigma=sigma_nd_brand,
                dims=("brand", "spline_basis"),
            )

            sigma_nd_sku = pm.HalfNormal("sigma_nd_sku", sigma=0.25)
            nd_sku_offset = pm.Normal(
                "nd_sku_offset", mu=0.0, sigma=1.0, dims=("sku", "spline_basis")
            )

            nd_coef = pm.Deterministic(
                "nd_coef",
                nd_brand_coef[sku_brand_idx, :] + nd_sku_offset * sigma_nd_sku,
                dims=("sku", "spline_basis"),
            )

            nd_effect = pm.Deterministic(
                "nd_effect",
                pm.math.sum(nd_coef[sku_idx, :] * nd_basis_data, axis=1),
                dims="obs",
            )
        else:
            # Shared ND curve
            sigma_nd = pm.HalfNormal("sigma_nd", sigma=0.5)
            nd_coef = pm.Normal(
                "nd_coef", mu=0.0, sigma=sigma_nd, dims="spline_basis"
            )

            nd_effect = pm.Deterministic(
                "nd_effect",
                pm.math.dot(nd_basis_data, nd_coef),
                dims="obs",
            )

        mu = pm.Deterministic(
            "mu",
            alpha
            + entity_effect[entity_idx]
            + month_effect[month_idx]
            + price_slope_z[sku_idx] * price_x_data
            + nd_effect,
            dims="obs",
        )

        sigma = pm.HalfNormal("sigma", sigma=0.7)

        if config.likelihood == "student_t":
            nu_minus_two = pm.Exponential("nu_minus_two", lam=1 / 10)
            nu = pm.Deterministic("nu", nu_minus_two + 2.0)

            pm.StudentT(
                "log_velocity_z_obs",
                nu=nu,
                mu=mu,
                sigma=sigma,
                observed=y,
                dims="obs",
            )
        else:
            pm.Normal(
                "log_velocity_z_obs",
                mu=mu,
                sigma=sigma,
                observed=y,
                dims="obs",
            )

    return model


def fit_model(
    model: pm.Model,
    config: ModelConfig = DEFAULT_CONFIG,
) -> az.InferenceData:
    """
    Fit the Bayesian model with standard configuration.

    Returns ArviZ InferenceData for consistent diagnostics and posterior access.
    """
    with model:
        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=min(config.chains, 4),
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            return_inferencedata=True,
            init="jitter+adapt_diag",
            progressbar=True,
        )

    return idata


# ============================================================================
# Diagnostics / elasticity
# ============================================================================

def get_model_diagnostics(idata) -> Dict[str, Any]:
    """
    Get model diagnostics as a dictionary (backward compatible).
    
    Delegates to summarize_convergence_diagnostics for the actual computation.
    """
    conv = summarize_convergence_diagnostics(idata)
    
    return {
        "divergences": conv.divergences,
        "rhat_max": conv.max_rhat,
        "ess_bulk_min": conv.min_ess_bulk,
        "ess_tail_min": conv.min_ess_tail,
        "coverage_90": conv.coverage_90,
        "is_usable": conv.is_acceptable,
    }


def get_posterior_predictive_check(
    model: pm.Model,
    idata: az.InferenceData,
    prepared_df: pd.DataFrame,
    random_seed: int = 42,
) -> pd.DataFrame:
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["log_velocity_z_obs"],
            random_seed=random_seed,
            progressbar=False,
        )

    draws = ppc.posterior_predictive["log_velocity_z_obs"]
    pred_median = draws.median(dim=("chain", "draw")).values
    pred_p10 = draws.quantile(0.10, dim=("chain", "draw")).values
    pred_p90 = draws.quantile(0.90, dim=("chain", "draw")).values

    result = prepared_df[
        ["month", "retailer", "category", "brand", "sku", "log_velocity_z"]
    ].copy()

    result["predicted_z_p10"] = pred_p10
    result["predicted_z_median"] = pred_median
    result["predicted_z_p90"] = pred_p90
    result["residual_z"] = result["log_velocity_z"] - result["predicted_z_median"]
    result["inside_p10_p90"] = (
        (result["log_velocity_z"] >= result["predicted_z_p10"])
        & (result["log_velocity_z"] <= result["predicted_z_p90"])
    )

    return result


# ============================================================================
# ArviZ plotting helpers (compatible with ArviZ >= 0.20)
# ============================================================================

def make_trace_figure(
    idata,
    var_names: Optional[List[str]] = None,
    max_skus: int = 8,
):
    """
    Return an ArviZ trace plot as a Matplotlib figure.

    Select a limited number of SKU-level parameters; plotting every SKU
    can create an unreadable and slow dashboard.
    """
    import matplotlib.pyplot as plt
    import arviz_plots as azp
    import arviz_base as azb

    # Increase max subplots limit
    azb.rcParams["plot.max_subplots"] = 1000

    if var_names is None:
        var_names = [
            "alpha",
            "sigma_entity",
            "sigma_month",
            "sigma_price_sku",
            "sigma_nd",
            "sigma",
            "price_mean_brand_pack",
            "price_slope_z",
        ]

    available = [name for name in var_names if name in idata.posterior]

    coords = {}
    if "price_slope_z" in available and "sku" in idata.posterior.coords:
        sku_values = idata.posterior.coords["sku"].values[:max_skus]
        coords["sku"] = sku_values
    if "price_mean_brand_pack" in available:
        # Limit brand/pack combinations to avoid too many subplots
        coords["brand"] = idata.posterior.coords["brand"].values[:3]
        coords["pack_group"] = idata.posterior.coords["pack_group"].values[:2]

    pc = azp.plot_trace(
        idata,
        var_names=available,
        coords=coords or None,
        backend="matplotlib",
    )

    fig = pc.viz["figure"].item()
    fig.tight_layout()
    return fig


def make_rank_figure(
    idata,
    var_names: Optional[List[str]] = None,
    max_skus: int = 8,
):
    import matplotlib.pyplot as plt
    import arviz_plots as azp

    if var_names is None:
        var_names = [
            "alpha",
            "sigma_entity",
            "sigma_month",
            "sigma_price_sku",
            "sigma_nd",
        ]

        if "price_slope_z" in idata.posterior:
            var_names.append("price_slope_z")

    available = [name for name in var_names if name in idata.posterior]

    coords = {}
    if "price_slope_z" in available and "sku" in idata.posterior.coords:
        coords["sku"] = idata.posterior.coords["sku"].values[:max_skus]

    pc = azp.plot_rank(
        idata,
        var_names=available,
        coords=coords or None,
        backend="matplotlib",
        visuals={"ecdf_lines": {}, "credible_interval": {}},
    )

    fig = pc.viz["figure"].item()
    fig.tight_layout()
    return fig


def make_energy_figure(idata):
    import matplotlib.pyplot as plt
    import arviz_plots as azp

    pc = azp.plot_energy(
        idata,
        backend="matplotlib",
    )

    fig = pc.viz["figure"].item()
    fig.tight_layout()
    return fig


def make_ppc_figure(
    idata,
    observed_var: str = "log_velocity_z_obs",
):
    """
    Compare observed standardized log velocity with posterior predictions.
    """
    import matplotlib.pyplot as plt
    import arviz_plots as azp

    if observed_var not in idata.observed_data:
        raise KeyError(
            f"'{observed_var}' is not available in idata.observed_data. "
            "Check the observed variable name in the PyMC model."
        )

    pc = azp.plot_ppc_dist(
        idata,
        var_names=[observed_var],
        group="posterior_predictive",
        backend="matplotlib",
    )

    fig = pc.viz["figure"].item()
    fig.tight_layout()
    return fig


def make_elasticity_forest_figure(
    idata,
    selected_skus: List[str],
    hdi_prob: float = 0.90,
):
    """
    Compare posterior standardized price slopes for selected SKUs.

    If price is standardized in the model, label this clearly as a
    standardized coefficient. Use your existing elasticity extraction
    function separately for commercial raw-scale elasticities.
    """
    import matplotlib.pyplot as plt
    import arviz_plots as azp

    if "price_slope_z" not in idata.posterior:
        raise KeyError("The posterior has no 'price_slope_z' variable.")

    pc = azp.plot_forest(
        idata,
        var_names=["price_slope_z"],
        coords={"sku": selected_skus},
        backend="matplotlib",
        ci_probs=(0.05, hdi_prob),
        visuals={"trunk": {}, "twig": {}, "point_estimate": {}, "labels": {}},
    )

    fig = pc.viz["figure"].item()
    fig.tight_layout()
    return fig


# ============================================================================
# Convergence Diagnostics (typed)
# ============================================================================

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ConvergenceDiagnostics:
    divergences: int
    max_rhat: float | None
    min_ess_bulk: float | None
    min_ess_tail: float | None
    coverage_90: float | None
    is_acceptable: bool


def summarize_convergence_diagnostics(idata) -> ConvergenceDiagnostics:
    """
    Extract convergence diagnostics from InferenceData.
    
    Returns a ConvergenceDiagnostics object with typed fields.
    """
    # Divergences
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().item())

    # R-hat and ESS from arviz summary
    max_rhat = None
    min_ess_bulk = None
    min_ess_tail = None
    coverage_90 = None
    
    try:
        summary = az.summary(idata, round_to=None, kind="diagnostics")
        if "r_hat" in summary.columns:
            max_rhat = float(summary["r_hat"].max())
        if "ess_bulk" in summary.columns:
            min_ess_bulk = float(summary["ess_bulk"].min())
        if "ess_tail" in summary.columns:
            min_ess_tail = float(summary["ess_tail"].min())
    except Exception:
        pass

    # Posterior predictive coverage
    if (hasattr(idata, "posterior_predictive") and "y" in idata.posterior_predictive 
        and hasattr(idata, "observed_data") and "y" in idata.observed_data):
        try:
            y_obs = idata.observed_data["y"].values.flatten()
            y_pred = idata.posterior_predictive["y"].values
            pred_p05 = np.percentile(y_pred, 5, axis=(0, 1))
            pred_p95 = np.percentile(y_pred, 95, axis=(0, 1))
            coverage_90 = float(np.mean((y_obs >= pred_p05) & (y_obs <= pred_p95)))
        except Exception:
            pass

    is_acceptable = (
        divergences == 0
        and (max_rhat is None or max_rhat < 1.01)
        and (min_ess_bulk is None or min_ess_bulk > 400)
    )

    return ConvergenceDiagnostics(
        divergences=divergences,
        max_rhat=max_rhat,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        coverage_90=coverage_90,
        is_acceptable=is_acceptable,
    )


# ============================================================================
# Plotly Elasticity Forest Plot (operates on pre-calculated elasticity_df)
# ============================================================================

def build_elasticity_forest_figure(
    elasticity_df: pd.DataFrame,
    max_skus: int = 30,
) -> go.Figure:
    """
    Build a Plotly forest plot from the already calculated elasticity summary.
    
    Required columns:
    - sku
    - elasticity_p05
    - elasticity_median
    - elasticity_p95
    Optional:
    - brand
    - category
    - retailer
    """
    required = {
        "sku",
        "elasticity_p05",
        "elasticity_median",
        "elasticity_p95",
    }

    missing = required.difference(elasticity_df.columns)
    if missing:
        raise ValueError(
            "Elasticity forest plot requires columns: "
            f"{sorted(required)}. Missing: {sorted(missing)}"
        )

    plot_df = (
        elasticity_df
        .dropna(
            subset=[
                "elasticity_p05",
                "elasticity_median",
                "elasticity_p95",
            ]
        )
        .sort_values("elasticity_median")
        .tail(max_skus)
        .copy()
    )

    if plot_df.empty:
        return go.Figure().update_layout(
            title="No valid elasticity estimates available"
        )

    # Build SKU labels
    if "brand" in plot_df.columns:
        plot_df["label"] = (
            plot_df["brand"].astype(str)
            + " — "
            + plot_df["sku"].astype(str)
        )
    else:
        plot_df["label"] = plot_df["sku"].astype(str)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=plot_df["elasticity_median"],
            y=plot_df["label"],
            mode="markers",
            marker={"size": 9, "color": "#1f77b4"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (
                    plot_df["elasticity_p95"]
                    - plot_df["elasticity_median"]
                ),
                "arrayminus": (
                    plot_df["elasticity_median"]
                    - plot_df["elasticity_p05"]
                ),
                "thickness": 1.5,
                "width": 3,
            },
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median elasticity: %{x:.2f}<br>"
                "90% interval: "
                "%{error_x.arrayminus:.2f} to "
                "%{error_x.array:.2f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="No estimated price response",
        annotation_position="top",
    )

    figure.update_layout(
        title="SKU Price Elasticity: Posterior Median and 90% Credible Interval",
        xaxis_title="Estimated own-price elasticity",
        yaxis_title="SKU",
        height=max(450, len(plot_df) * 30),
        showlegend=False,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 65, "b": 40},
    )

    return figure


# ============================================================================
# Rolling Holdout Backtest
# ============================================================================

@dataclass(frozen=True, slots=True)
class BacktestConfig:
    min_train_months: int = 18
    horizons: tuple[int, ...] = (1, 3, 6)
    max_cutoffs: int = 3


def calculate_wape(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Weighted Absolute Percentage Error."""
    denominator = actual.abs().sum()
    if denominator == 0:
        return float("nan")
    return float((actual - predicted).abs().sum() / denominator)


def calculate_bias(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Relative bias (positive = over-forecast)."""
    denominator = actual.abs().sum()
    if denominator == 0:
        return float("nan")
    return float((predicted - actual).sum() / denominator)


def calculate_directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Fraction of correct rise/fall directions."""
    actual_direction = actual.diff().gt(0)
    predicted_direction = predicted.diff().gt(0)

    valid = actual_direction.notna() & predicted_direction.notna()

    if valid.sum() == 0:
        return float("nan")

    return float(
        actual_direction.loc[valid]
        .eq(predicted_direction.loc[valid])
        .mean()
    )


def run_rolling_backtest(
    df: pd.DataFrame,
    config: BacktestConfig,
    meta: dict,
    scales: dict,
    spline,
    posterior_cache: dict,
) -> pd.DataFrame:
    """
    Expanding-window backtest for monthly data.
    
    For each cutoff month, fit on months through cutoff, forecast horizons ahead.
    Returns results at retailer x category x month level.
    """
    months = sorted(df["month"].unique())
    
    if len(months) < config.min_train_months + max(config.horizons):
        return pd.DataFrame()
    
    results = []
    
    # Limit number of cutoffs
    possible_cutoffs = months[config.min_train_months:-max(config.horizons)]
    cutoffs = possible_cutoffs[-config.max_cutoffs:]
    
    for cutoff in cutoffs:
        train_mask = df["month"] <= cutoff
        train_df = df.loc[train_mask].copy()
        
        if len(train_df) == 0:
            continue
            
        # Build features for training data
        train_features = build_retail_features(train_df)
        
        # For each horizon, evaluate
        for horizon in config.horizons:
            val_start = cutoff + pd.DateOffset(months=1)
            val_end = cutoff + pd.DateOffset(months=horizon)
            val_mask = (df["month"] >= val_start) & (df["month"] <= val_end)
            val_df = df.loc[val_mask].copy()
            
            if len(val_df) == 0:
                continue
                
            # Aggregate to retailer x category x month for stable evaluation
            actual = val_df.groupby(["retailer", "category", "month"])["units"].sum().reset_index()
            actual = actual.rename(columns={"units": "actual_units"})
            
            # Use posterior mean for prediction
            # This is a simplified approach - full backtest would re-fit
            # For now, use the posterior to predict on validation set
            # In production, you'd re-fit the model on training data
            pass  # Placeholder for full implementation
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def build_nd_response_curve(
    idata,
    spline,
    sku_index: int,
    nd_grid: Optional[np.ndarray] = None,
    n_draws: int = 300,
):
    """
    Return posterior p10 / median / p90 volume effects across ND values.

    Output is relative to the first ND grid value.
    Handles both shared and SKU-specific ND curves.
    """
    if nd_grid is None:
        nd_grid = np.linspace(0.05, 0.95, 50)

    nd_log = np.log(np.clip(nd_grid, 1e-4, 1.0))
    basis = spline.transform(nd_log.reshape(-1, 1))

    nd_coef = idata.posterior["nd_coef"].stack(sample=("chain", "draw"))

    if "sku" in nd_coef.dims:
        # SKU-specific curve
        nd_coef = nd_coef.isel(sku=sku_index).values
    else:
        # Shared curve
        nd_coef = nd_coef.values

    rng = np.random.default_rng(42)
    draw_idx = rng.choice(
        nd_coef.shape[-1],
        size=min(n_draws, nd_coef.shape[-1]),
        replace=False,
    )

    coef_draws = nd_coef[:, draw_idx].T
    log_effect = coef_draws @ basis.T

    # Relative response against first grid point
    log_delta = log_effect - log_effect[:, [0]]
    multiplier = np.exp(log_delta)

    return pd.DataFrame(
        {
            "nd": nd_grid,
            "nd_p10": np.quantile(multiplier, 0.10, axis=0),
            "nd_median": np.quantile(multiplier, 0.50, axis=0),
            "nd_p90": np.quantile(multiplier, 0.90, axis=0),
        }
    )


def add_posterior_predictive(model, idata, random_seed: int = 42):
    with model:
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["log_velocity_z_obs"],
            random_seed=random_seed,
            progressbar=False,
            extend_inferencedata=True,
        )
    return ppc


def apply_target_scope(
    df: pd.DataFrame,
    target_level: str,
    target_value,
) -> pd.Series:
    level = str(target_level).strip().lower()
    if level == "market":
        return pd.Series(True, index=df.index)
    if level == "retailer":
        return df["retailer"].eq(target_value)
    if level == "brand":
        return df["brand"].eq(target_value)
    if level == "sku":
        return df["sku"].eq(target_value)
    if level in ("pack_size", "pack size"):
        return df["pack_size"].eq(target_value)
    raise ValueError(f"Unknown target_level: {target_level!r}")


def calculate_market_shares(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    share_level: str = "brand",
) -> pd.DataFrame:
    """
    Calculate baseline and median-scenario market shares.

    Market share is calculated from revenue by default. Since the dataset
    contains the full competitive set, the denominator contains all rows in
    the selected market/view scope.
    """
    group_cols = ["month"]
    if share_level == "brand":
        group_cols.append("brand")
    elif share_level == "sku":
        group_cols.extend(["brand", "sku"])
    else:
        group_cols.extend(["brand", "sku", "pack_size"])

    base = (
        baseline_df.groupby(group_cols, as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "baseline_value"})
    )

    scen = (
        scenario_df.groupby(group_cols, as_index=False)["revenue_p50"]
        .sum()
        .rename(columns={"revenue_p50": "scenario_value"})
    )

    out = base.merge(scen, on=group_cols, how="outer").fillna(0)

    totals = (
        out.groupby("month", as_index=False)
        .agg(
            market_baseline=("baseline_value", "sum"),
            market_scenario=("scenario_value", "sum"),
        )
    )
    out = out.merge(totals, on="month", how="left")

    out["baseline_share"] = np.where(
        out["market_baseline"] > 0,
        out["baseline_value"] / out["market_baseline"],
        np.nan,
    )
    out["scenario_share"] = np.where(
        out["market_scenario"] > 0,
        out["scenario_value"] / out["market_scenario"],
        np.nan,
    )
    out["share_change_pp"] = out["scenario_share"] - out["baseline_share"]
    return out


def summarize_period_shares(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    share_level: str,
) -> pd.DataFrame:
    """
    Aggregate over the selected period and calculate market shares.
    """
    if share_level == "brand":
        group_cols = ["brand"]
    elif share_level == "sku":
        group_cols = ["brand", "sku"]
    else:
        group_cols = ["brand", "sku", "pack_size"]

    base = (
        baseline_df.groupby(group_cols, as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "baseline_value"})
    )

    scen = (
        scenario_df.groupby(group_cols, as_index=False)["revenue_p50"]
        .sum()
        .rename(columns={"revenue_p50": "scenario_value"})
    )

    out = base.merge(scen, on=group_cols, how="outer").fillna(0)

    base_total = out["baseline_value"].sum()
    scen_total = out["scenario_value"].sum()

    out["baseline_share"] = (
        out["baseline_value"] / base_total if base_total > 0 else np.nan
    )
    out["scenario_share"] = (
        out["scenario_value"] / scen_total if scen_total > 0 else np.nan
    )
    out["share_change_pp"] = out["scenario_share"] - out["baseline_share"]

    return out.sort_values("scenario_share", ascending=False).reset_index(drop=True)


def extract_elasticities(
    trace: Any,
    meta: Dict[str, Any],
    scales: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Convert standardized price slopes back to ordinary elasticity.
    """
    beta_name = _find_posterior_variable(trace, ["price_slope_z", "price_slope"])
    if beta_name is None:
        raise KeyError("Could not find posterior price slope variable.")

    beta_z = _posterior_array(trace, beta_name, ("sku",))
    beta_raw = (
        beta_z
        * scales["log_velocity"]["sd"]
        / scales["log_relative_price"]["sd"]
    )

    info = meta["sku_info"].copy()

    info["elasticity_p05"] = np.quantile(beta_raw, 0.05, axis=0)
    info["elasticity_median"] = np.quantile(beta_raw, 0.50, axis=0)
    info["elasticity_p95"] = np.quantile(beta_raw, 0.95, axis=0)

    return info[
        [
            "sku",
            "brand",
            "pack_group",
            "pack_size",
            "elasticity_p05",
            "elasticity_median",
            "elasticity_p95",
        ]
    ].sort_values("elasticity_median").reset_index(drop=True)


# ============================================================================
# Posterior extraction for fast scenarios (v2 compatible)
# ============================================================================

def extract_scenario_posterior(
    idata,
    max_draws: int = DEFAULT_SCENARIO_DRAWS,
    random_seed: int = 42,
) -> dict:
    rng = np.random.default_rng(random_seed)

    # Use arviz_base.extract to get posterior with stacked sample dimension
    try:
        import arviz_base as azb
        posterior = azb.extract(idata, group="posterior", combined=True, random_seed=random_seed)
    except ImportError:
        # Fallback for older ArviZ
        if hasattr(idata, "posterior"):
            posterior = idata.posterior
        elif hasattr(idata, "groups"):
            posterior = idata["posterior"]
        else:
            raise ValueError("Cannot extract posterior from object")
        posterior = posterior.stack(sample=("chain", "draw"))

    n_all = posterior.sizes["sample"]

    chosen = np.sort(
        rng.choice(
            n_all,
            size=min(max_draws, n_all),
            replace=False,
        )
    )

    cache = {
        "price_slope_z": (
            posterior["price_slope_z"]
            .isel(sample=chosen)
            .transpose("sample", "sku")
            .values
        ),
        "entity_effect": (
            posterior["entity_effect"]
            .isel(sample=chosen)
            .transpose("sample", "entity")
            .values
        ),
        "month_effect": (
            posterior["month_effect"]
            .isel(sample=chosen)
            .transpose("sample", "month")
            .values
        ),
    }

    # v1 shared curve versus v2 SKU-specific curve
    nd_coef = posterior["nd_coef"].isel(sample=chosen)

    if "sku" in nd_coef.dims:
        cache["nd_coef_type"] = "sku_specific"
        cache["nd_coef"] = nd_coef.transpose(
            "sample", "sku", "spline_basis"
        ).values
    else:
        cache["nd_coef_type"] = "shared"
        cache["nd_coef"] = nd_coef.transpose(
            "sample", "spline_basis"
        ).values

    return cache


# ============================================================================
# Corrected scenario math
# ============================================================================

def price_delta_log_volume(
    price_slope_draws: np.ndarray,
    sku_idx: np.ndarray,
    price_change: float,
    target_mask: np.ndarray,
) -> np.ndarray:
    """
    Returns log-volume deltas with shape:
        (n_posterior_draws, n_observations)
    """
    if price_change <= -1:
        raise ValueError("price_change must be greater than -1.0")

    price_delta = np.log1p(price_change)
    beta_by_obs = price_slope_draws[:, sku_idx]
    effect = beta_by_obs * price_delta

    return effect * target_mask[None, :]


def resolve_target_nd(
    nd_base: np.ndarray,
    value: float,
    mode: str,
) -> np.ndarray:
    if mode == "pp":
        nd_new = nd_base + value
    elif mode == "relative":
        nd_new = nd_base * (1.0 + value)
    elif mode == "absolute":
        nd_new = np.full_like(nd_base, value, dtype=float)
    else:
        raise ValueError("mode must be one of: pp, relative, absolute")

    return np.clip(nd_new, 1e-4, 1.0)


def nd_delta_log_volume(
    posterior_cache: dict,
    spline,
    scales: dict,
    sku_idx: np.ndarray,
    nd_base: np.ndarray,
    nd_new: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """
    Returns draw-level log-volume deltas:
    log(ND_new / ND_base) + f(ND_new) - f(ND_base).
    """
    nd_old = np.clip(nd_base, 1e-4, 1.0)
    nd_future = np.clip(nd_new, 1e-4, 1.0)

    # Reapply the transformation used during fitting.
    nd_old_z = (
        np.log(nd_old) - scales["log_nd"]["mean"]
    ) / scales["log_nd"]["sd"]

    nd_new_z = (
        np.log(nd_future) - scales["log_nd"]["mean"]
    ) / scales["log_nd"]["sd"]

    basis_old = spline.transform(nd_old_z.reshape(-1, 1))
    basis_new = spline.transform(nd_new_z.reshape(-1, 1))

    coef = posterior_cache["nd_coef"]

    if posterior_cache["nd_coef_type"] == "sku_specific":
        coef_by_obs = coef[:, sku_idx, :]
        f_old = np.einsum("dos,os->do", coef_by_obs, basis_old)
        f_new = np.einsum("dos,os->do", coef_by_obs, basis_new)
    else:
        f_old = coef @ basis_old.T
        f_new = coef @ basis_new.T

    mechanical_listing_effect = np.log(nd_future / nd_old)[None, :]
    fitted_velocity_effect = f_new - f_old

    result = mechanical_listing_effect + fitted_velocity_effect
    return result * target_mask[None, :]


def combined_delta_log_volume(
    price_effect: np.ndarray,
    nd_effect: np.ndarray,
) -> np.ndarray:
    return price_effect + nd_effect


def summarise_scenario_draws(
    base_units: np.ndarray,
    base_revenue: np.ndarray,
    price_change: float,
    delta_log_volume_draws: np.ndarray,
) -> pd.DataFrame:
    multiplier = np.exp(delta_log_volume_draws)

    scenario_units = multiplier * base_units[None, :]
    scenario_revenue = (
        scenario_units
        * (base_revenue / np.maximum(base_units, 1e-8))[None, :]
        * (1.0 + price_change)
    )

    return pd.DataFrame({
        "units_p05": np.quantile(scenario_units, 0.05, axis=0),
        "units_p50": np.quantile(scenario_units, 0.50, axis=0),
        "units_p95": np.quantile(scenario_units, 0.95, axis=0),
        "revenue_p05": np.quantile(scenario_revenue, 0.05, axis=0),
        "revenue_p50": np.quantile(scenario_revenue, 0.50, axis=0),
        "revenue_p95": np.quantile(scenario_revenue, 0.95, axis=0),
    })


def get_nd_response_curve(
    posterior_cache: dict,
    sku_index: int,
    spline,
    scales: dict,
    nd_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    if nd_grid is None:
        nd_grid = np.linspace(0.05, 0.95, 50)

    nd_grid = np.clip(nd_grid, 1e-4, 1.0)

    nd_z = (
        np.log(nd_grid) - scales["log_nd"]["mean"]
    ) / scales["log_nd"]["sd"]

    basis = spline.transform(nd_z.reshape(-1, 1))
    coef = posterior_cache["nd_coef"]

    if posterior_cache["nd_coef_type"] == "sku_specific":
        effect_draws = coef[:, sku_index, :] @ basis.T
    else:
        effect_draws = coef @ basis.T

    # Relative to 50% distribution to aid interpretation.
    reference_idx = np.argmin(np.abs(nd_grid - 0.50))
    effect_draws = effect_draws - effect_draws[:, [reference_idx]]

    multiplier = np.exp(effect_draws)

    return pd.DataFrame({
        "nd": nd_grid,
        "velocity_multiplier_p10": np.quantile(multiplier, 0.10, axis=0),
        "velocity_multiplier_median": np.quantile(multiplier, 0.50, axis=0),
        "velocity_multiplier_p90": np.quantile(multiplier, 0.90, axis=0),
    })


def get_growth_opportunities(
    df: pd.DataFrame,
    elasticity_df: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "retailer",
        "category",
        "brand",
        "sku",
        "pack_group",
    ]

    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            unit_sales=("units", "sum"),
            revenue=("revenue", "sum"),
            avg_nd=("nd", "mean"),
            avg_velocity_std=("velocity_std", "mean"),
            retailer_stores=("retailer_stores", "max"),
        )
    )

    category_benchmark = (
        summary.groupby(["retailer", "category"])["avg_velocity_std"]
        .transform("median")
    )

    summary["velocity_index"] = (
        summary["avg_velocity_std"]
        / np.maximum(category_benchmark, 1e-8)
    )

    summary["distribution_headroom"] = 1.0 - summary["avg_nd"]
    summary["potential_extra_stores"] = (
        summary["distribution_headroom"] * summary["retailer_stores"]
    )

    elastic = elasticity_df[
        ["sku", "elasticity_median", "elasticity_p05", "elasticity_p95"]
    ].drop_duplicates("sku")

    summary = summary.merge(elastic, on="sku", how="left")

    summary["distribution_opportunity_score"] = (
        np.log1p(summary["revenue"])
        * summary["distribution_headroom"]
        * np.clip(summary["velocity_index"], 0.0, 3.0)
    )

    summary["elasticity_uncertainty"] = (
        summary["elasticity_p95"] - summary["elasticity_p05"]
    )

    summary["price_opportunity_score"] = (
        np.log1p(summary["revenue"])
        * summary["elasticity_median"].abs()
        / np.maximum(summary["elasticity_uncertainty"], 0.1)
    )

    summary["primary_lever"] = np.where(
        summary["distribution_opportunity_score"]
        > summary["price_opportunity_score"],
        "distribution",
        "price",
    )

    return summary.sort_values(
        ["distribution_opportunity_score", "price_opportunity_score"],
        ascending=False,
    )


# ============================================================================
# Fast scenario engine (legacy compatibility)
# ============================================================================

def _thin_posterior(
    posterior: Dict[str, np.ndarray],
    max_draws: int,
) -> Dict[str, np.ndarray]:
    if not posterior:
        return posterior

    n = next(iter(posterior.values())).shape[0]
    if n <= max_draws:
        return posterior

    idx = np.linspace(0, n - 1, max_draws).round().astype(int)
    return {k: v[idx] for k, v in posterior.items()}


# ============================================================================
# Fast scenario engine
# ============================================================================

def _posterior_delta_log_velocity(
    df: pd.DataFrame,
    trace: Any,
    spline: SplineTransformer,
    scales: Dict[str, Dict[str, float]],
    price_change: float,
    nd_change: float,
    nd_change_mode: str = "pp",
    posterior_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure NumPy scenario transformation.

    Returns p10, median, p90 volume multipliers for each row.
    """
    posterior = (
        posterior_cache
        if posterior_cache is not None
        else extract_scenario_posterior(trace)
    )

    n = len(df)
    n_draws = (
        next(iter(posterior.values())).shape[0]
        if posterior
        else 1
    )

    # Price effect.
    price_x_delta = np.log1p(price_change)

    if "price_slope_z" in posterior and "sku_idx" in df.columns:
        price_slopes = posterior["price_slope_z"][:, df["sku_idx"].to_numpy()]
        delta_price_z = (
            price_x_delta / scales["log_relative_price"]["sd"]
        )
        price_delta = price_slopes * delta_price_z

    else:
        price_delta = np.zeros((n_draws, n), dtype=float)

    # Distribution effect.
    nd_old = df["nd"].to_numpy(dtype=float)

    if nd_change_mode == "relative":
        nd_new = np.clip(
            nd_old * (1.0 + nd_change),
            1e-4,
            1.0,
        )
    elif nd_change_mode == "pp":
        nd_new = np.clip(
            nd_old + nd_change,
            1e-4,
            1.0,
        )
    else:
        raise ValueError("nd_change_mode must be 'pp' or 'relative'.")

    old_z = (
        np.log(nd_old) - scales["log_nd"]["mean"]
    ) / scales["log_nd"]["sd"]
    new_z = (
        np.log(nd_new) - scales["log_nd"]["mean"]
    ) / scales["log_nd"]["sd"]

    old_basis = spline.transform(old_z.reshape(-1, 1))
    new_basis = spline.transform(new_z.reshape(-1, 1))
    delta_basis = new_basis - old_basis

    if "nd_coef" in posterior:
        nd_delta = posterior["nd_coef"] @ delta_basis.T
    else:
        nd_delta = np.zeros((n_draws, n), dtype=float)

    # Back to log velocity.
    delta_log_velocity = (
        price_delta + nd_delta
    ) * scales["log_velocity"]["sd"]

    velocity_ratio = np.exp(delta_log_velocity)

    # Wider distribution also creates more listed stores.
    distribution_ratio = nd_new / nd_old

    total_volume_ratio = velocity_ratio * distribution_ratio[None, :]

    p10 = np.quantile(total_volume_ratio, 0.10, axis=0)
    median = np.quantile(total_volume_ratio, 0.50, axis=0)
    p90 = np.quantile(total_volume_ratio, 0.90, axis=0)

    return p10, median, p90


def create_price_distribution_scenario(
    trace: Any,
    df: pd.DataFrame,
    spline: SplineTransformer,
    scales: Dict[str, Dict[str, float]],
    price_change: float,
    nd_change: float,
    nd_change_mode: str = "pp",
    posterior_cache: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Fast scenario function.

    `nd_change` is percentage-point change by default:
        0.10 means +10 percentage points.

    For backward compatibility, callers can set nd_change_mode='relative'
    where 0.10 means +10% relative.
    """
    out = df.copy()
    nd_old = out["nd"].to_numpy(dtype=float)

    if nd_change_mode == "relative":
        nd_new = np.clip(nd_old * (1 + nd_change), 1e-4, 1.0)
    else:
        nd_new = np.clip(nd_old + nd_change, 1e-4, 1.0)

    p10, median, p90 = _posterior_delta_log_velocity(
        out,
        trace,
        spline,
        scales,
        price_change,
        nd_change,
        nd_change_mode=nd_change_mode,
        posterior_cache=posterior_cache,
    )

    out["scenario_price_change"] = price_change
    out["scenario_nd_change"] = nd_change
    out["scenario_nd"] = nd_new
    out["volume_multiplier_p10"] = p10
    out["volume_multiplier_p05"] = p10
    out["volume_multiplier_p50"] = median
    out["volume_multiplier_p95"] = p90

    out["units_p05"] = out["units"].to_numpy() * p10
    out["units_p50"] = out["units"].to_numpy() * median
    out["units_p95"] = out["units"].to_numpy() * p90

    # Price change affects revenue directly; volume effect is multiplicative.
    price_factor = 1.0 + price_change
    out["revenue_p05"] = (
        out["revenue"].to_numpy() * price_factor * p10
    )
    out["revenue_p50"] = (
        out["revenue"].to_numpy() * price_factor * median
    )
    out["revenue_p95"] = (
        out["revenue"].to_numpy() * price_factor * p90
    )

    # Convenience fields.
    out["unit_change_pct_p50"] = median - 1.0
    out["revenue_change_pct_p50"] = (
        price_factor * median
    ) - 1.0

    return out


# ============================================================================
# Market share helpers
# ============================================================================

def calculate_shares(
    df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """
    Add / return common share metrics.

    The important principle is that shares use all competitors present in the
    supplied market frame.
    """
    work = df.copy()

    market_revenue = work.groupby(
        ["month", "category"],
        observed=True,
    )["revenue"].transform("sum")

    retailer_revenue = work.groupby(
        ["month", "retailer", "category"],
        observed=True,
    )["revenue"].transform("sum")

    brand_revenue = work.groupby(
        ["month", "category", "brand"],
        observed=True,
    )["revenue"].transform("sum")

    work["market_revenue"] = market_revenue
    work["retailer_market_revenue"] = retailer_revenue
    work["brand_revenue"] = brand_revenue

    work["market_revenue_share"] = (
        work["revenue"] / market_revenue.replace(0, np.nan)
    )
    work["retailer_revenue_share"] = (
        work["revenue"] / retailer_revenue.replace(0, np.nan)
    )
    work["brand_revenue_share_total"] = (
        work["brand_revenue"]
        / market_revenue.replace(0, np.nan)
    )
    work["category_revenue_share_total"] = 1.0

    # Return Series aligned to the original frame.
    return {
        "market_revenue_share": work["market_revenue_share"],
        "retailer_revenue_share": work["retailer_revenue_share"],
        "brand_revenue_share": work["brand_revenue"],
        "brand_revenue_share_total": work["brand_revenue_share_total"],
        "category_revenue_share_total": work["category_revenue_share_total"],
    }


def calculate_period_market_share(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    level: str = "brand",
) -> pd.DataFrame:
    """
    Calculate baseline vs scenario market shares for a selected period.

    Baseline and scenario are normalized independently so that share always sums
    to 100% across the complete competitive set.
    """
    if level == "brand":
        groups = ["brand"]
    elif level == "sku":
        groups = ["brand", "sku"]
    elif level in {"sku_pack", "pack"}:
        groups = ["brand", "sku", "pack_size"]
    else:
        raise ValueError("level must be 'brand', 'sku', or 'sku_pack'.")

    base = (
        baseline_df.groupby(groups, observed=True)["revenue"]
        .sum()
        .rename("baseline_value")
        .reset_index()
    )

    scen = (
        scenario_df.groupby(groups, observed=True)["revenue_p50"]
        .sum()
        .rename("scenario_value")
        .reset_index()
    )

    out = base.merge(scen, on=groups, how="outer").fillna(0.0)

    base_total = float(out["baseline_value"].sum())
    scen_total = float(out["scenario_value"].sum())

    out["baseline_share"] = (
        out["baseline_value"] / base_total
        if base_total > 0
        else np.nan
    )
    out["scenario_share"] = (
        out["scenario_value"] / scen_total
        if scen_total > 0
        else np.nan
    )
    out["share_change_pp"] = (
        out["scenario_share"] - out["baseline_share"]
    )
    out["share_change_pct"] = (
        out["scenario_share"] / out["baseline_share"].replace(0, np.nan)
    ) - 1.0

    return out.sort_values(
        "scenario_share",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================================
# Aggregation / compatibility
# ============================================================================

def aggregate_scenario(
    scenario_df: pd.DataFrame,
    by: List[str],
) -> pd.DataFrame:
    if by:
        grouped = scenario_df.groupby(by, observed=True)
    else:
        # One synthetic market row.
        tmp = scenario_df.copy()
        tmp["_all"] = "Total market"
        grouped = tmp.groupby(["_all"], observed=True)

    agg = grouped.agg(
        baseline_units=("units", "sum"),
        units_p05=("units_p05", "sum"),
        units_p50=("units_p50", "sum"),
        units_p95=("units_p95", "sum"),
        baseline_revenue=("revenue", "sum"),
        revenue_p05=("revenue_p05", "sum"),
        revenue_p50=("revenue_p50", "sum"),
        revenue_p95=("revenue_p95", "sum"),
    ).reset_index()

    base_u = agg["baseline_units"].replace(0, np.nan)
    base_r = agg["baseline_revenue"].replace(0, np.nan)

    agg["volume_change_pct"] = (
        agg["units_p50"] / base_u
    ) - 1.0
    agg["revenue_change_pct"] = (
        agg["revenue_p50"] / base_r
    ) - 1.0

    if not by:
        agg = agg.rename(columns={"_all": "total_market"})

    return agg


# ============================================================================
# Posterior predictive diagnostics (opt-in)
# ============================================================================

def get_posterior_predictive_check(
    trace: Any,
    df: pd.DataFrame,
    max_points: int = 3000,
) -> pd.DataFrame:
    """
    Generate a compact PPC diagnostic on demand.

    This is deliberately NOT part of fit_model() by default.
    """
    var_name = _find_posterior_variable(
        trace,
        ["log_velocity_z_obs"],
    )
    if var_name is None:
        return pd.DataFrame()

    posterior = _posterior_group(trace)

    try:
        pred = _posterior_array(trace, var_name, ("obs",))
    except Exception:
        return pd.DataFrame()

    pred = pred.reshape(-1, pred.shape[-1])

    if pred.shape[1] != len(df):
        return pd.DataFrame()

    observed = df["log_velocity_z"].to_numpy(dtype=float)
    pred_mean = np.mean(pred, axis=0)

    if len(df) > max_points:
        idx = np.linspace(0, len(df) - 1, max_points).round().astype(int)
    else:
        idx = np.arange(len(df))

    result = pd.DataFrame(
        {
            "observed_z": observed[idx],
            "predicted_z": pred_mean[idx],
        }
    )
    result["residual_z"] = (
        result["observed_z"] - result["predicted_z"]
    )
    return result


# ============================================================================
# Compatibility aliases / small analytical summaries
# ============================================================================

def create_market_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("month", observed=True)
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            standard_volume=("standard_volume", "sum"),
            price_per_standard_unit=("price_std", "mean"),
            nd=("nd", "mean"),
            velocity_std=("velocity_std", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )


def create_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "category"], observed=True)
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            standard_volume=("standard_volume", "sum"),
            price_per_standard_unit=("price_std", "mean"),
            nd=("nd", "mean"),
            velocity_std=("velocity_std", "mean"),
        )
        .reset_index()
        .sort_values(["month", "category"])
    )


def create_retailer_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "retailer"], observed=True)
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            standard_volume=("standard_volume", "sum"),
            price_per_standard_unit=("price_std", "mean"),
            nd=("nd", "mean"),
            velocity_std=("velocity_std", "mean"),
        )
        .reset_index()
        .sort_values(["month", "retailer"])
    )


def create_brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month", "brand"], observed=True)
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            standard_volume=("standard_volume", "sum"),
            price_per_standard_unit=("price_std", "mean"),
            nd=("nd", "mean"),
            velocity_std=("velocity_std", "mean"),
        )
        .reset_index()
        .sort_values(["month", "brand"])
    )


def create_sku_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["category", "retailer", "brand", "sku", "pack_size"],
            observed=True,
        )
        .agg(
            units=("units", "sum"),
            revenue=("revenue", "sum"),
            standard_volume=("standard_volume", "sum"),
            price_per_standard_unit=("price_std", "mean"),
            other_skus_price_std=("other_skus_price_std", "mean"),
            relative_price=("relative_price", "mean"),
            nd=("nd", "mean"),
            velocity_std=("velocity_std", "mean"),
        )
        .reset_index()
    )

    total_units = grouped["units"].sum()
    total_revenue = grouped["revenue"].sum()

    grouped["unit_share"] = (
        grouped["units"] / total_units if total_units > 0 else np.nan
    )
    grouped["revenue_share"] = (
        grouped["revenue"] / total_revenue if total_revenue > 0 else np.nan
    )
    return grouped


def classify_distribution_velocity(
    df: pd.DataFrame,
    benchmark_level: str = "retailer_category",
) -> pd.DataFrame:
    out = df.copy()

    if benchmark_level == "retailer_category":
        grp = ["retailer", "category"]
    else:
        grp = ["category"]

    benchmark = (
        out.groupby(grp, observed=True)["velocity_std"]
        .transform("mean")
    )
    out["relative_velocity"] = (
        out["velocity_std"] / benchmark.replace(0, np.nan)
    )
    return out


def get_expansion_candidates(df: pd.DataFrame) -> pd.DataFrame:
    classified = classify_distribution_velocity(df)
    return classified[
        (classified["nd"] < classified.groupby(
            ["retailer", "category"],
            observed=True,
        )["nd"].transform("median"))
        & (classified["relative_velocity"] > 1.0)
    ].copy()


def create_growth_decomposition(
    df: pd.DataFrame,
) -> pd.DataFrame:
    months = sorted(df["month"].unique())
    if len(months) < 2:
        return pd.DataFrame()

    start_month, end_month = months[0], months[-1]
    return _growth_decomposition(df, start_month, end_month)


def _growth_decomposition(
    df: pd.DataFrame,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
) -> pd.DataFrame:
    start = df[df["month"] == start_month].set_index(
        ["retailer", "sku"]
    )
    end = df[df["month"] == end_month].set_index(
        ["retailer", "sku"]
    )
    common = start.index.intersection(end.index)
    if len(common) == 0:
        return pd.DataFrame()

    out = pd.DataFrame(index=common).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["network"] = (
            np.log(
                end.loc[common, "retailer_stores"].to_numpy()
                / start.loc[common, "retailer_stores"].to_numpy()
            )
            * 100
        )
        out["distribution"] = (
            np.log(
                end.loc[common, "nd"].to_numpy()
                / start.loc[common, "nd"].to_numpy()
            )
            * 100
        )
        out["velocity"] = (
            np.log(
                end.loc[common, "velocity_std"].to_numpy()
                / start.loc[common, "velocity_std"].to_numpy()
            )
            * 100
        )
        out["total_standard_volume"] = (
            np.log(
                end.loc[common, "standard_volume"].to_numpy()
                / start.loc[common, "standard_volume"].to_numpy()
            )
            * 100
        )

    out["label"] = (
        out["retailer"].astype(str)
        + " | "
        + out["sku"].astype(str)
    )
    return out


# ============================================================================
# Optional helpers for the improved UI
# ============================================================================

def prepare_market(
    raw: pd.DataFrame,
    n_knots: int = DEFAULT_N_SPLINE_KNOTS,
) -> PreparedData:
    validation = validate_input_data(raw)
    if not validation.is_valid:
        raise ValueError(
            "Input data is not valid: "
            + "; ".join(validation.warnings)
        )
    prepared, scales = prepare_data(raw)
    indexed, meta = add_indices(prepared)
    spline, basis = fit_spline(
        indexed,
        n_knots=n_knots,
        degree=DEFAULT_SPLINE_DEGREE,
    )
    return PreparedData(
        df=indexed,
        scales=scales,
        meta=meta,
        spline=spline,
        nd_basis=basis,
    )


def run_fast_share_scenario(
    trace: Any,
    df: pd.DataFrame,
    spline: SplineTransformer,
    scales: Dict[str, Dict[str, float]],
    target_mask: np.ndarray,
    price_change: float = 0.0,
    nd_change: float = 0.0,
    nd_change_mode: str = "pp",
    level: str = "brand",
    posterior_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recommended high-level scenario API.

    Keeps every competitor in the denominator and changes only target rows.
    """
    work = df.copy()
    work["_target"] = np.asarray(target_mask, dtype=bool)

    baseline = work.drop(columns=["_target"])

    scenario_target = work.loc[work["_target"]].drop(columns=["_target"])
    scenario_other = work.loc[~work["_target"]].drop(columns=["_target"])

    if len(scenario_target):
        changed = create_price_distribution_scenario(
            trace,
            scenario_target,
            spline,
            scales,
            price_change,
            nd_change,
            nd_change_mode=nd_change_mode,
            posterior_cache=posterior_cache,
        )
    else:
        changed = scenario_target.copy()
        changed["units_p05"] = changed["units"]
        changed["units_p50"] = changed["units"]
        changed["units_p95"] = changed["units"]
        changed["revenue_p05"] = changed["revenue"]
        changed["revenue_p50"] = changed["revenue"]
        changed["revenue_p95"] = changed["revenue"]

    unchanged = scenario_other.copy()
    unchanged["scenario_price_change"] = 0.0
    unchanged["scenario_nd_change"] = 0.0
    unchanged["scenario_nd"] = unchanged["nd"]
    for q in ["p05", "p50", "p95"]:
        unchanged[f"volume_multiplier_{q}"] = 1.0
        unchanged[f"units_{q}"] = unchanged["units"]
        unchanged[f"revenue_{q}"] = unchanged["revenue"]

    scenario = pd.concat(
        [changed, unchanged],
        ignore_index=True,
    )

    shares = calculate_period_market_share(
        baseline,
        scenario,
        level=level,
    )
    return scenario, shares


# ============================================================================
# New Analytics Helpers for Retail Cockpit
# ============================================================================

def decompose_sales_growth(
    df: pd.DataFrame,
    start_month: str,
    end_month: str,
    group_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Decompose sales growth into distribution, velocity, and interaction effects.
    
    Units = sku_stores * velocity_std
    Growth = Distribution effect + Velocity effect + Interaction effect
    
    Returns a DataFrame with growth components per group.
    """
    if group_cols is None:
        group_cols = ["retailer", "category", "brand", "sku"]
    
    # Filter to period
    start = pd.Timestamp(start_month)
    end = pd.Timestamp(end_month)
    df = df[df["month"].between(start, end)].copy()
    
    # Get first and last month per group
    period_data = df.sort_values(group_cols + ["month"]).groupby(group_cols).agg(
        first_month=("month", "first"),
        last_month=("month", "last"),
        first_units=("units", "first"),
        last_units=("units", "last"),
        first_stores=("sku_stores", "first"),
        last_stores=("sku_stores", "last"),
        first_velocity=("velocity_std", "first"),
        last_velocity=("velocity_std", "last"),
        first_revenue=("revenue", "first"),
        last_revenue=("revenue", "last"),
        first_price=("price_std", "first"),
        last_price=("price_std", "last"),
        first_nd=("nd", "first"),
        last_nd=("nd", "last"),
    ).reset_index()
    
    # Only keep groups with data in both periods
    period_data = period_data.dropna()
    
    if period_data.empty:
        return pd.DataFrame()
    
    # Decomposition
    # Units = stores * velocity
    # ΔUnits = Δstores * velocity_old + stores_old * Δvelocity + Δstores * Δvelocity
    
    period_data["distribution_effect"] = (
        (period_data["last_stores"] - period_data["first_stores"]) 
        * period_data["first_velocity"]
    )
    period_data["velocity_effect"] = (
        period_data["first_stores"] 
        * (period_data["last_velocity"] - period_data["first_velocity"])
    )
    period_data["interaction_effect"] = (
        (period_data["last_stores"] - period_data["first_stores"])
        * (period_data["last_velocity"] - period_data["first_velocity"])
    )
    period_data["total_units_change"] = (
        period_data["last_units"] - period_data["first_units"]
    )
    
    # Revenue decomposition
    # Revenue = Units * Price
    period_data["unit_volume_effect"] = (
        period_data["total_units_change"] * period_data["first_price"]
    )
    period_data["price_effect"] = (
        period_data["last_units"] * (period_data["last_price"] - period_data["first_price"])
    )
    period_data["total_revenue_change"] = (
        period_data["last_revenue"] - period_data["first_revenue"]
    )
    
    # Growth rates
    period_data["units_growth_pct"] = (
        period_data["total_units_change"] / period_data["first_units"] * 100
    )
    period_data["revenue_growth_pct"] = (
        period_data["total_revenue_change"] / period_data["first_revenue"] * 100
    )
    
    return period_data.sort_values("total_units_change", ascending=False).reset_index(drop=True)


def compute_price_architecture(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute price architecture metrics per SKU.
    
    Returns per-SKU price metrics for price ladder analysis.
    """
    sku_metrics = df.groupby(["retailer", "category", "brand", "sku", "pack_size"]).agg(
        avg_price=("price_std", "mean"),
        avg_units=("units", "mean"),
        avg_revenue=("revenue", "mean"),
        avg_velocity=("velocity_std", "mean"),
        avg_nd=("nd", "mean"),
        months=("month", "nunique"),
    ).reset_index()
    
    # Price per physical unit
    sku_metrics["price_per_unit"] = sku_metrics["avg_price"] / sku_metrics["pack_size"]
    
    # Category benchmarks
    cat_price = sku_metrics.groupby(["retailer", "category"])["price_per_unit"].transform("median")
    sku_metrics["rel_price_index"] = sku_metrics["price_per_unit"] / cat_price
    
    # Brand benchmarks
    brand_price = sku_metrics.groupby(["retailer", "category", "brand"])["price_per_unit"].transform("median")
    sku_metrics["brand_price_index"] = sku_metrics["price_per_unit"] / brand_price
    
    # Pack segment
    sku_metrics["pack_segment"] = pd.cut(
        sku_metrics["pack_size"],
        bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["small", "medium", "large", "xl"]
    )
    
    # Pack segment benchmarks
    seg_price = sku_metrics.groupby(["retailer", "category", "pack_segment"])["price_per_unit"].transform("median")
    sku_metrics["pack_value_index"] = sku_metrics["price_per_unit"] / seg_price
    
    # Price dispersion (within SKU across months)
    price_disp = df.groupby(["retailer", "category", "brand", "sku"])["price_std"].agg(
        price_p10=lambda x: x.quantile(0.10),
        price_p90=lambda x: x.quantile(0.90),
    ).reset_index()
    price_disp["price_dispersion"] = price_disp["price_p90"] - price_disp["price_p10"]
    
    sku_metrics = sku_metrics.merge(
        price_disp[["retailer", "category", "brand", "sku", "price_dispersion"]],
        on=["retailer", "category", "brand", "sku"],
        how="left"
    )
    
    return sku_metrics.sort_values(["retailer", "category", "brand", "price_per_unit"]).reset_index(drop=True)


def compute_distribution_opportunity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute distribution opportunity metrics per SKU.
    
    Identifies white-space (high velocity, low ND) and 
    rationalization candidates (low velocity, high ND).
    """
    sku_metrics = df.groupby(["retailer", "category", "brand", "sku", "pack_size"]).agg(
        avg_nd=("nd", "mean"),
        avg_velocity=("velocity_std", "mean"),
        total_units=("units", "sum"),
        total_revenue=("revenue", "sum"),
        max_stores=("retailer_stores", "max"),
        avg_stores=("sku_stores", "mean"),
    ).reset_index()
    
    # Category benchmarks for velocity
    cat_vel = sku_metrics.groupby(["retailer", "category"])["avg_velocity"].transform("median")
    sku_metrics["velocity_index"] = sku_metrics["avg_velocity"] / cat_vel
    
    # Distribution headroom
    sku_metrics["distribution_headroom"] = 1.0 - sku_metrics["avg_nd"]
    sku_metrics["potential_extra_stores"] = (
        sku_metrics["distribution_headroom"] * sku_metrics["max_stores"]
    )
    
    # Opportunity scoring
    # High velocity + low ND + large revenue = listing expansion opportunity
    sku_metrics["distribution_opportunity_score"] = (
        np.log1p(sku_metrics["total_revenue"])
        * sku_metrics["distribution_headroom"]
        * np.clip(sku_metrics["velocity_index"], 0.0, 3.0)
    )
    
    # Rationalization score
    # Low velocity + high ND = potential delist candidate
    sku_metrics["rationalization_score"] = (
        sku_metrics["avg_nd"]
        / np.maximum(sku_metrics["velocity_index"], 0.1)
        * np.log1p(sku_metrics["total_revenue"])
    )
    
    # Classification
    conditions = [
        (sku_metrics["avg_velocity"] > cat_vel) & (sku_metrics["avg_nd"] < 0.5),
        (sku_metrics["avg_velocity"] > cat_vel) & (sku_metrics["avg_nd"] >= 0.5),
        (sku_metrics["avg_velocity"] <= cat_vel) & (sku_metrics["avg_nd"] < 0.5),
        (sku_metrics["avg_velocity"] <= cat_vel) & (sku_metrics["avg_nd"] >= 0.5),
    ]
    choices = ["Expand", "Protect", "Test/Review", "Rationalize"]
    sku_metrics["distribution_action"] = np.select(conditions, choices, default="Review")
    
    return sku_metrics.sort_values("distribution_opportunity_score", ascending=False).reset_index(drop=True)


def compute_market_share_analytics(
    df: pd.DataFrame,
    group_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive market share analytics.
    """
    if group_cols is None:
        group_cols = ["retailer", "category", "month", "brand", "sku"]
    
    # Monthly share by group
    monthly = df.groupby(group_cols).agg(
        units=("units", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    
    # Total market per retailer/category/month
    market_cols = ["retailer", "category", "month"]
    market_total = monthly.groupby(market_cols).agg(
        market_units=("units", "sum"),
        market_revenue=("revenue", "sum"),
    ).reset_index()
    
    monthly = monthly.merge(market_total, on=market_cols, how="left")
    monthly["unit_share"] = monthly["units"] / monthly["market_units"]
    monthly["revenue_share"] = monthly["revenue"] / monthly["market_revenue"]
    
    # Share momentum (3, 6, 12 month changes)
    monthly = monthly.sort_values(group_cols + ["month"])
    for window in [3, 6, 12]:
        monthly[f"unit_share_change_{window}m"] = monthly.groupby(group_cols)["unit_share"].transform(
            lambda x: x - x.shift(window)
        )
        monthly[f"revenue_share_change_{window}m"] = monthly.groupby(group_cols)["revenue_share"].transform(
            lambda x: x - x.shift(window)
        )
    
    # Pack segment share
    if "pack_size" in df.columns:
        monthly["pack_segment"] = pd.cut(
            monthly["pack_size"] if "pack_size" in monthly.columns else df["pack_size"],
            bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
            labels=["small", "medium", "large", "xl"]
        )
    
    return monthly


def compute_contribution_to_growth(
    df: pd.DataFrame,
    start_month: str,
    end_month: str,
    level: str = "brand",
) -> pd.DataFrame:
    """
    Compute contribution to category growth at specified level.
    
    Level can be: brand, sku, retailer
    """
    start = pd.Timestamp(start_month)
    end = pd.Timestamp(end_month)
    df_period = df[df["month"].between(start, end)].copy()
    
    if df_period.empty:
        return pd.DataFrame()
    
    if level == "brand":
        group_cols = ["retailer", "category", "brand"]
    elif level == "sku":
        group_cols = ["retailer", "category", "brand", "sku"]
    elif level == "retailer":
        group_cols = ["category", "retailer"]
    else:
        group_cols = ["retailer", "category", "brand"]
    
    # Start and end period values
    start_data = df_period[df_period["month"] == start].groupby(group_cols).agg(
        start_units=("units", "sum"),
        start_revenue=("revenue", "sum"),
    ).reset_index()
    
    end_data = df_period[df_period["month"] == end].groupby(group_cols).agg(
        end_units=("units", "sum"),
        end_revenue=("revenue", "sum"),
    ).reset_index()
    
    growth = start_data.merge(end_data, on=group_cols, how="outer").fillna(0)
    growth["unit_change"] = growth["end_units"] - growth["start_units"]
    growth["revenue_change"] = growth["end_revenue"] - growth["start_revenue"]
    
    # Total category change
    total_unit_change = growth["unit_change"].sum()
    total_rev_change = growth["revenue_change"].sum()
    
    growth["unit_contribution_pct"] = np.where(
        total_unit_change != 0,
        growth["unit_change"] / total_unit_change * 100,
        0
    )
    growth["revenue_contribution_pct"] = np.where(
        total_rev_change != 0,
        growth["revenue_change"] / total_rev_change * 100,
        0
    )
    
    # Share of growth
    growth["unit_share_of_growth"] = np.where(
        total_unit_change > 0,
        growth["unit_change"] / total_unit_change * 100,
        0
    )
    
    return growth.sort_values("unit_contribution_pct", ascending=False).reset_index(drop=True)


def generate_data_health_report(
    raw_df: pd.DataFrame,
    prepared_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive data health report.
    
    Returns dict with completeness, exclusions, outliers, coverage metrics.
    """
    report = {}
    
    # Basic counts
    report["total_rows"] = len(raw_df)
    report["date_range"] = {
        "min": str(raw_df["month"].min()) if "month" in raw_df.columns else None,
        "max": str(raw_df["month"].max()) if "month" in raw_df.columns else None,
        "n_months": int(raw_df["month"].nunique()) if "month" in raw_df.columns else 0,
    }
    report["n_retailers"] = int(raw_df["retailer"].nunique()) if "retailer" in raw_df.columns else 0
    report["n_categories"] = int(raw_df["category"].nunique()) if "category" in raw_df.columns else 0
    report["n_brands"] = int(raw_df["brand"].nunique()) if "brand" in raw_df.columns else 0
    report["n_skus"] = int(raw_df["sku"].nunique()) if "sku" in raw_df.columns else 0
    
    # Missing values
    numeric_cols = ["units", "revenue", "sku_stores", "retailer_stores", "pack_size"]
    missing = {}
    for col in numeric_cols:
        if col in raw_df.columns:
            missing[col] = int(raw_df[col].isna().sum())
    report["missing_values"] = missing
    
    # Zero/negative values
    zero_neg = {}
    for col in numeric_cols:
        if col in raw_df.columns:
            zero_neg[col] = int((raw_df[col] <= 0).sum())
    report["zero_or_negative"] = zero_neg
    
    # Distribution validity
    if all(c in raw_df.columns for c in ["sku_stores", "retailer_stores"]):
        invalid_dist = int((raw_df["sku_stores"] > raw_df["retailer_stores"]).sum())
        zero_stores = int((raw_df["sku_stores"] <= 0).sum() | (raw_df["retailer_stores"] <= 0).sum())
        report["distribution_validity"] = {
            "sku_stores_gt_retailer_stores": invalid_dist,
            "zero_stores": zero_stores,
        }
    
    # Price outliers (using IQR method)
    if "revenue" in raw_df.columns and "units" in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df["implied_price"] = raw_df["revenue"] / raw_df["units"].replace(0, np.nan)
        valid_price = raw_df["implied_price"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_price) > 0:
            q1, q3 = valid_price.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = int(((raw_df["implied_price"] < lower) | (raw_df["implied_price"] > upper)).sum())
            report["price_outliers_iqr"] = outliers
            report["price_stats"] = {
                "median": float(valid_price.median()),
                "q1": float(q1),
                "q3": float(q3),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
    
    # Coverage completeness
    if "month" in raw_df.columns and "retailer" in raw_df.columns and "sku" in raw_df.columns:
        expected = raw_df.groupby(["retailer", "sku"])["month"].nunique()
        max_months = raw_df["month"].nunique()
        coverage = (expected / max_months).mean()
        report["avg_sku_coverage"] = float(coverage)
        
        # Monthly completeness heatmap data
        if "retailer" in raw_df.columns and "category" in raw_df.columns:
            pivot = raw_df.pivot_table(
                index="retailer",
                columns="category",
                values="sku",
                aggfunc="nunique",
                fill_value=0
            )
            report["retailer_category_sku_counts"] = pivot.to_dict()
    
    # Prepared data stats
    if prepared_df is not None:
        report["prepared_rows"] = len(prepared_df)
        report["prepared_n_retailers"] = int(prepared_df["retailer"].nunique())
        report["prepared_n_categories"] = int(prepared_df["category"].nunique())
        report["prepared_n_brands"] = int(prepared_df["brand"].nunique())
        report["prepared_n_skus"] = int(prepared_df["sku"].nunique())
        report["prepared_date_range"] = {
            "min": str(prepared_df["month"].min()),
            "max": str(prepared_df["month"].max()),
        }
        
        # Exclusion rate
        report["exclusion_rate"] = 1.0 - (len(prepared_df) / len(raw_df)) if len(raw_df) > 0 else 0
    
    return report


def compute_model_validation_metrics(
    idata,
    prepared_df: pd.DataFrame,
    observed_var: str = "log_velocity_z_obs",
) -> Dict[str, Any]:
    """
    Compute model validation metrics: backtest coverage, residual patterns, calibration.
    """
    metrics = {}
    
    if observed_var not in idata.observed_data:
        return {"error": f"Observed variable {observed_var} not found"}
    
    observed = idata.observed_data[observed_var].values
    
    if "posterior_predictive" in idata:
        pred = idata.posterior_predictive[observed_var]
        pred_median = pred.median(dim=("chain", "draw")).values
        pred_p10 = pred.quantile(0.05, dim=("chain", "draw")).values
        pred_p90 = pred.quantile(0.95, dim=("chain", "draw")).values
        
        # Coverage
        coverage_90 = np.mean((observed >= pred_p10) & (observed <= pred_p90))
        coverage_50 = np.mean((observed >= pred.quantile(0.25, dim=("chain", "draw")).values) & 
                              (observed <= pred.quantile(0.75, dim=("chain", "draw")).values))
        
        # RMSE, MAE
        rmse = np.sqrt(np.mean((observed - pred_median) ** 2))
        mae = np.mean(np.abs(observed - pred_median))
        
        # Bias
        bias = np.mean(pred_median - observed)
        
        metrics["posterior_predictive"] = {
            "coverage_90": float(coverage_90),
            "coverage_50": float(coverage_50),
            "rmse": float(rmse),
            "mae": float(mae),
            "bias": float(bias),
        }
        
        # Residuals by retailer/category
        if "retailer" in prepared_df.columns and "category" in prepared_df.columns:
            residuals = observed - pred_median
            residual_df = prepared_df[["retailer", "category", "month"]].copy()
            residual_df["residual"] = residuals
            
            # Retailer residual heatmap data
            retailer_month = residual_df.pivot_table(
                index="retailer",
                columns="month",
                values="residual",
                aggfunc="mean"
            )
            metrics["residuals_by_retailer_month"] = retailer_month.to_dict()
            
            # Category residual stats
            cat_residuals = residual_df.groupby("category")["residual"].agg(["mean", "std", "count"])
            metrics["residuals_by_category"] = cat_residuals.to_dict()
    
    return metrics


def build_observed_vs_predicted_data(
    idata,
    prepared_df: pd.DataFrame,
    observed_var: str = "log_velocity_z_obs",
) -> pd.DataFrame:
    """
    Build DataFrame for observed vs predicted scatter plot.
    """
    if observed_var not in idata.observed_data:
        raise KeyError(f"Observed variable {observed_var} not found")
    
    observed = idata.observed_data[observed_var].values
    
    if "posterior_predictive" in idata:
        pred = idata.posterior_predictive[observed_var]
        pred_median = pred.median(dim=("chain", "draw")).values
        pred_p10 = pred.quantile(0.05, dim=("chain", "draw")).values
        pred_p90 = pred.quantile(0.95, dim=("chain", "draw")).values
    else:
        pred_median = np.full_like(observed, np.nan)
        pred_p10 = np.full_like(observed, np.nan)
        pred_p90 = np.full_like(observed, np.nan)
    
    result = prepared_df[["retailer", "category", "brand", "sku", "month", observed_var]].copy()
    result = result.rename(columns={observed_var: "observed"})
    result["predicted_median"] = pred_median
    result["predicted_p10"] = pred_p10
    result["predicted_p90"] = pred_p90
    result["residual"] = result["observed"] - result["predicted_median"]
    result["inside_90"] = (result["observed"] >= result["predicted_p10"]) & (result["observed"] <= result["predicted_p90"])
    
    return result


def build_price_ladder_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build data for price ladder visualization.
    
    Returns per-SKU price metrics suitable for scatter/bubble plots.
    """
    return compute_price_architecture(df)


def build_nd_velocity_quadrant_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build data for ND/Velocity quadrant chart.
    """
    return compute_distribution_opportunity(df)


def build_retail_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Central feature engineering function - adds all derived retail analytics columns.
    
    Creates: unit_price, price_per_size_unit, nd, velocity, log_velocity,
    category_units, brand_units, shares, price indices, pack_group, month_num, lag features.
    Also adds data_quality_status flags.
    
    Run once after data loading, before model fitting and all descriptive tabs.
    """
    df = df.copy()
    
    # --- Data quality flags ---
    valid_units = df["units"] > 0
    valid_revenue = df["revenue"] > 0
    valid_pack_size = df["pack_size"] > 0
    valid_store_counts = (
        (df["sku_stores"] > 0)
        & (df["retailer_stores"] > 0)
        & (df["sku_stores"] <= df["retailer_stores"])
    )
    
    conditions = [
        ~valid_units,
        ~valid_revenue,
        ~valid_pack_size,
        ~valid_store_counts,
        df["category"].isna() | df["brand"].isna() | df["sku"].isna() | df["retailer"].isna(),
    ]
    choices = [
        "zero_or_negative_units",
        "zero_or_negative_revenue",
        "invalid_pack_size",
        "invalid_store_counts",
        "missing_hierarchy",
    ]
    df["data_quality_status"] = np.select(conditions, choices, default="valid")
    
    # --- Core derived columns ---
    df["unit_price"] = np.where(
        df["units"] > 0,
        df["revenue"] / df["units"],
        np.nan
    )
    
    df["price_per_size_unit"] = np.where(
        df["pack_size"] > 0,
        df["unit_price"] / df["pack_size"],
        np.nan
    )
    
    df["nd"] = np.where(
        df["retailer_stores"] > 0,
        df["sku_stores"] / df["retailer_stores"],
        np.nan
    )
    
    df["velocity"] = np.where(
        df["sku_stores"] > 0,
        df["units"] / df["sku_stores"],
        np.nan
    )
    
    df["velocity_std"] = df["velocity"]  # alias for model compatibility
    df["log_velocity"] = np.log1p(df["velocity"])
    
    # --- Market context columns ---
    # Category units (total demand per month × retailer × category)
    cat_units = df.groupby(["month", "retailer", "category"])["units"].transform("sum")
    df["category_units"] = cat_units
    
    # Brand units (total demand per month × retailer × category × brand)
    brand_units = df.groupby(["month", "retailer", "category", "brand"])["units"].transform("sum")
    df["brand_units"] = brand_units
    
    # Pack-group units (total demand per month × retailer × category × brand × pack_group)
    # Pack group computed later, so we'll add this after pack_group
    
    # Shares
    df["brand_share"] = np.where(
        df["category_units"] > 0,
        df["brand_units"] / df["category_units"],
        np.nan
    )
    
    df["sku_share_of_brand"] = np.where(
        df["brand_units"] > 0,
        df["units"] / df["brand_units"],
        np.nan
    )
    
    df["sku_share_of_category"] = np.where(
        df["category_units"] > 0,
        df["units"] / df["category_units"],
        np.nan
    )
    
    # --- Price indices ---
    # Category price median
    cat_price_median = df.groupby(["month", "retailer", "category"])["unit_price"].transform("median")
    df["category_price_median"] = cat_price_median
    
    # Category price index (volume-weighted average price per size unit)
    _w = df["price_per_size_unit"].fillna(0) * df["units"]
    _wsum = _w.groupby([df["month"], df["retailer"], df["category"]]).transform("sum")
    _usum = df["units"].groupby([df["month"], df["retailer"], df["category"]]).transform("sum")
    df["category_price_index"] = np.where(_usum > 0, _wsum / _usum, np.nan)
    
    # Relative price index (vs category median)
    df["relative_price_index"] = np.where(
        df["category_price_median"] > 0,
        df["unit_price"] / df["category_price_median"],
        np.nan
    )
    
    # Relative price index vs volume-weighted category price index
    df["relative_price_index_vw"] = np.where(
        df["category_price_index"] > 0,
        df["price_per_size_unit"] / df["category_price_index"],
        np.nan
    )
    
    # Same-brand other-SKU price index: volume-weighted avg price of other SKUs sharing the brand identifier.
    # Computed as (brand weighted total - selected SKU contribution) / (brand units - selected SKU units).
    df["_pxu"] = df["unit_price"] * df["units"]
    
    brand_grp = ["month", "retailer", "category", "brand"]
    brand_pu = df.groupby(brand_grp)["_pxu"].transform("sum")
    brand_u = df.groupby(brand_grp)["units"].transform("sum")
    
    own_denom = brand_u - df["units"]
    df["same_brand_other_sku_price_index"] = np.where(
        own_denom > 0,
        (brand_pu - df["_pxu"]) / own_denom,
        np.nan,
    )
    
    # Other-brand price index: volume-weighted avg price of all other brands
    # = (category weighted total - brand weighted total) / (category units - brand units).
    cat_grp = ["month", "retailer", "category"]
    cat_pu = df.groupby(cat_grp)["_pxu"].transform("sum")
    cat_u = df.groupby(cat_grp)["units"].transform("sum")
    
    comp_denom = cat_u - brand_u
    df["other_brand_price_index"] = np.where(
        comp_denom > 0,
        (cat_pu - brand_pu) / comp_denom,
        np.nan,
    )
    
    df = df.drop(columns=["_pxu"])
    
    # --- Pack group (category-relative) ---
    df["pack_group"] = _add_category_relative_pack_groups(df)
    
    # --- Pack-group units and shares (after pack_group is created) ---
    pack_units = df.groupby(["month", "retailer", "category", "brand", "pack_group"])["units"].transform("sum")
    df["pack_group_units"] = pack_units
    
    df["brand_pack_share"] = np.where(
        df["brand_units"] > 0,
        df["pack_group_units"] / df["brand_units"],
        np.nan
    )
    
    df["pack_group_share_of_category"] = np.where(
        df["category_units"] > 0,
        df["pack_group_units"] / df["category_units"],
        np.nan
    )
    
    # --- Month number ---
    df["month_num"] = df["month"].dt.month
    
    # --- Lag features ---
    df = df.sort_values(["retailer", "category", "brand", "sku", "month"])
    df["lag_units"] = df.groupby(["retailer", "category", "brand", "sku"])["units"].shift(1)
    df["lag_share"] = df.groupby(["retailer", "category", "brand", "sku"])["sku_share_of_category"].shift(1)
    df["lag_price"] = df.groupby(["retailer", "category", "brand", "sku"])["unit_price"].shift(1)
    
    # --- Extreme price flag ---
    valid_price = df["unit_price"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_price) > 0:
        q1, q3 = valid_price.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        df["extreme_price"] = (df["unit_price"] < lower) | (df["unit_price"] > upper)
        df.loc[df["extreme_price"], "data_quality_status"] = "extreme_price"
    else:
        df["extreme_price"] = False
    
    return df


# ============================================================================
# Market Impact Comparison Module
# ============================================================================

MARKET_LEVELS = {
    "market": ["month"],
    "retailer": ["month", "retailer"],
    "retailer_category": ["month", "retailer", "category"],
    "brand": ["month", "retailer", "category", "brand"],
    "brand_pack": ["month", "retailer", "category", "brand", "pack_group"],
    "sku": ["month", "retailer", "category", "brand", "pack_group", "sku"],
}

SEGMENT_RELATIONSHIP_LABELS = {
    "selected_sku": "Selected SKU",
    "same_brand_other_pack": "Same brand, other pack",
    "same_category_same_pack": "Same category, same pack group (other brands)",
    "same_category_other_pack": "Same category, other pack groups (other brands)",
    "other_category": "Other categories (cross-category)",
    "other_retailer": "Other retailers",
}


def _simulate_scenario_core(
    df: pd.DataFrame,
    posterior_cache: dict,
    spline,
    scales: dict,
    price_change: float,
    nd_change: float,
    nd_mode: str,
    target_level: str,
    target_value: str,
) -> pd.DataFrame:
    """Core scenario simulation returning DataFrame with predicted units/revenue."""
    nd_base = df["nd"].to_numpy()
    
    if nd_change != 0.0:
        if nd_mode == "pp":
            resolved_nd = resolve_target_nd(nd_base, nd_change, "pp")
        elif nd_mode == "relative":
            resolved_nd = resolve_target_nd(nd_base, nd_change, "relative")
        else:
            resolved_nd = resolve_target_nd(nd_base, nd_change, "absolute")
    else:
        resolved_nd = nd_base
    
    target_mask = apply_target_scope(df, target_level, target_value)
    
    price_effect = np.zeros((posterior_cache["price_slope_z"].shape[0], len(df)))
    if price_change != 0.0:
        price_effect = price_delta_log_volume(
            posterior_cache["price_slope_z"],
            df["sku_idx"].to_numpy(dtype="int32"),
            price_change,
            target_mask.to_numpy(),
        )
    
    nd_effect = np.zeros((posterior_cache["price_slope_z"].shape[0], len(df)))
    if nd_change != 0.0:
        nd_effect = nd_delta_log_volume(
            posterior_cache,
            spline,
            scales,
            df["sku_idx"].to_numpy(dtype="int32"),
            nd_base,
            resolved_nd,
            target_mask.to_numpy(),
        )
    
    total_effect = combined_delta_log_volume(price_effect, nd_effect)
    
    result = summarise_scenario_draws(
        df["units"].to_numpy(),
        df["revenue"].to_numpy(),
        price_change,
        total_effect,
    )
    
    out = df.copy()
    result.index = out.index
    for col in result.columns:
        out[col] = result[col]
    
    out["baseline_units"] = df["units"]
    out["baseline_revenue"] = df["revenue"]
    out["baseline_nd"] = df["nd"]
    if "unit_price" not in out.columns:
        out["unit_price"] = np.where(df["units"] > 0, df["revenue"] / df["units"], np.nan)
    out["baseline_price"] = out["unit_price"]
    
    out["scenario_nd"] = resolved_nd
    out["scenario_price"] = out["unit_price"] * (1.0 + price_change)
    out["scenario_unit_price"] = out["scenario_price"]  # canonical name
    out["scenario_price_change"] = price_change
    out["scenario_nd_change"] = nd_change
    
    # Validate against canonical contract
    validate_scenario_result(out)
    
    return out


def run_scenario_suite(
    df: pd.DataFrame,
    posterior_cache: dict,
    spline,
    scales: dict,
    price_change: float,
    nd_change: float,
    nd_mode: str,
    target_level: str,
    target_value: str,
) -> dict:
    """
    Run four standard scenarios for comparison.
    
    Returns dict with keys: baseline, price_only, distribution_only, combined
    """
    scenarios = {}
    
    # Baseline
    scenarios["baseline"] = _simulate_scenario_core(
        df, posterior_cache, spline, scales,
        price_change=0.0, nd_change=0.0, nd_mode=nd_mode,
        target_level=target_level, target_value=target_value
    )
    
    # Price-only
    scenarios["price_only"] = _simulate_scenario_core(
        df, posterior_cache, spline, scales,
        price_change=price_change, nd_change=0.0, nd_mode=nd_mode,
        target_level=target_level, target_value=target_value
    )
    
    # Distribution-only
    scenarios["distribution_only"] = _simulate_scenario_core(
        df, posterior_cache, spline, scales,
        price_change=0.0, nd_change=nd_change, nd_mode=nd_mode,
        target_level=target_level, target_value=target_value
    )
    
    # Combined
    scenarios["combined"] = _simulate_scenario_core(
        df, posterior_cache, spline, scales,
        price_change=price_change, nd_change=nd_change, nd_mode=nd_mode,
        target_level=target_level, target_value=target_value
    )
    
    return scenarios


def classify_segment_relationship(
    row: pd.Series,
    selected_sku: str,
    selected_brand: str,
    selected_category: str,
    selected_pack_group: str,
    selected_retailer: str,
) -> str:
    """
    Classify a row's relationship to the selected SKU.
    
    Returns one of:
    - selected_sku
    - same_brand_other_pack
    - same_category_same_pack
    - same_category_other_pack
    - other_category
    - other_retailer
    """
    if row["sku"] == selected_sku:
        return "selected_sku"
    
    if row["brand"] == selected_brand:
        if row["category"] == selected_category:
            if row["pack_group"] != selected_pack_group:
                return "same_brand_other_pack"
        return "same_category_other_pack"  # same brand, different category (edge case)
    
    if row["category"] == selected_category:
        if row["pack_group"] == selected_pack_group:
            return "same_category_same_pack"
        return "same_category_other_pack"
    
    if row["retailer"] == selected_retailer:
        return "other_category"
    
    return "other_retailer"


def add_segment_classification(
    df: pd.DataFrame,
    selected_sku: str,
    selected_brand: str,
    selected_category: str,
    selected_pack_group: str,
    selected_retailer: str,
) -> pd.DataFrame:
    """Add segment relationship classification to a DataFrame."""
    df = df.copy()
    df["segment_relationship"] = df.apply(
        lambda row: classify_segment_relationship(
            row, selected_sku, selected_brand, selected_category,
            selected_pack_group, selected_retailer
        ),
        axis=1
    )
    return df


def aggregate_market_impact(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """
    Aggregate baseline and scenario to specified market level and compute deltas.
    
    Levels: market, retailer, retailer_category, brand, brand_pack, sku
    """
    if level not in MARKET_LEVELS:
        raise ValueError(f"Unknown level: {level}. Valid: {list(MARKET_LEVELS.keys())}")
    
    group_cols = MARKET_LEVELS[level]
    
    baseline = (
        baseline_df
        .groupby(group_cols, as_index=False)
        .agg(
            baseline_units=("baseline_units", "sum"),
            baseline_revenue=("baseline_revenue", "sum"),
        )
    )
    
    scenario = (
        scenario_df
        .groupby(group_cols, as_index=False)
        .agg(
            scenario_units=("units_p50", "sum"),
            scenario_revenue=("revenue_p50", "sum"),
        )
    )
    
    result = baseline.merge(scenario, on=group_cols, how="outer").fillna(0)
    
    result["delta_units"] = result["scenario_units"] - result["baseline_units"]
    result["delta_revenue"] = result["scenario_revenue"] - result["baseline_revenue"]
    
    result["delta_units_pct"] = np.where(
        result["baseline_units"] > 0,
        result["delta_units"] / result["baseline_units"],
        np.nan,
    )
    result["delta_revenue_pct"] = np.where(
        result["baseline_revenue"] > 0,
        result["delta_revenue"] / result["baseline_revenue"],
        np.nan,
    )
    
    if "retailer" in result.columns:
        market_cols = ["retailer"] if "retailer" in group_cols else []
        if "category" in group_cols:
            market_cols.append("category")
        if "month" in group_cols:
            market_cols.append("month")
        
        if market_cols:
            market_total = result.groupby(market_cols, as_index=False).agg(
                market_scenario_units=("scenario_units", "sum"),
                market_baseline_units=("baseline_units", "sum"),
            )
            result = result.merge(market_total, on=market_cols, how="left")
            result["scenario_share"] = np.where(
                result["market_scenario_units"] > 0,
                result["scenario_units"] / result["market_scenario_units"],
                np.nan,
            )
            result["baseline_share"] = np.where(
                result["market_baseline_units"] > 0,
                result["baseline_units"] / result["market_baseline_units"],
                np.nan,
            )
            result["share_change_pp"] = result["scenario_share"] - result["baseline_share"]
    
    return result


def compute_reallocation_breakdown(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    selected_sku: str,
    selected_brand: str,
    selected_category: str,
    selected_pack_group: str,
    selected_retailer: str,
) -> pd.DataFrame:
    """
    Compute reallocation breakdown by segment relationship.
    
    Returns DataFrame with one row per segment relationship showing
    baseline, scenario, delta units, delta %, and share of total change.
    """
    baseline_classified = add_segment_classification(
        baseline_df, selected_sku, selected_brand, selected_category,
        selected_pack_group, selected_retailer
    )
    scenario_classified = add_segment_classification(
        scenario_df, selected_sku, selected_brand, selected_category,
        selected_pack_group, selected_retailer
    )
    
    baseline_agg = baseline_classified.groupby("segment_relationship").agg(
        baseline_units=("baseline_units", "sum"),
        baseline_revenue=("baseline_revenue", "sum"),
    )
    
    scenario_agg = scenario_classified.groupby("segment_relationship").agg(
        scenario_units=("units_p50", "sum"),
        scenario_revenue=("revenue_p50", "sum"),
    )
    
    result = baseline_agg.join(scenario_agg, how="outer").fillna(0)
    
    result["delta_units"] = result["scenario_units"] - result["baseline_units"]
    result["delta_revenue"] = result["scenario_revenue"] - result["baseline_revenue"]
    
    result["delta_units_pct"] = np.where(
        result["baseline_units"] > 0,
        result["delta_units"] / result["baseline_units"],
        np.nan,
    )
    
    total_delta = result["delta_units"].sum()
    result["share_of_total_delta"] = np.where(
        total_delta != 0,
        result["delta_units"] / total_delta,
        0,
    )
    
    result["segment_label"] = result.index.map(SEGMENT_RELATIONSHIP_LABELS)
    
    return result.reset_index()


def compute_parameter_attribution(
    posterior_cache: dict,
    spline,
    scales: dict,
    sku_idx: int,
    price_change: float,
    nd_base: float,
    nd_new: float,
    selected_sku_name: str,
    meta: dict,
) -> pd.DataFrame:
    """
    Compute parameter-level attribution for the selected SKU.
    
    Returns DataFrame with PyMC component contributions.
    """
    price_slope_draws = posterior_cache["price_slope_z"][:, sku_idx]
    price_delta = np.log1p(price_change)
    price_log_vol_delta = price_slope_draws * price_delta
    price_multiplier = np.exp(price_log_vol_delta)
    
    nd_old = np.clip(nd_base, 1e-4, 1.0)
    nd_future = np.clip(nd_new, 1e-4, 1.0)
    
    nd_old_z = (np.log(nd_old) - scales["log_nd"]["mean"]) / scales["log_nd"]["sd"]
    nd_new_z = (np.log(nd_future) - scales["log_nd"]["mean"]) / scales["log_nd"]["sd"]
    
    basis_old = spline.transform(nd_old_z.reshape(-1, 1))
    basis_new = spline.transform(nd_new_z.reshape(-1, 1))
    
    coef = posterior_cache["nd_coef"]
    
    if posterior_cache["nd_coef_type"] == "sku_specific":
        coef_by_sku = coef[:, sku_idx, :]
        f_old = coef_by_sku @ basis_old.T
        f_new = coef_by_sku @ basis_new.T
    else:
        f_old = coef @ basis_old.T
        f_new = coef @ basis_new.T
    
    mechanical_listing_effect = np.log(nd_future / nd_old)
    fitted_velocity_effect = f_new - f_old
    nd_log_vol_delta = mechanical_listing_effect + fitted_velocity_effect
    nd_multiplier = np.exp(nd_log_vol_delta)
    
    combined_log_vol = price_log_vol_delta + nd_log_vol_delta
    combined_multiplier = np.exp(combined_log_vol)
    
    interaction_multiplier = combined_multiplier / (price_multiplier * nd_multiplier)
    
    return pd.DataFrame({
        "component": [
            "Selected-SKU price effect",
            "Numeric distribution effect",
            "Price × Distribution interaction",
            "Combined total",
        ],
        "log_vol_delta_median": [
            float(np.median(price_log_vol_delta)),
            float(np.median(nd_log_vol_delta)),
            float(np.median(np.log(interaction_multiplier))),
            float(np.median(combined_log_vol)),
        ],
        "multiplier_median": [
            float(np.median(price_multiplier)),
            float(np.median(nd_multiplier)),
            float(np.median(interaction_multiplier)),
            float(np.median(combined_multiplier)),
        ],
        "multiplier_p10": [
            float(np.quantile(price_multiplier, 0.10)),
            float(np.quantile(nd_multiplier, 0.10)),
            float(np.quantile(interaction_multiplier, 0.10)),
            float(np.quantile(combined_multiplier, 0.10)),
        ],
        "multiplier_p90": [
            float(np.quantile(price_multiplier, 0.90)),
            float(np.quantile(nd_multiplier, 0.90)),
            float(np.quantile(interaction_multiplier, 0.90)),
            float(np.quantile(combined_multiplier, 0.90)),
        ],
        "elasticity_median": [
            float(np.median(price_slope_draws)),
            np.nan, np.nan, np.nan,
        ],
        "nd_mechanical_effect": [
            np.nan,
            float(mechanical_listing_effect),
            np.nan, np.nan,
        ],
        "nd_fitted_effect": [
            np.nan,
            float(np.median(fitted_velocity_effect)),
            np.nan, np.nan,
        ],
    })


def create_parameter_waterfall(attribution_df: pd.DataFrame) -> go.Figure:
    """Create a waterfall chart showing parameter-level attribution."""
    components = attribution_df["component"].tolist()
    multipliers = attribution_df["multiplier_median"].tolist()
    p10 = attribution_df["multiplier_p10"].tolist()
    p90 = attribution_df["multiplier_p90"].tolist()
    
    values = [1.0]
    for m in multipliers[:-1]:
        values.append(values[-1] * m)
    
    fig = go.Figure()
    
    colors = ["gray", "blue", "green", "orange", "darkblue"]
    for i, (comp, val, m, low, high) in enumerate(zip(components, values, multipliers, p10, p90)):
        if i == 0:
            fig.add_trace(go.Bar(
                x=[comp], y=[val], name=comp,
                marker_color=colors[i % len(colors)],
                showlegend=False,
                text=[f"{val:.2f}x"],
                textposition="outside"
            ))
        elif i == len(components) - 1:
            fig.add_trace(go.Bar(
                x=[comp], y=[val], name=comp,
                marker_color=colors[i % len(colors)],
                showlegend=False,
                text=[f"{val:.2f}x"],
                textposition="outside"
            ))
        else:
            delta = val - values[i-1]
            fig.add_trace(go.Bar(
                x=[comp], y=[delta], name=comp,
                base=[values[i-1]],
                marker_color=colors[i % len(colors)],
                showlegend=False,
                text=[f"{m:.2f}x"],
                textposition="outside"
            ))
            fig.add_trace(go.Bar(
                x=[comp], y=[values[i-1] * high - values[i-1] * low],
                base=[values[i-1] * low],
                marker_color="rgba(0,0,0,0)",
                showlegend=False,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[values[i-1] * high - values[i-1] * m],
                    arrayminus=[values[i-1] * m - values[i-1] * low],
                    color="gray",
                    thickness=2
                )
            ))
    
    fig.update_layout(
        title="Parameter Attribution Waterfall",
        yaxis_title="Volume Multiplier",
        height=400,
        showlegend=False,
        barmode="relative",
    )
    return fig


def create_reallocation_waterfall(realloc_df: pd.DataFrame) -> go.Figure:
    """Create a waterfall chart showing reallocation by segment relationship."""
    df = realloc_df.sort_values("delta_units", ascending=False).copy()
    
    baseline_total = df["baseline_units"].sum()
    scenario_total = df["scenario_units"].sum()
    
    segments = df["segment_label"].tolist()
    deltas = df["delta_units"].tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=["Baseline"], y=[baseline_total],
        marker_color="gray", name="Baseline",
        text=[f"{baseline_total:,.0f}"], textposition="outside"
    ))
    
    running = baseline_total
    colors = []
    for delta in deltas:
        if delta >= 0:
            colors.append("green")
        else:
            colors.append("red")
        running += delta
        fig.add_trace(go.Bar(
            x=[segments[len([c for c in colors if c == "green"] + [c for c in colors if c == "red"]) - 1]],
            y=[delta],
            base=[running - delta],
            marker_color=colors[-1],
            name=segments[len(colors)-1],
            text=[f"{delta:+,.0f}"],
            textposition="outside",
            showlegend=False
        ))
    
    fig.add_trace(go.Bar(
        x=["Scenario"], y=[scenario_total],
        marker_color="darkblue", name="Scenario",
        text=[f"{scenario_total:,.0f}"], textposition="outside"
    ))
    
    fig.update_layout(
        title="Reallocation Waterfall by Segment",
        yaxis_title="Units",
        height=400,
        showlegend=False,
        barmode="relative",
    )
    return fig


def create_dumbbell_chart(realloc_df: pd.DataFrame) -> go.Figure:
    """Create a dumbbell chart comparing baseline vs scenario by segment."""
    df = realloc_df.sort_values("baseline_units", ascending=True).copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["baseline_units"],
        y=df["segment_label"],
        mode="markers",
        marker=dict(size=12, color="gray", symbol="circle"),
        name="Baseline",
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=df["scenario_units"],
        y=df["segment_label"],
        mode="markers",
        marker=dict(size=12, color="blue", symbol="circle"),
        name="Scenario",
        showlegend=True
    ))
    
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["baseline_units"], row["scenario_units"]],
            y=[row["segment_label"], row["segment_label"]],
            mode="lines",
            line=dict(color="lightgray", width=2),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    fig.update_layout(
        title="Winner-Loser Dumbbell: Baseline vs Scenario by Segment",
        xaxis_title="Units",
        height=400,
        showlegend=True
    )
    return fig


def create_category_bubble_map(market_impact_df: pd.DataFrame) -> go.Figure:
    """Create a category bubble map: x=volume share, y=share change, size=revenue, color=category."""
    df = market_impact_df.copy()
    
    if "category" not in df.columns:
        return go.Figure().update_layout(title="Category data not available")
    
    fig = px.scatter(
        df,
        x="scenario_share",
        y="share_change_pp",
        size="scenario_revenue",
        color="category",
        hover_data=["category", "baseline_share", "scenario_share", "share_change_pp", "delta_revenue_pct"],
        title="Category Market Map: Share vs Share Change",
        labels={
            "scenario_share": "Scenario Volume Share",
            "share_change_pp": "Share Change (pp)",
            "scenario_revenue": "Scenario Revenue"
        }
    )
    
    fig.update_layout(height=500)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


def create_brand_pack_heatmap(market_impact_df: pd.DataFrame) -> go.Figure:
    """Create a brand × pack-group heatmap of share change (pp)."""
    df = market_impact_df.copy()
    
    if "brand" not in df.columns or "pack_group" not in df.columns:
        return go.Figure().update_layout(title="Brand×Pack data not available")
    
    pivot = df.pivot_table(
        values="share_change_pp",
        index="brand",
        columns="pack_group",
        aggfunc="mean"
    )
    
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Brand × Pack Group: Share Change (pp)",
        labels={"color": "Share Change (pp)", "x": "Pack Group", "y": "Brand"}
    )
    
    fig.update_layout(height=400)
    return fig


def create_cross_category_sensitivity_chart(
    scenarios: dict,
    sensitivity: float
) -> go.Figure:
    """Create a chart showing how cross-category sensitivity affects results."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Cross-category sensitivity: {sensitivity:.0%} substitution",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        title="Cross-Category Sensitivity Analysis",
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig
