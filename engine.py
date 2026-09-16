"""
engine.py - Core analytics engine for retail demand, pricing, and distribution analysis.

Contains all non-UI functionality: data validation, preparation, modelling, scenarios,
and summary computations. No Streamlit dependencies.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from sklearn.preprocessing import SplineTransformer


# ============================================================
# Configuration Constants
# ============================================================

REQUIRED_COLUMNS = [
    "month", "retailer", "category", "brand", "sku",
    "units", "revenue", "sku_stores", "retailer_stores", "pack_size"
]

NUMERIC_COLUMNS = ["units", "revenue", "sku_stores", "retailer_stores", "pack_size"]
KEY_COLUMNS = ["month", "retailer", "category", "brand", "sku"]

# Default model settings
DEFAULT_DRAWS = 500
DEFAULT_TUNE = 500
DEFAULT_CHAINS = 2
DEFAULT_TARGET_ACCEPT = 0.95
DEFAULT_N_SPLINE_KNOTS = 4
DEFAULT_SPLINE_DEGREE = 3

# Advanced model settings
ADVANCED_DRAWS = 1500
ADVANCED_TUNE = 1500
ADVANCED_CHAINS = 4


# ============================================================
# Data Classes for Structured Returns
# ============================================================

@dataclass
class ValidationReport:
    """Results of input data validation."""
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
    """Prepared data with all derived metrics."""
    df: pd.DataFrame
    scales: Dict[str, Dict[str, float]]
    meta: Dict[str, Any]
    spline: SplineTransformer
    nd_basis: np.ndarray


@dataclass
class ModelOutput:
    """Model fitting results."""
    trace: xr.DataTree
    df: pd.DataFrame
    meta: Dict[str, Any]
    spline: SplineTransformer
    scales: Dict[str, Dict[str, float]]
    settings: Dict[str, Any]


@dataclass
class ScenarioResult:
    """Scenario analysis results."""
    df: pd.DataFrame
    price_change: float
    nd_change: float
    summary: pd.DataFrame


# ============================================================
# Data Validation
# ============================================================

def validate_input_data(raw: pd.DataFrame) -> ValidationReport:
    """
    Validate input data against required schema.
    
    Parameters
    ----------
    raw : pd.DataFrame
        Raw input data.
        
    Returns
    -------
    ValidationReport
        Validation results with detailed diagnostics.
    """
    warnings_list = []
    invalid_reasons = {}
    
    # Check missing columns
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
            info={"available_columns": list(raw.columns)}
        )
    
    df = raw.copy()
    original_count = len(df)
    
    # Check duplicates on key columns
    dup_mask = df.duplicated(subset=KEY_COLUMNS, keep=False)
    duplicate_count = int(dup_mask.sum())
    if duplicate_count:
        invalid_reasons["duplicates"] = duplicate_count
        warnings_list.append(f"Found {duplicate_count} duplicate records on key columns")
    
    # Parse month
    try:
        df["month"] = pd.to_datetime(df["month"])
    except Exception as e:
        invalid_reasons["month_parse"] = len(df)
        warnings_list.append(f"Failed to parse month column: {e}")
    
    # Convert numeric columns
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Identify invalid rows
    invalid_mask = pd.Series(False, index=df.index)
    
    # Missing key fields
    key_missing = df[KEY_COLUMNS].isna().any(axis=1)
    invalid_mask |= key_missing
    invalid_reasons["missing_key_fields"] = int(key_missing.sum())
    
    # Missing numeric fields
    num_missing = df[NUMERIC_COLUMNS].isna().any(axis=1)
    invalid_mask |= num_missing
    invalid_reasons["missing_numeric"] = int(num_missing.sum())
    
    # Non-positive values
    for col in NUMERIC_COLUMNS:
        non_pos = (df[col] <= 0)
        invalid_mask |= non_pos
        invalid_reasons[f"non_positive_{col}"] = int(non_pos.sum())
    
    # sku_stores > retailer_stores
    store_invalid = (df["sku_stores"] > df["retailer_stores"])
    invalid_mask |= store_invalid
    invalid_reasons["sku_stores_gt_retailer_stores"] = int(store_invalid.sum())
    if store_invalid.any():
        warnings_list.append(
            f"{int(store_invalid.sum())} rows have sku_stores > retailer_stores; "
            "numeric distribution would exceed 100%"
        )
    
    invalid_count = int(invalid_mask.sum())
    valid_count = original_count - invalid_count
    
    # Data quality warnings
    n_months = df.loc[~invalid_mask, "month"].nunique() if valid_count > 0 else 0
    if n_months < 12:
        warnings_list.append(f"Only {n_months} months of data; model may be unstable")
    
    # Check for single SKU per retailer-category-month
    if valid_count > 0:
        valid_df = df.loc[~invalid_mask].copy()
        cell_counts = valid_df.groupby(["retailer", "category", "month"])["sku"].nunique()
        single_sku_cells = (cell_counts == 1).sum()
        if single_sku_cells > 0:
            warnings_list.append(
                f"{single_sku_cells} retailer-category-month cells have only one SKU; "
                "competitor price cannot be computed"
            )
        
        # Price variation check
        price_std = valid_df.groupby(["retailer", "category", "month"])["revenue"].sum() / \
                    (valid_df.groupby(["retailer", "category", "month"])["units"].sum() * 
                     valid_df.groupby(["retailer", "category", "month"])["pack_size"].first())
        # Simplified check - will be done properly in prepare_data
        
        # Observations per SKU
        sku_obs = valid_df.groupby("sku")["month"].nunique()
        low_obs_skus = (sku_obs < 6).sum()
        if low_obs_skus > 0:
            warnings_list.append(
                f"{low_obs_skus} SKUs have fewer than 6 monthly observations"
            )
        
        # Pack size variation
        pack_per_cat = valid_df.groupby("category")["pack_size"].nunique()
        single_pack_cats = (pack_per_cat == 1).sum()
        if single_pack_cats > 0:
            warnings_list.append(
                f"{single_pack_cats} categories have only one pack size; "
                "pack-size effects cannot be estimated"
            )
    
    info = {
        "original_rows": original_count,
        "valid_rows": valid_count,
        "dropped_rows": invalid_count,
        "n_months": int(n_months),
        "n_retailers": int(df.loc[~invalid_mask, "retailer"].nunique()) if valid_count > 0 else 0,
        "n_categories": int(df.loc[~invalid_mask, "category"].nunique()) if valid_count > 0 else 0,
        "n_brands": int(df.loc[~invalid_mask, "brand"].nunique()) if valid_count > 0 else 0,
        "n_skus": int(df.loc[~invalid_mask, "sku"].nunique()) if valid_count > 0 else 0,
        "date_range": {
            "min": str(df.loc[~invalid_mask, "month"].min()) if valid_count > 0 else None,
            "max": str(df.loc[~invalid_mask, "month"].max()) if valid_count > 0 else None
        }
    }
    
    return ValidationReport(
        is_valid=valid_count > 0,
        row_count=original_count,
        missing_columns=missing,
        duplicate_count=duplicate_count,
        invalid_rows=invalid_count,
        invalid_reasons=invalid_reasons,
        warnings=warnings_list,
        info=info
    )


# ============================================================
# Data Preparation
# ============================================================

def prepare_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Prepare data: compute derived metrics, standardize, create indices.
    
    Parameters
    ----------
    raw : pd.DataFrame
        Validated raw data.
        
    Returns
    -------
    df : pd.DataFrame
        Prepared data with all derived columns.
    scales : dict
        Standardization parameters for log_velocity, log_relative_price, log_nd.
    """
    df = raw.copy()
    df["month"] = pd.to_datetime(df["month"])
    
    # Convert numeric
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing key fields or non-positive numerics
    invalid = (
        df[KEY_COLUMNS].isna().any(axis=1)
        | df[NUMERIC_COLUMNS].isna().any(axis=1)
        | (df["units"] <= 0)
        | (df["revenue"] <= 0)
        | (df["sku_stores"] <= 0)
        | (df["retailer_stores"] <= 0)
        | (df["pack_size"] <= 0)
        | (df["sku_stores"] > df["retailer_stores"])
    )
    if invalid.any():
        warnings.warn(f"Dropping {invalid.sum()} invalid rows during preparation")
        df = df.loc[~invalid].copy()
    
    # 1. Standard physical volume
    df["standard_volume"] = df["units"] * df["pack_size"]
    
    # 2. Standard unit price
    df["price_std"] = df["revenue"] / df["standard_volume"]
    
    # 3. Numeric distribution
    df["nd"] = df["sku_stores"] / df["retailer_stores"]
    
    # 4. Standard velocity per listed store
    df["velocity_std"] = df["standard_volume"] / df["sku_stores"]
    
    # 5. Competitor basket price (excl. focal SKU)
    cell = ["retailer", "category", "month"]
    df["category_revenue"] = df.groupby(cell)["revenue"].transform("sum")
    df["category_std_volume"] = df.groupby(cell)["standard_volume"].transform("sum")
    df["competitor_revenue"] = df["category_revenue"] - df["revenue"]
    df["competitor_std_volume"] = df["category_std_volume"] - df["standard_volume"]
    df["competitor_price_std"] = df["competitor_revenue"] / df["competitor_std_volume"]
    
    # 6. Relative price
    df["relative_price"] = df["price_std"] / df["competitor_price_std"]
    
    # Filter valid competitor prices and distributions
    valid = (
        (df["price_std"] > 0)
        & (df["competitor_price_std"] > 0)
        & (df["nd"] > 0)
        & (df["nd"] <= 1)
        & (df["velocity_std"] > 0)
        & (df["relative_price"] > 0)
    )
    if (~valid).any():
        warnings.warn(f"Dropping {(~valid).sum()} rows without valid competitor price or distribution")
        df = df.loc[valid].copy()
    
    # Log transforms
    df["log_velocity"] = np.log(df["velocity_std"])
    df["log_relative_price"] = np.log(df["relative_price"])
    df["log_nd"] = np.log(df["nd"])
    
    # Standardization
    scales = {}
    for col in ["log_velocity", "log_relative_price", "log_nd"]:
        mean = float(df[col].mean())
        sd = float(df[col].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"'{col}' has no usable variation (mean={mean}, sd={sd})")
        df[f"{col}_z"] = (df[col] - mean) / sd
        scales[col] = {"mean": mean, "sd": sd}
    
    # Pack group (exact pack size as string)
    df["pack_group"] = df["pack_size"].astype(str)
    
    # Final cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if df.empty:
        raise ValueError("No rows remain after cleaning")
    
    return df, scales


def add_indices(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add integer indices for PyMC model coordinates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared data.
        
    Returns
    -------
    df : pd.DataFrame
        Data with index columns added.
    meta : dict
        Metadata including level mappings and SKU info.
    """
    df = df.copy()
    level_map = {}
    
    for col in ["retailer", "sku", "brand", "month", "pack_group"]:
        codes, levels = pd.factorize(df[col], sort=True)
        df[f"{col}_idx"] = codes.astype("int32")
        level_map[col] = levels
    
    # Entity = retailer x SKU combination
    df["entity"] = df["retailer"].astype(str) + "__" + df["sku"].astype(str)
    entity_codes, entity_levels = pd.factorize(df["entity"], sort=True)
    df["entity_idx"] = entity_codes.astype("int32")
    
    # SKU-level metadata
    sku_info = (
        df[["sku_idx", "brand_idx", "pack_group_idx", "sku", "brand", "pack_group", "pack_size"]]
        .sort_values("sku_idx")
        .drop_duplicates("sku_idx")
    )
    
    # Verify SKU indexing is contiguous
    if not np.array_equal(sku_info["sku_idx"].to_numpy(), np.arange(df["sku_idx"].nunique())):
        raise ValueError("SKU indexing is inconsistent")
    
    meta = {
        "retailer_levels": level_map["retailer"],
        "sku_levels": level_map["sku"],
        "brand_levels": level_map["brand"],
        "month_levels": level_map["month"],
        "pack_group_levels": level_map["pack_group"],
        "entity_levels": entity_levels,
        "sku_brand_idx": sku_info["brand_idx"].to_numpy(dtype="int32"),
        "sku_pack_idx": sku_info["pack_group_idx"].to_numpy(dtype="int32"),
        "sku_info": sku_info,
    }
    return df, meta


def fit_spline(df: pd.DataFrame, n_knots: int = DEFAULT_N_SPLINE_KNOTS, 
               degree: int = DEFAULT_SPLINE_DEGREE) -> Tuple[SplineTransformer, np.ndarray]:
    """
    Fit B-spline basis for numeric distribution effect.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared data with log_nd_z column.
    n_knots : int
        Number of spline knots.
    degree : int
        Spline degree.
        
    Returns
    -------
    spline : SplineTransformer
        Fitted spline transformer.
    basis : np.ndarray
        Spline basis matrix (n_obs x n_basis).
    """
    spline = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=False,
        knots="quantile",
        extrapolation="linear",
    )
    basis = spline.fit_transform(df[["log_nd_z"]]).astype("float64")
    return spline, basis


# ============================================================
# Market Metrics and Shares
# ============================================================

def calculate_market_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total market monthly metrics."""
    market = df.groupby("month", as_index=False).agg(
        units=("units", "sum"),
        revenue=("revenue", "sum"),
        standard_volume=("standard_volume", "sum"),
        sku_stores=("sku_stores", "sum"),
        retailer_stores=("retailer_stores", "first"),  # same per retailer per month
    )
    market["price_per_standard_unit"] = market["revenue"] / market["standard_volume"]
    market["avg_nd"] = market["sku_stores"] / market["retailer_stores"]
    market["avg_velocity"] = market["standard_volume"] / market["sku_stores"]
    return market


def calculate_shares(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate various market shares at different hierarchy levels."""
    shares = {}
    
    # SKU shares within retailer-category-month
    cell = ["retailer", "category", "month"]
    df["sku_unit_share"] = df["units"] / df.groupby(cell)["units"].transform("sum")
    df["sku_revenue_share"] = df["revenue"] / df.groupby(cell)["revenue"].transform("sum")
    
    # Brand shares within retailer-category-month
    brand_cell = ["retailer", "category", "brand", "month"]
    brand_rev = df.groupby(brand_cell)["revenue"].transform("sum")
    brand_units = df.groupby(brand_cell)["units"].transform("sum")
    cell_rev = df.groupby(cell)["revenue"].transform("sum")
    cell_units = df.groupby(cell)["units"].transform("sum")
    df["brand_unit_share"] = brand_units / cell_units
    df["brand_revenue_share"] = brand_rev / cell_rev
    
    # Retailer share of total market revenue by month
    total_rev_month = df.groupby("month")["revenue"].transform("sum")
    df["retailer_revenue_share"] = df.groupby(["retailer", "month"])["revenue"].transform("sum") / total_rev_month
    
    # Brand share of total market revenue by month
    df["brand_revenue_share_total"] = df.groupby(["brand", "month"])["revenue"].transform("sum") / total_rev_month
    
    # Category share of total market revenue by month
    df["category_revenue_share_total"] = df.groupby(["category", "month"])["revenue"].transform("sum") / total_rev_month
    
    return {
        "sku_unit_share": df["sku_unit_share"],
        "sku_revenue_share": df["sku_revenue_share"],
        "brand_unit_share": df["brand_unit_share"],
        "brand_revenue_share": df["brand_revenue_share"],
        "retailer_revenue_share": df["retailer_revenue_share"],
        "brand_revenue_share_total": df["brand_revenue_share_total"],
        "category_revenue_share_total": df["category_revenue_share_total"],
    }


def create_growth_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose standard volume growth into network, distribution, velocity effects.
    
    Q_standard = retailer_stores * ND * velocity_std
    ln(Q1/Q0) = ln(retailer_stores1/retailer_stores0) + ln(ND1/ND0) + ln(V1/V0)
    """
    months = np.sort(df["month"].unique())
    if len(months) < 2:
        return pd.DataFrame()
    
    first_month = pd.Timestamp(months[0])
    last_month = pd.Timestamp(months[-1])
    
    start = df[df["month"] == first_month].set_index(["retailer", "sku"])
    end = df[df["month"] == last_month].set_index(["retailer", "sku"])
    common = start.index.intersection(end.index)
    
    if len(common) == 0:
        return pd.DataFrame()
    
    decomp = pd.DataFrame(index=common).reset_index()
    decomp["network"] = 100 * np.log(
        end.loc[common, "retailer_stores"].to_numpy() / start.loc[common, "retailer_stores"].to_numpy()
    )
    decomp["distribution"] = 100 * np.log(
        end.loc[common, "nd"].to_numpy() / start.loc[common, "nd"].to_numpy()
    )
    decomp["velocity"] = 100 * np.log(
        end.loc[common, "velocity_std"].to_numpy() / start.loc[common, "velocity_std"].to_numpy()
    )
    decomp["total_standard_volume"] = 100 * np.log(
        end.loc[common, "standard_volume"].to_numpy() / start.loc[common, "standard_volume"].to_numpy()
    )
    decomp["label"] = decomp["retailer"].astype(str) + " | " + decomp["sku"].astype(str)
    decomp = decomp.sort_values("total_standard_volume")
    
    return decomp


# ============================================================
# Summary Tables
# ============================================================

def create_market_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create total market summary table."""
    market = calculate_market_metrics(df)
    return market


def create_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create category-level summary."""
    cat = df.groupby(["month", "category"], as_index=False).agg(
        revenue=("revenue", "sum"),
        units=("units", "sum"),
        standard_volume=("standard_volume", "sum"),
        price_std=("price_std", "mean"),
        nd=("nd", "mean"),
        velocity_std=("velocity_std", "mean"),
    )
    cat["price_per_standard_unit"] = cat["revenue"] / cat["standard_volume"]
    return cat


def create_retailer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create retailer-level summary."""
    ret = df.groupby(["month", "retailer"], as_index=False).agg(
        revenue=("revenue", "sum"),
        units=("units", "sum"),
        standard_volume=("standard_volume", "sum"),
        price_std=("price_std", "mean"),
        nd=("nd", "mean"),
        velocity_std=("velocity_std", "mean"),
    )
    ret["price_per_standard_unit"] = ret["revenue"] / ret["standard_volume"]
    return ret


def create_brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create brand-level summary."""
    brand = df.groupby(["month", "brand"], as_index=False).agg(
        revenue=("revenue", "sum"),
        units=("units", "sum"),
        standard_volume=("standard_volume", "sum"),
        price_std=("price_std", "mean"),
        nd=("nd", "mean"),
        velocity_std=("velocity_std", "mean"),
    )
    brand["price_per_standard_unit"] = brand["revenue"] / brand["standard_volume"]
    return brand


def create_sku_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create SKU-level summary with all metrics."""
    sku = df.groupby(["category", "retailer", "brand", "sku", "pack_size"], as_index=False).agg(
        units=("units", "sum"),
        revenue=("revenue", "sum"),
        standard_volume=("standard_volume", "sum"),
        price_std=("price_std", "mean"),
        relative_price=("relative_price", "mean"),
        nd=("nd", "mean"),
        velocity_std=("velocity_std", "mean"),
        sku_stores=("sku_stores", "mean"),
        retailer_stores=("retailer_stores", "mean"),
    )
    sku["price_per_standard_unit"] = sku["revenue"] / sku["standard_volume"]
    
    # Add shares
    total_rev = df["revenue"].sum()
    total_units = df["units"].sum()
    sku["revenue_share"] = sku["revenue"] / total_rev
    sku["unit_share"] = sku["units"] / total_units
    
    return sku


def create_data_quality_report(raw: pd.DataFrame, prepared: pd.DataFrame, 
                                validation: ValidationReport) -> Dict[str, Any]:
    """Create comprehensive data quality report."""
    report = {
        "input_row_count": validation.row_count,
        "valid_model_row_count": len(prepared),
        "dropped_row_count": validation.invalid_rows,
        "dropped_reasons": validation.invalid_reasons,
        "missing_values": raw.isnull().sum().to_dict(),
        "duplicate_records": validation.duplicate_count,
        "date_coverage": {
            "min": str(prepared["month"].min()) if len(prepared) > 0 else None,
            "max": str(prepared["month"].max()) if len(prepared) > 0 else None,
            "n_months": prepared["month"].nunique() if len(prepared) > 0 else 0,
        },
        "counts": {
            "retailers": prepared["retailer"].nunique() if len(prepared) > 0 else 0,
            "categories": prepared["category"].nunique() if len(prepared) > 0 else 0,
            "brands": prepared["brand"].nunique() if len(prepared) > 0 else 0,
            "skus": prepared["sku"].nunique() if len(prepared) > 0 else 0,
            "pack_sizes": prepared["pack_size"].nunique() if len(prepared) > 0 else 0,
        },
        "distribution_validity": {
            "nd_min": float(prepared["nd"].min()) if len(prepared) > 0 else None,
            "nd_max": float(prepared["nd"].max()) if len(prepared) > 0 else None,
            "nd_gt_1_count": int((prepared["nd"] > 1).sum()) if len(prepared) > 0 else 0,
        },
        "price_variation": {
            "rel_price_min": float(prepared["relative_price"].min()) if len(prepared) > 0 else None,
            "rel_price_max": float(prepared["relative_price"].max()) if len(prepared) > 0 else None,
            "rel_price_std": float(prepared["relative_price"].std()) if len(prepared) > 0 else None,
        },
        "observations_per_retailer_sku": prepared.groupby(["retailer", "sku"])["month"].nunique().describe().to_dict() if len(prepared) > 0 else {},
        "observations_per_sku": prepared.groupby("sku")["month"].nunique().describe().to_dict() if len(prepared) > 0 else {},
        "low_observation_skus": prepared.groupby("sku")["month"].nunique()[prepared.groupby("sku")["month"].nunique() < 6].index.tolist() if len(prepared) > 0 else [],
        "warnings": validation.warnings,
    }
    return report


# ============================================================
# PyMC Model
# ============================================================

def build_pymc_model(df: pd.DataFrame, meta: Dict[str, Any], 
                      nd_basis: np.ndarray) -> pm.Model:
    """
    Build hierarchical PyMC model for log velocity.
    
    log(V_standard) = alpha + entity_effect + month_effect + 
                      beta_price * log_relative_price + f(ND) + error
    """
    coords = {
        "observation": np.arange(len(df)),
        "retailer": meta["retailer_levels"],
        "sku": meta["sku_levels"],
        "brand": meta["brand_levels"],
        "month": meta["month_levels"].astype(str),
        "pack_group": meta["pack_group_levels"],
        "entity": meta["entity_levels"],
        "nd_basis": np.arange(nd_basis.shape[1]),
    }
    
    with pm.Model(coords=coords) as model:
        # Data containers
        retailer_idx = pm.Data("retailer_idx", df["retailer_idx"].to_numpy(), dims="observation")
        sku_idx = pm.Data("sku_idx", df["sku_idx"].to_numpy(), dims="observation")
        month_idx = pm.Data("month_idx", df["month_idx"].to_numpy(), dims="observation")
        entity_idx = pm.Data("entity_idx", df["entity_idx"].to_numpy(), dims="observation")
        x_price = pm.Data("x_price", df["log_relative_price_z"].to_numpy(), dims="observation")
        x_nd = pm.Data("x_nd", nd_basis, dims=("observation", "nd_basis"))
        
        # Global intercept
        alpha = pm.Normal("alpha", 0, 1)
        
        # Entity effect (retailer x SKU persistent effect)
        sigma_entity = pm.HalfNormal("sigma_entity", 0.5)
        entity_offset = pm.Normal("entity_offset", 0, 1, dims="entity")
        entity_effect = pm.Deterministic("entity_effect", entity_offset * sigma_entity, dims="entity")
        
        # Month effect (seasonality)
        sigma_month = pm.HalfNormal("sigma_month", 0.5)
        month_offset = pm.Normal("month_offset", 0, 1, dims="month")
        month_effect = pm.Deterministic("month_effect", month_offset * sigma_month, dims="month")
        
        # SKU price response with brand x pack-size partial pooling
        price_mean = pm.Normal("price_mean", -0.4, 0.5, dims=("brand", "pack_group"))
        sigma_price_sku = pm.HalfNormal("sigma_price_sku", 0.3)
        price_offset = pm.Normal("price_offset", 0, 1, dims="sku")
        price_slope = pm.Deterministic(
            "price_slope_z",
            price_mean[meta["sku_brand_idx"], meta["sku_pack_idx"]] + price_offset * sigma_price_sku,
            dims="sku",
        )
        
        # Non-linear numeric distribution effect, grouped by brand and pack size
        nd_mean = pm.Normal("nd_mean", 0, 0.4, dims=("brand", "pack_group", "nd_basis"))
        sigma_nd_sku = pm.HalfNormal("sigma_nd_sku", 0.25)
        nd_offset = pm.Normal("nd_offset", 0, 1, dims=("sku", "nd_basis"))
        nd_weights = pm.Deterministic(
            "nd_weights",
            nd_mean[meta["sku_brand_idx"], meta["sku_pack_idx"], :] + nd_offset * sigma_nd_sku,
            dims=("sku", "nd_basis"),
        )
        nd_effect = pm.math.sum(nd_weights[sku_idx] * x_nd, axis=1)
        
        # Linear predictor
        mu = (
            alpha
            + entity_effect[entity_idx]
            + month_effect[month_idx]
            + price_slope[sku_idx] * x_price
            + nd_effect
        )
        
        # Observation noise
        sigma = pm.HalfNormal("sigma", 0.5)
        pm.Normal("velocity_z_obs", mu=mu, sigma=sigma, 
                  observed=df["log_velocity_z"].to_numpy(), dims="observation")
    
    return model


def fit_model(model: pm.Model, draws: int = DEFAULT_DRAWS, tune: int = DEFAULT_TUNE,
              chains: int = DEFAULT_CHAINS, target_accept: float = DEFAULT_TARGET_ACCEPT,
              random_seed: int = 42) -> xr.DataTree:
    """Fit PyMC model and generate posterior predictive samples."""
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )
        trace = pm.sample_posterior_predictive(
            trace,
            var_names=["velocity_z_obs"],
            random_seed=random_seed,
            extend_inferencedata=True,
        )
    return trace


# ============================================================
# Posterior Extraction and Elasticities
# ============================================================

def _get_posterior_array(trace: xr.DataTree, variable: str, dimensions: Tuple[str, ...]) -> np.ndarray:
    """Extract posterior array with consistent handling of DataTree/InferenceData."""
    if hasattr(trace, "__getitem__") and "posterior" in trace:
        posterior = trace["posterior"]
        if hasattr(posterior, "ds"):
            posterior = posterior.ds
        elif hasattr(posterior, "dataset"):
            posterior = posterior.dataset
    elif hasattr(trace, "posterior"):
        posterior = trace.posterior
    else:
        raise KeyError("Trace has no 'posterior' group")
    
    arr = posterior[variable].stack(sample=("chain", "draw")).transpose(*dimensions, "sample").to_numpy()
    return arr


def extract_elasticities(trace: xr.DataTree, meta: Dict[str, Any], 
                         scales: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Extract SKU-level price elasticities with credible intervals.
    
    Raw elasticity = beta_standardised * SD(log velocity) / SD(log relative price)
    """
    beta_z = _get_posterior_array(trace, "price_slope_z", ("sku",))
    beta_raw = beta_z * scales["log_velocity"]["sd"] / scales["log_relative_price"]["sd"]
    
    sku_results = meta["sku_info"].copy()
    sku_results["elasticity_p05"] = np.quantile(beta_raw, 0.05, axis=1)
    sku_results["elasticity_median"] = np.median(beta_raw, axis=1)
    sku_results["elasticity_p95"] = np.quantile(beta_raw, 0.95, axis=1)
    
    return sku_results


def get_posterior_predictive_check(trace: xr.DataTree, df: pd.DataFrame) -> pd.DataFrame:
    """Get posterior predictive check data."""
    try:
        if hasattr(trace, "__getitem__") and "posterior_predictive" in trace:
            ppc = trace["posterior_predictive"]
            if hasattr(ppc, "ds"):
                ppc = ppc.ds
            elif hasattr(ppc, "dataset"):
                ppc = ppc.dataset
        elif hasattr(trace, "posterior_predictive"):
            ppc = trace.posterior_predictive
        else:
            raise KeyError("No posterior_predictive group")
        
        ppc_mean = ppc["velocity_z_obs"].mean(dim=("chain", "draw")).to_numpy()
        ppc_df = pd.DataFrame({
            "observed_z": df["log_velocity_z"].to_numpy(),
            "predicted_z": ppc_mean,
        })
        ppc_df["residual_z"] = ppc_df["observed_z"] - ppc_df["predicted_z"]
        return ppc_df
    except Exception as e:
        warnings.warn(f"PPC extraction failed: {e}")
        return pd.DataFrame()


def get_model_diagnostics(trace: xr.DataTree) -> Dict[str, Any]:
    """Extract model diagnostics: divergences, R-hat, ESS."""
    diagnostics = {
        "divergences": 0,
        "max_rhat": 1.0,
        "min_ess": float('inf'),
    }
    
    try:
        # Divergences
        if hasattr(trace, "sample_stats"):
            if hasattr(trace.sample_stats, "diverging"):
                diagnostics["divergences"] = int(trace.sample_stats.diverging.sum())
            elif "diverging" in trace.sample_stats:
                diagnostics["divergences"] = int(trace.sample_stats["diverging"].sum())
        
        # R-hat and ESS from posterior
        import arviz as az
        summary = az.summary(trace, var_names=["alpha", "sigma", "sigma_entity", "sigma_month", 
                                                "sigma_price_sku", "sigma_nd_sku"])
        if not summary.empty:
            diagnostics["max_rhat"] = float(summary["r_hat"].max())
            diagnostics["min_ess"] = float(summary["ess_bulk"].min())
    except Exception:
        pass
    
    return diagnostics


# ============================================================
# Scenario Analysis
# ============================================================

def create_price_distribution_scenario(
    trace: xr.DataTree,
    df: pd.DataFrame,
    spline: SplineTransformer,
    scales: Dict[str, Dict[str, float]],
    price_change: float,
    nd_change: float,
) -> pd.DataFrame:
    """
    Create price and/or distribution change scenario.
    
    Parameters
    ----------
    trace : xr.DataTree
        Fitted model trace.
    df : pd.DataFrame
        Prepared data (filtered to scenario universe).
    spline : SplineTransformer
        Fitted spline for ND effect.
    scales : dict
        Standardization scales.
    price_change : float
        Relative price change (e.g., 0.05 for +5%).
    nd_change : float
        Relative ND change (e.g., 0.10 for +10%).
        
    Returns
    -------
    pd.DataFrame
        Scenario results with p10, median, p90 multipliers.
    """
    price_slope_z = _get_posterior_array(trace, "price_slope_z", ("sku",))
    nd_weights = _get_posterior_array(trace, "nd_weights", ("sku", "nd_basis"))
    
    sku_idx = df["sku_idx"].to_numpy()
    nd_old = df["nd"].to_numpy()
    nd_new = np.clip(nd_old * (1 + nd_change), 1e-4, 1.0)
    
    # Standardized ND
    nd_old_z = (np.log(nd_old) - scales["log_nd"]["mean"]) / scales["log_nd"]["sd"]
    nd_new_z = (np.log(nd_new) - scales["log_nd"]["mean"]) / scales["log_nd"]["sd"]
    
    # Spline basis difference
    delta_basis = spline.transform(nd_new_z.reshape(-1, 1)) - spline.transform(nd_old_z.reshape(-1, 1))
    
    # ND effect change (on z-scale)
    nd_delta_z = np.einsum("ok,oks->os", delta_basis, nd_weights[sku_idx])
    
    # Price effect change (on z-scale)
    price_delta_z = price_slope_z[sku_idx] * np.log1p(price_change) / scales["log_relative_price"]["sd"]
    
    # Combined effect on log velocity
    delta_log_velocity = (nd_delta_z + price_delta_z) * scales["log_velocity"]["sd"]
    
    # Volume multiplier = velocity ratio * ND ratio
    velocity_ratio = np.exp(delta_log_velocity)
    total_volume_ratio = velocity_ratio * (nd_new / nd_old)[:, None]
    
    out = df[["month", "retailer", "category", "brand", "sku", "pack_size", 
              "units", "revenue", "sku_stores", "retailer_stores", "nd", "price_std"]].copy()
    out["scenario_price_change"] = price_change
    out["scenario_nd_change"] = nd_change
    out["scenario_nd"] = nd_new
    out["volume_multiplier_p10"] = np.quantile(total_volume_ratio, 0.10, axis=1)
    out["volume_multiplier_median"] = np.median(total_volume_ratio, axis=1)
    out["volume_multiplier_p90"] = np.quantile(total_volume_ratio, 0.90, axis=1)
    out["expected_units_median"] = out["units"] * out["volume_multiplier_median"]
    out["expected_revenue_median"] = out["revenue"] * (1 + price_change) * out["volume_multiplier_median"]
    
    return out


def aggregate_scenario(scenario_df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Aggregate scenario results by specified columns."""
    agg = scenario_df.groupby(by, as_index=False).agg(
        baseline_units=("units", "sum"),
        expected_units_p10=("expected_units_median", lambda x: np.quantile(x, 0.10)),
        expected_units_median=("expected_units_median", "median"),
        expected_units_p90=("expected_units_median", lambda x: np.quantile(x, 0.90)),
        baseline_revenue=("revenue", "sum"),
        expected_revenue_median=("expected_revenue_median", "sum"),
    )
    agg["volume_change_pct"] = 100 * (agg["expected_units_median"] / agg["baseline_units"] - 1)
    agg["revenue_change_pct"] = 100 * (agg["expected_revenue_median"] / agg["baseline_revenue"] - 1)
    return agg





# ============================================================
# Distribution-Velocity Classification
# ============================================================

def classify_distribution_velocity(df: pd.DataFrame, benchmark_level: str = "retailer_category") -> pd.DataFrame:
    """
    Classify SKUs into distribution-velocity quadrants.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared data with nd, velocity_std, brand, pack_size.
    benchmark_level : str
        Level for velocity benchmark: 'retailer_category', 'category', or 'total'.
        
    Returns
    -------
    pd.DataFrame
        Data with relative_velocity and quadrant classification.
    """
    df = df.copy()
    
    if benchmark_level == "retailer_category":
        bench = df.groupby(["retailer", "category", "month"])["velocity_std"].transform("median")
    elif benchmark_level == "category":
        bench = df.groupby(["category", "month"])["velocity_std"].transform("median")
    else:
        bench = df.groupby("month")["velocity_std"].transform("median")
    
    df["relative_velocity"] = df["velocity_std"] / bench
    
    # Medians for quadrant boundaries
    nd_median = df["nd"].median()
    rv_median = df["relative_velocity"].median()
    
    def classify(row):
        if row["nd"] < nd_median and row["relative_velocity"] > rv_median:
            return "Expansion candidate"
        elif row["nd"] >= nd_median and row["relative_velocity"] > rv_median:
            return "Scale efficiently"
        elif row["nd"] < nd_median and row["relative_velocity"] <= rv_median:
            return "Diagnose"
        else:
            return "Rationalise/review"
    
    df["dv_quadrant"] = df.apply(classify, axis=1)
    return df


def get_expansion_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Get SKUs classified as expansion candidates (low ND, high velocity)."""
    classified = classify_distribution_velocity(df)
    return classified[classified["dv_quadrant"] == "Expansion candidate"].copy()


# ============================================================
# Utility: Convert trace to netcdf-compatible format
# ============================================================

def save_trace(trace: xr.DataTree, path: Path) -> None:
    """Save trace using native xarray/netcdf method."""
    trace.to_netcdf(path)


def load_trace(path: Path) -> xr.DataTree:
    """Load trace from netcdf."""
    return xr.open_datatree(path)