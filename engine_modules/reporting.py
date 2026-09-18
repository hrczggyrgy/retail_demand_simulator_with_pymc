"""
Reporting and aggregation utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Common aggregation metrics for descriptive summaries
_SUMMARY_METRICS = {
    "revenue": ("revenue", "sum"),
    "units": ("units", "sum"),
    "standard_volume": ("standard_volume", "sum"),
    "avg_price_std": ("price_std", "mean"),
    "avg_nd": ("nd", "mean"),
    "avg_velocity_std": ("velocity_std", "mean"),
}


def aggregate_scenario(
    scenario_df: pd.DataFrame,
    by: list[str],
) -> pd.DataFrame:
    """Aggregate scenario results by specified columns."""
    if by:
        grouped = scenario_df.groupby(by, observed=True)
    else:
        grouped = [("", scenario_df)]
    
    return grouped.agg(
        baseline_units=("baseline_units", "sum"),
        scenario_units=("scenario_units", "sum"),
        delta_units=("delta_units", "sum"),
        delta_units_pct=("delta_units_pct", "mean"),
    ).reset_index()


def calculate_shares(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Calculate market shares at various levels.
    
    Returns dict with share series for different groupings.
    """
    shares = {}
    
    # Brand share within retailer×category×month
    total = df.groupby(["retailer", "category", "month"])["standard_volume"].transform("sum")
    shares["brand_share"] = df["standard_volume"] / total
    
    # SKU share within brand×retailer×category×month
    brand_total = df.groupby(["retailer", "category", "brand", "month"])["standard_volume"].transform("sum")
    shares["sku_share"] = df["standard_volume"] / brand_total
    
    return shares


def decompose_sales_growth(
    df: pd.DataFrame,
    start_month: str,
    end_month: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Decompose sales growth into volume/price/mix components.
    
    Uses the decomposition identity:
    standard_volume = retailer_stores × nd × velocity_std
    """
    if group_cols is None:
        group_cols = ["retailer", "category", "brand", "sku"]
    
    start = pd.Timestamp(start_month)
    end = pd.Timestamp(end_month)
    
    mask_start = df["month"] == start
    mask_end = df["month"] == end
    
    df_start = df[mask_start].copy()
    df_end = df[mask_end].copy()
    
    # Merge on group_cols
    merged = df_start[group_cols + ["standard_volume", "units", "revenue", "sku_stores", "retailer_stores", "nd", "velocity_std", "price_std"]].merge(
        df_end[group_cols + ["standard_volume", "units", "revenue", "sku_stores", "retailer_stores", "nd", "velocity_std", "price_std"]],
        on=group_cols,
        suffixes=("_start", "_end"),
    )
    
    merged["volume_change"] = merged["standard_volume_end"] - merged["standard_volume_start"]
    merged["pct_change"] = np.where(
        merged["standard_volume_start"] > 0,
        merged["volume_change"] / merged["standard_volume_start"] * 100,
        0,
    )
    
    # Decompose using log-differential
    merged["stores_contrib"] = np.log(merged["retailer_stores_end"] / merged["retailer_stores_start"].clip(lower=1))
    merged["nd_contrib"] = np.log(merged["nd_end"].clip(lower=1e-4) / merged["nd_start"].clip(lower=1e-4))
    merged["vel_contrib"] = np.log(merged["velocity_std_end"].clip(lower=1e-4) / merged["velocity_std_start"].clip(lower=1e-4))
    
    return merged


def prepare_market(
    raw: pd.DataFrame,
    n_knots: int = 4,
    spline_degree: int = 3,
) -> dict[str, Any]:
    """
    High-level market preparation for the improved UI.
    
    Returns dict with validated data, features, and metadata.
    """
    from engine_modules.validation import prepare_data
    
    prepared = prepare_data(raw, n_knots, spline_degree)
    
    return {
        "df": prepared.df,
        "scales": prepared.scales,
        "meta": prepared.meta,
        "spline": prepared.spline,
        "nd_basis": prepared.nd_basis,
    }


def build_market_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Build market overview summary for dashboard."""
    total_volume = df["standard_volume"].sum()
    total_revenue = df["revenue"].sum()
    total_units = df["units"].sum()
    avg_price = total_revenue / total_units if total_units > 0 else 0
    
    n_skus = df["sku"].nunique()
    n_brands = df["brand"].nunique()
    n_retailers = df["retailer"].nunique()
    n_categories = df["category"].nunique()
    n_months = df["month"].nunique()
    
    return {
        "total_standard_volume": total_volume,
        "total_revenue": total_revenue,
        "total_units": total_units,
        "avg_unit_price": avg_price,
        "n_skus": n_skus,
        "n_brands": n_brands,
        "n_retailers": n_retailers,
        "n_categories": n_categories,
        "n_months": n_months,
    }


def build_price_pack_architecture(df: pd.DataFrame) -> pd.DataFrame:
    """Build price-pack architecture table."""
    return df.groupby(["pack_group", "brand"]).agg(
        n_skus=("sku", "nunique"),
        avg_price_std=("price_std", "mean"),
        avg_velocity_std=("velocity_std", "mean"),
        total_volume=("standard_volume", "sum"),
    ).reset_index()


def build_nd_velocity_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    """Build ND-Velocity quadrant classification."""
    df = df.copy()
    nd_median = df["nd"].median()
    vel_median = df["velocity_std"].median()
    
    df["quadrant"] = pd.cut(
        df["nd"],
        bins=[-np.inf, nd_median, np.inf],
        labels=["Low ND", "High ND"],
    )
    # Actually do 2x2
    df["nd_quad"] = np.where(df["nd"] > nd_median, "High ND", "Low ND")
    df["vel_quad"] = np.where(df["velocity_std"] > vel_median, "High Velocity", "Low Velocity")
    df["quadrant"] = df["nd_quad"] + " / " + df["vel_quad"]
    
    return df.groupby("quadrant").agg(
        n_skus=("sku", "nunique"),
        avg_nd=("nd", "mean"),
        avg_velocity=("velocity_std", "mean"),
        total_volume=("standard_volume", "sum"),
    ).reset_index()
