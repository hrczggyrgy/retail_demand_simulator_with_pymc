"""
Validation and data preparation functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from contracts import (
    KEY_COLUMNS,
    REQUIRED_COLUMNS,
    ValidationReport,
)
from contracts import (
    NUMERIC_RAW_COLUMNS as NUMERIC_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class PreparedData:
    df: pd.DataFrame
    scales: dict[str, dict[str, float]]
    meta: dict[str, Any]
    spline: Any  # SplineTransformer
    nd_basis: np.ndarray


def make_pack_group(pack_size: pd.Series) -> pd.Series:
    """Map continuous pack_size to discrete pack groups."""
    bins = [0, 0.75, 1.25, 2.5, 100]
    labels = ["small", "medium", "large", "xlarge"]
    return pd.cut(pack_size, bins=bins, labels=labels, right=False)


def validate_input_data(raw: pd.DataFrame) -> ValidationReport:
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        return ValidationReport(
            is_valid=False,
            errors=tuple([f"Missing required columns: {missing}"]),
            warnings=tuple(),
            row_count=len(raw),
            column_count=len(raw.columns),
            missing_columns=tuple(missing),
            extra_columns=tuple(),
        )

    dup_count = int(raw.duplicated(subset=KEY_COLUMNS, keep=False).sum())
    if dup_count > 0:
        return ValidationReport(
            is_valid=False,
            errors=(f"Duplicate keys found on {', '.join(KEY_COLUMNS)}.",),
            warnings=tuple(),
            row_count=len(raw),
            column_count=len(raw.columns),
            missing_columns=tuple(),
            extra_columns=tuple(),
        )

    key_missing = raw[KEY_COLUMNS].isna().any(axis=1)
    if key_missing.any():
        return ValidationReport(
            is_valid=False,
            errors=(f"Missing key values in {key_missing.sum()} rows.",),
            warnings=tuple(),
            row_count=len(raw),
            column_count=len(raw.columns),
            missing_columns=tuple(),
            extra_columns=tuple(),
        )

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(raw[col]):
            return ValidationReport(
                is_valid=False,
                errors=(f"Column '{col}' must be numeric.",),
                warnings=tuple(),
                row_count=len(raw),
                column_count=len(raw.columns),
                missing_columns=tuple(),
                extra_columns=tuple(),
            )

    warnings = []
    numeric_missing = raw[NUMERIC_COLUMNS].isna().any(axis=1)
    if numeric_missing.any():
        warnings.append(f"{numeric_missing.sum()} rows have missing numeric values")

    for col in NUMERIC_COLUMNS:
        neg = (raw[col] < 0).sum()
        if neg:
            warnings.append(f"{neg} rows have negative {col}")

    if (raw["sku_stores"] > raw["retailer_stores"]).any():
        warnings.append("Some rows have sku_stores > retailer_stores")

    if (raw["units"] <= 0).any():
        warnings.append("Some rows have non-positive units")

    if (raw["revenue"] <= 0).any():
        warnings.append("Some rows have non-positive revenue")

    return ValidationReport(
        is_valid=True,
        errors=tuple(),
        warnings=tuple(warnings),
        row_count=len(raw),
        column_count=len(raw.columns),
        missing_columns=tuple(),
        extra_columns=tuple(),
    )


def prepare_data(
    raw: pd.DataFrame,
    n_knots: int = 4,
    spline_degree: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full validation + feature engineering pipeline. Returns (df, meta)."""
    report = validate_input_data(raw)
    if not report.is_valid:
        raise ValueError("Validation failed: " + "; ".join(report.errors))

    df = raw.copy()
    df = df.reset_index(drop=True)

    # Ensure month is datetime
    df["month"] = pd.to_datetime(df["month"])

    # --- Data quality flags ---
    valid_units = df["units"] > 0
    valid_revenue = df["revenue"] > 0
    valid_pack_size = df["pack_size"] > 0
    valid_store_counts = (
        (df["sku_stores"] > 0)
        & (df["retailer_stores"] > 0)
        & (df["sku_stores"] <= df["retailer_stores"])
    )
    valid_hierarchy = (
        df["category"].notna()
        & df["brand"].notna()
        & df["sku"].notna()
        & df["retailer"].notna()
    )

    conditions = [
        ~valid_units,
        ~valid_revenue,
        ~valid_pack_size,
        ~valid_store_counts,
        ~valid_hierarchy,
    ]
    choices = [
        "zero_or_negative_units",
        "zero_or_negative_revenue",
        "invalid_pack_size",
        "invalid_store_counts",
        "missing_hierarchy",
    ]
    df["data_quality_status"] = np.select(conditions, choices, default="valid")

    # Core derived fields
    df["standard_volume"] = df["units"] / df["pack_size"]
    df["price_std"] = df["revenue"] / df["standard_volume"]
    df["nd"] = df["sku_stores"] / df["retailer_stores"]
    df["velocity_std"] = df["standard_volume"] / df["sku_stores"].replace(0, np.nan)
    df["unit_price"] = df["revenue"] / df["units"]

    # Market context columns
    cat_std_vol = df.groupby(["month", "retailer", "category"])["standard_volume"].transform("sum")
    df["category_std_volume"] = cat_std_vol

    brand_std_vol = df.groupby(["month", "retailer", "category", "brand"])["standard_volume"].transform("sum")
    df["brand_std_volume"] = brand_std_vol

    df["brand_share"] = np.where(
        df["category_std_volume"] > 0,
        df["brand_std_volume"] / df["category_std_volume"],
        np.nan,
    )

    # Relative price against category/pack-peer price basket
    peer_groups = ["retailer", "category", "pack_group", "month"]
    df["pack_group"] = make_pack_group(df["pack_size"])
    peer_rev = df.groupby(peer_groups, observed=True)["revenue"].transform("sum")
    peer_vol = df.groupby(peer_groups, observed=True)["standard_volume"].transform("sum")
    df["peer_price_std"] = peer_rev / peer_vol
    df["relative_price"] = df["price_std"] / df["peer_price_std"]
    df["relative_price"] = df["relative_price"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Month number for seasonality
    df["month_num"] = df["month"].dt.month

    # Log transforms
    df["log_nd"] = np.log(df["nd"].clip(lower=1e-4))
    df["log_price_std"] = np.log(df["price_std"].clip(lower=1e-4))
    df["log_velocity_std"] = np.log(df["velocity_std"].clip(lower=1e-4))
    df["log_relative_price"] = np.log(df["relative_price"].clip(lower=1e-4))

    # Standardize (z-score)
    scales: dict[str, dict[str, float]] = {}
    for col in ["log_nd", "log_price_std", "log_velocity_std", "log_relative_price"]:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            std = 1.0
        scales[col] = {"mean": mean, "std": std}
        df[f"{col}_z"] = (df[col] - mean) / std

    # Build spline basis for ND
    from scipy.interpolate import BSpline
    
    log_nd_vals = df["log_nd"].values
    log_nd_min = float(log_nd_vals.min())
    log_nd_max = float(log_nd_vals.max())
    log_nd_range = log_nd_max - log_nd_min
    
    min_knots = spline_degree + 1
    if log_nd_range < n_knots * 0.1:
        effective_knots = max(min_knots, min(n_knots, 2))
        center = float(log_nd_vals.mean())
        internal_knots = np.linspace(center - 2.0, center + 2.0, effective_knots + 2)[1:-1]
    else:
        internal_knots = np.linspace(log_nd_min, log_nd_max, n_knots + 2)[1:-1]
    
    if len(np.unique(internal_knots)) < min_knots:
        internal_knots = internal_knots + np.linspace(-0.01, 0.01, len(internal_knots))
    
    # Full knot vector: boundaries + internal knots + boundaries
    t = np.concatenate((
        [log_nd_min] * (spline_degree + 1),
        internal_knots,
        [log_nd_max] * (spline_degree + 1),
    ))
    nd_basis = BSpline.design_matrix(log_nd_vals, t, spline_degree).toarray()

    class SplineTransformer:
        def __init__(self, internal_knots: np.ndarray, degree: int, left_bound: float, right_bound: float):
            self.internal_knots = internal_knots
            self.degree = degree
            self.left_bound = left_bound
            self.right_bound = right_bound
            self.t = np.concatenate((
                [left_bound] * (degree + 1),
                internal_knots,
                [right_bound] * (degree + 1),
            ))

        def transform(self, x: np.ndarray) -> np.ndarray:
            return BSpline.design_matrix(x, self.t, self.degree).toarray()

    spline = SplineTransformer(internal_knots, spline_degree, log_nd_min, log_nd_max)

    meta = {
        "n_rows": len(df),
        "n_skus": df["sku"].nunique(),
        "n_retailers": df["retailer"].nunique(),
        "n_categories": df["category"].nunique(),
        "n_brands": df["brand"].nunique(),
        "months": sorted(df["month"].unique()),
    }

    return df, meta


def build_retail_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convenience function returning only the enriched DataFrame."""
    return prepare_data(raw)[0]
