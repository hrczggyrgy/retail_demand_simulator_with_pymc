"""
Feature engineering utilities for retail demand modeling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_pack_group(pack_size: pd.Series) -> pd.Series:
    """Map continuous pack_size to discrete pack groups."""
    bins = [0, 0.75, 1.25, 2.5, 100]
    labels = ["small", "medium", "large", "xlarge"]
    return pd.cut(pack_size, bins=bins, labels=labels, right=False)


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add core derived fields: unit_price, price_std, nd, standard_volume, velocity_std."""
    df = df.copy()
    df["unit_price"] = df["revenue"] / df["units"]
    df["price_std"] = df["unit_price"] / df["pack_size"]
    df["nd"] = df["sku_stores"] / df["retailer_stores"]
    df["standard_volume"] = df["units"] / df["pack_size"]
    df["velocity_std"] = df["standard_volume"] / df["sku_stores"].replace(0, np.nan)
    return df


def add_pack_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add pack_group column based on pack_size."""
    df = df.copy()
    df["pack_group"] = make_pack_group(df["pack_size"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month_num and other time-based features."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["month_num"] = df["month"].dt.month
    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed features."""
    df = df.copy()
    df["log_nd"] = np.log(df["nd"].clip(lower=1e-4))
    df["log_price_std"] = np.log(df["price_std"].clip(lower=1e-4))
    df["log_velocity_std"] = np.log(df["velocity_std"].clip(lower=1e-4))
    return df


def add_standardized_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Add z-scored features and return scales for reuse."""
    df = df.copy()
    scales: dict[str, dict[str, float]] = {}
    for col in ["log_nd", "log_price_std", "log_velocity_std"]:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            std = 1.0
        scales[col] = {"mean": float(mean), "std": float(std)}
        df[f"{col}_z"] = (df[col] - mean) / std
    return df, scales


def build_spline_basis(
    log_nd: np.ndarray,
    n_knots: int = 4,
    spline_degree: int = 3,
) -> tuple[np.ndarray, Any]:
    """Build B-spline basis matrix for ND effect."""
    from scipy.interpolate import BSpline
    
    knots = np.linspace(log_nd.min(), log_nd.max(), n_knots + 2)[1:-1]
    t = np.concatenate(([knots[0]] * (spline_degree + 1), knots, [knots[-1]] * (spline_degree + 1)))
    nd_basis = BSpline.design_matrix(log_nd, t, spline_degree).toarray()

    class SplineTransformer:
        def __init__(self, knots: np.ndarray, degree: int):
            self.knots = knots
            self.degree = degree
            self.t = np.concatenate(([knots[0]] * (degree + 1), knots, [knots[-1]] * (degree + 1)))

        def transform(self, x: np.ndarray) -> np.ndarray:
            return BSpline.design_matrix(x, self.t, self.degree).toarray()

    spline = SplineTransformer(knots, spline_degree)
    return nd_basis, spline


def apply_standardization(
    df: pd.DataFrame,
    scales: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply pre-computed standardization scales to new data."""
    df = df.copy()
    for col, params in scales.items():
        mean = params["mean"]
        std = params["std"]
        if std == 0:
            std = 1.0
        df[f"{col}_z"] = (df[col] - mean) / std
    return df


def build_all_features(
    raw: pd.DataFrame,
    n_knots: int = 4,
    spline_degree: int = 3,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], Any, np.ndarray]:
    """Full feature engineering pipeline."""
    df = raw.copy()
    df = df.reset_index(drop=True)
    df = add_core_features(df)
    df = add_pack_group(df)
    df = add_time_features(df)
    df = add_log_features(df)
    df, scales = add_standardized_features(df)
    nd_basis, spline = build_spline_basis(df["log_nd"].values, n_knots, spline_degree)
    return df, scales, spline, nd_basis
