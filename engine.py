
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
from sklearn.preprocessing import SplineTransformer


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

DEFAULT_DRAWS = 400
DEFAULT_TUNE = 400
DEFAULT_CHAINS = 2
DEFAULT_TARGET_ACCEPT = 0.90
DEFAULT_N_SPLINE_KNOTS = 4
DEFAULT_SPLINE_DEGREE = 3

ADVANCED_DRAWS = 1000
ADVANCED_TUNE = 1000
ADVANCED_CHAINS = 4

# How many posterior draws to retain for interactive scenarios.
DEFAULT_SCENARIO_DRAWS = 400


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

    df["competitor_revenue"] = df["category_revenue"] - df["revenue"]
    df["competitor_std_volume"] = (
        df["category_std_volume"] - df["standard_volume"]
    )

    df["competitor_price_std"] = (
        df["competitor_revenue"] / df["competitor_std_volume"]
    )
    df["relative_price"] = df["price_std"] / df["competitor_price_std"]

    usable = (
        df["price_std"].gt(0)
        & df["competitor_price_std"].gt(0)
        & df["nd"].gt(0)
        & df["nd"].le(1)
        & df["velocity_std"].gt(0)
        & df["relative_price"].gt(0)
    )

    dropped = int((~usable).sum())
    if dropped:
        warnings.warn(
            f"Dropping {dropped:,} rows without usable competitor price, "
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

    df["pack_group"] = df["pack_size"].astype(str)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after final numeric cleanup.")

    return df, scales


def add_indices(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()

    level_map: Dict[str, np.ndarray] = {}
    for col in ["retailer", "sku", "brand", "month", "pack_group"]:
        codes, levels = pd.factorize(df[col], sort=True)
        df[f"{col}_idx"] = codes.astype("int32")
        level_map[col] = levels

    df["entity"] = (
        df["retailer"].astype(str) + "__" + df["sku"].astype(str)
    )
    entity_codes, entity_levels = pd.factorize(df["entity"], sort=True)
    df["entity_idx"] = entity_codes.astype("int32")

    sku_info = (
        df[
            [
                "sku_idx",
                "brand_idx",
                "pack_group_idx",
                "sku",
                "brand",
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


def fit_model(
    model: pm.Model,
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    target_accept: float = DEFAULT_TARGET_ACCEPT,
    random_seed: int = 42,
    sample_posterior_predictive: bool = False,
) -> xr.Dataset:
    """
    Fit the Bayesian model.

    Normal application mode deliberately skips posterior-predictive sampling.
    This is the largest unnecessary cost in the original interactive flow.
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
            idata_kwargs={"log_likelihood": False},
        )

        if sample_posterior_predictive:
            # Kept as an explicit opt-in diagnostic path.
            ppc = pm.sample_posterior_predictive(
                trace,
                var_names=["log_velocity_z_obs"],
                random_seed=random_seed,
                progressbar=False,
            )
            trace.extend(ppc)

    return trace


# ============================================================================
# Diagnostics / elasticity
# ============================================================================

def get_model_diagnostics(trace: Any) -> Dict[str, Any]:
    posterior = _posterior_group(trace)

    divergences = 0
    if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
        divergences = int(np.asarray(trace.sample_stats["diverging"]).sum())

    max_rhat = np.nan
    min_ess = np.nan

    try:
        # Avoid requiring a full ArviZ import path; calculate directly.
        chains = max(1, int(posterior.sizes.get("chain", 1)))
        draws = max(1, int(posterior.sizes.get("draw", 1)))

        rhats: List[float] = []
        esses: List[float] = []

        for var in ["price_slope_z", "nd_coef", "sigma", "sigma_entity"]:
            if var not in posterior:
                continue

            arr = np.asarray(posterior[var])
            if arr.ndim < 2:
                continue

            # Flatten parameter dimensions.
            arr = arr.reshape(chains, draws, -1)

            # R-hat and ESS across every parameter element.
            chain_means = arr.mean(axis=1)
            chain_vars = arr.var(axis=1, ddof=1)

            if chains > 1 and draws > 1:
                between = draws * chain_means.var(axis=0, ddof=1)
                within = chain_vars.mean(axis=0)
                var_hat = ((draws - 1) / draws) * within + between / draws
                rhat = np.sqrt(var_hat / np.maximum(within, 1e-12))
                rhats.extend(rhat[np.isfinite(rhat)].ravel().tolist())

            # Simple bulk ESS approximation from lag-1 autocorrelation.
            x = arr - arr.mean(axis=1, keepdims=True)
            var = np.mean(x * x, axis=1)
            ac_num = np.mean(x[:, 1:] * x[:, :-1], axis=1)
            rho = np.clip(
                ac_num / np.maximum(var, 1e-12),
                -0.99,
                0.99,
            )
            ess = (chains * draws) * (1.0 - rho.mean(axis=0)) / (
                1.0 + rho.mean(axis=0)
            )
            esses.extend(ess[np.isfinite(ess)].ravel().tolist())

        if rhats:
            max_rhat = float(np.max(rhats))
        if esses:
            min_ess = float(np.min(esses))
    except Exception:
        pass

    return {
        "divergences": divergences,
        "max_rhat": max_rhat if np.isfinite(max_rhat) else None,
        "min_ess": min_ess if np.isfinite(min_ess) else None,
    }


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
# Posterior extraction for fast scenarios
# ============================================================================

def extract_scenario_posterior(
    trace: Any,
    max_draws: int = DEFAULT_SCENARIO_DRAWS,
) -> Dict[str, np.ndarray]:
    """
    Extract only the posterior quantities needed by a scenario.

    This makes repeated slider interactions dramatically cheaper than carrying
    the full PyMC trace through every scenario calculation.
    """
    result: Dict[str, np.ndarray] = {}

    price_name = _find_posterior_variable(trace, ["price_slope_z", "price_slope"])
    if price_name is not None:
        result["price_slope_z"] = _posterior_array(
            trace, price_name, ("sku",)
        )

    nd_name = _find_posterior_variable(
        trace,
        ["nd_coef", "distribution_coef", "nd_slope"],
    )
    if nd_name is not None:
        result["nd_coef"] = _posterior_array(
            trace, nd_name, ("spline_basis",)
        )

    return _thin_posterior(result, max_draws=max_draws)


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
    out["volume_multiplier_median"] = median
    out["volume_multiplier_p90"] = p90

    out["expected_units_p10"] = out["units"].to_numpy() * p10
    out["expected_units_median"] = out["units"].to_numpy() * median
    out["expected_units_p90"] = out["units"].to_numpy() * p90

    # Price change affects revenue directly; volume effect is multiplicative.
    price_factor = 1.0 + price_change
    out["expected_revenue_p10"] = (
        out["revenue"].to_numpy() * price_factor * p10
    )
    out["expected_revenue_median"] = (
        out["revenue"].to_numpy() * price_factor * median
    )
    out["expected_revenue_p90"] = (
        out["revenue"].to_numpy() * price_factor * p90
    )

    # Convenience fields.
    out["unit_change_pct_median"] = median - 1.0
    out["revenue_change_pct_median"] = (
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
        scenario_df.groupby(groups, observed=True)["expected_revenue_median"]
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
        expected_units_p10=("expected_units_p10", "sum"),
        expected_units_median=("expected_units_median", "sum"),
        expected_units_p90=("expected_units_p90", "sum"),
        baseline_revenue=("revenue", "sum"),
        expected_revenue_p10=("expected_revenue_p10", "sum"),
        expected_revenue_median=("expected_revenue_median", "sum"),
        expected_revenue_p90=("expected_revenue_p90", "sum"),
    ).reset_index()

    base_u = agg["baseline_units"].replace(0, np.nan)
    base_r = agg["baseline_revenue"].replace(0, np.nan)

    agg["volume_change_pct"] = (
        agg["expected_units_median"] / base_u
    ) - 1.0
    agg["revenue_change_pct"] = (
        agg["expected_revenue_median"] / base_r
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
            competitor_price_std=("competitor_price_std", "mean"),
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
        changed["expected_units_p10"] = changed["units"]
        changed["expected_units_median"] = changed["units"]
        changed["expected_units_p90"] = changed["units"]
        changed["expected_revenue_p10"] = changed["revenue"]
        changed["expected_revenue_median"] = changed["revenue"]
        changed["expected_revenue_p90"] = changed["revenue"]

    unchanged = scenario_other.copy()
    unchanged["scenario_price_change"] = 0.0
    unchanged["scenario_nd_change"] = 0.0
    unchanged["scenario_nd"] = unchanged["nd"]
    for q in ["p10", "median", "p90"]:
        unchanged[f"volume_multiplier_{q}"] = 1.0
        unchanged[f"expected_units_{q}"] = unchanged["units"]
        unchanged[f"expected_revenue_{q}"] = unchanged["revenue"]

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
