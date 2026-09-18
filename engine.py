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

# Legacy model output dataclasses that might be referenced
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Re-export everything from engine_modules for backward compatibility
from engine_modules import *
from engine_modules.model import (
    extract_joint_posterior,
)
from engine_modules.reporting import (
    decompose_sales_growth,
    prepare_market,
)
from engine_modules.scenarios import (
    aggregate_scenario_result,
    run_joint_scenario_draws,
)

# Additional legacy functions that app.py might need


@dataclass(frozen=True, slots=True)
class ModelOutput:
    trace: Any
    df: Any
    meta: dict[str, Any]
    spline: Any
    scales: dict[str, dict[str, float]]
    settings: dict[str, Any]


# Fast / safe utilities that were in engine.py
def _posterior_group(trace: Any):
    """Return the posterior group for InferenceData/DataTree-like objects."""
    if hasattr(trace, "posterior"):
        return trace.posterior
    if hasattr(trace, "groups"):
        try:
            return trace["posterior"]
        except (KeyError, TypeError):
            pass
    try:
        return trace["posterior"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Trace does not contain a posterior group") from exc


def _posterior_array(
    trace: Any,
    variable: str,
    dimensions: tuple[str, ...] | None = None,
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


# Rolling Holdout Backtest
@dataclass(frozen=True, slots=True)
class BacktestConfig:
    min_train_months: int = 18
    horizons: tuple[int, ...] = (1, 3, 6)


def run_rolling_backtest(
    df: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
    model_config: ModelConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Run rolling holdout backtest. Not implemented yet."""
    raise NotImplementedError("Rolling backtest not yet implemented")


# Corrected scenario math (legacy compatibility)
def price_delta_log_volume(
    price_slope_draws: np.ndarray,
    sku_idx: np.ndarray,
    price_change: float,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Compute price effect on log volume."""
    return price_slope_draws[:, sku_idx] * np.log1p(price_change) * target_mask


def nd_delta_log_volume(
    gamma1_draws: np.ndarray,
    gamma2_draws: np.ndarray,
    sku_idx: np.ndarray,
    nd_base: np.ndarray,
    nd_change: float,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Compute ND effect on log volume."""
    nd_new = nd_base * (1 + nd_change)
    return (gamma1_draws[:, sku_idx] * (nd_new - nd_base) +
            gamma2_draws[:, sku_idx] * (nd_new**2 - nd_base**2)) * target_mask


# Fast scenario engine (legacy compatibility)
def _thin_posterior(
    posterior: dict[str, np.ndarray],
    max_draws: int,
) -> dict[str, np.ndarray]:
    if not posterior:
        return posterior
    n_draws = next(iter(posterior.values())).shape[0]
    if n_draws <= max_draws:
        return posterior
    idx = np.random.default_rng(42).choice(n_draws, size=max_draws, replace=False)
    return {k: v[idx] for k, v in posterior.items()}


def extract_scenario_posterior(
    idata: Any,
    max_draws: int = 400,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Extract posterior draws for fast scenarios (v2 compatible)."""
    return extract_joint_posterior(idata, max_draws, random_seed)


# Optional helpers for the improved UI
def prepare_market(
    raw: pd.DataFrame,
    n_knots: int = 4,
    spline_degree: int = 3,
) -> dict[str, Any]:
    """High-level market preparation for the improved UI."""
    from engine_modules.validation import prepare_data as _prepare_data
    prepared = _prepare_data(raw, n_knots, spline_degree)
    return {
        "df": prepared.df,
        "scales": prepared.scales,
        "meta": prepared.meta,
        "spline": prepared.spline,
        "nd_basis": prepared.nd_basis,
    }


# New Analytics Helpers for Retail Cockpit
def decompose_sales_growth(
    df: pd.DataFrame,
    start_month: str,
    end_month: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Decompose sales growth into volume/price/mix components."""
    from engine_modules.reporting import decompose_sales_growth as _decompose
    return _decompose(df, start_month, end_month, group_cols)


# Scenario Action Table Support (New simplified workflow)
@dataclass(frozen=True, slots=True)
class ActionTableRow:
    """A single scenario action for a specific retailer × SKU relationship."""
    retailer: str
    category: str
    brand: str
    sku: str
    month: str
    baseline_price: float
    baseline_nd: float
    baseline_velocity: float
    price_change_pct: float = 0.0
    nd_change_pp: float = 0.0
    include: bool = False
    # Computed fields
    scenario_price: float = 0.0
    scenario_nd: float = 0.0
    scenario_velocity: float = 0.0
    delta_volume: float = 0.0
    delta_revenue: float = 0.0


def create_scenario_action_table(
    df: pd.DataFrame,
    posterior_cache: dict[str, np.ndarray] | None = None,
    choice_data: ChoiceSetData | None = None,
) -> pd.DataFrame:
    """
    Create the editable scenario action table.
    
    If called with only df (legacy signature), validates enriched columns and returns
    the action table from the DataFrame directly.
    If called with posterior_cache and choice_data, builds from the choice set data.
    """
    # Legacy signature: only df provided
    if posterior_cache is None and choice_data is None:
        required_cols = {"month", "retailer", "category", "brand", "sku", "pack_size",
                         "price_std", "nd", "standard_volume", "velocity_std", "revenue"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                "Scenario action table requires enriched market data. "
                f"Missing columns: {sorted(missing)}"
            )
        
        action_cols = ["month", "retailer", "category", "brand", "sku", "pack_size"]
        baseline = df[action_cols + ["standard_volume", "price_std", "nd", "velocity_std", 
                                      "relative_price", "revenue"]].copy()
        
        baseline["include"] = False
        baseline["price_change_pct"] = 0.0
        baseline["nd_change_pp"] = 0.0
        baseline["nd_mode"] = "pp"
        
        if "relative_price" not in baseline.columns:
            cat_mean_price = df.groupby(["month", "category"])["price_std"].transform("mean")
            baseline["relative_price"] = baseline["price_std"] / cat_mean_price
        
        cols_order = [
            "include", "month", "retailer", "category", "brand", "sku", "pack_size",
            "standard_volume", "price_std", "nd", "velocity_std", "relative_price",
            "price_change_pct", "nd_change_pp", "nd_mode",
            "baseline_revenue", "baseline_volume",
        ]
        
        baseline["baseline_revenue"] = baseline["revenue"]
        baseline["baseline_volume"] = baseline["standard_volume"]
        
        return baseline[cols_order]
    
    # New signature: with posterior_cache and choice_data
    if posterior_cache is None or choice_data is None:
        raise ValueError("Both posterior_cache and choice_data must be provided together")
    
    rows = []
    for m_id in choice_data.market_ids:
        for s_idx, sku in enumerate(choice_data.sku_ids):
            rows.append({
                "retailer": choice_data.retailer_levels[choice_data.market_retailer_idx[choice_data.market_ids.index(m_id)]],
                "category": choice_data.category_levels[choice_data.market_category_idx[choice_data.market_ids.index(m_id)]],
                "brand": choice_data.brand_levels[choice_data.sku_brand_idx[s_idx]],
                "sku": sku,
                "month": choice_data.month_levels[choice_data.market_month_idx[choice_data.market_ids.index(m_id)]],
                "baseline_price": choice_data.price_std[choice_data.market_ids.index(m_id), s_idx],
                "baseline_nd": choice_data.nd[choice_data.market_ids.index(m_id), s_idx],
                "baseline_velocity": choice_data.observed_units[choice_data.market_ids.index(m_id), s_idx] / max(choice_data.nd[choice_data.market_ids.index(m_id), s_idx], 1e-4),
                "price_change_pct": 0.0,
                "nd_change_pp": 0.0,
                "include": False,
            })
    return pd.DataFrame(rows)


def run_scenario_suite_actions(
    action_df: pd.DataFrame,
    choice_data: ChoiceSetData,
    posterior_cache: dict[str, np.ndarray],
    n_draws: int = 400,
) -> dict[str, pd.DataFrame]:
    """Run scenario suite from action table (legacy compatibility)."""
    # Convert to ScenarioAction objects
    actions = []
    for _, row in action_df[action_df["include"]].iterrows():
        actions.append(ScenarioAction(
            retailer=row["retailer"],
            category=row["category"],
            brand=row["brand"],
            sku=row["sku"],
            month=row["month"],
            new_price_std=row["baseline_price"] * (1 + row["price_change_pct"]) if row["price_change_pct"] != 0 else None,
            new_nd=row["baseline_nd"] + row["nd_change_pp"] if row["nd_change_pp"] != 0 else None,
            nd_mode="pp",
        ))
    
    # Run four counterfactuals
    baseline = run_joint_scenario_draws(choice_data, posterior_cache, [], n_draws=n_draws)
    price_only = run_joint_scenario_draws(choice_data, posterior_cache, 
        [a for a in actions if a.new_price_std is not None], n_draws=n_draws)
    dist_only = run_joint_scenario_draws(choice_data, posterior_cache,
        [a for a in actions if a.new_nd is not None], n_draws=n_draws)
    combined = run_joint_scenario_draws(choice_data, posterior_cache, actions, n_draws=n_draws)
    
    # Aggregate
    return {
        "baseline": aggregate_scenario_result(baseline, choice_data, "sku"),
        "price_only": aggregate_scenario_result(price_only, choice_data, "sku"),
        "distribution_only": aggregate_scenario_result(dist_only, choice_data, "sku"),
        "combined": aggregate_scenario_result(combined, choice_data, "sku"),
    }


def fit_spline(
    df: pd.DataFrame,
    n_knots: int = DEFAULT_N_SPLINE_KNOTS,
    degree: int = DEFAULT_SPLINE_DEGREE,
) -> tuple[Any, np.ndarray]:
    """Fit spline to log_nd_z (legacy compatibility with sklearn SplineTransformer)."""
    from sklearn.preprocessing import SplineTransformer as SkSplineTransformer
    
    x = df["log_nd_z"].to_numpy(dtype=float).reshape(-1, 1)
    spline = SkSplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=False,
    )
    basis = spline.fit_transform(x).astype(np.float64, copy=False)
    return spline, basis


def validate_choice_set(choice_data: ChoiceSetData) -> dict[str, Any]:
    """
    Validate choice-set data integrity.
    
    Returns a dict with validation results.
    """
    results = {
        "n_markets": len(choice_data.market_ids),
        "n_skus": len(choice_data.sku_ids),
        "total_observed_units": int(choice_data.observed_units.sum()),
        "market_totals_match": True,
        "unavailable_have_zero_units": True,
        "shares_sum_to_one": True,
        "relative_price_range": (float(choice_data.relative_price.min()), 
                                  float(choice_data.relative_price.max())),
        "nd_range": (float(choice_data.nd.min()), float(choice_data.nd.max())),
        "availability_rate": float(choice_data.available_mask.mean()),
    }
    
    # Check market totals
    market_sums = choice_data.observed_units.sum(axis=1)
    if not np.allclose(market_sums, choice_data.market_total_units, rtol=1e-10):
        results["market_totals_match"] = False
    
    # Check unavailable SKUs have zero units
    unavailable_units = choice_data.observed_units[~choice_data.available_mask].sum()
    if unavailable_units > 0:
        results["unavailable_have_zero_units"] = False
    
    # Check shares sum to 1 per market (for available SKUs)
    for m in range(len(choice_data.market_ids)):
        avail = choice_data.available_mask[m]
        if avail.any():
            total = choice_data.observed_units[m, avail].sum()
            if total > 0:
                shares = choice_data.observed_units[m, avail] / total
                if not np.isclose(shares.sum(), 1.0, atol=1e-10):
                    results["shares_sum_to_one"] = False
    
    return results
