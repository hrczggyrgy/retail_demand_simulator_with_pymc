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

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

# Re-export everything from engine_modules for backward compatibility
from engine_modules import *
from engine_modules.model import (
    JointModelConfig,
    extract_joint_posterior,
)
from engine_modules.model import (
    build_joint_model as _build_joint_model,
)

_DEFAULT_JOINT_MODEL_CONFIG = JointModelConfig()

def fit_joint_model(
    model: pm.Model,
    config: JointModelConfig | None = None,
) -> az.InferenceData:
    if config is None:
        config = _DEFAULT_JOINT_MODEL_CONFIG
    """Fit the joint SKU share model. Wrapper for backward compatibility."""
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



# Export JointModelConfig with legacy name (already imported as JointModelConfig)

@dataclass(frozen=True, slots=True)
class NestModelConfig:
    """Configuration for the retailer-category nest allocation model (Layer B). Legacy class."""
    draws: int = 800
    tune: int = 800
    chains: int = 4
    target_accept: float = 0.95
    random_seed: int = 42
    
    # Prior scales
    sigma_alpha_rc: float = 0.5
    sigma_cpi: float = 0.3
    sigma_assortment: float = 0.3
    sigma_nesting: float = 0.5
    sigma_rc_month: float = 0.3
    sigma_outside: float = 0.5


_DEFAULT_NEST_MODEL_CONFIG = NestModelConfig()

def build_nest_allocation_model(
    choice_data: ChoiceSetData,
    joint_posterior_cache: dict,
    config: NestModelConfig | None = None,
) -> pm.Model:
    if config is None:
        config = _DEFAULT_NEST_MODEL_CONFIG
    """Build the retailer-category nest allocation model (Layer B). Legacy wrapper.
    
    This model allocates total observed market volume across retailer-category nests
    using the inclusive value from the SKU-level model (Layer C).
    
    Args:
        choice_data: ChoiceSetData from SKU-level model
        joint_posterior_cache: Posterior cache from joint SKU share model (for IV)
        config: NestModelConfig
    
    Returns:
        PyMC model for nest allocation
    """
    n_markets = len(choice_data.market_ids)
    
    # Get unique retailer-category combinations
    rc_combos = []
    for m_idx in range(n_markets):
        retailer = choice_data.retailer_levels[choice_data.market_retailer_idx[m_idx]]
        category = choice_data.category_levels[choice_data.market_category_idx[m_idx]]
        rc_combos.append((retailer, category))
    
    unique_rc = sorted(set(rc_combos))
    n_rc = len(unique_rc)
    
    # Map market to rc index
    market_to_rc = np.array([unique_rc.index(rc) for rc in rc_combos], dtype=np.int32)
    
    # Get months
    months = sorted({choice_data.month_levels[choice_data.market_month_idx[m_idx]] for m_idx in range(n_markets)})
    n_months = len(months)
    month_to_idx = {m: i for i, m in enumerate(months)}
    market_month_idx = np.array([month_to_idx[choice_data.month_levels[choice_data.market_month_idx[m_idx]]] for m_idx in range(n_markets)], dtype=np.int32)
    
    # Get categories for nesting parameter
    rc_categories = [rc[1] for rc in unique_rc]
    rc_category_idx = np.array([np.where(choice_data.category_levels == cat)[0][0] for cat in rc_categories], dtype=np.int32)
    
    # Market totals (total units per retailer-category-month)
    market_totals = choice_data.market_total_units
    
    # Inclusive values from joint model
    if "inclusive_value" in joint_posterior_cache:
        iv_per_market = joint_posterior_cache["inclusive_value"].mean(axis=0)  # (n_markets,)
    else:
        iv_per_market = np.log(np.maximum(choice_data.available_mask.sum(axis=1), 1))
    
    # Aggregate IV to retailer-category level (mean across months)
    iv_rc = np.zeros(n_rc)
    iv_rc_counts = np.zeros(n_rc)
    for m_idx in range(n_markets):
        rc_idx = market_to_rc[m_idx]
        iv_rc[rc_idx] += iv_per_market[m_idx]
        iv_rc_counts[rc_idx] += 1
    iv_rc = np.where(iv_rc_counts > 0, iv_rc / iv_rc_counts, 0.0)
    
    coords = {
        "rc": np.arange(n_rc),
        "month": np.arange(n_months),
        "category": choice_data.category_levels,
    }
    
    with pm.Model(coords=coords) as model:
        # Data containers
        market_totals_data = pm.Data("market_totals", market_totals, dims="market")
        iv_data = pm.Data("iv_rc", iv_rc, dims="rc")
        market_to_rc_data = pm.Data("market_to_rc", market_to_rc, dims="market")
        market_month_idx_data = pm.Data("market_month_idx", market_month_idx, dims="market")
        
        # Retailer-category baseline attractiveness
        alpha_rc = pm.Normal("alpha_rc", 0.0, config.sigma_alpha_rc, dims="rc")
        
        # Category-level CPI effect
        theta_c = pm.Normal("theta_c", 0.0, config.sigma_cpi, dims="category")
        cpi_rc = pm.Data("cpi_rc", np.zeros(n_rc), dims="rc")
        
        # Category-level assortment effect
        omega_c = pm.Normal("omega_c", 0.0, config.sigma_assortment, dims="category")
        assortment_rc = pm.Data("assortment_rc", np.zeros(n_rc), dims="rc")
        
        # Nesting parameter lambda_c (0 < lambda <= 1)
        lambda_raw = pm.Normal("lambda_raw", 0.0, config.sigma_nesting, dims="category")
        lambda_c = pm.Deterministic("lambda_c", pm.math.sigmoid(lambda_raw), dims="category")
        
        # Outside option
        sigma_outside = pm.HalfNormal("sigma_outside", config.sigma_outside)
        outside = pm.Normal("outside", 0.0, sigma_outside)
        
        # Utility for each retailer-category nest (average over months)
        # U_rc = alpha_rc + theta_c * CPI_rc + omega_c * Assortment_rc + lambda_c * IV_rc
        utility_rc = (
            alpha_rc
            + theta_c[rc_category_idx] * cpi_rc
            + omega_c[rc_category_idx] * assortment_rc
            + lambda_c[rc_category_idx] * iv_data
        )
        
        # Nest shares (multinomial logit with outside option)
        # w_rc = exp(U_rc) / (sum(exp(U_rc')) + exp(U_outside))
        exp_utility = pm.math.exp(utility_rc)
        exp_outside = pm.math.exp(outside)
        denom = exp_utility.sum() + exp_outside
        nest_shares = exp_utility / denom
        
        # Nest shares with outside option
        w_rc = pm.Deterministic("w_rc", nest_shares, dims="rc")
        w_outside = pm.Deterministic("w_outside", exp_outside / denom)
        
        # Since we don't observe nest-level breakdown, we model the total market volume
        # with a simple Normal likelihood on the total (placeholder)
        # The nest shares are available for scenario analysis
        total_units_pred = pm.Deterministic("total_units_pred", market_totals_data, dims="market")
        sigma_total = pm.HalfNormal("sigma_total", 0.5)
        pm.Normal(
            "market_totals_obs",
            mu=total_units_pred,
            sigma=sigma_total,
            observed=market_totals_data,
            dims="market",
        )
    
    return model


def build_joint_sku_share_model(
    choice_data: ChoiceSetData,
    config: JointModelConfig | None = None,
) -> pm.Model:
    if config is None:
        config = JointModelConfig()
    """Build joint SKU share model with legacy name. Wrapper for backward compatibility."""
    return _build_joint_model(choice_data, config)


from engine_modules.choice_sets import (
    ChoiceSetData,
)
from engine_modules.choice_sets import (
    build_choice_set_data as _build_choice_set_data,
)
from engine_modules.reporting import (
    decompose_sales_growth,
    prepare_market,
)
from engine_modules.scenarios import (
    aggregate_scenario_result,
    compute_source_destination_flows,
    run_joint_scenario_draws,
    create_reallocation_sankey,
)


def build_source_destination_summary(
    result: ScenarioResult,
    choice_data: ChoiceSetData,
    action: ScenarioAction,
) -> pd.DataFrame:
    """Build source-destination reallocation summary aggregated by segment. Legacy wrapper."""
    flows = compute_source_destination_flows(result, choice_data, action.sku)
    # Aggregate by segment
    agg = flows.groupby("source_relationship").agg({
        "baseline_standard_volume_p50": "sum",
        "scenario_standard_volume_p50": "sum",
        "delta_standard_volume_p50": "sum",
        "delta_standard_volume_p05": "sum",
        "delta_standard_volume_p95": "sum",
    }).reset_index()
    return agg


def build_choice_set_data(
    df: pd.DataFrame,
    meta: dict[str, Any],
    price_col: str = "price_std",
    volume_col: str = "standard_volume",
) -> ChoiceSetData:
    """Build choice set data with legacy signature. Wrapper for backward compatibility."""
    return _build_choice_set_data(df, price_col=price_col)



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


_DEFAULT_BACKTEST_CONFIG = BacktestConfig()
_DEFAULT_MODEL_CONFIG = DEFAULT_CONFIG

def run_rolling_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = _DEFAULT_BACKTEST_CONFIG
    if model_config is None:
        model_config = _DEFAULT_MODEL_CONFIG
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


# =============================================================================
# Legacy compatibility functions (kept for existing tests)
# =============================================================================

def _resolve_target_nd(nd_base: np.ndarray, nd_change: float, nd_mode: str) -> np.ndarray:
    """Resolve target ND values from base and change specification."""
    if nd_mode == "pp":
        # Percentage points: nd_change is absolute change in ND (0-1 scale)
        return np.clip(nd_base + nd_change, 0.0, 1.0)
    elif nd_mode == "relative":
        # Relative: nd_change is multiplicative factor
        return np.clip(nd_base * (1.0 + nd_change), 0.0, 1.0)
    else:  # absolute
        # Absolute: nd_change is the new ND value
        return np.full_like(nd_base, np.clip(nd_change, 0.0, 1.0))


def _apply_target_scope(df: pd.DataFrame, target_level: str, target_value: str) -> pd.Series:
    """Build boolean mask for target scope."""
    target_level = target_level.lower()
    if target_level == "market":
        return pd.Series(True, index=df.index)
    elif target_level == "retailer":
        return df["retailer"] == target_value
    elif target_level == "category":
        return df["category"] == target_value
    elif target_level == "brand":
        return df["brand"] == target_value
    elif target_level == "sku":
        return df["sku"] == target_value
    else:
        raise ValueError(f"Unknown target_level: {target_level}")


def _price_delta_log_volume(
    price_slope_z: np.ndarray,
    sku_idx: np.ndarray,
    price_change: float,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Compute log-volume change from price change (vectorized over draws)."""
    n_draws = price_slope_z.shape[0]
    n_obs = len(sku_idx)
    effect = np.zeros((n_draws, n_obs))
    
    for d in range(n_draws):
        slopes = price_slope_z[d, sku_idx]
        effect[d, target_mask] = slopes[target_mask] * price_change
    
    return effect


def _nd_delta_log_volume(
    posterior_cache: dict,
    spline,
    scales: dict,
    sku_idx: np.ndarray,
    nd_base: np.ndarray,
    nd_new: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Compute log-volume change from ND change using spline (vectorized over draws)."""
    log_nd_base = np.log(np.clip(nd_base, 1e-4, None))
    log_nd_new = np.log(np.clip(nd_new, 1e-4, None))
    
    # Standardize using stored scales
    mean_nd = scales["log_nd"]["mean"]
    sd_nd = scales["log_nd"]["sd"]
    z_base = (log_nd_base - mean_nd) / sd_nd
    z_new = (log_nd_new - mean_nd) / sd_nd
    
    # Transform through spline basis
    basis_base = spline.transform(z_base.reshape(-1, 1))
    basis_new = spline.transform(z_new.reshape(-1, 1))
    
    n_draws = posterior_cache["nd_coef"].shape[0]
    n_obs = len(sku_idx)
    effect = np.zeros((n_draws, n_obs))
    
    if posterior_cache.get("nd_coef_type") == "shared":
        # Shared coefficients across SKUs
        for d in range(n_draws):
            coef = posterior_cache["nd_coef"][d]
            pred_base = basis_base @ coef
            pred_new = basis_new @ coef
            effect[d, target_mask] = (pred_new - pred_base)[target_mask]
    else:
        # Per-SKU coefficients (not expected in current tests)
        for d in range(n_draws):
            for s in range(n_obs):
                if target_mask[s]:
                    coef = posterior_cache["nd_coef"][d, sku_idx[s]]
                    pred_base = basis_base[s] @ coef
                    pred_new = basis_new[s] @ coef
                    effect[d, s] = pred_new - pred_base
    
    return effect


def _combined_delta_log_volume(
    price_effect: np.ndarray,
    nd_effect: np.ndarray,
) -> np.ndarray:
    """Combine price and ND effects (additive in log space)."""
    return price_effect + nd_effect


def _summarise_scenario_draws(
    units: np.ndarray,
    revenue: np.ndarray,
    price_change: float,
    total_effect: np.ndarray,
) -> pd.DataFrame:
    """Summarize posterior draws into median/p10/p90 predictions."""
    n_draws, n_obs = total_effect.shape
    
    # Convert log-volume effects to volume
    volume_pred = units * np.exp(total_effect)
    
    # Revenue uses new price * new volume
    # Price change is applied as multiplicative factor
    price_factor = 1.0 + price_change
    revenue_pred = volume_pred * revenue * price_factor / units
    
    # Percentiles across draws
    result = pd.DataFrame({
        "units_p50": np.median(volume_pred, axis=0),
        "units_p10": np.percentile(volume_pred, 10, axis=0),
        "units_p90": np.percentile(volume_pred, 90, axis=0),
        "revenue_p50": np.median(revenue_pred, axis=0),
        "revenue_p10": np.percentile(revenue_pred, 10, axis=0),
        "revenue_p90": np.percentile(revenue_pred, 90, axis=0),
    })
    
    return result


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
    """Core scenario simulation returning DataFrame with predicted units/revenue.
    
    Legacy function kept for backward compatibility with tests.
    """
    nd_base = df["nd"].to_numpy()
    sku_idx = df["sku_idx"].to_numpy(dtype="int32")
    target_mask = _apply_target_scope(df, target_level, target_value).to_numpy()

    if nd_change != 0.0:
        resolved_nd = _resolve_target_nd(nd_base, nd_change, nd_mode)
    else:
        resolved_nd = nd_base

    price_effect = np.zeros((posterior_cache["price_slope_z"].shape[0], len(df)))
    if price_change != 0.0:
        price_effect = _price_delta_log_volume(
            posterior_cache["price_slope_z"],
            sku_idx,
            price_change,
            target_mask,
        )

    nd_effect = np.zeros((posterior_cache["price_slope_z"].shape[0], len(df)))
    if nd_change != 0.0:
        nd_effect = _nd_delta_log_volume(
            posterior_cache,
            spline,
            scales,
            sku_idx,
            nd_base,
            resolved_nd,
            target_mask,
        )

    total_effect = _combined_delta_log_volume(price_effect, nd_effect)

    result = _summarise_scenario_draws(
        df["units"].to_numpy(),
        df["revenue"].to_numpy(),
        price_change,
        total_effect,
    )

    out = df.copy()
    result.index = out.index
    for col in result.columns:
        out[col] = result[col]
    
    # Add scenario_nd column
    out["scenario_nd"] = np.where(target_mask, resolved_nd, nd_base)

    return out


def calculate_market_shares(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    share_level: str = "brand",
) -> pd.DataFrame:
    """
    Calculate baseline and median-scenario market shares.
    
    Legacy function for backward compatibility.
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

    merged = pd.merge(base, scen, on=group_cols, how="outer").fillna(0.0)

    total_base = merged.groupby("month")["baseline_value"].transform("sum")
    total_scen = merged.groupby("month")["scenario_value"].transform("sum")

    merged["baseline_share"] = merged["baseline_value"] / total_base
    merged["scenario_share"] = merged["scenario_value"] / total_scen
    merged["share_delta_pp"] = (merged["scenario_share"] - merged["baseline_share"]) * 100

    return merged


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
    Legacy function for backward compatibility.
    """
    return {
        "baseline": _simulate_scenario_core(
            df, posterior_cache, spline, scales,
            price_change=0.0, nd_change=0.0, nd_mode=nd_mode,
            target_level=target_level, target_value=target_value
        ),
        "price_only": _simulate_scenario_core(
            df, posterior_cache, spline, scales,
            price_change=price_change, nd_change=0.0, nd_mode=nd_mode,
            target_level=target_level, target_value=target_value
        ),
        "distribution_only": _simulate_scenario_core(
            df, posterior_cache, spline, scales,
            price_change=0.0, nd_change=nd_change, nd_mode=nd_mode,
            target_level=target_level, target_value=target_value
        ),
        "combined": _simulate_scenario_core(
            df, posterior_cache, spline, scales,
            price_change=price_change, nd_change=nd_change, nd_mode=nd_mode,
            target_level=target_level, target_value=target_value
        ),
    }


# Legacy constants for backward compatibility
MARKET_LEVELS = {
    "market": ["month"],
    "retailer": ["month", "retailer"],
    "retailer_category": ["month", "retailer", "category"],
    "brand": ["month", "retailer", "category", "brand"],
    "brand_pack": ["month", "retailer", "category", "brand", "pack_group"],
    "sku": ["month", "retailer", "category", "brand", "pack_group", "sku"],
}


def aggregate_market_impact(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """
    Aggregate baseline and scenario to specified market level and compute deltas.
    
    Levels: market, retailer, retailer_category, brand, brand_pack, sku
    Legacy function for backward compatibility.
    """
    if level not in MARKET_LEVELS:
        raise ValueError(f"Unknown level: {level}. Valid: {list(MARKET_LEVELS.keys())}")

    group_cols = MARKET_LEVELS[level]

    baseline = (
        baseline_df
        .groupby(group_cols, as_index=False, observed=True)
        .agg(
            baseline_units=("units", "sum"),
            baseline_revenue=("revenue", "sum"),
        )
    )

    scenario = (
        scenario_df
        .groupby(group_cols, as_index=False, observed=True)
        .agg(
            scenario_units=("units_p50", "sum"),
            scenario_revenue=("revenue_p50", "sum"),
        )
    )

    result = baseline.merge(scenario, on=group_cols, how="outer")
    # Avoid fillna on categorical columns
    for col in result.columns:
        if result[col].dtype.name == "category":
            result[col] = result[col].astype(object)
    result = result.fillna(0)

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

    return result


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
    Legacy function for backward compatibility.
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
    """Add segment relationship classification to a DataFrame. Legacy function."""
    df = df.copy()
    df["segment_relationship"] = df.apply(
        lambda row: classify_segment_relationship(
            row, selected_sku, selected_brand, selected_category,
            selected_pack_group, selected_retailer
        ),
        axis=1,
    )
    return df


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
    Legacy function for backward compatibility.
    """
    baseline_classified = add_segment_classification(
        baseline_df, selected_sku, selected_brand, selected_category,
        selected_pack_group, selected_retailer
    )
    scenario_classified = add_segment_classification(
        scenario_df, selected_sku, selected_brand, selected_category,
        selected_pack_group, selected_retailer
    )

    baseline_agg = baseline_classified.groupby("segment_relationship", observed=True).agg(
        baseline_units=("units", "sum"),
        baseline_revenue=("revenue", "sum"),
    )

    scenario_agg = scenario_classified.groupby("segment_relationship", observed=True).agg(
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
    Legacy function for backward compatibility.
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
    })


def _find_posterior_variable(trace: Any, candidates: list[str]) -> str | None:
    """Find first matching posterior variable name. Legacy function."""
    posterior = _posterior_group(trace)
    for name in candidates:
        if name in posterior:
            return name
    return None


def extract_elasticities(
    trace: Any,
    meta: dict[str, Any],
    scales: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Convert standardized price slopes back to ordinary elasticity.
    Legacy function for backward compatibility.
    """
    beta_name = _find_posterior_variable(trace, ["price_slope_z", "price_slope"])
    if beta_name is None:
        raise KeyError("Could not find posterior price slope variable.")

    beta_z = _posterior_array(trace, beta_name, ("sku",))
    beta_raw = (
        beta_z
        * scales["log_velocity_std"]["sd"]
        / scales["log_relative_price_std"]["sd"]
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


def create_market_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create monthly market summary. Legacy function."""
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
    """Create monthly category summary. Legacy function."""
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
    """Create monthly retailer summary. Legacy function."""
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
    """Create monthly brand summary. Legacy function."""
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
    """Create SKU summary. Legacy function."""
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


def compute_price_architecture(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute price architecture metrics per SKU.
    
    Returns per-SKU price metrics for price ladder analysis.
    Legacy function.
    """
    sku_metrics = df.groupby(["retailer", "category", "brand", "sku", "pack_size"]).agg(
        avg_price_std=("price_std", "mean"),
        avg_units=("units", "mean"),
        avg_revenue=("revenue", "mean"),
        avg_velocity_std=("velocity_std", "mean"),
        avg_nd=("nd", "mean"),
        months=("month", "nunique"),
    ).reset_index()

    # Category benchmarks (price per standard unit)
    cat_price = sku_metrics.groupby(["retailer", "category"])["avg_price_std"].transform("median")
    sku_metrics["rel_price_index"] = sku_metrics["avg_price_std"] / cat_price

    # Brand benchmarks
    brand_price = sku_metrics.groupby(["retailer", "category", "brand"])["avg_price_std"].transform("median")
    sku_metrics["brand_price_index"] = sku_metrics["avg_price_std"] / brand_price

    # Pack segment
    sku_metrics["pack_segment"] = pd.cut(
        sku_metrics["pack_size"],
        bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["small", "medium", "large", "xl"]
    )

    # Pack segment benchmarks
    seg_price = sku_metrics.groupby(["retailer", "category", "pack_segment"], observed=True)["avg_price_std"].transform("median")
    sku_metrics["pack_value_index"] = sku_metrics["avg_price_std"] / seg_price

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

    return sku_metrics.sort_values(["retailer", "category", "brand", "avg_price_std"]).reset_index(drop=True)


def compute_distribution_opportunity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute distribution opportunity metrics per SKU.
    
    Identifies white-space (high velocity_std, low ND) and 
    rationalization candidates (low velocity_std, high ND).
    Legacy function.
    """
    sku_metrics = df.groupby(["retailer", "category", "brand", "sku", "pack_size"]).agg(
        avg_nd=("nd", "mean"),
        avg_velocity_std=("velocity_std", "mean"),
        total_units=("units", "sum"),
        total_revenue=("revenue", "sum"),
        max_stores=("retailer_stores", "max"),
        avg_stores=("sku_stores", "mean"),
    ).reset_index()

    # Category benchmarks for velocity
    cat_vel = sku_metrics.groupby(["retailer", "category"])["avg_velocity_std"].transform("median")
    sku_metrics["velocity_index"] = sku_metrics["avg_velocity_std"] / cat_vel

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
        (sku_metrics["avg_velocity_std"] > cat_vel) & (sku_metrics["avg_nd"] < 0.5),
        (sku_metrics["avg_velocity_std"] > cat_vel) & (sku_metrics["avg_nd"] >= 0.5),
        (sku_metrics["avg_velocity_std"] <= cat_vel) & (sku_metrics["avg_nd"] < 0.5),
        (sku_metrics["avg_velocity_std"] <= cat_vel) & (sku_metrics["avg_nd"] >= 0.5),
    ]
    choices = ["Expand", "Protect", "Test/Review", "Rationalize"]
    sku_metrics["distribution_action"] = np.select(conditions, choices, default="Review")

    return sku_metrics.sort_values("distribution_opportunity_score", ascending=False).reset_index(drop=True)


def compute_market_share_analytics(
    df: pd.DataFrame,
    group_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive market share analytics using standard volume.
    Legacy function.
    """
    if group_cols is None:
        group_cols = ["retailer", "category", "month", "brand", "sku"]

    # Monthly share by group
    monthly = df.groupby(group_cols).agg(
        standard_volume=("standard_volume", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()

    # Total market per retailer/category/month
    market_cols = ["retailer", "category", "month"]
    market_total = monthly.groupby(market_cols).agg(
        market_standard_volume=("standard_volume", "sum"),
        market_revenue=("revenue", "sum"),
    ).reset_index()

    monthly = monthly.merge(market_total, on=market_cols, how="left")
    monthly["standard_volume_share"] = monthly["standard_volume"] / monthly["market_standard_volume"]
    monthly["revenue_share"] = monthly["revenue"] / monthly["market_revenue"]

    # Share momentum (3, 6, 12 month changes)
    monthly = monthly.sort_values(group_cols + ["month"])
    for window in [3, 6, 12]:
        monthly[f"standard_volume_share_change_{window}m"] = monthly.groupby(group_cols)["standard_volume_share"].transform(
            lambda x: x - x.shift(window)
        )
        monthly[f"revenue_share_change_{window}m"] = monthly.groupby(group_cols)["revenue_share"].transform(
            lambda x: x - x.shift(window)
        )

    return monthly


def compute_contribution_to_growth(
    df: pd.DataFrame,
    start_month: str,
    end_month: str,
    level: str = "brand",
) -> pd.DataFrame:
    """
    Compute contribution to category growth at specified level using standard volume.
    
    Level can be: brand, sku, retailer
    Legacy function.
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
        start_standard_volume=("standard_volume", "sum"),
        start_revenue=("revenue", "sum"),
    ).reset_index()

    end_data = df_period[df_period["month"] == end].groupby(group_cols).agg(
        end_standard_volume=("standard_volume", "sum"),
        end_revenue=("revenue", "sum"),
    ).reset_index()

    growth = start_data.merge(end_data, on=group_cols, how="outer").fillna(0)
    growth["standard_volume_change"] = growth["end_standard_volume"] - growth["start_standard_volume"]
    growth["revenue_change"] = growth["end_revenue"] - growth["start_revenue"]

    # Total category change
    total_standard_volume_change = growth["standard_volume_change"].sum()
    total_rev_change = growth["revenue_change"].sum()

    growth["standard_volume_contribution_pct"] = np.where(
        total_standard_volume_change != 0,
        growth["standard_volume_change"] / total_standard_volume_change * 100,
        0
    )
    growth["revenue_contribution_pct"] = np.where(
        total_rev_change != 0,
        growth["revenue_change"] / total_rev_change * 100,
        0
    )

    # Share of growth
    growth["standard_volume_share_of_growth"] = np.where(
        total_standard_volume_change > 0,
        growth["standard_volume_change"] / total_standard_volume_change * 100,
        0
    )

    return growth.sort_values("standard_volume_contribution_pct", ascending=False).reset_index(drop=True)


def _growth_decomposition(
    df: pd.DataFrame,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
) -> pd.DataFrame:
    """Internal growth decomposition. Legacy function."""
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


def create_growth_decomposition(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create growth decomposition. Legacy function."""
    months = sorted(df["month"].unique())
    if len(months) < 2:
        return pd.DataFrame()

    start_month, end_month = months[0], months[-1]
    return _growth_decomposition(df, start_month, end_month)


def classify_distribution_velocity(
    df: pd.DataFrame,
    benchmark_level: str = "retailer_category",
) -> pd.DataFrame:
    """Classify distribution velocity relative to benchmark. Legacy function."""
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
    """Get expansion candidates. Legacy function."""
    classified = classify_distribution_velocity(df)
    return classified[
        (classified["nd"] < classified.groupby(
            ["retailer", "category"],
            observed=True,
        )["nd"].transform("median"))
        & (classified["relative_velocity"] > 1.0)
    ].copy()


def add_indices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add index columns for modeling. Legacy function."""
    df = df.copy()

    level_map: dict[str, np.ndarray] = {}
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
