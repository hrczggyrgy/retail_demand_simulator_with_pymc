"""
PyMC model building for retail demand/share models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm

from contracts import DEFAULT_CONFIG, ModelConfig
from engine_modules.choice_sets import (
    ChoiceSetData,
)


def build_pymc_model(
    df: pd.DataFrame,
    meta: dict[str, Any],
    nd_basis: np.ndarray,
    config: ModelConfig = DEFAULT_CONFIG,
) -> pm.Model:
    """
    Build the hierarchical retail demand model (v1).

    Model:
    log(V_std) = alpha + entity_effect + month_effect + price_slope * log_rel_price_z + nd_effect

    With:
    - entity_effect: retailer×SKU non-centered
    - month_effect: non-centered
    - price_slope: SKU-level with brand×pack pooling
    - nd_effect: B-spline
    """
    n_obs = len(df)
    n_entities = meta["n_entities"]
    n_months = meta["n_months"]
    n_skus = meta["n_skus"]
    n_brand_packs = meta["n_brand_packs"]
    n_spline_basis = nd_basis.shape[1]

    # Index arrays
    entity_idx = df["entity_idx"].values
    month_idx = df["month_idx"].values
    sku_idx = df["sku_idx"].values
    bp_idx = df["brand_pack_idx"].values
    log_rel_price_z = df["log_relative_price_z"].values

    coords = {
        "obs_id": np.arange(n_obs),
        "entity": np.arange(n_entities),
        "month": np.arange(n_months),
        "sku": np.arange(n_skus),
        "brand_pack": np.arange(n_brand_packs),
        "spline_basis": np.arange(n_spline_basis),
    }

    with pm.Model(coords=coords) as model:
        # Data containers
        pm.Data("entity_idx", entity_idx, dims="obs_id")
        pm.Data("month_idx", month_idx, dims="obs_id")
        pm.Data("sku_idx", sku_idx, dims="obs_id")
        pm.Data("brand_pack_idx", bp_idx, dims="obs_id")
        pm.Data("log_rel_price_z", log_rel_price_z, dims="obs_id")
        pm.Data("nd_basis", nd_basis, dims=("obs_id", "spline_basis"))

        # Global intercept
        alpha = pm.Normal("alpha", 0.0, 0.5)

        # Entity effect (retailer×SKU) - non-centered
        sigma_entity = pm.HalfNormal("sigma_entity", 0.5)
        entity_raw = pm.Normal("entity_raw", 0.0, 1.0, dims="entity")
        entity_effect = pm.Deterministic("entity_effect", sigma_entity * entity_raw, dims="entity")

        # Month effect - non-centered
        sigma_month = pm.HalfNormal("sigma_month", 0.35)
        month_raw = pm.Normal("month_raw", 0.0, 1.0, dims="month")
        month_effect = pm.Deterministic("month_effect", sigma_month * month_raw, dims="month")

        # Price slope with brand×pack pooling
        price_bp_mu = pm.Normal("price_bp_mu", 0.0, 0.5, dims="brand_pack")
        price_sku_sigma = pm.HalfNormal("price_sku_sigma", 0.35)
        price_sku_raw = pm.Normal("price_sku_raw", 0.0, 1.0, dims="sku")
        price_slope = pm.Deterministic(
            "price_slope",
            price_bp_mu[bp_idx] + price_sku_sigma * price_sku_raw,
            dims="sku",
        )

        # ND effect - B-spline
        nd_coef = pm.Normal("nd_coef", 0.0, 0.35, dims="spline_basis")
        nd_effect = pm.Deterministic(
            "nd_effect",
            pm.math.dot(pm.Data("nd_basis", nd_basis, dims=("obs_id", "spline_basis")), nd_coef),
            dims="obs_id",
        )

        # Linear predictor
        mu = pm.Deterministic(
            "mu",
            alpha
            + entity_effect[entity_idx]
            + month_effect[month_idx]
            + price_slope[sku_idx] * log_rel_price_z
            + nd_effect,
            dims="obs_id",
        )

        # Likelihood
        if config.likelihood == "student_t":
            nu = pm.Exponential("nu", 1.0 / 29.0) + 1
            sigma = pm.HalfNormal("sigma", 0.5)
            pm.StudentT(
                "log_velocity_std_z_obs",
                nu=nu,
                mu=mu,
                sigma=sigma,
                observed=df["log_velocity_std_z"].values,
                dims="obs_id",
            )
        else:
            sigma = pm.HalfNormal("sigma", 0.5)
            pm.Normal(
                "log_velocity_std_z_obs",
                mu=mu,
                sigma=sigma,
                observed=df["log_velocity_std_z"].values,
                dims="obs_id",
            )

    return model


def build_pymc_model_v2(
    df: pd.DataFrame,
    meta: dict[str, Any],
    nd_basis: np.ndarray,
    config: ModelConfig = DEFAULT_CONFIG,
) -> pm.Model:
    """
    Build the improved hierarchical model (v2) with better pooling structure.

    Adds:
    - category×month effects
    - retailer×category effects
    """
    n_obs = len(df)
    n_entities = meta["n_entities"]
    n_months = meta["n_months"]
    n_skus = meta["n_skus"]
    n_brand_packs = meta["n_brand_packs"]
    n_categories = meta["n_categories"]
    n_retailers = meta["n_retailers"]
    n_spline_basis = nd_basis.shape[1]

    entity_idx = df["entity_idx"].values
    month_idx = df["month_idx"].values
    sku_idx = df["sku_idx"].values
    bp_idx = df["brand_pack_idx"].values
    cat_idx = df["category_idx"].values
    ret_idx = df["retailer_idx"].values
    log_rel_price_z = df["log_relative_price_z"].values

    coords = {
        "obs_id": np.arange(n_obs),
        "entity": np.arange(n_entities),
        "month": np.arange(n_months),
        "sku": np.arange(n_skus),
        "brand_pack": np.arange(n_brand_packs),
        "category": np.arange(n_categories),
        "retailer": np.arange(n_retailers),
        "spline_basis": np.arange(n_spline_basis),
    }

    with pm.Model(coords=coords) as model:
        pm.Data("entity_idx", entity_idx, dims="obs_id")
        pm.Data("month_idx", month_idx, dims="obs_id")
        pm.Data("sku_idx", sku_idx, dims="obs_id")
        pm.Data("brand_pack_idx", bp_idx, dims="obs_id")
        pm.Data("category_idx", cat_idx, dims="obs_id")
        pm.Data("retailer_idx", ret_idx, dims="obs_id")
        pm.Data("log_rel_price_z", log_rel_price_z, dims="obs_id")
        pm.Data("nd_basis", nd_basis, dims=("obs_id", "spline_basis"))

        alpha = pm.Normal("alpha", 0.0, 0.5)

        sigma_entity = pm.HalfNormal("sigma_entity", 0.5)
        entity_raw = pm.Normal("entity_raw", 0.0, 1.0, dims="entity")
        entity_effect = pm.Deterministic("entity_effect", sigma_entity * entity_raw, dims="entity")

        sigma_month = pm.HalfNormal("sigma_month", 0.35)
        month_raw = pm.Normal("month_raw", 0.0, 1.0, dims="month")
        month_effect = pm.Deterministic("month_effect", sigma_month * month_raw, dims="month")

        # Category × month interaction
        sigma_cat_month = pm.HalfNormal("sigma_cat_month", 0.2)
        cat_month_raw = pm.Normal("cat_month_raw", 0.0, 1.0, dims=("category", "month"))
        cat_month_effect = pm.Deterministic(
            "cat_month_effect", sigma_cat_month * cat_month_raw, dims=("category", "month")
        )

        # Retailer × category
        sigma_ret_cat = pm.HalfNormal("sigma_ret_cat", 0.2)
        ret_cat_raw = pm.Normal("ret_cat_raw", 0.0, 1.0, dims=("retailer", "category"))
        ret_cat_effect = pm.Deterministic(
            "ret_cat_effect", sigma_ret_cat * ret_cat_raw, dims=("retailer", "category")
        )

        price_bp_mu = pm.Normal("price_bp_mu", 0.0, 0.5, dims="brand_pack")
        price_sku_sigma = pm.HalfNormal("price_sku_sigma", 0.35)
        price_sku_raw = pm.Normal("price_sku_raw", 0.0, 1.0, dims="sku")
        price_slope = pm.Deterministic(
            "price_slope",
            price_bp_mu[bp_idx] + price_sku_sigma * price_sku_raw,
            dims="sku",
        )

        nd_coef = pm.Normal("nd_coef", 0.0, 0.35, dims="spline_basis")
        nd_effect = pm.Deterministic(
            "nd_effect",
            pm.math.dot(pm.Data("nd_basis", nd_basis, dims=("obs_id", "spline_basis")), nd_coef),
            dims="obs_id",
        )

        mu = pm.Deterministic(
            "mu",
            alpha
            + entity_effect[entity_idx]
            + month_effect[month_idx]
            + cat_month_effect[cat_idx, month_idx]
            + ret_cat_effect[ret_idx, cat_idx]
            + price_slope[sku_idx] * log_rel_price_z
            + nd_effect,
            dims="obs_id",
        )

        if config.likelihood == "student_t":
            nu = pm.Exponential("nu", 1.0 / 29.0) + 1
            sigma = pm.HalfNormal("sigma", 0.5)
            pm.StudentT(
                "log_velocity_std_z_obs",
                nu=nu,
                mu=mu,
                sigma=sigma,
                observed=df["log_velocity_std_z"].values,
                dims="obs_id",
            )
        else:
            sigma = pm.HalfNormal("sigma", 0.5)
            pm.Normal(
                "log_velocity_std_z_obs",
                mu=mu,
                sigma=sigma,
                observed=df["log_velocity_std_z"].values,
                dims="obs_id",
            )

    return model


# =============================================================================
# Joint SKU Market-Share Model (Dirichlet-Multinomial)
# =============================================================================

from dataclasses import dataclass

# Utility value for unavailable SKUs: exp(-30) ≈ 9.3e-14 share
# This ensures DirichletMultinomial concentration parameters a = concentration * share > 0
# for all SKUs, satisfying the distribution's requirement that all a > 0.
UNAVAILABLE_UTILITY = -30.0
UNAVAILABLE_SHARE_TOLERANCE = 1e-12  # Tests should assert share < this, not == 0


@dataclass(frozen=True, slots=True)
class JointModelConfig:
    """Configuration for the joint SKU market-share model."""
    draws: int = 1200
    tune: int = 1500
    chains: int = 4
    target_accept: float = 0.98
    random_seed: int = 42
    
    # Legacy hierarchy options (kept for backward compatibility, not used in current model)
    use_category_price_pooling: bool = True
    use_sku_nd_effects: bool = True
    use_category_nesting: bool = True
    
    # Legacy prior scales (kept for backward compatibility)
    beta_category_mu: float = -1.0
    beta_category_sigma: float = 0.75
    sigma_beta_brand: float = 0.35
    sigma_beta_sku: float = 0.25
    sigma_gamma1: float = 0.5
    sigma_gamma2: float = 0.5
    sigma_alpha_retailer_sku: float = 0.7
    sigma_alpha_sku: float = 0.5
    
    # Legacy concentration prior
    concentration_sigma: float = 1.0


def build_joint_model(
    choice_data: ChoiceSetData,
    config: JointModelConfig = JointModelConfig(),
) -> pm.Model:
    """
    Build the joint SKU market-share model using Dirichlet-Multinomial.

    Model:
    - Utilities: U[m,s] = alpha_rs + beta_s * log_rel_price + gamma1_s * ND + gamma2_s * ND^2 + pack + cat_month
    - Shares: softmax(U)
    - Observed: Dirichlet-Multinomial(total_units, shares * concentration)
    
    Initialization fixes:
    - Tighter priors on hierarchical SDs for stable NUTS sampling
    - Explicit initval=0.0 for raw parameters to avoid extreme initial utilities
    - Concentration prior shifted to encourage reasonable shares
    """
    n_markets, n_skus = choice_data.observed_units.shape
    n_retailers = len(choice_data.retailer_levels)
    n_categories = len(choice_data.category_levels)
    n_months = len(choice_data.month_levels)
    n_pack_groups = len(choice_data.pack_group_levels)
    n_skus_total = n_skus

    # Build indices
    market_retailer = choice_data.market_retailer_idx
    market_category = choice_data.market_category_idx
    market_month = choice_data.market_month_idx
    sku_brand = choice_data.sku_brand_idx
    sku_category = choice_data.sku_category_idx
    sku_pack_group = choice_data.sku_pack_group_idx
    brand_category = choice_data.brand_category_idx

    # Pre-compute log relative price and ND
    log_rel_price = np.log(np.maximum(choice_data.relative_price, 1e-4))
    nd = choice_data.nd
    nd_sq = nd ** 2

    with pm.Model() as model:
        # Data containers
        pm.Data("market_retailer", market_retailer)
        pm.Data("market_category", market_category)
        pm.Data("market_month", market_month)
        pm.Data("sku_brand", sku_brand)
        pm.Data("sku_category", sku_category)
        pm.Data("sku_pack_group", sku_pack_group)
        pm.Data("brand_category", brand_category)
        pm.Data("log_rel_price", log_rel_price)
        pm.Data("nd", nd)
        pm.Data("nd_sq", nd_sq)
        pm.Data("observed_units", choice_data.observed_units)
        pm.Data("market_totals", choice_data.market_total_units)
        pm.Data("available_mask", choice_data.available_mask.astype(float))

        # Global intercept - slightly tighter
        alpha = pm.Normal("alpha", 0.0, 0.35)

        # Retailer×SKU effects (non-centered) - tighter prior, explicit init
        sigma_rs = pm.HalfNormal("sigma_rs", 0.3)
        alpha_rs_raw = pm.Normal("alpha_rs_raw", 0.0, 1.0, shape=(n_retailers, n_skus), initval=np.zeros((n_retailers, n_skus)))
        alpha_rs = pm.Deterministic("alpha_rs", sigma_rs * alpha_rs_raw)

        # Price slope by SKU (non-centered, pooled by brand) - tighter prior, negative expected
        sigma_beta = pm.HalfNormal("sigma_beta", 0.3)
        beta_sku_raw = pm.Normal("beta_sku_raw", 0.0, 1.0, shape=n_skus, initval=np.zeros(n_skus))
        beta_sku = pm.Deterministic("beta_sku", -sigma_beta * beta_sku_raw)  # Negative elasticity

        # ND linear coefficient by SKU - tighter prior
        sigma_gamma1 = pm.HalfNormal("sigma_gamma1", 0.3)
        gamma1_sku_raw = pm.Normal("gamma1_sku_raw", 0.0, 1.0, shape=n_skus, initval=np.zeros(n_skus))
        gamma1_sku = pm.Deterministic("gamma1_sku", sigma_gamma1 * gamma1_sku_raw)

        # ND quadratic coefficient by SKU - tighter prior, negative for concave
        sigma_gamma2 = pm.HalfNormal("sigma_gamma2", 0.3)
        gamma2_sku_raw = pm.Normal("gamma2_sku_raw", 0.0, 1.0, shape=n_skus, initval=np.zeros(n_skus))
        gamma2_sku = pm.Deterministic("gamma2_sku", -sigma_gamma2 * gamma2_sku_raw)  # Concave

        # Pack group effect - tighter prior
        pack_effect = pm.Normal("pack_effect", 0.0, 0.35, shape=n_pack_groups)

        # Category-month effect - tighter prior
        sigma_cat_month = pm.HalfNormal("sigma_cat_month", 0.25)
        cat_month_raw = pm.Normal("cat_month_raw", 0.0, 1.0, shape=(n_categories, n_months), initval=np.zeros((n_categories, n_months)))
        cat_month_effect = pm.Deterministic("cat_month_effect", sigma_cat_month * cat_month_raw)

        # Concentration parameter for Dirichlet-Multinomial - shifted prior
        concentration = pm.HalfNormal("concentration", 1.5) + 0.5  # Ensures > 0.5

        # Deterministic utility computation
        def _compute_utility(
            alpha_rs_val, beta_sku_val, gamma1_sku_val, gamma2_sku_val,
            pack_effect_val, cat_month_effect_val,
            market_retailer_val, market_category_val, market_month_val,
            sku_brand_val, sku_category_val, sku_pack_group_val,
            log_rel_price_val, nd_val, nd_sq_val,
        ):
            # alpha_rs_val: (n_retailers, n_skus)
            # market_retailer_val: (n_markets,)
            # alpha_rs_val[market_retailer_val, :] -> (n_markets, n_skus)
            alpha_rs_market = alpha_rs_val[market_retailer_val, :]
            
            # beta_sku_val: (n_skus,), log_rel_price_val: (n_markets, n_skus)
            # Reshape beta_sku_val to (1, n_skus) for explicit broadcasting
            price_term = beta_sku_val.reshape((1, n_skus)) * log_rel_price_val
            
            # gamma1_sku_val: (n_skus,), nd_val: (n_markets, n_skus)
            nd_term1 = gamma1_sku_val.reshape((1, n_skus)) * nd_val
            
            # gamma2_sku_val: (n_skus,), nd_sq_val: (n_markets, n_skus)
            nd_term2 = gamma2_sku_val.reshape((1, n_skus)) * nd_sq_val
            
            # pack_effect_val: (n_pack_groups,), sku_pack_group_val: (n_skus,)
            # pack_effect_val[sku_pack_group_val] -> (n_skus,)
            # Reshape to (1, n_skus) for broadcasting over markets
            pack_term = pack_effect_val[sku_pack_group_val].reshape((1, n_skus))
            
            # cat_month_effect_val: (n_categories, n_months)
            # market_category_val: (n_markets,), market_month_val: (n_markets,)
            # cat_month_effect_val[market_category_val, market_month_val] -> (n_markets,)
            # Reshape to (n_markets, 1) for broadcasting over SKUs
            cat_month_term = cat_month_effect_val[market_category_val, market_month_val].reshape((n_markets, 1))
            
            utility = (
                alpha
                + alpha_rs_val[market_retailer_val, :]
                + price_term
                + gamma1_sku_val.reshape((1, n_skus)) * nd_val
                + gamma2_sku_val.reshape((1, n_skus)) * nd_sq_val
                + pack_term
                + cat_month_term
            )
            # Mask unavailable: use UNAVAILABLE_UTILITY so softmax gives small but positive shares
            # (exp(-30) ≈ 9e-14). This ensures a = concentration * sku_share > 0 for all SKUs,
            # satisfying DirichletMultinomial's requirement that all concentration parameters a > 0.
            utility = pm.math.where(
                choice_data.available_mask,
                utility,
                UNAVAILABLE_UTILITY
            )
            return utility

        utility = _compute_utility(
            alpha_rs, beta_sku, gamma1_sku, gamma2_sku,
            pack_effect, cat_month_effect,
            market_retailer, market_category, market_month,
            sku_brand, sku_category, sku_pack_group,
            log_rel_price, nd, nd_sq,
        )

        # Softmax shares
        sku_share = pm.Deterministic("sku_share", pm.math.softmax(utility, axis=1))

        # Inclusive value (log-sum-exp of utilities) for nested logit compatibility
        inclusive_value = pm.Deterministic("inclusive_value", pm.math.logsumexp(utility, axis=1))

        # Dirichlet-Multinomial likelihood
        alpha_dm = concentration * sku_share
        pm.DirichletMultinomial(
            "sku_units_obs",
            a=alpha_dm,
            n=choice_data.market_total_units,
            observed=choice_data.observed_units,
        )

    return model


def extract_joint_posterior(
    idata: Any,
    max_draws: int = 500,
    random_seed: int = 42,
    choice_data: Any = None,
) -> dict[str, np.ndarray]:
    """Extract posterior draws for joint model scenario evaluation.
    
    Args:
        idata: InferenceData object from PyMC sampling
        max_draws: Maximum number of draws to return
        random_seed: Random seed for subsampling
        choice_data: Optional ChoiceSetData to compute sku_share deterministics
    """
    rng = np.random.default_rng(random_seed)

    def get_var(name: str) -> np.ndarray:
        posterior = idata.posterior
        if name not in posterior:
            raise KeyError(f"Posterior variable '{name}' not found")
        arr = posterior[name].values
        # Combine chains and draws
        if arr.ndim >= 2 and arr.shape[0] * arr.shape[1] > 1:
            arr = arr.reshape((-1,) + arr.shape[2:])
        return arr

    # Extract all needed parameters
    posterior = {
        "alpha": get_var("alpha"),
        "alpha_rs": get_var("alpha_rs"),
        "beta_sku": get_var("beta_sku"),
        "gamma1_sku": get_var("gamma1_sku"),
        "gamma2_sku": get_var("gamma2_sku"),
        "pack_effect": get_var("pack_effect"),
        "cat_month_effect": get_var("cat_month_effect"),
        "concentration": get_var("concentration"),
    }

    # Backward compatibility aliases
    # alpha_sku: average over retailers
    posterior["alpha_sku"] = posterior["alpha_rs"].mean(axis=1)  # (draws, n_skus)
    # alpha_retailer_sku: alias for alpha_rs
    posterior["alpha_retailer_sku"] = posterior["alpha_rs"]
    # concentration_category: alias for concentration (single value per draw)
    posterior["concentration_category"] = posterior["concentration"]

    # Compute sku_share if choice_data is provided
    if choice_data is not None:
        posterior["sku_share"] = _compute_sku_share_from_posterior(posterior, choice_data)

    # Subsample draws if needed
    n_draws = posterior["alpha"].shape[0]
    if n_draws > max_draws:
        idx = rng.choice(n_draws, size=max_draws, replace=False)
        for key in posterior:
            if posterior[key] is not None:
                posterior[key] = posterior[key][idx]

    return posterior


def _compute_sku_share_from_posterior(
    posterior: dict[str, np.ndarray],
    choice_data: Any,
) -> np.ndarray:
    """Compute sku_share for each posterior draw using choice_data."""
    n_draws = posterior["alpha"].shape[0]
    n_markets = len(choice_data.market_ids)
    n_skus = choice_data.observed_units.shape[1]
    
    # Indices
    market_retailer = choice_data.market_retailer_idx
    market_category = choice_data.market_category_idx
    market_month = choice_data.market_month_idx
    sku_category = choice_data.sku_category_idx
    sku_pack_group = choice_data.sku_pack_group_idx
    log_rel_price = np.log(np.maximum(choice_data.relative_price, 1e-4))
    nd = choice_data.nd
    nd_sq = nd ** 2
    
    sku_shares = np.zeros((n_draws, n_markets, n_skus))
    
    for d in range(n_draws):
        # Get draw parameters
        alpha = posterior["alpha"][d]
        alpha_rs = posterior["alpha_rs"][d]  # (n_retailers, n_skus)
        beta_sku = posterior["beta_sku"][d]  # (n_skus,)
        gamma1_sku = posterior["gamma1_sku"][d]  # (n_skus,)
        gamma2_sku = posterior["gamma2_sku"][d]  # (n_skus,)
        pack_effect = posterior["pack_effect"][d]  # (n_pack_groups,)
        cat_month_effect = posterior["cat_month_effect"][d]  # (n_categories, n_months)
        
        # Compute utility for each market and SKU
        # alpha_rs[market_retailer, :] -> (n_markets, n_skus)
        alpha_rs_market = alpha_rs[market_retailer, :]
        
        # beta_sku * log_rel_price -> (n_markets, n_skus)
        price_term = beta_sku.reshape((1, n_skus)) * log_rel_price
        
        # gamma1_sku * nd -> (n_markets, n_skus)
        nd_term1 = gamma1_sku.reshape((1, n_skus)) * nd
        
        # gamma2_sku * nd_sq -> (n_markets, n_skus)
        nd_term2 = gamma2_sku.reshape((1, n_skus)) * nd_sq
        
        # pack_effect[sku_pack_group] -> (n_skus,) -> (1, n_skus)
        pack_term = pack_effect[sku_pack_group].reshape((1, n_skus))
        
        # cat_month_effect[market_category, market_month] -> (n_markets,) -> (n_markets, 1)
        cat_month_term = cat_month_effect[market_category, market_month].reshape((n_markets, 1))
        
        utility = (
            alpha
            + alpha_rs_market
            + price_term
            + nd_term1
            + nd_term2
            + pack_term
            + cat_month_term
        )
        
        # Mask unavailable
        utility = np.where(
            choice_data.available_mask,
            utility,
            UNAVAILABLE_UTILITY
        )
        
        # Softmax
        sku_shares[d] = np.exp(utility) / np.exp(utility).sum(axis=1, keepdims=True)
    
    return sku_shares
