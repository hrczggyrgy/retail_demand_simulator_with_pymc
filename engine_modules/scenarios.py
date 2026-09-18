"""
Scenario engine for counterfactual evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contracts import (
    ChoiceSetData,
    MarketOutcome,
    ScenarioAction,
    ScenarioOutcome,
    ScenarioPlan,
    ScenarioResult,
)
from engine_modules.choice_sets import _recompute_relative_prices


def run_scenario_plan(
    plan: ScenarioPlan,
    choice_data: ChoiceSetData,
    posterior_cache: dict[str, np.ndarray],
    n_draws: int | None = None,
    random_seed: int = 42,
) -> ScenarioOutcome:
    """
    Run a full ScenarioPlan evaluation: four true counterfactuals + market-first aggregation.
    
    This is the primary entry point for scenario evaluation. It:
    1. Runs four counterfactuals: baseline, price_only, distribution_only, combined
    2. Computes per-draw per-market reconciliation checks
    3. Aggregates to market-level (retailer×category) for commercial decision-making
    4. Provides SKU-level detail for diagnostic drill-down
    5. Builds driver waterfall and source-destination flows
    
    Args:
        plan: High-level ScenarioPlan with focal actions
        choice_data: ChoiceSetData with baseline arrays
        posterior_cache: Posterior parameter cache
        n_draws: Number of posterior draws to use
        random_seed: Random seed for reproducible draws
    
    Returns:
        ScenarioOutcome with market-level outcomes (primary) and SKU detail (secondary)
    """
    # Build action sets for four counterfactuals
    full_actions = list(plan.actions)
    
    price_actions = [
        ScenarioAction(
            retailer=a.retailer,
            category=a.category,
            brand=a.brand,
            sku=a.sku,
            month=a.month,
            new_price_std=a.new_price_std,
            new_nd=None,
            nd_mode=a.nd_mode,
        )
        for a in full_actions if a.new_price_std is not None
    ]
    
    nd_actions = [
        ScenarioAction(
            retailer=a.retailer,
            category=a.category,
            brand=a.brand,
            sku=a.sku,
            month=a.month,
            new_price_std=None,
            new_nd=a.new_nd,
            nd_mode=a.nd_mode,
        )
        for a in full_actions if a.new_nd is not None
    ]
    
    # Run four counterfactuals
    baseline_result = run_joint_scenario_draws(
        choice_data=choice_data,
        posterior_cache=posterior_cache,
        actions=[],
        n_draws=n_draws,
        random_seed=random_seed,
    )
    
    price_only_result = run_joint_scenario_draws(
        choice_data=choice_data,
        posterior_cache=posterior_cache,
        actions=price_actions,
        n_draws=n_draws,
        random_seed=random_seed,
    ) if price_actions else baseline_result
    
    dist_only_result = run_joint_scenario_draws(
        choice_data=choice_data,
        posterior_cache=posterior_cache,
        actions=nd_actions,
        n_draws=n_draws,
        random_seed=random_seed,
    ) if nd_actions else baseline_result
    
    combined_result = run_joint_scenario_draws(
        choice_data=choice_data,
        posterior_cache=posterior_cache,
        actions=full_actions,
        n_draws=n_draws,
        random_seed=random_seed,
    )
    
    # Reconciliation check
    recon = check_scenario_reconciliation(combined_result, choice_data)
    
    # Market-first aggregation (retailer×category = decision unit)
    market_outcomes = _aggregate_to_market_outcomes(
        baseline_result, combined_result, choice_data, plan
    )
    
    # SKU-level detail for drill-down
    sku_outcomes = _aggregate_to_sku_outcomes(
        baseline_result, combined_result, choice_data
    )
    
    # Driver waterfall
    driver_waterfall = create_driver_waterfall(
        aggregate_scenario_result(baseline_result, choice_data, "sku"),
        aggregate_scenario_result(price_only_result, choice_data, "sku"),
        aggregate_scenario_result(dist_only_result, choice_data, "sku"),
        aggregate_scenario_result(combined_result, choice_data, "sku"),
        selected_sku=plan.actions[0].sku if len(plan.actions) == 1 else None,
    )
    
    # Source-destination (if single SKU)
    source_destination = None
    if len(plan.actions) == 1:
        source_destination = compute_source_destination_flows(
            combined_result, choice_data, plan.actions[0].sku
        )
    
    # Diagnostics
    diagnostics = None
    if "divergences" in posterior_cache:
        # Build ConvergenceDiagnostics from posterior cache if available
        pass  # Will be provided by caller
    
    return ScenarioOutcome(
        plan=plan,
        market_outcomes=market_outcomes,
        sku_outcomes=sku_outcomes,
        driver_waterfall=driver_waterfall,
        source_destination=source_destination,
        reconciliation_passed=recon["all_passed"],
        diagnostics=diagnostics,
    )


def _aggregate_to_market_outcomes(
    baseline_result: ScenarioResult,
    combined_result: ScenarioResult,
    choice_data: ChoiceSetData,
    plan: ScenarioPlan,
) -> tuple[MarketOutcome, ...]:
    """
    Aggregate scenario results to market-level (retailer×category).
    
    This is the PRIMARY commercial output - the decision unit.
    Shows total market impact, focal SKU/brand impact, and competitor sources.
    """
    n_markets = len(choice_data.market_ids)
    market_outcomes = []
    
    # Determine focal SKU/brand
    focal_sku = plan.actions[0].sku if len(plan.actions) == 1 else None
    focal_brand = plan.actions[0].brand if len(plan.actions) == 1 else None
    
    # Compute source-destination for competitor attribution
    source_flows = {}
    if focal_sku:
        try:
            source_df = compute_source_destination_flows(combined_result, choice_data, focal_sku)
            # Aggregate by relationship
            if not source_df.empty:
                source_flows = (
                    source_df.groupby("source_relationship")["delta_standard_volume_p50"]
                    .sum()
                    .to_dict()
                )
        except Exception:
            pass
    
    for m_idx in range(n_markets):
        market_id = choice_data.market_ids[m_idx]
        parts = market_id.split("|")
        month_str = parts[0]
        retailer = parts[1] if len(parts) > 1 else ""
        category = parts[2] if len(parts) > 2 else ""
        
        # Baseline and scenario volumes
        baseline_vol = baseline_result.baseline_units_p50[m_idx].sum()
        scenario_vol = combined_result.scenario_units_p50[m_idx].sum()
        delta_vol = scenario_vol - baseline_vol
        delta_pct = delta_vol / baseline_vol * 100 if baseline_vol > 0 else 0.0
        
        # Share
        baseline_share = 1.0  # This market is 100% of itself
        scenario_share = scenario_vol / baseline_vol if baseline_vol > 0 else 1.0
        
        # Revenue (approximate via price_std)
        # Would need price data for exact revenue
        baseline_rev = baseline_vol  # Placeholder
        scenario_rev = scenario_vol
        
        # Focal SKU impact
        focal_baseline = None
        focal_scenario = None
        focal_delta = None
        if focal_sku:
            sku_idx = choice_data.sku_to_idx.get(focal_sku)
            if sku_idx is not None:
                focal_baseline = baseline_result.baseline_units_p50[m_idx, sku_idx]
                focal_scenario = combined_result.scenario_units_p50[m_idx, sku_idx]
                focal_delta = focal_scenario - focal_baseline
        
        # Probability positive (from per-draw deltas)
        delta_draws = (combined_result.scenario_units - baseline_result.baseline_units)[:, m_idx].sum(axis=1)
        prob_positive = float((delta_draws > 0).mean())
        
        market_outcomes.append(MarketOutcome(
            market_id=market_id,
            retailer=retailer,
            category=category,
            baseline_volume=baseline_vol,
            scenario_volume=scenario_vol,
            delta_volume=delta_vol,
            delta_volume_pct=delta_pct,
            baseline_share=baseline_share,
            scenario_share=scenario_share,
            delta_share_pp=(scenario_share - baseline_share) * 100,
            baseline_revenue=baseline_rev,
            scenario_revenue=scenario_rev,
            delta_revenue=scenario_rev - baseline_rev,
            focal_sku=focal_sku,
            focal_baseline_volume=focal_baseline,
            focal_scenario_volume=focal_scenario,
            focal_delta_volume=focal_delta,
            competitor_sources=source_flows if source_flows else None,
            probability_positive=prob_positive,
            is_decision_ready=False,  # Set by caller based on diagnostics
        ))
    
    return tuple(market_outcomes)


def _aggregate_to_sku_outcomes(
    baseline_result: ScenarioResult,
    combined_result: ScenarioResult,
    choice_data: ChoiceSetData,
) -> pd.DataFrame:
    """Aggregate to SKU-level detail with intervals for drill-down."""
    n_markets = len(choice_data.market_ids)
    n_skus = len(choice_data.sku_ids)
    n_draws = baseline_result.baseline_units.shape[0]
    
    rows = []
    for m_idx in range(n_markets):
        market_id = choice_data.market_ids[m_idx]
        retailer = choice_data.retailer_levels[choice_data.market_retailer_idx[m_idx]]
        category = choice_data.category_levels[choice_data.market_category_idx[m_idx]]
        month = choice_data.month_levels[choice_data.market_month_idx[m_idx]]
        
        for s_idx in range(n_skus):
            sku = choice_data.sku_ids[s_idx]
            brand = choice_data.brand_levels[choice_data.sku_brand_idx[s_idx]]
            pack_group = choice_data.pack_group_levels[choice_data.sku_pack_group_idx[s_idx]]
            
            base_draws = baseline_result.baseline_units[:, m_idx, s_idx]
            scen_draws = combined_result.scenario_units[:, m_idx, s_idx]
            delta_draws = scen_draws - base_draws
            
            rows.append({
                "market_id": market_id,
                "retailer": retailer,
                "category": category,
                "month": month,
                "sku": sku,
                "brand": brand,
                "pack_group": pack_group,
                "baseline_units_p05": float(np.percentile(base_draws, 5)),
                "baseline_units_p50": float(np.median(base_draws)),
                "baseline_units_p95": float(np.percentile(base_draws, 95)),
                "scenario_units_p05": float(np.percentile(scen_draws, 5)),
                "scenario_units_p50": float(np.median(scen_draws)),
                "scenario_units_p95": float(np.percentile(scen_draws, 95)),
                "delta_units_p05": float(np.percentile(delta_draws, 5)),
                "delta_units_p50": float(np.median(delta_draws)),
                "delta_units_p95": float(np.percentile(delta_draws, 95)),
                "probability_positive": float((delta_draws > 0).mean()),
            })
    
    return pd.DataFrame(rows)


def run_joint_scenario_draws(
    choice_data: ChoiceSetData,
    posterior_cache: dict[str, np.ndarray],
    actions: list[ScenarioAction],
    n_draws: int | None = None,
    random_seed: int = 42,
) -> ScenarioResult:
    """
    Run counterfactual scenario by applying actions to the joint model.
    
    For each posterior draw:
    1. Compute baseline utilities and shares
    2. Apply scenario actions (price/ND changes) to absolute price_std
    3. Recalculate peer prices for affected markets
    4. Recompute relative prices for all SKUs in affected markets
    5. Recalculate utilities and shares
    6. Draw baseline and scenario package units from Dirichlet-Multinomial
    7. Compute deltas
    
    Args:
        choice_data: ChoiceSetData with baseline arrays (including price_std)
        posterior_cache: Posterior parameter cache
        actions: List of ScenarioAction objects
        n_draws: Number of posterior draws to use (None = all in cache)
        random_seed: Random seed for reproducible draws
    """
    n_markets, n_skus = choice_data.observed_units.shape
    
    # Baseline absolute price and ND
    baseline_price_std = choice_data.price_std.copy()
    baseline_nd = choice_data.nd.copy()
    baseline_available = choice_data.available_mask.copy()
    baseline_volume = choice_data.observed_units.copy()
    market_totals = choice_data.market_total_units.copy()
    
    # Build market lookup
    market_lookup = {m_id: m_idx for m_idx, m_id in enumerate(choice_data.market_ids)}
    sku_lookup = choice_data.sku_to_idx
    
    # Prepare scenario absolute price_std and ND
    scenario_price_std = baseline_price_std.copy()
    scenario_nd = baseline_nd.copy()
    scenario_available = baseline_available.copy()
    
    # Apply actions to scenario arrays
    affected_markets = set()
    
    for action in actions:
        month_str = pd.Timestamp(action.month).strftime("%Y-%m")
        m_id = f"{month_str}|{action.retailer}|{action.category}"
        
        if m_id not in market_lookup:
            continue
        m_idx = market_lookup[m_id]
        
        s_idx = sku_lookup.get(action.sku)
        if s_idx is None:
            continue
        
        affected_markets.add(m_idx)
        
        # Price change - apply absolute price change
        if action.new_price_std is not None:
            scenario_price_std[m_idx, s_idx] = action.new_price_std
        
        # ND change - allow exact zero for ND=0 (unavailable)
        if action.new_nd is not None:
            if action.nd_mode == "pp":
                scenario_nd[m_idx, s_idx] = baseline_nd[m_idx, s_idx] + action.new_nd
            elif action.nd_mode == "relative":
                scenario_nd[m_idx, s_idx] = baseline_nd[m_idx, s_idx] * (1 + action.new_nd)
            elif action.nd_mode == "absolute":
                scenario_nd[m_idx, s_idx] = action.new_nd
            # Clip to [0, 1] - allow exact 0 for unavailable SKUs
            scenario_nd[m_idx, s_idx] = np.clip(scenario_nd[m_idx, s_idx], 0.0, 1.0)
            # Available if ND > 0
            scenario_available[m_idx, s_idx] = scenario_nd[m_idx, s_idx] > 0
    
    # Recalculate peer prices and relative prices for affected markets
    scenario_rel_price = _recompute_relative_prices(
        choice_data, scenario_price_std, affected_markets
    )
    
    # Baseline relative price (precomputed in choice_data)
    baseline_rel_price = choice_data.relative_price
    
    # Determine number of draws
    if n_draws is None:
        n_draws = posterior_cache["beta_sku"].shape[0]
    else:
        n_draws = min(n_draws, posterior_cache["beta_sku"].shape[0])
    
    # Storage for results
    baseline_units_all = np.zeros((n_draws, n_markets, n_skus), dtype=np.float64)
    scenario_units_all = np.zeros((n_draws, n_markets, n_skus), dtype=np.float64)
    baseline_shares_all = np.zeros((n_draws, n_markets, n_skus), dtype=np.float64)
    scenario_shares_all = np.zeros((n_draws, n_markets, n_skus), dtype=np.float64)
    
    rng = np.random.default_rng(random_seed)
    
    for d in range(n_draws):
        # Baseline
        baseline_utility = _compute_utility(
            choice_data, posterior_cache, d,
            baseline_rel_price,
            baseline_nd,
            baseline_available,
        )
        baseline_shares = _softmax(baseline_utility)
        baseline_shares_all[d] = baseline_shares
        
        # Draw baseline units from Dirichlet-Multinomial
        for m_idx in range(n_markets):
            avail = baseline_available[m_idx]
            if avail.any():
                total = market_totals[m_idx]
                alpha = posterior_cache["concentration"][d] * baseline_shares[m_idx, avail]
                baseline_units_all[d, m_idx, avail] = rng.dirichlet(alpha) * total
        
        # Scenario
        scenario_utility = _compute_utility(
            choice_data, posterior_cache, d,
            scenario_rel_price,
            scenario_nd,
            scenario_available,
        )
        scenario_shares = _softmax(scenario_utility)
        scenario_shares_all[d] = scenario_shares
        
        # Draw scenario units
        for m_idx in range(n_markets):
            avail = scenario_available[m_idx]
            if avail.any():
                total = market_totals[m_idx]
                alpha = posterior_cache["concentration"][d] * scenario_shares[m_idx, avail]
                scenario_units_all[d, m_idx, avail] = rng.dirichlet(alpha) * total
    
    # Compute summaries
    baseline_units_p50 = np.median(baseline_units_all, axis=0)
    scenario_units_p50 = np.median(scenario_units_all, axis=0)
    baseline_units_p05 = np.percentile(baseline_units_all, 5, axis=0)
    scenario_units_p05 = np.percentile(scenario_units_all, 5, axis=0)
    baseline_units_p95 = np.percentile(baseline_units_all, 95, axis=0)
    scenario_units_p95 = np.percentile(scenario_units_all, 95, axis=0)
    
    delta_units_p50 = scenario_units_p50 - baseline_units_p50
    delta_units_p05 = scenario_units_p05 - baseline_units_p05
    delta_units_p95 = scenario_units_p95 - baseline_units_p95
    
    return ScenarioResult(
        baseline_units=baseline_units_all,
        scenario_units=scenario_units_all,
        baseline_shares=baseline_shares_all,
        scenario_shares=scenario_shares_all,
        baseline_units_p50=baseline_units_p50,
        scenario_units_p50=scenario_units_p50,
        baseline_units_p05=baseline_units_p05,
        scenario_units_p05=scenario_units_p05,
        baseline_units_p95=baseline_units_p95,
        scenario_units_p95=scenario_units_p95,
        delta_units_p50=delta_units_p50,
        delta_units_p05=delta_units_p05,
        delta_units_p95=delta_units_p95,
        choice_data=choice_data,
        actions=tuple(actions),
    )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _compute_utility(
    choice_data: ChoiceSetData,
    posterior_cache: dict[str, np.ndarray],
    draw_idx: int,
    relative_price: np.ndarray,
    nd: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    """
    Compute SKU utility for a single posterior draw.
    
    U[m, s] = alpha_rs + beta_s * log_rel_price + gamma1_s * ND + gamma2_s * ND^2 + pack + cat_month
    """
    n_markets, n_skus = relative_price.shape
    
    # Extract draw-specific parameters
    beta_sku = posterior_cache["beta_sku"][draw_idx]
    gamma1_sku = posterior_cache["gamma1_sku"][draw_idx]
    gamma2_sku = posterior_cache["gamma2_sku"][draw_idx]
    alpha_sku = posterior_cache["alpha_rs"][draw_idx]
    pack_effect = posterior_cache["pack_effect"][draw_idx]
    cat_month_effect = posterior_cache["cat_month_effect"][draw_idx]
    
    # Mappings
    market_retailer_idx = choice_data.market_retailer_idx
    market_category_idx = choice_data.market_category_idx
    market_month_idx = choice_data.market_month_idx
    sku_pack_group_idx = choice_data.sku_pack_group_idx
    
    # Pre-compute
    log_rel_price = np.log(np.maximum(relative_price, 1e-4))
    nd_sq = nd ** 2
    
    # Vectorized utility computation
    utility = np.zeros((n_markets, n_skus))
    
    for m_idx in range(n_markets):
        retailer = market_retailer_idx[m_idx]
        category = market_category_idx[m_idx]
        month = market_month_idx[m_idx]
        
        # Base utility for this market
        base_u = (
            alpha_sku[retailer, :]
            + beta_sku * log_rel_price[m_idx, :]
            + gamma1_sku * nd[m_idx, :]
            + gamma2_sku * nd_sq[m_idx, :]
            + pack_effect[sku_pack_group_idx]
            + cat_month_effect[category, month]
        )
        
        # Mask unavailable
        base_u = np.where(available[m_idx], base_u, -1e9)
        utility[m_idx] = base_u
    
    return utility


# =============================================================================
# Source-Destination Analysis
# =============================================================================

SOURCE_DESTINATION_COLUMNS = (
    "selected_sku",
    "source_relationship",
    "baseline_standard_volume_p50",
    "scenario_standard_volume_p50",
    "delta_standard_volume_p50",
    "delta_standard_volume_p05",
    "delta_standard_volume_p95",
)


def compute_source_destination_flows(
    result: ScenarioResult,
    choice_data: ChoiceSetData,
    selected_sku: str,
) -> pd.DataFrame:
    """
    Compute source-destination reallocation flows for a focal SKU.
    
    Segments:
    - Selected SKU
    - Same-brand other SKUs
    - Other brands (same pack / other pack)
    - Other categories (same retailer)
    - Same category other retailers
    """
    n_markets, n_skus = choice_data.observed_units.shape
    sku_ids = choice_data.sku_ids
    sku_to_idx = choice_data.sku_to_idx
    
    selected_idx = sku_to_idx[selected_sku]
    
    # Get metadata
    sku_brand = [choice_data.brand_levels[choice_data.sku_brand_idx[i]] for i in range(n_skus)]
    sku_category = [choice_data.category_levels[choice_data.sku_category_idx[i]] for i in range(n_skus)]
    sku_pack_group = [choice_data.pack_group_levels[choice_data.sku_pack_group_idx[i]] for i in range(n_skus)]
    sku_retailer = [choice_data.retailer_levels[choice_data.market_retailer_idx[m]] for m in range(n_markets)]
    
    selected_brand = sku_brand[selected_idx]
    selected_category = sku_category[selected_idx]
    selected_pack = sku_pack_group[selected_idx]
    selected_retailer = sku_retailer  # varies by market
    
    delta_p50 = result.delta_units_p50
    delta_p05 = result.delta_units_p05
    delta_p95 = result.delta_units_p95
    
    rows = []
    for m_idx in range(n_markets):
        ret = sku_retailer[m_idx]
        cat = sku_category  # same across markets
        
        for s_idx in range(n_skus):
            if s_idx == selected_idx:
                continue
            
            brand = sku_brand[s_idx]
            cat_s = sku_category[s_idx]
            pack = sku_pack_group[s_idx]
            
            # Determine relationship
            if brand == selected_brand:
                relationship = "Same Brand"
            elif cat_s == selected_category and pack == selected_pack:
                relationship = "Other Brand (Same Pack)"
            elif cat_s == selected_category:
                relationship = "Other Brand (Other Pack)"
            elif ret == selected_retailer[m_idx]:
                relationship = "Other Category (Same Retailer)"
            else:
                relationship = "Other Retailer (Same Category)"
            
            rows.append({
                "selected_sku": selected_sku,
                "source_relationship": relationship,
                "baseline_standard_volume_p50": result.baseline_units_p50[m_idx, s_idx],
                "scenario_standard_volume_p50": result.scenario_units_p50[m_idx, s_idx],
                "delta_standard_volume_p50": delta_p50[m_idx, s_idx],
                "delta_standard_volume_p05": delta_p05[m_idx, s_idx],
                "delta_standard_volume_p95": delta_p95[m_idx, s_idx],
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# Market Impact Comparison
# =============================================================================

MARKET_LEVELS = {
    "market": ["month"],
    "retailer": ["month", "retailer"],
    "retailer_category": ["month", "retailer", "category"],
    "brand": ["month", "retailer", "category", "brand"],
    "brand_pack": ["month", "retailer", "category", "brand", "pack_group"],
    "sku": ["month", "retailer", "category", "brand", "pack_group", "sku"],
}


def aggregate_scenario_result(
    result: ScenarioResult,
    choice_data: ChoiceSetData,
    level: str = "sku",
) -> pd.DataFrame:
    """
    Aggregate scenario results to specified level.
    
    Levels: market, retailer, category, brand, brand_pack, sku
    
    Aggregation is done per-draw first, then quantiles are computed
    over the aggregated draw distribution (correct uncertainty propagation).
    """
    n_markets, n_skus = choice_data.observed_units.shape
    sku_ids = choice_data.sku_ids
    n_draws = result.baseline_units.shape[0]
    
    # Build SKU metadata DataFrame
    sku_meta = []
    for s_idx, sku in enumerate(sku_ids):
        brand_idx = choice_data.sku_brand_idx[s_idx]
        brand = choice_data.brand_levels[brand_idx]
        cat_idx = choice_data.sku_category_idx[s_idx]
        category = choice_data.category_levels[cat_idx]
        pg_idx = choice_data.sku_pack_group_idx[s_idx]
        pack_group = choice_data.pack_group_levels[pg_idx]
        sku_meta.append({
            "sku_idx": s_idx,
            "sku": sku,
            "brand": brand,
            "category": category,
            "pack_group": pack_group,
        })
    sku_meta_df = pd.DataFrame(sku_meta)
    
    # Build market metadata
    market_meta = []
    for m_idx in range(n_markets):
        market_meta.append({
            "market_idx": m_idx,
            "market_id": choice_data.market_ids[m_idx],
            "retailer": choice_data.retailer_levels[choice_data.market_retailer_idx[m_idx]],
            "category": choice_data.category_levels[choice_data.market_category_idx[m_idx]],
            "month": choice_data.month_levels[choice_data.market_month_idx[m_idx]],
        })
    market_meta_df = pd.DataFrame(market_meta)
    
    # Determine grouping columns
    if level == "market":
        group_cols = []
    elif level == "retailer":
        group_cols = ["retailer"]
    elif level == "category":
        group_cols = ["retailer", "category"]
    elif level == "brand":
        group_cols = ["retailer", "category", "brand"]
    elif level == "brand_pack":
        group_cols = ["retailer", "category", "brand", "pack_group"]
    elif level == "sku":
        group_cols = ["retailer", "category", "brand", "pack_group", "sku"]
    else:
        raise ValueError(f"Unknown level: {level}")
    
    # Build groupby keys for each SKU/market combination
    group_keys = {}
    for m_idx in range(n_markets):
        m_info = market_meta_df.iloc[m_idx]
        for s_idx in range(n_skus):
            s_info = sku_meta_df.iloc[s_idx]
            if group_cols:
                key = tuple(
                    m_info[col] if col in m_info else s_info[col]
                    for col in group_cols
                )
            else:
                key = ()
            group_keys[(m_idx, s_idx)] = key
    
    unique_keys = sorted(set(group_keys.values()))
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    n_groups = len(unique_keys)
    
    # Aggregate per draw, then compute quantiles
    baseline_agg_draws = np.zeros((n_draws, n_groups), dtype=np.float64)
    scenario_agg_draws = np.zeros((n_draws, n_groups), dtype=np.float64)
    
    for d in range(n_draws):
        for (m_idx, s_idx), key in group_keys.items():
            g_idx = key_to_idx[key]
            baseline_agg_draws[d, g_idx] += result.baseline_units[d, m_idx, s_idx]
            scenario_agg_draws[d, g_idx] += result.scenario_units[d, m_idx, s_idx]
    
    delta_agg_draws = scenario_agg_draws - baseline_agg_draws
    
    def quantiles(arr, axis=0):
        return (
            np.quantile(arr, 0.05, axis=axis),
            np.quantile(arr, 0.50, axis=axis),
            np.quantile(arr, 0.95, axis=axis),
        )
    
    baseline_p05, baseline_p50, baseline_p95 = quantiles(baseline_agg_draws)
    scenario_p05, scenario_p50, scenario_p95 = quantiles(scenario_agg_draws)
    delta_p05, delta_p50, delta_p95 = quantiles(delta_agg_draws)
    
    # Build result DataFrame
    if group_cols:
        rows = []
        for key in unique_keys:
            row = {}
            for i, col in enumerate(group_cols):
                row[col] = key[i]
            rows.append(row)
        agg = pd.DataFrame(rows)
    else:
        agg = pd.DataFrame([{}])
    
    agg["baseline_units"] = baseline_p50
    agg["baseline_units_p05"] = baseline_p05
    agg["baseline_units_p95"] = baseline_p95
    agg["scenario_units"] = scenario_p50
    agg["scenario_units_p05"] = scenario_p05
    agg["scenario_units_p95"] = scenario_p95
    agg["delta_units"] = delta_p50
    agg["delta_units_p05"] = delta_p05
    agg["delta_units_p50"] = delta_p50
    agg["delta_units_p95"] = delta_p95
    
    # Add pct change
    agg["delta_units_pct"] = np.where(
        agg["baseline_units"] > 0,
        agg["delta_units"] / agg["baseline_units"] * 100,
        0.0,
    )
    
    return agg


def check_scenario_reconciliation(
    result: ScenarioResult,
    choice_data: ChoiceSetData,
    tol: float = 1e-6,
) -> dict[str, bool]:
    """
    Check scenario invariants and reconciliation.
    
    Returns dict of checks with boolean pass/fail.
    """
    checks = {
        "baseline_shares_sum_to_one": True,
        "scenario_shares_sum_to_one": True,
        "scenario_units_match_totals": True,
        "delta_consistency": True,
        "all_passed": True,
        "details": [],
    }
    
    # Baseline shares sum to 1 per market
    baseline_share_sum = result.baseline_shares.sum(axis=2)
    if not np.allclose(baseline_share_sum, 1.0, atol=tol):
        checks["baseline_shares_sum_to_one"] = False
        checks["details"].append("Baseline shares don't sum to 1")
    
    # Scenario shares sum to 1 per market
    scenario_share_sum = result.scenario_shares.sum(axis=2)
    if not np.allclose(scenario_share_sum, 1.0, atol=tol):
        checks["scenario_shares_sum_to_one"] = False
        checks["details"].append("Scenario shares don't sum to 1")
    
    # Scenario units match market totals
    scenario_units_sum = result.scenario_units.sum(axis=2)
    if not np.allclose(scenario_units_sum, choice_data.market_total_units, atol=tol):
        checks["scenario_units_match_totals"] = False
        checks["details"].append("Scenario units don't match market totals")
    
    # Delta consistency: sum of deltas should be zero per market per draw
    delta_per_draw_market = (result.scenario_units - result.baseline_units).sum(axis=2)
    if not np.allclose(delta_per_draw_market, 0.0, atol=tol):
        checks["delta_consistency"] = False
        max_delta = np.abs(delta_per_draw_market).max()
        checks["details"].append(f"Max per-market per-draw delta = {max_delta:.4f} (expected 0)")
    
    # Also check total delta across all markets
    total_delta = delta_per_draw_market.sum(axis=1)
    if not np.allclose(total_delta, 0.0, atol=tol):
        checks["delta_consistency"] = False
        max_total_delta = np.abs(total_delta).max()
        checks["details"].append(f"Max total delta across markets = {max_total_delta:.4f} (expected 0)")
    
    checks["all_passed"] = all([
        checks["baseline_shares_sum_to_one"],
        checks["scenario_shares_sum_to_one"],
        checks["scenario_units_match_totals"],
        checks["delta_consistency"],
    ])
    
    return checks


def create_driver_waterfall(
    baseline_df: pd.DataFrame,
    price_only_df: pd.DataFrame,
    distribution_only_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    selected_sku: str | None = None,
) -> pd.DataFrame:
    """
    Create driver decomposition waterfall data from four counterfactual scenarios.
    
    Decomposition:
    - Baseline standard volume
    - Price effect = price_only - baseline
    - ND effect = distribution_only - baseline
    - Interaction = combined - price_only - distribution_only + baseline
    - Scenario standard volume
    
    This follows the standard 2x2 factorial decomposition.
    """
    if selected_sku:
        base = baseline_df[baseline_df["sku"] == selected_sku].copy()
        price = price_only_df[price_only_df["sku"] == selected_sku].copy()
        dist = distribution_only_df[distribution_only_df["sku"] == selected_sku].copy()
        comb = combined_df[combined_df["sku"] == selected_sku].copy()
    else:
        base = baseline_df.copy()
        price = price_only_df.copy()
        dist = distribution_only_df.copy()
        comb = combined_df.copy()
    
    if base.empty or comb.empty:
        return pd.DataFrame()
    
    # Aggregate standard volume
    base_q = base["standard_volume"].sum()
    price_q = price["standard_volume"].sum() if not price.empty else base_q
    dist_q = dist["standard_volume"].sum() if not dist.empty else base_q
    comb_q = comb["standard_volume"].sum()
    
    # Compute components
    price_effect = price_q - base_q
    nd_effect = dist_q - base_q
    interaction = comb_q - price_q - dist_q + base_q
    
    components = pd.DataFrame({
        "component": [
            "Baseline standard volume",
            "Price effect",
            "ND effect",
            "Price × ND interaction",
            "Scenario standard volume",
        ],
        "value": [
            base_q,
            price_effect,
            nd_effect,
            interaction,
            comb_q,
        ],
        "is_delta": [False, True, True, True, False],
    })
    
    return components
