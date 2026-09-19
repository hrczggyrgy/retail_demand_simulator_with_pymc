"""Tests for scenario engine and counterfactuals."""
import numpy as np
import pandas as pd
import pytest

import demo_data
import engine


def _get_test_setup():
    """Shared test setup for scenario tests."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "std": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}
    return raw, df, meta, scales, spline, pc


def test_summarise_scenario_draws():
    """Test scenario core simulation produces valid output."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc2 = {"price_slope_z": np.random.normal(-1.5,0.3,(50,df["sku"].nunique())),
           "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}
    scope = df.copy()
    res = engine._simulate_scenario_core(
        scope, pc2, spline, scales, 0.05, 0.1, "pp", "brand", scope["brand"].iloc[0]
    )
    for c in ["units_p50","revenue_p50"]:
        assert res[c].notna().all(), f"NaN in {c}"
    assert "scenario_nd" in res.columns
    assert res["scenario_nd"].notna().all()
    target_brand = scope["brand"].iloc[0]
    target = res[res["brand"]==target_brand]
    non = res[res["brand"]!=target_brand]
    assert not np.allclose(target["units_p50"]/target["units"], 1.0), "Target brand should change"
    assert np.allclose(non["units_p50"]/non["units"], 1.0), "Non-target should not change"
    print("✓ _simulate_scenario_core: no NaN, target effect only")


def test_calculate_market_shares():
    """Test market share calculation at multiple levels."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    scope = df.copy()
    res = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", scope["brand"].iloc[0]
    )
    for c in res.columns:
        scope[c] = res[c].to_numpy()
    for level in ["brand", "category", "retailer", "sku"]:
        sh = engine.calculate_market_shares(scope, scope, level)
        assert "baseline_share" in sh.columns
        assert "scenario_share" in sh.columns
        for period, g in sh.groupby("month"):
            baseline_sum = g["baseline_share"].sum()
            scenario_sum = g["scenario_share"].sum()
            assert np.isclose(baseline_sum, 1.0, atol=1e-3), f"baseline shares sum != 1 at {period}"
            assert np.isclose(scenario_sum, 1.0, atol=1e-3), f"scenario shares sum != 1 at {period}"
    print("✓ calculate_market_shares sums to 1 at all levels")


def test_scenario_suite():
    """Test full scenario suite runs correctly."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    scenarios = engine.run_scenario_suite(
        df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0]
    )
    assert set(scenarios.keys()) == {"baseline","price_only","distribution_only","combined"}
    for k, v in scenarios.items():
        assert v["units_p50"].notna().all(), f"{k} has NaN units_p50"
        assert v["revenue_p50"].notna().all(), f"{k} has NaN revenue_p50"
    b = scenarios["baseline"]
    p = scenarios["price_only"]
    d = scenarios["distribution_only"]
    assert not np.allclose(b["units_p50"], p["units_p50"])
    assert not np.allclose(b["units_p50"], d["units_p50"])
    print("✓ run_scenario_suite: 4 scenarios, no NaN, effects differ")


def test_aggregate_market_impact():
    """Test market impact aggregation at all levels."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    scenarios = engine.run_scenario_suite(
        df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0]
    )
    for level in ["market", "retailer", "brand", "sku"]:
        agg = engine.aggregate_market_impact(scenarios["baseline"], scenarios["combined"], level)
        assert "baseline_revenue" in agg.columns
        assert "scenario_revenue" in agg.columns
        assert "delta_revenue_pct" in agg.columns
    print("✓ aggregate_market_impact works at all levels")


def test_reallocation_breakdown():
    """Test reallocation breakdown has key segments."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    scenarios = engine.run_scenario_suite(
        df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0]
    )
    target_sku = df["sku"].iloc[0]
    target_row = df[df["sku"]==target_sku].iloc[0]
    rb = engine.compute_reallocation_breakdown(
        scenarios["baseline"], scenarios["combined"],
        target_sku, target_row["brand"], target_row["category"],
        target_row["pack_group"], target_row["retailer"]
    )
    assert "segment_relationship" in rb.columns
    assert "share_of_total_delta" in rb.columns
    relationships = set(rb["segment_relationship"].unique())
    assert "selected_sku" in relationships
    assert "same_brand_other_pack" in relationships
    assert "other_category" in relationships
    print("✓ compute_reallocation_breakdown: key segments present")


def test_parameter_attribution():
    """Test parameter attribution has price and ND components."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    _ = engine.run_scenario_suite(
        df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0]
    )
    target_sku = df["sku"].iloc[0]
    target_row = df[df["sku"] == target_sku].iloc[0]
    sku_idx = target_row["sku_idx"]
    nd_base = target_row["nd"]
    pa = engine.compute_parameter_attribution(
        pc, spline, scales, sku_idx, 0.05, nd_base, nd_base + 0.1, target_sku, meta
    )
    assert "component" in pa.columns
    assert "multiplier_median" in pa.columns
    components = set(pa["component"].unique())
    assert "Selected-SKU price effect" in components
    assert "Numeric distribution effect" in components
    print("✓ compute_parameter_attribution: price + nd components present")


def test_zero_price_and_nd_scenario_equals_baseline() -> None:
    """Zero price and zero ND change should produce baseline-equivalent output."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    scope = df.copy()
    res = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.0, 0.0, "pp", "brand", scope["brand"].iloc[0]
    )
    np.testing.assert_allclose(
        res["units_p50"].to_numpy(), scope["units"].to_numpy(), rtol=1e-10
    )
    np.testing.assert_allclose(
        res["revenue_p50"].to_numpy(), scope["revenue"].to_numpy(), rtol=1e-10
    )
    print("✓ Zero price/ND scenario equals baseline")


def test_target_scope_case_insensitive_or_raises() -> None:
    """Target scope matching should be case-insensitive or raise clear error."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    scope = df.copy()
    target_brand = scope["brand"].iloc[0]

    res1 = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", target_brand
    )
    assert res1["units_p50"].notna().all()

    res2 = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "Brand", target_brand
    )
    assert res2["units_p50"].notna().all()

    with pytest.raises(ValueError, match="Unknown target_level"):
        engine._simulate_scenario_core(
            scope, pc, spline, scales, 0.05, 0.1, "pp", "invalid_scope", target_brand
        )
    print("✓ Target scope case-insensitive and raises on invalid")


def test_filtered_dataframe_non_contiguous_index_no_nans() -> None:
    """Filtered dataframe with non-contiguous index should not create NaNs."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    scope = df.iloc[::3].copy()
    res = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", scope["brand"].iloc[0]
    )
    assert res["units_p50"].notna().all()
    assert res["revenue_p50"].notna().all()
    print("✓ Non-contiguous index produces no NaNs")


def test_scenario_runs_on_full_market_not_filtered() -> None:
    """Scenario should run on full uploaded market, not display-filtered rows."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    filtered = df[df["category"] == "CSD"].copy()
    full_res = engine._simulate_scenario_core(
        df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0]
    )
    filtered_res = engine._simulate_scenario_core(
        filtered, pc, spline, scales, 0.05, 0.1, "pp", "brand", filtered["brand"].iloc[0]
    )
    assert len(full_res) > len(filtered_res)
    assert len(full_res) == len(df)
    print("✓ Scenario runs on full market (not filtered)")


def test_scenario_segment_deltas_reconcile_to_market_delta() -> None:
    """Segment deltas should sum to market-level delta (reconciliation)."""
    raw, df, meta, scales, spline, pc = _get_test_setup()
    scenarios = engine.run_scenario_suite(
        df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0]
    )
    target_sku = df["sku"].iloc[0]
    target_row = df[df["sku"] == target_sku].iloc[0]

    rb = engine.compute_reallocation_breakdown(
        scenarios["baseline"], scenarios["combined"],
        target_sku, target_row["brand"], target_row["category"],
        target_row["pack_group"], target_row["retailer"]
    )

    total_segment_delta = rb["delta_units"].sum()
    market_baseline = scenarios["baseline"]["units_p50"].sum()
    market_scenario = scenarios["combined"]["units_p50"].sum()
    market_delta = market_scenario - market_baseline

    np.testing.assert_allclose(total_segment_delta, market_delta, rtol=1e-5)
    print("✓ Scenario segment deltas reconcile to market delta")


# ============================================================================
# Phase 4: Draw-by-draw joint scenario tests
# ============================================================================

def test_joint_scenario_runs() -> None:
    """Test that joint scenario runs and produces results."""
    raw = demo_data.generate_demo_data()
    df, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    enriched = engine.build_retail_features(indexed)

    choice_data = engine.build_choice_set_data(enriched, meta)

    fast_config = engine.JointModelConfig(
        draws=30, tune=30, chains=1, target_accept=0.9,
        use_category_price_pooling=True, use_sku_nd_effects=True,
    )
    model = engine.build_joint_sku_share_model(choice_data, fast_config)
    idata = engine.fit_joint_model(model, fast_config)

    cache = engine.extract_joint_posterior(idata, max_draws=10)

    actions = [
        engine.ScenarioAction(
            retailer="Apex Supermarkets",
            category="CSD",
            brand="Fizz Up",
            sku="FIZZ_500ML",
            month="2024-07-01",
            new_nd=0.1,
            nd_mode="pp",
        )
    ]

    result = engine.run_joint_scenario_draws(choice_data, cache, actions, n_draws=5, random_seed=42)

    assert isinstance(result, engine.ScenarioResult)
    assert result.baseline_units_p50.shape == (len(choice_data.market_ids), 12)
    assert result.scenario_units_p50.shape == (len(choice_data.market_ids), 12)
    assert result.delta_units_p50.shape == (len(choice_data.market_ids), 12)
    print("✓ Joint scenario runs and produces results")


def test_joint_scenario_reconciliation() -> None:
    """Test scenario reconciliation for joint model."""
    raw = demo_data.generate_demo_data()
    df, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    enriched = engine.build_retail_features(indexed)

    choice_data = engine.build_choice_set_data(enriched, meta)

    fast_config = engine.JointModelConfig(
        draws=30, tune=30, chains=1, target_accept=0.9,
        use_category_price_pooling=True, use_sku_nd_effects=True,
    )
    model = engine.build_joint_sku_share_model(choice_data, fast_config)
    idata = engine.fit_joint_model(model, fast_config)

    cache = engine.extract_joint_posterior(idata, max_draws=10)

    actions = [
        engine.ScenarioAction(
            retailer="Apex Supermarkets",
            category="CSD",
            brand="Fizz Up",
            sku="FIZZ_500ML",
            month="2024-07-01",
            new_nd=0.1,
            nd_mode="pp",
        )
    ]

    result = engine.run_joint_scenario_draws(choice_data, cache, actions, n_draws=5, random_seed=42)

    recon = engine.check_scenario_reconciliation(result, choice_data)

    assert recon["baseline_shares_sum_to_one"] is True
    assert recon["scenario_shares_sum_to_one"] is True
    assert recon["baseline_units_match_totals"] is True
    assert recon["scenario_units_match_totals"] is True
    print("✓ Joint scenario reconciliation checks pass")


def test_scenario_aggregation() -> None:
    """Test scenario aggregation at multiple levels."""
    raw = demo_data.generate_demo_data()
    df, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    enriched = engine.build_retail_features(indexed)

    choice_data = engine.build_choice_set_data(enriched, meta)

    fast_config = engine.JointModelConfig(
        draws=30, tune=30, chains=1, target_accept=0.9,
        use_category_price_pooling=True, use_sku_nd_effects=True,
    )
    model = engine.build_joint_sku_share_model(choice_data, fast_config)
    idata = engine.fit_joint_model(model, fast_config)

    cache = engine.extract_joint_posterior(idata, max_draws=10)

    actions = [
        engine.ScenarioAction(
            retailer="Apex Supermarkets",
            category="CSD",
            brand="Fizz Up",
            sku="FIZZ_500ML",
            month="2024-07-01",
            new_nd=0.1,
            nd_mode="pp",
        )
    ]

    result = engine.run_joint_scenario_draws(choice_data, cache, actions, n_draws=5, random_seed=42)

    for level in ["market", "retailer", "category", "brand", "sku", "brand_pack", "retailer_category"]:
        agg = engine.aggregate_scenario_result(result, choice_data, level=level)
        assert "baseline_units" in agg.columns
        assert "scenario_units" in agg.columns
        assert "delta_units" in agg.columns
        assert "delta_units_pct" in agg.columns
        observed_total = choice_data.observed_units.sum()
        rel_error_baseline = abs(agg["baseline_units"].sum() - observed_total) / observed_total
        rel_error_scenario = abs(agg["scenario_units"].sum() - observed_total) / observed_total
        assert rel_error_baseline < 0.01
        assert rel_error_scenario < 0.01
    print("✓ Scenario aggregation works at all levels")


if __name__ == "__main__":
    test_summarise_scenario_draws()
    test_calculate_market_shares()
    test_scenario_suite()
    test_aggregate_market_impact()
    test_reallocation_breakdown()
    test_parameter_attribution()
    test_zero_price_and_nd_scenario_equals_baseline()
    test_target_scope_case_insensitive_or_raises()
    test_filtered_dataframe_non_contiguous_index_no_nans()
    test_scenario_runs_on_full_market_not_filtered()
    test_scenario_segment_deltas_reconcile_to_market_delta()
    test_joint_scenario_runs()
    test_joint_scenario_reconciliation()
    test_scenario_aggregation()
    print("\n=== ALL SCENARIO TESTS PASSED ===")
