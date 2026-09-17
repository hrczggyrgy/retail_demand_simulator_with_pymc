#!/usr/bin/env python
"""Comprehensive test of app functionality: data, model, tabs, scenarios, reasonable outputs."""
import numpy as np
import pytest

import demo_data
import engine
from engine import REQUIRED_COLUMNS

# ============================================================
# Issue 2: Deterministic fixtures and engine tests
# ============================================================

def test_demo_schema_contains_only_raw_columns() -> None:
    df = demo_data.get_minimal_test_fixture()
    assert list(df.columns) == demo_data.RAW_COLUMNS


def test_numeric_distribution_is_bounded() -> None:
    df = engine.build_retail_features(demo_data.get_minimal_test_fixture())
    valid = df.loc[df["data_quality_status"].eq("valid")]
    assert valid["nd"].between(0.0, 1.0).all()


def test_brand_share_reconciles_within_category() -> None:
    df = engine.build_retail_features(demo_data.get_minimal_test_fixture())
    # brand_share is per-SKU; deduplicate by brand to sum correctly
    brand_level = df.drop_duplicates(subset=["month", "retailer", "category", "brand"])
    total = brand_level.groupby(
        ["month", "retailer", "category"], observed=True
    )["brand_share"].sum()
    np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-10)


# ============================================================
# Original comprehensive tests
# ============================================================

def test_demo_data_schema():
    raw = demo_data.generate_demo_data()
    assert list(raw.columns) == REQUIRED_COLUMNS, f"Schema mismatch: {list(raw.columns)}"
    assert len(raw) == 1152, f"Expected 1152 rows, got {len(raw)}"
    assert raw["month"].nunique() == 24
    assert raw["retailer"].nunique() == 4
    assert raw["category"].nunique() == 3
    assert raw["brand"].nunique() == 6
    assert raw["sku"].nunique() == 12
    assert (raw["units"] >= 0).all()
    assert (raw["revenue"] >= 0).all()
    assert (raw["sku_stores"] > 0).all()
    assert (raw["retailer_stores"] > 0).all()
    print("✓ demo_data schema & basic checks")

def test_prepare_data():
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    # Required columns after prepare
    for c in ["month","retailer","category","brand","sku","pack_size","pack_group",
              "units","revenue","sku_stores","retailer_stores","nd",
              "log_nd","log_relative_price","log_velocity",
              "log_nd_z","log_relative_price_z","log_velocity_z"]:
        assert c in df.columns, f"Missing: {c}"
    assert len(df) == 1152
    # sku_idx added by app after prepare
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    assert df["sku_idx"].max() == df["sku"].nunique() - 1 == 11
    assert df["pack_group"].nunique() <= 3  # tertiles per category
    print("✓ prepare_data columns & indices")

def test_build_retail_features():
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    enriched = engine.build_retail_features(df)
    for c in ["other_skus_price_std","same_brand_other_sku_price_index",
              "other_brand_price_index","unit_price","price_per_size_unit"]:
        assert c in enriched.columns, f"Missing: {c}"
    assert enriched["same_brand_other_sku_price_index"].notna().all()
    assert enriched["other_brand_price_index"].notna().all()
    print("✓ build_retail_features indices complete (no NaN)")

def test_model_config_presets():
    for name, cfg in [("FAST", engine.FAST_CONFIG), ("DEFAULT", engine.DEFAULT_CONFIG),
                       ("ADVANCED", engine.ADVANCED_CONFIG)]:
        assert hasattr(cfg, "chains"), f"{name} missing chains"
        assert hasattr(cfg, "scenario_draws"), f"{name} missing scenario_draws"
        assert cfg.chains >= 1
    print("✓ ModelConfig presets exist")

def test_fit_spline():
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    spline, _ = engine.fit_spline(df)
    assert hasattr(spline, "transform")
    out = spline.transform(np.array([[0.5]]))
    assert out.shape[1] >= 3  # at least 3 basis functions
    print("✓ fit_spline produces basis")

def test_summarise_scenario_draws():
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}
    scope = df.copy()
    res = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", scope["brand"].iloc[0]
    )
    # No NaN in median columns
    for c in ["units_p50","revenue_p50"]:
        assert res[c].notna().all(), f"NaN in {c}"
    # scenario_nd should also be present and not NaN
    assert "scenario_nd" in res.columns
    assert res["scenario_nd"].notna().all()
    # Target rows have effect, non-target don't
    target_brand = scope["brand"].iloc[0]
    target = res[res["brand"]==target_brand]
    non = res[res["brand"]!=target_brand]
    assert not np.allclose(target["units_p50"]/target["units"], 1.0), "Target brand should change"
    assert np.allclose(non["units_p50"]/non["units"], 1.0), "Non-target should not change"
    print("✓ _simulate_scenario_core: no NaN, target effect only")

def test_calculate_market_shares():
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}
    scope = df.copy()
    res = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", scope["brand"].iloc[0]
    )
    for c in res.columns:
        scope[c] = res[c].to_numpy()
    scope["expected_units_median"] = scope["units_p50"].to_numpy()
    scope["expected_revenue_median"] = scope["revenue_p50"].to_numpy()
    for level in ["brand", "category", "retailer", "sku"]:
        sh = engine.calculate_market_shares(scope, scope, level)
        assert "baseline_share" in sh.columns
        assert "scenario_share" in sh.columns
        # Shares should sum to ~1 within each period
        for period, g in sh.groupby("month"):
            baseline_sum = g["baseline_share"].sum()
            scenario_sum = g["scenario_share"].sum()
            assert np.isclose(baseline_sum, 1.0, atol=1e-3), (
                f"baseline shares sum != 1 at {period}"
            )
            assert np.isclose(scenario_sum, 1.0, atol=1e-3), (
                f"scenario shares sum != 1 at {period}"
            )
    print("✓ calculate_market_shares sums to 1 at all levels")

def test_scenario_suite():
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}
    scenarios = engine.run_scenario_suite(
        df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0]
    )
    assert set(scenarios.keys()) == {"baseline","price_only","distribution_only","combined"}
    for k, v in scenarios.items():
        assert v["units_p50"].notna().all(), f"{k} has NaN units_p50"
        assert v["revenue_p50"].notna().all(), f"{k} has NaN revenue_p50"
    # Baseline == price_only with 0 price change, etc.
    # Combined should have larger effect than individual
    b = scenarios["baseline"]
    p = scenarios["price_only"]
    d = scenarios["distribution_only"]
    # At least some target rows differ
    assert not np.allclose(b["units_p50"], p["units_p50"])
    assert not np.allclose(b["units_p50"], d["units_p50"])
    print("✓ run_scenario_suite: 4 scenarios, no NaN, effects differ")

def test_aggregate_market_impact():
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}
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
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}
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
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}
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

def test_elasticity_extraction():
    # Quick sanity: elasticity function runs without error on synthetic posterior
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    spline, _ = engine.fit_spline(df)
    np.random.seed(42)
    # extract_elasticities expects arviz InferenceData - skip full test, just verify it's callable
    assert callable(engine.extract_elasticities)
    print("✓ extract_elasticities callable")

def test_analytics_helpers():
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    months = sorted(df["month"].unique())
    start_month, end_month = str(months[0]), str(months[-1])
    # These should run without error
    _ = engine.create_market_summary(df)
    _ = engine.create_category_summary(df)
    _ = engine.create_retailer_summary(df)
    _ = engine.create_brand_summary(df)
    _ = engine.create_sku_summary(df)
    _ = engine.compute_price_architecture(df)
    _ = engine.compute_distribution_opportunity(df)
    _ = engine.compute_market_share_analytics(df)
    _ = engine.compute_contribution_to_growth(df, start_month, end_month)
    # compute_model_validation_metrics needs idata - skip
    _ = engine.create_growth_decomposition(df)
    _ = engine.classify_distribution_velocity(df)
    _ = engine.get_expansion_candidates(df)
    print("✓ All analytics helpers execute")


# ============================================================
# Issue 2: Additional engine contract tests
# ============================================================

def test_zero_price_and_nd_scenario_equals_baseline() -> None:
    """Zero price and zero ND change should produce baseline-equivalent output."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}

    # Zero change scenario
    scope = df.copy()
    res = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.0, 0.0, "pp", "brand", scope["brand"].iloc[0]
    )

    # Units should equal baseline (within floating point)
    np.testing.assert_allclose(
        res["units_p50"].to_numpy(), scope["units"].to_numpy(), rtol=1e-10
    )
    np.testing.assert_allclose(
        res["revenue_p50"].to_numpy(), scope["revenue"].to_numpy(), rtol=1e-10
    )
    print("✓ Zero price/ND scenario equals baseline")


def test_target_scope_case_insensitive_or_raises() -> None:
    """Target scope matching should be case-insensitive or raise clear error for invalid scope."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}

    scope = df.copy()
    target_brand = scope["brand"].iloc[0]

# Should work with lowercase
    res1 = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", target_brand
    )
    assert res1["units_p50"].notna().all()

    # Should work with mixed case
    res2 = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "Brand", target_brand
    )
    assert res2["units_p50"].notna().all()

    # Invalid scope should raise ValueError
    with pytest.raises(ValueError, match="Unknown target_level"):
        engine._simulate_scenario_core(
            scope, pc, spline, scales, 0.05, 0.1, "pp", "invalid_scope", target_brand
        )
    print("✓ Target scope case-insensitive and raises on invalid")


def test_filtered_dataframe_non_contiguous_index_no_nans() -> None:
    """Filtered dataframe with non-contiguous index should not create NaNs in scenario output."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}

    # Create non-contiguous index (e.g., every 3rd row)
    scope = df.iloc[::3].copy()
    res = engine._simulate_scenario_core(
        scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", scope["brand"].iloc[0]
    )

    assert res["units_p50"].notna().all()
    assert res["revenue_p50"].notna().all()
    print("✓ Non-contiguous index produces no NaNs")


def test_scenario_runs_on_full_market_not_filtered() -> None:
    """Scenario should run on full uploaded market, not display-filtered rows."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}

    # Simulate a filtered view (e.g., one category)
    filtered = df[df["category"] == "CSD"].copy()

    # Run scenario on FULL market
    full_res = engine._simulate_scenario_core(
        df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0]
    )

    # Run scenario on filtered view (this would be wrong behavior)
    filtered_res = engine._simulate_scenario_core(
        filtered, pc, spline, scales, 0.05, 0.1, "pp", "brand", filtered["brand"].iloc[0]
    )

    # Full market should have more rows
    assert len(full_res) > len(filtered_res)
    assert len(full_res) == len(df)
    print("✓ Scenario runs on full market (not filtered)")


def test_same_brand_other_sku_price_index_excludes_focal_sku() -> None:
    """same_brand_other_sku_price_index must exclude the focal SKU itself."""
    # Create a brand with multiple SKUs
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    enriched = engine.build_retail_features(df)

    # For a given brand, the same_brand_other_sku_price_index should not include the SKU's own price
    # Pick a brand with multiple SKUs
    brand_skus = df.groupby("brand")["sku"].nunique()
    multi_sku_brand = brand_skus[brand_skus > 1].index[0]
    brand_rows = enriched[enriched["brand"] == multi_sku_brand]

    # The index is computed per month/retailer/category/brand group
    group_cols = ["month", "retailer", "category", "brand"]
    for _, group in brand_rows.groupby(group_cols, observed=True):
        for sku in group["sku"].unique():
            sku_rows = group[group["sku"] == sku]
            # The index for this SKU should be the volume-weighted
            # avg price of OTHER SKUs in the same brand within this group
            other_skus = group[group["sku"] != sku]
            if len(other_skus) > 0:
                # Volume-weighted avg price = sum(price * units) / sum(units) for other SKUs
                expected_idx = (
                    other_skus["unit_price"] * other_skus["units"]
                ).sum() / other_skus["units"].sum()
                actual_idx = sku_rows["same_brand_other_sku_price_index"].iloc[0]
                np.testing.assert_allclose(actual_idx, expected_idx, rtol=1e-10)
    print("✓ same_brand_other_sku_price_index excludes focal SKU")


def test_one_sku_brand_produces_nan_for_same_brand_index() -> None:
    """A one-SKU brand should produce NaN (not infinity or zero)
    for same-brand-other-SKU price index."""
    # Create minimal fixture with single-SKU brand
    df = demo_data.get_minimal_test_fixture()
    enriched = engine.build_retail_features(df)

    # Brand B has only one SKU (B_500)
    b_rows = enriched[enriched["brand"] == "B"]
    assert len(b_rows) == 1
    assert b_rows["same_brand_other_sku_price_index"].isna().all()
    print("✓ One-SKU brand produces NaN for same-brand index")


def test_scenario_segment_deltas_reconcile_to_market_delta() -> None:
    """Segment deltas should sum to market-level delta (reconciliation)."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    n_skus = df["sku"].nunique()
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    nd = df["nd"].clip(1e-4).to_numpy()
    scales = {"log_nd": {"mean": float(np.log(nd).mean()), "sd": float(np.log(nd).std())}}
    spline, _ = engine.fit_spline(df)
    n_basis = int(spline.transform(np.array([[0.0]])).shape[1])
    np.random.seed(42)
    pc = {"price_slope_z": np.random.normal(-1.5,0.3,(50,n_skus)),
          "nd_coef": np.random.normal(0,0.5,(50,n_basis)), "nd_coef_type":"shared"}

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

    # Sum of segment delta_units should equal market delta
    total_segment_delta = rb["delta_units"].sum()
    market_baseline = scenarios["baseline"]["units_p50"].sum()
    market_scenario = scenarios["combined"]["units_p50"].sum()
    market_delta = market_scenario - market_baseline

    np.testing.assert_allclose(total_segment_delta, market_delta, rtol=1e-5)
    print("✓ Scenario segment deltas reconcile to market delta")


if __name__ == "__main__":
    # Issue 2 deterministic fixture tests
    test_demo_schema_contains_only_raw_columns()
    test_numeric_distribution_is_bounded()
    test_brand_share_reconciles_within_category()

    # Original comprehensive tests
    test_demo_data_schema()
    test_prepare_data()
    test_build_retail_features()
    test_model_config_presets()
    test_fit_spline()
    test_summarise_scenario_draws()
    test_calculate_market_shares()
    test_scenario_suite()
    test_aggregate_market_impact()
    test_reallocation_breakdown()
    test_parameter_attribution()
    test_elasticity_extraction()
    test_analytics_helpers()

    # Issue 2 additional engine contract tests
    test_zero_price_and_nd_scenario_equals_baseline()
    test_target_scope_case_insensitive_or_raises()
    test_filtered_dataframe_non_contiguous_index_no_nans()
    test_scenario_runs_on_full_market_not_filtered()
    test_same_brand_other_sku_price_index_excludes_focal_sku()
    test_one_sku_brand_produces_nan_for_same_brand_index()
    test_scenario_segment_deltas_reconcile_to_market_delta()

    print("\n=== ALL TESTS PASSED ===")
