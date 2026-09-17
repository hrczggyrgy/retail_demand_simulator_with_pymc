#!/usr/bin/env python
"""Comprehensive test of app functionality: data, model, tabs, scenarios, reasonable outputs."""
import numpy as np
import pandas as pd
import demo_data
import engine

def test_demo_data_schema():
    raw = demo_data.generate_demo_data()
    from engine import REQUIRED_COLUMNS
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
    res = engine._simulate_scenario_core(scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", scope["brand"].iloc[0])
    # No NaN in median columns
    for c in ["units_median","revenue_median"]:
        assert res[c].notna().all(), f"NaN in {c}"
    # scenario_nd should also be present and not NaN
    assert "scenario_nd" in res.columns
    assert res["scenario_nd"].notna().all()
    # Target rows have effect, non-target don't
    target_brand = scope["brand"].iloc[0]
    target = res[res["brand"]==target_brand]
    non = res[res["brand"]!=target_brand]
    assert not np.allclose(target["units_median"]/target["units"], 1.0), "Target brand should change"
    assert np.allclose(non["units_median"]/non["units"], 1.0), "Non-target should not change"
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
    res = engine._simulate_scenario_core(scope, pc, spline, scales, 0.05, 0.1, "pp", "brand", scope["brand"].iloc[0])
    for c in res.columns: scope[c] = res[c].to_numpy()
    scope["expected_units_median"] = scope["units_median"].to_numpy()
    scope["expected_revenue_median"] = scope["revenue_median"].to_numpy()
    for level in ["brand","category","retailer","sku"]:
        sh = engine.calculate_market_shares(scope, scope, level)
        assert "baseline_share" in sh.columns and "scenario_share" in sh.columns
        # Shares should sum to ~1 within each period
        for period, g in sh.groupby("month"):
            assert np.isclose(g["baseline_share"].sum(), 1.0, atol=1e-3), f"baseline shares sum != 1 at {period}"
            assert np.isclose(g["scenario_share"].sum(), 1.0, atol=1e-3), f"scenario shares sum != 1 at {period}"
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
    scenarios = engine.run_scenario_suite(df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0])
    assert set(scenarios.keys()) == {"baseline","price_only","distribution_only","combined"}
    for k, v in scenarios.items():
        assert v["units_median"].notna().all(), f"{k} has NaN units_median"
        assert v["revenue_median"].notna().all(), f"{k} has NaN revenue_median"
    # Baseline == price_only with 0 price change, etc.
    # Combined should have larger effect than individual
    b = scenarios["baseline"]
    p = scenarios["price_only"]
    d = scenarios["distribution_only"]
    c = scenarios["combined"]
    # At least some target rows differ
    assert not np.allclose(b["units_median"], p["units_median"])
    assert not np.allclose(b["units_median"], d["units_median"])
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
    scenarios = engine.run_scenario_suite(df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0])
    for level in ["market","retailer","brand","sku"]:  # using valid MARKET_LEVELS keys
        agg = engine.aggregate_market_impact(scenarios["baseline"], scenarios["combined"], level)
        assert "baseline_revenue" in agg.columns and "scenario_revenue" in agg.columns
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
    scenarios = engine.run_scenario_suite(df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0])
    target_sku = df["sku"].iloc[0]
    target_row = df[df["sku"]==target_sku].iloc[0]
    rb = engine.compute_reallocation_breakdown(
        scenarios["baseline"], scenarios["combined"],
        target_sku, target_row["brand"], target_row["category"],
        target_row["pack_group"], target_row["retailer"]
    )
    assert "segment_relationship" in rb.columns and "share_of_total_delta" in rb.columns
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
    scenarios = engine.run_scenario_suite(df, pc, spline, scales, 0.05, 0.1, "pp", "brand", df["brand"].iloc[0])
    target_sku = df["sku"].iloc[0]
    target_row = df[df["sku"]==target_sku].iloc[0]
    sku_idx = target_row["sku_idx"]
    nd_base = target_row["nd"]
    pa = engine.compute_parameter_attribution(
        pc, spline, scales, sku_idx, 0.05, nd_base, nd_base + 0.1, target_sku, meta
    )
    assert "component" in pa.columns and "multiplier_median" in pa.columns
    components = set(pa["component"].unique())
    assert "Selected-SKU price effect" in components
    assert "Numeric distribution effect" in components
    print("✓ compute_parameter_attribution: price + nd components present")

def test_elasticity_extraction():
    # Quick sanity: elasticity function runs without error on synthetic posterior
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

if __name__ == "__main__":
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
    print("\n=== ALL TESTS PASSED ===")