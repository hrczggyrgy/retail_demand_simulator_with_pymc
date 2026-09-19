"""Tests for reporting and analytics helpers."""
import numpy as np
import pandas as pd
import pytest

import demo_data
import engine


def test_analytics_helpers():
    """Test all analytics helpers execute without error."""
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
    _ = engine.create_growth_decomposition(df, start_month, end_month)
    _ = engine.classify_distribution_velocity(df)
    _ = engine.get_expansion_candidates(df)
    print("✓ All analytics helpers execute")


def test_decompose_sales_growth():
    """Test sales growth decomposition."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    months = sorted(df["month"].unique())
    start_month, end_month = str(months[0]), str(months[-1])
    result = engine.decompose_sales_growth(df, start_month, end_month)
    assert "stores_contrib" in result.columns
    assert "nd_contrib" in result.columns
    assert "vel_contrib" in result.columns
    assert "volume_change" in result.columns
    print("✓ Sales growth decomposition works")


def test_prepare_market():
    """Test market preparation for reporting."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    result = engine.prepare_market(df)
    assert "df" in result
    assert "scales" in result
    assert "meta" in result
    print("✓ prepare_market works")


def test_build_market_overview():
    """Test market overview building."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    overview = engine.build_market_overview(df)
    assert isinstance(overview, dict)
    assert "total_revenue" in overview
    assert "total_units" in overview
    print("✓ build_market_overview works")


def test_build_price_pack_architecture():
    """Test price pack architecture."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    enriched = engine.build_retail_features(df)
    result = engine.build_price_pack_architecture(enriched)
    assert isinstance(result, pd.DataFrame)
    assert "pack_group" in result.columns
    print("✓ build_price_pack_architecture works")


def test_build_nd_velocity_quadrant():
    """Test ND velocity quadrant."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    enriched = engine.build_retail_features(df)
    result = engine.build_nd_velocity_quadrant(enriched)
    assert isinstance(result, pd.DataFrame)
    assert "quadrant" in result.columns
    assert "avg_velocity" in result.columns
    print("✓ build_nd_velocity_quadrant works")


def test_calculate_shares():
    """Test share calculation."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    result = engine.calculate_shares(df)
    assert "brand_share" in result
    assert "sku_share" in result
    print("✓ calculate_shares works")


if __name__ == "__main__":
    test_analytics_helpers()
    test_decompose_sales_growth()
    test_prepare_market()
    test_build_market_overview()
    test_build_price_pack_architecture()
    test_build_nd_velocity_quadrant()
    test_calculate_shares()
    print("\n=== ALL REPORTING TESTS PASSED ===")
