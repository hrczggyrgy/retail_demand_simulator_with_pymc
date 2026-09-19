"""Tests for feature engineering and retail features."""
import numpy as np
import pandas as pd
import pytest

import demo_data
import engine


def test_demo_data_schema():
    """Test that demo data has the correct schema."""
    raw = demo_data.generate_demo_data()
    assert list(raw.columns) == engine.REQUIRED_COLUMNS, f"Schema mismatch: {list(raw.columns)}"
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
    """Test data preparation produces required columns."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    # Required columns after prepare
    for c in ["month","retailer","category","brand","sku","pack_size","pack_group",
              "units","revenue","sku_stores","retailer_stores","nd",
              "log_nd","log_relative_price","log_velocity_std",
              "log_nd_z","log_relative_price_z","log_velocity_std_z"]:
        assert c in df.columns, f"Missing: {c}"
    assert len(df) == 1152
    # sku_idx added by app after prepare
    df["sku_idx"] = df["sku"].map({s:i for i,s in enumerate(sorted(df["sku"].unique()))})
    assert df["sku_idx"].max() == df["sku"].nunique() - 1 == 11
    assert df["pack_group"].nunique() <= 3  # tertiles per category
    print("✓ prepare_data columns & indices")


def test_build_retail_features():
    """Test retail feature engineering produces core columns."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    enriched = engine.build_retail_features(df)
    # Core derived columns
    for c in ["unit_price", "price_std", "nd", "standard_volume", "velocity_std",
              "brand_share", "pack_group", "month_num",
              "log_nd", "log_price_std", "log_velocity_std", "log_relative_price",
              "log_nd_z", "log_price_std_z", "log_velocity_std_z", "log_relative_price_z",
              "data_quality_status"]:
        assert c in enriched.columns, f"Missing: {c}"
    # Data quality status should be valid for demo data
    assert (enriched["data_quality_status"] == "valid").all()
    print("✓ build_retail_features core columns present")


def test_brand_share_reconciles_within_category() -> None:
    """Test brand shares sum to 1 within each category."""
    df = engine.build_retail_features(demo_data.get_minimal_test_fixture())
    brand_level = df.drop_duplicates(subset=["month", "retailer", "category", "brand"])
    total = brand_level.groupby(
        ["month", "retailer", "category"], observed=True
    )["brand_share"].sum()
    np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-10)


def test_numeric_distribution_is_bounded() -> None:
    """Test ND is bounded between 0 and 1."""
    df = engine.build_retail_features(demo_data.get_minimal_test_fixture())
    valid = df.loc[df["data_quality_status"].eq("valid")]
    assert valid["nd"].between(0.0, 1.0).all()


def test_demo_schema_contains_only_raw_columns() -> None:
    """Test demo data fixture contains only raw columns."""
    df = demo_data.get_minimal_test_fixture()
    assert list(df.columns) == list(demo_data.RAW_COLUMNS)


def test_same_brand_other_sku_price_index_excludes_focal_sku() -> None:
    """same_brand_other_sku_price_index must exclude the focal SKU itself."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    enriched = engine.build_retail_features(df)

    # Pick a brand with multiple SKUs
    brand_skus = df.groupby("brand")["sku"].nunique()
    multi_sku_brand = brand_skus[brand_skus > 1].index[0]
    brand_rows = enriched[enriched["brand"] == multi_sku_brand]

    group_cols = ["month", "retailer", "category", "brand"]
    for _, group in brand_rows.groupby(group_cols, observed=True):
        for sku in group["sku"].unique():
            sku_rows = group[group["sku"] == sku]
            other_skus = group[group["sku"] != sku]
            if len(other_skus) > 0:
                expected_idx = (
                    other_skus["price_std"] * other_skus["standard_volume"]
                ).sum() / other_skus["standard_volume"].sum()
                actual_idx = sku_rows["same_brand_other_sku_price_index"].iloc[0]
                np.testing.assert_allclose(actual_idx, expected_idx, rtol=1e-10)
    print("✓ same_brand_other_sku_price_index excludes focal SKU")


def test_one_sku_brand_produces_nan_for_same_brand_index() -> None:
    """A one-SKU brand should produce NaN for same-brand-other-SKU price index."""
    df = demo_data.get_minimal_test_fixture()
    enriched = engine.build_retail_features(df)

    b_rows = enriched[enriched["brand"] == "B"]
    assert len(b_rows) == 1
    assert b_rows["same_brand_other_sku_price_index"].isna().all()
    print("✓ One-SKU brand produces NaN for same-brand index")


if __name__ == "__main__":
    test_demo_data_schema()
    test_prepare_data()
    test_build_retail_features()
    test_brand_share_reconciles_within_category()
    test_numeric_distribution_is_bounded()
    test_demo_schema_contains_only_raw_columns()
    test_same_brand_other_sku_price_index_excludes_focal_sku()
    test_one_sku_brand_produces_nan_for_same_brand_index()
    print("\n=== ALL FEATURE TESTS PASSED ===")
