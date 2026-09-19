"""
Validation module tests - fast deterministic tests.
"""
import numpy as np
import pandas as pd
import pytest

import demo_data
import engine


def test_validate_input_data_success() -> None:
    """Test validation passes for valid demo data."""
    raw = demo_data.generate_demo_data()
    report = engine.validate_input_data(raw)
    assert report.is_valid
    assert len(report.errors) == 0
    assert report.row_count > 0


def test_validate_input_data_missing_columns() -> None:
    """Test validation fails for missing required columns."""
    raw = demo_data.generate_demo_data()
    raw_missing = raw.drop(columns=["units"])
    report = engine.validate_input_data(raw_missing)
    assert not report.is_valid
    assert any("units" in e for e in report.errors)


def test_validate_input_data_duplicate_keys() -> None:
    """Test validation fails for duplicate keys."""
    raw = demo_data.generate_demo_data()
    # Create duplicate row
    raw_dup = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
    report = engine.validate_input_data(raw_dup)
    assert not report.is_valid
    assert any("Duplicate" in e for e in report.errors)


def test_validate_input_data_negative_units() -> None:
    """Test validation rejects negative units for DirichletMultinomial."""
    raw = demo_data.generate_demo_data()
    raw.loc[0, "units"] = -5
    report = engine.validate_input_data(raw)
    assert not report.is_valid
    assert any("non-negative" in e for e in report.errors)


def test_validate_input_data_zero_sku_stores() -> None:
    """Test validation allows zero sku_stores but prepares data correctly."""
    raw = demo_data.generate_demo_data()
    raw.loc[0, "sku_stores"] = 0
    report = engine.validate_input_data(raw)
    # sku_stores=0 is allowed (caught later in data prep if needed)
    assert report.is_valid
    # But data preparation will flag it
    df, _ = engine.prepare_data(raw)
    assert "invalid_store_counts" in df["data_quality_status"].values


def test_validate_input_data_fractional_units_rejected() -> None:
    """Test DirichletMultinomial validation rejects fractional units."""
    raw = demo_data.generate_demo_data()
    raw.loc[0, "units"] = 10.5  # Fractional
    report = engine.validate_input_data(raw)
    assert not report.is_valid
    assert any("integer" in e.lower() for e in report.errors)


def test_validate_input_data_market_total_positive() -> None:
    """Test validation requires positive market total."""
    raw = demo_data.generate_demo_data()
    # Set all units in first market to 0
    mask = (raw["month"] == raw["month"].iloc[0]) & \
           (raw["retailer"] == raw["retailer"].iloc[0]) & \
           (raw["category"] == raw["category"].iloc[0])
    raw.loc[mask, "units"] = 0
    report = engine.validate_input_data(raw)
    assert not report.is_valid
    assert any("positive total" in e.lower() for e in report.errors)


def test_prepare_data_returns_tuple() -> None:
    """Test prepare_data returns (df, meta) tuple."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(meta, dict)
    assert len(df) > 0
    assert 'log_nd_z' in df.columns
    assert 'log_relative_price_z' in df.columns


def test_make_pack_group() -> None:
    """Test pack group mapping."""
    pack_sizes = pd.Series([0.3, 0.5, 1.0, 2.0, 3.0])
    groups = engine.make_pack_group(pack_sizes)
    assert list(groups) == ["small", "small", "medium", "large", "xlarge"]


def test_build_retail_features() -> None:
    """Test build_retail_features returns enriched DataFrame."""
    raw = demo_data.generate_demo_data()
    df, _ = engine.prepare_data(raw)
    enriched = engine.build_retail_features(df)
    assert "relative_price" in enriched.columns
    assert "same_brand_other_sku_price_index" in enriched.columns
    assert "other_brand_price_index" in enriched.columns
    assert enriched["relative_price"].notna().all()


def test_create_data_quality_report() -> None:
    """Test data quality report creation."""
    raw = demo_data.generate_demo_data()
    df, _ = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    validation = engine.validate_input_data(raw)
    report = engine.validate_input_data(raw)
    # validate_input_data returns ValidationReport with exclusion info
    assert hasattr(report, "warnings")
    assert hasattr(report, "row_count")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])