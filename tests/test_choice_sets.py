"""Tests for choice-set data layer."""
import numpy as np
import pandas as pd
import pytest

import demo_data
import engine


def test_choice_set_data_layer() -> None:
    """Test that choice set data layer builds correctly."""
    raw = demo_data.generate_demo_data()
    df, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    enriched = engine.build_retail_features(indexed)

    choice_data = engine.build_choice_set_data(enriched, meta)

    assert len(choice_data.market_ids) > 0
    assert len(choice_data.sku_ids) == 12
    assert choice_data.observed_units.shape == (len(choice_data.market_ids), 12)
    assert choice_data.relative_price.shape == (len(choice_data.market_ids), 12)
    assert choice_data.nd.shape == (len(choice_data.market_ids), 12)
    assert choice_data.available_mask.shape == (len(choice_data.market_ids), 12)
    assert choice_data.market_total_units.shape == (len(choice_data.market_ids),)
    assert choice_data.market_retailer_idx.shape == (len(choice_data.market_ids),)
    assert choice_data.market_category_idx.shape == (len(choice_data.market_ids),)
    assert choice_data.market_month_idx.shape == (len(choice_data.market_ids),)
    assert choice_data.sku_brand_idx.shape == (12,)
    assert choice_data.sku_category_idx.shape == (12,)
    assert choice_data.sku_pack_group_idx.shape == (12,)
    assert choice_data.brand_category_idx.shape == (6,)
    assert len(choice_data.sku_to_idx) == 12
    assert len(choice_data.idx_to_sku) == 12
    print("✓ Choice set data layer builds correctly")


def test_choice_set_validation() -> None:
    """Test choice set validation passes for valid data."""
    raw = demo_data.generate_demo_data()
    df, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    enriched = engine.build_retail_features(indexed)

    choice_data = engine.build_choice_set_data(enriched, meta)
    validation = engine.validate_choice_set(choice_data)

    assert validation["market_totals_match"] is True
    assert validation["unavailable_have_zero_units"] is True
    assert validation["shares_sum_to_one"] is True
    assert validation["n_markets"] == len(choice_data.market_ids)
    assert validation["n_skus"] == 12
    print("✓ Choice set validation passes")


def test_choice_set_unavailable_zero_units() -> None:
    """Test that unavailable SKUs have zero observed units."""
    raw = demo_data.generate_demo_data()
    df, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    enriched = engine.build_retail_features(indexed)

    choice_data = engine.build_choice_set_data(enriched, meta)

    unavailable_units = choice_data.observed_units[~choice_data.available_mask].sum()
    assert unavailable_units == 0.0

    available_units = choice_data.observed_units[choice_data.available_mask]
    assert (available_units >= 0).all()
    print("✓ Unavailable SKUs have zero units")


if __name__ == "__main__":
    test_choice_set_data_layer()
    test_choice_set_validation()
    test_choice_set_unavailable_zero_units()
    print("\n=== ALL CHOICE SET TESTS PASSED ===")
