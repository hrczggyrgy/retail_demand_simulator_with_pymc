"""Tests for model structure and building."""
import numpy as np
import pytest

import demo_data
import engine


def test_joint_model_builds() -> None:
    """Test that joint SKU share model builds without error."""
    raw = demo_data.generate_demo_data()
    df, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    enriched = engine.build_retail_features(indexed)

    choice_data = engine.build_choice_set_data(enriched, meta)

    fast_config = engine.JointModelConfig(
        draws=50, tune=50, chains=1, target_accept=0.9,
        use_category_price_pooling=True, use_sku_nd_effects=True,
    )
    model = engine.build_joint_sku_share_model(choice_data, fast_config)

    assert "beta_sku" in model.named_vars
    assert "gamma1_sku" in model.named_vars
    assert "gamma2_sku" in model.named_vars
    assert "sku_share" in model.named_vars
    assert "inclusive_value" in model.named_vars
    assert "sku_units_obs" in model.named_vars
    print("✓ Joint model builds with all required variables")


def test_nest_model_builds() -> None:
    """Test that nest allocation model builds without error."""
    raw = demo_data.generate_demo_data()
    df, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(df)
    enriched = engine.build_retail_features(indexed)

    choice_data = engine.build_choice_set_data(enriched, meta)

    # Create minimal posterior cache for testing
    n_draws = 10
    n_rc = 12  # 4 retailers × 3 categories
    n_categories = 3
    n_months = 24
    n_markets = len(choice_data.market_ids)
    
    cache = {
        "inclusive_value": np.random.randn(n_draws, n_markets),
        "sku_share": np.random.randn(n_draws, n_markets, 12),
    }

    nest_config = engine.NestModelConfig(
        draws=30, tune=30, chains=1, target_accept=0.9,
    )
    model = engine.build_nest_allocation_model(choice_data, cache, nest_config)

    assert "alpha_rc" in model.named_vars
    assert "theta_c" in model.named_vars
    assert "lambda_c" in model.named_vars
    assert "w_rc" in model.named_vars
    assert "w_outside" in model.named_vars
    assert "market_totals_obs" in model.named_vars
    print("✓ Nest allocation model builds with all required variables")


if __name__ == "__main__":
    test_joint_model_builds()
    test_nest_model_builds()
    print("\n=== ALL MODEL STRUCTURE TESTS PASSED ===")
