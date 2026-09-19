"""Tests for diagnostics and convergence."""
import numpy as np
import pytest

import demo_data
import engine


def test_joint_model_fits() -> None:
    """Test that joint model fits and produces posterior."""
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

    assert hasattr(idata, "posterior")
    assert "beta_sku" in idata.posterior
    assert "gamma1_sku" in idata.posterior
    assert "sku_share" in idata.posterior
    assert "inclusive_value" in idata.posterior
    print("✓ Joint model fits and produces posterior")


def test_joint_posterior_extraction() -> None:
    """Test posterior extraction for joint model."""
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

    cache = engine.extract_joint_posterior(idata, max_draws=10, choice_data=choice_data)

    assert "beta_sku" in cache
    assert "gamma1_sku" in cache
    assert "gamma2_sku" in cache
    assert "alpha_sku" in cache
    assert "alpha_retailer_sku" in cache
    assert "pack_effect" in cache
    assert "cat_month_effect" in cache
    assert "concentration_category" in cache
    assert "sku_share" in cache
    assert cache["beta_sku"].shape[1] == 12  # n_skus
    assert cache["sku_share"].shape[1:] == (len(choice_data.market_ids), 12)
    print("✓ Joint posterior extraction works")


def test_get_model_diagnostics() -> None:
    """Test model diagnostics computation."""
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

    diagnostics = engine.get_model_diagnostics(idata, model_mode="Joint")
    
    assert "divergences" in diagnostics
    assert "max_rhat" in diagnostics
    assert "min_ess_bulk" in diagnostics
    assert "min_ess_tail" in diagnostics
    assert "is_acceptable" in diagnostics
    print("✓ Model diagnostics computed")


def test_summarize_convergence_diagnostics() -> None:
    """Test convergence diagnostics summary."""
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

    conv_diag = engine.summarize_convergence_diagnostics(idata)
    
    assert conv_diag is not None
    assert hasattr(conv_diag, 'max_rhat')
    assert hasattr(conv_diag, 'divergences')
    print("✓ Convergence diagnostics summarized")


def test_elasticity_extraction() -> None:
    """Test elasticity extraction is callable."""
    assert callable(engine.extract_elasticities)
    print("✓ extract_elasticities callable")


if __name__ == "__main__":
    test_joint_model_fits()
    test_joint_posterior_extraction()
    test_get_model_diagnostics()
    test_summarize_convergence_diagnostics()
    test_elasticity_extraction()
    print("\n=== ALL DIAGNOSTICS TESTS PASSED ===")
