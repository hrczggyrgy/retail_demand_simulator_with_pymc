import pandas as pd
import numpy as np
import pytest

import demo_data
import engine


def test_app_integration_smoke():
    """End-to-end smoke test of app pipeline."""
    raw = demo_data.generate_demo_data()
    
    # Validation
    validation = engine.validate_input_data(raw)
    assert validation.is_valid
    
    # Data preparation
    prepared, scales = engine.prepare_data(raw)
    indexed, meta = engine.add_indices(prepared)
    enriched = engine.build_retail_features(indexed)
    assert len(enriched) == 1152
    
    # Choice set
    choice_data = engine.build_choice_set_data(enriched, meta)
    assert len(choice_data.market_ids) > 0
    assert len(choice_data.sku_ids) == 12
    
    # Model building
    joint_config = engine.JointModelConfig(
        draws=10, tune=10, chains=1, target_accept=0.9,
    )
    model = engine.build_joint_sku_share_model(choice_data, joint_config)
    assert "beta_sku" in model.named_vars
    
    # Model fitting
    idata = engine.fit_joint_model(model, joint_config)
    assert hasattr(idata, "posterior")
    
    # Diagnostics
    diagnostics = engine.get_model_diagnostics(idata, model_mode="Joint")
    assert "divergences" in diagnostics
    
    # Posterior extraction
    cache = engine.extract_joint_posterior(idata, max_draws=5, choice_data=choice_data)
    assert "beta_sku" in cache
    assert "sku_share" in cache
    
    # Scenario
    action = engine.ScenarioAction(
        retailer=choice_data.retailer_levels[choice_data.market_retailer_idx[0]],
        category=choice_data.category_levels[choice_data.market_category_idx[0]],
        brand=choice_data.brand_levels[choice_data.sku_brand_idx[0]],
        sku=choice_data.sku_ids[0],
        month=str(choice_data.month_levels[choice_data.market_month_idx[0]]).split(" ")[0],
        new_price_std=choice_data.price_std[0, 0] * 0.95,
        new_nd=None,
        nd_mode="pp",
    )
    result = engine.run_joint_scenario_draws(choice_data, cache, [action], n_draws=3, random_seed=42)
    assert isinstance(result, engine.ScenarioResult)
    
    # Reconciliation
    recon = engine.check_scenario_reconciliation(result, choice_data)
    assert recon["baseline_shares_sum_to_one"] is True
    
    print("✓ End-to-end app integration smoke test passed")


def test_create_scenario_action_table():
    """Test scenario action table creation."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    enriched = engine.build_retail_features(df)
    
    action_table = engine.create_scenario_action_table(enriched)
    assert isinstance(action_table, pd.DataFrame)
    assert "retailer" in action_table.columns
    assert "sku" in action_table.columns
    assert "include" in action_table.columns
    print("✓ create_scenario_action_table works")


def test_create_category_bubble_map():
    """Test category bubble map creation."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    # Create mock market impact data
    impact = pd.DataFrame({
        "category": ["CSD", "Water", "Juice"],
        "scenario_share": [0.4, 0.3, 0.3],
        "share_change_pp": [1.0, -0.5, -0.5],
        "scenario_revenue": [1000, 800, 600],
        "baseline_share": [0.39, 0.305, 0.305],
        "delta_revenue_pct": [5.0, -2.0, -1.0],
    })
    fig = engine.create_category_bubble_map(impact)
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    print("✓ create_category_bubble_map works")


def test_create_brand_pack_heatmap():
    """Test brand pack heatmap creation."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    impact = pd.DataFrame({
        "brand": ["A", "A", "B", "B"],
        "pack_group": ["Small", "Large", "Small", "Large"],
        "share_change_pp": [1.0, -0.5, 0.5, -1.0],
    })
    fig = engine.create_brand_pack_heatmap(impact)
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    print("✓ create_brand_pack_heatmap works")


def test_create_winner_loser_dumbbell():
    """Test winner loser dumbbell creation."""
    raw = demo_data.generate_demo_data()
    df, meta = engine.prepare_data(raw)
    baseline = df.copy()
    scenario = df.copy()
    scenario["units"] = scenario["units"] * 1.1
    fig = engine.create_winner_loser_dumbbell(baseline, scenario, "brand", 5)
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure)
    print("✓ create_winner_loser_dumbbell works")


if __name__ == "__main__":
    test_app_integration_smoke()
    test_create_scenario_action_table()
    test_create_category_bubble_map()
    test_create_brand_pack_heatmap()
    test_create_winner_loser_dumbbell()
    print("\n=== ALL APP SMOKE TESTS PASSED ===")
