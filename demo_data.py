# ============================================================
# Template and Demo Data
# ============================================================

from typing import Dict, Any

import numpy as np
import pandas as pd

from engine import REQUIRED_COLUMNS


def get_input_template() -> pd.DataFrame:
    """Return empty DataFrame with required columns for template download."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def generate_demo_data(months_count: int = 24, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic demo data matching the notebook's generator."""
    np.random.seed(seed)
    
    months = pd.date_range(start="2024-01-01", periods=months_count, freq="MS").strftime("%Y-%m-%d").tolist()
    
    retailers = {
        "Apex Supermarkets": {"base_stores": 310, "growth": 6, "type": "supermarket"},
        "Horizon Discount": {"base_stores": 240, "growth": 10, "type": "discount"},
        "Vanguard Express": {"base_stores": 160, "growth": 4, "type": "convenience"},
        "Beacon Hypermarket": {"base_stores": 55, "growth": 1, "type": "hypermarket"},
    }
    
    products = [
        # CSD
        {"category": "CSD", "brand": "Fizz Up", "sku": "FIZZ_500ML", "pack_size": 0.5, "base_price": 1.99, "base_vel": 180, "season": "csd"},
        {"category": "CSD", "brand": "Fizz Up", "sku": "FIZZ_2000ML", "pack_size": 2.0, "base_price": 2.99, "base_vel": 120, "season": "csd"},
        {"category": "CSD", "brand": "Sparkle Cola", "sku": "SPK_500ML", "pack_size": 0.5, "base_price": 1.89, "base_vel": 210, "season": "csd"},
        {"category": "CSD", "brand": "Sparkle Cola", "sku": "SPK_2000ML", "pack_size": 2.0, "base_price": 2.79, "base_vel": 150, "season": "csd"},
        # Juice
        {"category": "Juice", "brand": "SunHarvest", "sku": "SH_ORANGE_1000ML", "pack_size": 1.0, "base_price": 3.49, "base_vel": 110, "season": "juice"},
        {"category": "Juice", "brand": "SunHarvest", "sku": "SH_APPLE_1000ML", "pack_size": 1.0, "base_price": 3.29, "base_vel": 95, "season": "juice"},
        {"category": "Juice", "brand": "PureNature", "sku": "PN_ORANGE_1000ML", "pack_size": 1.0, "base_price": 3.99, "base_vel": 80, "season": "juice"},
        {"category": "Juice", "brand": "PureNature", "sku": "PN_GRAPE_1000ML", "pack_size": 1.0, "base_price": 4.19, "base_vel": 65, "season": "juice"},
        # Ice Tea
        {"category": "Ice Tea", "brand": "ChillLeaf", "sku": "CL_LEMON_500ML", "pack_size": 0.5, "base_price": 2.19, "base_vel": 140, "season": "icetea"},
        {"category": "Ice Tea", "brand": "ChillLeaf", "sku": "CL_PEACH_1500ML", "pack_size": 1.5, "base_price": 3.29, "base_vel": 90, "season": "icetea"},
        {"category": "Ice Tea", "brand": "Breeze Tea", "sku": "BT_LEMON_500ML", "pack_size": 0.5, "base_price": 1.99, "base_vel": 160, "season": "icetea"},
        {"category": "Ice Tea", "brand": "Breeze Tea", "sku": "BT_GREEN_500ML", "pack_size": 0.5, "base_price": 2.09, "base_vel": 115, "season": "icetea"},
    ]
    
    rows = []
    for month_idx, month_str in enumerate(months):
        m_num = int(month_str.split("-")[1])
        
        csd_season = 1.25 if m_num in [6, 7, 8] else (1.20 if m_num == 12 else (0.85 if m_num in [1, 2] else 1.00))
        icetea_season = 1.55 if m_num in [6, 7, 8] else (1.25 if m_num in [5, 9] else (0.60 if m_num in [11, 12, 1, 2] else 0.90))
        juice_season = 1.20 if m_num in [11, 12, 1, 2, 3] else (0.85 if m_num in [6, 7, 8] else 1.00)
        
        inflation_factor = 1.0 + (month_idx / 24.0) * 0.07
        
        for ret_name, ret_info in retailers.items():
            monthly_growth = (ret_info["growth"] / 12.0) * month_idx
            ret_stores = int(round(ret_info["base_stores"] + monthly_growth + np.random.randint(-1, 2)))
            
            if ret_info["type"] == "hypermarket":
                channel_vel_mult, listing_bias = 2.8, 0.94
            elif ret_info["type"] == "supermarket":
                channel_vel_mult, listing_bias = 1.2, 0.88
            elif ret_info["type"] == "discount":
                channel_vel_mult, listing_bias = 1.5, 0.78
            else:
                channel_vel_mult, listing_bias = 0.7, 0.70
            
            for p in products:
                if p["pack_size"] <= 0.5 and ret_info["type"] == "convenience":
                    sku_listing_rate = min(0.98, listing_bias + 0.15)
                elif p["pack_size"] >= 1.5 and ret_info["type"] == "convenience":
                    sku_listing_rate = max(0.35, listing_bias - 0.25)
                else:
                    sku_listing_rate = listing_bias + np.random.uniform(-0.05, 0.05)
                
                sku_stores = max(1, min(int(round(ret_stores * sku_listing_rate)), ret_stores))
                
                season_mult = csd_season if p["season"] == "csd" else (icetea_season if p["season"] == "icetea" else juice_season)
                
                is_promo = np.random.rand() < 0.15
                price_discount = np.random.uniform(0.12, 0.22) if is_promo else 0.0
                promo_lift = 1.0 + (price_discount * np.random.uniform(2.2, 3.2)) if is_promo else 1.0
                
                effective_price = p["base_price"] * inflation_factor * (1.0 - price_discount) * np.random.uniform(0.98, 1.02)
                base_velocity = p["base_vel"] * channel_vel_mult * season_mult * promo_lift
                actual_velocity = base_velocity * np.random.uniform(0.92, 1.08)
                
                units = int(round(sku_stores * actual_velocity))
                revenue = round(units * effective_price, 2)
                
                rows.append({
                    "month": month_str,
                    "retailer": ret_name,
                    "category": p["category"],
                    "brand": p["brand"],
                    "sku": p["sku"],
                    "units": units,
                    "revenue": revenue,
                    "sku_stores": sku_stores,
                    "retailer_stores": ret_stores,
                    "pack_size": p["pack_size"],
                })
    
    df = pd.DataFrame(rows)
    df = df.dropna()
    df = df[(df["units"] > 0) & (df["revenue"] > 0) & (df["sku_stores"] <= df["retailer_stores"])]
    df = df[REQUIRED_COLUMNS].copy()
    unexpected = set(df.columns) - set(REQUIRED_COLUMNS)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing or unexpected:
        raise ValueError(
            f"Demo data schema violation. Missing: {sorted(missing)}, unexpected: {sorted(unexpected)}"
        )
    return df


# ============================================================
# Scenario Guardrails
# ============================================================

def check_scenario_guardrails(df: pd.DataFrame, price_change: float, nd_change: float) -> Dict[str, Any]:
    """Check if scenario values are within observed ranges."""
    warnings = []
    
    # ND bounds
    nd_new = df["nd"] * (1 + nd_change)
    if (nd_new > 1).any():
        warnings.append(f"ND change of {nd_change:.0%} would push {(nd_new > 1).sum()} rows above 100% ND; capped at 100%")
    
    # Price range
    rel_price_new = df["relative_price"] * (1 + price_change)
    hist_min, hist_max = df["relative_price"].min(), df["relative_price"].max()
    if (rel_price_new < hist_min).any() or (rel_price_new > hist_max).any():
        outside = ((rel_price_new < hist_min) | (rel_price_new > hist_max)).sum()
        warnings.append(f"Price change of {price_change:.0%} would push {outside} rows outside historical relative price range [{hist_min:.2f}, {hist_max:.2f}]")
    
    return {"warnings": warnings, "affected_rows": len(df)}