"""
Choice-set data layer for the joint SKU market-share model.

Builds the ChoiceSetData container with all arrays and metadata needed
for the Dirichlet-Multinomial choice model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from contracts import (
    ChoiceSetData,
    DERIVED_COLUMNS,
    PRICE_STD,
    RELATIVE_PRICE,
    ND,
)


def _get_category_pack_peer_price(
    df: pd.DataFrame,
    price_col: str = PRICE_STD,
    volume_col: str = "standard_volume",
) -> pd.Series:
    """
    Compute category/pack-peer price basket for each SKU.

    Peer basket = volume-weighted average price of other SKUs in same
    retailer × category × pack_group × month, excluding focal SKU.
    Prefers other brands when available.
    """
    df = df.copy()
    df["_pxv"] = df[price_col] * df[volume_col]

    # Group by retailer, category, pack_group, month
    peer_groups = ["retailer", "category", "pack_group", "month"]

    total_pxv = df.groupby(peer_groups)["_pxv"].transform("sum")
    total_vol = df.groupby(peer_groups)[volume_col].transform("sum")

    # Exclude self
    self_pxv = df["_pxv"]
    self_vol = df[volume_col]

    peer_pxv = total_pxv - self_pxv
    peer_vol = total_vol - self_vol

    peer_price = peer_pxv / peer_vol
    peer_price = peer_price.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return peer_price


def build_choice_set_data(
    df: pd.DataFrame,
    price_col: str = PRICE_STD,
) -> ChoiceSetData:
    """
    Build ChoiceSetData from enriched DataFrame.

    Expects df to have columns from DERIVED_COLUMNS plus:
    - month (datetime)
    - retailer, category, brand, sku
    - price_std, relative_price, nd
    """
    # Market definition: retailer × category × month
    df["_market_id"] = df["retailer"] + "|" + df["category"] + "|" + df["month"].dt.strftime("%Y-%m")
    market_ids = sorted(df["_market_id"].unique())
    market_to_idx = {m: i for i, m in enumerate(market_ids)}

    # SKU universe
    sku_ids = sorted(df["sku"].unique())
    sku_to_idx = {s: i for i, s in enumerate(sku_ids)}
    idx_to_sku = {i: s for i, s in enumerate(sku_ids)}
    n_markets = len(market_ids)
    n_skus = len(sku_ids)

    # Factorize categorical variables
    market_retailer_idx, retailer_levels = pd.factorize(
        df.groupby("_market_id")["retailer"].first()
    )
    market_category_idx, category_levels = pd.factorize(
        df.groupby("_market_id")["category"].first()
    )
    market_month_idx, month_levels = pd.factorize(
        df.groupby("_market_id")["month"].first()
    )

    sku_brand_idx, brand_levels = pd.factorize(
        df.groupby("sku")["brand"].first()
    )
    sku_category_idx, category_levels_sku = pd.factorize(
        df.groupby("sku")["category"].first()
    )
    sku_pack_group_idx, pack_group_levels = pd.factorize(
        df.groupby("sku")["pack_group"].first()
    )

    # Brand -> category mapping
    brand_to_cat = df.drop_duplicates("brand").set_index("brand")["category"]
    brand_cat_idx = np.array([
        np.where(category_levels_sku == brand_to_cat[b])[0][0]
        for b in brand_levels
    ])

    # Initialize padded arrays
    observed_units = np.zeros((n_markets, n_skus), dtype=np.float64)
    price_std = np.zeros((n_markets, n_skus), dtype=np.float64)
    relative_price = np.ones((n_markets, n_skus), dtype=np.float64)
    nd_array = np.zeros((n_markets, n_skus), dtype=np.float64)
    available_mask = np.zeros((n_markets, n_skus), dtype=bool)

    # Fill arrays
    for _, row in df.iterrows():
        m_idx = market_to_idx[row["_market_id"]]
        s_idx = sku_to_idx[row["sku"]]
        observed_units[m_idx, s_idx] = row["units"]
        price_std[m_idx, s_idx] = row[price_col]
        relative_price[m_idx, s_idx] = row[RELATIVE_PRICE]
        nd_array[m_idx, s_idx] = row[ND]
        # Available if ND > 0 (listed in at least one store)
        available_mask[m_idx, s_idx] = row[ND] > 0

    # Market totals
    market_total_units = observed_units.sum(axis=1)

    return ChoiceSetData(
        observed_units=observed_units,
        market_total_units=market_total_units,
        price_std=price_std,
        relative_price=relative_price,
        nd=nd_array,
        available_mask=available_mask,
        market_retailer_idx=market_retailer_idx,
        market_category_idx=market_category_idx,
        market_month_idx=market_month_idx,
        sku_brand_idx=sku_brand_idx,
        sku_category_idx=sku_category_idx,
        sku_pack_group_idx=sku_pack_group_idx,
        brand_category_idx=brand_cat_idx,
        market_ids=market_ids,
        sku_ids=sku_ids,
        retailer_levels=retailer_levels,
        category_levels=category_levels,
        brand_levels=brand_levels,
        month_levels=month_levels,
        pack_group_levels=pack_group_levels,
        sku_to_idx=sku_to_idx,
        idx_to_sku=idx_to_sku,
    )


def _compute_peer_price_matrix(
    choice_data: ChoiceSetData,
    price_std: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Compute category/pack-peer price for each market × SKU.

    Peer price = volume-weighted average of OTHER SKUs in same
    retailer × category × pack_group × month.
    """
    n_markets, n_skus = price_std.shape
    peer_price = np.ones((n_markets, n_skus), dtype=np.float64)

    # Group markets by (retailer, category, pack_group, month)
    sku_pack_group = choice_data.sku_pack_group_idx

    for m_idx in range(n_markets):
        retailer = choice_data.market_retailer_idx[m_idx]
        category = choice_data.market_category_idx[m_idx]
        month = choice_data.market_month_idx[m_idx]

        # Find available SKUs in this market
        avail = choice_data.available_mask[m_idx]
        market_skus = np.where(avail)[0]

        if len(market_skus) <= 1:
            continue

        # Group by pack_group
        for s_idx in market_skus:
            pg = sku_pack_group[s_idx]
            # Find other SKUs in same pack group
            other_skus = [s for s in market_skus if s != s_idx and sku_pack_group[s] == pg]
            if len(other_skus) == 0:
                continue
            other_vol = volume[m_idx, other_skus]
            other_price = price_std[m_idx, other_skus]
            if other_vol.sum() > 0:
                peer_price[m_idx, s_idx] = np.sum(other_price * other_vol) / np.sum(other_vol)
            else:
                peer_price[m_idx, s_idx] = price_std[m_idx, s_idx]

    return peer_price


def _recompute_relative_prices(
    choice_data: ChoiceSetData,
    price_std: np.ndarray,
    affected_markets: set[int],
) -> np.ndarray:
    """
    Recompute relative prices for all SKUs in affected markets.

    Relative price = price_std / peer_price, where peer_price is the
    volume-weighted average of OTHER SKUs in the same retailer-category-pack_group-month.
    """
    n_markets, n_skus = price_std.shape
    rel_price = choice_data.relative_price.copy()

    sku_pack_group = choice_data.sku_pack_group_idx
    baseline_volume = choice_data.observed_units

    for m_idx in affected_markets:
        avail = choice_data.available_mask[m_idx]
        market_skus = np.where(avail)[0]

        if len(market_skus) <= 1:
            for s_idx in market_skus:
                rel_price[m_idx, s_idx] = 1.0
            continue

        for s_idx in market_skus:
            pg = sku_pack_group[s_idx]
            other_skus = [s for s in market_skus if s != s_idx and sku_pack_group[s] == pg]
            if len(other_skus) == 0:
                rel_price[m_idx, s_idx] = 1.0
                continue

            other_vol = baseline_volume[m_idx, other_skus]
            other_price = price_std[m_idx, other_skus]
            if other_vol.sum() > 0:
                peer_price = np.sum(other_price * other_vol) / np.sum(other_vol)
                rel_price[m_idx, s_idx] = price_std[m_idx, s_idx] / peer_price
            else:
                rel_price[m_idx, s_idx] = 1.0

    rel_price = np.where(np.isfinite(rel_price), rel_price, 1.0)
    return rel_price