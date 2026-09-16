
"""
Retail Demand & Share Simulator
--------------------------------
A simplified Streamlit UI focused on one question:

    "If price or distribution changes, how will market share change?"

The analytical engine remains in engine.py. This app treats the uploaded
dataset as a complete competitive market and keeps competitors in the
share denominator even when the user targets only one brand/SKU/retailer.
"""

from __future__ import annotations

import io
import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import demo_data
import engine


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Retail Share Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        .scenario-card {
            border: 1px solid rgba(128,128,128,.25);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: .8rem;
        }
        .small-note {
            color: rgba(100,100,100,.9);
            font-size: .88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compact_number(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    a = abs(value)
    if a >= 1e9:
        return f"{value/1e9:.1f}B"
    if a >= 1e6:
        return f"{value/1e6:.1f}M"
    if a >= 1e3:
        return f"{value/1e3:.1f}K"
    return f"{value:,.0f}"


def pct(value: float, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


def pp(value: float, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{decimals}f} pp"


def month_label(value) -> str:
    return pd.Timestamp(value).strftime("%b %Y")


def unique_sorted(df: pd.DataFrame, col: str) -> List:
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())


def style_change(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:+.1f} pp"


def fmt_share(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.1%}"


def build_filter_mask(
    df: pd.DataFrame,
    category: Optional[str] = None,
    retailer: Optional[str] = None,
    brand: Optional[str] = None,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if category and category != "All":
        mask &= df["category"].eq(category)
    if retailer and retailer != "All":
        mask &= df["retailer"].eq(retailer)
    if brand and brand != "All":
        mask &= df["brand"].eq(brand)
    return mask


def calculate_market_shares(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    share_level: str = "brand",
) -> pd.DataFrame:
    """
    Calculate baseline and median-scenario market shares.

    Market share is calculated from revenue by default. Since the dataset
    contains the full competitive set, the denominator contains all rows in
    the selected market/view scope.
    """
    group_cols = ["month"]
    if share_level == "brand":
        group_cols.append("brand")
    elif share_level == "sku":
        group_cols.extend(["brand", "sku"])
    else:
        group_cols.extend(["brand", "sku", "pack_size"])

    base = (
        baseline_df.groupby(group_cols, as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "baseline_value"})
    )

    scen = (
        scenario_df.groupby(group_cols, as_index=False)["expected_revenue_median"]
        .sum()
        .rename(columns={"expected_revenue_median": "scenario_value"})
    )

    out = base.merge(scen, on=group_cols, how="outer").fillna(0)

    totals = (
        out.groupby("month", as_index=False)
        .agg(
            market_baseline=("baseline_value", "sum"),
            market_scenario=("scenario_value", "sum"),
        )
    )
    out = out.merge(totals, on="month", how="left")

    out["baseline_share"] = np.where(
        out["market_baseline"] > 0,
        out["baseline_value"] / out["market_baseline"],
        np.nan,
    )
    out["scenario_share"] = np.where(
        out["market_scenario"] > 0,
        out["scenario_value"] / out["market_scenario"],
        np.nan,
    )
    out["share_change_pp"] = out["scenario_share"] - out["baseline_share"]
    return out


def summarize_period_shares(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    share_level: str,
) -> pd.DataFrame:
    """
    Aggregate over the selected period and calculate market shares.
    """
    if share_level == "brand":
        group_cols = ["brand"]
    elif share_level == "sku":
        group_cols = ["brand", "sku"]
    else:
        group_cols = ["brand", "sku", "pack_size"]

    base = (
        baseline_df.groupby(group_cols, as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "baseline_value"})
    )

    scen = (
        scenario_df.groupby(group_cols, as_index=False)["expected_revenue_median"]
        .sum()
        .rename(columns={"expected_revenue_median": "scenario_value"})
    )

    out = base.merge(scen, on=group_cols, how="outer").fillna(0)

    base_total = out["baseline_value"].sum()
    scen_total = out["scenario_value"].sum()

    out["baseline_share"] = (
        out["baseline_value"] / base_total if base_total > 0 else np.nan
    )
    out["scenario_share"] = (
        out["scenario_value"] / scen_total if scen_total > 0 else np.nan
    )
    out["share_change_pp"] = out["scenario_share"] - out["baseline_share"]

    return out.sort_values("scenario_share", ascending=False).reset_index(drop=True)


def apply_target_scope(
    df: pd.DataFrame,
    target_level: str,
    target_value,
) -> pd.Series:
    if target_level == "Market":
        return pd.Series(True, index=df.index)
    if target_level == "Retailer":
        return df["retailer"].eq(target_value)
    if target_level == "Brand":
        return df["brand"].eq(target_value)
    if target_level == "SKU":
        return df["sku"].eq(target_value)
    if target_level == "Pack size":
        return df["pack_size"].eq(target_value)
    return pd.Series(False, index=df.index)


def run_targeted_scenario(
    full_df: pd.DataFrame,
    trace,
    spline,
    scales,
    price_change: float,
    nd_change: float,
    target_level: str,
    target_value,
) -> pd.DataFrame:
    """
    Run the scenario for the complete competitive set, but change only the
    selected target rows.

    This is important for share simulation: competitors remain in the market
    denominator even when the action targets one brand/SKU/retailer.
    """
    work = full_df.copy()
    work["_scenario_row_id"] = np.arange(len(work), dtype=int)

    target_mask = apply_target_scope(work, target_level, target_value)

    # Baseline scenario: zero change for everyone.
    baseline_scenario = engine.create_price_distribution_scenario(
        trace,
        work,
        spline,
        scales,
        0.0,
        0.0,
    )

    if not target_mask.any() or (price_change == 0 and nd_change == 0):
        return baseline_scenario

    target_rows = work.loc[target_mask].copy()

    target_scenario = engine.create_price_distribution_scenario(
        trace,
        target_rows,
        spline,
        scales,
        price_change,
        nd_change,
    )

    # Replace only target rows with the changed scenario result.
    baseline_by_id = baseline_scenario.set_index("_scenario_row_id")
    target_by_id = target_scenario.set_index("_scenario_row_id")

    replace_cols = [
        "scenario_price_change",
        "scenario_nd_change",
        "scenario_nd",
        "volume_multiplier_p10",
        "volume_multiplier_median",
        "volume_multiplier_p90",
        "expected_units_median",
        "expected_revenue_median",
    ]
    replace_cols = [c for c in replace_cols if c in baseline_by_id.columns and c in target_by_id.columns]

    baseline_by_id.loc[target_by_id.index, replace_cols] = target_by_id[replace_cols]

    out = baseline_by_id.reset_index()
    return out


def create_share_change_chart(
    share_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    plot_df = share_df.nlargest(top_n, "scenario_share").copy()
    plot_df = plot_df.sort_values("share_change_pp")

    if "sku" in plot_df.columns:
        labels = plot_df["brand"].astype(str) + " | " + plot_df["sku"].astype(str)
    else:
        labels = plot_df["brand"].astype(str)

    fig = go.Figure(
        go.Bar(
            x=plot_df["share_change_pp"] * 100,
            y=labels,
            orientation="h",
            text=[style_change(x * 100) for x in plot_df["share_change_pp"]],
            textposition="outside",
            hovertemplate=(
                "%{y}<br>"
                "Baseline share: %{customdata[0]:.1%}<br>"
                "Scenario share: %{customdata[1]:.1%}<br>"
                "Change: %{x:.1f} pp<extra></extra>"
            ),
            customdata=np.column_stack(
                [plot_df["baseline_share"], plot_df["scenario_share"]]
            ),
        )
    )

    fig.add_vline(x=0, line_width=1)
    fig.update_layout(
        title="How market share moves",
        xaxis_title="Change in market share (percentage points)",
        yaxis_title="",
        height=max(420, min(720, 35 * len(plot_df) + 120)),
        margin=dict(l=20, r=80, t=60, b=50),
    )
    return fig


def create_share_comparison_chart(
    share_df: pd.DataFrame,
    top_n: int = 12,
) -> go.Figure:
    plot_df = share_df.nlargest(top_n, "scenario_share").copy()

    if "sku" in plot_df.columns:
        labels = plot_df["brand"].astype(str) + " | " + plot_df["sku"].astype(str)
    else:
        labels = plot_df["brand"].astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Baseline",
            x=plot_df["baseline_share"] * 100,
            y=labels,
            orientation="h",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Scenario",
            x=plot_df["scenario_share"] * 100,
            y=labels,
            orientation="h",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Baseline vs scenario share",
        xaxis_title="Market share (%)",
        yaxis_title="",
        height=max(420, min(720, 35 * len(plot_df) + 120)),
        margin=dict(l=20, r=30, t=60, b=50),
        legend_title_text="",
    )
    return fig


def create_trend_chart(
    share_df: pd.DataFrame,
    share_col: str,
    title: str,
    group_col: str = "brand",
    top_n: int = 8,
) -> go.Figure:
    leaders = (
        share_df.groupby(group_col)[share_col]
        .mean()
        .nlargest(top_n)
        .index
    )
    plot_df = share_df[share_df[group_col].isin(leaders)].copy()

    fig = go.Figure()
    for key, sub in plot_df.groupby(group_col):
        sub = sub.sort_values("month")
        fig.add_trace(
            go.Scatter(
                x=sub["month"],
                y=sub[share_col] * 100,
                mode="lines",
                name=str(key),
                hovertemplate=f"{key}<br>%{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Market share (%)",
        height=420,
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Session state / data loading
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "raw_data": None,
        "prepared_data": None,
        "scales": None,
        "meta": None,
        "spline": None,
        "nd_basis": None,
        "validation_report": None,
        "quality_report": None,
        "model_result": None,
        "model_diagnostics": None,
        "model_settings": None,
        "elasticities": None,
        "scenario_df": None,
        "scenario_shares": None,
        "scenario_target": None,
        "scenario_price": 0.0,
        "scenario_nd": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data
def read_uploaded_file(file_bytes: bytes, file_type: str, sheet_name: Optional[str]) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    if file_type == "csv":
        return pd.read_csv(bio)
    return pd.read_excel(bio, sheet_name=sheet_name)


@st.cache_data
def demo_data_cached() -> pd.DataFrame:
    return demo_data.generate_demo_data(months_count=24, seed=42)


def clear_analysis_state() -> None:
    for key in [
        "prepared_data",
        "scales",
        "meta",
        "spline",
        "nd_basis",
        "validation_report",
        "quality_report",
        "model_result",
        "model_diagnostics",
        "model_settings",
        "elasticities",
        "scenario_df",
        "scenario_shares",
        "scenario_target",
    ]:
        st.session_state[key] = None


def load_dataframe(df: pd.DataFrame) -> None:
    clear_analysis_state()
    st.session_state.raw_data = df.copy()


def prepare_data_if_needed() -> None:
    if st.session_state.raw_data is None or st.session_state.prepared_data is not None:
        return

    raw = st.session_state.raw_data

    with st.spinner("Checking and preparing the market..."):
        validation = engine.validate_input_data(raw)

        if not validation.is_valid:
            st.session_state.validation_report = validation
            st.error("The data could not be prepared. Check the required columns and data quality.")
            return

        prepared, scales = engine.prepare_data(raw)
        indexed, meta = engine.add_indices(prepared)
        spline, nd_basis = engine.fit_spline(
            indexed,
            n_knots=engine.DEFAULT_N_SPLINE_KNOTS,
            degree=engine.DEFAULT_SPLINE_DEGREE,
        )
        quality = engine.create_data_quality_report(raw, indexed, validation)

        st.session_state.prepared_data = indexed
        st.session_state.scales = scales
        st.session_state.meta = meta
        st.session_state.spline = spline
        st.session_state.nd_basis = nd_basis
        st.session_state.validation_report = validation
        st.session_state.quality_report = quality


def fit_model_if_needed() -> None:
    if st.session_state.model_result is not None:
        return
    if st.session_state.prepared_data is None:
        return

    df = st.session_state.prepared_data

    with st.spinner("Estimating price and distribution response..."):
        settings = {
            "draws": engine.DEFAULT_DRAWS,
            "tune": engine.DEFAULT_TUNE,
            "chains": engine.DEFAULT_CHAINS,
            "target_accept": engine.DEFAULT_TARGET_ACCEPT,
        }

        model = engine.build_pymc_model(
            df,
            st.session_state.meta,
            st.session_state.nd_basis,
        )
        trace = engine.fit_model(
            model,
            draws=settings["draws"],
            tune=settings["tune"],
            chains=settings["chains"],
            target_accept=settings["target_accept"],
        )

        st.session_state.model_result = trace
        st.session_state.model_settings = settings
        st.session_state.model_diagnostics = engine.get_model_diagnostics(trace)
        st.session_state.elasticities = engine.extract_elasticities(
            trace,
            st.session_state.meta,
            st.session_state.scales,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.title("Retail Share Simulator")
        st.caption("Understand how price and distribution can change market share.")

        st.subheader("1. Market")

        uploaded = st.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx"],
            help="The file should contain the complete competitive market: all retailers and competitor SKUs.",
        )

        if uploaded is not None:
            sheet_name = None
            if uploaded.name.lower().endswith(".xlsx"):
                sheets = pd.ExcelFile(uploaded).sheet_names
                sheet_name = st.selectbox("Worksheet", sheets)

            if st.button("Load uploaded data", use_container_width=True, type="primary"):
                ftype = "xlsx" if uploaded.name.lower().endswith(".xlsx") else "csv"
                load_dataframe(read_uploaded_file(uploaded.getvalue(), ftype, sheet_name))
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use demo market", use_container_width=True):
                load_dataframe(demo_data_cached())
                st.rerun()
        with col2:
            template = demo_data.get_input_template()
            st.download_button(
                "Template",
                template.to_csv(index=False),
                "retail_data_template.csv",
                "text/csv",
                use_container_width=True,
            )

        if st.session_state.prepared_data is None:
            return

        st.divider()
        st.subheader("2. View filters")

        df = st.session_state.prepared_data

        categories = ["All"] + unique_sorted(df, "category")
        retailers = ["All"] + unique_sorted(df, "retailer")
        brands = ["All"] + unique_sorted(df, "brand")

        st.session_state.view_category = st.selectbox(
            "Category", categories, index=0, key="view_category_widget"
        )
        st.session_state.view_retailer = st.selectbox(
            "Retailer", retailers, index=0, key="view_retailer_widget"
        )
        st.session_state.view_brand = st.selectbox(
            "Brand", brands, index=0, key="view_brand_widget"
        )

        months = sorted(df["month"].dropna().unique())
        if months:
            st.session_state.view_start = st.selectbox(
                "From", months, index=max(0, len(months) - 12), format_func=month_label
            )
            st.session_state.view_end = st.selectbox(
                "To", months, index=len(months) - 1, format_func=month_label
            )

        st.caption(
            "These filters change the market you are viewing. They do not remove "
            "competitors from the scenario denominator unless you explicitly choose "
            "a different market scope on the scenario page."
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()
    render_sidebar()

    st.title("What happens to market share if price or distribution changes?")
    st.markdown(
        """
        The simulator uses the complete competitive market in the data. You choose
        **one action**, the products/retailers it applies to, and the app estimates
        how **brand or SKU market shares move**.
        """
    )

    if st.session_state.raw_data is None:
        st.info(
            "Start with **Use demo market** or upload your complete market dataset. "
            "The model needs competitor SKUs to calculate relative price and share."
        )
        return

    prepare_data_if_needed()

    if st.session_state.prepared_data is None:
        return

    # Global view
    df = st.session_state.prepared_data.copy()

    view_category = st.session_state.get("view_category", "All")
    view_retailer = st.session_state.get("view_retailer", "All")
    view_brand = st.session_state.get("view_brand", "All")

    view_mask = build_filter_mask(
        df,
        category=view_category,
        retailer=view_retailer,
        brand=view_brand,
    )

    start_month = st.session_state.get("view_start", df["month"].min())
    end_month = st.session_state.get("view_end", df["month"].max())

    if pd.Timestamp(start_month) > pd.Timestamp(end_month):
        start_month, end_month = end_month, start_month

    view_mask &= df["month"].between(pd.Timestamp(start_month), pd.Timestamp(end_month))
    view_df = df.loc[view_mask].copy()

    if view_df.empty:
        st.warning("No data matches the selected view.")
        return

    # Model should generally fit the full market, not the filtered view.
    fit_model_if_needed()

    if st.session_state.model_result is None:
        return

    diagnostics = st.session_state.model_diagnostics or {}
    model_ok = (
        diagnostics.get("divergences", 0) == 0
        and diagnostics.get("max_rhat", 1.0) <= 1.01
    )

    # -----------------------------------------------------------------------
    # Header KPI strip
    # -----------------------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Market value", compact_number(view_df["revenue"].sum()))
    with col2:
        st.metric("Brands", int(view_df["brand"].nunique()))
    with col3:
        st.metric("SKUs", int(view_df["sku"].nunique()))
    with col4:
        st.metric("Retailers", int(view_df["retailer"].nunique()))
    with col5:
        st.metric(
            "Model",
            "Ready" if model_ok else "Check",
            help="Model diagnostics are shown in the advanced section below.",
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Tabs are now only two simple modes.
    # -----------------------------------------------------------------------

    simulate_tab, understand_tab = st.tabs(["🎯 Simulate", "🔎 Understand the model"])

    with simulate_tab:

        st.subheader("Build your what-if")

        left, right = st.columns([1.15, 1])

        with left:
            st.markdown("### 1. Change")
            price_change_pct = st.slider(
                "Price change",
                min_value=-30,
                max_value=30,
                value=0,
                step=1,
                format="%d%%",
                help="Change the target's price relative to its current price.",
            )

            nd_change_pp = st.slider(
                "Distribution change",
                min_value=-30,
                max_value=30,
                value=0,
                step=1,
                format="%d pp",
                help="Change store coverage. +10 pp means, for example, 60% → 70%.",
            )

            st.markdown("### 2. Apply the change to")

            target_level = st.radio(
                "Target",
                ["Market", "Retailer", "Brand", "SKU"],
                horizontal=True,
                index=2,
                help="The selected target receives the price and distribution changes. "
                     "All other competitors stay in the market.",
            )

            target_value = None
            if target_level == "Retailer":
                target_value = st.selectbox(
                    "Choose retailer", unique_sorted(view_df, "retailer")
                )
            elif target_level == "Brand":
                target_value = st.selectbox(
                    "Choose brand", unique_sorted(view_df, "brand")
                )
            elif target_level == "SKU":
                sku_options = (
                    view_df[["brand", "sku"]]
                    .drop_duplicates()
                    .sort_values(["brand", "sku"])
                )
                sku_labels = [
                    f"{row.brand} | {row.sku}" for row in sku_options.itertuples()
                ]
                selected_label = st.selectbox("Choose SKU", sku_labels)
                target_value = selected_label.split(" | ", 1)[1]

            run = st.button(
                "Run share simulation",
                type="primary",
                use_container_width=True,
            )

        with right:
            st.markdown("### What the simulator is doing")

            target_text = (
                "the complete market"
                if target_level == "Market"
                else f"{target_level.lower()} **{target_value}**"
            )

            st.info(
                f"**Action:** {price_change_pct:+d}% price, "
                f"{nd_change_pp:+d} pp distribution\n\n"
                f"**Target:** {target_text}\n\n"
                "The share denominator keeps the other competitor brands and SKUs "
                "in the market."
            )

            st.caption(
                "Interpretation: this is a model-based historical scenario estimate. "
                "It is not a causal guarantee of what will happen after an intervention."
            )

        if run:
            price_change = price_change_pct / 100.0
            nd_change = nd_change_pp / 100.0

            # The scenario itself should use the full competitive data, but only
            # the selected market/view scope. That preserves the user's chosen
            # market definition while retaining all competitors within it.
            market_scope_df = df[
                df["month"].between(pd.Timestamp(start_month), pd.Timestamp(end_month))
            ].copy()

            if view_category != "All":
                market_scope_df = market_scope_df[
                    market_scope_df["category"].eq(view_category)
                ]

            if market_scope_df.empty:
                st.warning("No market rows remain for the selected scope.")
            else:
                with st.spinner("Calculating scenario and share shifts..."):
                    scenario = run_targeted_scenario(
                        market_scope_df,
                        st.session_state.model_result,
                        st.session_state.spline,
                        st.session_state.scales,
                        price_change,
                        nd_change,
                        target_level,
                        target_value,
                    )

                    # For share calculations, use the entire selected market
                    # denominator. Retailer / brand / SKU target is only the action scope.
                    share_level = st.radio(
                        "Show share at",
                        ["Brand", "SKU"],
                        horizontal=True,
                        key="share_level_after_run",
                    )

                    share_df = summarize_period_shares(
                        market_scope_df,
                        scenario,
                        share_level.lower(),
                    )

                    st.session_state.scenario_df = scenario
                    st.session_state.scenario_shares = share_df
                    st.session_state.scenario_target = {
                        "level": target_level,
                        "value": target_value,
                        "price_change": price_change,
                        "nd_change": nd_change,
                        "category": view_category,
                        "start": start_month,
                        "end": end_month,
                        "share_level": share_level.lower(),
                    }

        if st.session_state.scenario_shares is not None:
            share_df = st.session_state.scenario_shares
            settings = st.session_state.scenario_target or {}

            st.divider()
            st.subheader("Estimated market-share impact")

            target_level = settings.get("level")
            target_value = settings.get("value")
            price_change = settings.get("price_change", 0.0)
            nd_change = settings.get("nd_change", 0.0)

            target_label = (
                "Entire market"
                if target_level == "Market"
                else f"{target_level}: {target_value}"
            )

            st.caption(
                f"{target_label} · price {price_change:+.0%} · "
                f"distribution {nd_change * 100:+.0f} pp"
            )

            # Identify target rows in the share result.
            if target_level == "Brand" and "brand" in share_df.columns:
                target_rows = share_df[share_df["brand"].eq(target_value)]
            elif target_level == "SKU" and "sku" in share_df.columns:
                target_rows = share_df[share_df["sku"].eq(target_value)]
            elif target_level == "Retailer":
                # Retailer is a scenario target but shares are market-wide brand/SKU shares.
                target_rows = pd.DataFrame()
            else:
                target_rows = share_df

            if not target_rows.empty:
                target_share_change = target_rows["share_change_pp"].sum()
                target_baseline_share = target_rows["baseline_share"].sum()
                target_scenario_share = target_rows["scenario_share"].sum()
            else:
                target_share_change = np.nan
                target_baseline_share = np.nan
                target_scenario_share = np.nan

            a, b, c, d = st.columns(4)
            with a:
                st.metric(
                    "Target baseline share",
                    fmt_share(target_baseline_share),
                )
            with b:
                st.metric(
                    "Target scenario share",
                    fmt_share(target_scenario_share),
                )
            with c:
                st.metric(
                    "Share change",
                    style_change(target_share_change * 100)
                    if not pd.isna(target_share_change)
                    else "See table",
                )
            with d:
                total_base = share_df["baseline_value"].sum()
                total_scen = share_df["scenario_value"].sum()
                total_rev_change = total_scen / total_base - 1 if total_base > 0 else np.nan
                st.metric(
                    "Market value change",
                    pct(total_rev_change),
                )

            col_left, col_right = st.columns(2)
            with col_left:
                st.plotly_chart(
                    create_share_change_chart(share_df),
                    use_container_width=True,
                )
            with col_right:
                st.plotly_chart(
                    create_share_comparison_chart(share_df),
                    use_container_width=True,
                )

            st.markdown("### Where does the share move?")
            display_cols = []
            if "brand" in share_df.columns:
                display_cols.append("brand")
            if "sku" in share_df.columns:
                display_cols.append("sku")
            display_cols += [
                "baseline_share",
                "scenario_share",
                "share_change_pp",
            ]

            display = share_df[display_cols].copy()
            display = display.sort_values("share_change_pp", ascending=False)
            display = display.rename(
                columns={
                    "brand": "Brand",
                    "sku": "SKU",
                    "baseline_share": "Baseline share",
                    "scenario_share": "Scenario share",
                    "share_change_pp": "Share change (pp)",
                }
            )

            st.dataframe(
                display.style.format(
                    {
                        "Baseline share": "{:.1%}",
                        "Scenario share": "{:.1%}",
                        "Share change (pp)": lambda x: f"{x * 100:+.1f} pp",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Volume / revenue impact for target scope
            scenario_df = st.session_state.scenario_df
            target_mask = apply_target_scope(
                scenario_df,
                target_level,
                target_value,
            )
            target_scenario_df = scenario_df.loc[target_mask].copy()

            if not target_scenario_df.empty:
                st.markdown("### What happened to the target itself?")

                baseline_units = target_scenario_df["units"].sum()
                scenario_units = target_scenario_df["expected_units_median"].sum()
                baseline_revenue = target_scenario_df["revenue"].sum()
                scenario_revenue = target_scenario_df["expected_revenue_median"].sum()

                u1, u2, u3, u4 = st.columns(4)
                with u1:
                    st.metric("Target baseline units", compact_number(baseline_units))
                with u2:
                    st.metric(
                        "Target scenario units",
                        compact_number(scenario_units),
                        f"{(scenario_units / baseline_units - 1) * 100:+.1f}%"
                        if baseline_units > 0 else None,
                    )
                with u3:
                    st.metric("Target baseline value", compact_number(baseline_revenue))
                with u4:
                    st.metric(
                        "Target scenario value",
                        compact_number(scenario_revenue),
                        f"{(scenario_revenue / baseline_revenue - 1) * 100:+.1f}%"
                        if baseline_revenue > 0 else None,
                    )

    with understand_tab:
        st.subheader("How the model works")

        st.markdown(
            """
            ### The business question

            The model estimates how **sales velocity** changes when a product's
            **relative price** or **store coverage** changes. The share calculation
            then compares the resulting value across the complete competitive set.

            **Think of the flow as:**

            `Price / Distribution change`
            → `Estimated sales response`
            → `Brand / SKU value`
            → `Market share change`
            """
        )

        st.divider()

        diagnostics = st.session_state.model_diagnostics or {}
        d1, d2, d3 = st.columns(3)
        with d1:
            div = diagnostics.get("divergences")
            st.metric("Sampling divergences", "0" if div == 0 else str(div))
        with d2:
            rhat = diagnostics.get("max_rhat")
            st.metric("Max R-hat", f"{rhat:.3f}" if rhat is not None else "—")
        with d3:
            ess = diagnostics.get("min_ess")
            st.metric("Min bulk ESS", f"{ess:.0f}" if ess is not None else "—")

        if diagnostics.get("divergences", 0) == 0 and diagnostics.get("max_rhat", 1) <= 1.01:
            st.success("The fitted model passes the basic convergence checks used by this app.")
        else:
            st.warning(
                "The model should be treated cautiously because one or more convergence "
                "checks are outside the app's thresholds."
            )

        st.divider()

        st.subheader("Observed share today")

        full_view = view_df.copy()
        current_brand_share = (
            full_view.groupby("brand", as_index=False)["revenue"].sum()
        )
        current_brand_share["share"] = (
            current_brand_share["revenue"] / current_brand_share["revenue"].sum()
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_trend_chart(
                    calculate_market_shares(
                        full_view,
                        engine.create_price_distribution_scenario(
                            st.session_state.model_result,
                            full_view,
                            st.session_state.spline,
                            st.session_state.scales,
                            0.0,
                            0.0,
                        ),
                        "brand",
                    ),
                    "baseline_share",
                    "Observed brand share trend",
                    "brand",
                ),
                use_container_width=True,
            )

        with col2:
            top = current_brand_share.nlargest(10, "share").sort_values("share")
            fig = go.Figure(
                go.Bar(
                    x=top["share"] * 100,
                    y=top["brand"],
                    orientation="h",
                    text=[f"{x:.1%}" for x in top["share"]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Current brand share",
                xaxis_title="Share (%)",
                yaxis_title="",
                height=420,
                margin=dict(l=20, r=60, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "The model estimates historical conditional associations. "
            "Scenario outputs should not be interpreted as proof of causal price or "
            "distribution effects."
        )

        with st.expander("Technical details"):
            st.latex(
                r"""
                \log(V^{std}_{r,s,t}) =
                \alpha + \eta_{r,s} + \mu_t +
                \beta_s \log(P^{rel}_{r,s,t}) +
                f(ND_{r,s,t}) + \varepsilon_{r,s,t}
                """
            )
            st.write(
                """
                • Relative price compares the focal SKU with the competitor basket.

                • Distribution is numeric distribution / store coverage.

                • The price response is hierarchical by SKU, with pooling across
                  brand and pack-size groups.

                • Distribution uses a non-linear spline effect.

                • The share simulator applies the selected change only to the target
                  while keeping the complete competitive market in the denominator.
                """
            )

        st.divider()

        st.subheader("Downloads")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.download_button(
                "Prepared data",
                st.session_state.prepared_data.to_csv(index=False),
                "prepared_data.csv",
                "text/csv",
                use_container_width=True,
            )
        with c2:
            if st.session_state.elasticities is not None:
                st.download_button(
                    "Price sensitivity",
                    st.session_state.elasticities.to_csv(index=False),
                    "price_sensitivity.csv",
                    "text/csv",
                    use_container_width=True,
                )
        with c3:
            qr = st.session_state.quality_report
            if qr is not None:
                st.download_button(
                    "Data quality",
                    json.dumps(qr, indent=2, default=str),
                    "data_quality.json",
                    "application/json",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()