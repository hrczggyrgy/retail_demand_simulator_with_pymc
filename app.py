
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
import matplotlib.pyplot as plt

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


def _model_config_from_mode(mode: str) -> engine.ModelConfig:
    if mode == "Fast (2 chains)":
        return engine.FAST_CONFIG
    elif mode == "Advanced (4 chains, Student-t)":
        return engine.ADVANCED_CONFIG
    return engine.DEFAULT_CONFIG


def fit_model_if_needed() -> None:
    if st.session_state.model_result is not None:
        return
    if st.session_state.prepared_data is None:
        return

    df = st.session_state.prepared_data

    config = _model_config_from_mode(st.session_state.get("model_mode", "Default (4 chains)"))

    with st.spinner("Estimating price and distribution response..."):
        if config.use_category_price_pooling or config.use_sku_nd_effects:
            model = engine.build_pymc_model_v2(df, st.session_state.meta, st.session_state.nd_basis, config)
        else:
            model = engine.build_pymc_model(df, st.session_state.meta, st.session_state.nd_basis)

        idata = engine.fit_model(model, config)

        # Add posterior predictive samples for PPC plots
        idata = engine.add_posterior_predictive(model, idata)

    st.session_state.model_result = idata
    st.session_state.fitted_model = model
    st.session_state.model_settings = {"config": config}
    st.session_state.model_diagnostics = engine.get_model_diagnostics(idata)
    st.session_state.elasticities = engine.extract_elasticities(
        idata,
        st.session_state.meta,
        st.session_state.scales,
    )

    # Extract posterior cache for fast scenarios
    st.session_state.posterior_cache = engine.extract_scenario_posterior(
        idata,
        max_draws=config.scenario_draws,
        random_seed=config.random_seed,
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
        st.subheader("2. Model configuration")

        model_mode = st.selectbox(
            "Model mode",
            ["Fast (2 chains)", "Default (4 chains)", "Advanced (4 chains, Student-t)"],
            index=1,
            help=(
                "Fast: quick exploratory fits. Default: balanced speed/quality. "
                "Advanced: full hierarchy with robust likelihood (slower)."
            ),
        )
        st.session_state.model_mode = model_mode

        st.divider()
        st.subheader("3. View filters")

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

    idata = st.session_state.model_result
    meta = st.session_state.meta
    spline = st.session_state.spline
    scales = st.session_state.scales
    posterior_cache = st.session_state.posterior_cache
    elasticity_df = st.session_state.elasticities
    diagnostics = st.session_state.model_diagnostics or {}

    tab_data, tab_model, tab_explorer, tab_scenario = st.tabs([
        "1. Data quality",
        "2. Model diagnostics",
        "3. Price & distribution",
        "4. Scenario builder",
    ])

    with tab_data:
        report = st.session_state.validation_report
        quality = st.session_state.quality_report

        if report is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{report.row_count:,}")
            c2.metric("Duplicates", f"{report.duplicate_count:,}")
            c3.metric("Invalid rows", f"{report.invalid_rows:,}")
            c4.metric("Valid", "Yes" if report.is_valid else "No")

        if quality is not None:
            st.divider()
            st.subheader("Market overview")

            c1, c2, c3, c4 = st.columns(4)
            counts = quality.get("counts", {})
            c1.metric("Retailers", counts.get("retailers", 0))
            c2.metric("Categories", counts.get("categories", 0))
            c3.metric("Brands", counts.get("brands", 0))
            c4.metric("SKUs", counts.get("skus", 0))

            date_cov = quality.get("date_coverage", {})
            if date_cov:
                st.caption(
                    f"Date range: {date_cov.get('min')} to {date_cov.get('max')} "
                    f"({date_cov.get('n_months', 0)} months)"
                )

            st.divider()
            st.subheader("Data quality flags")

            col1, col2 = st.columns(2)
            with col1:
                miss = quality.get("missing_values", {})
                if miss:
                    st.write("**Missing values**")
                    for k, v in miss.items():
                        st.write(f"• {k}: {v:,}")
                else:
                    st.write("**Missing values**: None")

            with col2:
                dist_val = quality.get("distribution_validity", {})
                if dist_val:
                    st.write("**Distribution validity**")
                    st.write(f"• ND range: [{dist_val.get('nd_min', 0):.3f}, {dist_val.get('nd_max', 0):.3f}]")
                    st.write(f"• ND > 1 count: {dist_val.get('nd_gt_1_count', 0)}")

            low_obs = quality.get("low_observation_skus", [])
            if low_obs:
                st.write(f"**Low-observation SKUs (< 6 months)**: {len(low_obs)}")
                with st.expander("Show SKUs"):
                    st.write(low_obs)

            if report.warnings:
                st.divider()
                st.write("**Warnings**")
                for w in report.warnings:
                    st.warning(w)

    with tab_model:
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Divergences", diagnostics.get("divergences", 0))
        with d2:
            st.metric("Max R-hat", f'{diagnostics.get("max_rhat", np.nan):.3f}')
        with d3:
            st.metric("Min bulk ESS", f'{diagnostics.get("min_ess_bulk", np.nan):.0f}')

        if diagnostics.get("is_usable"):
            st.success("Model diagnostics passed the configured checks.")
        else:
            for warning in diagnostics.get("warnings", []):
                st.warning(warning)

        st.divider()

        # Parameter group selector for trace/rank plots
        param_group = st.selectbox(
            "Diagnostic parameter group",
            [
                "Global parameters",
                "Price elasticities",
                "Distribution parameters",
            ],
            key="diag_param_group",
        )

        param_groups = {
            "Global parameters": [
                "alpha",
                "sigma_entity",
                "sigma_month",
                "sigma_price_sku",
                "sigma_nd",
                "sigma",
            ],
            "Price elasticities": ["price_slope_z"],
            "Distribution parameters": ["nd_coef"],
        }

        selected_var_names = param_groups[param_group]

        with st.expander("Trace plots", expanded=(param_group == "Global parameters")):
            try:
                fig = engine.make_trace_figure(idata, var_names=selected_var_names)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Trace plot failed: {e}")

        with st.expander("Rank plots", expanded=False):
            try:
                fig = engine.make_rank_figure(idata, var_names=selected_var_names)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Rank plot failed: {e}")

        with st.expander("Sampler energy", expanded=False):
            try:
                fig = engine.make_energy_figure(idata)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"Energy plot failed: {e}")

        with st.expander("Posterior predictive check", expanded=True):
            try:
                fig = engine.make_ppc_figure(idata)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except Exception as e:
                st.error(f"PPC plot failed: {e}. Run posterior predictive sampling first.")

    with tab_explorer:
        if elasticity_df is not None and posterior_cache is not None:
            # Filters
            cat_col, brand_col, sku_col = st.columns(3)

            categories = sorted(df["category"].dropna().unique())
            selected_category = cat_col.selectbox("Category", categories, key="explorer_cat")

            brands_in_cat = sorted(
                df.loc[df["category"] == selected_category, "brand"].dropna().unique()
            )
            selected_brand = brand_col.selectbox("Brand", brands_in_cat, key="explorer_brand")

            skus_in_brand = sorted(
                df.loc[
                    (df["category"] == selected_category) & (df["brand"] == selected_brand),
                    "sku",
                ].dropna().unique()
            )

            # Elasticity forest plot (ArviZ)
            st.subheader("Price elasticity (posterior forest plot)")

            brand_skus = [
                s for s in skus_in_brand if s in idata.posterior.coords.get("sku", {}).values
            ]

            if brand_skus:
                selected_skus_forest = st.multiselect(
                    "SKUs to compare",
                    options=brand_skus,
                    default=brand_skus[:min(8, len(brand_skus))],
                    key="explorer_skus_forest",
                )

                if selected_skus_forest:
                    try:
                        fig = engine.make_elasticity_forest_figure(
                            idata,
                            selected_skus=selected_skus_forest,
                            hdi_prob=0.90,
                        )
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Elasticity forest plot failed: {e}")

                st.caption(
                    "Intervals show 90% posterior credible intervals. "
                    "More negative own-price elasticities imply greater expected "
                    "volume sensitivity to relative price changes."
                )
            else:
                st.info("No SKUs available for forest plot in selected brand.")

            st.divider()

            # SKU-level elasticity table
            st.subheader("Price elasticity (raw scale)")
            sku_options = skus_in_brand
            selected_sku = sku_col.selectbox("SKU", sku_options, key="explorer_sku")

            sku_elasticity = elasticity_df[elasticity_df["sku"] == selected_sku]
            if not sku_elasticity.empty:
                st.dataframe(
                    sku_elasticity[
                        ["sku", "brand", "pack_group", "pack_size", "elasticity_p05", "elasticity_median", "elasticity_p95"]
                    ].style.format({
                        "elasticity_p05": "{:.3f}",
                        "elasticity_median": "{:.3f}",
                        "elasticity_p95": "{:.3f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            # ND response curve (using engine's build_nd_response_curve)
            st.divider()
            st.subheader("Distribution response curve")

            sku_index = meta["sku_levels"].tolist().index(selected_sku)

            try:
                nd_curve = engine.build_nd_response_curve(idata, spline, sku_index=sku_index)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=nd_curve["nd"],
                    y=nd_curve["nd_p90"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=nd_curve["nd"],
                    y=nd_curve["nd_p10"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(30, 136, 229, 0.18)",
                    name="80% credible interval",
                ))
                fig.add_trace(go.Scatter(
                    x=nd_curve["nd"],
                    y=nd_curve["nd_median"],
                    mode="lines",
                    line={"color": "#1E88E5", "width": 3},
                    name="Median expected multiplier",
                ))
                fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50% ND (ref)")
                fig.update_layout(
                    title=f"Distribution response: {selected_sku}",
                    xaxis_title="Numeric distribution",
                    yaxis_title="Volume multiplier vs minimum ND",
                    template="plotly_white",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"ND response curve failed: {e}")

            st.caption(
                "Shows how estimated velocity changes with distribution, holding price constant. "
                "Relative to 50% numeric distribution. Based on historical associations, not causal effects."
            )
        else:
            st.info("Fit the model first to explore price and distribution responses.")

    with tab_scenario:
        if posterior_cache is None:
            st.info("Fit the model first to run scenarios.")
        else:
            st.subheader("Build your what-if scenario")

            # Scenario type
            scenario_type = st.radio(
                "Scenario type",
                ["Price only", "Distribution only", "Combined"],
                horizontal=True,
                help="Price only: change relative price. Distribution only: change ND. Combined: both."
            )

            # Target definition
            st.markdown("### Target")
            target_level = st.radio(
                "Apply to",
                ["Market", "Retailer", "Brand", "SKU"],
                horizontal=True,
                index=2,
            )

            target_value = None
            if target_level == "Retailer":
                target_value = st.selectbox("Choose retailer", unique_sorted(view_df, "retailer"))
            elif target_level == "Brand":
                target_value = st.selectbox("Choose brand", unique_sorted(view_df, "brand"))
            elif target_level == "SKU":
                sku_options = (
                    view_df[["brand", "sku"]]
                    .drop_duplicates()
                    .sort_values(["brand", "sku"])
                )
                sku_labels = [f"{r.brand} | {r.sku}" for r in sku_options.itertuples()]
                selected = st.selectbox("Choose SKU", sku_labels)
                target_value = selected.split(" | ", 1)[1]

            # Price control
            price_change = 0.0
            if scenario_type in ["Price only", "Combined"]:
                st.divider()
                st.markdown("### Price change")
                price_change_pct = st.slider(
                    "Price change",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=1,
                    format="%d%%",
                )
                price_change = price_change_pct / 100.0

            # Distribution control
            nd_new = None
            nd_mode = "pp"
            if scenario_type in ["Distribution only", "Combined"]:
                st.divider()
                st.markdown("### Distribution change")
                nd_mode = st.radio(
                    "Input mode",
                    ["Percentage points", "Relative change", "Target ND"],
                    horizontal=True,
                )

                if nd_mode == "Percentage points":
                    nd_change_pp = st.slider(
                        "ND change (pp)",
                        min_value=-30,
                        max_value=30,
                        value=0,
                        step=1,
                    )
                    nd_change_val = nd_change_pp / 100.0
                elif nd_mode == "Relative change":
                    nd_rel = st.slider(
                        "ND relative change",
                        min_value=-50,
                        max_value=100,
                        value=0,
                        step=1,
                        format="%d%%",
                    )
                    nd_change_val = nd_rel / 100.0
                else:
                    nd_target = st.slider(
                        "Target ND",
                        min_value=0.01,
                        max_value=1.00,
                        value=0.50,
                        step=0.01,
                    )
                    nd_change_val = nd_target

            if st.button("Run scenario", type="primary", use_container_width=True):
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
                    with st.spinner("Calculating scenario..."):
                        nd_base = market_scope_df["nd"].to_numpy()
                        if scenario_type in ["Distribution only", "Combined"]:
                            if nd_mode == "Percentage points":
                                resolved_nd = engine.resolve_target_nd(nd_base, nd_change_val, "pp")
                            elif nd_mode == "Relative change":
                                resolved_nd = engine.resolve_target_nd(nd_base, nd_change_val, "relative")
                            else:
                                resolved_nd = engine.resolve_target_nd(nd_base, nd_change_val, "absolute")
                        else:
                            resolved_nd = nd_base

                        target_mask = engine.apply_target_scope(market_scope_df, target_level, target_value)

                        price_effect = np.zeros((posterior_cache["price_slope_z"].shape[0], len(market_scope_df)))
                        if scenario_type in ["Price only", "Combined"] and price_change != 0:
                            price_effect = engine.price_delta_log_volume(
                                posterior_cache["price_slope_z"],
                                market_scope_df["sku_idx"].to_numpy(dtype="int32"),
                                price_change,
                                target_mask.to_numpy(),
                            )

                        nd_effect = np.zeros((posterior_cache["price_slope_z"].shape[0], len(market_scope_df)))
                        if scenario_type in ["Distribution only", "Combined"]:
                            nd_effect = engine.nd_delta_log_volume(
                                posterior_cache,
                                spline,
                                scales,
                                market_scope_df["sku_idx"].to_numpy(dtype="int32"),
                                nd_base,
                                resolved_nd,
                                target_mask.to_numpy(),
                            )

                        total_effect = engine.combined_delta_log_volume(price_effect, nd_effect)

                        result = engine.summarise_scenario_draws(
                            market_scope_df["units"].to_numpy(),
                            market_scope_df["revenue"].to_numpy(),
                            price_change,
                            total_effect,
                        )

                        for col in result.columns:
                            market_scope_df[col] = result[col]

                        share_df = engine.calculate_market_shares(market_scope_df, market_scope_df, "brand")
                        sku_share_df = engine.calculate_market_shares(market_scope_df, market_scope_df, "sku")

                        st.session_state.scenario_df = market_scope_df
                        st.session_state.scenario_shares = share_df
                        st.session_state.scenario_sku_shares = sku_share_df
                        st.session_state.scenario_settings = {
                            "type": scenario_type,
                            "target_level": target_level,
                            "target_value": target_value,
                            "price_change": price_change,
                            "nd_mode": nd_mode,
                            "nd_change": nd_change_val if 'nd_change_val' in locals() else 0,
                            "resolved_nd": resolved_nd if 'resolved_nd' in locals() else None,
                        }

            if "scenario_shares" in st.session_state and st.session_state.scenario_shares is not None:
                share_df = st.session_state.scenario_shares
                settings = st.session_state.scenario_settings or {}

                st.divider()
                st.subheader("Scenario results")

                target_level = settings.get("target_level")
                target_value = settings.get("target_value")
                price_change = settings.get("price_change", 0.0)

                target_label = (
                    "Entire market"
                    if target_level == "Market"
                    else f"{target_level}: {target_value}"
                )

                st.caption(
                    f"**Target**: {target_label}  |  "
                    f"**Price**: {price_change:+.0%}  |  "
                    f"**Type**: {settings.get('type', 'Combined')}"
                )

                total_base = share_df["baseline_value"].sum()
                total_scen = share_df["scenario_value"].sum()
                total_rev_change = total_scen / total_base - 1 if total_base > 0 else np.nan

                a, b, c, d = st.columns(4)
                with a:
                    st.metric("Market value change", f"{total_rev_change:+.1%}")
                with b:
                    if target_level in ["Brand", "SKU"] and target_value:
                        target_rows = share_df[share_df[target_level.lower()] == target_value]
                        if not target_rows.empty:
                            ts_change = target_rows["share_change_pp"].sum()
                            st.metric("Target share change", f"{ts_change * 100:+.1f} pp")
                with c:
                    scenario_df = st.session_state.scenario_df
                    target_mask = engine.apply_target_scope(scenario_df, target_level, target_value)
                    if target_mask.any():
                        bu = scenario_df.loc[target_mask, "units"].sum()
                        su = scenario_df.loc[target_mask, "units_median"].sum()
                        st.metric("Target units change", f"{(su/bu-1)*100:+.1f}%" if bu > 0 else "—")
                with d:
                    if target_mask.any():
                        br = scenario_df.loc[target_mask, "revenue"].sum()
                        sr = scenario_df.loc[target_mask, "revenue_median"].sum()
                        st.metric("Target revenue change", f"{(sr/br-1)*100:+.1f}%" if br > 0 else "—")

                st.divider()

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        engine.create_share_change_chart(share_df),
                        use_container_width=True,
                    )
                with col2:
                    st.plotly_chart(
                        engine.create_share_comparison_chart(share_df),
                        use_container_width=True,
                    )

                # Detailed table
                st.markdown("### Share changes by brand")
                display = share_df.sort_values("share_change_pp", ascending=False)[
                    ["brand", "baseline_share", "scenario_share", "share_change_pp"]
                ].copy()
                display = display.rename(columns={
                    "brand": "Brand",
                    "baseline_share": "Baseline share",
                    "scenario_share": "Scenario share",
                    "share_change_pp": "Change (pp)",
                })
                st.dataframe(
                    display.style.format({
                        "Baseline share": "{:.1%}",
                        "Scenario share": "{:.1%}",
                        "Change (pp)": lambda x: f"{x * 100:+.1f} pp",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                if "scenario_sku_shares" in st.session_state:
                    with st.expander("SKU-level share changes"):
                        sku_share_df = st.session_state.scenario_sku_shares
                        sku_display = sku_share_df.sort_values("share_change_pp", ascending=False)[
                            ["brand", "sku", "baseline_share", "scenario_share", "share_change_pp"]
                        ].copy()
                        sku_display = sku_display.rename(columns={
                            "brand": "Brand",
                            "sku": "SKU",
                            "baseline_share": "Baseline share",
                            "scenario_share": "Scenario share",
                            "share_change_pp": "Change (pp)",
                        })
                        st.dataframe(
                            sku_display.style.format({
                                "Baseline share": "{:.1%}",
                                "Scenario share": "{:.1%}",
                                "Change (pp)": lambda x: f"{x * 100:+.1f} pp",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

                st.divider()
                st.subheader("Growth opportunities")
                opportunities = engine.get_growth_opportunities(df, elasticity_df)
                opp_display = opportunities.head(20)[
                    ["retailer", "category", "brand", "sku", "pack_group",
                     "unit_sales", "revenue", "avg_nd", "avg_velocity_std",
                     "distribution_headroom", "velocity_index",
                     "distribution_opportunity_score", "price_opportunity_score",
                     "primary_lever"]
                ].copy()
                opp_display = opp_display.rename(columns={
                    "retailer": "Retailer", "category": "Category", "brand": "Brand",
                    "sku": "SKU", "pack_group": "Pack group",
                    "unit_sales": "Units", "revenue": "Revenue",
                    "avg_nd": "Avg ND", "avg_velocity_std": "Avg velocity (std)",
                    "distribution_headroom": "Dist. headroom", "velocity_index": "Velocity index",
                    "distribution_opportunity_score": "Dist. opp. score",
                    "price_opportunity_score": "Price opp. score",
                    "primary_lever": "Primary lever",
                })
                st.dataframe(
                    opp_display.style.format({
                        "Revenue": "{:,.0f}", "Units": "{:,.0f}",
                        "Avg ND": "{:.1%}", "Avg velocity (std)": "{:.2f}",
                        "Dist. headroom": "{:.1%}", "Velocity index": "{:.2f}",
                        "Dist. opp. score": "{:.1f}", "Price opp. score": "{:.1f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "**Interpretation**: Historical model-based estimates. Not causal guarantees. "
                    "Distribution opportunity: high velocity + low ND + large revenue. "
                    "Price opportunity: high elasticity + revenue, adjusted for uncertainty."
                )


if __name__ == "__main__":
    main()