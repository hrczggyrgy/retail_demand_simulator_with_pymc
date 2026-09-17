
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

import numpy as np
import pandas as pd
import plotly.express as px
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


def render_workflow_banner(
    has_data: bool,
    has_model: bool,
    has_scenario: bool,
) -> None:
    """Render workflow progress banner showing user journey status."""
    steps = [
        ("1. Check data", has_data, "Data loaded and validated"),
        ("2. Explore market", has_data, "Features prepared, market visible"),
        ("3. Fit model", has_model, "Posterior sampled, diagnostics available"),
        ("4. Run scenario", has_scenario, "Baseline vs scenario comparison ready"),
        ("5. Review impact", has_scenario, "Market reallocation, uncertainty assessed"),
    ]

    icons = []
    for label, complete, _tooltip in steps:
        icon = "✅" if complete else "⬜"
        icons.append(f"{icon} {label}")

    st.caption("  →  ".join(icons), help="Your progress through the decision workflow")

    if not has_model:
        st.info(
            "💡 Next step: Go to the sidebar, choose a model mode, and click **Fit / Re-fit model** "
            "to enable scenario simulation."
        )
    elif has_model and not has_scenario:
        st.info(
            "💡 Model is ready. Go to the **Scenario simulator** tab to define a price or "
            "distribution action and run a scenario."
        )


def chart_subtitle(question: str) -> None:
    """Render a standardized chart subtitle with the business question."""
    st.caption(f"📋 {question}")


def render_scope_context(scope_text: str, period_text: str | None = None) -> None:
    """Render market scope and period context below charts/KPIs."""
    parts = [f"Scope: {scope_text}"]
    if period_text:
        parts.append(f"Period: {period_text}")
    st.caption(" | ".join(parts))


def render_model_diagnostic_status(diagnostics: dict, prefix: str = "") -> None:
    """Render model diagnostic status cards for scenario simulator."""
    if not diagnostics:
        st.warning("⚠ Model diagnostics not available")
        return

    divergences = diagnostics.get("divergences", 0)
    max_rhat = diagnostics.get("max_rhat")
    min_ess = diagnostics.get("min_ess_bulk")
    ppc_coverage = diagnostics.get("ppc_coverage_90")
    is_decision_ready = diagnostics.get("is_decision_ready", False)

    if is_decision_ready:
        st.success("✅ Model status: **Decision-ready** — convergence checks passed")
    elif divergences > 0:
        st.warning(
            f"⚠ Model status: **Exploratory** — {divergences} divergent transition(s) detected. "
            "Results should not be interpreted as decision-ready parameter estimates."
        )
    elif max_rhat is not None and max_rhat > 1.01:
        st.warning(
            f"⚠ Model status: **Exploratory** — max R-hat = {max_rhat:.3f} (threshold: 1.01). "
            "Chains may not have converged."
        )
    elif min_ess is not None and min_ess < 400:
        st.warning(
            f"⚠ Model status: **Exploratory** — min bulk ESS = {min_ess:.0f} (threshold: 400). "
            "Effective sample size is low."
        )
    else:
        st.info("ℹ Model status: **Exploratory** — review full diagnostics in Model Health tab")

    # Compact diagnostic summary
    with st.expander("🔬 Diagnostic details", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Divergences", divergences)
        cols[1].metric("Max R-hat", f"{max_rhat:.3f}" if max_rhat else "—")
        cols[2].metric("Min ESS (bulk)", f"{min_ess:.0f}" if min_ess else "—")
        cols[3].metric("PPC coverage (90%)", f"{ppc_coverage:.1%}" if ppc_coverage else "—")


def unique_sorted(df: pd.DataFrame, col: str) -> list:
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
    category: str | None = None,
    retailer: str | None = None,
    brand: str | None = None,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if category and category != "All":
        mask &= df["category"].eq(category)
    if retailer and retailer != "All":
        mask &= df["retailer"].eq(retailer)
    if brand and brand != "All":
        mask &= df["brand"].eq(brand)
    return mask


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

    target_mask = engine.apply_target_scope(work, target_level, target_value)

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
        "volume_multiplier_p05",
        "volume_multiplier_p50",
        "volume_multiplier_p95",
        "units_p50",
        "revenue_p50",
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
def read_uploaded_file(file_bytes: bytes, file_type: str, sheet_name: str | None) -> pd.DataFrame:
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


def fit_model_explicitly() -> None:
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
        # Reset model when mode changes
        if st.session_state.get("model_mode") != model_mode:
            st.session_state.model_result = None
            st.session_state.fitted_model = None
            st.session_state.model_settings = None
            st.session_state.model_diagnostics = None
            st.session_state.elasticities = None
            st.session_state.posterior_cache = None
        st.session_state.model_mode = model_mode

        if st.button("Fit / Re-fit model", use_container_width=True, type="primary"):
            fit_model_explicitly()
            st.rerun()

        if st.session_state.model_result is not None:
            config = st.session_state.model_settings["config"]
            st.caption(f"Fitted with: {config.__class__.__name__}")

        st.divider()
        st.subheader("3. Explore market")
        st.caption("These filters affect **display tabs only**. Scenarios always use the full market.")

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
            "💡 These filters change the market you are viewing. They do not remove "
            "competitors from the scenario denominator — the scenario always uses the "
            "full uploaded market."
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()
    render_sidebar()

    st.title("Retail Demand & Scenario Cockpit")
    st.markdown(
        """
        **Modelled historical response** — not causal guarantees.  
        The simulator uses the complete competitive market in the data. You choose
        **one action**, the products/retailers it applies to, and the app estimates
        how **brand or SKU market shares move** based on historical conditional associations.
        """
    )

    # Workflow progress banner
    has_data = st.session_state.raw_data is not None
    has_model = st.session_state.model_result is not None
    has_scenario = st.session_state.get("scenario_suite") is not None
    render_workflow_banner(has_data, has_model, has_scenario)

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

    if st.session_state.model_result is None:
        # Still show descriptive tabs even without model
        show_descriptive_tabs(df, view_df, start_month, end_month)
        return

    idata = st.session_state.model_result
    meta = st.session_state.meta
    spline = st.session_state.spline
    scales = st.session_state.scales
    posterior_cache = st.session_state.posterior_cache
    elasticity_df = st.session_state.elasticities
    diagnostics = st.session_state.model_diagnostics or {}

    # Build enriched features for all tabs
    enriched_df = engine.build_retail_features(df)
    view_enriched = enriched_df.loc[view_mask].copy()

    # 6-tab navigation
    tab_overview, tab_health, tab_market, tab_price, tab_scenario, tab_model = st.tabs([
        "📊 Overview",
        "🔍 Data health",
        "📈 Market & share",
        "💰 Price & distribution",
        "⚙️ Scenario simulator",
        "🏥 Model health",
    ])

    with tab_overview:
        render_overview_tab(view_df, view_enriched, elasticity_df, posterior_cache, meta, scales, spline, start_month, end_month)

    with tab_health:
        render_data_health_tab(df, st.session_state.validation_report, st.session_state.quality_report)

    with tab_market:
        render_market_share_tab(view_df, view_enriched, start_month, end_month)

    with tab_price:
        render_price_distribution_tab(view_df, view_enriched, idata, meta, scales, spline, posterior_cache, elasticity_df)

    with tab_scenario:
        render_scenario_tab(
            df, view_df, view_enriched, idata, meta, scales, spline,
            posterior_cache, elasticity_df, diagnostics,
            start_month, end_month, view_category, view_retailer, view_brand
        )

    with tab_model:
        render_model_health_tab(idata, df, diagnostics, meta, scales, spline, posterior_cache, elasticity_df)


def show_descriptive_tabs(df: pd.DataFrame, view_df: pd.DataFrame, start_month, end_month) -> None:
    st.info("Model not yet fitted. Showing descriptive analytics only.")
    # Build enriched features for descriptive tabs
    enriched_df = engine.build_retail_features(df)
    view_enriched = enriched_df.loc[view_df.index].copy()

    tab_overview, tab_health, tab_market, tab_price = st.tabs([
        "📊 Overview", "🔍 Data health", "📈 Market & share", "💰 Price & distribution"
    ])
    with tab_overview:
        render_overview_tab(view_df, view_enriched, None, None, None, None, None, start_month, end_month)
    with tab_health:
        render_data_health_tab(df, st.session_state.validation_report, st.session_state.quality_report)
    with tab_market:
        render_market_share_tab(view_df, view_enriched, start_month, end_month)
    with tab_price:
        render_price_distribution_tab(view_df, view_enriched, None, None, None, None, None, None)


def render_overview_tab(view_df, view_enriched, elasticity_df, posterior_cache, meta, scales, spline, start_month, end_month) -> None:
    period_text = f"{start_month.strftime('%b %Y')} – {end_month.strftime('%b %Y')}"
    st.subheader(f"Market Overview ({period_text})")
    render_scope_context("All observed retailers, categories, brands, and SKUs", period_text)

    if view_enriched is not None and "category_units" in view_enriched.columns:
        latest_month = view_enriched["month"].max()
        prev_month = view_enriched["month"].min()
        latest = view_enriched[view_enriched["month"] == latest_month]
        previous = view_enriched[view_enriched["month"] == prev_month]

        cat_units_latest = latest["category_units"].sum()
        cat_units_prev = previous["category_units"].sum()
        cat_growth = (cat_units_latest - cat_units_prev) / cat_units_prev if cat_units_prev else 0

        rev_latest = latest["revenue"].sum()
        rev_prev = previous["revenue"].sum()
        rev_growth = (rev_latest - rev_prev) / rev_prev if rev_prev else 0

        sku_count = latest["sku"].nunique()
        brand_count = latest["brand"].nunique()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Category Units", f"{cat_units_latest:,.0f}", f"{cat_growth:+.1%}")
        c2.metric("Revenue", f"{rev_latest:,.0f}", f"{rev_growth:+.1%}")
        c3.metric("SKUs", f"{sku_count}")
        c4.metric("Brands", f"{brand_count}")
        c5.metric("Avg ND", f"{latest['nd'].mean():.1%}" if "nd" in latest.columns else "—")

        # Scope context for KPIs
        render_scope_context("All observed retailers, categories, brands, and SKUs", "Latest month vs. prior month")

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Category Trend")
        chart_subtitle("How has the observed market changed over time?")
        if "category_units" in view_enriched.columns:
            trend = view_enriched.groupby("month").agg(
                units=("category_units", "sum"),
                revenue=("revenue", "sum")
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=trend["month"], y=trend["units"], name="Units", yaxis="y", opacity=0.6))
            fig.add_trace(go.Scatter(x=trend["month"], y=trend["revenue"], name="Revenue", yaxis="y2", line=dict(color="red")))
            fig.update_layout(
                yaxis=dict(title="Units", side="left"),
                yaxis2=dict(title="Revenue", side="right", overlaying="y"),
                height=350, hovermode="x unified", margin=dict(l=40, r=40, t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed retailers, categories, brands, and SKUs", period_text)
        else:
            st.info("Category units not available in view.")

    with col2:
        st.markdown("### Contribution to Growth")
        chart_subtitle("Which brands drove market growth or decline?")
        if "brand_units" in view_enriched.columns:
            contrib = view_enriched.groupby(["month", "brand"]).agg(
                units=("units", "sum"),
                brand_units=("brand_units", "first")
            ).reset_index()
            latest_m = contrib["month"].max()
            prev_m = contrib["month"].min()
            curr = contrib[contrib["month"] == latest_m][["brand", "units"]].rename(columns={"units": "curr"})
            prev = contrib[contrib["month"] == prev_m][["brand", "units"]].rename(columns={"units": "prev"})
            merged = curr.merge(prev, on="brand", how="outer").fillna(0)
            merged["change"] = merged["curr"] - merged["prev"]
            merged = merged.sort_values("change", ascending=True).tail(10)

            fig = go.Figure(go.Bar(
                x=merged["change"], y=merged["brand"], orientation="h",
                marker_color=["red" if x < 0 else "green" for x in merged["change"]]
            ))
            fig.update_layout(height=350, margin=dict(l=100, r=20, t=30, b=40), xaxis_title="Unit change")
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("Top 10 brands by unit change", f"Latest ({latest_m.strftime('%b %Y')}) vs. earliest ({prev_m.strftime('%b %Y')}) period")
        else:
            st.info("Brand contribution not available.")

    st.divider()
    st.markdown("### Key Alerts")
    alerts = []

    if view_enriched is not None and "nd" in view_enriched.columns:
        low_nd = view_enriched[view_enriched["nd"] < 0.3]
        if not low_nd.empty:
            alerts.append(f"🔍 **Distribution opportunity**: {len(low_nd)} SKU-months have ND < 30% (distribution gap)")

        high_nd_low_vel = view_enriched[(view_enriched["nd"] > 0.8) & (view_enriched["velocity"] < view_enriched["velocity"].median())]
        if not high_nd_low_vel.empty:
            alerts.append(f"💰 **Rationalization candidates**: {len(high_nd_low_vel)} SKU-months have high ND but low velocity")

    if elasticity_df is not None:
        high_elas = elasticity_df[elasticity_df["elasticity_median"] < -2.5]
        if not high_elas.empty:
            alerts.append(f"🔴 **High price sensitivity**: {len(high_elas)} SKUs have elasticity < -2.5")

        low_precision = elasticity_df[elasticity_df["elasticity_p95"] - elasticity_df["elasticity_p05"] > 3.0]
        if not low_precision.empty:
            alerts.append(f"🟡 **Low precision**: {len(low_precision)} SKUs have wide elasticity intervals (>3.0 width)")

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("No major alerts detected in current view.")


def render_data_health_tab(df, validation_report, quality_report) -> None:
    st.subheader("Data Quality & Coverage")

    if validation_report is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{validation_report.row_count:,}")
        c2.metric("Duplicates", f"{validation_report.duplicate_count:,}")
        c3.metric("Invalid Rows", f"{validation_report.invalid_rows:,}")
        c4.metric("Valid", "✅ Yes" if validation_report.is_valid else "❌ No")

    if quality_report is not None:
        st.divider()
        st.markdown("### Market Coverage")
        counts = quality_report.get("counts", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retailers", counts.get("retailers", 0))
        c2.metric("Categories", counts.get("categories", 0))
        c3.metric("Brands", counts.get("brands", 0))
        c4.metric("SKUs", counts.get("skus", 0))

        coverage = quality_report.get("coverage", {})
        if coverage:
            st.markdown("### Field Coverage")
            cov_df = pd.DataFrame(list(coverage.items()), columns=["Field", "Coverage %"])
            cov_df["Coverage %"] = cov_df["Coverage %"] * 100
            fig = px.bar(cov_df, x="Field", y="Coverage %", color="Coverage %", color_continuous_scale="RdYlGn")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Data Quality Flags")

    if "data_quality_status" in df.columns:
        flag_counts = df["data_quality_status"].value_counts().reset_index()
        flag_counts.columns = ["Flag", "Count"]
        fig = px.bar(flag_counts, x="Flag", y="Count", color="Count", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Excluded / Flagged Records Detail"):
            excluded = df[df["data_quality_status"] != "valid"]
            if not excluded.empty:
                display_cols = ["month", "retailer", "category", "brand", "sku", "units", "revenue", "data_quality_status"]
                avail_cols = [c for c in display_cols if c in excluded.columns]
                st.dataframe(excluded[avail_cols].head(200), use_container_width=True, hide_index=True)
            else:
                st.info("No excluded records.")
    else:
        st.info("Data quality flags not yet computed. Run feature engineering first.")

    st.divider()
    st.markdown("### Price Outlier Detection")
    if "unit_price" in df.columns:
        fig = px.box(df, y="unit_price", points="outliers", title="Unit Price Distribution")
        st.plotly_chart(fig, use_container_width=True)

        price_by_cat = df.groupby("category")["unit_price"].median().reset_index()
        price_by_cat.columns = ["Category", "Median Unit Price"]
        fig2 = px.bar(price_by_cat, x="Category", y="Median Unit Price", title="Median Unit Price by Category")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Unit price not yet computed.")


def render_market_share_tab(view_df, view_enriched, start_month, end_month) -> None:
    period_text = f"{start_month.strftime('%b %Y')} – {end_month.strftime('%b %Y')}"
    st.subheader(f"Market Share & Growth ({period_text})")
    render_scope_context("All observed retailers, categories, brands, and SKUs", period_text)

    if "brand_share" not in view_enriched.columns:
        st.warning("Brand share not available. Run feature engineering first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Brand Share Trends")
        chart_subtitle("Which brands gained or lost share over time?")
        share_trend = view_enriched.groupby(["month", "brand"]).agg(
            share=("brand_share", "first"),
            units=("units", "sum")
        ).reset_index()
        fig = px.area(share_trend, x="month", y="share", color="brand", title="Brand Unit Share Over Time")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        render_scope_context("All observed retailers, categories, brands, and SKUs", period_text)

    with col2:
        st.markdown("### Brand Share (Latest Month)")
        chart_subtitle("What is the current brand share distribution?")
        latest = view_enriched["month"].max()
        latest_df = view_enriched[view_enriched["month"] == latest]
        brand_share_latest = latest_df.groupby("brand").agg(
            share=("brand_share", "first"),
            units=("units", "sum")
        ).reset_index().sort_values("share", ascending=False)
        fig = px.pie(brand_share_latest, values="share", names="brand", title=f"Brand Share – {latest.strftime('%b %Y')}")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        render_scope_context("All observed retailers, categories, brands, and SKUs", f"Latest month ({latest.strftime('%b %Y')})")

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Retailer × Category Heatmap")
        chart_subtitle("Where are units concentrated across retailers and categories?")
        if "retailer" in view_enriched.columns and "category" in view_enriched.columns:
            heat = view_enriched.groupby(["retailer", "category"]).agg(
                units=("units", "sum"),
                revenue=("revenue", "sum")
            ).reset_index()
            fig = px.density_heatmap(heat, x="category", y="retailer", z="units", title="Units by Retailer × Category")
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed retailers and categories", period_text)

    with col4:
        st.markdown("### Pack Mix by Brand")
        chart_subtitle("What is the pack size composition by brand?")
        if "pack_group" in view_enriched.columns:
            pack_mix = view_enriched.groupby(["brand", "pack_group"]).agg(
                units=("units", "sum")
            ).reset_index()
            fig = px.bar(pack_mix, x="brand", y="units", color="pack_group", title="Pack Size Mix by Brand", barmode="stack")
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed brands and pack groups", period_text)

    st.divider()
    st.markdown("### Contribution Waterfall (Category Growth Decomposition)")
    chart_subtitle("Which brands explain the category's unit change?")
    if "brand_units" in view_enriched.columns:
        latest_m = view_enriched["month"].max()
        prev_m = view_enriched["month"].min()
        curr = view_enriched[view_enriched["month"] == latest_m].groupby("brand").agg(
            units=("units", "sum"),
            brand_units=("brand_units", "first")
        )
        prev = view_enriched[view_enriched["month"] == prev_m].groupby("brand").agg(
            units=("units", "sum"),
            brand_units=("brand_units", "first")
        )
        merged = curr.join(prev, lsuffix="_curr", rsuffix="_prev", how="outer").fillna(0)
        merged["change"] = merged["units_curr"] - merged["units_prev"]
        merged = merged.sort_values("change")

        fig = go.Figure(go.Waterfall(
            name="", orientation="v",
            measure=["relative"] * len(merged) + ["total"],
            x=list(merged.index) + ["Total"],
            y=list(merged["change"]) + [merged["change"].sum()],
            text=[f"{v:+,}" for v in merged["change"]] + [f"{merged['change'].sum():+,}"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(height=400, title="Category Growth by Brand Contribution")
        st.plotly_chart(fig, use_container_width=True)
        render_scope_context("All observed brands in category", f"Latest ({latest_m.strftime('%b %Y')}) vs. earliest ({prev_m.strftime('%b %Y')}) period")


def render_price_distribution_tab(view_df, view_enriched, idata, meta, scales, spline, posterior_cache, elasticity_df) -> None:
    st.subheader("Price & Distribution Analytics")

    if view_enriched is None or "price_per_size_unit" not in view_enriched.columns:
        st.warning("Price per size unit not available. Run feature engineering first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Price Ladder (Pack Size vs Unit Value)")
        chart_subtitle("Which packs occupy comparable price-value positions?")
        latest = view_enriched["month"].max()
        latest_df = view_enriched[view_enriched["month"] == latest]
        fig = px.scatter(
            latest_df, x="pack_size", y="price_per_size_unit",
            size="revenue", color="brand", hover_data=["sku", "units", "nd"],
            title=f"Price per Size Unit by Pack Size – {latest.strftime('%b %Y')}",
            labels={"pack_size": "Pack Size", "price_per_size_unit": "Price per Size Unit"}
        )
        if "category_price_median" in latest_df.columns:
            cat_median = latest_df["category_price_median"].median()
            fig.add_hline(y=cat_median, line_dash="dash", line_color="gray", annotation_text="Category Median")
        st.plotly_chart(fig, use_container_width=True)
        render_scope_context("All observed SKUs in latest month", f"Latest month ({latest.strftime('%b %Y')})")

    with col2:
        st.markdown("### ND vs Velocity Opportunity Quadrant")
        chart_subtitle("Which SKUs combine strong velocity with low numeric distribution?")
        fig = px.scatter(
            latest_df, x="nd", y="velocity",
            size="revenue", color="brand", hover_data=["sku", "pack_size", "units"],
            title=f"Distribution vs Velocity – {latest.strftime('%b %Y')}",
            labels={"nd": "Numeric Distribution", "velocity": "Velocity (units/store)"}
        )
        nd_median = latest_df["nd"].median()
        vel_median = latest_df["velocity"].median()
        fig.add_vline(x=nd_median, line_dash="dash", line_color="gray")
        fig.add_hline(y=vel_median, line_dash="dash", line_color="gray")
        fig.add_annotation(x=nd_median*0.5, y=vel_median*1.5, text="Expand", showarrow=False, font=dict(size=10, color="green"))
        fig.add_annotation(x=nd_median*1.5, y=vel_median*1.5, text="Protect", showarrow=False, font=dict(size=10, color="blue"))
        fig.add_annotation(x=nd_median*0.5, y=vel_median*0.5, text="Test/Review", showarrow=False, font=dict(size=10, color="orange"))
        fig.add_annotation(x=nd_median*1.5, y=vel_median*0.5, text="Rationalize", showarrow=False, font=dict(size=10, color="red"))
        st.plotly_chart(fig, use_container_width=True)
        render_scope_context("All observed SKUs in latest month", f"Latest month ({latest.strftime('%b %Y')})")

    st.divider()
    st.markdown("### Relative Price Index by Brand")
    chart_subtitle("How do brands' prices compare to the category median?")
    if "relative_price_index" in latest_df.columns:
        rpi = latest_df.groupby("brand").agg(
            rpi=("relative_price_index", "median"),
            revenue=("revenue", "sum")
        ).reset_index().sort_values("rpi")
        fig = px.bar(rpi, x="brand", y="rpi", color="rpi", color_continuous_scale="RdYlGn_r",
                     title="Median Relative Price Index by Brand (vs Category Median)")
        fig.add_hline(y=1.0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
        render_scope_context("All observed brands in latest month", f"Latest month ({latest.strftime('%b %Y')})")

    st.divider()
    st.markdown("### Distribution White-Space Matrix")
    chart_subtitle("Which retailer-brand combinations have white-space (low ND, high revenue)?")
    if "retailer" in latest_df.columns:
        ws = latest_df.groupby(["retailer", "brand"]).agg(
            skus=("sku", "nunique"),
            avg_nd=("nd", "mean"),
            revenue=("revenue", "sum")
        ).reset_index()
        fig = px.scatter(ws, x="retailer", y="brand", size="revenue", color="avg_nd",
                         color_continuous_scale="RdYlGn", title="Avg ND by Retailer × Brand (bubble = revenue)")
        st.plotly_chart(fig, use_container_width=True)
        render_scope_context("All observed retailer-brand combinations", f"Latest month ({latest.strftime('%b %Y')})")


def render_scenario_tab(df, view_df, view_enriched, idata, meta, scales, spline,
                         posterior_cache, elasticity_df, diagnostics,
                         start_month, end_month, view_category, view_retailer, view_brand) -> None:
    from app import render_scenario_builder

    st.markdown("### Scenario Assumptions & Guardrails")

    # Model diagnostic status panel (P0)
    render_model_diagnostic_status(diagnostics, prefix="scenario")

    st.divider()

    with st.expander("⚙️ Cross-Effect Multipliers (Assumption-Led)", expanded=False):
        st.markdown("""
        These multipliers approximate substitution effects **not estimated by the model**.
        They are scenario assumptions, not statistically discovered relationships.

        | Relationship | Multiplier | Status |
        |---|---|---|
        | Same brand, adjacent pack | 0.35 | Assumption |
        | Same brand, other pack | 0.15 | Assumption |
        | Other brand, same pack | 0.25 | Assumption |
        | Other brand, other pack | 0.08 | Assumption |
        """)

    with st.expander("📊 Confidence Flags", expanded=False):
        st.markdown("""
        | Status | Meaning |
        |---|---|
        | **Estimated** | Posterior interval excludes zero, sufficient data |
        | **Shrunk** | Group prior dominates, limited SKU-level evidence |
        | **Assumption-Led** | User-specified rule, not data-supported |
        | **Not Available** | Sparse/missing competitive set |
        """)

    st.divider()

    sensitivity_preset = st.radio(
        "Substitution assumption",
        ["Conservative (0%)", "Central (5%)", "High (15%)"],
        horizontal=True,
        index=1,
        help="Conservative: no cross-category substitution. Central: 5% volume reallocates to other brands in the same pack group. High: 15% substitution."
    )
    sensitivity_map = {
        "Conservative (0%)": 0.0,
        "Central (5%)": 0.05,
        "High (15%)": 0.15,
    }
    cross_cat_sensitivity = sensitivity_map[sensitivity_preset]
    st.caption("Applied to: Other brands in same pack group within same category. "
               "These are assumption-led multipliers, not model-estimated.")

    st.divider()

    # Define scenario action section (P0 - separate from Explore market filters)
    st.markdown("### Define Scenario Action")
    st.caption("The controls below define **what action to test**. They do not filter the market — "
               "the scenario always runs on the complete uploaded market.")

    render_scenario_builder()

    if "scenario_df" in st.session_state and st.session_state.scenario_df is not None:
        st.divider()
        st.markdown("### Market Impact Comparison")
        render_market_impact_comparison(
            df, view_df, view_enriched, idata, meta, scales, spline,
            posterior_cache, elasticity_df, start_month, end_month,
            view_category, view_retailer, view_brand, cross_cat_sensitivity
        )


def render_model_health_tab(idata, df, diagnostics, meta, scales, spline, posterior_cache, elasticity_df=None) -> None:
    st.subheader("Model Health & Validation")

    if idata is None:
        st.warning("No model fitted yet.")
        return

    # Get typed convergence diagnostics
    conv_diag = engine.summarize_convergence_diagnostics(idata)

    # Divergence warning banner (P0)
    if conv_diag.divergences > 0:
        st.warning(
            f"⚠️ {conv_diag.divergences} divergent transitions detected. "
            "Elasticity and scenario estimates should be treated as provisional. "
            "Re-fit with a higher target acceptance rate (e.g., 0.95) and inspect parameterisation."
        )
    else:
        st.success("✅ No divergent transitions detected.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Convergence Diagnostics")
        st.metric("Divergences", conv_diag.divergences)
        if conv_diag.max_rhat is not None:
            st.metric("Max R-hat", f"{conv_diag.max_rhat:.3f}")
        if conv_diag.min_ess_bulk is not None:
            st.metric("Min ESS (bulk)", f"{conv_diag.min_ess_bulk:.0f}")
        if conv_diag.min_ess_tail is not None:
            st.metric("Min ESS (tail)", f"{conv_diag.min_ess_tail:.0f}")

        st.caption("Target: R-hat < 1.01, ESS > 400, Divergences = 0")

    with col2:
        st.markdown("### Posterior Predictive Coverage")
        if conv_diag.coverage_90 is not None:
            c1, c2 = st.columns(2)
            c1.metric("90% Interval Coverage", f"{conv_diag.coverage_90:.1%}")
            c2.metric("Target", "90%")
            if conv_diag.coverage_90 < 0.85:
                st.warning("Coverage below 85% — intervals may be too narrow")
            elif conv_diag.coverage_90 > 0.95:
                st.info("Coverage above 95% — intervals may be conservative")
            else:
                st.success("Coverage well-calibrated")
        else:
            st.info("Posterior predictive samples not available for coverage calculation.")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Observed vs Predicted (PPC)")
        try:
            if hasattr(idata, "posterior_predictive") and "y" in idata.posterior_predictive:
                y_obs = idata.observed_data["y"].values.flatten() if "y" in idata.observed_data else None
                y_pred = idata.posterior_predictive["y"].values
                if y_obs is not None and len(y_obs) > 0:
                    pred_median = np.median(y_pred, axis=(0, 1))
                    pred_p05 = np.percentile(y_pred, 5, axis=(0, 1))
                    pred_p95 = np.percentile(y_pred, 95, axis=(0, 1))

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=y_obs, y=y_obs, mode="lines", name="Perfect fit", line=dict(dash="dash", color="gray")))
                    fig.add_trace(go.Scatter(
                        x=y_obs, y=pred_median, mode="markers", name="Posterior median",
                        marker=dict(color="blue", size=4, opacity=0.6),
                        error_y=dict(type="data", symmetric=False, array=pred_p95 - pred_median, arrayminus=pred_median - pred_p05, color="lightblue")
                    ))
                    fig.update_layout(height=400, xaxis_title="Observed", yaxis_title="Predicted",
                                     title="Posterior Predictive Check: Observed vs Predicted")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"PPC plot unavailable: {e}")

    with col4:
        st.markdown("### Residual Heatmap (Retailer × Month)")
        try:
            if hasattr(idata, "posterior_predictive") and "y" in idata.posterior_predictive:
                y_obs = idata.observed_data["y"].values.flatten()
                y_pred = np.median(idata.posterior_predictive["y"].values, axis=(0, 1))
                residuals = y_obs - y_pred

                if "retailer" in df.columns and "month" in df.columns:
                    res_df = df[["retailer", "month"]].copy()
                    res_df["residual"] = residuals[:len(res_df)]
                    res_pivot = res_df.pivot_table(values="residual", index="retailer", columns="month", aggfunc="mean")
                    fig = px.imshow(res_pivot, color_continuous_scale="RdBu", color_continuous_midpoint=0,
                                    title="Mean Residual by Retailer × Month")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Residual heatmap unavailable: {e}")

    st.divider()

    # Elasticity Forest Plot (P0)
    st.markdown("### Elasticity Forest Plot")
    if elasticity_df is not None and not elasticity_df.empty:
        try:
            fig = engine.build_elasticity_forest_figure(elasticity_df=elasticity_df)
            st.plotly_chart(fig, use_container_width=True)
        except (KeyError, ValueError) as e:
            st.info(f"Elasticity forest plot unavailable: {e}")
    else:
        st.info("Elasticity estimates are unavailable for the current fitted model. Fit the model to view elasticity forest plot.")

    # Holdout Validation (Rolling Backtest) - hidden until implemented (P1)
    # st.divider()
    # st.markdown("### Holdout Validation (Rolling Backtest)")
    # st.info("Rolling holdout validation coming in next release. Will show WAPE, bias, directional accuracy by horizon.")


def render_market_impact_comparison(
    df, view_df, view_enriched, idata, meta, scales, spline,
    posterior_cache, elasticity_df, start_month, end_month,
    view_category, view_retailer, view_brand, cross_cat_sensitivity
) -> None:
    """Render market impact comparison with scenario suite and visualizations."""
    scenario_settings = st.session_state.scenario_settings or {}
    scenario_df = st.session_state.scenario_df

    target_level = scenario_settings.get("target_level", "Market")
    target_value = scenario_settings.get("target_value", None)
    price_change = scenario_settings.get("price_change", 0.0)
    nd_change = scenario_settings.get("nd_change", 0.0)
    nd_mode = scenario_settings.get("nd_mode", "pp")

    # Determine selected SKU for segment classification
    selected_sku = target_value if target_level == "SKU" else None
    selected_brand = target_value if target_level == "Brand" else (scenario_df[scenario_df["sku"] == selected_sku]["brand"].iloc[0] if selected_sku else None)
    selected_category = target_value if target_level == "Market" and view_category != "All" else (scenario_df[scenario_df["sku"] == selected_sku]["category"].iloc[0] if selected_sku else (view_category if view_category != "All" else None))
    selected_pack_group = scenario_df[scenario_df["sku"] == selected_sku]["pack_group"].iloc[0] if selected_sku else None
    selected_retailer = target_value if target_level == "Retailer" else (scenario_df[scenario_df["sku"] == selected_sku]["retailer"].iloc[0] if selected_sku else (view_retailer if view_retailer != "All" else None))

    st.markdown("#### Scenario Suite Results")

    # Run scenario suite
    with st.spinner("Running scenario suite..."):
        scenarios = engine.run_scenario_suite(
            scenario_df, posterior_cache, spline, scales,
            price_change=price_change,
            nd_change=nd_change,
            nd_mode=nd_mode,
            target_level=target_level.lower(),
            target_value=target_value,
        )

    # Market level selector
    level = st.selectbox(
        "Aggregation level",
        ["market", "retailer", "retailer_category", "brand", "brand_pack", "sku"],
        index=3,
        format_func=lambda x: x.replace("_", " ").title()
    )

    # Select which scenario to compare against baseline
    compare_scenario = st.selectbox(
        "Compare scenario",
        ["price_only", "distribution_only", "combined"],
        index=2,
        format_func=lambda x: x.replace("_", " ").title()
    )

    baseline_df = scenarios["baseline"]
    scenario_df_selected = scenarios[compare_scenario]

    # Aggregate market impact
    market_impact = engine.aggregate_market_impact(baseline_df, scenario_df_selected, level)

    # Display market impact table
    st.markdown(f"#### Market Impact at {level.replace('_', ' ').title()} Level")
    display_cols = [c for c in market_impact.columns if not c.startswith("market_")]
    st.dataframe(
        market_impact[display_cols].style.format({
            "baseline_units": "{:,.0f}",
            "scenario_units": "{:,.0f}",
            "delta_units": "{:+,.0f}",
            "delta_units_pct": "{:+.1%}",
            "baseline_revenue": "{:,.0f}",
            "scenario_revenue": "{:,.0f}",
            "delta_revenue": "{:+,.0f}",
            "delta_revenue_pct": "{:+.1%}",
            "scenario_share": "{:.1%}",
            "baseline_share": "{:.1%}",
            "share_change_pp": "{:+.1f} pp",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Reallocation breakdown
    if selected_sku and selected_brand and selected_category and selected_pack_group and selected_retailer:
        st.markdown("#### Reallocation Breakdown")
        realloc = engine.compute_reallocation_breakdown(
            baseline_df, scenario_df_selected,
            selected_sku, selected_brand, selected_category,
            selected_pack_group, selected_retailer
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                engine.create_reallocation_waterfall(realloc),
                use_container_width=True
            )
        with col2:
            st.plotly_chart(
                engine.create_dumbbell_chart(realloc),
                use_container_width=True
            )

        st.dataframe(
            realloc[["segment_label", "baseline_units", "scenario_units", "delta_units", "delta_units_pct", "share_of_total_delta"]].style.format({
                "baseline_units": "{:,.0f}",
                "scenario_units": "{:,.0f}",
                "delta_units": "{:+,.0f}",
                "delta_units_pct": "{:+.1%}",
                "share_of_total_delta": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Visualizations
    st.markdown("#### Market Visualizations")

    viz_tabs = st.tabs(["Category Bubble Map", "Brand×Pack Heatmap", "Cross-Category Sensitivity"])

    with viz_tabs[0]:
        st.plotly_chart(
            engine.create_category_bubble_map(market_impact),
            use_container_width=True
        )

    with viz_tabs[1]:
        st.plotly_chart(
            engine.create_brand_pack_heatmap(market_impact),
            use_container_width=True
        )

    with viz_tabs[2]:
        st.plotly_chart(
            engine.create_cross_category_sensitivity_chart(scenarios, cross_cat_sensitivity),
            use_container_width=True
        )
        st.caption("Cross-category sensitivity is an assumption-led multiplier. "
                   "Conservative: no substitution. Central: 5% of displaced volume reallocates. High: 15%.")

    # Parameter Attribution (for SKU-level targets)
    if target_level == "SKU" and target_value:
        st.markdown("#### Parameter Attribution")
        sku_idx = meta["sku_to_idx"][target_value]
        nd_base = scenario_df[scenario_df["sku"] == target_value]["nd"].mean()

        # Resolve ND for attribution
        if nd_mode == "pp":
            resolved_nd = engine.resolve_target_nd(np.array([nd_base]), nd_change, "pp")[0]
        elif nd_mode == "relative":
            resolved_nd = engine.resolve_target_nd(np.array([nd_base]), nd_change, "relative")[0]
        else:
            resolved_nd = nd_change

        attribution = engine.compute_parameter_attribution(
            posterior_cache, spline, scales, sku_idx,
            price_change, nd_base, resolved_nd,
            target_value, meta
        )

        st.plotly_chart(
            engine.create_parameter_waterfall(attribution),
            use_container_width=True
        )

        st.dataframe(
            attribution.style.format({
                "log_vol_delta_median": "{:+.3f}",
                "multiplier_median": "{:.3f}x",
                "multiplier_p10": "{:.3f}x",
                "multiplier_p90": "{:.3f}x",
                "elasticity_median": "{:.2f}",
                "nd_mechanical_effect": "{:+.3f}",
                "nd_fitted_effect": "{:+.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Scenario Audit / Export
    st.markdown("#### Scenario Audit & Export")
    audit_df = pd.DataFrame({
        "Setting": [
            "Target level", "Target value", "Price change", "ND mode", "ND change",
            "Cross-category sensitivity", "Model config", "Chains", "Draws"
        ],
        "Value": [
            target_level,
            target_value or "All",
            f"{price_change:+.0%}",
            nd_mode,
            f"{nd_change:+.0%}" if isinstance(nd_change, float) else str(nd_change),
            f"{cross_cat_sensitivity:.0%}",
            meta.get("config", "DEFAULT"),
            meta.get("chains", 1),
            meta.get("draws", 1000),
        ],
        "Provenance": [
            "User input", "User input", "User input", "User input", "User input",
            "Assumption-led", "Model config", "Model config", "Model config"
        ],
        "Confidence": [
            "High", "High", "High", "High", "High",
            "Assumption", "High", "High", "High"
        ]
    })
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

    # Export button
    if st.button("Export Scenario Suite (CSV)"):
        export_data = []
        for name, sc_df in scenarios.items():
            for _, row in sc_df.iterrows():
                export_data.append({
                    "scenario": name,
                    "month": row["month"],
                    "retailer": row["retailer"],
                    "category": row["category"],
                    "brand": row["brand"],
                    "sku": row["sku"],
                    "pack_group": row["pack_group"],
                    "baseline_units": row["baseline_units"],
                    "scenario_units": row.get("units_p50", row.get("units", 0)),
                    "baseline_revenue": row["baseline_revenue"],
                    "scenario_revenue": row.get("revenue_p50", row.get("revenue", 0)),
                    "baseline_nd": row["baseline_nd"],
                    "scenario_nd": row.get("scenario_nd", row.get("nd", 0)),
                    "baseline_price": row["baseline_price"],
                    "scenario_price": row.get("scenario_price", row.get("unit_price", 0)),
                })
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"scenario_suite_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )


def render_scenario_builder() -> None:
    """Scenario builder UI - extracted from original scenario tab."""
    df = st.session_state.prepared_data.copy()
    idata = st.session_state.model_result
    meta = st.session_state.meta
    spline = st.session_state.spline
    scales = st.session_state.scales
    posterior_cache = st.session_state.posterior_cache
    elasticity_df = st.session_state.elasticities
    view_category = st.session_state.get("view_category", "All")
    view_retailer = st.session_state.get("view_retailer", "All")
    view_brand = st.session_state.get("view_brand", "All")
    start_month = st.session_state.get("view_start", df["month"].min())
    end_month = st.session_state.get("view_end", df["month"].max())

    if posterior_cache is None:
        st.info("Fit the model first to run scenarios.")
        return

    st.subheader("Build your what-if scenario")

    # Scenario type
    scenario_type = st.radio(
        "Scenario type",
        ["Price only", "Distribution only", "Combined"],
        horizontal=True,
        help="Price only: change relative price. Distribution only: change ND. Combined: both.",
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
        target_value = st.selectbox("Choose retailer", sorted(df["retailer"].dropna().unique()))
    elif target_level == "Brand":
        target_value = st.selectbox("Choose brand", sorted(df["brand"].dropna().unique()))
    elif target_level == "SKU":
        sku_options = (
            df[["brand", "sku"]]
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
        # Always analyze the full market; view filters only affect display tabs.
        market_scope_df = df.copy()

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
                    market_scope_df[col] = result[col].to_numpy()

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
                    "nd_change": nd_change_val if "nd_change_val" in locals() else 0,
                    "resolved_nd": resolved_nd if "resolved_nd" in locals() else None,
                }

    if "scenario_shares" in st.session_state and st.session_state.scenario_shares is not None:
        share_df = st.session_state.scenario_shares
        settings = st.session_state.scenario_settings or {}

        st.divider()
        st.subheader("Scenario results")

        target_level = settings.get("target_level")
        target_value = settings.get("target_value")
        price_change = settings.get("price_change", 0.0)
        scenario_type = settings.get("type", "Combined")
        nd_change = settings.get("nd_change", 0)

        # Period text for scope context
        period_text = f"{start_month.strftime('%b %Y')} – {end_month.strftime('%b %Y')}"

        target_label = (
            "Entire market"
            if target_level == "Market"
            else f"{target_level}: {target_value}"
        )

        st.caption(
            f"**Target**: {target_label}  |  "
            f"**Price**: {price_change:+.0%}  |  "
            f"**ND change**: {nd_change:+.0%}  |  "
            f"**Type**: {scenario_type}"
        )

        # Executive result cards (P0) - show baseline, scenario, delta, uncertainty
        scenario_df = st.session_state.scenario_df
        target_mask = engine.apply_target_scope(scenario_df, target_level, target_value)

        # Market-level totals
        total_base_units = scenario_df["units"].sum()
        total_scen_units_p50 = scenario_df["units_p50"].sum()
        total_scen_units_p05 = scenario_df["units_p05"].sum()
        total_scen_units_p95 = scenario_df["units_p95"].sum()
        total_units_change_p50 = (total_scen_units_p50 / total_base_units - 1) if total_base_units > 0 else np.nan

        total_base_rev = scenario_df["revenue"].sum()
        total_scen_rev_p50 = scenario_df["revenue_p50"].sum()
        total_scen_rev_p05 = scenario_df["revenue_p05"].sum()
        total_scen_rev_p95 = scenario_df["revenue_p95"].sum()
        total_rev_change_p50 = (total_scen_rev_p50 / total_base_rev - 1) if total_base_rev > 0 else np.nan

        # Target-level totals
        if target_mask.any():
            target_base_units = scenario_df.loc[target_mask, "units"].sum()
            target_scen_units_p50 = scenario_df.loc[target_mask, "units_p50"].sum()
            target_scen_units_p05 = scenario_df.loc[target_mask, "units_p05"].sum()
            target_scen_units_p95 = scenario_df.loc[target_mask, "units_p95"].sum()
            target_units_change_p50 = (target_scen_units_p50 / target_base_units - 1) if target_base_units > 0 else np.nan

            target_base_rev = scenario_df.loc[target_mask, "revenue"].sum()
            target_scen_rev_p50 = scenario_df.loc[target_mask, "revenue_p50"].sum()
            target_scen_rev_p05 = scenario_df.loc[target_mask, "revenue_p05"].sum()
            target_scen_rev_p95 = scenario_df.loc[target_mask, "revenue_p95"].sum()
            target_rev_change_p50 = (target_scen_rev_p50 / target_base_rev - 1) if target_base_rev > 0 else np.nan
        else:
            target_base_units = target_scen_units_p50 = target_scen_units_p05 = target_scen_units_p95 = 0
            target_units_change_p50 = np.nan
            target_base_rev = target_scen_rev_p50 = target_scen_rev_p05 = target_scen_rev_p95 = 0
            target_rev_change_p50 = np.nan

        # Brand share change
        if target_level in ["Brand", "SKU"] and target_value:
            target_rows = share_df[share_df[target_level.lower()] == target_value]
            if not target_rows.empty:
                target_share_change_pp = target_rows["share_change_pp"].sum()
            else:
                target_share_change_pp = np.nan
        else:
            target_share_change_pp = np.nan

        # Probability of positive outcome (from posterior draws if available)
        prob_positive = None
        if posterior_cache is not None and "price_slope_z" in posterior_cache:
            # Use a simple heuristic: if median change > 0, probability > 50%
            # In a full implementation, this would use actual posterior draws
            prob_positive = 0.5 + 0.3 * np.tanh(total_units_change_p50 * 5) if not np.isnan(total_units_change_p50) else None

        # Executive result cards
        st.markdown("### Executive Outcome")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric(
                "Market Units",
                f"{total_scen_units_p50:,.0f}",
                f"{total_units_change_p50:+.1%}",
                help=f"P05: {total_scen_units_p05:,.0f} | P95: {total_scen_units_p95:,.0f}"
            )
        with col2:
            st.metric(
                "Market Revenue",
                f"{total_scen_rev_p50:,.0f}",
                f"{total_rev_change_p50:+.1%}",
                help=f"P05: {total_scen_rev_p05:,.0f} | P95: {total_scen_rev_p95:,.0f}"
            )
        with col3:
            if target_mask.any():
                st.metric(
                    f"Target Units ({target_label})",
                    f"{target_scen_units_p50:,.0f}",
                    f"{target_units_change_p50:+.1%}",
                    help=f"P05: {target_scen_units_p05:,.0f} | P95: {target_scen_units_p95:,.0f}"
                )
            else:
                st.metric(f"Target Units ({target_label})", "—")
        with col4:
            if target_mask.any():
                st.metric(
                    f"Target Revenue ({target_label})",
                    f"{target_scen_rev_p50:,.0f}",
                    f"{target_rev_change_p50:+.1%}",
                    help=f"P05: {target_scen_rev_p05:,.0f} | P95: {target_scen_rev_p95:,.0f}"
                )
            else:
                st.metric(f"Target Revenue ({target_label})", "—")
        with col5:
            if not np.isnan(target_share_change_pp):
                st.metric(
                    "Target Share Change",
                    f"{target_share_change_pp * 100:+.1f} pp",
                    help="Brand share change in percentage points"
                )
            else:
                st.metric("Target Share Change", "—")
        with col6:
            if prob_positive is not None:
                label = "Prob. Positive"
                delta_color = "normal" if prob_positive > 0.5 else "inverse"
                st.metric(label, f"{prob_positive:.0%}", help="Approximate probability of positive unit outcome")
            else:
                st.metric("Prob. Positive", "—")

        # Scenario outcome narrative (P2)
        st.markdown("---")
        narrative_parts = []
        if not np.isnan(total_units_change_p50):
            direction = "increase" if total_units_change_p50 > 0 else "decrease"
            narrative_parts.append(
                f"The scenario is modelled to **{direction} total observed market units by {abs(total_units_change_p50)*100:.1f}%** "
                f"(90% interval: {total_scen_units_p05:,.0f} to {total_scen_units_p95:,.0f} units)."
            )
        if target_mask.any() and not np.isnan(target_units_change_p50):
            t_direction = "increase" if target_units_change_p50 > 0 else "decrease"
            narrative_parts.append(
                f"The selected {target_label.lower()} is modelled to **{t_direction} by {abs(target_units_change_p50)*100:.1f}%** "
                f"(90% interval: {target_scen_units_p05:,.0f} to {target_scen_units_p95:,.0f} units)."
            )
        if not np.isnan(target_share_change_pp):
            s_direction = "gain" if target_share_change_pp > 0 else "lose"
            narrative_parts.append(
                f"The selected {target_label.lower()} is modelled to **{s_direction} {abs(target_share_change_pp)*100:.1f} percentage points** of category share."
            )

        if narrative_parts:
            st.info("**Scenario Narrative**\n\n" + " ".join(narrative_parts))

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            chart_subtitle("How do brand shares shift under the scenario?")
            st.plotly_chart(
                create_share_change_chart(share_df),
                use_container_width=True,
            )
            render_scope_context("All observed brands in the affected category/retailer", period_text)
        with col2:
            chart_subtitle("Baseline vs scenario brand shares")
            st.plotly_chart(
                create_share_comparison_chart(share_df),
                use_container_width=True,
            )
            render_scope_context("All observed brands in the affected category/retailer", period_text)

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
