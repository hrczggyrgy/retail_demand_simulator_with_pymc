"""
Retail Demand & Share Simulator — Simplified 4-Page Workflow
--------------------------------------------------------------
Pages:
1. Market data       — Upload/sample data, raw table, data quality, market overview with decomposition identity
2. Market diagnostics — Share, price-pack architecture, ND/velocity quadrant, growth decomposition
3. Fit & validate    — Model mode, Fit/Re-fit button, diagnostics, PPC, elasticity forest
4. Scenario cockpit  — Editable action table (data_editor), guardrails, scenario outputs, market impact

Standardized columns: standard_volume, price_per_standard_unit, nd, velocity_std
Decomposition identity: standard_volume = retailer_stores × nd × velocity_std
"""

from __future__ import annotations

import io
from dataclasses import dataclass

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
        /* Style for data_editor checkbox column */
        .stDataFrame [data-testid="stCheckbox"] {
            transform: scale(1.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants & column mapping
# ---------------------------------------------------------------------------

# Standard column names used throughout the app
STANDARD_COLUMNS = {
    "month": "month",
    "retailer": "retailer",
    "category": "category",
    "brand": "brand",
    "sku": "sku",
    "units": "units",
    "revenue": "revenue",
    "sku_stores": "sku_stores",
    "retailer_stores": "retailer_stores",
    "pack_size": "pack_size",
    # Derived
    "standard_volume": "standard_volume",
    "price_per_standard_unit": "price_per_standard_unit",
    "nd": "nd",
    "velocity_std": "velocity_std",
    "relative_price_std": "relative_price_std",
}

REQUIRED_RAW_COLUMNS = {
    "month", "retailer", "category", "brand", "sku",
    "units", "revenue", "sku_stores", "retailer_stores", "pack_size"
}

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
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return "—"
        return ts.strftime("%b %Y")
    except (ValueError, TypeError):
        return "—"


def render_workflow_banner(
    step: int,
) -> None:
    """Render workflow progress banner showing user journey status.
    
    Steps: 1=Data, 2=Diagnostics, 3=Model, 4=Scenario
    """
    steps = [
        ("1. Market data", "Data loaded and validated"),
        ("2. Market diagnostics", "Market structure visible"),
        ("3. Fit & validate", "Posterior sampled, diagnostics available"),
        ("4. Scenario cockpit", "Actions defined, impact compared"),
    ]

    icons = []
    for i, (label, _tooltip) in enumerate(steps, 1):
        if i < step:
            icon = "✅"
        elif i == step:
            icon = "🔄"
        else:
            icon = "⬜"
        icons.append(f"{icon} {label}")

    st.caption("  →  ".join(icons), help="Your progress through the decision workflow")
    
    if step == 1:
        st.info("💡 Upload data or use demo market to begin.")
    elif step == 2:
        st.info("💡 Explore market structure, then click **Fit model** in sidebar.")
    elif step == 3:
        st.info("💡 Review diagnostics, then define scenario actions.")
    elif step == 4:
        st.success("✅ Ready to run scenarios and compare market impact.")


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
        st.info("ℹ Model status: **Exploratory** — review full diagnostics in Fit & validate page")

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


def check_decomposition_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Verify the decomposition identity: standard_volume = retailer_stores × nd × velocity_std.
    
    Returns a DataFrame with decomposition check results per SKU-month.
    """
    result = df.copy()
    required = ["standard_volume", "retailer_stores", "nd", "velocity_std"]
    if all(c in result.columns for c in required):
        result["decomposition_check"] = (
            result["retailer_stores"] * result["nd"] * result["velocity_std"]
        )
        result["decomposition_error"] = (
            result["decomposition_check"] - result["standard_volume"]
        ).abs()
        result["decomposition_error_pct"] = (
            result["decomposition_error"] / result["standard_volume"].replace(0, np.nan)
        ) * 100
    else:
        result["decomposition_check"] = np.nan
        result["decomposition_error"] = np.nan
        result["decomposition_error_pct"] = np.nan
    return result


# ---------------------------------------------------------------------------
# Session state / data loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AppReadiness:
    has_raw_data: bool
    has_prepared_data: bool
    has_model: bool
    has_valid_diagnostics: bool
    has_scenario: bool


def get_app_readiness() -> AppReadiness:
    diagnostics = st.session_state.get("diagnostics")
    return AppReadiness(
        has_raw_data="raw_data" in st.session_state and st.session_state.raw_data is not None,
        has_prepared_data="prepared_data" in st.session_state and st.session_state.prepared_data is not None,
        has_model="model_result" in st.session_state and st.session_state.model_result is not None,
        has_valid_diagnostics=(
            diagnostics is not None
            and getattr(diagnostics, "is_decision_ready", False)
        ),
        has_scenario="scenario_suite" in st.session_state and st.session_state.scenario_suite is not None,
    )


def require_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    chart_name: str,
) -> bool:
    """Check if required columns exist in dataframe; show info message if not."""
    missing = required_columns.difference(df.columns)

    if missing:
        st.info(
            f"{chart_name} is unavailable because required fields are missing: "
            f"{', '.join(sorted(missing))}."
        )
        return False

    if df.empty:
        st.info(
            f"{chart_name} is unavailable because no records match "
            "the selected filters."
        )
        return False

    return True


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
        "posterior_cache",
        "fitted_model",
        "scenario_suite",
        "scenario_actions",
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


def _model_config_from_mode(mode: str) -> engine.ModelConfig | engine.JointModelConfig:
    if mode == "Fast (2 chains)":
        return engine.FAST_CONFIG
    elif mode == "Advanced (4 chains, Student-t)":
        return engine.ADVANCED_CONFIG
    elif mode == "Joint SKU Share Model (Dirichlet-Multinomial)":
        return engine.JointModelConfig()
    return engine.DEFAULT_CONFIG


def _is_joint_model_mode(mode: str) -> bool:
    return mode == "Joint SKU Share Model (Dirichlet-Multinomial)"


def fit_model_explicitly() -> None:
    if st.session_state.prepared_data is None:
        return

    df = st.session_state.prepared_data
    mode = st.session_state.get("model_mode", "Default (4 chains)")
    config = _model_config_from_mode(mode)

    with st.spinner("Estimating price and distribution response..."):
        if _is_joint_model_mode(mode):
            # Build choice set data for joint model
            enriched = engine.build_retail_features(df)
            choice_data = engine.build_choice_set_data(enriched, st.session_state.meta)
            st.session_state.choice_data = choice_data
            
            model = engine.build_joint_sku_share_model(choice_data, config)
            idata = engine.fit_joint_model(model, config)
            
            # Posterior predictive
            idata = engine.sample_joint_posterior_predictive(model, idata)
            
            st.session_state.posterior_cache = engine.extract_joint_posterior(
                idata,
                max_draws=config.scenario_draws if hasattr(config, "scenario_draws") else 400,
                random_seed=config.random_seed,
            )
        else:
            if config.use_category_price_pooling or config.use_sku_nd_effects:
                model = engine.build_pymc_model_v2(df, st.session_state.meta, st.session_state.nd_basis, config)
            else:
                model = engine.build_pymc_model(df, st.session_state.meta, st.session_state.nd_basis)

            idata = engine.fit_model(model, config)
            idata = engine.add_posterior_predictive(model, idata)

            st.session_state.posterior_cache = engine.extract_scenario_posterior(
                idata,
                max_draws=config.scenario_draws,
                random_seed=config.random_seed,
            )

    st.session_state.model_result = idata
    st.session_state.fitted_model = model
    st.session_state.model_settings = {"config": config, "mode": mode}
    
    if _is_joint_model_mode(mode):
        st.session_state.model_diagnostics = engine.summarize_convergence_diagnostics(idata)
        # Extract elasticities from joint model (beta_sku)
        # TODO: implement joint model elasticity extraction
    else:
        st.session_state.model_diagnostics = engine.get_model_diagnostics(idata)
        st.session_state.elasticities = engine.extract_elasticities(
            idata,
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

        st.subheader("1. Market data")

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
        st.subheader("2. Fit & validate")

        model_mode = st.selectbox(
            "Model mode",
            [
                "Fast (2 chains)",
                "Default (4 chains)",
                "Advanced (4 chains, Student-t)",
                "Joint SKU Share Model (Dirichlet-Multinomial)",
            ],
            index=1,
            help=(
                "Fast: quick exploratory fits. Default: balanced speed/quality. "
                "Advanced: full hierarchy with robust likelihood (slower). "
                "Joint SKU Share Model: Dirichlet-Multinomial choice model for "
                "within retailer-category SKU allocation (experimental)."
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
        st.subheader("3. View filters")
        st.caption("These filters affect **display pages only**. Scenarios always use the full market.")

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
# Page 1: Market data
# ---------------------------------------------------------------------------

def render_market_data_page(
    df: pd.DataFrame | None = None,
    view_df: pd.DataFrame | None = None,
    view_enriched: pd.DataFrame | None = None,
    start_month=None,
    end_month=None,
) -> None:
    """Page 1: Market data — upload/sample data, raw table, data quality, market overview with decomposition identity."""
    st.subheader("Market Data")
    
    if st.session_state.raw_data is None:
        st.info(
            "Start with **Use demo market** or upload your complete market dataset. "
            "The model needs competitor SKUs to calculate relative price and share."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use demo market", use_container_width=True, type="primary"):
                load_dataframe(demo_data_cached())
                st.rerun()
        with col2:
            template = demo_data.get_input_template()
            st.download_button(
                "Download template",
                template.to_csv(index=False),
                "retail_data_template.csv",
                "text/csv",
                use_container_width=True,
            )
        return

    # Data is loaded - show overview
    raw = st.session_state.raw_data
    
    # Show raw data preview
    st.markdown("### Raw Data Preview")
    st.caption(f"Shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
    st.dataframe(raw.head(100), use_container_width=True, hide_index=True)
    
    # Column validation
    st.markdown("### Column Validation")
    missing = REQUIRED_RAW_COLUMNS - set(raw.columns)
    extra = set(raw.columns) - REQUIRED_RAW_COLUMNS
    
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    else:
        st.success("✅ All 10 required columns present")
    
    if extra:
        st.info(f"Extra columns (will be ignored): {', '.join(sorted(extra))}")
    
    # Data quality report
    if st.session_state.quality_report is not None:
        st.divider()
        st.markdown("### Data Quality Report")
        quality = st.session_state.quality_report
        
        counts = quality.get("counts", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retailers", counts.get("retailers", 0))
        c2.metric("Categories", counts.get("categories", 0))
        c3.metric("Brands", counts.get("brands", 0))
        c4.metric("SKUs", counts.get("skus", 0))
        
        coverage = quality.get("coverage", {})
        if coverage:
            st.markdown("#### Field Coverage")
            cov_df = pd.DataFrame(list(coverage.items()), columns=["Field", "Coverage %"])
            cov_df["Coverage %"] = cov_df["Coverage %"] * 100
            fig = px.bar(cov_df, x="Field", y="Coverage %", color="Coverage %", 
                         color_continuous_scale="RdYlGn", height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Prepared data overview with decomposition identity
    if view_enriched is not None and not view_enriched.empty:
        st.divider()
        period_text = f"{start_month.strftime('%b %Y')} – {end_month.strftime('%b %Y')}"
        st.markdown(f"### Market Overview ({period_text})")
        render_scope_context("All observed retailers, categories, brands, and SKUs", period_text)
        
        # Verify decomposition identity
        checked = check_decomposition_identity(view_enriched)
        max_error_pct = checked["decomposition_error_pct"].max()
        mean_error_pct = checked["decomposition_error_pct"].mean()
        
        c1, c2, c3, c4, c5 = st.columns(5)
        latest_month = view_enriched["month"].max()
        latest = view_enriched[view_enriched["month"] == latest_month]
        
        c1.metric("Total Std Volume", f"{latest['standard_volume'].sum():,.0f}")
        c2.metric("Total Revenue", f"{latest['revenue'].sum():,.0f}")
        c3.metric("SKUs", f"{latest['sku'].nunique()}")
        c4.metric("Avg ND", f"{latest['nd'].mean():.1%}" if "nd" in latest.columns else "—")
        c5.metric("Decomp. Error (max)", f"{max_error_pct:.2f}%" if not pd.isna(max_error_pct) else "—")
        
        if max_error_pct > 1.0:
            st.warning(f"⚠ Decomposition identity check: max error = {max_error_pct:.2f}% (should be < 1%)")
        elif not pd.isna(max_error_pct):
            st.success(f"✅ Decomposition identity holds: max error = {max_error_pct:.4f}%")
        
        # Category trend
        st.markdown("#### Category Trend")
        chart_subtitle("How has the observed market changed over time?")
        if "standard_volume" in view_enriched.columns:
            trend = view_enriched.groupby("month").agg(
                volume=("standard_volume", "sum"),
                revenue=("revenue", "sum")
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=trend["month"], y=trend["volume"], name="Std Volume", yaxis="y", opacity=0.6))
            fig.add_trace(go.Scatter(x=trend["month"], y=trend["revenue"], name="Revenue", yaxis="y2", line=dict(color="red")))
            fig.update_layout(
                yaxis=dict(title="Standard Volume", side="left"),
                yaxis2=dict(title="Revenue", side="right", overlaying="y"),
                height=350, hovermode="x unified", margin=dict(l=40, r=40, t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed retailers, categories, brands, and SKUs", period_text)
        
        # Raw data expander
        with st.expander("📋 Full raw data table", expanded=False):
            st.dataframe(raw, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 2: Market diagnostics
# ---------------------------------------------------------------------------

def render_market_diagnostics_page(
    view_df: pd.DataFrame,
    view_enriched: pd.DataFrame,
    start_month,
    end_month,
) -> None:
    """Page 2: Market diagnostics — share, price-pack architecture, ND/velocity quadrant, growth decomposition."""
    period_text = f"{start_month.strftime('%b %Y')} – {end_month.strftime('%b %Y')}"
    st.subheader(f"Market Diagnostics ({period_text})")
    render_scope_context("All observed retailers, categories, brands, and SKUs", period_text)
    
    if view_enriched is None or view_enriched.empty:
        st.warning("No data available for diagnostics.")
        return
    
    latest_month = view_enriched["month"].max()
    latest = view_enriched[view_enriched["month"] == latest_month]
    
    # Row 1: Brand share trends + Brand share latest
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Brand Share Trends")
        chart_subtitle("Which brands gained or lost share over time?")
        if "brand_share" in view_enriched.columns:
            share_trend = view_enriched.groupby(["month", "brand"]).agg(
                share=("brand_share", "first"),
                volume=("standard_volume", "sum")
            ).reset_index()
            fig = px.area(share_trend, x="month", y="share", color="brand", 
                          title="Brand Standard Volume Share Over Time")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed retailers, categories, brands, and SKUs", period_text)
        else:
            st.info("Brand share not available. Run feature engineering first.")
    
    with col2:
        st.markdown("### Brand Share (Latest Month)")
        chart_subtitle("What is the current brand share distribution?")
        if "brand_share" in latest.columns:
            brand_share_latest = latest.groupby("brand").agg(
                share=("brand_share", "first"),
                volume=("standard_volume", "sum")
            ).reset_index().sort_values("share", ascending=False)
            fig = px.pie(brand_share_latest, values="share", names="brand", 
                         title=f"Brand Share – {latest_month.strftime('%b %Y')}")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed retailers, categories, brands, and SKUs", 
                               f"Latest month ({latest_month.strftime('%b %Y')})")
        else:
            st.info("Brand share not available.")
    
    # Row 2: Price-Pack Architecture + ND/Velocity Quadrant
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Price-Pack Architecture")
        chart_subtitle("Which packs occupy comparable price-value positions?")
        if "price_per_standard_unit" in latest.columns and "pack_size" in latest.columns:
            fig = px.scatter(
                latest, x="pack_size", y="price_per_standard_unit",
                size="revenue", color="brand", hover_data=["sku", "standard_volume", "nd"],
                title=f"Price per Standard Unit by Pack Size – {latest_month.strftime('%b %Y')}",
                labels={"pack_size": "Pack Size", "price_per_standard_unit": "Price per Standard Unit"}
            )
            if "category_price_median" in latest.columns:
                cat_median = latest["category_price_median"].median()
                fig.add_hline(y=cat_median, line_dash="dash", line_color="gray", annotation_text="Category Median")
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed SKUs in latest month", f"Latest month ({latest_month.strftime('%b %Y')})")
        else:
            st.info("Price per standard unit not available.")
    
    with col4:
        st.markdown("### ND vs Velocity Opportunity Quadrant")
        chart_subtitle("Which SKUs combine strong velocity with low numeric distribution?")
        if "nd" in latest.columns and "velocity_std" in latest.columns:
            fig = px.scatter(
                latest, x="nd", y="velocity_std",
                size="revenue", color="brand", hover_data=["sku", "pack_size", "standard_volume"],
                title=f"Distribution vs Velocity (Std) – {latest_month.strftime('%b %Y')}",
                labels={"nd": "Numeric Distribution", "velocity_std": "Velocity Std (units/store)"}
            )
            nd_median = latest["nd"].median()
            vel_median = latest["velocity_std"].median()
            fig.add_vline(x=nd_median, line_dash="dash", line_color="gray")
            fig.add_hline(y=vel_median, line_dash="dash", line_color="gray")
            fig.add_annotation(x=nd_median*0.5, y=vel_median*1.5, text="Expand", showarrow=False, font=dict(size=10, color="green"))
            fig.add_annotation(x=nd_median*1.5, y=vel_median*1.5, text="Protect", showarrow=False, font=dict(size=10, color="blue"))
            fig.add_annotation(x=nd_median*0.5, y=vel_median*0.5, text="Test/Review", showarrow=False, font=dict(size=10, color="orange"))
            fig.add_annotation(x=nd_median*1.5, y=vel_median*0.5, text="Rationalize", showarrow=False, font=dict(size=10, color="red"))
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed SKUs in latest month", f"Latest month ({latest_month.strftime('%b %Y')})")
        else:
            st.info("ND or velocity not available.")
    
    # Row 3: Relative Price Index + Distribution White-Space
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("### Relative Price Index by Brand")
        chart_subtitle("How do brands' prices compare to the category median?")
        if "relative_price_std" in latest.columns:
            rpi = latest.groupby("brand").agg(
                rpi=("relative_price_std", "median"),
                revenue=("revenue", "sum")
            ).reset_index().sort_values("rpi")
            fig = px.bar(rpi, x="brand", y="rpi", color="rpi", color_continuous_scale="RdYlGn_r",
                         title="Median Relative Price Index by Brand (vs Category Median)")
            fig.add_hline(y=1.0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed brands in latest month", f"Latest month ({latest_month.strftime('%b %Y')})")
        else:
            st.info("Relative price index not available.")
    
    with col6:
        st.markdown("### Distribution White-Space Matrix")
        chart_subtitle("Which retailer-brand combinations have white-space (low ND, high revenue)?")
        if "retailer" in latest.columns:
            ws = latest.groupby(["retailer", "brand"]).agg(
                skus=("sku", "nunique"),
                avg_nd=("nd", "mean"),
                revenue=("revenue", "sum")
            ).reset_index()
            fig = px.scatter(ws, x="retailer", y="brand", size="revenue", color="avg_nd",
                             color_continuous_scale="RdYlGn", title="Avg ND by Retailer × Brand (bubble = revenue)")
            st.plotly_chart(fig, use_container_width=True)
            render_scope_context("All observed retailer-brand combinations", f"Latest month ({latest_month.strftime('%b %Y')})")
        else:
            st.info("Retailer data not available.")
    
    # Row 4: Growth Decomposition Waterfall
    st.divider()
    st.markdown("### Growth Decomposition Waterfall")
    chart_subtitle("Which brands explain the category's standard volume change?")
    if "brand_units" in view_enriched.columns:
        latest_m = view_enriched["month"].max()
        prev_m = view_enriched["month"].min()
        curr = view_enriched[view_enriched["month"] == latest_m].groupby("brand").agg(
            volume=("standard_volume", "sum"),
            brand_volume=("brand_units", "first")
        )
        prev = view_enriched[view_enriched["month"] == prev_m].groupby("brand").agg(
            volume=("standard_volume", "sum"),
            brand_volume=("brand_units", "first")
        )
        merged = curr.join(prev, lsuffix="_curr", rsuffix="_prev", how="outer").fillna(0)
        merged["change"] = merged["volume_curr"] - merged["volume_prev"]
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
        fig.update_layout(height=400, title="Category Growth by Brand Contribution (Standard Volume)")
        st.plotly_chart(fig, use_container_width=True)
        render_scope_context("All observed brands in category", 
                           f"Latest ({latest_m.strftime('%b %Y')}) vs. earliest ({prev_m.strftime('%b %Y')}) period")
    else:
        st.info("Brand units not available for growth decomposition.")


# ---------------------------------------------------------------------------
# Page 3: Fit & validate
# ---------------------------------------------------------------------------

def render_fit_validate_page(
    df: pd.DataFrame,
    view_df: pd.DataFrame,
    view_enriched: pd.DataFrame,
    start_month,
    end_month,
) -> None:
    """Page 3: Fit & validate — model mode, Fit/Re-fit button, diagnostics, PPC, elasticity forest."""
    st.subheader("Fit & Validate")
    
    # Show model config from sidebar
    model_mode = st.session_state.get("model_mode", "Default (4 chains)")
    st.caption(f"Model mode: {model_mode}")
    
    if st.session_state.model_result is None:
        st.info("Click **Fit / Re-fit model** in the sidebar to estimate the model.")
        st.markdown("### Model Specification")
        st.markdown("""
        The hierarchical Bayesian model estimates:
        
        ```
        log(velocity_std_{r,s,t}) = α + η_{r,s} + μ_t + β_s × log(relative_price_std_{r,s,t}) + f(ND_{r,s,t}) + ε_{r,s,t}
        ```
        
        Where:
        - **α**: Global intercept
        - **η_{r,s}**: Retailer×SKU entity effect (partial pooling)
        - **μ_t**: Month effect (seasonality)
        - **β_s**: SKU price elasticity with brand×pack pooling
        - **f(ND)**: B-spline for non-linear numeric distribution effect
        - **ε**: Observation noise (Student-t in Advanced mode)
        
        **Standardization**: All continuous variables are z-scored before modeling.
        """)
        return
    
    # Model is fitted - show diagnostics
    idata = st.session_state.model_result
    meta = st.session_state.meta
    spline = st.session_state.spline
    scales = st.session_state.scales
    posterior_cache = st.session_state.posterior_cache
    elasticity_df = st.session_state.elasticities
    diagnostics = st.session_state.model_diagnostics or {}
    
    # Get typed convergence diagnostics
    conv_diag = engine.summarize_convergence_diagnostics(idata)
    
    # Divergence warning banner
    if conv_diag.divergences > 0:
        st.warning(
            f"⚠️ {conv_diag.divergences} divergent transitions detected. "
            "Elasticity and scenario estimates should be treated as provisional. "
            "Re-fit with a higher target acceptance rate (e.g., 0.95) and inspect parameterisation."
        )
    else:
        st.success("✅ No divergent transitions detected.")
    
    # Convergence diagnostics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Convergence Diagnostics")
        st.metric("Divergences", conv_diag.divergences)
        if conv_diag.max_rhat is not None:
            st.metric("Max R-hat", f"{conv_diag.max_rhat:.3f}", 
                      delta="✅" if conv_diag.max_rhat < 1.01 else "⚠", 
                      delta_color="normal" if conv_diag.max_rhat < 1.01 else "inverse")
        if conv_diag.min_ess_bulk is not None:
            st.metric("Min ESS (bulk)", f"{conv_diag.min_ess_bulk:.0f}",
                      delta="✅" if conv_diag.min_ess_bulk > 400 else "⚠",
                      delta_color="normal" if conv_diag.min_ess_bulk > 400 else "inverse")
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
    
    # PPC plots
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
                    fig.add_trace(go.Scatter(x=y_obs, y=y_obs, mode="lines", name="Perfect fit", 
                                            line=dict(dash="dash", color="gray")))
                    fig.add_trace(go.Scatter(
                        x=y_obs, y=pred_median, mode="markers", name="Posterior median",
                        marker=dict(color="blue", size=4, opacity=0.6),
                        error_y=dict(type="data", symmetric=False, array=pred_p95 - pred_median, 
                                     arrayminus=pred_median - pred_p05, color="lightblue")
                    ))
                    fig.update_layout(height=400, xaxis_title="Observed (log velocity_std)", 
                                     yaxis_title="Predicted (log velocity_std)",
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
    
    # Elasticity Forest Plot
    st.markdown("### Elasticity Forest Plot")
    if elasticity_df is not None and not elasticity_df.empty:
        try:
            fig = engine.build_elasticity_forest_figure(elasticity_df=elasticity_df)
            st.plotly_chart(fig, use_container_width=True)
        except (KeyError, ValueError) as e:
            st.info(f"Elasticity forest plot unavailable: {e}")
    else:
        st.info("Elasticity estimates are unavailable for the current fitted model.")
    
    # Parameter summary table
    st.divider()
    st.markdown("### Parameter Summary")
    if elasticity_df is not None and not elasticity_df.empty:
        display_cols = [c for c in elasticity_df.columns if c in [
            "sku", "brand", "category", "elasticity_median", "elasticity_p05", "elasticity_p95",
            "nd_effect_p50", "shrinkage_flag"
        ]]
        st.dataframe(
            elasticity_df[display_cols].style.format({
                "elasticity_median": "{:.2f}",
                "elasticity_p05": "{:.2f}",
                "elasticity_p95": "{:.2f}",
                "nd_effect_p50": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page 4: Scenario cockpit
# ---------------------------------------------------------------------------

def create_scenario_action_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create the initial scenario action table from the prepared data."""
    action_df = df[["month", "retailer", "category", "brand", "sku",
                    "standard_volume", "price_per_standard_unit", "nd", "velocity_std",
                    "relative_price_std"]].copy()
    action_df["include"] = False
    action_df["price_change_pct"] = 0.0
    action_df["nd_change_pp"] = 0.0
    action_df["nd_mode"] = "pp"
    action_df["baseline_revenue"] = df["revenue"].values
    action_df["baseline_volume"] = df["standard_volume"].values
    return action_df


def render_scenario_action_editor(df: pd.DataFrame) -> pd.DataFrame:
    """Render the editable scenario action table using st.data_editor."""
    st.markdown("### Define Scenario Actions")
    st.caption(
        "Check **Include** to activate an action. Edit price_change_pct (%) and nd_change_pp (pp) per row. "
        "All checked actions run together in one scenario suite."
    )
    
    if "scenario_actions" not in st.session_state or st.session_state.scenario_actions is None:
        st.session_state.scenario_actions = create_scenario_action_table(df)
    
    action_df = st.session_state.scenario_actions.copy()
    
    key_cols = ["month", "retailer", "category", "brand", "sku"]
    if not action_df.empty:
        merged = df[key_cols].merge(
            action_df[key_cols + ["include", "price_change_pct", "nd_change_pp", "nd_mode"]],
            on=key_cols, how="left"
        )
        merged["include"] = merged["include"].fillna(False)
        merged["price_change_pct"] = merged["price_change_pct"].fillna(0.0)
        merged["nd_change_pp"] = merged["nd_change_pp"].fillna(0.0)
        merged["nd_mode"] = merged["nd_mode"].fillna("pp")
        
        merged["standard_volume"] = df["standard_volume"].values
        merged["price_per_standard_unit"] = df["price_per_standard_unit"].values
        merged["nd"] = df["nd"].values
        merged["velocity_std"] = df["velocity_std"].values
        merged["relative_price_std"] = df["relative_price_std"].values
        merged["baseline_revenue"] = df["revenue"].values
        merged["baseline_volume"] = df["standard_volume"].values
        
        action_df = merged
    
    column_config = {
        "include": st.column_config.CheckboxColumn(
            "Include",
            help="Check to include this SKU-month in the scenario",
            default=False,
        ),
        "month": st.column_config.TextColumn("Month", disabled=True),
        "retailer": st.column_config.TextColumn("Retailer", disabled=True),
        "category": st.column_config.TextColumn("Category", disabled=True),
        "brand": st.column_config.TextColumn("Brand", disabled=True),
        "sku": st.column_config.TextColumn("SKU", disabled=True),
        "standard_volume": st.column_config.NumberColumn("Std Volume", format="%,.0f", disabled=True),
        "price_per_standard_unit": st.column_config.NumberColumn("Price/Std Unit", format="%.2f", disabled=True),
        "nd": st.column_config.NumberColumn("ND", format="%.1%", disabled=True),
        "velocity_std": st.column_config.NumberColumn("Velocity Std", format="%.2f", disabled=True),
        "relative_price_std": st.column_config.NumberColumn("Rel Price Std", format="%.2f", disabled=True),
        "price_change_pct": st.column_config.NumberColumn(
            "Price Δ (%)", format="%.1f", step=1.0,
            help="Price change in percent (e.g., -10 = -10%)",
        ),
        "nd_change_pp": st.column_config.NumberColumn(
            "ND Δ (pp)", format="%.1f", step=1.0,
            help="ND change in percentage points (e.g., 5 = +5pp)",
        ),
        "nd_mode": st.column_config.SelectboxColumn(
            "ND Mode", options=["pp", "relative", "absolute"], default="pp",
            help="pp: percentage points, relative: % of current ND, absolute: target ND value",
        ),
        "baseline_revenue": st.column_config.NumberColumn("Revenue", format="%,.0f", disabled=True),
        "baseline_volume": st.column_config.NumberColumn("Volume", format="%,.0f", disabled=True),
    }
    
    display_cols = [
        "include", "month", "retailer", "category", "brand", "sku",
        "standard_volume", "price_per_standard_unit", "nd", "velocity_std", "relative_price_std",
        "price_change_pct", "nd_change_pp", "nd_mode",
        "baseline_revenue", "baseline_volume",
    ]
    
    edited_df = st.data_editor(
        action_df[display_cols],
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="scenario_action_editor",
    )
    
    st.session_state.scenario_actions = edited_df
    
    return edited_df


def run_scenario_from_actions(
    action_df: pd.DataFrame,
    df: pd.DataFrame,
    posterior_cache: dict,
    spline,
    scales,
    meta: dict,
) -> dict:
    """Run the scenario suite based on checked actions in the action table."""
    checked = action_df[action_df["include"]].copy()
    
    if checked.empty:
        st.warning("No actions selected. Check 'Include' for at least one row.")
        return {}
    
    scenario_df = df.copy()
    
    if len(checked) > 0:
        avg_price_change = checked["price_change_pct"].mean() / 100.0
        avg_nd_change = checked["nd_change_pp"].mean() / 100.0
        nd_mode = checked["nd_mode"].iloc[0]
    else:
        avg_price_change = 0.0
        avg_nd_change = 0.0
        nd_mode = "pp"
    
    if len(checked["sku"].unique()) == 1:
        target_level = "sku"
        target_value = checked["sku"].iloc[0]
    elif len(checked["brand"].unique()) == 1:
        target_level = "brand"
        target_value = checked["brand"].iloc[0]
    elif len(checked["retailer"].unique()) == 1:
        target_level = "retailer"
        target_value = checked["retailer"].iloc[0]
    elif len(checked["category"].unique()) == 1:
        target_level = "category"
        target_value = checked["category"].iloc[0]
    else:
        target_level = "market"
        target_value = None
    
    with st.spinner("Running scenario suite..."):
        scenarios = engine.run_scenario_suite(
            scenario_df, posterior_cache, spline, scales,
            price_change=avg_price_change,
            nd_change=avg_nd_change,
            nd_mode=nd_mode,
            target_level=target_level,
            target_value=target_value,
        )
    
    return scenarios


def render_scenario_cockpit_page(
    df: pd.DataFrame,
    view_df: pd.DataFrame,
    view_enriched: pd.DataFrame,
    idata,
    meta: dict,
    scales: dict,
    spline,
    posterior_cache: dict,
    elasticity_df,
    diagnostics: dict,
    start_month,
    end_month,
    view_category: str,
    view_retailer: str,
    view_brand: str,
) -> None:
    """Page 4: Scenario cockpit — editable action table, guardrails, scenario outputs, market impact."""
    st.subheader("Scenario Cockpit")
    
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
    
    action_df = render_scenario_action_editor(df)
    
    checked = action_df[action_df["include"]]
    if not checked.empty:
        st.markdown("#### Selected Actions Summary")
        summary_cols = ["month", "retailer", "category", "brand", "sku", 
                       "price_change_pct", "nd_change_pp", "nd_mode",
                       "baseline_volume", "baseline_revenue"]
        st.dataframe(checked[summary_cols].style.format({
            "price_change_pct": "{:+.1f}%",
            "nd_change_pp": "{:+.1f}pp",
            "baseline_volume": "{:,.0f}",
            "baseline_revenue": "{:,.0f}",
        }), use_container_width=True, hide_index=True)
        
        if st.button("Run Scenario Suite", type="primary", use_container_width=True):
            scenarios = run_scenario_from_actions(
                action_df, df, posterior_cache, spline, scales, meta
            )
            if scenarios:
                st.session_state.scenario_suite = scenarios
                st.session_state.scenario_cross_sensitivity = cross_cat_sensitivity
                st.rerun()
    else:
        st.info("No actions selected. Check 'Include' for at least one row to enable scenario run.")
    
    st.divider()
    
    if st.session_state.get("scenario_suite") is not None:
        scenarios = st.session_state.scenario_suite
        cross_cat_sensitivity = st.session_state.get("scenario_cross_sensitivity", 0.05)
        
        st.markdown("### Scenario Suite Results")
        
        level = st.selectbox(
            "Aggregation level",
            ["market", "retailer", "category", "brand", "sku"],
            index=2,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        
        compare_scenario = st.selectbox(
            "Compare scenario",
            ["price_only", "distribution_only", "combined"],
            index=2,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        baseline_df = scenarios["baseline"]
        scenario_df_selected = scenarios[compare_scenario]
        
        market_impact = engine.aggregate_market_impact(baseline_df, scenario_df_selected, level)
        
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
        
        st.markdown("#### Driver Waterfall")
        chart_subtitle("Which modelled components explain the scenario difference?")
        
        try:
            waterfall = engine.create_driver_waterfall(
                baseline_df, scenario_df_selected,
                posterior_cache, spline, scales, meta
            )
            st.plotly_chart(waterfall, use_container_width=True)
        except Exception as e:
            st.info(f"Driver waterfall unavailable: {e}")
        
        st.markdown("#### Market Visualizations")
        
        period_text = f"{start_month.strftime('%b %Y')} – {end_month.strftime('%b %Y')}"
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            chart_subtitle("Which categories gain or lose share vs. volume?")
            st.plotly_chart(
                engine.create_category_bubble_map(market_impact),
                use_container_width=True
            )
            render_scope_context("All observed categories in the affected market", period_text)
        
        with viz_col2:
            chart_subtitle("How do brand×pack combinations shift under the scenario?")
            st.plotly_chart(
                engine.create_brand_pack_heatmap(market_impact),
                use_container_width=True
            )
            render_scope_context("All observed brand×pack combinations", period_text)
        
        st.markdown("#### Winner/Loser Dumbbell")
        chart_subtitle("Which SKUs gain or lose the most standard volume?")
        try:
            dumbbell = engine.create_winner_loser_dumbbell(market_impact)
            st.plotly_chart(dumbbell, use_container_width=True)
        except Exception as e:
            st.info(f"Dumbbell chart unavailable: {e}")
        
        with st.expander("🔬 Advanced: Cross-Category Sensitivity", expanded=False):
            chart_subtitle("How does the substitution assumption affect other categories?")
            st.plotly_chart(
                engine.create_cross_category_sensitivity_chart(scenarios, cross_cat_sensitivity),
                use_container_width=True
            )
            st.caption("Cross-category sensitivity is an assumption-led multiplier. "
                       "Conservative: no substitution. Central: 5% of displaced volume reallocates. High: 15%.")
            render_scope_context("Other categories within the same retailer", period_text)
        
        checked = action_df[action_df["include"]]
        if len(checked["sku"].unique()) == 1:
            target_sku = checked["sku"].iloc[0]
            st.markdown("#### Parameter Attribution")
            sku_idx = meta["sku_to_idx"][target_sku]
            nd_base = checked["nd"].mean()
            nd_mode = checked["nd_mode"].iloc[0]
            nd_change = checked["nd_change_pp"].mean() / 100.0
            price_change = checked["price_change_pct"].mean() / 100.0
            
            if nd_mode == "pp":
                resolved_nd = engine.resolve_target_nd(np.array([nd_base]), nd_change, "pp")[0]
            elif nd_mode == "relative":
                resolved_nd = engine.resolve_target_nd(np.array([nd_base]), nd_change, "relative")[0]
            else:
                resolved_nd = nd_change
            
            attribution = engine.compute_parameter_attribution(
                posterior_cache, spline, scales, sku_idx,
                price_change, nd_base, resolved_nd,
                target_sku, meta
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
        
        st.markdown("#### Scenario Audit & Export")
        checked = action_df[action_df["include"]]
        audit_df = pd.DataFrame({
            "Setting": [
                "Target level", "Target value(s)", "Avg Price change", "ND mode", "Avg ND change",
                "Cross-category sensitivity", "Model config", "Chains", "Draws"
            ],
            "Value": [
                "Multiple SKUs" if len(checked) > 1 else "Single SKU",
                f"{len(checked)} SKU-months selected",
                f"{checked['price_change_pct'].mean():+.1f}%",
                checked["nd_mode"].iloc[0] if len(checked) > 0 else "pp",
                f"{checked['nd_change_pp'].mean():+.1f}pp",
                f"{cross_cat_sensitivity:.0%}",
                st.session_state.model_settings["config"].__class__.__name__,
                st.session_state.model_settings["config"].chains,
                st.session_state.model_settings["config"].draws,
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
                        "baseline_volume": row.get("baseline_units", row.get("standard_volume", 0)),
                        "scenario_volume": row.get("units_p50", row.get("standard_volume", 0)),
                        "baseline_revenue": row.get("baseline_revenue", row.get("revenue", 0)),
                        "scenario_revenue": row.get("revenue_p50", row.get("revenue", 0)),
                        "baseline_nd": row.get("baseline_nd", row.get("nd", 0)),
                        "scenario_nd": row.get("scenario_nd", row.get("nd", 0)),
                        "baseline_price": row.get("baseline_price", row.get("price_per_standard_unit", 0)),
                        "scenario_price": row.get("scenario_price", row.get("price_per_standard_unit", 0)),
                    })
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"scenario_suite_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()

    # Prepare data BEFORE rendering sidebar so sidebar can show model config
    if st.session_state.raw_data is not None:
        prepare_data_if_needed()

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

    if st.session_state.raw_data is None:
        st.info(
            "Start with **Use demo market** or upload your complete market dataset. "
            "The model needs competitor SKUs to calculate relative price and share."
        )
        # Show only market data page when no data
        render_market_data_page()
        return

    if st.session_state.prepared_data is None:
        st.error("Data preparation failed. Check the data quality report.")
        render_market_data_page()
        return

    if st.session_state.prepared_data is None:
        render_market_data_page()
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

    # Determine current workflow step
    has_model = st.session_state.model_result is not None
    has_scenario = st.session_state.get("scenario_suite") is not None

    if not has_model:
        current_step = 2  # Market diagnostics
    elif not has_scenario:
        current_step = 3  # Fit & validate
    else:
        current_step = 4  # Scenario cockpit

    render_workflow_banner(current_step)

    # 4-page navigation
    page_market, page_diagnostics, page_fit, page_scenario = st.tabs([
        "📊 1. Market data",
        "🔍 2. Market diagnostics",
        "🧠 3. Fit & validate",
        "⚙️ 4. Scenario cockpit",
    ])

    enriched_df = engine.build_retail_features(df)
    view_enriched = enriched_df.loc[view_mask].copy()

    with page_market:
        render_market_data_page(df, view_df, view_enriched, start_month, end_month)

    with page_diagnostics:
        render_market_diagnostics_page(view_df, view_enriched, start_month, end_month)

    with page_fit:
        if not has_model:
            st.info("Configure model in sidebar and click **Fit / Re-fit model** to proceed.")
        render_fit_validate_page(
            df, view_df, view_enriched, start_month, end_month
        )

    with page_scenario:
        if not has_model:
            st.warning("Fit the model first (page 3) to enable scenario simulation.")
        else:
            idata = st.session_state.model_result
            meta = st.session_state.meta
            spline = st.session_state.spline
            scales = st.session_state.scales
            posterior_cache = st.session_state.posterior_cache
            elasticity_df = st.session_state.elasticities
            diagnostics = st.session_state.model_diagnostics or {}
            render_scenario_cockpit_page(
                df, view_df, view_enriched, idata, meta, scales, spline,
                posterior_cache, elasticity_df, diagnostics,
                start_month, end_month, view_category, view_retailer, view_brand
            )


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
        "posterior_cache": None,
        "fitted_model": None,
        "scenario_suite": None,
        "scenario_actions": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


if __name__ == "__main__":
    main()
