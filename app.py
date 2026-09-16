"""
app.py - Streamlit frontend for Retail Demand and Distribution Analytics.

Uses engine.py as the analytical backend. Provides descriptive analytics,
Bayesian modelling, and scenario analysis for retail data.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import SplineTransformer

# Import engine functions
import engine
import demo_data


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Retail Demand and Distribution Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Constants and Configuration
# ============================================================

TABS = [
    "Overview",
    "Market and Categories",
    "Retailers",
    "Brands, SKUs and Pack Sizes",
    "Distribution and Velocity",
    "Bayesian Model",
    "Scenarios",
    "Data Quality",
]

# Colour sequences for consistent categorical mapping
RETAILER_COLOURS = px.colors.qualitative.Set2
BRAND_COLOURS = px.colors.qualitative.Set3
CATEGORY_COLOURS = px.colors.qualitative.T10

MODEL_PRESETS = {
    "Exploratory": {
        "draws": engine.DEFAULT_DRAWS,
        "tune": engine.DEFAULT_TUNE,
        "chains": engine.DEFAULT_CHAINS,
        "target_accept": engine.DEFAULT_TARGET_ACCEPT,
        "n_knots": engine.DEFAULT_N_SPLINE_KNOTS,
    },
    "Final": {
        "draws": engine.ADVANCED_DRAWS,
        "tune": engine.ADVANCED_TUNE,
        "chains": engine.ADVANCED_CHAINS,
        "target_accept": engine.DEFAULT_TARGET_ACCEPT,
        "n_knots": engine.DEFAULT_N_SPLINE_KNOTS,
    },
}


# ============================================================
# Utility Functions
# ============================================================

def compute_fingerprint(data: bytes) -> str:
    """Compute SHA256 fingerprint of data."""
    return hashlib.sha256(data).hexdigest()[:16]


def compute_filter_fingerprint(data_fp: str, filters: Dict) -> str:
    """Compute fingerprint combining data and filter state."""
    filter_str = json.dumps(filters, sort_keys=True, default=str)
    return compute_fingerprint((data_fp + filter_str).encode())


def invalidate_downstream(prefix: str) -> None:
    """Remove all session_state keys starting with prefix."""
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith(prefix)]
    for k in keys_to_delete:
        del st.session_state[k]


def format_compact_number(value: float) -> str:
    """Format number in compact notation (1.2K, 1.5M, 2.3B)."""
    if pd.isna(value):
        return "—"
    abs_val = abs(value)
    if abs_val >= 1e9:
        return f"{value/1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.1f}"


def format_pct(value: float) -> str:
    """Format as percentage."""
    if pd.isna(value):
        return "—"
    return f"{value:.1f}%"


def get_colour_map(values: List, palette: List) -> Dict:
    """Create consistent colour mapping for categorical values."""
    return {v: palette[i % len(palette)] for i, v in enumerate(sorted(values))}


# ============================================================
# Cached Data Operations
# ============================================================

@st.cache_data
def read_file_cached(file_bytes: bytes, file_type: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read CSV or XLSX from bytes."""
    bio = io.BytesIO(file_bytes)
    if file_type == "csv":
        return pd.read_csv(bio)
    else:
        return pd.read_excel(bio, sheet_name=sheet_name)


@st.cache_data
def get_xlsx_sheets(file_bytes: bytes) -> List[str]:
    """Get sheet names from XLSX."""
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    return xls.sheet_names


@st.cache_data
def validate_data_cached(raw_df: pd.DataFrame) -> engine.ValidationReport:
    """Validate input data."""
    return engine.validate_input_data(raw_df)


@st.cache_data
def prepare_data_cached(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Prepare data with derived metrics."""
    return engine.prepare_data(raw_df)


@st.cache_data
def add_indices_cached(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Add integer indices for modelling."""
    return engine.add_indices(df)


@st.cache_data
def fit_spline_cached(df: pd.DataFrame, n_knots: int, degree: int) -> Tuple[SplineTransformer, np.ndarray]:
    """Fit B-spline basis."""
    return engine.fit_spline(df, n_knots=n_knots, degree=degree)


@st.cache_data
def run_prep_pipeline(raw_df: pd.DataFrame, n_knots: int = engine.DEFAULT_N_SPLINE_KNOTS,
                       degree: int = engine.DEFAULT_SPLINE_DEGREE) -> Tuple:
    """Run full preparation pipeline: validate -> prepare -> add_indices -> fit_spline."""
    # Validate
    validation = engine.validate_input_data(raw_df)
    if not validation.is_valid:
        raise ValueError(f"Validation failed: {validation.invalid_reasons}")
    
    # Prepare
    prepared_df, scales = engine.prepare_data(raw_df)
    
    # Add indices
    indexed_df, meta = engine.add_indices(prepared_df)
    
    # Fit spline
    spline, nd_basis = engine.fit_spline(indexed_df, n_knots=n_knots, degree=degree)
    
    # Quality report
    quality_report = engine.create_data_quality_report(raw_df, indexed_df, validation)
    
    return indexed_df, scales, meta, spline, nd_basis, validation, quality_report


@st.cache_data
def calculate_shares_cached(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate market shares."""
    return engine.calculate_shares(df)


@st.cache_data
def create_growth_decomposition_cached(df: pd.DataFrame, start_month: Optional[pd.Timestamp] = None,
                                        end_month: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Create growth decomposition with optional period selection."""
    # Use engine's function but allow period override
    if start_month is None or end_month is None:
        return engine.create_growth_decomposition(df)
    
    # Custom period implementation
    start = df[df["month"] == start_month].set_index(["retailer", "sku"])
    end = df[df["month"] == end_month].set_index(["retailer", "sku"])
    common = start.index.intersection(end.index)
    
    if len(common) == 0:
        return pd.DataFrame()
    
    decomp = pd.DataFrame(index=common).reset_index()
    decomp["network"] = 100 * np.log(
        end.loc[common, "retailer_stores"].to_numpy() / start.loc[common, "retailer_stores"].to_numpy()
    )
    decomp["distribution"] = 100 * np.log(
        end.loc[common, "nd"].to_numpy() / start.loc[common, "nd"].to_numpy()
    )
    decomp["velocity"] = 100 * np.log(
        end.loc[common, "velocity_std"].to_numpy() / start.loc[common, "velocity_std"].to_numpy()
    )
    decomp["total_standard_volume"] = 100 * np.log(
        end.loc[common, "standard_volume"].to_numpy() / start.loc[common, "standard_volume"].to_numpy()
    )
    decomp["label"] = decomp["retailer"].astype(str) + " | " + decomp["sku"].astype(str)
    decomp = decomp.sort_values("total_standard_volume")
    
    return decomp


@st.cache_data
def create_market_summary_cached(df: pd.DataFrame) -> pd.DataFrame:
    return engine.create_market_summary(df)


@st.cache_data
def create_category_summary_cached(df: pd.DataFrame) -> pd.DataFrame:
    return engine.create_category_summary(df)


@st.cache_data
def create_retailer_summary_cached(df: pd.DataFrame) -> pd.DataFrame:
    return engine.create_retailer_summary(df)


@st.cache_data
def create_brand_summary_cached(df: pd.DataFrame) -> pd.DataFrame:
    return engine.create_brand_summary(df)


@st.cache_data
def create_sku_summary_cached(df: pd.DataFrame) -> pd.DataFrame:
    return engine.create_sku_summary(df)


@st.cache_data
def classify_dv_cached(df: pd.DataFrame, benchmark_level: str) -> pd.DataFrame:
    return engine.classify_distribution_velocity(df, benchmark_level)


@st.cache_data
def get_expansion_candidates_cached(df: pd.DataFrame) -> pd.DataFrame:
    return engine.get_expansion_candidates(df)


@st.cache_data
def generate_demo_data_cached(months_count: int = 24, seed: int = 42) -> pd.DataFrame:
    return demo_data.generate_demo_data(months_count, seed)


@st.cache_data
def get_input_template_cached() -> pd.DataFrame:
    return demo_data.get_input_template()


# ============================================================
# Chart Helper Functions
# ============================================================

def create_monthly_chart(df: pd.DataFrame, y_col: str, title: str, y_title: str,
                          color_col: Optional[str] = None, color_map: Optional[Dict] = None) -> go.Figure:
    """Create a monthly trend line chart."""
    if color_col and color_col in df.columns:
        fig = px.line(df, x="month", y=y_col, color=color_col, title=title,
                      color_discrete_map=color_map, markers=True)
    else:
        fig = px.line(df, x="month", y=y_col, title=title, markers=True)
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title=y_title,
        hovermode="x unified",
        legend_title_text="",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(tickformat="%b %Y")
    return fig


def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                      color_col: Optional[str] = None, color_map: Optional[Dict] = None,
                      orientation: str = "v") -> go.Figure:
    """Create a bar chart."""
    if orientation == "h":
        fig = px.bar(df, y=x_col, x=y_col, color=color_col, title=title,
                     color_discrete_map=color_map, orientation="h")
        fig.update_layout(yaxis_title="", xaxis_title=y_col, height=400)
    else:
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title,
                     color_discrete_map=color_map)
        fig.update_layout(xaxis_title="", yaxis_title=y_col, height=400)
    
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def create_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                          color_col: Optional[str] = None, color_map: Optional[Dict] = None,
                          size_col: Optional[str] = None, hover_data: Optional[List] = None) -> go.Figure:
    """Create a scatter plot."""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                     hover_data=hover_data, title=title, color_discrete_map=color_map)
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_heatmap(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, title: str) -> go.Figure:
    """Create a heatmap."""
    pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")
    fig = px.imshow(pivot, title=title, aspect="auto", color_continuous_scale="Blues")
    fig.update_layout(height=500, margin=dict(l=40, r=40, t=60, b=40))
    return fig


def create_forest_plot(elasticities: pd.DataFrame, title: str) -> go.Figure:
    """Create elasticity forest plot with credible intervals."""
    df = elasticities.sort_values("elasticity_median").copy()
    df["label"] = df["sku"] + " (" + df["brand"] + ", " + df["pack_group"] + ")"
    
    fig = go.Figure()
    
    # Add credible intervals
    fig.add_trace(go.Scatter(
        x=df["elasticity_p95"],
        y=df["label"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["elasticity_p05"],
        y=df["label"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.2)",
        showlegend=False,
        hoverinfo="skip",
    ))
    
    # Add median points
    fig.add_trace(go.Scatter(
        x=df["elasticity_median"],
        y=df["label"],
        mode="markers",
        marker=dict(size=8, color="steelblue"),
        name="Median",
        hovertemplate="SKU: %{y}<br>Elasticity: %{x:.3f}<extra></extra>",
    ))
    
    # Zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title=title,
        xaxis_title="Price Elasticity",
        yaxis_title="",
        height=max(400, len(df) * 25 + 100),
        margin=dict(l=200, r=40, t=60, b=40),
        showlegend=False,
    )
    return fig


def create_observed_vs_predicted(ppc_df: pd.DataFrame) -> go.Figure:
    """Create observed vs predicted scatter plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ppc_df["observed_z"],
        y=ppc_df["predicted_z"],
        mode="markers",
        marker=dict(size=4, opacity=0.5, color="steelblue"),
        name="Observations",
    ))
    
    # Perfect prediction line
    min_val = min(ppc_df["observed_z"].min(), ppc_df["predicted_z"].min())
    max_val = max(ppc_df["observed_z"].max(), ppc_df["predicted_z"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect Prediction",
    ))
    
    fig.update_layout(
        title="Posterior Predictive Check: Observed vs Predicted (z-scale)",
        xaxis_title="Observed (z)",
        yaxis_title="Predicted (z)",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_residual_histogram(ppc_df: pd.DataFrame) -> go.Figure:
    """Create residual histogram."""
    fig = px.histogram(ppc_df, x="residual_z", nbins=30, title="Residual Distribution (z-scale)")
    fig.update_layout(
        xaxis_title="Residual (z)",
        yaxis_title="Count",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ============================================================
# Data Loading and Initialization
# ============================================================

def initialize_session_state() -> None:
    """Initialize all session state keys."""
    defaults = {
        "raw_data": None,
        "raw_data_bytes": None,
        "worksheet_name": None,
        "prepared_data": None,
        "scales": None,
        "meta": None,
        "spline": None,
        "nd_basis": None,
        "validation_report": None,
        "quality_report": None,
        "data_fingerprint": None,
        "active_filters": {
            "categories": [],
            "retailers": [],
            "brands": [],
            "skus": [],
            "pack_sizes": [],
            "date_range": None,
        },
        "filtered_data": None,
        "filter_fingerprint": None,
        "model_result": None,
        "model_input_fingerprint": None,
        "model_settings": None,
        "model_runtime_seconds": None,
        "model_diagnostics": None,
        "elasticities_df": None,
        "ppc_df": None,
        "scenario_result": None,
        "scenario_agg": {},
        "scenario_settings": None,
        "scenario_fingerprint": None,
        "model_fit_status": None,  # None, "running", "success", "error"
        "model_fit_error": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_data(uploaded_file, sheet_name: Optional[str] = None) -> None:
    """Load uploaded file into session state."""
    file_bytes = uploaded_file.read()
    file_type = "xlsx" if uploaded_file.name.endswith(".xlsx") else "csv"
    
    # Store raw data
    st.session_state.raw_data_bytes = file_bytes
    st.session_state.raw_data = read_file_cached(file_bytes, file_type, sheet_name)
    st.session_state.worksheet_name = sheet_name
    st.session_state.data_fingerprint = compute_fingerprint(file_bytes)
    
    # Invalidate all downstream state
    invalidate_downstream("prepared_data")
    invalidate_downstream("scales")
    invalidate_downstream("meta")
    invalidate_downstream("spline")
    invalidate_downstream("nd_basis")
    invalidate_downstream("validation_report")
    invalidate_downstream("quality_report")
    invalidate_downstream("filtered_data")
    invalidate_downstream("filter_fingerprint")
    invalidate_downstream("model_result")
    invalidate_downstream("model_input_fingerprint")
    invalidate_downstream("model_settings")
    invalidate_downstream("model_runtime_seconds")
    invalidate_downstream("model_diagnostics")
    invalidate_downstream("elasticities_df")
    invalidate_downstream("ppc_df")
    invalidate_downstream("scenario_result")
    invalidate_downstream("scenario_agg")
    invalidate_downstream("scenario_settings")
    invalidate_downstream("scenario_fingerprint")
    invalidate_downstream("model_fit_status")
    invalidate_downstream("model_fit_error")


def load_demo_data() -> None:
    """Load generated demo data."""
    df = generate_demo_data_cached()
    csv_bytes = df.to_csv(index=False).encode()
    st.session_state.raw_data_bytes = csv_bytes
    st.session_state.raw_data = df
    st.session_state.worksheet_name = None
    st.session_state.data_fingerprint = compute_fingerprint(csv_bytes)
    
    # Invalidate all downstream state
    invalidate_downstream("prepared_data")
    invalidate_downstream("scales")
    invalidate_downstream("meta")
    invalidate_downstream("spline")
    invalidate_downstream("nd_basis")
    invalidate_downstream("validation_report")
    invalidate_downstream("quality_report")
    invalidate_downstream("filtered_data")
    invalidate_downstream("filter_fingerprint")
    invalidate_downstream("model_result")
    invalidate_downstream("model_input_fingerprint")
    invalidate_downstream("model_settings")
    invalidate_downstream("model_runtime_seconds")
    invalidate_downstream("model_diagnostics")
    invalidate_downstream("elasticities_df")
    invalidate_downstream("ppc_df")
    invalidate_downstream("scenario_result")
    invalidate_downstream("scenario_agg")
    invalidate_downstream("scenario_settings")
    invalidate_downstream("scenario_fingerprint")
    invalidate_downstream("model_fit_status")
    invalidate_downstream("model_fit_error")


def prepare_and_store() -> None:
    """Run preparation pipeline and store results."""
    raw_df = st.session_state.raw_data
    n_knots = st.session_state.get("model_n_knots", engine.DEFAULT_N_SPLINE_KNOTS)
    degree = engine.DEFAULT_SPLINE_DEGREE
    
    try:
        (prepared_df, scales, meta, spline, nd_basis,
         validation, quality_report) = run_prep_pipeline(raw_df, n_knots, degree)
        
        st.session_state.prepared_data = prepared_df
        st.session_state.scales = scales
        st.session_state.meta = meta
        st.session_state.spline = spline
        st.session_state.nd_basis = nd_basis
        st.session_state.validation_report = validation
        st.session_state.quality_report = quality_report
        st.session_state.filtered_data = prepared_df.copy()
        st.session_state.filter_fingerprint = compute_filter_fingerprint(
            st.session_state.data_fingerprint, st.session_state.active_filters
        )
        st.session_state.model_fit_status = None
        st.session_state.model_fit_error = None
    except Exception as e:
        st.error(f"Data preparation failed: {e}")
        st.session_state.prepared_data = None


def apply_filters() -> None:
    """Apply active filters to prepared data."""
    if st.session_state.prepared_data is None:
        return
    
    df = st.session_state.prepared_data.copy()
    filters = st.session_state.active_filters
    
    if filters["categories"]:
        df = df[df["category"].isin(filters["categories"])]
    if filters["retailers"]:
        df = df[df["retailer"].isin(filters["retailers"])]
    if filters["brands"]:
        df = df[df["brand"].isin(filters["brands"])]
    if filters["skus"]:
        df = df[df["sku"].isin(filters["skus"])]
    if filters["pack_sizes"]:
        df = df[df["pack_size"].isin(filters["pack_sizes"])]
    if filters["date_range"] and len(filters["date_range"]) == 2:
        start, end = filters["date_range"]
        df = df[(df["month"] >= pd.Timestamp(start)) & (df["month"] <= pd.Timestamp(end))]
    
    st.session_state.filtered_data = df
    st.session_state.filter_fingerprint = compute_filter_fingerprint(
        st.session_state.data_fingerprint, filters
    )
    
    # Invalidate model and scenario if filters changed
    if st.session_state.model_input_fingerprint != st.session_state.filter_fingerprint:
        invalidate_downstream("model_result")
        invalidate_downstream("model_input_fingerprint")
        invalidate_downstream("model_settings")
        invalidate_downstream("model_runtime_seconds")
        invalidate_downstream("model_diagnostics")
        invalidate_downstream("elasticities_df")
        invalidate_downstream("ppc_df")
        invalidate_downstream("scenario_result")
        invalidate_downstream("scenario_agg")
        invalidate_downstream("scenario_settings")
        invalidate_downstream("scenario_fingerprint")


def update_filter_options() -> None:
    """Update filter multiselect options based on current filtered data."""
    if st.session_state.filtered_data is None:
        return
    
    df = st.session_state.filtered_data
    filters = st.session_state.active_filters
    
    # Update available options (cascade)
    all_categories = sorted(df["category"].unique())
    all_retailers = sorted(df["retailer"].unique())
    all_brands = sorted(df["brand"].unique())
    all_skus = sorted(df["sku"].unique())
    all_pack_sizes = sorted(df["pack_size"].unique())
    date_min = df["month"].min()
    date_max = df["month"].max()
    
    # Store in session for sidebar rendering
    st.session_state.filter_options = {
        "categories": all_categories,
        "retailers": all_retailers,
        "brands": all_brands,
        "skus": all_skus,
        "pack_sizes": all_pack_sizes,
        "date_min": date_min,
        "date_max": date_max,
    }


def render_sidebar() -> None:
    """Render sidebar with all controls."""
    with st.sidebar:
        st.header("Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV or XLSX",
            type=["csv", "xlsx"],
            help="Required columns: month, retailer, category, brand, sku, units, revenue, sku_stores, retailer_stores, pack_size"
        )
        
        if uploaded_file is not None:
            file_type = "xlsx" if uploaded_file.name.endswith(".xlsx") else "csv"
            if file_type == "xlsx":
                sheet_names = get_xlsx_sheets(uploaded_file.getvalue())
                if len(sheet_names) > 1:
                    sheet_name = st.selectbox("Select worksheet", sheet_names)
                else:
                    sheet_name = sheet_names[0]
            else:
                sheet_name = None
            
            if st.button("Load File", type="primary"):
                load_data(uploaded_file, sheet_name)
                st.rerun()
        
        # Template download
        template_df = get_input_template_cached()
        st.download_button(
            "Download CSV Template",
            template_df.to_csv(index=False),
            "retail_data_template.csv",
            "text/csv",
            help="Empty template with required columns"
        )
        
        # Demo data
        if st.button("Load Demo Data"):
            load_demo_data()
            st.rerun()
        
        # Validation status
        if st.session_state.validation_report is not None:
            vr = st.session_state.validation_report
            if vr.is_valid:
                st.success(f"✓ Valid: {vr.info['valid_rows']} rows, {vr.info['n_months']} months")
            else:
                st.error("✗ Invalid data")
                for warn in vr.warnings:
                    st.warning(warn)
        
        st.divider()
        
        # Filters
        st.header("Filters")
        
        if st.session_state.filtered_data is not None:
            update_filter_options()
            opts = st.session_state.filter_options
            filters = st.session_state.active_filters
            
            # Category
            cats = st.multiselect("Category", opts["categories"], 
                                  default=filters["categories"],
                                  key="filter_category")
            if cats != filters["categories"]:
                st.session_state.active_filters["categories"] = cats
                apply_filters()
                st.rerun()
            
            # Retailer
            rets = st.multiselect("Retailer", opts["retailers"],
                                  default=filters["retailers"],
                                  key="filter_retailer")
            if rets != filters["retailers"]:
                st.session_state.active_filters["retailers"] = rets
                apply_filters()
                st.rerun()
            
            # Brand
            brands = st.multiselect("Brand", opts["brands"],
                                    default=filters["brands"],
                                    key="filter_brand")
            if brands != filters["brands"]:
                st.session_state.active_filters["brands"] = brands
                apply_filters()
                st.rerun()
            
            # SKU
            skus = st.multiselect("SKU", opts["skus"],
                                  default=filters["skus"],
                                  key="filter_sku")
            if skus != filters["skus"]:
                st.session_state.active_filters["skus"] = skus
                apply_filters()
                st.rerun()
            
            # Pack Size
            packs = st.multiselect("Pack Size", opts["pack_sizes"],
                                   default=filters["pack_sizes"],
                                   key="filter_pack")
            if packs != filters["pack_sizes"]:
                st.session_state.active_filters["pack_sizes"] = packs
                apply_filters()
                st.rerun()
            
            # Date Range
            date_range = st.date_input(
                "Date Range",
                value=(opts["date_min"], opts["date_max"]),
                min_value=opts["date_min"],
                max_value=opts["date_max"],
                key="filter_date"
            )
            if date_range != filters["date_range"]:
                st.session_state.active_filters["date_range"] = date_range
                apply_filters()
                st.rerun()
            
            # Clear filters
            if st.button("Clear All Filters"):
                st.session_state.active_filters = {
                    "categories": [],
                    "retailers": [],
                    "brands": [],
                    "skus": [],
                    "pack_sizes": [],
                    "date_range": None,
                }
                apply_filters()
                st.rerun()
            
            # Filter caption
            df = st.session_state.filtered_data
            st.caption(
                f"Date: {df['month'].min().strftime('%b %Y')} – {df['month'].max().strftime('%b %Y')}  |  "
                f"Rows: {len(df):,}  |  "
                f"Categories: {df['category'].nunique()}  |  "
                f"Retailers: {df['retailer'].nunique()}  |  "
                f"Brands: {df['brand'].nunique()}  |  "
                f"SKUs: {df['sku'].nunique()}"
            )
        
        st.divider()
        
        # Model Settings
        st.header("Model Settings")
        
        if st.session_state.filtered_data is not None:
            n_obs = len(st.session_state.filtered_data)
            st.caption(f"Model-ready observations: {n_obs:,}")
            if n_obs < 100:
                st.warning("⚠ Few observations; model may be unstable")
            elif n_obs < 500:
                st.info("ℹ Moderate observation count")
            
            # Presets
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Exploratory", use_container_width=True):
                    for k, v in MODEL_PRESETS["Exploratory"].items():
                        st.session_state[f"model_{k}"] = v
                    st.rerun()
            with col2:
                if st.button("Final", use_container_width=True):
                    for k, v in MODEL_PRESETS["Final"].items():
                        st.session_state[f"model_{k}"] = v
                    st.rerun()
            
            # Manual controls
            draws = st.number_input("Draws", 100, 5000, 
                                    value=st.session_state.get("model_draws", engine.DEFAULT_DRAWS),
                                    key="model_draws")
            tune = st.number_input("Tune", 100, 5000,
                                   value=st.session_state.get("model_tune", engine.DEFAULT_TUNE),
                                   key="model_tune")
            chains = st.number_input("Chains", 1, 8,
                                     value=st.session_state.get("model_chains", engine.DEFAULT_CHAINS),
                                     key="model_chains")
            target_accept = st.number_input("Target Accept", 0.8, 0.99,
                                            value=st.session_state.get("model_target_accept", engine.DEFAULT_TARGET_ACCEPT),
                                            step=0.01,
                                            key="model_target_accept")
            n_knots = st.number_input("Spline Knots", 2, 10,
                                      value=st.session_state.get("model_n_knots", engine.DEFAULT_N_SPLINE_KNOTS),
                                      key="model_n_knots")
            
            # Fit button
            model_fp = compute_filter_fingerprint(
                st.session_state.data_fingerprint, st.session_state.active_filters
            )
            model_ready = st.session_state.model_input_fingerprint == model_fp
            
            if st.button("Fit Bayesian Model", type="primary", disabled=model_ready or st.session_state.model_fit_status == "running"):
                run_model_fitting()
        
        st.divider()
        
        # Scenario Settings
        st.header("Scenario Settings")
        
        if st.session_state.model_result is not None:
            price_change = st.number_input("Price Change (%)", -50, 100, 5, step=1, key="scenario_price") / 100
            nd_change = st.number_input("ND Change (%)", -90, 100, 10, step=1, key="scenario_nd") / 100
            
            # Scope filters
            df = st.session_state.filtered_data
            cat_scope = st.multiselect("Category", sorted(df["category"].unique()), key="scenario_cat")
            ret_scope = st.multiselect("Retailer", sorted(df["retailer"].unique()), key="scenario_ret")
            brand_scope = st.multiselect("Brand", sorted(df["brand"].unique()), key="scenario_brand")
            sku_scope = st.multiselect("SKU", sorted(df["sku"].unique()), key="scenario_sku")
            pack_scope = st.multiselect("Pack Size", sorted(df["pack_size"].unique()), key="scenario_pack")
            
            if st.button("Run Scenario", type="primary"):
                run_scenario(price_change, nd_change, cat_scope, ret_scope, brand_scope, sku_scope, pack_scope)
        else:
            st.info("Fit a model first to enable scenarios")
        
        st.divider()
        
        # Downloads
        st.header("Downloads")
        render_download_buttons()


def render_download_buttons() -> None:
    """Render download buttons for all available outputs."""
    if st.session_state.prepared_data is not None:
        st.download_button(
            "Prepared Data (CSV)",
            st.session_state.prepared_data.to_csv(index=False),
            "prepared_data.csv",
            "text/csv",
        )
    
    if st.session_state.quality_report is not None:
        st.download_button(
            "Quality Report (JSON)",
            json.dumps(st.session_state.quality_report, indent=2, default=str),
            "quality_report.json",
            "application/json",
        )
    
    if st.session_state.elasticities_df is not None:
        st.download_button(
            "Elasticities (CSV)",
            st.session_state.elasticities_df.to_csv(index=False),
            "elasticities.csv",
            "text/csv",
        )
    
    if st.session_state.scenario_agg:
        for level, df in st.session_state.scenario_agg.items():
            if not df.empty:
                name = level if level else "total_market"
                st.download_button(
                    f"Scenario {name} (CSV)",
                    df.to_csv(index=False),
                    f"scenario_{name}.csv",
                    "text/csv",
                )
    
    if st.session_state.model_result is not None:
        # Serialize trace to bytes
        bio = io.BytesIO()
        st.session_state.model_result.to_netcdf(bio)
        st.download_button(
            "Model Trace (NetCDF)",
            bio.getvalue(),
            "model_trace.nc",
            "application/octet-stream",
        )


def run_model_fitting() -> None:
    """Run model fitting in background thread."""
    st.session_state.model_fit_status = "running"
    st.session_state.model_fit_error = None
    
    def fit_worker():
        try:
            df = st.session_state.filtered_data
            meta = st.session_state.meta
            nd_basis = st.session_state.nd_basis
            scales = st.session_state.scales
            
            settings = {
                "draws": st.session_state.model_draws,
                "tune": st.session_state.model_tune,
                "chains": st.session_state.model_chains,
                "target_accept": st.session_state.model_target_accept,
                "n_knots": st.session_state.model_n_knots,
            }
            
            start_time = time.time()
            
            # Build and fit model
            model = engine.build_pymc_model(df, meta, nd_basis)
            trace = engine.fit_model(
                model,
                draws=settings["draws"],
                tune=settings["tune"],
                chains=settings["chains"],
                target_accept=settings["target_accept"],
            )
            
            runtime = time.time() - start_time
            
            # Extract diagnostics and elasticities
            diagnostics = engine.get_model_diagnostics(trace)
            elasticities = engine.extract_elasticities(trace, meta, scales)
            ppc = engine.get_posterior_predictive_check(trace, df)
            
            # Store results
            st.session_state.model_result = trace
            st.session_state.model_input_fingerprint = compute_filter_fingerprint(
                st.session_state.data_fingerprint, st.session_state.active_filters
            )
            st.session_state.model_settings = settings
            st.session_state.model_runtime_seconds = runtime
            st.session_state.model_diagnostics = diagnostics
            st.session_state.elasticities_df = elasticities
            st.session_state.ppc_df = ppc
            st.session_state.model_fit_status = "success"
            
            # Invalidate scenarios
            invalidate_downstream("scenario_result")
            invalidate_downstream("scenario_agg")
            invalidate_downstream("scenario_settings")
            invalidate_downstream("scenario_fingerprint")
            
        except Exception as e:
            st.session_state.model_fit_status = "error"
            st.session_state.model_fit_error = str(e)
    
    thread = threading.Thread(target=fit_worker)
    thread.start()


def run_scenario(price_change: float, nd_change: float,
                 cat_scope: List, ret_scope: List, brand_scope: List,
                 sku_scope: List, pack_scope: List) -> None:
    """Run scenario analysis."""
    df = st.session_state.filtered_data.copy()
    
    # Apply scope filters
    if cat_scope:
        df = df[df["category"].isin(cat_scope)]
    if ret_scope:
        df = df[df["retailer"].isin(ret_scope)]
    if brand_scope:
        df = df[df["brand"].isin(brand_scope)]
    if sku_scope:
        df = df[df["sku"].isin(sku_scope)]
    if pack_scope:
        df = df[df["pack_size"].isin(pack_scope)]
    
    if df.empty:
        st.warning("No data matches scenario scope")
        return
    
    # Check guardrails
    guardrails = demo_data.check_scenario_guardrails(df, price_change, nd_change)
    for warn in guardrails["warnings"]:
        st.warning(warn)
    
    # Run scenario
    trace = st.session_state.model_result
    spline = st.session_state.spline
    scales = st.session_state.scales
    
    scenario_df = engine.create_price_distribution_scenario(
        trace, df, spline, scales, price_change, nd_change
    )
    
    # Aggregate at multiple levels
    agg_results = {
        "total_market": engine.aggregate_scenario(scenario_df, []),
        "category": engine.aggregate_scenario(scenario_df, ["category"]),
        "retailer": engine.aggregate_scenario(scenario_df, ["retailer"]),
        "brand": engine.aggregate_scenario(scenario_df, ["brand"]),
        "sku_pack": engine.aggregate_scenario(scenario_df, ["sku", "pack_size"]),
    }
    
    st.session_state.scenario_result = scenario_df
    st.session_state.scenario_agg = agg_results
    st.session_state.scenario_settings = {
        "price_change": price_change,
        "nd_change": nd_change,
        "scope": {
            "categories": cat_scope,
            "retailers": ret_scope,
            "brands": brand_scope,
            "skus": sku_scope,
            "pack_sizes": pack_scope,
        }
    }
    st.session_state.scenario_fingerprint = compute_filter_fingerprint(
        st.session_state.model_input_fingerprint,
        {"price": price_change, "nd": nd_change, "scope": st.session_state.scenario_settings["scope"]}
    )


# ============================================================
# Tab Rendering Functions
# ============================================================

def render_overview_tab() -> None:
    """Tab 1: Overview."""
    if st.session_state.filtered_data is None:
        st.info("Upload data to begin")
        return
    
    df = st.session_state.filtered_data
    
    # KPIs
    market = create_market_summary_cached(df)
    latest = market.iloc[-1] if len(market) > 0 else None
    first = market.iloc[0] if len(market) > 0 else None
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Revenue", format_compact_number(df["revenue"].sum()))
        st.metric("Total Units", format_compact_number(df["units"].sum()))
    with col2:
        st.metric("Total Std Volume", format_compact_number(df["standard_volume"].sum()))
        st.metric("Avg Price/Std Unit", f"{df['price_std'].mean():.2f}")
    with col3:
        st.metric("Avg Numeric Distribution", f"{df['nd'].mean():.1%}")
        st.metric("Avg Std Velocity", f"{df['velocity_std'].mean():.1f}")
    with col4:
        st.metric("Retailers", df["retailer"].nunique())
        st.metric("Brands", df["brand"].nunique())
    with col5:
        st.metric("SKUs", df["sku"].nunique())
        st.metric("Date Range", f"{df['month'].min().strftime('%b %Y')} – {df['month'].max().strftime('%b %Y')}")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_monthly_chart(market, "revenue", "Monthly Total Revenue", "Revenue")
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_monthly_chart(market, "standard_volume", "Monthly Total Standard Volume", "Standard Volume")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = create_monthly_chart(market, "units", "Monthly Total Units", "Units")
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = create_monthly_chart(market, "price_per_standard_unit", "Monthly Price per Standard Unit", "Price")
        st.plotly_chart(fig4, use_container_width=True)
    
    # Main observations
    st.subheader("Main Observations")
    if first is not None and latest is not None:
        rev_growth = 100 * (latest["revenue"] / first["revenue"] - 1)
        vol_growth = 100 * (latest["standard_volume"] / first["standard_volume"] - 1)
        
        top_retailer = df.groupby("retailer")["revenue"].sum().idxmax()
        top_brand = df.groupby("brand")["revenue"].sum().idxmax()
        top_sku = df.groupby("sku")["revenue"].sum().idxmax()
        
        st.write(f"• Revenue growth ({first['month'].strftime('%b %Y')} → {latest['month'].strftime('%b %Y')}): {rev_growth:.1f}%")
        st.write(f"• Standard volume growth: {vol_growth:.1f}%")
        st.write(f"• Top retailer by revenue: {top_retailer}")
        st.write(f"• Top brand by revenue: {top_brand}")
        st.write(f"• Top SKU by revenue: {top_sku}")


def render_market_categories_tab() -> None:
    """Tab 2: Market and Categories."""
    if st.session_state.filtered_data is None:
        st.info("Upload data to begin")
        return
    
    df = st.session_state.filtered_data
    n_cats = df["category"].nunique()
    
    if n_cats == 1:
        st.info("One category is currently selected; category-share comparison is not applicable.")
    
    # Charts
    cat_summary = create_category_summary_cached(df)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_monthly_chart(cat_summary, "revenue", "Revenue by Category over Time", "Revenue",
                                    color_col="category", color_map=get_colour_map(df["category"].unique(), CATEGORY_COLOURS))
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_monthly_chart(cat_summary, "standard_volume", "Standard Volume by Category over Time", "Standard Volume",
                                    color_col="category", color_map=get_colour_map(df["category"].unique(), CATEGORY_COLOURS))
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if n_cats > 1:
            # Calculate shares (modifies df in place)
            engine.calculate_shares(df)
            cat_share_df = df[["month", "category", "category_revenue_share_total"]].drop_duplicates()
            fig3 = create_monthly_chart(cat_share_df, "category_revenue_share_total", "Category Value Share Trend", "Revenue Share",
                                        color_col="category", color_map=get_colour_map(df["category"].unique(), CATEGORY_COLOURS))
            st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = create_monthly_chart(cat_summary, "price_per_standard_unit", "Category Average Price per Standard Unit", "Price",
                                    color_col="category", color_map=get_colour_map(df["category"].unique(), CATEGORY_COLOURS))
        st.plotly_chart(fig4, use_container_width=True)
    
    # Category summary table
    st.subheader("Category Summary")
    if len(cat_summary) > 0:
        # Add growth vs prior period
        cat_summary_display = cat_summary.copy()
        if len(cat_summary) > 1:
            first_month = cat_summary["month"].min()
            last_month = cat_summary["month"].max()
            first_vals = cat_summary[cat_summary["month"] == first_month].set_index("category")
            last_vals = cat_summary[cat_summary["month"] == last_month].set_index("category")
            common_cats = first_vals.index.intersection(last_vals.index)
            
            growth = {}
            for cat in common_cats:
                growth[cat] = 100 * (last_vals.loc[cat, "revenue"] / first_vals.loc[cat, "revenue"] - 1)
            
            cat_summary_display["revenue_growth_pct"] = cat_summary_display["category"].map(growth)
        
        # Format for display
        display_cols = ["category", "revenue", "units", "standard_volume", "price_per_standard_unit",
                        "nd", "velocity_std", "revenue_growth_pct"] if "revenue_growth_pct" in cat_summary_display.columns else \
                       ["category", "revenue", "units", "standard_volume", "price_per_standard_unit", "nd", "velocity_std"]
        
        st.dataframe(
            cat_summary_display[display_cols].style.format({
                "revenue": format_compact_number,
                "units": format_compact_number,
                "standard_volume": format_compact_number,
                "price_per_standard_unit": "{:.2f}",
                "nd": "{:.1%}",
                "velocity_std": "{:.1f}",
                "revenue_growth_pct": "{:.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )
        
        st.download_button(
            "Download Category Summary",
            cat_summary.to_csv(index=False),
            "category_summary.csv",
            "text/csv",
        )


def render_retailers_tab() -> None:
    """Tab 3: Retailers."""
    if st.session_state.filtered_data is None:
        st.info("Upload data to begin")
        return
    
    df = st.session_state.filtered_data
    
    # Retailer summaries
    ret_summary = create_retailer_summary_cached(df)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_monthly_chart(ret_summary, "revenue", "Revenue Trend by Retailer", "Revenue",
                                    color_col="retailer", color_map=get_colour_map(df["retailer"].unique(), RETAILER_COLOURS))
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_monthly_chart(ret_summary, "standard_volume", "Standard Volume Trend by Retailer", "Standard Volume",
                                    color_col="retailer", color_map=get_colour_map(df["retailer"].unique(), RETAILER_COLOURS))
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        # Calculate shares (modifies df in place)
        engine.calculate_shares(df)
        ret_share_df = df[["month", "retailer", "retailer_revenue_share"]].drop_duplicates()
        fig3 = create_monthly_chart(ret_share_df, "retailer_revenue_share", "Retailer Value Share Trend", "Revenue Share",
                                    color_col="retailer", color_map=get_colour_map(df["retailer"].unique(), RETAILER_COLOURS))
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = create_monthly_chart(ret_summary, "price_per_standard_unit", "Average Price per Standard Unit by Retailer", "Price",
                                    color_col="retailer", color_map=get_colour_map(df["retailer"].unique(), RETAILER_COLOURS))
        st.plotly_chart(fig4, use_container_width=True)
    
    # Retailer summary table
    st.subheader("Retailer Summary")
    st.dataframe(
        ret_summary.style.format({
            "revenue": format_compact_number,
            "units": format_compact_number,
            "standard_volume": format_compact_number,
            "price_per_standard_unit": "{:.2f}",
            "nd": "{:.1%}",
            "velocity_std": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button("Download Retailer Summary", ret_summary.to_csv(index=False),
                       "retailer_summary.csv", "text/csv")
    
    # Heatmaps
    st.subheader("Retailer × SKU Heatmaps")
    months_available = sorted(df["month"].unique())
    selected_month = st.selectbox("Select Month", months_available, format_func=lambda x: x.strftime("%b %Y"))
    
    month_df = df[df["month"] == selected_month]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = create_heatmap(month_df, "sku", "retailer", "nd", f"Numeric Distribution ({selected_month.strftime('%b %Y')})")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_heatmap(month_df, "sku", "retailer", "velocity_std", f"Standard Velocity ({selected_month.strftime('%b %Y')})")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = create_heatmap(month_df, "sku", "retailer", "relative_price", f"Relative Price ({selected_month.strftime('%b %Y')})")
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap source downloads
    st.download_button("Download ND Heatmap Data", month_df[["retailer", "sku", "nd"]].to_csv(index=False),
                       "heatmap_nd.csv", "text/csv")
    st.download_button("Download Velocity Heatmap Data", month_df[["retailer", "sku", "velocity_std"]].to_csv(index=False),
                       "heatmap_velocity.csv", "text/csv")
    st.download_button("Download Rel Price Heatmap Data", month_df[["retailer", "sku", "relative_price"]].to_csv(index=False),
                       "heatmap_rel_price.csv", "text/csv")


def render_brands_skus_tab() -> None:
    """Tab 4: Brands, SKUs and Pack Sizes."""
    if st.session_state.filtered_data is None:
        st.info("Upload data to begin")
        return
    
    df = st.session_state.filtered_data
    
    # Brand trends
    brand_summary = create_brand_summary_cached(df)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_monthly_chart(brand_summary, "revenue", "Brand Revenue Trend", "Revenue",
                                    color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS))
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_monthly_chart(brand_summary, "standard_volume", "Brand Standard Volume Trend", "Standard Volume",
                                    color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS))
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        # Calculate shares (modifies df in place)
        engine.calculate_shares(df)
        brand_share_df = df[["month", "brand", "brand_revenue_share_total"]].drop_duplicates()
        fig3 = create_monthly_chart(brand_share_df, "brand_revenue_share_total", "Brand Value Share Trend", "Revenue Share",
                                    color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS))
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = create_monthly_chart(brand_summary, "price_per_standard_unit", "Brand Price Position Trend", "Price",
                                    color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS))
        st.plotly_chart(fig4, use_container_width=True)
    
    # SKU charts
    sku_summary = create_sku_summary_cached(df)
    
    col1, col2 = st.columns(2)
    with col1:
        # Revenue by SKU (top 20)
        top_sku_rev = sku_summary.nlargest(20, "revenue")
        fig5 = create_bar_chart(top_sku_rev, "sku", "revenue", "Revenue by SKU (Top 20)",
                                color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS),
                                orientation="h")
        st.plotly_chart(fig5, use_container_width=True)
        
        # Revenue by Brand × Pack Size
        brand_pack = df.groupby(["brand", "pack_size"], as_index=False)["revenue"].sum()
        fig6 = create_bar_chart(brand_pack, "brand", "revenue", "Revenue by Brand × Pack Size",
                                color_col="pack_size", orientation="v")
        st.plotly_chart(fig6, use_container_width=True)
    with col2:
        # Standard Volume by SKU (top 20)
        top_sku_vol = sku_summary.nlargest(20, "standard_volume")
        fig7 = create_bar_chart(top_sku_vol, "sku", "standard_volume", "Standard Volume by SKU (Top 20)",
                                color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS),
                                orientation="h")
        st.plotly_chart(fig7, use_container_width=True)
        
        # Pack Size Mix by Brand
        pack_mix = df.groupby(["brand", "pack_size"], as_index=False)["units"].sum()
        fig8 = create_bar_chart(pack_mix, "brand", "units", "Pack Size Mix by Brand (Units)",
                                color_col="pack_size", orientation="v")
        st.plotly_chart(fig8, use_container_width=True)
    
    # Detailed SKU table
    st.subheader("SKU Detail Table")
    display_cols = ["category", "retailer", "brand", "sku", "pack_size", "units", "revenue",
                    "standard_volume", "price_per_standard_unit", "competitor_price_std",
                    "relative_price", "nd", "velocity_std", "unit_share", "revenue_share"]
    
    # Ensure all columns exist
    available_cols = [c for c in display_cols if c in sku_summary.columns]
    missing = [c for c in display_cols if c not in sku_summary.columns]
    if missing:
        # Add competitor_price_std if missing (it's in df but may not be in sku_summary)
        pass
    
    st.dataframe(
        sku_summary[available_cols].style.format({
            "revenue": format_compact_number,
            "units": format_compact_number,
            "standard_volume": format_compact_number,
            "price_per_standard_unit": "{:.2f}",
            "competitor_price_std": "{:.2f}",
            "relative_price": "{:.2f}",
            "nd": "{:.1%}",
            "velocity_std": "{:.1f}",
            "unit_share": "{:.1%}",
            "revenue_share": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    
    st.download_button("Download SKU Detail Table", sku_summary.to_csv(index=False),
                       "sku_detail.csv", "text/csv")


def render_distribution_velocity_tab() -> None:
    """Tab 5: Distribution and Velocity."""
    if st.session_state.filtered_data is None:
        st.info("Upload data to begin")
        return
    
    df = st.session_state.filtered_data
    
    # Classification thresholds
    with st.expander("Classification Thresholds"):
        nd_threshold = st.slider("ND Threshold (percentile)", 0, 100, 50, key="dv_nd_thresh")
        rv_threshold = st.slider("Relative Velocity Threshold (percentile)", 0, 100, 50, key="dv_rv_thresh")
    
    # Classify
    classified = classify_dv_cached(df, "retailer_category")
    
    # Override with user thresholds
    nd_thresh_val = classified["nd"].quantile(nd_threshold / 100)
    rv_thresh_val = classified["relative_velocity"].quantile(rv_threshold / 100)
    
    def classify_custom(row):
        if row["nd"] < nd_thresh_val and row["relative_velocity"] > rv_thresh_val:
            return "Expansion candidate"
        elif row["nd"] >= nd_thresh_val and row["relative_velocity"] > rv_thresh_val:
            return "Scale efficiently"
        elif row["nd"] < nd_thresh_val and row["relative_velocity"] <= rv_thresh_val:
            return "Diagnose"
        else:
            return "Rationalise/review"
    
    classified["dv_quadrant"] = classified.apply(classify_custom, axis=1)
    
    # Scatter charts
    col1, col2 = st.columns(2)
    with col1:
        hover_cols = ["retailer", "category", "brand", "sku", "pack_size", "revenue", "nd", "velocity_std", "price_std"]
        available_hover = [c for c in hover_cols if c in classified.columns]
        
        fig1 = create_scatter_chart(classified, "nd", "velocity_std", "Numeric Distribution vs Standard Velocity",
                                    color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS),
                                    size_col="revenue", hover_data=available_hover)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_scatter_chart(classified, "relative_price", "velocity_std", "Relative Price vs Standard Velocity",
                                    color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS),
                                    hover_data=available_hover)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = create_scatter_chart(classified, "nd", "relative_velocity", "Distribution Opportunity Matrix",
                                    color_col="brand", color_map=get_colour_map(df["brand"].unique(), BRAND_COLOURS),
                                    size_col="revenue", hover_data=available_hover)
        st.plotly_chart(fig3, use_container_width=True)
    
    # Expansion candidates
    st.subheader("Expansion Candidates")
    expansion = classified[classified["dv_quadrant"] == "Expansion candidate"].copy()
    if len(expansion) > 0:
        expansion = expansion.sort_values(["relative_velocity", "nd", "revenue"], ascending=[False, True, False])
        st.dataframe(
            expansion[["retailer", "category", "brand", "sku", "pack_size", "revenue", "nd", 
                       "velocity_std", "relative_velocity", "dv_quadrant"]].style.format({
                "revenue": format_compact_number,
                "nd": "{:.1%}",
                "velocity_std": "{:.1f}",
                "relative_velocity": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button("Download Expansion Candidates", expansion.to_csv(index=False),
                           "expansion_candidates.csv", "text/csv")
    else:
        st.info("No expansion candidates found with current thresholds")
    
    # Growth decomposition
    st.subheader("Growth Decomposition")
    months = sorted(df["month"].unique())
    if len(months) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            start_month = st.selectbox("Start Month", months[:-1], format_func=lambda x: x.strftime("%b %Y"))
        with col2:
            end_month = st.selectbox("End Month", months[1:], index=len(months)-2, format_func=lambda x: x.strftime("%b %Y"))
        
        if start_month < end_month:
            decomp = create_growth_decomposition_cached(df, start_month, end_month)
            if not decomp.empty:
                st.dataframe(
                    decomp.style.format({
                        "network": "{:.1f}",
                        "distribution": "{:.1f}",
                        "velocity": "{:.1f}",
                        "total_standard_volume": "{:.1f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
                
                # Aggregated views
                col1, col2 = st.columns(2)
                with col1:
                    ret_agg = decomp.groupby("retailer")[["network", "distribution", "velocity", "total_standard_volume"]].mean().reset_index()
                    st.write("**By Retailer (avg %)**")
                    st.dataframe(ret_agg.style.format({"network": "{:.1f}", "distribution": "{:.1f}", "velocity": "{:.1f}", "total_standard_volume": "{:.1f}"}), width="stretch", hide_index=True)
                with col2:
                    brand_agg = decomp.groupby("sku")[["network", "distribution", "velocity", "total_standard_volume"]].mean().reset_index()
                    # Merge brand info
                    brand_map = df[["sku", "brand"]].drop_duplicates()
                    brand_agg = brand_agg.merge(brand_map, on="sku")
                    brand_agg = brand_agg.groupby("brand")[["network", "distribution", "velocity", "total_standard_volume"]].mean().reset_index()
                    st.write("**By Brand (avg %)**")
                    st.dataframe(brand_agg.style.format({"network": "{:.1f}", "distribution": "{:.1f}", "velocity": "{:.1f}", "total_standard_volume": "{:.1f}"}), width="stretch", hide_index=True)
                
                st.download_button("Download Decomposition", decomp.to_csv(index=False),
                                   "growth_decomposition.csv", "text/csv")
            else:
                st.info("No common retailer-SKU pairs between selected periods")
    else:
        st.info("Insufficient months for decomposition")


def render_bayesian_model_tab() -> None:
    """Tab 6: Bayesian Model."""
    if st.session_state.filtered_data is None:
        st.info("Upload data to begin")
        return
    
    df = st.session_state.filtered_data
    
    # Pre-fit panel
    if st.session_state.model_result is None:
        st.subheader("Model Specification")
        with st.expander("Mathematical Description"):
            st.latex(r"""
            \log(V^{\text{std}}_{r,s,t}) = \alpha + \eta_{r,s} + \mu_t + 
            \beta_{s} \log(P^{\text{rel}}_{r,s,t}) + f(\text{ND}_{r,s,t}) + \varepsilon_{r,s,t}
            """)
            st.write(r"""
            - $\alpha$: Global intercept
            - $\eta_{r,s}$: Retailer×SKU entity effect (partial pooling)
            - $\mu_t$: Month effect (seasonality)
            - $\beta_{s}$: SKU price elasticity with brand×pack-size pooling
            - $f(\text{ND})$: B-spline for non-linear ND effect
            - $\varepsilon$: Observation noise
            """)
        
        st.warning("**Disclaimer**: This model estimates historical conditional associations between relative price, numeric distribution, and standard sales velocity. It does not prove causal price or listing effects without additional promotion, stock, execution, or experimental data.")
        
        st.write(f"**Model-ready observations:** {len(df):,}")
        st.write(f"**Retailers:** {df['retailer'].nunique()} | **Brands:** {df['brand'].nunique()} | **SKUs:** {df['sku'].nunique()} | **Pack Sizes:** {df['pack_size'].nunique()}")
        st.write(f"**Time coverage:** {df['month'].min().strftime('%b %Y')} – {df['month'].max().strftime('%b %Y')} ({df['month'].nunique()} months)")
        
        if df["month"].nunique() < 24:
            st.warning("⚠ Fewer than 24 months of data; model estimates may be unstable")
        
        if st.session_state.model_fit_status == "running":
            with st.status("Fitting Bayesian model...", expanded=True) as status:
                st.write(f"Settings: {st.session_state.model_settings}")
                # Wait for completion
                while st.session_state.model_fit_status == "running":
                    time.sleep(0.5)
                    st.write(f"Running... ({time.time() - st.session_state.get('_fit_start_time', time.time()):.0f}s)")
                if st.session_state.model_fit_status == "success":
                    status.update(label="Model fitted successfully", state="complete")
                else:
                    status.update(label=f"Model fitting failed: {st.session_state.model_fit_error}", state="error")
            st.rerun()
        
        elif st.session_state.model_fit_status == "error":
            st.error(f"Model fitting failed: {st.session_state.model_fit_error}")
        
    # Post-fit panels
    if st.session_state.model_result is not None:
        st.subheader("Model Results")
        
        # Settings and runtime
        settings = st.session_state.model_settings
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Draws:** {settings['draws']} | **Tune:** {settings['tune']} | **Chains:** {settings['chains']}")
        with col2:
            st.write(f"**Target Accept:** {settings['target_accept']} | **Spline Knots:** {settings['n_knots']}")
        with col3:
            st.write(f"**Runtime:** {st.session_state.model_runtime_seconds:.1f}s")
        
        # Diagnostics
        st.subheader("Convergence Diagnostics")
        diag = st.session_state.model_diagnostics
        
        col1, col2, col3 = st.columns(3)
        with col1:
            div = diag["divergences"]
            if div > 0:
                st.error(f"🔴 Divergences: {div}")
            else:
                st.success(f"✓ Divergences: {div}")
        with col2:
            rhat = diag["max_rhat"]
            if rhat > 1.01:
                st.error(f"🔴 Max R-hat: {rhat:.3f}")
            else:
                st.success(f"✓ Max R-hat: {rhat:.3f}")
        with col3:
            ess = diag["min_ess"]
            if ess < 400:
                st.error(f"🔴 Min ESS (bulk): {ess:.0f}")
            else:
                st.success(f"✓ Min ESS (bulk): {ess:.0f}")
        
        # PPC
        st.subheader("Posterior Predictive Check")
        ppc_df = st.session_state.ppc_df
        if not ppc_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = create_observed_vs_predicted(ppc_df)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_residual_histogram(ppc_df)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("PPC data not available")
        
        # Elasticities
        st.subheader("SKU Price Elasticities")
        elasticities = st.session_state.elasticities_df
        if elasticities is not None and len(elasticities) > 0:
            display_cols = ["sku", "brand", "pack_group", "elasticity_p05", "elasticity_median", "elasticity_p95"]
            st.dataframe(
                elasticities[display_cols].style.format({
                    "elasticity_p05": "{:.3f}",
                    "elasticity_median": "{:.3f}",
                    "elasticity_p95": "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            
            # Forest plot
            fig = create_forest_plot(elasticities, "Price Elasticity Forest Plot (90% CI)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
            **Interpretation**: Negative elasticity = higher relative price historically associated with lower standard velocity. 
            Positive elasticity may signal estimation uncertainty, special product dynamics, confounding, or insufficient variation; 
            do not automatically interpret as a positive pricing opportunity.
            """)
            
            # Downloads
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.download_button("Posterior Summary", 
                                   elasticities[display_cols].to_csv(index=False),
                                   "posterior_summary.csv", "text/csv")
            with col2:
                st.download_button("Elasticity Table",
                                   elasticities.to_csv(index=False),
                                   "elasticities.csv", "text/csv")
            with col3:
                st.download_button("Model-Ready Data",
                                   df.to_csv(index=False),
                                   "model_ready_data.csv", "text/csv")
            with col4:
                bio = io.BytesIO()
                st.session_state.model_result.to_netcdf(bio)
                st.download_button("Trace (NetCDF)", bio.getvalue(),
                                   "model_trace.nc", "application/octet-stream")


def render_scenarios_tab() -> None:
    """Tab 7: Scenarios."""
    if st.session_state.model_result is None:
        st.info("Fit a Bayesian model first to enable scenario analysis")
        return
    
    if not st.session_state.scenario_agg:
        st.info("Configure scenario settings in the sidebar and click 'Run Scenario'")
        return
    
    st.warning("**Model-based historical scenario estimate, not a causal guarantee.**")
    
    settings = st.session_state.scenario_settings
    st.write(f"Price change: {settings['price_change']:.0%} | ND change: {settings['nd_change']:.0%}")
    
    agg = st.session_state.scenario_agg
    
    # Total market summary
    st.subheader("Total Market Scenario Summary")
    total = agg["total_market"]
    if not total.empty:
        row = total.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Baseline Units", format_compact_number(row["baseline_units"]))
            st.metric("Expected Units (Median)", format_compact_number(row["expected_units_median"]))
        with col2:
            st.metric("Units p10", format_compact_number(row["expected_units_p10"]))
            st.metric("Units p90", format_compact_number(row["expected_units_p90"]))
        with col3:
            st.metric("Baseline Revenue", format_compact_number(row["baseline_revenue"]))
            st.metric("Expected Revenue (Median)", format_compact_number(row["expected_revenue_median"]))
        with col4:
            st.metric("Volume Change", format_pct(row["volume_change_pct"]))
            st.metric("Revenue Change", format_pct(row["revenue_change_pct"]))
    
    st.divider()
    
    # Aggregation tables
    for level_name, level_df in [("Category", "category"), ("Retailer", "retailer"), 
                                  ("Brand", "brand"), ("SKU + Pack Size", "sku_pack")]:
        if level_name in agg and not agg[level_name].empty:
            st.subheader(f"{level_name} Scenario Summary")
            df_display = agg[level_name].copy()
            
            format_dict = {
                "baseline_units": format_compact_number,
                "expected_units_p10": format_compact_number,
                "expected_units_median": format_compact_number,
                "expected_units_p90": format_compact_number,
                "baseline_revenue": format_compact_number,
                "expected_revenue_median": format_compact_number,
                "volume_change_pct": "{:.1f}%",
                "revenue_change_pct": "{:.1f}%",
            }
            
            st.dataframe(df_display.style.format(format_dict), width="stretch", hide_index=True)
            st.download_button(f"Download {level_name}", df_display.to_csv(index=False),
                               f"scenario_{level_name.lower().replace(' ', '_')}.csv", "text/csv")
    
    # Charts
    st.divider()
    st.subheader("Scenario Charts")
    
    if not agg["retailer"].empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_bar_chart(agg["retailer"], "retailer", "volume_change_pct", 
                                   "Retailer Volume Change (%)", color_col="retailer",
                                   color_map=get_colour_map(st.session_state.filtered_data["retailer"].unique(), RETAILER_COLOURS))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_bar_chart(agg["retailer"], "retailer", "revenue_change_pct", 
                                   "Retailer Revenue Change (%)", color_col="retailer",
                                   color_map=get_colour_map(st.session_state.filtered_data["retailer"].unique(), RETAILER_COLOURS))
            st.plotly_chart(fig, use_container_width=True)
    
    if not agg["brand"].empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_bar_chart(agg["brand"], "brand", "volume_change_pct", 
                                   "Brand Volume Change (%)", color_col="brand",
                                   color_map=get_colour_map(st.session_state.filtered_data["brand"].unique(), BRAND_COLOURS))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_bar_chart(agg["brand"], "brand", "revenue_change_pct", 
                                   "Brand Revenue Change (%)", color_col="brand",
                                   color_map=get_colour_map(st.session_state.filtered_data["brand"].unique(), BRAND_COLOURS))
            st.plotly_chart(fig, use_container_width=True)
    
    if not agg["sku_pack"].empty:
        # Top 20 by baseline revenue
        top_sku = agg["sku_pack"].nlargest(20, "baseline_revenue")
        col1, col2 = st.columns(2)
        with col1:
            fig = create_bar_chart(top_sku, "sku", "volume_change_pct", 
                                   "SKU Volume Change (%) (Top 20 by Revenue)",
                                   color_col="brand", color_map=get_colour_map(st.session_state.filtered_data["brand"].unique(), BRAND_COLOURS),
                                   orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_scatter_chart(top_sku, "volume_change_pct", "revenue_change_pct",
                                       "Revenue vs Volume Impact", color_col="brand",
                                       color_map=get_colour_map(st.session_state.filtered_data["brand"].unique(), BRAND_COLOURS),
                                       size_col="baseline_revenue")
            st.plotly_chart(fig, use_container_width=True)


def render_data_quality_tab() -> None:
    """Tab 8: Data Quality."""
    if st.session_state.quality_report is None:
        st.info("Upload data to begin")
        return
    
    qr = st.session_state.quality_report
    vr = st.session_state.validation_report
    
    st.subheader("Row Counts")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Input Rows", f"{qr['input_row_count']:,}")
    with col2:
        st.metric("Prepared Rows", f"{qr['valid_model_row_count']:,}")
    with col3:
        st.metric("Dropped Rows", f"{qr['dropped_row_count']:,}")
    with col4:
        st.metric("Excluded %", f"{100*qr['dropped_row_count']/qr['input_row_count']:.1f}%" if qr['input_row_count'] > 0 else "—")
    
    # Exclusion reasons
    if qr["dropped_reasons"]:
        st.subheader("Exclusion Reasons")
        excl_df = pd.DataFrame(list(qr["dropped_reasons"].items()), columns=["Reason", "Count"])
        st.dataframe(excl_df, width="stretch", hide_index=True)
    
    # Missing values
    if qr["missing_values"]:
        st.subheader("Missing Values by Column")
        miss_df = pd.DataFrame(list(qr["missing_values"].items()), columns=["Column", "Missing Count"])
        miss_df = miss_df[miss_df["Missing Count"] > 0]
        if not miss_df.empty:
            st.dataframe(miss_df, width="stretch", hide_index=True)
    
    # Duplicates
    st.metric("Duplicate Records", f"{qr['duplicate_records']:,}")
    
    # Date coverage
    dc = qr["date_coverage"]
    st.subheader("Date Coverage")
    st.write(f"Range: {dc['min']} to {dc['max']} | Months: {dc['n_months']}")
    
    # Entity counts
    st.subheader("Entity Counts")
    counts = qr["counts"]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Retailers", counts["retailers"])
    with col2:
        st.metric("Categories", counts["categories"])
    with col3:
        st.metric("Brands", counts["brands"])
    with col4:
        st.metric("SKUs", counts["skus"])
    with col5:
        st.metric("Pack Sizes", counts["pack_sizes"])
    
    # Distribution validity
    dv = qr["distribution_validity"]
    st.subheader("Distribution Validity")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ND Min", f"{dv['nd_min']:.2f}" if dv['nd_min'] is not None else "—")
    with col2:
        st.metric("ND Max", f"{dv['nd_max']:.2f}" if dv['nd_max'] is not None else "—")
    with col3:
        st.metric("ND > 100% Count", f"{dv['nd_gt_1_count']:,}")
    
    # Price variation
    pv = qr["price_variation"]
    st.subheader("Price Variation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rel Price Min", f"{pv['rel_price_min']:.2f}" if pv['rel_price_min'] is not None else "—")
    with col2:
        st.metric("Rel Price Max", f"{pv['rel_price_max']:.2f}" if pv['rel_price_max'] is not None else "—")
    with col3:
        st.metric("Rel Price Std", f"{pv['rel_price_std']:.2f}" if pv['rel_price_std'] is not None else "—")
    
    # Observations per retailer×SKU
    if qr["observations_per_retailer_sku"]:
        st.subheader("Observations per Retailer × SKU")
        st.json(qr["observations_per_retailer_sku"])
    
    if qr["observations_per_sku"]:
        st.subheader("Observations per SKU")
        st.json(qr["observations_per_sku"])
    
    # Low observation SKUs
    if qr["low_observation_skus"]:
        st.subheader("SKUs with <6 Observations")
        st.write(qr["low_observation_skus"])
    
    # Warnings
    if qr["warnings"]:
        st.subheader("Warnings")
        for w in qr["warnings"]:
            st.warning(w)
    
    # Prepared data preview
    st.divider()
    st.subheader("Prepared Data Preview")
    n_rows = st.number_input("Rows to preview", 10, 1000, 100, key="preview_rows")
    st.dataframe(
        st.session_state.prepared_data.head(n_rows).style.format({
            "revenue": format_compact_number,
            "units": format_compact_number,
            "standard_volume": format_compact_number,
            "price_std": "{:.2f}",
            "competitor_price_std": "{:.2f}",
            "relative_price": "{:.2f}",
            "nd": "{:.1%}",
            "velocity_std": "{:.1f}",
        }),
        use_container_width=True,
    )
    
    st.download_button(
        "Download Prepared Data (CSV)",
        st.session_state.prepared_data.to_csv(index=False),
        "prepared_data.csv",
        "text/csv",
    )


# ============================================================
# Main App
# ============================================================

def main() -> None:
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Check if model fitting is running
    if st.session_state.model_fit_status == "running":
        # Store start time for status display
        if "_fit_start_time" not in st.session_state:
            st.session_state._fit_start_time = time.time()
    
    # Header
    st.title("Retail Demand and Distribution Analytics")
    
    # Check if data is loaded
    if st.session_state.raw_data is None:
        st.info("👈 Upload a CSV or XLSX file, or click 'Load Demo Data' in the sidebar to begin.")
        st.markdown("""
        **Required columns:**
        - `month`: Monthly date (YYYY-MM or YYYY-MM-DD)
        - `retailer`: Retailer name or identifier
        - `category`: Category name or identifier
        - `brand`: Brand name or identifier
        - `sku`: SKU name or identifier
        - `units`: Monthly unit sales
        - `revenue`: Monthly revenue
        - `sku_stores`: Stores where SKU was listed/available
        - `retailer_stores`: Total retailer store count
        - `pack_size`: Physical product size (category-consistent unit)
        
        **Note:** `sku_stores` should represent stores where the SKU was *listed or available*, 
        not merely stores with positive sales.
        """)
        return
    
    # Prepare data if not done
    if st.session_state.prepared_data is None:
        with st.spinner("Preparing data..."):
            prepare_and_store()
        st.rerun()
    
    # Apply filters if needed
    if st.session_state.filtered_data is None:
        apply_filters()
    
    # Check if model needs refit warning
    if st.session_state.model_result is not None:
        current_fp = compute_filter_fingerprint(
            st.session_state.data_fingerprint, st.session_state.active_filters
        )
        if st.session_state.model_input_fingerprint != current_fp:
            st.warning("⚠ Filters have changed since model was fitted. Model results may not reflect current filter selection. Refit the model.")
    
    # Render tabs
    tab_objects = st.tabs(TABS)
    
    with tab_objects[0]:
        render_overview_tab()
    with tab_objects[1]:
        render_market_categories_tab()
    with tab_objects[2]:
        render_retailers_tab()
    with tab_objects[3]:
        render_brands_skus_tab()
    with tab_objects[4]:
        render_distribution_velocity_tab()
    with tab_objects[5]:
        render_bayesian_model_tab()
    with tab_objects[6]:
        render_scenarios_tab()
    with tab_objects[7]:
        render_data_quality_tab()


if __name__ == "__main__":
    main()