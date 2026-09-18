"""
Model diagnostics, convergence checks, and elasticity analysis.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from contracts import (
    ConvergenceDiagnostics,
)


def get_model_diagnostics(idata: Any, model_mode: str = "Default (4 chains)") -> dict[str, Any]:
    """Get model diagnostics as a dictionary (backward compatible)."""
    diagnostics = summarize_convergence_diagnostics(idata)
    return {
        "divergences": diagnostics.divergences,
        "max_rhat": diagnostics.max_rhat,
        "min_ess_bulk": diagnostics.min_ess_bulk,
        "min_ess_tail": diagnostics.min_ess_tail,
        "coverage_90": diagnostics.coverage_90,
        "is_acceptable": diagnostics.is_acceptable,
    }


def summarize_convergence_diagnostics(
    idata: Any,
    observed_var: str = "log_velocity_std_z_obs",
    predictive_var: str = "log_velocity_std_z_obs",
) -> ConvergenceDiagnostics:
    """
    Extract convergence diagnostics from InferenceData.
    
    Args:
        idata: ArviZ InferenceData object
        observed_var: Name of observed variable in idata.observed_data
        predictive_var: Name of posterior predictive variable in idata.posterior_predictive
    
    Returns a ConvergenceDiagnostics object with typed fields.
    """
    # Divergences
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().item())

    # R-hat and ESS from arviz summary
    max_rhat = None
    min_ess_bulk = None
    min_ess_tail = None

    try:
        summary = az.summary(idata, round_to=None, kind="diagnostics")
        if "r_hat" in summary.columns:
            max_rhat = float(summary["r_hat"].max())
        if "ess_bulk" in summary.columns:
            min_ess_bulk = float(summary["ess_bulk"].min())
        if "ess_tail" in summary.columns:
            min_ess_tail = float(summary["ess_tail"].min())
    except (ValueError, KeyError, AttributeError, RuntimeError):
        pass

    # E-BFMI and max tree depth from sample_stats
    min_bfmi = None
    max_tree_depth = None
    if hasattr(idata, "sample_stats"):
        if "bfmi" in idata.sample_stats:
            try:
                min_bfmi = float(idata.sample_stats["bfmi"].min().item())
            except (ValueError, AttributeError):
                pass
        if "tree_depth" in idata.sample_stats:
            try:
                max_tree_depth = float(idata.sample_stats["tree_depth"].max().item())
            except (ValueError, AttributeError):
                pass

    # Posterior predictive coverage
    coverage_90 = None
    if (hasattr(idata, "posterior_predictive") and predictive_var in idata.posterior_predictive
        and hasattr(idata, "observed_data") and observed_var in idata.observed_data):
        try:
            pred = idata.posterior_predictive[predictive_var]
            obs = idata.observed_data[observed_var]
            pred_flat = pred.values.reshape((-1,) + pred.shape[2:])
            obs_flat = obs.values.flatten()
            
            # Compute 90% prediction intervals
            pred_lower = np.percentile(pred_flat, 5, axis=0)
            pred_upper = np.percentile(pred_flat, 95, axis=0)
            
            in_interval = (obs_flat >= pred_lower) & (obs_flat <= pred_upper)
            coverage_90 = float(in_interval.mean())
        except (KeyError, ValueError, AttributeError, IndexError):
            pass

    # Missing diagnostics fail - require all checks to be present and pass
    is_acceptable = (
        divergences == 0
        and max_rhat is not None
        and max_rhat < 1.01
        and min_ess_bulk is not None
        and min_ess_bulk >= 400
        and min_ess_tail is not None
        and min_ess_tail >= 400
        and (min_bfmi is None or min_bfmi >= 0.3)
        and (max_tree_depth is None or max_tree_depth < 10)
    )

    return ConvergenceDiagnostics(
        divergences=divergences,
        max_rhat=max_rhat,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        coverage_90=coverage_90,
        min_bfmi=min_bfmi,
        max_tree_depth=max_tree_depth,
        is_acceptable=is_acceptable,
    )


def compute_sku_elasticities(
    posterior_cache: dict[str, np.ndarray],
    sku_ids: list[str],
    meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute price elasticities per SKU from posterior draws."""
    beta_sku = posterior_cache["beta_sku"]  # (n_draws, n_skus)
    
    # Elasticity = beta (since model is log-log)
    elasticities = beta_sku.mean(axis=0)
    elasticities_p05 = np.percentile(beta_sku, 5, axis=0)
    elasticities_p95 = np.percentile(beta_sku, 95, axis=0)
    
    df = pd.DataFrame({
        "sku": sku_ids,
        "elasticity_p50": elasticities,
        "elasticity_p05": elasticities_p05,
        "elasticity_p95": elasticities_p95,
    })
    
    # Add significance flag (90% CI excludes 0)
    df["significant"] = (df["elasticity_p05"] < 0) & (df["elasticity_p95"] < 0)
    
    return df


def build_elasticity_forest_figure(
    elasticity_df: pd.DataFrame,
    title: str = "SKU Price Elasticities (90% CI)",
) -> Any:
    """Build Plotly forest plot for elasticities."""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Sort by median elasticity
    df = elasticity_df.sort_values("elasticity_p50")
    
    fig.add_trace(go.Scatter(
        x=df["elasticity_p50"],
        y=df["sku"],
        mode="markers",
        marker=dict(size=10, color="darkblue"),
        name="Median",
        error_x=dict(
            type="data",
            symmetric=False,
            array=df["elasticity_p95"] - df["elasticity_p50"],
            arrayminus=df["elasticity_p50"] - df["elasticity_p05"],
            color="darkblue",
            thickness=2,
        ),
        hovertemplate="%{y}<br>Elasticity: %{x:.3f}<br>90% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
        customdata=np.column_stack([df["elasticity_p05"], df["elasticity_p95"]]),
    ))
    
    # Add zero reference line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="Price Elasticity",
        yaxis_title="SKU",
        height=max(400, len(df) * 25 + 100),
        margin=dict(l=150, r=50, t=80, b=50),
    )
    
    return fig


def compute_posterior_predictive_check(
    idata: Any,
    observed_var: str = "log_velocity_std_z_obs",
    predictive_var: str = "log_velocity_std_z_obs",
) -> dict[str, Any]:
    """Compute posterior predictive check statistics."""
    if (not hasattr(idata, "posterior_predictive") or predictive_var not in idata.posterior_predictive
        or not hasattr(idata, "observed_data") or observed_var not in idata.observed_data):
        return {"error": "Missing posterior predictive or observed data"}
    
    pred = idata.posterior_predictive[predictive_var]
    obs = idata.observed_data[observed_var]
    
    pred_flat = pred.values.reshape((-1,) + pred.shape[2:])
    obs_flat = obs.values.flatten()
    
    # Mean prediction vs observed
    pred_mean = pred_flat.mean(axis=0)
    
    # Correlation
    from scipy.stats import pearsonr
    corr, p_val = pearsonr(pred_mean, obs_flat)
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_mean - obs_flat) ** 2))
    
    # Coverage
    pred_lower = np.percentile(pred_flat, 5, axis=0)
    pred_upper = np.percentile(pred_flat, 95, axis=0)
    in_interval = (obs_flat >= pred_lower) & (obs_flat <= pred_upper)
    coverage = in_interval.mean()
    
    return {
        "correlation": float(corr),
        "p_value": float(p_val),
        "rmse": float(rmse),
        "coverage_90": float(coverage),
        "n_obs": len(obs_flat),
    }
