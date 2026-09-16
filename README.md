# Retail Demand and Distribution Analytics

A Streamlit application for retail demand, pricing, and distribution analytics with hierarchical Bayesian modeling.

## Overview

This application provides descriptive analytics, Bayesian modeling, and scenario analysis for retail data. It uses a hierarchical Bayesian model to estimate price elasticities and numeric distribution effects at the SKU level, with partial pooling across brands and pack sizes.

## Features

- **Data Management**: CSV/XLSX upload with schema validation, template download, and synthetic demo data
- **Descriptive Analytics**: 5 tabs covering market overview, categories, retailers, brands/SKUs/pack sizes, and distribution-velocity analysis
- **Bayesian Modeling**: Hierarchical model with entity effects, month effects, SKU-level price slopes with brand×pack pooling, and B-spline ND effects
- **Model Diagnostics**: Divergences, R-hat, ESS, posterior predictive checks (observed vs predicted, residuals)
- **Elasticity Analysis**: SKU-level price elasticities with 90% credible intervals, forest plots
- **Scenario Analysis**: Price and numeric distribution what-if scenarios with guardrails, multi-level aggregation (total, category, retailer, brand, SKU×pack)
- **Data Quality**: Comprehensive quality report with exclusion reasons, missing values, observation counts

## Installation

```bash
git clone <repository-url>
cd retail_demand_simulator_with_pymc
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Data Format

Required columns:

| Column | Description |
|--------|-------------|
| month | Monthly date (YYYY-MM or YYYY-MM-DD) |
| retailer | Retailer name or identifier |
| category | Category name or identifier |
| brand | Brand name or identifier |
| sku | SKU name or identifier |
| units | Monthly unit sales |
| revenue | Monthly revenue |
| sku_stores | Stores where SKU was listed/available |
| retailer_stores | Total retailer store count |
| pack_size | Physical product size (category-consistent unit) |

Note: `sku_stores` should represent stores where the SKU was listed or available, not merely stores with positive sales.

## Architecture

- `engine.py`: Core analytics engine (validation, preparation, modeling, scenarios, summaries)
- `app.py`: Streamlit frontend with 8 tabs, sidebar controls, fingerprint-based state management
- `demo_data.py`: Synthetic data generator and input template

## Model Specification

The Bayesian model estimates:

```
log(V_std_{r,s,t}) = alpha + eta_{r,s} + mu_t + beta_s * log(P_rel_{r,s,t}) + f(ND_{r,s,t}) + epsilon_{r,s,t}
```

Where:
- `alpha`: Global intercept
- `eta_{r,s}`: Retailer×SKU entity effect (partial pooling)
- `mu_t`: Month effect (seasonality)
- `beta_s`: SKU price elasticity with brand×pack-size pooling
- `f(ND)`: B-spline for non-linear numeric distribution effect
- `epsilon`: Observation noise

## Disclaimer

This model estimates historical conditional associations between relative price, numeric distribution, and standard sales velocity. It does not prove causal price or listing effects without additional promotion, stock, execution, or experimental data.

## License

MIT License