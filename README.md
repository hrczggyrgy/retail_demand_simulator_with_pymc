# Retail Demand and Distribution Analytics

A Streamlit application for retail demand, pricing, and distribution analytics with hierarchical Bayesian modeling.

## Overview

This application models an **observed total beverage market**. A selected SKU or brand is a **scenario focal point only** — it is not an "own" brand. The model estimates historical conditional associations between relative price, numeric distribution, and standard sales velocity across the full competitive set.

## Features

- **Data Management**: CSV/XLSX upload with schema validation, template download, and synthetic demo data (exactly 10 raw columns)
- **Descriptive Analytics**: 6 tabs — Overview, Data Health, Market & Share, Price & Distribution, Scenario Simulator, Model Health
- **Bayesian Modeling**: Hierarchical model with entity effects, month effects, SKU-level price slopes with brand×pack pooling, and B-spline ND effects
- **Explicit Model Fitting**: Model fits only when you click "Fit / Re-fit model" — no auto-fit on data load
- **Model Diagnostics**: Divergences, R-hat, ESS, posterior predictive coverage, observed vs predicted, residual heatmap
- **Elasticity Analysis**: SKU-level price elasticities with 90% credible intervals, Plotly forest plot
- **Scenario Analysis**: Price and numeric distribution what-if scenarios with guardrails, multi-level aggregation (market, retailer, category, brand, brand×pack, SKU)
- **Market Impact Comparison**: Baseline, price-only, distribution-only, and combined scenarios with reallocation breakdown and parameter attribution
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

### Workflow

1. **Upload Data** — Use "Load uploaded data" or "Use demo market" in the sidebar
2. **Configure Model** — Select model mode (Fast/Default/Advanced) in sidebar
3. **Fit Model** — Click "Fit / Re-fit model" button (model does not auto-fit on load)
4. **Explore Tabs** — View filters in sidebar affect display tabs only; scenarios always use the full market
5. **Run Scenarios** — In Scenario Simulator tab, set price/ND changes and target, then click "Run scenario"
6. **Compare Market Impact** — View scenario suite results, reallocation breakdown, and parameter attribution

### Data Contract

The application accepts **exactly these 10 raw source columns**:

| Column | Description |
|--------|-------------|
| `month` | Monthly date: `YYYY-MM` or `YYYY-MM-DD` |
| `retailer` | Retailer identifier |
| `category` | Category identifier |
| `brand` | Brand identifier |
| `sku` | SKU identifier |
| `units` | Monthly unit sales |
| `revenue` | Monthly revenue |
| `sku_stores` | Stores where SKU was listed/available |
| `retailer_stores` | Total retailer store count |
| `pack_size` | Category-consistent physical pack size |

**Derived quantities**: `revenue / units` is average realised unit price, not necessarily shelf/list price.

### Scenario Output Contract

Canonical scenario columns (no legacy aliases):

| Column | Description |
|--------|-------------|
| `units_p05`, `units_p50`, `units_p95` | Posterior unit volume quantiles |
| `revenue_p05`, `revenue_p50`, `revenue_p95` | Posterior revenue quantiles |
| `scenario_unit_price` | Scenario unit price (`unit_price * (1 + price_change)`) |
| `scenario_nd` | Scenario numeric distribution |
| `scenario_price_change` | Applied price change |
| `scenario_nd_change` | Applied ND change |

### Interpretation

Scenario results are **modelled historical relationships**, not causal proofs of price, distribution, promotion, profit, or household switching effects. The selected SKU is a focal intervention point in the observed market, not a product owned by the user.

Reallocation is labelled "modelled reallocation in the observed market" — segments are: Selected SKU, Same-brand other SKUs, Other brands (same pack / other pack), Other categories (same retailer), Same category other retailers.

## Quality Commands

```bash
# Run tests
pytest -q

# Lint
ruff check .

# Type check
mypy app.py engine.py demo_data.py
```

## Architecture

- `engine.py`: Core analytics engine (validation, preparation, modeling, scenarios, summaries) — organized by analytical layer
- `app.py`: Streamlit frontend with 6 tabs, sidebar controls, explicit fit button — organized by page-rendering functions
- `demo_data.py`: Raw 10-column data generator and input template only
- `tests/test_app.py`: Comprehensive test suite (data contracts, engine functions, scenario reconciliation)

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

## Standard Scenario Suite

Four scenarios run on the **same posterior draw set** for fair comparison:

| Scenario | Price Change | ND Change |
|----------|--------------|-----------|
| Baseline | 0% | 0 pp |
| Price only | User-specified | 0 pp |
| Distribution only | 0% | User-specified |
| Combined | User-specified | User-specified |

Aggregation at six levels: Market, Retailer, Category, Brand, Brand×Pack, SKU. Total observed-market delta equals sum of SKU deltas within floating-point tolerance.

## Disclaimer

This model estimates historical conditional associations between relative price, numeric distribution, and standard sales velocity. It does not prove causal price or listing effects without additional promotion, stock, execution, or experimental data.

## License

MIT License