# Retail Demand and Distribution Analytics

A Streamlit application for retail demand, pricing, and distribution analytics with hierarchical Bayesian modeling.

## Overview

This application models an **observed total beverage market**. A selected SKU or brand is a **scenario focal point only** — it is not an "own" brand. The model estimates historical conditional associations between relative price, numeric distribution, and standard sales velocity across the full competitive set.

## Features

- **Data Management**: CSV/XLSX upload with schema validation, template download, and synthetic demo data (exactly 10 raw columns)
- **Three-Page Workflow**: Market (data + diagnostics), Scenario (plan editor + outcomes), Model Quality (convergence + PPC + elasticities)
- **Bayesian Modeling**: Joint Dirichlet-Multinomial SKU share model with non-centered hierarchies for brand-pack price effects, ND spline effects, category-month effects
- **Explicit Model Fitting**: Model fits only when you click "Fit / Re-fit model" — no auto-fit on data load
- **Model Diagnostics**: Divergences, R-hat, ESS, BFMI, posterior predictive coverage, observed vs predicted, residual heatmap
- **Elasticity Analysis**: SKU-level price elasticities with 90% credible intervals
- **Scenario Analysis**: Four counterfactuals (baseline, price-only, distribution-only, combined) with per-draw aggregation and market-first outcomes
- **Reallocation & Attribution**: Source-destination flow breakdown and parameter-level driver waterfall
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

1. **Market Page** — Upload data or use demo market; explore market structure, shares, price-pack architecture, ND/velocity, growth decomposition
2. **Scenario Page** — Define `ScenarioPlan` actions (price/ND changes), run 4-counterfactual suite, review market-first outcomes and reallocation
3. **Model Quality Page** — Review convergence (R-hat, ESS, BFMI, divergences), posterior predictive checks, elasticity diagnostics

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
| `baseline_share_p05`, `baseline_share_p50`, `baseline_share_p95` | Posterior baseline share quantiles |
| `scenario_share_p05`, `scenario_share_p50`, `scenario_share_p95` | Posterior scenario share quantiles |
| `reallocated_units_p05`, `reallocated_units_p50`, `reallocated_units_p95` | Posterior reallocated unit quantiles |
| `delta_share_p05`, `delta_share_p50`, `delta_share_p95` | Posterior share delta quantiles |
| `delta_units_p05`, `delta_units_p50`, `delta_units_p95` | Posterior unit delta quantiles |
| `market_total_units` | Observed market total units (fixed) |
| `interval_kind` | `posterior_expected` or `posterior_predictive` |

### Interpretation

Scenario results are **modelled historical relationships**, not causal proofs of price, distribution, promotion, profit, or household switching effects. The selected SKU is a focal intervention point in the observed market, not a product owned by the user.

Reallocation is labelled "modelled reallocation in the observed market" — segments are: Selected SKU, Same-brand other SKUs, Other brands (same pack / other pack), Other categories (same retailer), Same category other retailers.

## Quality Commands

```bash
# Run tests (fast deterministic tests)
pytest -q tests/ -k "not slow" -n 1

# Run slow/MCMC integration tests separately
pytest -q tests/ -k "slow" -n 1

# Lint
ruff check .

# Type check
mypy app.py engine.py demo_data.py engine_modules/
```

## Architecture

```
├── app.py                    # Streamlit frontend (3 pages: Market, Scenario, Model Quality)
├── contracts.py              # Single source of truth: schemas, dataclasses, constants, thresholds
├── engine.py                 # Thin backward-compat re-export layer (legacy API only)
├── demo_data.py              # Raw 10-column data generator and input template only
├── engine_modules/           # Modular engine implementation
│   ├── validation.py         # Data validation & preparation
│   ├── features.py           # Feature engineering (relative price, ND, splines, indices)
│   ├── choice_sets.py        # ChoiceSetData construction for modeling
│   ├── model.py              # Joint Dirichlet-Multinomial model builder
│   ├── fitting.py            # PyMC sampling with non-centered parameterization
│   ├── diagnostics.py        # Convergence, PPC, health assessment
│   ├── scenarios.py          # ScenarioPlan runner, market-first aggregation, reallocation
│   └── reporting.py          # Summary tables, elasticity extraction, driver waterfall
└── tests/                    # Modular test suite
    ├── test_validation.py
    ├── test_features.py
    ├── test_choice_sets.py
    ├── test_model_structure.py
    ├── test_diagnostics.py
    ├── test_scenarios.py
    ├── test_reporting.py
    └── test_app_smoke.py
```

## Model Specification

The Bayesian joint model estimates SKU shares within each retailer×category market via Dirichlet-Multinomial:

```
V_{r,c,s,t} = exp(α + η_{r,c,s} + μ_t + β_s * log(P_rel) + f(ND) + γ_{c,t})
share_{r,c,s,t} ~ DirichletMultinomial(n=total_units, α=concentration * V / sum(V))
```

Where:
- `α`: Global intercept
- `η_{r,c,s}`: Retailer×Category×SKU entity effect (non-centered, partial pooling)
- `μ_t`: Month effect (seasonality)
- `β_s`: SKU price elasticity with brand×pack-size pooling (non-centered)
- `f(ND)`: B-spline for non-linear numeric distribution effect (non-centered)
- `γ_{c,t}`: Category×month interaction (non-centered)
- `concentration`: Dirichlet concentration parameter (HalfNormal + 0.5)

**Sampling defaults (ValidatedModelConfig)**: 4 chains, 1,200 draws, 1,500 tune, `target_accept=0.98`, seed=42.

**Health gates**: R-hat < 1.01, bulk/tail ESS > 400, BFMI > 0.3, 0 divergences, PPC coverage ≈ 90%.

## Standard Scenario Suite

Four scenarios run on the **same posterior draw set** for fair comparison:

| Scenario | Price Change | ND Change |
|----------|--------------|-----------|
| Baseline | 0% | 0 pp |
| Price only | User-specified | 0 pp |
| Distribution only | 0% | User-specified |
| Combined | User-specified | User-specified |

Aggregation is **per-draw before posterior quantiles** (fixing the quantile-of-sum vs sum-of-quantiles error). Market-first outcomes ensure total observed-market delta equals sum of SKU deltas.

## Disclaimer

This model estimates historical conditional associations between relative price, numeric distribution, and standard sales velocity. It does not prove causal price or listing effects without additional promotion, stock, execution, or experimental data.

## License

MIT License