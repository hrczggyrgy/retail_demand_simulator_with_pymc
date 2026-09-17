# Hierarchical Retail Market Demand System

## Scientific Blueprint for SKU-Level Market Reallocation

## 1. Purpose

Build a Bayesian retail demand system that answers the following counterfactual question:

> If a user changes realised price and/or numeric distribution for one or more SKUs at selected retailers, how does the **complete observed market** change across every SKU, brand, pack size, category, retailer, and an outside-market component?

The model must report not only the selected SKU response but also the **model-implied source and destination of volume**:

- Which same-brand SKUs lose or gain volume
- Which other-brand SKUs lose or gain volume
- Which pack-size groups gain or lose
- Which categories gain or lose
- Which retailers gain or lose
- How much is internal reallocation versus net observed-market expansion/contraction
- Posterior uncertainty for every result

This is a total-market model. It does not represent a particular manufacturer's portfolio.

## 2. Interpretation and limits

Use this statement in the application and documentation:

> **Modelled historical response — not causal guarantees.**

The model estimates historical conditional associations in the uploaded retail panel. It does not prove household-level switching or causal price impact.

Appropriate wording:

- Model-implied reallocation in the observed market
- Posterior-estimated historical price response
- Historical numeric-distribution response
- Scenario-associated volume movement

Inappropriate wording:

- Units stolen from a competitor
- Guaranteed incremental volume
- Causal price uplift
- Observed shopper switching

The model does not currently observe promotions, trade spend, stock-outs, shelf/display execution, cost/margin, household baskets, shopper location, or retailer trips. These omissions must be visible in methodology documentation and scenario exports.

## 3. Raw data contract

The uploaded dataset and demo generator must contain exactly these 10 source columns:

| Column | Meaning |
|---|---|
| `month` | Monthly date in `YYYY-MM` or `YYYY-MM-DD` format |
| `retailer` | Retailer name or identifier |
| `category` | Category name or identifier, e.g. CSD, Ice Tea, Juice |
| `brand` | Brand name or identifier |
| `sku` | SKU name or identifier |
| `units` | Monthly package units sold |
| `revenue` | Monthly realised revenue |
| `sku_stores` | Stores where the SKU was listed/available |
| `retailer_stores` | Total retailer store count |
| `pack_size` | Category-consistent physical pack size |

The raw data must not contain derived analytics, ownership flags, elasticity values, scenario values, or model outputs.

## 4. Market definition

The observed competitive market is:

\[
\text{Observed beverage market}
\rightarrow
\text{Retailer}
\rightarrow
\text{Category}
\rightarrow
\text{Brand}
\rightarrow
\text{SKU}
\rightarrow
\text{Pack size}
\]

The initial categories are:

- Carbonated soft drinks (CSD)
- Ice Tea
- Juice

There are no “own” or “rival” brands. The user selects a **focal SKU**, **focal brand**, or **target scope** for a scenario. All other records are simply alternatives in the observed market.

Use these neutral terms:

| Term | Definition |
|---|---|
| Selected SKU | SKU receiving a direct price/distribution change |
| Same-brand other SKU | Another SKU with the same `brand` identifier |
| Other brand | Any different `brand` identifier in the relevant observed market |
| Same pack group | Similar category-relative pack-size band |
| Other category | Different beverage category in the observed market |
| Other retailer | Different retailer in the observed market |
| Outside option | Demand not assigned to an observed retailer/category/SKU alternative |

## 5. Derived features

All derived fields are created in `engine.py`.

### 5.1 Standard volume and price

For retailer \(r\), SKU \(s\), and month \(t\):

\[
Q^{std}_{r,s,t}
=
Units_{r,s,t} \times PackSize_s
\]

\[
P^{std}_{r,s,t}
=
\frac{Revenue_{r,s,t}}{Q^{std}_{r,s,t}}
\]

Where:

- \(Q^{std}\) is physical standard volume, for example litres/month if pack size is litres
- \(P^{std}\) is realised price per physical standard unit

Keep package units and standard volume separate:

| Measure | Use |
|---|---|
| Package units | Exact SKU-level choice/allocation accounting |
| Standard volume | Cross-pack market size, volume, and value comparison |
| Revenue | Value impact |

### 5.2 Distribution and velocity

\[
ND_{r,s,t}
=
\frac{SKUStores_{r,s,t}}{RetailerStores_{r,t}}
\]

\[
Velocity^{std}_{r,s,t}
=
\frac{Q^{std}_{r,s,t}}{SKUStores_{r,s,t}}
\]

Accounting identity:

\[
Q^{std}_{r,s,t}
=
RetailerStores_{r,t}
\times
ND_{r,s,t}
\times
Velocity^{std}_{r,s,t}
\]

Interpretation:

- Retailer stores: potential retail network
- Numeric distribution: proportion of that network carrying the SKU
- Standard-volume velocity: physical volume sold per listed store per month

### 5.3 Shares

Within a retailer-category-month market:

\[
Share^{unit}_{r,s,t}
=
\frac{Units_{r,s,t}}
{\sum_{j \in \mathcal{J}_{r,c,t}} Units_{r,j,t}}
\]

\[
Share^{std}_{r,s,t}
=
\frac{Q^{std}_{r,s,t}}
{\sum_{j \in \mathcal{J}_{r,c,t}} Q^{std}_{r,j,t}}
\]

\[
Share^{revenue}_{r,s,t}
=
\frac{Revenue_{r,s,t}}
{\sum_{j \in \mathcal{J}_{r,c,t}} Revenue_{r,j,t}}
\]

Use package-unit share for the count allocation model. Show standard-volume and revenue shares in market reports.

### 5.4 Price-position features

Create a peer price basket for each retailer × category × month × pack group:

\[
P^{std,peer}_{r,s,t}
=
\frac{
\sum_{j \neq s} P^{std}_{r,j,t}Q^{std}_{r,j,t}
}{
\sum_{j \neq s}Q^{std}_{r,j,t}
}
\]

Then calculate relative price:

\[
RP_{r,s,t}
=
\log
\left(
\frac{P^{std}_{r,s,t}}
{P^{std,peer}_{r,s,t}}
\right)
\]

Also create:

- `same_brand_other_sku_price_index`
- `other_brand_price_index`
- price rank within retailer-category-pack group
- pack-size group: Small, Medium, Large, based on category-relative pack-size bands

## 6. Choice-set construction

The central implementation object is a market choice set.

Define:

\[
market\_id
=
month \times retailer \times category
\]

For every market row, construct an alternative set of all SKUs that can compete in that category:

```text
2026-07 | Retailer A | CSD

Alternatives:
- Brand A 330ml
- Brand A 500ml
- Brand A 1.5L
- Brand B 500ml
- Brand B 2L
- Brand C 330ml
- Brand C 1L
```

The SKU set is ragged because not every SKU is available in every retailer-category-month. Convert it to a padded array with an availability mask.

```text
sku_count_matrix:      (n_markets, n_skus)
relative_price_matrix: (n_markets, n_skus)
nd_matrix:             (n_markets, n_skus)
available_mask:        (n_markets, n_skus)
market_total_units:    (n_markets,)
```

For unavailable SKUs:

```text
observed units = 0
availability mask = False
utility = -1e9 before softmax
```

A SKU must be treated as unavailable if its numeric distribution is zero in that retailer-category-month.

## 7. Demand system architecture

Use a nested demand system with three layers.

### 7.1 Layer A: total observed beverage market

\[
T_t
=
\sum_{r,c,s}Q^{std}_{r,c,s,t}
\]

Baseline option:

\[
T^{scenario}_t = T^{baseline}_t
\]

This holds total observed market volume fixed and allocates volume internally.

Advanced option:

\[
\log(T_t)
=
\mu_T
+
\theta_T CPI_t
+
\omega_T Assortment_t
+
\delta^T_{m(t)}
+
\tau_Tt
+
\epsilon^T_t
\]

Use the advanced option only after lower-level allocation models validate successfully.

### 7.2 Layer B: retailer-category market allocation

Define retailer-category market standard volume:

\[
M_{r,c,t}
=
\sum_{s \in \mathcal{J}_{r,c,t}}
Q^{std}_{r,c,s,t}
\]

Retailer-category share of total observed market:

\[
w_{r,c,t}
=
\frac{M_{r,c,t}}{T_t}
\]

Retailer-category utility:

\[
U^{nest}_{r,c,t}
=
a_{r,c}
+
\theta_c CPI_{r,c,t}
+
\omega_c Assortment_{r,c,t}
+
\lambda_c IV_{r,c,t}
+
\delta_{r,c,m(t)}
\]

Then:

\[
w_{r,c,t}
=
\frac{
\exp(U^{nest}_{r,c,t})
}{
\sum_{r',c'}\exp(U^{nest}_{r',c',t})
+
\exp(U^{outside}_t)
}
\]

This layer is needed to estimate model-implied movement from:

- Retailer B CSD to Retailer A CSD
- Ice Tea/Juice to CSD
- Observed beverage market outside option to an observed retailer-category cell

### 7.3 Layer C: SKU allocation inside retailer-category market

For active SKU \(s\) within retailer \(r\), category \(c\), month \(t\):

\[
U_{r,s,t}
=
\alpha_{r,s}
+
\beta_s RP_{r,s,t}
+
\gamma_{1,s}ND_{r,s,t}
+
\gamma_{2,s}ND_{r,s,t}^{2}
+
\zeta_{pack(s)}
+
\delta_{c,m(t)}
+
\xi_{r,s,t}
\]

Conditional SKU share:

\[
s_{r,s,t \mid c}
=
\frac{
\exp(U_{r,s,t}/\lambda_c)
}{
\sum_{j \in \mathcal{J}_{r,c,t}}
\exp(U_{r,j,t}/\lambda_c)
}
\]

Inclusive value:

\[
IV_{r,c,t}
=
\lambda_c
\log
\left[
\sum_{j \in \mathcal{J}_{r,c,t}}
\exp(U_{r,j,t}/\lambda_c)
\right]
\]

Final standard-volume forecast:

\[
Q^{std}_{r,s,t}
=
T_t
\times
w_{r,c,t}
\times
s_{r,s,t \mid c}
\]

## 8. Likelihood design

### 8.1 SKU allocation likelihood

Use package units as the count outcome inside each retailer-category-month choice set:

\[
\mathbf{y}_{r,c,t}
\sim
DirichletMultinomial
(
N_{r,c,t},
\kappa_c \cdot \mathbf{p}_{r,c,t}
)
\]

Where:

- \(\mathbf{y}_{r,c,t}\) is the vector of SKU package-unit counts
- \(N_{r,c,t}\) is total package units in the retailer-category-month
- \(\mathbf{p}_{r,c,t}\) is the softmax choice probability vector
- \(\kappa_c\) controls overdispersion

Use Dirichlet-Multinomial instead of simple Multinomial because retail monthly shares have extra variability caused by unobserved promotions, execution, availability, and demand shocks.

### 8.2 Standard-volume model

Use standard volume for market-size and velocity diagnostics. It can be modelled separately with a Student-t log-volume/log-velocity model:

\[
\log(Velocity^{std}_{r,s,t})
=
\alpha_{r,s}
+
\beta_sRP_{r,s,t}
+
f_s(ND_{r,s,t})
+
\delta_{c,m(t)}
+
\epsilon_{r,s,t}
\]

\[
\epsilon_{r,s,t}
\sim
StudentT(\nu,0,\sigma)
\]

This auxiliary model is useful for standard-volume forecast diagnostics and the distribution/velocity decomposition.

## 9. Hierarchical priors

### 9.1 Price sensitivity

\[
\beta_s
\sim
\mathcal{N}
(
\mu_{\beta,brand(s),pack(s)},
\sigma_{\beta,brand(s),pack(s)}
)
\]

\[
\mu_{\beta,brand,pack}
\sim
\mathcal{N}
(
\mu_{\beta,category},
\sigma_{\beta,category}
)
\]

\[
\mu_{\beta,category}
\sim
\mathcal{N}(-1.0, 0.75)
\]

### 9.2 Distribution sensitivity

\[
\gamma_{1,s}
\sim
\mathcal{N}
(
\mu_{\gamma1,category(s)},
\sigma_{\gamma1,category(s)}
)
\]

\[
\gamma_{2,s}
\sim
\mathcal{N}
(
\mu_{\gamma2,category(s)},
\sigma_{\gamma2,category(s)}
)
\]

### 9.3 Retailer-SKU baseline

\[
\alpha_{r,s}
\sim
\mathcal{N}
(
\alpha_s,
\sigma_{\alpha,retailer}
)
\]

\[
\alpha_s
\sim
\mathcal{N}
(
\alpha_{brand(s)},
\sigma_{\alpha,sku}
)
\]

### 9.4 Nesting parameter

\[
0 < \lambda_c \leq 1
\]

Use:

\[
\lambda_c = sigmoid(\tilde{\lambda}_c)
\]

\[
\tilde{\lambda}_c
\sim
\mathcal{N}(0,1)
\]

A lower \(\lambda_c\) indicates stronger substitution among SKUs in the same retailer-category nest.

## 10. PyMC implementation design

### 10.1 Coordinates

```python
coords = {
    "market": market_ids,
    "retailer": retailer_levels,
    "category": category_levels,
    "brand": brand_levels,
    "sku": sku_levels,
    "pack_group": pack_group_levels,
    "month": month_levels,
}
```

### 10.2 Required data arrays

```python
observed_units: np.ndarray              # (market, sku)
market_total_units: np.ndarray          # (market,)
relative_price: np.ndarray              # (market, sku)
nd: np.ndarray                          # (market, sku)
available_mask: np.ndarray              # (market, sku)
market_retailer_idx: np.ndarray         # (market,)
market_category_idx: np.ndarray         # (market,)
market_month_idx: np.ndarray            # (market,)
sku_brand_idx: np.ndarray               # (sku,)
sku_category_idx: np.ndarray            # (sku,)
sku_pack_group_idx: np.ndarray          # (sku,)
brand_category_idx: np.ndarray          # (brand,)
```

### 10.3 PyMC graph outline

```python
with pm.Model(coords=coords) as model:
    observed_units = pm.Data(
        "observed_units",
        sku_count_matrix,
        dims=("market", "sku"),
    )

    total_units = pm.Data(
        "total_units",
        market_total_units,
        dims="market",
    )

    relative_price = pm.Data(
        "relative_price",
        relative_price_matrix,
        dims=("market", "sku"),
    )

    nd = pm.Data(
        "nd",
        nd_matrix,
        dims=("market", "sku"),
    )

    available = pm.Data(
        "available",
        availability_mask,
        dims=("market", "sku"),
    )

    beta_category = pm.Normal(
        "beta_category",
        mu=-1.0,
        sigma=0.75,
        dims="category",
    )

    beta_brand = pm.Normal(
        "beta_brand",
        mu=beta_category[brand_category_idx],
        sigma=sigma_beta_brand[brand_category_idx],
        dims="brand",
    )

    beta_sku = pm.Normal(
        "beta_sku",
        mu=beta_brand[sku_brand_idx],
        sigma=sigma_beta_sku[sku_brand_idx],
        dims="sku",
    )

    alpha_retailer_sku = pm.Normal(
        "alpha_retailer_sku",
        mu=0.0,
        sigma=sigma_alpha_retailer_sku,
        dims=("retailer", "sku"),
    )

    utility = (
        alpha_sku[None, :]
        + alpha_retailer_sku[market_retailer_idx, :]
        + beta_sku[None, :] * relative_price
        + gamma_nd_sku[None, :] * nd
        + gamma_nd2_sku[None, :] * pt.square(nd)
        + category_month[
            market_category_idx,
            market_month_idx,
        ][:, None]
    )

    masked_utility = pt.where(
        available,
        utility,
        -1e9,
    )

    sku_share = pm.Deterministic(
        "sku_share",
        pm.math.softmax(masked_utility, axis=1),
        dims=("market", "sku"),
    )

    concentration = concentration_category[
        market_category_idx
    ][:, None]

    pm.DirichletMultinomial(
        "sku_units_obs",
        n=total_units,
        a=concentration * sku_share,
        observed=observed_units,
        dims=("market", "sku"),
    )
```

Use PyTensor tensor operations inside the model graph.

## 11. Scenario methodology

### 11.1 User-editable scenario actions

A scenario action must be defined at:

\[
month \times retailer \times sku
\]

Editable fields:

- `new_price_per_standard_unit`
- `new_nd`
- `apply`

The scenario action table should show:

| Apply? | Month | Retailer | Category | Brand | SKU | Pack size | Baseline price | New price | Baseline ND | New ND |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|

### 11.2 Baseline scenario

\[
P^{std}_{new}=P^{std}_{base}
\]

\[
ND_{new}=ND_{base}
\]

### 11.3 Price scenario

\[
P^{std}_{i,new}
=
P^{std}_{i,base}(1+\Delta P_i)
\]

Recalculate:

- Peer-price indices
- Relative price of selected SKU
- Relative price of other affected SKUs if peer basket changes
- All utilities in the affected market
- All SKU conditional shares
- Retailer-category allocation, if Layer B is active
- Total market, if Layer A is active

### 11.4 Distribution scenario

\[
ND_{i,new}
=
\frac{SKUStores_{i,new}}{RetailerStores_i}
\]

Recalculate selected SKU utility:

\[
U^{new}_{i}
=
U^{base}_{i}
+
\gamma_{1,i}(ND_{new}-ND_{base})
+
\gamma_{2,i}(ND_{new}^2-ND_{base}^2)
\]

If `new_nd = 0`, set SKU availability to false.

### 11.5 Combined scenario

Apply both price and distribution changes, then recalculate all model layers.

### 11.6 Posterior-draw simulation

For every posterior draw \(d\):

1. Create baseline price, ND, availability, utility, shares, and volume.
2. Apply user scenario actions.
3. Recalculate affected peer prices.
4. Recalculate all utilities and softmax shares.
5. Recalculate retailer-category and total-market allocation if enabled.
6. Calculate baseline and scenario volumes for every SKU.
7. Calculate deltas for every SKU.
8. Aggregate deltas by brand, pack group, category, retailer, and total market.
9. Compute percentiles across posterior draws.

Do not aggregate posterior medians and then calculate deltas. Calculate all counterfactual deltas per posterior draw first.

## 12. Exact model-implied source/destination accounting

For selected SKU \(i\), posterior draw \(d\):

\[
\Delta Q_i^{(d)}
=
Q_{i,scenario}^{(d)}
-
Q_{i,baseline}^{(d)}
\]

For every alternative SKU \(j\):

\[
\Delta Q_j^{(d)}
=
Q_{j,scenario}^{(d)}
-
Q_{j,baseline}^{(d)}
\]

If \(\Delta Q_j^{(d)} < 0\), alternative \(j\) is a model-implied source of selected SKU growth.

\[
SourceShare_j^{(d)}
=
\frac{-\Delta Q_j^{(d)}}
{\sum_{k \neq i, \Delta Q_k^{(d)}<0}-\Delta Q_k^{(d)}}
\]

Use a neutral relationship taxonomy:

```text
selected_sku
same_brand_other_sku
other_brand_same_pack
other_brand_other_pack
other_category_same_retailer
same_category_other_retailer
other_category_other_retailer
outside_option
```

Report P05, P50, and P95 for all gains, losses, source shares, and market totals.

## 13. Required reconciliation

For every posterior draw and market:

\[
\sum_{s \in \mathcal{J}_{r,c,t}}
Share_{r,s,t}=1
\]

\[
\sum_{s \in \mathcal{J}_{r,c,t}}
Q_{r,s,t}=M_{r,c,t}
\]

\[
\sum_{r,c}M_{r,c,t}=T_t
\]

When total observed market size is held fixed:

\[
\sum_{j}\Delta Q_j=0
\]

When an outside option is included:

\[
\sum_{j}\Delta Q_j
+
\Delta Q_{outside}=0
\]

All UI tables, visualisations, and exports must pass these reconciliation checks.

## 14. App workflow

### Screen 1: Market data

Purpose:

- Upload a raw market dataset or load sample market data
- Validate the raw ten-column schema
- Show data-quality and coverage checks
- Confirm market scope, period, retailers, categories, brands, SKUs

### Screen 2: Market diagnostics

Purpose:

- Understand the market before fitting a model

Primary outputs:

- Total-market/category standard-volume trend
- Brand and SKU shares
- Pack-size mix
- Price-pack ladder
- Numeric-distribution versus standard-volume-velocity quadrant
- Growth decomposition: network, ND, and velocity

### Screen 3: Fit and validate

Purpose:

- Fit PyMC model explicitly
- Validate statistical reliability before scenario use

Primary outputs:

- Divergences
- R-hat
- Effective sample size
- Posterior predictive checks
- Observed versus predicted output
- Elasticity forest plot
- Advanced trace/rank/energy diagnostics in expanders

### Screen 4: Scenario cockpit

Purpose:

- Apply one or more price/distribution actions
- View full observed-market impact

Order:

1. Editable scenario-actions table
2. Guardrails and model-health warning
3. Selected SKU baseline versus scenario cards
4. Parameter bridge waterfall
5. Exact SKU-level winner/loser table
6. Reallocation Sankey chart
7. Brand/category/retailer/market summary
8. Uncertainty fan or interval charts
9. Exportable scenario audit

## 15. Required visualisations

### 15.1 Parameter bridge waterfall

```text
Baseline selected SKU volume
+ Price-associated response
+ Distribution mechanical network response
+ Distribution velocity response
+ Price × distribution interaction
= Scenario selected SKU volume
```

Name it:

> Modelled scenario decomposition

Do not name it causal decomposition.

### 15.2 SKU winner-loser table

| Retailer | Category | Brand | SKU | Pack group | Baseline | Scenario | Delta | P05 | P95 | Relationship |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|

### 15.3 Reallocation Sankey

```text
Selected SKU gain
    ← Same-brand other SKUs
    ← Other-brand same-pack SKUs
    ← Other-brand other-pack SKUs
    ← Other categories
    ← Other retailers
    ← Outside option / market expansion
```

### 15.4 Uncertainty chart

Use P05/P50/P95 interval bars for a one-month scenario and a fan chart for multi-month scenario horizons.

### 15.5 Market impact table

| Level | Baseline standard volume | Scenario standard volume | Delta | Delta % | P05 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Selected SKU | | | | | | |
| Selected brand | | | | | | |
| Retailer-category | | | | | | |
| Category total | | | | | | |
| Total observed market | | | | | | |

## 16. Validation requirements

### 16.1 Statistical diagnostics

- Divergences: target zero
- R-hat: below 1.01
- Effective sample size: above 400 where practical
- Posterior predictive checks for SKU allocation and market totals
- Residual checks for standard-volume velocity model
- Rolling out-of-time holdout tests
- Calibration: actual results inside 90% intervals approximately 90% of the time

### 16.2 Scenario tests

```text
- Zero price and ND change reproduces baseline.
- SKU shares sum to one for every retailer-category-month.
- Unavailable SKU has zero or near-zero scenario share.
- SKU deltas reconcile to brand, category, retailer, and total-market deltas.
- Selected SKU gain has a complete model-implied source/destination account.
- Scenario always uses complete uploaded market, not display-filtered rows.
- Target action affects only selected retailer × SKU × month rows.
- Invalid ND below zero or above one is rejected.
- Peer price excludes focal SKU where intended.
```

## 17. Phased implementation

### Phase 1: Standard-volume velocity foundation

Implement:

- Standard volume
- Standard price
- Numeric distribution
- Standard-volume velocity
- Peer price and relative price
- Hierarchical retailer-SKU velocity model
- Price and ND scenario for selected SKU
- Posterior predictive checks

Output:

- Selected-SKU response with P05/P50/P95 uncertainty

### Phase 2: Within retailer-category choice model

Implement:

- Market choice-set tensor
- SKU availability mask
- Dirichlet-Multinomial SKU share model
- SKU-level baseline/scenario share allocation
- Exact model-implied competitor SKU winner/loser output within retailer-category

Output:

- Which CSD SKU at Retailer A gains/loses when selected SKU price/ND changes

### Phase 3: Retailer-category allocation

Implement:

- Retailer-category nest utility
- Inclusive value from SKU alternatives
- Retailer-category shares
- Category expansion versus internal allocation

Output:

- Whether a scenario expands CSD at Retailer A or reallocates existing CSD demand

### Phase 4: Cross-retailer and cross-category allocation

Implement:

- Retailer-category share model across CSD, Ice Tea, and Juice
- Outside option
- Cross-retailer scenario allocation
- Cross-category scenario sensitivity

Output:

- Model-implied volume movement from Retailer B CSD, Ice Tea, Juice, or outside observed market

### Phase 5: Advanced extensions

Only after validation:

- Price × ND interaction
- Time-varying elasticities


## 18. Definition of done

The model is ready for decision-oriented demonstration when:

- The raw input contains only the defined ten columns.
- The data pipeline creates consistent standard-volume, price, ND, and velocity features.
- Choice-set arrays correctly represent every retailer-category-month alternative set.
- Shares and volumes reconcile at SKU, brand, retailer-category, and total-market levels.
- Scenario actions change only selected retailer-SKU-month observations.
- All selected-SKU changes are allocated across explicit observed-market alternatives and outside option.
- Every output has P05, P50, and P95 uncertainty where posterior draws are available.
- Posterior predictive checks and MCMC diagnostics are displayed.
- The app uses modelled historical-response language, not causal guarantees.
- All scenario, aggregation, and reconciliation logic has automated test coverage.
