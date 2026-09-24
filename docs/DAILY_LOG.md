# Daily Log

## Day 1 - Business & Measurement Design

**Status:** Completed
**Focus:** Define the business problem, measurement framework, causal design, and synthetic-world structure before implementation.

---

## Objective

Establish the business and analytical design for the Marketing Incrementality & Budget Allocation project before generating data or writing analysis code.

The goal was to make the core decisions explicit enough that the synthetic data, experiment, and later optimization represent one consistent business problem.

---

## 1. Business Question

Defined the project around a realistic marketing investment decision:

> Given an additional $2M marketing budget for the next quarter, how should the budget be allocated across Google, Meta, TikTok, and CRM to maximize expected incremental contribution margin?

The existing annual marketing environment is approximately **$10M**.

The additional $2M is the decision variable for the final budget allocation.

### Decision

The eligible channels are:

* Google
* Meta
* TikTok
* CRM

Organic is included for measurement and baseline context but cannot receive the additional budget.

The existing paid-channel starting mix is:

* Google: 40%
* Meta: 30%
* TikTok: 20%
* CRM: 10%

The 4:3:2:1 mix is the **starting condition**, not the optimization result.

---

## 2. Business Outcome

Defined **incremental contribution margin** as the primary business outcome.

The underlying concept is:

```text
Incremental Outcome
=
Observed Outcome - Counterfactual Outcome
```

Revenue and purchases will be supporting outcomes, but revenue alone is not sufficient for the final investment decision.

### Decision

Standard campaign performance will use **Purchase** as the campaign conversion outcome.

First purchase and repeat purchase behavior may be used later for customer economics and customer value analysis.

### Reasoning

We decided not to routinely split campaign performance into first purchase versus repeat purchase. This keeps the standard campaign analysis focused on the overall purchase outcome while allowing longer-term behavior to be handled separately in customer economics.

---

## 3. Measurement Framework

Defined four connected measurement layers:

1. Funnel
2. Attribution
3. Incrementality
4. Economics

The planned funnel is:

```text
Impression
    ↓
Click
    ↓
Visit
    ↓
Booking
    ↓
Purchase
    ↓
Revenue
    ↓
Contribution Margin
```

### Key distinction

A central principle was established:

> Attribution is not the same as incrementality.

Attribution describes which channel receives credit under a measurement methodology.

Incrementality asks what additional business would not have happened without the marketing activity.

Attribution will therefore be treated as descriptive measurement, not direct causal evidence.

---

## 4. Causal Framework

Defined a **geographic controlled experiment** as the primary source of aggregate causal evidence.

### Experiment design

* 20 markets
* 10 treatment markets
* 10 control markets
* 28-day experiment
* Treatment receives a 25% increase in total marketing spend
* Control remains at baseline spend

### Market definition

A **market** is a geographic business unit with independently observable marketing activity and customer outcomes.

Markets do not need to be identical in size, purchasing power, or customer behavior.

### Assignment approach

Treatment and control markets will be assigned using **stratified randomization based on pre-experiment market characteristics**.

Potential characteristics discussed for stratification and balance checks include:

* Baseline purchases
* Baseline revenue
* Baseline marketing spend
* Market size
* Historical purchase behavior
* Average order value
* Customer mix

### Reasoning

We considered the fact that geographic markets can differ materially in purchasing power and purchase behavior.

Rather than manually selecting markets that appear similar, the design uses pre-period characteristics to structure randomization, followed by baseline-balance and pre-trend diagnostics.

With only 20 markets, perfect balance is not expected. The objective is credible comparability, not identical markets.

---

## 5. Difference-in-Differences

The primary causal estimator will be **Difference-in-Differences (DiD)**.

Conceptually:

```text
Incremental Effect
=
Change in Treatment
-
Change in Control
```

The primary causal outcome will be contribution margin.

Supporting outcomes include:

* Purchases
* Revenue

The key identifying assumption is that treatment and control markets would have followed sufficiently similar trends in the absence of the treatment.

Planned diagnostics include:

* Baseline balance
* Pre-treatment trends
* Treatment exposure
* Contamination
* Seasonality
* Market shocks
* Statistical uncertainty

No experiment estimation was performed today.

---

## 6. Aggregate Experiment vs Channel Effects

A key limitation of the experiment was clarified.

Because treatment increases total marketing investment across the paid channels, the experiment identifies the **aggregate effect of increased marketing investment**.

It does not automatically identify the causal effect of Google, Meta, TikTok, or CRM individually.

### Decision

Treatment spend is intended to preserve the existing 4:3:2:1 channel mix while increasing total marketing spend by 25%.

For example:

```text
Baseline:   $100K
Treatment:  $125K

Google:     $40K  → $50K
Meta:       $30K  → $37.5K
TikTok:     $20K  → $25K
CRM:        $10K  → $12.5K
```

### Reasoning

The experiment provides aggregate causal evidence.

Channel-level marginal economics will instead need to come from channel response models and the broader synthetic data, with the aggregate experiment used as an important piece of evidence.

The project will not attribute the aggregate experiment effect to an individual channel without sufficient identification.

---

## 7. Synthetic World Design

Defined the synthetic data as one coherent business and data-generating process rather than independent random CSV files.

### Historical period

The planned design contains approximately **18 months of historical data**, followed by the experiment period and an observation window for downstream customer behavior.

Exact dates have not yet been defined.

### Market heterogeneity

Markets may differ in:

* Market size
* Baseline demand
* Purchase propensity
* Average order value
* Customer mix
* Marketing intensity
* Seasonality

### Customer heterogeneity

Customers may differ in:

* Purchase propensity
* Average order value
* Likelihood of repeat purchase
* Product/category preference
* Price sensitivity

### Channel behavior

Google, Meta, TikTok, and CRM will have different underlying response characteristics.

Marketing effects will be probabilistic rather than deterministic.

Diminishing returns will be included so that marginal-return analysis is meaningful.

### Design principle

The final relative performance of the channels will not be predetermined.

The synthetic data should contain enough variation to require actual analysis rather than making the eventual answer obvious.

---

## 8. Hidden Simulation Truth

Defined four hidden ground-truth artifacts for later validation:

```text
data/simulation_truth/

true_channel_incrementality.csv
true_customer_value.csv
true_market_effect.csv
true_response_curves.csv
```

These files represent underlying relationships in the synthetic data-generating process.

### Decision

Ground truth will be used for validation rather than treated as a primary analytical input.

The purpose is to evaluate whether the analytical methods can recover known relationships from noisy simulated data.

The analytical estimates are not expected to match the ground truth exactly.

---

## 9. Data Quality Design

Discussed the need for realistic but controlled data-quality problems.

Potential issues include:

* Missing campaign IDs
* Missing UTMs
* Duplicate records
* Duplicate transactions
* Customer identity inconsistencies
* Attribution-window differences
* Timezone differences
* Platform versus warehouse discrepancies
* Unmatched records
* Experiment contamination

### Boundary

No data-quality issues were implemented today.

The exact issues and their rates remain open until the synthetic data-generation design is implemented.

The intention is to create realistic analytical problems without making the dataset artificially messy or unnecessarily complex.

---

## 10. What Was Actually Completed

Day 1 was completed as a **business and measurement design milestone**.

The following were defined and agreed:

* Business question
* $10M annual starting environment
* Additional $2M budget
* Eligible channels
* 4:3:2:1 starting mix
* Organic as measurement-only
* Incremental contribution margin as the primary outcome
* Purchase as the standard campaign conversion outcome
* Funnel / attribution / incrementality / economics framework
* Definition of a market
* Geographic experiment structure
* 20-market treatment/control design
* 25% treatment spend increase
* Stratified randomization approach
* Baseline-balance and pre-trend validation approach
* Difference-in-Differences as the primary estimator
* Aggregate versus channel-level causal distinction
* Synthetic-world design
* Hidden simulation-truth structure

No implementation was completed.

---

## 11. Not Completed

The following were discussed as future work but were **not completed today**:

* Data generation
* Data-quality injection
* Data cleaning
* Data validation
* Funnel analysis
* Attribution analysis
* Experiment estimation
* Customer economics
* LTV analysis
* Channel response curves
* Marginal-return analysis
* Budget optimization
* Scenario analysis
* Final $2M allocation
* Executive recommendation

---

## 12. Open Questions

The following implementation details remain unresolved:

* Exact historical dates
* Exact post-experiment observation window
* Final market characteristics and stratification variables
* Exact market identifiers
* Exact market-level implementation of the 25% treatment increase
* Exact channel response-curve structure
* Detailed customer behavior rules
* Product categories and variable-cost structure
* Exact data-quality issues and their rates
* How aggregate experiment evidence will be connected to channel-level marginal economics
* Final operational constraints for the $2M allocation

These are open design questions, not missing results.

---

## Day 1 Conclusion

Day 1 established the business question, measurement framework, causal experiment design, and synthetic-world structure needed to build the project consistently.

The important boundary is that **Day 1 produced design decisions, not analytical results**. No synthetic data or analysis was executed yet.