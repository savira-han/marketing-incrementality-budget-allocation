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

* 20 areas
* 10 treatment areas
* 10 control areas
* 28-day experiment
* Treatment receives a 25% increase in total marketing spend
* Control remains at baseline spend

### Area definition

An **area** is a geographic business unit with independently observable marketing activity and customer outcomes.

The project uses **area** rather than market as the geographic unit.

### Assignment approach

Treatment and control areas will be assigned using **stratified randomization based on pre-experiment area characteristics**.

Potential characteristics discussed for stratification and balance checks include:

* Baseline purchases
* Baseline revenue
* Baseline marketing spend
* Area size
* Historical purchase behavior
* Average order value
* Customer mix

### Reasoning

We considered the fact that geographic areas can differ materially in purchasing power and purchase behavior.

Rather than manually selecting areas that appear similar, the design uses pre-period characteristics to structure randomization, followed by baseline-balance and pre-trend diagnostics.

With only 20 areas, perfect balance is not expected. The objective is credible comparability, not identical areas.

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

The key identifying assumption is that treatment and control areas would have followed sufficiently similar trends in the absence of the treatment.

Planned diagnostics include:

* Baseline balance
* Pre-treatment trends
* Treatment exposure
* Contamination
* Seasonality
* Area shocks
* Statistical uncertainty

No experiment estimation was performed on Day 1.

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

### Business context

The company is an **online travel platform (OTA) focused on hotel bookings**.

The customer journey is expected to involve research, comparison, multiple visits, and interactions across several marketing channels before purchase.

### Historical period

The planned design contains **18 months of historical data**, followed by the experiment period and a post-experiment observation window.

Exact dates were not defined on Day 1.

### Area heterogeneity

Areas may differ in:

* Area size
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
true_area_effect.csv
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

No data-quality issues were implemented on Day 1.

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
* Definition of an area
* Geographic experiment structure
* 20-area treatment/control design
* 25% treatment spend increase
* Stratified randomization approach
* Baseline-balance and pre-trend validation approach
* Difference-in-Differences as the primary estimator
* Aggregate versus channel-level causal distinction
* OTA / hotel-booking business context
* Synthetic-world design
* Hidden simulation-truth structure

No implementation was completed.

---

## 11. Not Completed

The following were discussed as future work but were **not completed on Day 1**:

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

## Day 1 Conclusion

Day 1 established the business question, measurement framework, causal experiment design, and synthetic-world structure needed to build the project consistently.

The important boundary is that **Day 1 produced design decisions, not analytical results**. No synthetic data or analysis was executed yet.

---

# Day 2 - Data Architecture

**Status:** Completed
**Focus:** Translate the Day 1 business and measurement design into a concrete source-data architecture before generating synthetic data.

---

## Objective

Define what data is required, what each row represents, how records are identified and joined, and how the source data will support attribution, incrementality, customer economics, channel response, and budget optimization.

The main goal was to establish the architecture before implementation so that Day 3 can generate one coherent synthetic business rather than disconnected datasets.

---

## 1. Source Data Architecture

Six source tables were defined and locked:

1. `areas`
2. `experiment_assignment`
3. `marketing_performance`
4. `customers`
5. `marketing_touchpoints`
6. `transactions`

The source layer is intentionally separated from downstream analytical transformations.

---

## 2. `areas`

**Grain:** one row per area.

| Column      | Purpose               |
| ----------- | --------------------- |
| `area_id`   | Primary identifier    |
| `area_name` | Descriptive area name |
| `area_size` | Optional metadata     |

### Decision

`area_size` will not be used analytically.

Baseline demand, revenue, purchase behavior, and historical marketing performance will not be stored in this table. Those characteristics will be derived from historical data.

---

## 3. `experiment_assignment`

**Grain:** one row per area per experiment.

| Column             | Purpose               |
| ------------------ | --------------------- |
| `experiment_id`    | Experiment identifier |
| `area_id`          | Assigned area         |
| `experiment_group` | Treatment or control  |
| `assignment_date`  | Assignment date       |

**Key:** `experiment_id + area_id`

### Decision

Baseline metrics, treatment spend, channel, and treatment effects will not be stored in this table.

---

## 4. `marketing_performance`

**Raw grain:** one campaign × channel × date.

| Column          | Purpose                                   |
| --------------- | ----------------------------------------- |
| `date`          | Marketing activity date                   |
| `channel`       | Google, Meta, TikTok, or CRM              |
| `campaign_id`   | Campaign identifier                       |
| `campaign_name` | Campaign name containing area information |
| `spend`         | Marketing spend                           |
| `impressions`   | Impressions delivered                     |
| `clicks`        | Clicks generated                          |

### Decision

`area_id` will **not** be stored in the raw marketing table.

Area will be derived from the campaign naming convention.

Example:

```text
BrandSearch_JKT
        ↓
campaign_name parsing
        ↓
area_id = JKT
```

### Reasoning

This reflects a realistic marketing-data situation where geographic information may be embedded in campaign naming rather than provided as a clean analytical field.

---

## 5. `customers`

**Grain:** one row per customer.

| Column        | Purpose             |
| ------------- | ------------------- |
| `customer_id` | Customer identifier |
| `area_id`     | Customer's area     |

### Decision

`acquisition_date` and `customer_segment` will not be stored as source attributes.

They will be derived later from observed customer behavior.

This keeps analytical definitions separate from the source customer data.

---

## 6. `marketing_touchpoints`

**Raw grain:** one customer click or marketing interaction.

| Column                 | Purpose                                   |
| ---------------------- | ----------------------------------------- |
| `touchpoint_id`        | Unique touchpoint identifier              |
| `customer_id`          | Customer associated with the interaction  |
| `touchpoint_timestamp` | Exact interaction timestamp               |
| `channel`              | Marketing channel                         |
| `campaign_id`          | Campaign identifier                       |
| `campaign_name`        | Campaign name containing area information |

### Decision

`area_id` will be derived from `campaign_name`, consistent with `marketing_performance`.

### Touchpoint definition

A touchpoint represents a meaningful customer-level marketing interaction, specifically a **click** for this project.

Impressions will not be stored as individual customer-level touchpoints.

### Reasoning

Impressions are important for overall marketing performance but would create a much larger customer-level dataset without adding equivalent value to the planned multi-touch attribution analysis.

Keeping clicks as touchpoints also makes the customer journey more interpretable:

```text
Marketing exposure
      ↓
Click
      ↓
Customer touchpoint
      ↓
Purchase
```

---

## 7. `transactions`

**Grain:** one transaction / purchase.

| Column             | Purpose                                  |
| ------------------ | ---------------------------------------- |
| `transaction_id`   | Transaction identifier                   |
| `customer_id`      | Customer associated with the transaction |
| `transaction_date` | Purchase date                            |
| `revenue`          | Transaction revenue                      |
| `subsidy`          | Business-funded subsidy                  |
| `discount`         | Discount applied                         |
| `cancelled`        | Cancellation indicator                   |

### Decision

Transaction economics are defined as:

```text
Variable Cost
=
Subsidy + Discount

Contribution Margin
=
Revenue - Subsidy - Discount
```

Marketing spend remains exclusively in `marketing_performance` and is not included in transaction-level variable cost.

---

## 8. Source Table Relationships

The core relationships were defined as:

```text
areas
  │
  ├── experiment_assignment
  │
  ├── customers
  │
  └── marketing_performance
          │
          └── campaign_name → derive area_id


customers
  │
  ├── marketing_touchpoints
  │       │
  │       └── campaign_name → derive area_id
  │
  └── transactions
```

The customer journey therefore connects:

```text
Customer
   ↓
Marketing Touchpoints
   ↓
Transaction
```

while the experiment connects:

```text
Area
   ↓
Experiment Assignment
   ↓
Marketing Spend
   ↓
Customer Behavior
   ↓
Transactions
```

---

## 9. Attribution Architecture

The attribution design was finalized during Day 2.

### Decision

**Primary methodology:** Multi-touch attribution

**Comparison baseline:** Last-touch attribution

Last-touch is not the primary methodology and will only be used as a comparison point.

### Attribution as a derived layer

Attribution weights will not be generated as source data.

The source data will contain:

* Customer interactions
* Marketing touchpoints
* Transactions

The attribution layer will then be calculated from those records.

This creates the following architecture:

```text
Marketing Touchpoints
        +
Transactions
        ↓
Multi-touch Attribution
```

This keeps attribution logic separate from the underlying business data.

---

## 10. Attribution Window

The primary attribution window was set to **14 days**.

A touchpoint is eligible to receive attribution credit for a purchase if it occurred within 14 days before that purchase.

### Reasoning

The business is an OTA / hotel-booking platform.

The customer journey can involve:

* Research
* Hotel comparison
* Multiple visits
* Waiting before purchase
* Multiple marketing interactions

A 7-day window was considered too restrictive for this type of consideration journey.

A 30-day window could increase the likelihood of assigning credit to older interactions that have a weaker relationship with the eventual purchase.

Therefore:

**14 days is the primary attribution window.**

Other windows may be considered later as sensitivity checks, but they are not part of the primary methodology.

---

## 11. Attribution vs Incrementality

The distinction between attribution and incrementality was reinforced through the data architecture.

### Attribution

```text
Which marketing touchpoints receive credit for a purchase?
```

### Incrementality

```text
Did additional marketing activity cause additional business outcomes?
```

The attribution layer is therefore not used as causal evidence.

Incrementality will be evaluated through the geographic experiment and Difference-in-Differences.

---

## 12. Time Architecture

The project timeline was expanded and finalized.

### Historical period

**18 months**

Used to establish:

* Historical area differences
* Baseline demand
* Customer behavior
* Marketing behavior
* Seasonality
* Pre-experiment trends

### Experiment period

**28 days**

Used for the geographic treatment/control experiment.

### Post-experiment observation

**90 days**

Used to observe downstream customer behavior and support customer economics.

### Important distinction

The **14-day attribution window** and **90-day observation period** serve different purposes.

The attribution window determines whether a marketing touchpoint can receive credit for a purchase.

The observation period determines how long downstream customer behavior can be observed after the experiment.

---

## 13. Support for Downstream Analysis

The architecture was checked against the later analytical requirements.

| Analytical Need         | Primary Data                                                       |
| ----------------------- | ------------------------------------------------------------------ |
| Marketing performance   | `marketing_performance`                                            |
| Incrementality          | `marketing_performance` + `experiment_assignment` + `transactions` |
| Multi-touch attribution | `marketing_touchpoints` + `transactions`                           |
| Customer economics      | `customers` + `transactions`                                       |
| Channel response        | Marketing activity + business outcomes                             |
| Marginal economics      | Channel response + experiment evidence                             |
| Budget optimization     | Marginal economics + response curves + business constraints        |

The architecture is designed to support the full analytical path without adding unnecessary source tables.

---

## 14. Data Generation Principle

A key design decision was to generate the synthetic data as **one coherent business process**.

The intended dependency structure is:

```text
Areas
  ↓
Customers
  ↓
Marketing Activity / Interactions
  ↓
Transactions
```

with the experiment affecting marketing investment:

```text
Areas
  ↓
Experiment Assignment
  ↓
Marketing Spend
  ↓
Customer Behavior
  ↓
Transactions
```

The objective is to avoid generating independent random CSV files that do not represent the same underlying business.

---

## 15. What Was Actually Completed

Day 2 was completed as a **data architecture milestone**.

The following were defined and locked:

* OTA / hotel-booking business context
* Six source tables
* Source-table grains
* Primary and join keys
* Raw versus derived data boundaries
* Campaign-name-based area extraction
* Customer-level click-only touchpoints
* Multi-touch attribution as the primary methodology
* Last-touch as a comparison baseline
* 14-day primary attribution window
* 18-month historical period
* 28-day experiment period
* 90-day post-experiment observation period
* Transaction-level subsidy and discount structure
* Contribution-margin calculation
* Source-table relationships
* Architecture required for downstream customer economics
* Architecture required for channel response and budget optimization
* One coherent data-generation approach

---

## 16. Not Completed

The following were discussed or planned but were **not completed during Day 2**:

* Synthetic data generation
* Python data-generation implementation
* SQL transformations
* Attribution implementation
* Data-quality injection
* Data validation
* Funnel reconstruction
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

## 17. Open Questions

The following remain open for implementation:

* Exact historical dates
* Exact experiment dates
* Exact area identifiers and names
* Exact stratification variables
* Exact implementation of the 25% treatment increase at campaign level
* Exact customer-generation parameters
* Exact purchase-behavior rules
* Exact data-quality issues and rates
* Exact multi-touch attribution weighting logic
* Exact channel response-curve structure
* Exact customer economics and cohort rules
* Exact operational constraints for the $2M optimization
* How aggregate experiment evidence will be connected to channel-level marginal economics

These are implementation and modeling questions, not completed analytical results.

---

## Day 2 Conclusion

Day 2 established and locked the project's source-data architecture.

The project now has defined source tables, row grains, keys, relationships, raw versus derived fields, customer touchpoint rules, attribution-window logic, and the overall timeline needed to support the later causal and economic analysis.

The important boundary is that **Day 2 produced the data blueprint, not the data or analytical results**.

The synthetic business has not yet been generated.