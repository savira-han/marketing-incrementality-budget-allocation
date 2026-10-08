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

# Day 3 - Customer Lifecycle

**Status:** In Progress
**Focus:** Build and validate a coherent customer acquisition and purchase lifecycle across the historical, experiment, and post-experiment periods.

---

## Objective

Implement the first major component of the synthetic business simulator: a persistent customer lifecycle connecting customer entry, marketing exposure, first purchase, repeat purchase, and customer-level behavioral characteristics.

The objective was to ensure that customers and transactions are generated from the same underlying business process rather than as independent random datasets.

---

## 1. Customer Generation Dependency

### Change

Customer generation was moved to occur after marketing performance generation and validation.

The customer-generation process now receives both:

* `area_characteristics`
* `marketing_performance`

Customer entry dates were extended across the full simulation period:

```text
2025-01-01 → 2026-10-26
```

### Why

The original customer-generation structure relied primarily on fixed customer-volume targets and area weights.

This did not adequately represent customer acquisition as a response to the underlying business and marketing environment.

The revised dependency is:

```text
Area characteristics
        +
Marketing activity
        ↓
Customer acquisition
        ↓
Customer population
```

This also allows new customers to enter during the experiment and post-experiment periods.

---

## 2. Marketing-Responsive Customer Acquisition

### Change

Customer acquisition was redesigned around area-day marketing activity.

Marketing performance is aggregated to area and date, with clicks used as a measure of marketing pressure.

Customer acquisition combines:

* baseline area demand
* customer volume potential
* marketing responsiveness
* daily marketing activity
* diminishing marketing response
* Poisson sampling

### Why

Customer acquisition should respond to the marketing environment while retaining natural stochastic variation.

The diminishing-response function prevents customer acquisition from increasing linearly without limit as marketing activity increases.

The Poisson process introduces realistic variation around expected customer acquisition.

---

## 3. Customer-Level Persistent Characteristics

### Change

Persistent hidden customer characteristics were retained and connected to downstream behavior:

* `purchase_propensity`
* `aov_tendency`
* `repeat_purchase_tendency`
* `price_sensitivity`
* Google responsiveness
* Meta responsiveness
* TikTok responsiveness
* CRM responsiveness

These characteristics remain simulation state and are not exposed in the analyst-facing `customers` source table.

### Why

Customers should exhibit heterogeneous behavior rather than behaving identically.

Persistent characteristics allow the same customer to maintain behavioral tendencies across marketing exposure, purchasing, repeat behavior, and later customer economics.

---

## 4. Customer Lifecycle Redesign

### Change

The transaction-generation process was redesigned from continuous daily purchase eligibility to a state-based lifecycle.

The previous mechanism allowed an active customer to attempt a purchase every day after entry.

The revised lifecycle introduces an explicit `next_purchase_date` state.

The resulting structure is:

```text
Customer entry
      ↓
First purchase opportunity
      ↓
Completed purchase
      ↓
Repeat waiting period
      ↓
Next purchase opportunity
      ↓
Repeat purchase
      ↓
Potential additional repeat purchase
```

### Why

The previous implementation did not represent realistic repeat-purchase timing.

A customer could repeatedly attempt to purchase immediately after a previous purchase, making repeat behavior resemble repeated independent purchase trials.

The revised state-based approach creates a meaningful interval between completed purchases and allows `repeat_purchase_tendency` to influence customer return timing.

---

## 5. Repeat Purchase Behavior

### Change

Repeat purchase timing is now influenced by customer-level `repeat_purchase_tendency`.

Customers with stronger repeat tendencies receive shorter expected return intervals, while stochastic variation is retained.

A minimum waiting period is enforced between completed purchases.

### Why

`repeat_purchase_tendency` should affect observable customer behavior rather than function only as a probability multiplier.

The revised mechanism creates a direct relationship between:

```text
Customer characteristic
        ↓
Return timing
        ↓
Repeat purchase behavior
```

---

## 6. Cancellation and Customer State

### Change

Cancelled transactions remain in the `transactions` table but do not advance the completed-purchase lifecycle.

Completed purchases update:

* purchase count
* last purchase date
* next purchase date

Cancelled transactions create another purchase opportunity after a short interval.

### Why

A cancelled transaction represents a transaction event but not a completed customer purchase.

Separating transaction events from completed purchases is required for later analysis of:

* cancellation rate
* realized revenue
* subsidy
* discount
* contribution margin
* customer value

---

## 7. Customer Lifecycle Validation

The revised lifecycle was validated using the generated synthetic data.

| Metric                       |   Result |
| ---------------------------- | -------: |
| Customers                    |  121,637 |
| Transactions                 |   51,619 |
| Customers with 0 purchases   |   82,376 |
| Customers with 1 purchase    |   30,379 |
| Customers with 2 purchases   |    7,353 |
| Customers with 3+ purchases  |    1,529 |
| Cancelled transactions       |    1,756 |
| Cancellation rate            |    3.40% |
| First purchases              |   39,261 |
| Repeat purchases             |   10,602 |
| Minimum purchase gap         |  19 days |
| Median purchase gap          | 191 days |
| 75th percentile purchase gap | 275 days |
| Maximum purchase gap         | 652 days |

The resulting population contains non-purchasers, first-time purchasers, repeat purchasers, and frequent purchasers.

---

## 8. Repeat Behavior Validation

Customer behavior was compared across quintiles of `repeat_purchase_tendency`.

| Repeat Tendency | Average Purchases | Repeat Customer Rate |
| --------------- | ----------------: | -------------------: |
| Lowest 20%      |             0.407 |                6.82% |
| 20-40%          |             0.405 |                7.08% |
| 40-60%          |             0.405 |                7.01% |
| 60-80%          |             0.412 |                7.51% |
| Highest 20%     |             0.422 |                8.09% |

Higher repeat tendency produces a higher repeat-customer rate.

The relationship is not perfectly monotonic because repeat behavior is also affected by marketing exposure, purchase propensity, seasonality, area characteristics, and customer observation time.

---

## 9. Repeat Timing Validation

Repeat purchase timing was compared across quintiles of `repeat_purchase_tendency`.

| Repeat Tendency | Median Gap | Average Gap |
| --------------- | ---------: | ----------: |
| Lowest 20%      | 173.0 days |  182.1 days |
| 20-40%          | 160.5 days |  175.9 days |
| 40-60%          | 165.0 days |  178.1 days |
| 60-80%          | 153.5 days |  175.1 days |
| Highest 20%     | 151.5 days |  165.4 days |

Higher repeat tendency is associated with shorter repeat-purchase intervals.

This confirms that the hidden customer characteristic influences observable repeat-purchase timing as intended.

---

## 10. Customer State Consistency

The final lifecycle consistency checks produced:

| Validation                             |  Result |
| -------------------------------------- | ------: |
| Transactions with valid customer IDs   |  51,619 |
| Transactions before customer entry     |       0 |
| Transactions outside simulation period |       0 |
| Completed first purchases              |  39,261 |
| Completed repeat purchases             |  10,602 |
| Customers with completed purchases     |  39,261 |
| Invalid purchase ordering              |       0 |
| Minimum completed-purchase gap         | 19 days |

All customer lifecycle consistency checks passed.

---

## 11. Day 3 Decisions

The following decisions were finalized during this stage:

* Customer acquisition is responsive to the marketing environment.
* Customer entry spans the full simulation period.
* Customer-level behavioral characteristics remain persistent hidden simulation state.
* Customer lifecycle is state-based rather than continuously purchase-eligible.
* Repeat purchases require a waiting period.
* Repeat tendency influences repeat-purchase timing.
* Cancelled transactions do not advance completed-purchase state.
* Treatment spend remains isolated to the defined experiment period.
* Treatment and control customer counts are not forced to be identical.
* Customer lifecycle validation is based on structural consistency and behavioral relationships rather than fixed target counts.

---

## 12. Day 3 Conclusion

The customer lifecycle has been implemented and validated as a coherent component of the synthetic business simulator.

The current lifecycle is considered sufficiently robust for downstream development and is locked for the current simulation stage.

The validated dependency is:

```text
Area characteristics
        ↓
Customer entry
        ↓
Persistent customer behavior
        ↓
Marketing exposure
        ↓
First purchase
        ↓
Repeat waiting period
        ↓
Repeat purchase
        ↓
Customer value
```

Further lifecycle tuning is not required at this stage.

The next stage will extend the validated customer lifecycle into **customer economics**, including AOV, revenue, subsidy, discount, cancellation economics, contribution margin, and customer value.

# Day 4 - Customer Economics & Purchasing Behavior

Status: Completed
Focus: Build and validate customer-level transaction economics, repeat-purchase behavior, and contribution-margin inputs as part of the synthetic business simulator.

---

## Objective

Extend the validated customer lifecycle into a coherent customer economics system.

The objective was to ensure that transaction value, promotional costs, contribution margin, customer value, and repeat behavior are generated from persistent customer and area characteristics rather than independent random values.

The economic system also needed to remain consistent with the project's primary business outcome:

> Incremental Contribution Margin

---

## 1. Economic Currency

### Decision

All transaction economics are denominated in USD.

This includes:

* Revenue
* AOV
* Discount
* Subsidy
* Contribution margin

### Change

The original AOV scale was adjusted from an IDR-like magnitude to a USD-denominated hotel-booking scale.

The customer and area AOV tendencies were changed to use:

```
`np.log(60)`
```

while preserving the existing distribution structure and variation.

### Why

Marketing spend was already denominated in USD.

Using a consistent currency across marketing spend and transaction economics makes the later contribution-margin and budget-allocation analysis internally coherent.

The distribution shape was preserved rather than changing the underlying variability of customer economics.

---

## 2. Transaction Economics

### Decision

The raw `transactions` table contains economic inputs rather than a pre-calculated contribution-margin field.

The source fields remain:

```
`transaction_id
customer_id
transaction_date
revenue
subsidy
discount
cancelled
`
```

Contribution margin is derived later as:

```
`Contribution Margin
=
Revenue - Subsidy - Discount
`
```

Marketing spend remains separate in `marketing_performance`.

It is not deducted from transaction-level contribution margin.

### Why

This keeps the raw transaction data focused on observable transaction economics while allowing contribution margin to remain a derived analytical measure.

Marketing profitability and incremental contribution margin from marketing investment will be evaluated later by connecting transaction economics with marketing spend and causal evidence.

---

## 3. AOV Generation

### Change

Transaction revenue is generated from persistent customer and area-level AOV tendencies with additional transaction-level variation.

The dependency is:

```
`Customer AOV tendency
        +
Area AOV tendency
        +
Transaction-level variation
        ↓
Realized transaction revenue
`
```

### Why

Customers should have persistent differences in their typical booking value while transactions should still contain natural stochastic variation.

This avoids making AOV completely deterministic from a hidden customer characteristic.

---

## 4. Promotional Cost Generation

### Change

Discount and subsidy are generated as transaction-level rates applied to the underlying booking value.

Customer `price_sensitivity` influences both rates.

The resulting structure is:

```
`Price sensitivity
        ↓
Discount / Subsidy rate
        ↓
Promotional cost
        ↓
Contribution margin
`
```

Discount and subsidy are set to zero for cancelled transactions.

### Why

Price sensitivity should affect the economics of a customer's transaction rather than being stored as an unused hidden characteristic.

This creates a direct connection between persistent customer behavior and realized transaction economics.

---

## 5. Repeat Purchase Economics

### Change

Repeat purchase behavior was strengthened so that `repeat_purchase_tendency` influences both:

* Repeat-purchase probability
* Time between completed purchases

The repeat-purchase probability uses a stronger tendency effect:

```
`1.0 + 1.25 × repeat_purchase_tendency`
```

The expected repeat gap was revised to:

```
`120 - 90 × repeat_purchase_tendency`
```

with the existing minimum waiting period and stochastic Gamma variation retained.

### Why

The initial mechanism produced only a weak observable relationship between repeat tendency and overall purchase frequency.

The revised mechanism makes repeat tendency sufficiently influential to appear in observable repeat behavior while preserving stochastic variation from marketing exposure, purchase propensity, seasonality, area characteristics, and observation time.

---

## 6. Customer Value

Customer value is generated naturally through the combination of:

```
`Purchase frequency
      ×
Realized transaction economics
      ↓
Customer revenue
      ↓
Customer contribution margin
`
```

No fixed customer lifetime value is assigned during transaction generation.

Customer-level value is derived from completed transaction history.

### Validation

Customer value increased naturally with purchase frequency:

| Purchase Group | Customers | Mean Revenue | Mean Contribution Margin |
| -------------- | --------: | -----------: | -----------------------: |
| 1 purchase     |    28,551 |       $64.68 |                   $52.10 |
| 2 purchases    |     8,308 |      $129.21 |                  $104.58 |
| 3+ purchases   |     2,343 |      $203.66 |                  $165.03 |

This confirms that customer value emerges from actual purchasing behavior rather than being directly imposed.

---

## 7. Economic Validation

The generated transaction economics were validated after the USD scale revision.

| Metric                                | Result  |
| ------------------------------------- | ------- |
| Total transactions                    | 54,507  |
| Completed transactions                | 52,664  |
| Cancelled transactions                | 1,843   |
| Cancellation rate                     | 3.38%   |
| Mean AOV                              | $64.51  |
| Median AOV                            | $57.59  |
| P25 AOV                               | $42.47  |
| P75 AOV                               | $78.39  |
| P95 AOV                               | $125.19 |
| Mean discount rate                    | 9.16%   |
| Mean subsidy rate                     | 10.11%  |
| Mean contribution margin              | $52.09  |
| Contribution-margin rate              | 80.74%  |
| Negative contribution-margin bookings | 0.00%   |
| Mean customer revenue                 | $86.66  |
| Mean customer contribution margin     | $69.97  |

The resulting AOV distribution is consistent with the intended USD-denominated hotel-booking economics.

---

## 8. Area-Level Economic Variation

Area-level economics were checked to ensure that transaction value was not identical across geographic units.

The resulting area-level AOV distribution was:

```
`Minimum AOV: $41.50
Maximum AOV: $114.48
Mean AOV: $64.81
Standard deviation: $17.08
`
```

This confirms that area-level AOV tendencies create meaningful variation while remaining within a plausible overall range.

---

## 9. Persistent Characteristic Validation

The economic and behavioral mechanisms were validated against the hidden customer characteristics.

| Metric                                   | Result |
| ---------------------------------------- | ------ |
| AOV tendency → realized AOV              | 0.735  |
| Purchase propensity → purchase frequency | 0.328  |
| Repeat tendency → repeat purchases       | 0.128  |
| Repeat tendency → actual repeat gap      | -0.126 |
| Price sensitivity → discount rate        | 0.175  |
| Price sensitivity → subsidy rate         | 0.183  |

These relationships are not expected to be deterministic.

The objective is for each persistent characteristic to influence the behavior it is intended to represent while allowing other customer, area, marketing, seasonal, and stochastic factors to affect the final outcome.

---

## 10. Repeat Behavior Validation

Repeat customer rates were compared across quartiles of `repeat_purchase_tendency`.

| Tendency Quartile | Customers | Repeat Customer Rate |
| ----------------- | --------: | -------------------: |
| Q1                |    30,410 |                7.41% |
| Q2                |    30,409 |                8.11% |
| Q3                |    30,409 |                9.02% |
| Q4                |    30,409 |               10.49% |

Higher repeat tendency produces a higher repeat-customer rate.

Among repeat customers, the correlation between repeat tendency and number of repeat purchases was:

```
`0.128`
```

Actual repeat-purchase gaps were also validated.

| Tendency Quartile | Mean Repeat Gap | Median Repeat Gap |
| ----------------- | --------------: | ----------------: |
| Q1                |      175.3 days |        163.0 days |
| Q2                |      174.1 days |        160.0 days |
| Q3                |      162.5 days |        144.0 days |
| Q4                |      151.5 days |        131.5 days |

The correlation between repeat tendency and actual repeat gap was:

```
`-0.126`
```

Higher repeat tendency is therefore associated with both a higher likelihood of repeat purchasing and shorter intervals between purchases.

---

## 11. Price Sensitivity Validation

The original validation of `price_sensitivity` against customer contribution margin was not considered an appropriate mechanism test.

Contribution margin is affected by several factors, including:

* AOV
* Purchase frequency
* Discount
* Subsidy

Therefore, customer contribution margin is not expected to have a strong direct relationship with price sensitivity.

The mechanism was instead validated against the economic inputs directly.

The resulting correlations were:

```
`Price sensitivity → discount rate: 0.175

Price sensitivity → subsidy rate: 0.183
`
```

Both relationships are positive and consistent with the intended simulation mechanism.

---

## 12. Cancellation Economics

Cancelled transactions remain in the transaction table as transaction events but do not contribute realized revenue or promotional costs.

Validation produced:

```
`Cancelled revenue: $0.00
Cancelled discount: $0.00
Cancelled subsidy: $0.00
Cancelled contribution margin: $0.00
`
```

The cancellation rate was:

```
`3.38%`
```

This keeps cancelled transactions available for operational analysis while preventing cancelled bookings from contributing to realized customer economics.

---

## 13. First vs Repeat Purchase Economics

First and repeat purchases were compared to determine whether the revised lifecycle created unrealistic differences in transaction value.

| Metric              | First Purchase | Repeat Purchase |
| ------------------- | -------------: | --------------: |
| AOV                 |         $64.61 |          $64.23 |
| Contribution Margin |         $52.23 |          $51.66 |

The results are close.

This is considered acceptable because the current simulation is designed for repeat behavior to primarily influence purchase frequency and timing rather than impose a separate AOV regime for repeat customers.

Customer value therefore grows primarily through additional completed purchases rather than artificially increasing the value of each repeat transaction.

---

## 14. Day 4 Decisions

The following decisions were finalized during this stage:

* Transaction economics are denominated in USD.
* AOV is generated from persistent customer and area tendencies with transaction-level variation.
* Discount and subsidy are transaction-level economic inputs.
* Price sensitivity influences both discount and subsidy rates.
* Contribution margin is derived as `revenue - subsidy - discount`.
* Contribution margin is not stored as a raw transaction field.
* Marketing spend remains separate from transaction-level contribution margin.
* Customer value emerges from completed transaction history.
* Repeat purchase tendency influences repeat-purchase probability.
* Repeat purchase tendency influences repeat-purchase timing.
* First and repeat purchases do not require materially different AOV distributions.
* Cancellation economics remain separate from realized customer economics.
* Persistent customer characteristics are validated against the behavior they are intended to influence.
* Economic mechanisms are considered sufficiently realistic for downstream development.

---

## 15. What Was Actually Completed

Customer economics and purchasing behavior were implemented and validated as the next component of the synthetic business simulator.

The following were completed:

* USD-denominated transaction economics
* Customer and area AOV tendencies
* Transaction-level AOV variation
* Discount generation
* Subsidy generation
* Price-sensitivity mechanism
* Contribution-margin derivation
* Customer-level economic aggregation
* Repeat-purchase probability mechanism
* Repeat-purchase timing mechanism
* Customer-value validation
* Area-level economic validation
* Cancellation economics validation
* First versus repeat purchase validation
* Persistent-characteristic validation

---

## 16. Day 4 Conclusion

Customer economics and purchasing behavior have been implemented and validated as a coherent extension of the customer lifecycle.

The resulting system produces plausible USD-denominated booking economics while maintaining meaningful customer-level and area-level variation.

The key validated dependency is:

```
`Persistent customer characteristics
            ↓
    Purchase behavior
            ↓
    Transaction economics
            ↓
    Customer value
            ↓
    Contribution margin
`
```

The economic mechanisms are considered sufficiently robust and are locked for the current simulation stage.

Further changes to the economic generation mechanism are not required at this stage.

The next stage will extend the synthetic business into the marketing response system.

# Day 5 - Marketing Response System

Status: Completed
Focus: Build and validate a coherent marketing response mechanism connecting marketing activity, customer-level responsiveness, diminishing returns, purchase probability, and transactions.

---

## Objective

Extend the validated customer economics system into a marketing response system.

The objective was to ensure that marketing activity affects customer purchasing behavior through a coherent mechanism rather than being independently generated from transaction outcomes.

The intended dependency is:

```
`Marketing spend
        ↓
    Marketing clicks
        ↓
  Marketing pressure
        ↓
Channel responsiveness
        ↓
 Marketing response
        ↓
 Diminishing returns
        ↓
 Purchase probability
        ↓
   Transactions
        ↓
Contribution margin
`
```

The mechanism needed to preserve customer and area heterogeneity while producing a positive but diminishing response to additional marketing pressure.

---

## 1. Marketing Pressure

### Change

Marketing performance was aggregated from campaign-level data into daily area-level channel activity.

Marketing pressure was defined as:

```
`Marketing Pressure
=
Channel Clicks / Area Customer Count
`
```

Area information was derived from the campaign naming convention.

### Validation

Experiment-period treatment and control pressure was compared by channel.

| Channel | Treatment / Control Pressure |
| ------- | ---------------------------- |
| Google  | 1.3185                       |
| Meta    | 1.3170                       |
| TikTok  | 1.2962                       |
| CRM     | 1.3012                       |

Treatment areas consistently received higher marketing pressure than control areas.

The increase is consistent with the defined 25% treatment spend increase while preserving the existing channel mix.

### Conclusion

The marketing-performance layer produces the intended difference in customer-level marketing pressure between treatment and control areas.

---

## 2. Channel Responsiveness

### Change

Customer-level channel responsiveness was connected to area-level marketing pressure.

The response mechanism is:

```
`Channel pressure
        ×
Customer channel responsiveness
        ↓
Marketing response
`
```

Customers were evaluated across responsiveness quintiles for each channel.

### Validation

The relationship between responsiveness and modeled marketing response was monotonic across all four channels.

| Channel | Q1 Response | Q5 Response |
| ------- | ----------- | ----------- |
| Google  | 0.6390      | 1.4763      |
| Meta    | 0.3103      | 0.7205      |
| TikTok  | 0.1989      | 0.4615      |
| CRM     | 0.0881      | 0.2041      |

Higher customer responsiveness therefore produces higher marketing response.

### Conclusion

Customer-level channel responsiveness is functioning as an observable driver of the simulated marketing-response mechanism.

---

## 3. Diminishing Returns

### Change

The marketing-effect function used by the transaction-generation process was independently reconstructed and tested across an increasing range of marketing response.

The mechanism follows:

```
`Marketing Effect
=
1
+
0.18
× log(1 + Marketing Response / 0.03)
× Area Marketing Responsiveness
`
```

### Validation

Marketing effect increased as marketing response increased, while the incremental effect declined.

| Response | Marketing Effect | Incremental Effect |
| -------- | ---------------- | ------------------ |
| 0.4008   | 1.4946           | —                  |
| 0.8006   | 1.6164           | 0.1219             |
| 1.2004   | 1.6894           | 0.0729             |
| 1.6002   | 1.7416           | 0.0522             |
| 2.0000   | 1.7823           | 0.0407             |

### Conclusion

The response function produces the intended diminishing-return behavior.

This is important for the later marginal-return and budget-allocation analysis because additional marketing investment does not generate a constant incremental response.

---

## 4. Treatment Response

### Change

Experiment-period transaction outcomes were compared between treatment and control areas.

Contribution margin was derived as:

```
`Contribution Margin
=
Revenue - Subsidy - Discount
`
```

The primary treatment-response diagnostics focused on transactions and completed transactions rather than requiring contribution margin to move in a specific direction.

### Validation

Treatment and control outcomes were:

| Metric                              | Control | Treatment | Treatment / Control |
| ----------------------------------- | ------- | --------- | ------------------- |
| Customers                           | 59,220  | 62,417    | —                   |
| Transactions per customer           | —       | —         | 1.0017              |
| Completed transactions per customer | —       | —         | 1.0010              |
| Contribution margin per customer    | $1.6917 | $1.5715   | 0.9289              |

Contribution margin was retained as an economic outcome diagnostic rather than a directional mechanism assertion because it is affected by AOV, discounts, subsidies, cancellations, customer composition, and stochastic purchasing behavior.

### Conclusion

The realized transaction uplift is modest and noisy, but this does not by itself indicate a problem with the marketing-response mechanism.

---

## 5. Entry Timing Diagnostic

### Change

The treatment-response comparison was repeated using only customers whose entry date occurred before the experiment began.

This isolates the experiment-period response from differences in the number of newly acquired customers entering treatment and control areas during the experiment.

### Validation

The pre-existing customer population produced:

| Metric                              | Control | Treatment | Treatment / Control |
| ----------------------------------- | ------- | --------- | ------------------- |
| Customers                           | 48,636  | 51,128    | —                   |
| Transactions                        | 1,827   | 1,924     | —                   |
| Completed transactions per customer | 0.0362  | 0.0363    | 1.0020              |
| Contribution margin per customer    | $2.0235 | $1.8747   | 0.9265              |

Transactions per customer remained approximately flat between treatment and control.

### Conclusion

Differences in customer entry timing are not the primary explanation for the weak realized treatment uplift.

The analysis therefore proceeded to isolate the modeled purchase-probability mechanism directly.

---

## 6. Purchase Probability Diagnostic

### Change

The purchase-probability calculation used by the transaction-generation process was independently reconstructed.

The diagnostic incorporated:

* Customer purchase propensity
* Area purchase propensity
* Customer channel responsiveness
* Area marketing responsiveness
* Experiment-period marketing pressure
* July seasonality
* The modeled marketing-effect function

A first-purchase scenario was used to isolate marketing response from repeat-purchase behavior.

### Validation

Actual treatment and control populations produced:

| Metric               | Control  | Treatment | Treatment / Control |
| -------------------- | -------- | --------- | ------------------- |
| Marketing response   | 1.697843 | 2.183515  | 1.2861              |
| Marketing effect     | 1.786628 | 1.777966  | 0.9952              |
| Purchase probability | 0.001411 | 0.001407  | 0.9973              |

Treatment areas therefore had higher modeled marketing response, but differences in customer and area characteristics largely offset the response at the purchase-probability level.

### Conclusion

A direct comparison of actual treatment and control populations is not sufficient to evaluate the underlying marketing mechanism because population composition affects the modeled outcome.

A counterfactual comparison was therefore used as the final mechanism diagnostic.

---

## 7. Counterfactual Marketing Response

### Change

A counterfactual diagnostic was constructed using the same customers and the same customer and area characteristics under two scenarios:

* Control-level marketing pressure
* Treatment-level marketing pressure

Only marketing pressure was changed.

This isolates the effect of marketing pressure from treatment/control population composition.

### Validation

| Metric               | Control Pressure | Treatment Pressure | Treatment / Control |
| -------------------- | ---------------- | ------------------ | ------------------- |
| Marketing response   | 1.699475         | 2.231833           | 1.3132              |
| Marketing effect     | 1.761743         | 1.812321           | 1.0287              |
| Purchase probability | 0.001394         | 0.001434           | 1.0288              |

Holding customer and area characteristics constant:

* Treatment pressure produces 31.3% higher marketing response.
* Diminishing returns reduce this to approximately 2.9% higher marketing effect.
* Purchase probability is approximately 2.9% higher under treatment-level pressure.

### Conclusion

The counterfactual diagnostic confirms that the marketing-response mechanism operates in the intended direction.

The relatively small purchase-probability effect is a direct consequence of the diminishing-return function and the other components of the purchase-probability model.

---

## 8. Day 5 Decisions

The following decisions were finalized during this stage:

* Marketing pressure is defined from area-level channel clicks relative to area customer count.
* Treatment areas receive higher marketing pressure than control areas.
* Customer-level channel responsiveness creates heterogeneous marketing response.
* Marketing response increases with marketing pressure.
* Marketing effect follows a diminishing-return function.
* Marketing response contributes positively to purchase probability.
* Treatment/control population composition can materially affect the observed treatment comparison.
* Counterfactual pressure comparison is used to isolate the underlying marketing-response mechanism.
* The realized experiment does not need to reproduce the modeled treatment effect exactly because transactions remain stochastic.
* Contribution margin is treated as an economic outcome rather than a directional validation requirement for the marketing mechanism.
* The marketing-response mechanism is considered sufficiently credible for downstream customer journey, attribution, and incrementality analysis.
* No changes to `generate_transactions()` are required at this stage.

---

## 9. What Was Actually Completed

The marketing response system was implemented and validated as the next component of the synthetic business simulator.

The following were completed:

* Area-level marketing pressure
* Treatment/control pressure validation
* Customer-level channel responsiveness
* Channel responsiveness validation
* Diminishing-return response function
* Treatment-response diagnostics
* Entry-timing diagnostic
* Purchase-probability reconstruction
* Counterfactual marketing-response validation
* Marketing-response mechanism validation

---

## 10. Day 5 Conclusion

The marketing response system has been implemented and validated as a coherent extension of the customer lifecycle and customer economics systems.

The validated dependency is:

```
`Marketing spend
        ↓
    Marketing clicks
        ↓
  Marketing pressure
        ↓
Customer responsiveness
        ↓
 Marketing response
        ↓
 Diminishing returns
        ↓
 Purchase probability
        ↓
   Transactions
        ↓
Contribution margin
`
```

The counterfactual validation confirms that higher marketing pressure produces higher modeled purchase probability when customer and area characteristics are held constant.

The resulting marketing-response mechanism is considered sufficiently credible for downstream development and is locked for the current simulation stage.

Further tuning of the marketing-response generation mechanism is not required at this stage.

The next stage will extend the validated synthetic business into customer journeys and attribution.
