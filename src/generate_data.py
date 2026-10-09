import os
import resource

from pathlib import Path

import numpy as np
import pandas as pd


# ======
# memory check function
# ======
def print_memory(label):
    memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[MEMORY] {label}: {memory_mb:,.0f} MB")


# ============================================================
# Simulation configuration
# ============================================================

SEED = 42

HISTORICAL_START = pd.Timestamp("2025-01-01")
HISTORICAL_END = pd.Timestamp("2026-06-30")

EXPERIMENT_START = pd.Timestamp("2026-07-01")
EXPERIMENT_END = pd.Timestamp("2026-07-28")

POST_EXPERIMENT_START = pd.Timestamp("2026-07-29")
POST_EXPERIMENT_END = pd.Timestamp("2026-10-26")

ATTRIBUTION_WINDOW_DAYS = 14
POST_EXPERIMENT_DAYS = 90

N_AREAS = 20
N_TREATMENT_AREAS = 10
N_CONTROL_AREAS = 10

TARGET_MIN_CUSTOMERS = 120_000
TARGET_MAX_CUSTOMERS = 150_000

TREATMENT_SPEND_MULTIPLIER = 1.25

CHANNEL_MIX = {
    "Google": 0.40,
    "Meta": 0.30,
    "TikTok": 0.20,
    "CRM": 0.10,
}

ELIGIBLE_CHANNELS = list(CHANNEL_MIX.keys())

RANDOM_GENERATOR = np.random.default_rng(SEED)

# ============================================================
# Channel Parameters
# ============================================================

CHANNEL_PARAMETERS = {
    "Google": {
        "cpc": 2.20,
        "impressions_per_dollar": 220,
        "base_ctr": 0.035,
    },
    "Meta": {
        "cpc": 1.60,
        "impressions_per_dollar": 480,
        "base_ctr": 0.012,
    },
    "TikTok": {
        "cpc": 1.10,
        "impressions_per_dollar": 700,
        "base_ctr": 0.009,
    },
    "CRM": {
        "cpc": 0.35,
        "impressions_per_dollar": 180,
        "base_ctr": 0.025,
    },
}

CAMPAIGN_TYPE_MULTIPLIERS = {
    "BrandSearch": {
        "spend": 0.80,
        "ctr": 1.40,
    },
    "GenericHotelSearch": {
        "spend": 1.00,
        "ctr": 1.10,
    },
    "DestinationSearch": {
        "spend": 0.90,
        "ctr": 1.00,
    },
    "Prospecting": {
        "spend": 1.10,
        "ctr": 0.85,
    },
    "Retargeting": {
        "spend": 0.80,
        "ctr": 1.20,
    },
    "HotelPromotion": {
        "spend": 1.00,
        "ctr": 1.00,
    },
    "TravelDiscovery": {
        "spend": 1.10,
        "ctr": 0.80,
    },
    "DestinationContent": {
        "spend": 0.90,
        "ctr": 0.85,
    },
    "Reengagement": {
        "spend": 0.80,
        "ctr": 1.10,
    },
    "RepeatBooking": {
        "spend": 0.70,
        "ctr": 1.20,
    },
    "PromotionalOffer": {
        "spend": 1.00,
        "ctr": 1.00,
    },
}

CAMPAIGN_TYPE_WEIGHTS = {
    "Google": {
        "BrandSearch": 0.80,
        "GenericHotelSearch": 1.00,
        "DestinationSearch": 0.90,
    },
    "Meta": {
        "Prospecting": 1.10,
        "Retargeting": 0.80,
        "HotelPromotion": 1.00,
    },
    "TikTok": {
        "TravelDiscovery": 1.10,
        "DestinationContent": 0.90,
        "HotelPromotion": 1.00,
    },
    "CRM": {
        "Reengagement": 0.80,
        "RepeatBooking": 0.70,
        "PromotionalOffer": 1.00,
    },
}

BASE_DAILY_SPEND = 27_400

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SIMULATION_TRUTH_DIR = PROJECT_ROOT / "data" / "simulation_truth"

def generate_areas():
    """Generate observable area records and hidden simulation characteristics."""

    area_names = [
        "Jakarta",
        "Bandung",
        "Surabaya",
        "Yogyakarta",
        "Semarang",
        "Medan",
        "Denpasar",
        "Makassar",
        "Malang",
        "Palembang",
        "Tangerang",
        "Bekasi",
        "Depok",
        "Bogor",
        "Batam",
        "Balikpapan",
        "Solo",
        "Lombok",
        "Manado",
        "Padang",
    ]

    if len(area_names) != N_AREAS:
        raise ValueError(
            f"Expected {N_AREAS} areas, found {len(area_names)}."
        )

    area_ids = [f"AREA_{i:03d}" for i in range(1, N_AREAS + 1)]

    areas = pd.DataFrame(
        {
            "area_id": area_ids,
            "area_name": area_names,
            "area_size": RANDOM_GENERATOR.integers(
                low=40,
                high=1_200,
                size=N_AREAS,
            ).astype(float),
        }
    )

    # Introduce realistic missingness in optional metadata.
    missing_indices = RANDOM_GENERATOR.choice(
        N_AREAS,
        size=3,
        replace=False,
    )

    areas.loc[missing_indices, "area_size"] = np.nan

    # Hidden characteristics used only by the simulator.
    area_characteristics = pd.DataFrame(
        {
            "area_id": area_ids,
            "baseline_demand": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.35,
                size=N_AREAS,
            ),
            "customer_volume_potential": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=N_AREAS,
            ),
            "purchase_propensity": RANDOM_GENERATOR.beta(
                a=5,
                b=25,
                size=N_AREAS,
            ),
            "aov_tendency": RANDOM_GENERATOR.lognormal(
                mean=np.log(60),
                sigma=0.20,
                size=N_AREAS,
            ),
            "seasonality_sensitivity": RANDOM_GENERATOR.normal(
                loc=1.0,
                scale=0.12,
                size=N_AREAS,
            ),
            "marketing_responsiveness": RANDOM_GENERATOR.normal(
                loc=1.0,
                scale=0.15,
                size=N_AREAS,
            ),
        }
    )

    return areas, area_characteristics


def generate_campaign_configuration(area_characteristics):
    """Create campaign metadata used by the marketing simulator."""

    area_ids = area_characteristics["area_id"].tolist()

    campaign_types = {
        "Google": [
            "BrandSearch",
            "GenericHotelSearch",
            "DestinationSearch",
        ],
        "Meta": [
            "Prospecting",
            "Retargeting",
            "HotelPromotion",
        ],
        "TikTok": [
            "TravelDiscovery",
            "DestinationContent",
            "HotelPromotion",
        ],
        "CRM": [
            "Reengagement",
            "RepeatBooking",
            "PromotionalOffer",
        ],
    }

    campaign_records = []

    campaign_number = 1

    for channel, campaign_names in campaign_types.items():
        for campaign_type in campaign_names:
            for area_id in area_ids:

                campaign_id = f"CMP_{campaign_number:04d}"

                campaign_name = (
                    f"{channel}_{campaign_type}_{area_id}"
                )

                campaign_records.append(
                    {
                        "campaign_id": campaign_id,
                        "channel": channel,
                        "campaign_type": campaign_type,
                        "area_id": area_id,
                        "campaign_name": campaign_name,
                    }
                )

                campaign_number += 1

    return pd.DataFrame(campaign_records)

def generate_festival_campaign_configuration(area_characteristics):
    """Create dedicated 10.10 Travel Festival campaign metadata."""

    area_ids = area_characteristics["area_id"].tolist()

    campaign_records = []

    campaign_number = 1000

    for channel in ELIGIBLE_CHANNELS:
        for area_id in area_ids:

            campaign_id = f"CMP_{campaign_number:04d}"

            campaign_name = (
                f"{channel}_10.10_{area_id}"
            )

            campaign_records.append(
                {
                    "campaign_id": campaign_id,
                    "channel": channel,
                    "campaign_type": "10.10",
                    "area_id": area_id,
                    "campaign_name": campaign_name,
                }
            )

            campaign_number += 1

    return pd.DataFrame(campaign_records)

def generate_experiment_assignment(area_characteristics):
    """Randomly assign areas to treatment/control with balanced baseline potential."""

    assignment_date = EXPERIMENT_START

    assignment_base = area_characteristics[
        [
            "area_id",
            "baseline_demand",
            "customer_volume_potential",
            "marketing_responsiveness",
        ]
    ].copy()

    # Calculate the baseline acquisition potential that
    # drives customer entry before experiment treatment.
    assignment_base["baseline_acquisition_potential"] = (
        assignment_base["baseline_demand"]
        * assignment_base["customer_volume_potential"]
        * assignment_base["marketing_responsiveness"]
    )

    area_ids = assignment_base["area_id"].to_numpy()

    potential_lookup = (
        assignment_base
        .set_index("area_id")[
            "baseline_acquisition_potential"
        ]
    )

    total_potential = (
        assignment_base["baseline_acquisition_potential"].sum()
    )

    best_treatment_ids = None
    best_difference = np.inf

    # Search many randomized 10-treatment / 10-control
    # assignments and retain the most balanced one.
    for _ in range(10_000):

        shuffled_area_ids = area_ids.copy()

        RANDOM_GENERATOR.shuffle(
            shuffled_area_ids
        )

        treatment_ids = shuffled_area_ids[:10]

        control_ids = shuffled_area_ids[10:]

        treatment_potential = (
            potential_lookup.loc[treatment_ids].sum()
        )

        control_potential = (
            potential_lookup.loc[control_ids].sum()
        )

        difference = abs(
            treatment_potential
            - control_potential
        )

        if difference < best_difference:
            best_difference = difference
            best_treatment_ids = treatment_ids.copy()

    treatment_ids = set(best_treatment_ids)

    assignment_records = []

    for area_id in area_ids:

        if area_id in treatment_ids:
            experiment_group = "treatment"
        else:
            experiment_group = "control"

        assignment_records.append(
            {
                "experiment_id": "EXP_001",
                "area_id": area_id,
                "experiment_group": experiment_group,
                "assignment_date": assignment_date,
            }
        )

    assignment_df = pd.DataFrame(
        assignment_records
    )

    # Validate exactly 10 areas per group.
    group_counts = (
        assignment_df["experiment_group"]
        .value_counts()
    )

    assert group_counts["treatment"] == 10
    assert group_counts["control"] == 10

    # Validate that all areas received exactly one assignment.
    assert (
        assignment_df["area_id"].nunique()
        == len(area_characteristics)
    )

    assert (
        assignment_df["area_id"].duplicated().sum()
        == 0
    )

    return assignment_df

def generate_marketing_performance(
    campaign_configuration,
    festival_campaign_configuration,
    area_characteristics,
    experiment_assignment,
):
    """Generate daily campaign spend, impressions, and clicks."""

    all_dates = pd.date_range(
        HISTORICAL_START,
        POST_EXPERIMENT_END,
        freq="D",
    )

    # --------------------------------------------------------
    # Lookups
    # --------------------------------------------------------

    assignment_lookup = (
        experiment_assignment
        .set_index("area_id")["experiment_group"]
        .to_dict()
    )

    area_weights = (
        area_characteristics["customer_volume_potential"]
        * area_characteristics["baseline_demand"]
    )

    area_weights = (
        area_weights / area_weights.sum()
    )

    area_weight_lookup = dict(
        zip(
            area_characteristics["area_id"],
            area_weights,
        )
    )

    # --------------------------------------------------------
    # Seasonal demand/spend pattern
    # --------------------------------------------------------

    month_multipliers = {
        1: 0.95,
        2: 0.90,
        3: 1.00,
        4: 0.98,
        5: 1.02,
        6: 1.00,
        7: 1.05,
        8: 1.00,
        9: 1.05,
        10: 1.20,
        11: 1.05,
        12: 1.10,
    }

    # --------------------------------------------------------
    # Combine normal and festival campaign configurations
    # --------------------------------------------------------

    normal_campaigns = campaign_configuration.copy()
    normal_campaigns["is_festival"] = False

    festival_campaigns = festival_campaign_configuration.copy()
    festival_campaigns["is_festival"] = True

    all_campaigns = pd.concat(
        [
            normal_campaigns,
            festival_campaigns,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Generate campaign-day performance
    # --------------------------------------------------------

    campaign_frames = []

    for _, campaign in all_campaigns.iterrows():

        area_id = campaign["area_id"]
        channel = campaign["channel"]
        campaign_type = campaign["campaign_type"]

        channel_params = CHANNEL_PARAMETERS[channel]
        area_share = area_weight_lookup[area_id]

        # ----------------------------------------------------
        # Determine campaign allocation
        # ----------------------------------------------------

        if not campaign["is_festival"]:

            campaign_weights = CAMPAIGN_TYPE_WEIGHTS[channel]

            total_campaign_weight = sum(
                campaign_weights.values()
            )

            campaign_type_share = (
                campaign_weights[campaign_type]
                / total_campaign_weight
            )

            spend_multiplier = 1.00

            ctr_multiplier = (
                CAMPAIGN_TYPE_MULTIPLIERS[
                    campaign_type
                ]["ctr"]
            )

        else:

            # Festival campaigns receive additional spend
            # on top of the normal channel allocation.
            campaign_type_share = 1.00
            spend_multiplier = 1.60
            ctr_multiplier = 1.15

        daily_records = []

        for simulation_date in all_dates:

            # Festival campaigns only run during October.
            if (
                campaign["is_festival"]
                and simulation_date.month != 10
            ):
                continue

            month_multiplier = month_multipliers[
                simulation_date.month
            ]

            # Random daily variation represents normal
            # campaign-level volatility.
            daily_noise = RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.12,
            )

            channel_daily_spend = (
                BASE_DAILY_SPEND
                * CHANNEL_MIX[channel]
            )

            # ------------------------------------------------
            # Baseline spend before experiment treatment
            # ------------------------------------------------

            baseline_spend = (
                channel_daily_spend
                * campaign_type_share
                * area_share
                * month_multiplier
                * daily_noise
                * spend_multiplier
            )

            # ------------------------------------------------
            # Experiment treatment
            # ------------------------------------------------

            is_treatment_period = (
                EXPERIMENT_START
                <= simulation_date
                <= EXPERIMENT_END
                and assignment_lookup[area_id]
                == "treatment"
            )

            if is_treatment_period:

                spend = (
                    baseline_spend
                    * TREATMENT_SPEND_MULTIPLIER
                )

                # Validate the treatment multiplier against
                # the exact baseline value used for this
                # campaign-day.
                assert np.isclose(
                    spend / baseline_spend,
                    TREATMENT_SPEND_MULTIPLIER,
                )

            else:

                spend = baseline_spend

            # ------------------------------------------------
            # Impressions and clicks
            # ------------------------------------------------

            impressions = (
                spend
                * channel_params[
                    "impressions_per_dollar"
                ]
                * RANDOM_GENERATOR.lognormal(
                    mean=0.0,
                    sigma=0.08,
                )
            )

            ctr = (
                channel_params["base_ctr"]
                * ctr_multiplier
            )

            clicks = RANDOM_GENERATOR.binomial(
                int(round(impressions)),
                min(ctr, 0.50),
            )

            daily_records.append(
                {
                    "date": simulation_date,
                    "channel": channel,
                    "campaign_id": campaign["campaign_id"],
                    "campaign_name": campaign["campaign_name"],
                    "spend": spend,
                    "impressions": impressions,
                    "clicks": clicks,
                }
            )

        campaign_frames.append(
            pd.DataFrame(daily_records)
        )

    return pd.concat(
        campaign_frames,
        ignore_index=True,
    )



def validate_marketing_performance(
    marketing_performance,
    experiment_assignment,
):
    """Validate the generated marketing performance data."""

    required_columns = [
        "date",
        "channel",
        "campaign_id",
        "campaign_name",
        "spend",
        "impressions",
        "clicks",
    ]

    assert list(marketing_performance.columns) == required_columns

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    assert marketing_performance["date"].min() == HISTORICAL_START
    assert marketing_performance["date"].max() == POST_EXPERIMENT_END

    assert marketing_performance["campaign_id"].notna().all()
    assert marketing_performance["channel"].notna().all()
    assert marketing_performance["campaign_name"].notna().all()

    assert (
        marketing_performance[
            ["campaign_id", "date"]
        ].duplicated().sum()
        == 0
    )

    assert marketing_performance["spend"].gt(0).all()
    assert marketing_performance["impressions"].gt(0).all()
    assert marketing_performance["clicks"].ge(0).all()

    assert (
        marketing_performance["clicks"]
        <= marketing_performance["impressions"].round()
    ).all()

    # --------------------------------------------------------
    # Channel validation
    # --------------------------------------------------------

    channels = set(
        marketing_performance["channel"].unique()
    )

    assert channels == set(ELIGIBLE_CHANNELS)

    # --------------------------------------------------------
    # Campaign validation
    # --------------------------------------------------------

    festival_mask = (
        marketing_performance["campaign_name"]
        .str.contains("_10.10_", regex=False)
    )

    normal_data = marketing_performance.loc[
        ~festival_mask
    ]

    festival_data = marketing_performance.loc[
        festival_mask
    ]

    # Festival campaigns should only run during October.
    assert (
        festival_data["date"].dt.month == 10
    ).all()

    # Normal campaigns should cover the full simulation period.
    assert (
        normal_data["date"].between(
            HISTORICAL_START,
            POST_EXPERIMENT_END,
        )
    ).all()

    # --------------------------------------------------------
    # Experiment period validation
    # --------------------------------------------------------

    experiment_mask = marketing_performance["date"].between(
        EXPERIMENT_START,
        EXPERIMENT_END,
    )

    experiment_data = marketing_performance.loc[
        experiment_mask
    ].copy()

    experiment_area_lookup = (
        experiment_assignment
        .set_index("area_id")["experiment_group"]
        .to_dict()
    )

    # Extract area_id from campaign_name.
    experiment_data["area_id"] = (
        experiment_data["campaign_name"]
        .str.extract(r"(AREA_\d{3})$")
    )

    experiment_data["experiment_group"] = (
        experiment_data["area_id"]
        .map(experiment_area_lookup)
    )

    # Every experiment-period campaign row must map
    # to a valid experiment group.
    assert experiment_data["area_id"].notna().all()

    assert (
        experiment_data["experiment_group"]
        .isin(["treatment", "control"])
    ).all()

    # Both experiment groups must be present.
    assert set(
        experiment_data["experiment_group"].unique()
    ) == {"treatment", "control"}

    # --------------------------------------------------------
    # Treatment uplift validation
    # --------------------------------------------------------
    #
    # The generator applies:
    #
    #     treatment spend = baseline spend × 1.25
    #
    # However, each campaign-day also contains independent
    # daily noise. Therefore, raw treatment/control spend
    # cannot be expected to have an exact 1.25 ratio.
    #
    # Areas also have different underlying spend weights.
    # We therefore normalize each area's experiment-period
    # spend by its area allocation weight before comparing
    # treatment and control.
    # --------------------------------------------------------

    area_weights = (
        area_characteristics["customer_volume_potential"]
        * area_characteristics["baseline_demand"]
    )

    area_weights = (
        area_weights / area_weights.sum()
    )

    area_weight_lookup = dict(
        zip(
            area_characteristics["area_id"],
            area_weights,
        )
    )

    area_spend = (
        experiment_data
        .groupby(
            ["area_id", "experiment_group"],
            as_index=False,
        )["spend"]
        .sum()
    )

    area_spend["area_weight"] = (
        area_spend["area_id"]
        .map(area_weight_lookup)
    )

    assert area_spend["area_weight"].notna().all()

    area_spend["normalized_spend"] = (
        area_spend["spend"]
        / area_spend["area_weight"]
    )

    treatment_normalized_spend = (
        area_spend.loc[
            area_spend["experiment_group"] == "treatment",
            "normalized_spend",
        ]
        .mean()
    )

    control_normalized_spend = (
        area_spend.loc[
            area_spend["experiment_group"] == "control",
            "normalized_spend",
        ]
        .mean()
    )

    treatment_control_ratio = (
        treatment_normalized_spend
        / control_normalized_spend
    )

    print("\nTreatment uplift validation:")
    print(
        f"Normalized treatment/control spend ratio: "
        f"{treatment_control_ratio:.4f}"
    )
    print(
        f"Expected treatment multiplier: "
        f"{TREATMENT_SPEND_MULTIPLIER:.2f}"
    )

    # Daily noise and finite simulation size mean that the
    # observed ratio will not be exactly 1.25.
    assert (
        1.15
        <= treatment_control_ratio
        <= 1.35
    )

    print(
        "Treatment uplift validation passed."
    )

    # --------------------------------------------------------
    # Channel mix validation outside treatment period
    # --------------------------------------------------------

    pre_experiment_data = marketing_performance.loc[
        marketing_performance["date"] < EXPERIMENT_START
    ]

    pre_channel_spend = (
        pre_experiment_data
        .groupby("channel")["spend"]
        .sum()
    )

    pre_channel_share = (
        pre_channel_spend
        / pre_channel_spend.sum()
    )

    print("\nPre-experiment channel spend share:")
    print(pre_channel_share)

    for channel, expected_share in CHANNEL_MIX.items():

        observed_share = pre_channel_share[channel]

        assert abs(
            observed_share - expected_share
        ) < 0.03

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nMarketing performance validation passed.")

def generate_customers(area_characteristics, marketing_performance):
    """Generate customers and their hidden behavioral characteristics."""

    marketing = marketing_performance.copy()

    marketing["date"] = pd.to_datetime(
        marketing["date"]
    )

    marketing["area_id"] = marketing["campaign_name"].str.extract(
        r"(AREA_\d{3})"
    )

    area_day_marketing = (
        marketing
        .groupby(["date", "area_id"], as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            spend=("spend", "sum"),
        )
    )

    area_day_marketing = area_day_marketing.merge(
        area_characteristics[
            [
                "area_id",
                "baseline_demand",
                "customer_volume_potential",
                "marketing_responsiveness",
            ]
        ],
        on="area_id",
        how="left",
    )

    # Baseline daily customer acquisition rate.
    # This represents underlying customer demand before
    # marketing response is applied.
    area_day_marketing["baseline_acquisition_rate"] = (
        area_day_marketing["baseline_demand"]
        * area_day_marketing["customer_volume_potential"]
        * 4.5
    )

    # Marketing increases customer acquisition with
    # diminishing returns as clicks increase.
    area_day_marketing["marketing_response"] = (
        1
        + 0.08
        * np.log1p(
            area_day_marketing["clicks"]
        )
        * area_day_marketing["marketing_responsiveness"]
    )

    # Expected customer entries for each area-day.
    area_day_marketing["expected_customers"] = (
        area_day_marketing["baseline_acquisition_rate"]
        * area_day_marketing["marketing_response"]
    )

    # Realized customer entries follow a Poisson process.
    area_day_marketing["new_customers"] = (
        RANDOM_GENERATOR.poisson(
            area_day_marketing["expected_customers"]
        )
    )

    customer_area_ids = []
    customer_entry_dates = []

    for row in area_day_marketing.itertuples(index=False):
        if row.new_customers > 0:
            customer_area_ids.extend(
                [row.area_id] * row.new_customers
            )
            customer_entry_dates.extend(
                [row.date] * row.new_customers
            )

    customer_count = len(customer_area_ids)

    customer_ids = [
        f"CUST_{i:06d}"
        for i in range(1, customer_count + 1)
    ]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "area_id": customer_area_ids,
        }
    )

    entry_dates = pd.Series(
        customer_entry_dates,
        dtype="datetime64[ns]",
    )

    hidden_customer_characteristics = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "entry_date": entry_dates,
            "purchase_propensity": RANDOM_GENERATOR.beta(
                a=2.5,
                b=35,
                size=customer_count,
            ),
            "aov_tendency": RANDOM_GENERATOR.lognormal(
                mean=np.log(60),
                sigma=0.35,
                size=customer_count,
            ),
            "repeat_purchase_tendency": RANDOM_GENERATOR.beta(
                a=2,
                b=5,
                size=customer_count,
            ),
            "price_sensitivity": RANDOM_GENERATOR.beta(
                a=2.5,
                b=4,
                size=customer_count,
            ),
        }
    )

    channel_responsiveness = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "Google": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=customer_count,
            ),
            "Meta": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=customer_count,
            ),
            "TikTok": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=customer_count,
            ),
            "CRM": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=customer_count,
            ),
        }
    )

    return (
        customers,
        hidden_customer_characteristics,
        channel_responsiveness,
    )

def generate_transactions(
    customers,
    hidden_customer_characteristics,
    channel_responsiveness,
    area_characteristics,
    marketing_performance,
):
    """Generate probabilistic customer transactions."""

    # --------------------------------------------------------
    # Prepare customer-level simulation arrays
    # --------------------------------------------------------

    customer_data = (
        customers
        .merge(
            hidden_customer_characteristics,
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            channel_responsiveness,
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
    )

    assert len(customer_data) == len(customers)

    customer_ids = customer_data["customer_id"].to_numpy()
    customer_area_ids = customer_data["area_id"].to_numpy()

    customer_entry_dates = (
        customer_data["entry_date"]
        .to_numpy()
        .astype("datetime64[ns]")
    )

    customer_purchase_propensity = (
        customer_data["purchase_propensity"].to_numpy()
    )

    customer_aov_tendency = (
        customer_data["aov_tendency"].to_numpy()
    )

    customer_repeat_tendency = (
        customer_data["repeat_purchase_tendency"].to_numpy()
    )

    customer_price_sensitivity = (
        customer_data["price_sensitivity"].to_numpy()
    )

    customer_google_response = (
        customer_data["Google"].to_numpy()
    )

    customer_meta_response = (
        customer_data["Meta"].to_numpy()
    )

    customer_tiktok_response = (
        customer_data["TikTok"].to_numpy()
    )

    customer_crm_response = (
        customer_data["CRM"].to_numpy()
    )

    # --------------------------------------------------------
    # Area-level simulation parameters
    # --------------------------------------------------------

    area_lookup = (
        area_characteristics
        .set_index("area_id")
        .to_dict("index")
    )

    customer_area_purchase_propensity = np.array(
        [
            area_lookup[area_id]["purchase_propensity"]
            for area_id in customer_area_ids
        ]
    )

    customer_area_aov_tendency = np.array(
        [
            area_lookup[area_id]["aov_tendency"]
            for area_id in customer_area_ids
        ]
    )

    customer_area_seasonality = np.array(
        [
            area_lookup[area_id]["seasonality_sensitivity"]
            for area_id in customer_area_ids
        ]
    )

    customer_area_marketing_response = np.array(
        [
            area_lookup[area_id]["marketing_responsiveness"]
            for area_id in customer_area_ids
        ]
    )

    # --------------------------------------------------------
    # Aggregate marketing activity to area × date
    # --------------------------------------------------------

    marketing_by_area = marketing_performance.copy()

    marketing_by_area["area_id"] = (
        marketing_by_area["campaign_name"]
        .str.extract(r"(AREA_\d{3})$")
    )

    assert marketing_by_area["area_id"].notna().all()

    daily_area_clicks = (
        marketing_by_area
        .groupby(
            ["date", "area_id", "channel"],
            as_index=False,
        )["clicks"]
        .sum()
    )

    daily_area_clicks = (
        daily_area_clicks
        .pivot_table(
            index=["date", "area_id"],
            columns="channel",
            values="clicks",
            fill_value=0,
        )
        .reset_index()
    )

    for channel in ELIGIBLE_CHANNELS:
        if channel not in daily_area_clicks.columns:
            daily_area_clicks[channel] = 0

    # --------------------------------------------------------
    # Convert area-level clicks into customer-level
    # marketing pressure
    # --------------------------------------------------------

    area_customer_counts = (
        customers
        .groupby("area_id")["customer_id"]
        .count()
        .to_dict()
    )

    daily_area_clicks["customer_count"] = (
        daily_area_clicks["area_id"]
        .map(area_customer_counts)
    )

    assert daily_area_clicks["customer_count"].notna().all()

    for channel in ELIGIBLE_CHANNELS:
        daily_area_clicks[f"{channel}_pressure"] = (
            daily_area_clicks[channel]
            / daily_area_clicks["customer_count"]
        )

    # --------------------------------------------------------
    # Simulation state
    # --------------------------------------------------------

    # Number of completed purchases for each customer.
    purchase_count = np.zeros(
        len(customer_data),
        dtype=np.int16,
    )

    # Date of the most recent completed purchase.
    last_purchase_date = np.full(
        len(customer_data),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )

    # Earliest date when a customer can attempt another
    # purchase after completing a previous purchase.
    next_purchase_date = customer_entry_dates.copy()

    transaction_records = []

    transaction_number = 1

    # --------------------------------------------------------
    # Daily purchase simulation
    # --------------------------------------------------------

    all_dates = pd.date_range(
        HISTORICAL_START,
        POST_EXPERIMENT_END,
        freq="D",
    )

    # Purchase probability is calibrated around a low daily
    # probability because customers have many opportunities
    # across the observation period.
    month_multipliers = {
        1: 0.95,
        2: 0.90,
        3: 1.00,
        4: 0.98,
        5: 1.02,
        6: 1.00,
        7: 1.05,
        8: 1.00,
        9: 1.05,
        10: 1.20,
        11: 1.05,
        12: 1.10,
    }

    for simulation_date in all_dates:

        simulation_date_np = np.datetime64(
            simulation_date,
            "ns",
        )

        # ----------------------------------------------------
        # Customer lifecycle eligibility
        # ----------------------------------------------------

        active_mask = (
            customer_entry_dates
            <= simulation_date_np
        )

        purchase_opportunity_mask = (
            next_purchase_date
            <= simulation_date_np
        )

        eligible_mask = (
            active_mask
            & purchase_opportunity_mask
        )

        eligible_indices = np.flatnonzero(
            eligible_mask
        )

        if len(eligible_indices) == 0:
            continue

        # ----------------------------------------------------
        # Daily area marketing pressure
        # ----------------------------------------------------

        daily_marketing = daily_area_clicks.loc[
            daily_area_clicks["date"] == simulation_date
        ]

        daily_marketing_lookup = (
            daily_marketing
            .set_index("area_id")
            .to_dict("index")
        )

        customer_eligible_area_ids = (
            customer_area_ids[eligible_indices]
        )

        google_pressure = np.array(
            [
                daily_marketing_lookup[area_id][
                    "Google_pressure"
                ]
                for area_id in customer_eligible_area_ids
            ]
        )

        meta_pressure = np.array(
            [
                daily_marketing_lookup[area_id][
                    "Meta_pressure"
                ]
                for area_id in customer_eligible_area_ids
            ]
        )

        tiktok_pressure = np.array(
            [
                daily_marketing_lookup[area_id][
                    "TikTok_pressure"
                ]
                for area_id in customer_eligible_area_ids
            ]
        )

        crm_pressure = np.array(
            [
                daily_marketing_lookup[area_id][
                    "CRM_pressure"
                ]
                for area_id in customer_eligible_area_ids
            ]
        )

        # ----------------------------------------------------
        # Customer-specific marketing response
        # ----------------------------------------------------

        marketing_response = (
            google_pressure
            * customer_google_response[
                eligible_indices
            ]
            + meta_pressure
            * customer_meta_response[
                eligible_indices
            ]
            + tiktok_pressure
            * customer_tiktok_response[
                eligible_indices
            ]
            + crm_pressure
            * customer_crm_response[
                eligible_indices
            ]
        )

        # Diminishing returns from increasingly high
        # marketing pressure.
        marketing_effect = (
            1.0
            + 0.18
            * np.log1p(
                marketing_response / 0.03
            )
            * customer_area_marketing_response[
                eligible_indices
            ]
        )

        # ----------------------------------------------------
        # Seasonality
        # ----------------------------------------------------

        seasonality = (
            month_multipliers[simulation_date.month]
            ** customer_area_seasonality[
                eligible_indices
            ]
        )

        # ----------------------------------------------------
        # First purchase vs repeat purchase behavior
        # ----------------------------------------------------

        previous_purchase = (
            purchase_count[eligible_indices] > 0
        )

        repeat_multiplier = np.where(
            previous_purchase,
            (
                1.0
                + 1.25
                * customer_repeat_tendency[
                    eligible_indices
                ]
            ),
            1.0,
        )

        # ----------------------------------------------------
        # 10.10 Travel Festival demand effect
        # ----------------------------------------------------

        festival_multiplier = (
            1.0
            if simulation_date.month != 10
            else 1.35
        )

        # ----------------------------------------------------
        # Purchase probability
        # ----------------------------------------------------

        base_probability = (
            0.00075
            * customer_purchase_propensity[
                eligible_indices
            ]
            / customer_purchase_propensity.mean()
        )

        purchase_probability = (
            base_probability
            * customer_area_purchase_propensity[
                eligible_indices
            ]
            / customer_area_purchase_propensity.mean()
            * seasonality
            * marketing_effect
            * repeat_multiplier
            * festival_multiplier
        )

        purchase_probability = np.clip(
            purchase_probability,
            0.0,
            0.05,
        )

        purchase_events = (
            RANDOM_GENERATOR.random(
                len(eligible_indices)
            )
            < purchase_probability
        )

        purchase_indices = eligible_indices[
            purchase_events
        ]

        if len(purchase_indices) == 0:
            continue

        # ----------------------------------------------------
        # Generate transaction economics
        # ----------------------------------------------------

        purchase_aov = (
            customer_aov_tendency[
                purchase_indices
            ]
            * customer_area_aov_tendency[
                purchase_indices
            ]
            / customer_area_aov_tendency.mean()
            * RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.18,
                size=len(purchase_indices),
            )
        )

        # Festival promotions create stronger discounting
        # and subsidy.
        if simulation_date.month == 10:

            discount_rate = (
                RANDOM_GENERATOR.beta(
                    a=2.5,
                    b=18,
                    size=len(purchase_indices),
                )
                + 0.025
            )

            subsidy_rate = (
                RANDOM_GENERATOR.beta(
                    a=2.2,
                    b=16,
                    size=len(purchase_indices),
                )
                + 0.035
            )

        else:

            discount_rate = (
                RANDOM_GENERATOR.beta(
                    a=2.0,
                    b=25,
                    size=len(purchase_indices),
                )
            )

            subsidy_rate = (
                RANDOM_GENERATOR.beta(
                    a=2.0,
                    b=22,
                    size=len(purchase_indices),
                )
            )

        # More price-sensitive customers receive somewhat
        # higher promotional cost.
        discount_rate *= (
            0.75
            + 0.75
            * customer_price_sensitivity[
                purchase_indices
            ]
        )

        subsidy_rate *= (
            0.75
            + 0.75
            * customer_price_sensitivity[
                purchase_indices
            ]
        )

        discount_rate = np.clip(
            discount_rate,
            0.0,
            0.60,
        )

        subsidy_rate = np.clip(
            subsidy_rate,
            0.0,
            0.60,
        )

        discount = (
            purchase_aov
            * discount_rate
        )

        subsidy = (
            purchase_aov
            * subsidy_rate
        )

        # ----------------------------------------------------
        # Cancellation
        # ----------------------------------------------------

        cancelled = (
            RANDOM_GENERATOR.random(
                len(purchase_indices)
            )
            < (
                0.025
                + 0.025
                * customer_price_sensitivity[
                    purchase_indices
                ]
            )
        )

        revenue = np.where(
            cancelled,
            0.0,
            purchase_aov,
        )

        subsidy = np.where(
            cancelled,
            0.0,
            subsidy,
        )

        discount = np.where(
            cancelled,
            0.0,
            discount,
        )

        # ----------------------------------------------------
        # Store transaction records and update lifecycle
        # ----------------------------------------------------

        for index, customer_index in enumerate(
            purchase_indices
        ):

            transaction_records.append(
                {
                    "transaction_id": (
                        f"TXN_{transaction_number:08d}"
                    ),
                    "customer_id": (
                        customer_ids[customer_index]
                    ),
                    "transaction_date": simulation_date,
                    "revenue": float(
                        revenue[index]
                    ),
                    "subsidy": float(
                        subsidy[index]
                    ),
                    "discount": float(
                        discount[index]
                    ),
                    "cancelled": bool(
                        cancelled[index]
                    ),
                }
            )

            # A completed purchase changes the customer's
            # lifecycle state.
            if not cancelled[index]:

                purchase_count[customer_index] += 1

                last_purchase_date[
                    customer_index
                ] = simulation_date_np

                # Customers with stronger repeat tendency
                # generally return sooner, while retaining
                # substantial natural variation.
                repeat_tendency = (
                    customer_repeat_tendency[
                        customer_index
                    ]
                )

                mean_repeat_gap = (
                    120
                    - 90 * repeat_tendency
                )

                repeat_gap = max(
                    14,
                    int(
                        RANDOM_GENERATOR.gamma(
                            shape=4.0,
                            scale=(
                                mean_repeat_gap / 4.0
                            ),
                        )
                    ),
                )

                next_purchase_date[
                    customer_index
                ] = (
                    simulation_date_np
                    + np.timedelta64(
                        repeat_gap,
                        "D",
                    )
                )

            else:

                # A cancelled booking does not create a
                # completed-purchase state. Give the customer
                # another opportunity after a short interval.
                next_purchase_date[
                    customer_index
                ] = (
                    simulation_date_np
                    + np.timedelta64(
                        7,
                        "D",
                    )
                )

            transaction_number += 1

    # --------------------------------------------------------
    # Build final transaction table
    # --------------------------------------------------------

    transactions = pd.DataFrame(
        transaction_records,
        columns=[
            "transaction_id",
            "customer_id",
            "transaction_date",
            "revenue",
            "subsidy",
            "discount",
            "cancelled",
        ],
    )

    return transactions

def generate_marketing_touchpoints(
    customers,
    hidden_customer_characteristics,
    channel_responsiveness,
    marketing_performance,
):
    """Generate customer-level marketing click touchpoints (vectorized)."""

    TOUCHPOINT_CAPTURE_RATE = 0.02

    customer_data = (
        customers
        .merge(
            hidden_customer_characteristics[["customer_id", "entry_date"]],
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            channel_responsiveness,
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
    )

    assert len(customer_data) == len(customers)
    assert customer_data["entry_date"].notna().all()

    customer_data["entry_date"] = pd.to_datetime(customer_data["entry_date"])

    marketing_data = marketing_performance.reset_index(drop=True)

    area_ids = marketing_data["campaign_name"].str.extract(
        r"(AREA_\d{3})$", expand=False
    )
    assert area_ids.notna().all()

    # --------------------------------------------------------
    # 1. Captured clicks per campaign-day, all rows at once
    # --------------------------------------------------------
    captured = RANDOM_GENERATOR.binomial(
        marketing_data["clicks"].to_numpy().astype(np.int64),
        TOUCHPOINT_CAPTURE_RATE,
    )

    # One entry per touchpoint, pointing back to its campaign-day row
    row_pos = np.repeat(np.arange(len(marketing_data)), captured)
    n_touchpoints = len(row_pos)

    tp_dates = marketing_data["date"].to_numpy()[row_pos]
    tp_areas = area_ids.to_numpy()[row_pos]
    tp_channels = marketing_data["channel"].to_numpy()[row_pos]

    # --------------------------------------------------------
    # 2. Sample customers per (area, channel)
    # --------------------------------------------------------
    tp_customer = np.empty(n_touchpoints, dtype=object)
    valid = np.zeros(n_touchpoints, dtype=bool)

    group_keys = pd.DataFrame({"area": tp_areas, "channel": tp_channels})

    for (area_id, channel), positions in (
        group_keys.groupby(["area", "channel"]).indices.items()
    ):
        group = (
            customer_data[customer_data["area_id"] == area_id]
            .sort_values("entry_date")
        )

        entry = group["entry_date"].to_numpy()
        ids = group["customer_id"].to_numpy()
        cum_weights = np.cumsum(group[channel].to_numpy(dtype=float))

        # Number of customers already active on each touchpoint's date
        active_n = np.searchsorted(entry, tp_dates[positions], side="right")

        ok = active_n > 0
        pos = positions[ok]
        k = active_n[ok]

        # Weighted draw restricted to the first k (active) customers
        u = RANDOM_GENERATOR.random(len(pos)) * cum_weights[k - 1]
        idx = np.minimum(
            np.searchsorted(cum_weights, u, side="right"),
            k - 1,
        )

        tp_customer[pos] = ids[idx]
        valid[pos] = True

    # --------------------------------------------------------
    # 3. Timestamps and final table
    # --------------------------------------------------------
    seconds_in_day = RANDOM_GENERATOR.integers(0, 86_400, size=n_touchpoints)
    timestamps = tp_dates + seconds_in_day.astype("timedelta64[s]")

    marketing_touchpoints = pd.DataFrame(
        {
            "customer_id": tp_customer,
            "touchpoint_timestamp": timestamps,
            "channel": tp_channels,
            "campaign_id": marketing_data["campaign_id"].to_numpy()[row_pos],
            "campaign_name": marketing_data["campaign_name"].to_numpy()[row_pos],
        }
    )

    marketing_touchpoints = (
        marketing_touchpoints[valid]
        .sort_values("touchpoint_timestamp")
        .reset_index(drop=True)
    )

    marketing_touchpoints.insert(
        0,
        "touchpoint_id",
        [f"TP_{i:09d}" for i in range(1, len(marketing_touchpoints) + 1)],
    )

    return marketing_touchpoints

def validate_day_1(
    area_characteristics,
    experiment_assignment,
    customers,
    hidden_customer_characteristics,
    marketing_performance,
    transactions,
):
    
    print("\n" + "=" * 70)
    print("DAY 1 - CUSTOMER LIFECYCLE")
    print("=" * 70) 
    # ---------------------------------------------------------
    # 2. Validate experiment assignment
    # ---------------------------------------------------------

    assert len(experiment_assignment) == N_AREAS

    assert (
        experiment_assignment["area_id"].nunique()
        == N_AREAS
    )

    group_counts = (
        experiment_assignment["experiment_group"]
        .value_counts()
    )

    assert (
        group_counts["treatment"]
        == N_TREATMENT_AREAS
    )

    assert (
        group_counts["control"]
        == N_CONTROL_AREAS
    )

    assert (
        experiment_assignment["assignment_date"]
        .eq(EXPERIMENT_START)
        .all()
    )

    # ---------------------------------------------------------
    # 3. Customer acquisition summary
    # ---------------------------------------------------------

    entry_dates = (
        hidden_customer_characteristics["entry_date"]
    )

    historical = (
        entry_dates <= pd.Timestamp("2026-06-30")
    ).sum()

    experiment = (
        (
            entry_dates >= pd.Timestamp("2026-07-01")
        )
        & (
            entry_dates <= pd.Timestamp("2026-07-28")
        )
    ).sum()

    post = (
        entry_dates >= pd.Timestamp("2026-07-29")
    ).sum()

    customer_summary = (
        hidden_customer_characteristics[
            ["customer_id", "entry_date"]
        ]
        .merge(
            customers[
                ["customer_id", "area_id"]
            ],
            on="customer_id",
            how="left",
        )
        .merge(
            experiment_assignment[
                ["area_id", "experiment_group"]
            ],
            on="area_id",
            how="left",
        )
    )

    customer_summary["period"] = np.select(
        [
            customer_summary["entry_date"]
            <= pd.Timestamp("2026-06-30"),
            (
                customer_summary["entry_date"]
                <= pd.Timestamp("2026-07-28")
            ),
        ],
        [
            "pre_experiment",
            "experiment",
        ],
        default="post_experiment",
    )

    # ---------------------------------------------------------
    # 4. Pre-experiment acquisition trend
    # ---------------------------------------------------------

    pre_period = customer_summary[
        customer_summary["period"] == "pre_experiment"
    ].copy()

    pre_period["week"] = (
        pre_period["entry_date"]
        .dt.to_period("W")
        .dt.start_time
    )

    weekly_acquisition = (
        pre_period
        .groupby(["week", "experiment_group"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nPre-experiment weekly customer acquisition:")
    print(weekly_acquisition)

    # ---------------------------------------------------------
    # 4. Baseline acquisition balance
    # ---------------------------------------------------------

    assignment_check = (
        area_characteristics[
            [
                "area_id",
                "baseline_demand",
                "customer_volume_potential",
                "marketing_responsiveness",
            ]
        ]
        .copy()
    )

    assignment_check["baseline_acquisition_potential"] = (
        assignment_check["baseline_demand"]
        * assignment_check["customer_volume_potential"]
        * assignment_check["marketing_responsiveness"]
    )

    assignment_check = assignment_check.merge(
        experiment_assignment[
            ["area_id", "experiment_group"]
        ],
        on="area_id",
        how="left",
    )

    baseline_potential = (
        assignment_check
        .groupby("experiment_group")[
            "baseline_acquisition_potential"
        ]
        .sum()
    )

    # ---------------------------------------------------------
    # 5. Compact validation summary
    # ---------------------------------------------------------

    print("\n=== Synthetic Data Generation Summary ===")

    print("\nExperiment assignment:")
    print(group_counts)

    print("\nBaseline acquisition potential:")
    print(baseline_potential)

    print("\nCustomer count:")
    print(len(customers))

    print("\nCustomer entry date range:")
    print(
        entry_dates.min(),
        "to",
        entry_dates.max(),
    )

    print("\nCustomers acquired by period:")
    print(
        pd.Series(
            {
                "Historical": historical,
                "Experiment": experiment,
                "Post": post,
            }
        )
    )

    print("\nCustomer acquisition by experiment group:")
    print(
        customer_summary[
            customer_summary["period"] == "experiment"
        ]
        .groupby("experiment_group")
        .size()
    )

    print("\nCustomer acquisition by group and period:")
    print(
        customer_summary
        .groupby(
            ["period", "experiment_group"]
        )
        .size()
    )

    print("\nMarketing performance:")
    print(
        f"{len(marketing_performance):,} rows | "
        f"{marketing_performance['date'].min().date()} "
        f"to "
        f"{marketing_performance['date'].max().date()}"
    )

    print("\nTransactions:")
    print(f"{len(transactions):,} rows")

    print("\nMarketing touchpoints:")
    print(f"{len(marketing_touchpoints):,} rows")

    # --------------------------------------------------------
    # Customer lifecycle validation
    # --------------------------------------------------------

    customer_count = len(customers)

    transaction_count = len(transactions)

    completed_transactions = transactions.loc[
        ~transactions["cancelled"]
    ].copy()

    purchase_counts = (
        completed_transactions
        .groupby("customer_id")
        .size()
    )

    customer_purchase_counts = (
        customers[["customer_id"]]
        .merge(
            purchase_counts.rename("purchase_count"),
            on="customer_id",
            how="left",
        )
        .fillna({"purchase_count": 0})
    )

    customers_with_0 = (
        customer_purchase_counts["purchase_count"] == 0
    ).sum()

    customers_with_1 = (
        customer_purchase_counts["purchase_count"] == 1
    ).sum()

    customers_with_2 = (
        customer_purchase_counts["purchase_count"] == 2
    ).sum()

    customers_with_3_plus = (
        customer_purchase_counts["purchase_count"] >= 3
    ).sum()

    cancelled_count = (
        transactions["cancelled"]
        .sum()
    )

    cancelled_rate = (
        cancelled_count / transaction_count
        if transaction_count > 0
        else 0
    )

    first_purchase_count = (
        completed_transactions
        .sort_values(
            ["customer_id", "transaction_date"]
        )
        .groupby("customer_id")
        .head(1)
        .shape[0]
    )

    repeat_purchase_count = (
        len(completed_transactions)
        - first_purchase_count
    )

    # --------------------------------------------------------
    # Purchase gap analysis
    # --------------------------------------------------------

    completed_transactions_sorted = (
        completed_transactions
        .sort_values(
            ["customer_id", "transaction_date"]
        )
    )

    completed_transactions_sorted["purchase_gap_days"] = (
        completed_transactions_sorted
        .groupby("customer_id")["transaction_date"]
        .diff()
        .dt.days
    )

    purchase_gaps = (
        completed_transactions_sorted[
            "purchase_gap_days"
        ]
        .dropna()
    )

    # --------------------------------------------------------
    # Print lifecycle summary
    # --------------------------------------------------------

    print("\nCustomer Lifecycle Summary")
    print("-" * 40)

    print(f"Customer count: {customer_count:,}")
    print(f"Transaction count: {transaction_count:,}")

    print(
        f"Customers with 0 purchases: "
        f"{customers_with_0:,}"
    )

    print(
        f"Customers with 1 purchase: "
        f"{customers_with_1:,}"
    )

    print(
        f"Customers with 2 purchases: "
        f"{customers_with_2:,}"
    )

    print(
        f"Customers with 3+ purchases: "
        f"{customers_with_3_plus:,}"
    )

    print(
        f"Cancelled transactions: "
        f"{cancelled_count:,} "
        f"({cancelled_rate:.2%})"
    )

    print(
        f"First purchase count: "
        f"{first_purchase_count:,}"
    )

    print(
        f"Repeat purchase count: "
        f"{repeat_purchase_count:,}"
    )

    if len(purchase_gaps) > 0:

        print(
            f"Minimum purchase gap: "
            f"{purchase_gaps.min():.0f} days"
        )

        print(
            f"Median purchase gap: "
            f"{purchase_gaps.median():.0f} days"
        )

        print(
            f"75th percentile purchase gap: "
            f"{purchase_gaps.quantile(0.75):.0f} days"
        )

        print(
            f"Maximum purchase gap: "
            f"{purchase_gaps.max():.0f} days"
        )

    else:

        print("No repeat purchase gaps available.")

    # --------------------------------------------------------
    # Repeat tendency validation
    # --------------------------------------------------------

    customer_purchase_summary = (
        customers[["customer_id"]]
        .merge(
            hidden_customer_characteristics[
                [
                    "customer_id",
                    "repeat_purchase_tendency",
                ]
            ],
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            completed_transactions
            .groupby("customer_id")
            .size()
            .rename("purchase_count"),
            on="customer_id",
            how="left",
        )
        .fillna({"purchase_count": 0})
    )

    customer_purchase_summary["is_repeat_customer"] = (
        customer_purchase_summary["purchase_count"] >= 2
    )

    print("\nRepeat Tendency Validation")
    print("-" * 40)

    print(
        customer_purchase_summary
        .groupby(
            pd.qcut(
                customer_purchase_summary[
                    "repeat_purchase_tendency"
                ],
                5,
                duplicates="drop",
            )
        )
        .agg(
            customers=("customer_id", "count"),
            avg_purchases=("purchase_count", "mean"),
            repeat_customer_rate=(
                "is_repeat_customer",
                "mean",
            ),
        )
    )

    # --------------------------------------------------------
    # Repeat timing validation
    # --------------------------------------------------------

    repeat_transactions = (
        completed_transactions
        .sort_values(
            ["customer_id", "transaction_date"]
        )
        .copy()
    )

    repeat_transactions["purchase_number"] = (
        repeat_transactions
        .groupby("customer_id")
        .cumcount()
        + 1
    )

    repeat_transactions = repeat_transactions[
        repeat_transactions["purchase_number"] >= 2
    ].copy()

    repeat_transactions["purchase_gap_days"] = (
        repeat_transactions
        .groupby("customer_id")["transaction_date"]
        .diff()
        .dt.days
    )

    repeat_transactions = repeat_transactions[
        repeat_transactions["purchase_gap_days"].notna()
    ].copy()

    repeat_timing_validation = (
        repeat_transactions[
            [
                "customer_id",
                "purchase_gap_days",
            ]
        ]
        .merge(
            hidden_customer_characteristics[
                [
                    "customer_id",
                    "repeat_purchase_tendency",
                ]
            ],
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
    )

    print("\nRepeat Timing Validation")
    print("-" * 40)

    print(
        repeat_timing_validation
        .groupby(
            pd.qcut(
                repeat_timing_validation[
                    "repeat_purchase_tendency"
                ],
                5,
                duplicates="drop",
            )
        )
        .agg(
            repeat_purchases=(
                "customer_id",
                "count",
            ),
            median_gap_days=(
                "purchase_gap_days",
                "median",
            ),
            avg_gap_days=(
                "purchase_gap_days",
                "mean",
            ),
        )
    )

    # --------------------------------------------------------
    # Customer state consistency validation
    # --------------------------------------------------------

    print("\nCustomer State Consistency Validation")
    print("-" * 40)

    transaction_check = (
        transactions
        .merge(
            customers,
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            hidden_customer_characteristics[
                [
                    "customer_id",
                    "entry_date",
                ]
            ],
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # Check 1: Every transaction belongs to a valid customer
    # --------------------------------------------------------

    missing_customers = (
        transaction_check["area_id"].isna().sum()
    )

    assert missing_customers == 0, (
        "Transactions contain unknown customer IDs."
    )

    print(
        f"Transactions with valid customer IDs: "
        f"{len(transaction_check):,}"
    )

    # --------------------------------------------------------
    # Check 2: No transaction occurs before customer entry
    # --------------------------------------------------------

    transaction_before_entry = (
        transaction_check["transaction_date"]
        < transaction_check["entry_date"]
    )

    before_entry_count = transaction_before_entry.sum()

    assert before_entry_count == 0, (
        "Transactions occur before customer entry date."
    )

    print(
        f"Transactions before customer entry: "
        f"{before_entry_count:,}"
    )

    # --------------------------------------------------------
    # Check 3: Transaction dates are within simulation period
    # --------------------------------------------------------

    invalid_transaction_dates = (
        (transaction_check["transaction_date"] < HISTORICAL_START)
        | (
            transaction_check["transaction_date"]
            > POST_EXPERIMENT_END
        )
    )

    invalid_date_count = invalid_transaction_dates.sum()

    assert invalid_date_count == 0, (
        "Transactions fall outside the simulation period."
    )

    print(
        f"Transactions outside simulation period: "
        f"{invalid_date_count:,}"
    )

    # --------------------------------------------------------
    # Check 4: Completed purchases per customer
    # --------------------------------------------------------

    completed_transactions_check = (
        transaction_check[
            ~transaction_check["cancelled"]
        ]
        .sort_values(
            ["customer_id", "transaction_date"]
        )
        .copy()
    )

    completed_transactions_check["purchase_number"] = (
        completed_transactions_check
        .groupby("customer_id")
        .cumcount()
        + 1
    )

    first_purchase_count_check = (
        (
            completed_transactions_check[
                "purchase_number"
            ]
            == 1
        )
        .sum()
    )

    repeat_purchase_count_check = (
        (
            completed_transactions_check[
                "purchase_number"
            ]
            >= 2
        )
        .sum()
    )

    print(
        f"Completed first purchases: "
        f"{first_purchase_count_check:,}"
    )

    print(
        f"Completed repeat purchases: "
        f"{repeat_purchase_count_check:,}"
    )

    # --------------------------------------------------------
    # Check 5: Repeat purchases must have a previous purchase
    # --------------------------------------------------------

    repeat_without_previous = (
        completed_transactions_check[
            "purchase_number"
        ]
        < 2
    ).sum()

    # This should simply equal the number of first purchases.
    assert (
        first_purchase_count_check
        == completed_transactions_check[
            "customer_id"
        ].nunique()
    )

    print(
        f"Customers with completed purchases: "
        f"{completed_transactions_check['customer_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Check 6: Completed purchase dates must increase
    # --------------------------------------------------------

    purchase_date_gaps = (
        completed_transactions_check
        .groupby("customer_id")["transaction_date"]
        .diff()
        .dt.days
        .dropna()
    )

    invalid_purchase_order = (
        purchase_date_gaps <= 0
    ).sum()

    assert invalid_purchase_order == 0, (
        "A customer has multiple completed purchases "
        "on the same day or in reverse chronological order."
    )

    print(
        f"Invalid purchase ordering: "
        f"{invalid_purchase_order:,}"
    )

    # --------------------------------------------------------
    # Check 7: Minimum repeat gap
    # --------------------------------------------------------

    if len(purchase_date_gaps) > 0:

        minimum_repeat_gap = (
            purchase_date_gaps.min()
        )

        print(
            f"Minimum completed-purchase gap: "
            f"{minimum_repeat_gap:.0f} days"
        )

        assert minimum_repeat_gap >= 14, (
            "Repeat purchases occur too soon after "
            "the previous purchase."
        )

    # --------------------------------------------------------
    # Final lifecycle consistency check
    # --------------------------------------------------------

    print(
        "\nAll customer lifecycle consistency checks passed."
    )

def validate_day_2(
    customers,
    hidden_customer_characteristics,
    transactions,
):
    # ============================================================
    # DAY 2 - CUSTOMER ECONOMICS AUDIT
    # ============================================================


    economic_audit = transactions.copy()

    economic_audit["contribution_margin"] = (
        economic_audit["revenue"]
        - economic_audit["subsidy"]
        - economic_audit["discount"]
    )

    economic_audit["discount_rate"] = (
        economic_audit["discount"] / economic_audit["revenue"]
    )

    economic_audit["subsidy_rate"] = (
        economic_audit["subsidy"] / economic_audit["revenue"]
    )

    completed = economic_audit[~economic_audit["cancelled"]].copy()

    print("\n" + "=" * 70)
    print("DAY 2 - CUSTOMER ECONOMICS AUDIT")
    print("=" * 70)

    print("\nTRANSACTION VOLUME")
    print(f"Total transactions: {len(economic_audit):,}")
    print(f"Completed transactions: {len(completed):,}")
    print(f"Cancelled transactions: {economic_audit['cancelled'].sum():,}")
    print(
        f"Cancellation rate: "
        f"{economic_audit['cancelled'].mean():.2%}"
    )

    print("\nREVENUE")
    print(f"Mean AOV: ${completed['revenue'].mean():,.2f}")
    print(f"Median AOV: ${completed['revenue'].median():,.2f}")
    print(f"P25 AOV: ${completed['revenue'].quantile(0.25):,.2f}")
    print(f"P75 AOV: ${completed['revenue'].quantile(0.75):,.2f}")
    print(f"P95 AOV: ${completed['revenue'].quantile(0.95):,.2f}")

    print("\nDISCOUNT")
    print(
        f"Mean discount rate: "
        f"{completed['discount_rate'].mean():.2%}"
    )
    print(
        f"Median discount rate: "
        f"{completed['discount_rate'].median():.2%}"
    )
    print(
        f"Mean discount per booking: "
        f"${completed['discount'].mean():,.2f}"
    )

    print("\nSUBSIDY")
    print(
        f"Mean subsidy rate: "
        f"{completed['subsidy_rate'].mean():.2%}"
    )
    print(
        f"Median subsidy rate: "
        f"{completed['subsidy_rate'].median():.2%}"
    )
    print(
        f"Mean subsidy per booking: "
        f"${completed['subsidy'].mean():,.2f}"
    )

    print("\nCONTRIBUTION MARGIN")
    print(
        f"Mean CM per booking: "
        f"${completed['contribution_margin'].mean():,.2f}"
    )
    print(
        f"Median CM per booking: "
        f"${completed['contribution_margin'].median():,.2f}"
    )
    print(
        f"CM margin: "
        f"{completed['contribution_margin'].sum() / completed['revenue'].sum():.2%}"
    )
    print(
        f"Negative CM bookings: "
        f"{(completed['contribution_margin'] < 0).mean():.2%}"
    )

    print("\nCUSTOMER ECONOMICS")
    customer_economics = (
        completed
        .groupby("customer_id")
        .agg(
            purchases=("transaction_id", "count"),
            revenue=("revenue", "sum"),
            subsidy=("subsidy", "sum"),
            discount=("discount", "sum"),
            contribution_margin=("contribution_margin", "sum"),
        )
    )

    print(
        f"Customers with completed purchases: "
        f"{len(customer_economics):,}"
    )
    print(
        f"Mean customer revenue: "
        f"${customer_economics['revenue'].mean():,.2f}"
    )
    print(
        f"Median customer revenue: "
        f"${customer_economics['revenue'].median():,.2f}"
    )
    print(
        f"Mean customer CM: "
        f"${customer_economics['contribution_margin'].mean():,.2f}"
    )
    print(
        f"Median customer CM: "
        f"${customer_economics['contribution_margin'].median():,.2f}"
    )

    print("\nFIRST VS REPEAT PURCHASES")

    purchase_sequence = (
        completed
        .sort_values(["customer_id", "transaction_date"])
        .copy()
    )

    purchase_sequence["purchase_number"] = (
        purchase_sequence
        .groupby("customer_id")
        .cumcount() + 1
    )

    first_purchase = purchase_sequence[
        purchase_sequence["purchase_number"] == 1
    ]

    repeat_purchase = purchase_sequence[
        purchase_sequence["purchase_number"] > 1
    ]

    print(
        f"First-purchase AOV: "
        f"${first_purchase['revenue'].mean():,.2f}"
    )
    print(
        f"Repeat-purchase AOV: "
        f"${repeat_purchase['revenue'].mean():,.2f}"
    )

    print(
        f"First-purchase CM: "
        f"${first_purchase['contribution_margin'].mean():,.2f}"
    )
    print(
        f"Repeat-purchase CM: "
        f"${repeat_purchase['contribution_margin'].mean():,.2f}"
    )

    print("\nECONOMIC BEHAVIOR VALIDATION")

    # ------------------------------------------------------------
    # Customer value by purchase frequency
    # ------------------------------------------------------------

    customer_economics["purchase_group"] = pd.cut(
        customer_economics["purchases"],
        bins=[0, 1, 2, float("inf")],
        labels=["1 purchase", "2 purchases", "3+ purchases"],
    )

    customer_value_by_frequency = (
        customer_economics
        .groupby("purchase_group", observed=False)
        .agg(
            customers=("purchases", "size"),
            mean_revenue=("revenue", "mean"),
            mean_contribution_margin=(
                "contribution_margin",
                "mean",
            ),
        )
    )

    print("\nCUSTOMER VALUE BY PURCHASE FREQUENCY")
    print(customer_value_by_frequency.to_string())

    # ------------------------------------------------------------
    # Area-level economic variation
    # ------------------------------------------------------------

    area_economics = (
        completed
        .groupby("customer_id")
        .agg(
            area_id=("customer_id", lambda x: customers.loc[
                customers["customer_id"].eq(x.iloc[0]),
                "area_id"
            ].iloc[0]),
        )
        .reset_index()
    )

    area_economics = (
        completed
        .merge(
            customers[["customer_id", "area_id"]],
            on="customer_id",
            how="left",
        )
        .groupby("area_id")
        .agg(
            transactions=("transaction_id", "count"),
            aov=("revenue", "mean"),
            contribution_margin=("contribution_margin", "mean"),
        )
    )

    print("\nAREA-LEVEL ECONOMIC VARIATION")
    print(
        f"AOV range: "
        f"${area_economics['aov'].min():,.2f} - "
        f"${area_economics['aov'].max():,.2f}"
    )
    print(
        f"Mean area AOV: "
        f"${area_economics['aov'].mean():,.2f}"
    )
    print(
        f"Area AOV standard deviation: "
        f"${area_economics['aov'].std():,.2f}"
    )

    # ------------------------------------------------------------
    # Promotional cost behavior
    # ------------------------------------------------------------

    print("\nPROMOTIONAL COST VALIDATION")
    print(
        f"Discount rate range: "
        f"{completed['discount_rate'].min():.2%} - "
        f"{completed['discount_rate'].max():.2%}"
    )
    print(
        f"Subsidy rate range: "
        f"{completed['subsidy_rate'].min():.2%} - "
        f"{completed['subsidy_rate'].max():.2%}"
    )

    print(
        f"Discount as % of revenue: "
        f"{completed['discount'].sum() / completed['revenue'].sum():.2%}"
    )
    print(
        f"Subsidy as % of revenue: "
        f"{completed['subsidy'].sum() / completed['revenue'].sum():.2%}"
    )

    # ------------------------------------------------------------
    # Cancellation economics
    # ------------------------------------------------------------

    cancelled = economic_audit[
        economic_audit["cancelled"]
    ].copy()

    print("\nCANCELLATION ECONOMICS")
    print(
        f"Cancelled revenue: "
        f"${cancelled['revenue'].sum():,.2f}"
    )
    print(
        f"Cancelled discount: "
        f"${cancelled['discount'].sum():,.2f}"
    )
    print(
        f"Cancelled subsidy: "
        f"${cancelled['subsidy'].sum():,.2f}"
    )
    print(
        f"Cancelled CM: "
        f"${cancelled['contribution_margin'].sum():,.2f}"
    )

    print("\nPERSISTENT CHARACTERISTICS VS REALIZED BEHAVIOR")

    customer_behavior = (
        customers[["customer_id"]]
        .merge(
            hidden_customer_characteristics,
            on="customer_id",
            how="left",
        )
        .merge(
            customer_economics[
                [
                    "purchases",
                    "revenue",
                    "contribution_margin",
                ]
            ],
            left_on="customer_id",
            right_index=True,
            how="left",
        )
    )

    customer_behavior["purchases"] = (
        customer_behavior["purchases"]
        .fillna(0)
    )

    customer_behavior["revenue"] = (
        customer_behavior["revenue"]
        .fillna(0)
    )

    customer_behavior["contribution_margin"] = (
        customer_behavior["contribution_margin"]
        .fillna(0)
    )

    customer_behavior["realized_aov"] = np.where(
        customer_behavior["purchases"] > 0,
        customer_behavior["revenue"]
        / customer_behavior["purchases"],
        np.nan,
    )

    print("\nAOV TENDENCY VS REALIZED AOV")

    aov_validation = customer_behavior[
        customer_behavior["purchases"] > 0
    ]

    print(
        f"Correlation: "
        f"{aov_validation['aov_tendency'].corr(
            aov_validation['realized_aov']
        ):.3f}"
    )

    print("\nPRICE SENSITIVITY VS CUSTOMER CM")

    print(
        f"Correlation: "
        f"{customer_behavior['price_sensitivity'].corr(
            customer_behavior['contribution_margin']
        ):.3f}"
    )

    print("\nREPEAT PURCHASE TENDENCY VS PURCHASE FREQUENCY")

    print(
        f"Correlation: "
        f"{customer_behavior['repeat_purchase_tendency'].corr(
            customer_behavior['purchases']
        ):.3f}"
    )

    print("\nPURCHASE PROPENSITY VS PURCHASE FREQUENCY")

    print(
        f"Correlation: "
        f"{customer_behavior['purchase_propensity'].corr(
            customer_behavior['purchases']
        ):.3f}"
    )

    print("\nREPEAT BEHAVIOR DIAGNOSTIC")

    # --------------------------------------------------------
    # Identify first and repeat purchases
    # --------------------------------------------------------

    purchase_sequence = (
        completed
        .sort_values(["customer_id", "transaction_date"])
        .copy()
    )

    purchase_sequence["purchase_number"] = (
        purchase_sequence
        .groupby("customer_id")
        .cumcount()
        + 1
    )

    # --------------------------------------------------------
    # 1. Does repeat tendency affect becoming a repeat customer?
    # --------------------------------------------------------

    customer_purchase_counts = (
        purchase_sequence
        .groupby("customer_id")["purchase_number"]
        .max()
    )

    repeat_customer_flag = (
        customer_purchase_counts > 1
    ).rename("is_repeat_customer")

    repeat_customer_test = (
        hidden_customer_characteristics[
            ["customer_id", "repeat_purchase_tendency"]
        ]
        .merge(
            repeat_customer_flag,
            on="customer_id",
            how="left",
        )
    )

    repeat_customer_test["is_repeat_customer"] = (
        repeat_customer_test["is_repeat_customer"]
        .fillna(False)
    )

    repeat_customer_test["tendency_quartile"] = pd.qcut(
        repeat_customer_test["repeat_purchase_tendency"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    repeat_rate_by_quartile = (
        repeat_customer_test
        .groupby(
            "tendency_quartile",
            observed=False,
        )
        .agg(
            customers=("customer_id", "size"),
            repeat_customer_rate=(
                "is_repeat_customer",
                "mean",
            ),
        )
    )

    print("\nREPEAT CUSTOMER RATE BY TENDENCY QUARTILE")
    print(
        repeat_rate_by_quartile.to_string()
    )

    # --------------------------------------------------------
    # 2. Among repeat customers, does tendency affect
    #    number of repeat purchases?
    # --------------------------------------------------------

    repeat_customer_ids = repeat_customer_flag[
        repeat_customer_flag
    ].index

    repeat_frequency_test = (
        customer_economics
        .loc[
            customer_economics.index.isin(
                repeat_customer_ids
            )
        ]
        .reset_index()
        .merge(
            hidden_customer_characteristics[
                [
                    "customer_id",
                    "repeat_purchase_tendency",
                ]
            ],
            on="customer_id",
            how="left",
        )
    )

    repeat_frequency_test["repeat_purchases"] = (
        repeat_frequency_test["purchases"] - 1
    )

    repeat_frequency_correlation = (
        repeat_frequency_test[
            "repeat_purchase_tendency"
        ].corr(
            repeat_frequency_test[
                "repeat_purchases"
            ]
        )
    )

    print("\nREPEAT PURCHASES VS TENDENCY")
    print(
        f"Correlation: "
        f"{repeat_frequency_correlation:.3f}"
    )

    # --------------------------------------------------------
    # 3. Does tendency affect repeat purchase timing?
    # --------------------------------------------------------

    repeat_gap_data = (
        purchase_sequence
        .groupby("customer_id")
        .agg(
            first_purchase_date=(
                "transaction_date",
                "min",
            ),
            last_purchase_date=(
                "transaction_date",
                "max",
            ),
            purchases=(
                "purchase_number",
                "max",
            ),
        )
        .reset_index()
    )

    repeat_gap_data = repeat_gap_data[
        repeat_gap_data["purchases"] > 1
    ].copy()

    repeat_gap_data = repeat_gap_data.merge(
        hidden_customer_characteristics[
            [
                "customer_id",
                "repeat_purchase_tendency",
            ]
        ],
        on="customer_id",
        how="left",
    )

    repeat_gap_data["days_from_first_to_last"] = (
        repeat_gap_data["last_purchase_date"]
        - repeat_gap_data["first_purchase_date"]
    ).dt.days

    repeat_gap_correlation = (
        repeat_gap_data[
            "repeat_purchase_tendency"
        ].corr(
            repeat_gap_data[
                "days_from_first_to_last"
            ]
        )
    )

    print(
        "\nREPEAT TENDENCY VS DAYS BETWEEN PURCHASES"
    )

    print(
        f"Correlation: "
        f"{repeat_gap_correlation:.3f}"
    )

    print("\nREPEAT PURCHASE GAP VALIDATION")

    repeat_gap_sequence = (
        purchase_sequence[
            purchase_sequence["purchase_number"] > 1
        ]
        .copy()
    )

    repeat_gap_sequence["previous_purchase_date"] = (
        repeat_gap_sequence
        .groupby("customer_id")["transaction_date"]
        .shift(1)
    )

    repeat_gap_sequence["repeat_gap_days"] = (
        repeat_gap_sequence["transaction_date"]
        - repeat_gap_sequence["previous_purchase_date"]
    ).dt.days

    repeat_gap_sequence = repeat_gap_sequence.merge(
        hidden_customer_characteristics[
            [
                "customer_id",
                "repeat_purchase_tendency",
            ]
        ],
        on="customer_id",
        how="left",
    )

    repeat_gap_correlation = (
        repeat_gap_sequence[
            "repeat_purchase_tendency"
        ].corr(
            repeat_gap_sequence[
                "repeat_gap_days"
            ]
        )
    )

    print(
        f"Repeat tendency vs actual repeat gap: "
        f"{repeat_gap_correlation:.3f}"
    )

    print("\nAVERAGE REPEAT GAP BY TENDENCY QUARTILE")

    repeat_gap_sequence["tendency_quartile"] = pd.qcut(
        repeat_gap_sequence["repeat_purchase_tendency"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    repeat_gap_by_quartile = (
        repeat_gap_sequence
        .groupby(
            "tendency_quartile",
            observed=False,
        )
        .agg(
            repeat_purchases=("customer_id", "size"),
            mean_repeat_gap_days=(
                "repeat_gap_days",
                "mean",
            ),
            median_repeat_gap_days=(
                "repeat_gap_days",
                "median",
            ),
        )
    )

    print(
        repeat_gap_by_quartile.to_string()
    )

    print("\nPRICE SENSITIVITY VALIDATION")

    price_sensitivity_test = (
        completed
        .merge(
            hidden_customer_characteristics[
                [
                    "customer_id",
                    "price_sensitivity",
                ]
            ],
            on="customer_id",
            how="left",
        )
    )

    price_sensitivity_test["discount_rate"] = (
        price_sensitivity_test["discount"]
        / price_sensitivity_test["revenue"]
    )

    price_sensitivity_test["subsidy_rate"] = (
        price_sensitivity_test["subsidy"]
        / price_sensitivity_test["revenue"]
    )

    discount_correlation = (
        price_sensitivity_test[
            "price_sensitivity"
        ].corr(
            price_sensitivity_test[
                "discount_rate"
            ]
        )
    )

    subsidy_correlation = (
        price_sensitivity_test[
            "price_sensitivity"
        ].corr(
            price_sensitivity_test[
                "subsidy_rate"
            ]
        )
    )

    print(
        f"Price sensitivity vs discount rate: "
        f"{discount_correlation:.3f}"
    )

    print(
        f"Price sensitivity vs subsidy rate: "
        f"{subsidy_correlation:.3f}"
    )

    print("\n" + "=" * 70)

    print("\nAll current validations passed.")

def validate_day_3(
    area_characteristics,
    experiment_assignment,
    customers,
    hidden_customer_characteristics,
    channel_responsiveness,
    marketing_performance,
    transactions,
):
    # ============================================================
    # DAY 3 - MARKETING RESPONSE SYSTEM
    # ============================================================

    # ------------------------------------------------------------
    # 1. Marketing pressure diagnostic
    # ------------------------------------------------------------

    marketing_by_area = marketing_performance.copy()

    marketing_by_area["area_id"] = (
        marketing_by_area["campaign_name"]
        .str.extract(r"(AREA_\d{3})$")
    )

    assert marketing_by_area["area_id"].notna().all()

    daily_area_clicks = (
        marketing_by_area
        .groupby(
            ["date", "area_id", "channel"],
            as_index=False,
        )["clicks"]
        .sum()
    )

    daily_area_clicks = (
        daily_area_clicks
        .pivot_table(
            index=["date", "area_id"],
            columns="channel",
            values="clicks",
            fill_value=0,
        )
        .reset_index()
    )

    for channel in ELIGIBLE_CHANNELS:
        if channel not in daily_area_clicks.columns:
            daily_area_clicks[channel] = 0

    area_customer_counts = (
        customers
        .groupby("area_id")["customer_id"]
        .count()
        .to_dict()
    )

    daily_area_clicks["customer_count"] = (
        daily_area_clicks["area_id"]
        .map(area_customer_counts)
    )

    assert daily_area_clicks["customer_count"].notna().all()
    assert (daily_area_clicks["customer_count"] > 0).all()

    for channel in ELIGIBLE_CHANNELS:
        daily_area_clicks[f"{channel}_pressure"] = (
            daily_area_clicks[channel]
            / daily_area_clicks["customer_count"]
        )

    # ------------------------------------------------------------
    # Pressure integrity
    # ------------------------------------------------------------

    pressure_columns = [
        f"{channel}_pressure"
        for channel in ELIGIBLE_CHANNELS
    ]

    assert daily_area_clicks[pressure_columns].notna().all().all()
    assert (
        daily_area_clicks[pressure_columns] >= 0
    ).all().all()

    print("\n" + "=" * 60)
    print("DAY 3 - MARKETING RESPONSE SYSTEM")
    print("=" * 60)

    print("\n1. Marketing pressure")
    print("-" * 60)

    print("\nPressure summary by channel:")

    pressure_summary = (
        daily_area_clicks[pressure_columns]
        .describe()
        .T[
            ["mean", "50%", "std", "min", "max"]
        ]
    )

    pressure_summary = pressure_summary.rename(
        columns={"50%": "median"}
    )

    print(
        pressure_summary.round(4)
    )

    # ------------------------------------------------------------
    # Clicks -> pressure relationship
    # ------------------------------------------------------------

    click_pressure_checks = []

    for channel in ELIGIBLE_CHANNELS:

        correlation = (
            daily_area_clicks[
                [channel, f"{channel}_pressure"]
            ]
            .corr()
            .loc[channel, f"{channel}_pressure"]
        )

        click_pressure_checks.append(
            {
                "channel": channel,
                "clicks_pressure_correlation": correlation,
            }
        )

    click_pressure_summary = pd.DataFrame(
        click_pressure_checks
    )

    print("\nClicks -> pressure correlation:")

    print(
        click_pressure_summary.round(4)
    )

    # ------------------------------------------------------------
    # Experiment-period treatment vs control pressure
    # ------------------------------------------------------------

    experiment_pressure = daily_area_clicks[
        daily_area_clicks["date"].between(
            EXPERIMENT_START,
            EXPERIMENT_END,
        )
    ].merge(
        experiment_assignment[
            ["area_id", "experiment_group"]
        ],
        on="area_id",
        how="left",
        validate="many_to_one",
    )

    assert experiment_pressure[
        "experiment_group"
    ].notna().all()

    pressure_by_group = (
        experiment_pressure
        .groupby("experiment_group")[
            pressure_columns
        ]
        .mean()
    )

    print("\nExperiment-period pressure by group:")

    print(
        pressure_by_group.round(4)
    )

    for channel in ELIGIBLE_CHANNELS:

        treatment_pressure = pressure_by_group.loc[
            "treatment",
            f"{channel}_pressure",
        ]

        control_pressure = pressure_by_group.loc[
            "control",
            f"{channel}_pressure",
        ]

        ratio = (
            treatment_pressure
            / control_pressure
        )

        print(
            f"\n{channel}: "
            f"Treatment / Control pressure ratio = "
            f"{ratio:.4f}"
        )

        assert treatment_pressure > control_pressure

    print(
        "\nMarketing pressure validation passed."
    )

    # ------------------------------------------------------------
    # 2. Channel responsiveness
    # ------------------------------------------------------------

    print("\n2. Channel responsiveness")
    print("-" * 60)

    # Use experiment-period average pressure by area.
    # This avoids creating a large customer x day table while
    # preserving the same pressure mechanism used by
    # generate_transactions().

    experiment_area_pressure = (
        experiment_pressure
        .groupby("area_id")[pressure_columns]
        .mean()
        .reset_index()
    )

    # Attach each customer's hidden channel responsiveness
    # and their area's experiment-period average pressure.
    customer_response_data = (
        customers[
            ["customer_id", "area_id"]
        ]
        .merge(
            channel_responsiveness[
                ["customer_id"] + ELIGIBLE_CHANNELS
            ],
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            experiment_area_pressure,
            on="area_id",
            how="left",
            validate="many_to_one",
        )
    )

    assert len(customer_response_data) == len(customers)

    assert (
        customer_response_data[ELIGIBLE_CHANNELS]
        .notna()
        .all()
        .all()
    )

    assert (
        customer_response_data[ELIGIBLE_CHANNELS] >= 0
    ).all().all()

    assert (
        customer_response_data[pressure_columns]
        .notna()
        .all()
        .all()
    )

    # ------------------------------------------------------------
    # Channel-specific modeled response
    # ------------------------------------------------------------

    responsiveness_checks = []

    for channel in ELIGIBLE_CHANNELS:

        response_column = f"{channel}_response"

        pressure_column = f"{channel}_pressure"

        customer_response_data[response_column] = (
            customer_response_data[pressure_column]
            * customer_response_data[channel]
        )

        # Divide customers into responsiveness quintiles.
        customer_response_data[
            f"{channel}_responsiveness_quintile"
        ] = pd.qcut(
            customer_response_data[channel],
            q=5,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
            duplicates="drop",
        )

        quintile_summary = (
            customer_response_data
            .groupby(
                f"{channel}_responsiveness_quintile",
                observed=True,
            )[response_column]
            .mean()
        )

        print(f"\n{channel} responsiveness:")

        print(
            quintile_summary.round(4)
        )

        responsiveness_checks.append(
            {
                "channel": channel,
                "Q1": quintile_summary.iloc[0],
                "Q5": quintile_summary.iloc[-1],
            }
        )

        # Higher channel responsiveness must produce
        # higher modeled channel response.
        assert (
            quintile_summary.iloc[-1]
            > quintile_summary.iloc[0]
        )

    responsiveness_summary = pd.DataFrame(
        responsiveness_checks
    )

    print("\nResponsiveness summary:")

    print(
        responsiveness_summary.round(4)
    )

    print(
        "\nChannel responsiveness validation passed."
    )

    # ------------------------------------------------------------
    # 3. Diminishing returns
    # ------------------------------------------------------------

    print("\n3. Diminishing returns")
    print("-" * 60)

    # The transaction generator applies a logarithmic response curve:
    #
    # marketing_effect =
    #     1.0
    #     + 0.18
    #     * log1p(marketing_response / 0.03)
    #     * customer_area_marketing_response
    #
    # Validate this mechanism directly using the same formula.
    # A representative area-level marketing responsiveness is used
    # so that the validation isolates the diminishing-return curve.

    representative_marketing_responsiveness = (
        area_characteristics["marketing_responsiveness"].mean()
    )

    response_grid = np.linspace(
        0.001,
        2.0,
        100,
    )

    effect_grid = (
        1.0
        + 0.18
        * np.log1p(
            response_grid / 0.03
        )
        * representative_marketing_responsiveness
    )

    marginal_effect = np.diff(effect_grid)

    # The effect must increase as marketing response increases.
    assert np.all(
        np.diff(effect_grid) > 0
    )

    # The incremental gain must generally decline.
    # Because the formula is deterministic, we expect a fully
    # monotonic decline in the marginal increments.
    assert np.all(
        np.diff(marginal_effect) < 0
    )

    # ------------------------------------------------------------
    # Response quantile summary
    # ------------------------------------------------------------

    response_quantiles = np.quantile(
        response_grid,
        [0.20, 0.40, 0.60, 0.80, 1.00],
    )

    effect_quantiles = (
        1.0
        + 0.18
        * np.log1p(
            response_quantiles / 0.03
        )
        * representative_marketing_responsiveness
    )

    diminishing_returns_summary = pd.DataFrame(
        {
            "response": response_quantiles,
            "marketing_effect": effect_quantiles,
        }
    )

    diminishing_returns_summary[
        "incremental_effect"
    ] = (
        diminishing_returns_summary[
            "marketing_effect"
        ].diff()
    )

    print(
        "\nMarketing response -> marketing effect:"
    )

    print(
        diminishing_returns_summary.round(4)
    )

    # The first row has no preceding point, so exclude it
    # when checking incremental effect.
    incremental_effect = (
        diminishing_returns_summary[
            "incremental_effect"
        ].dropna()
    )

    assert (
        incremental_effect > 0
    ).all()

    assert (
        incremental_effect.diff().dropna() < 0
    ).all()

    print(
        "\nDiminishing returns validation passed."
    )

    # ------------------------------------------------------------
    # 4. Treatment response
    # ------------------------------------------------------------

    print("\n4. Treatment response")
    print("-" * 60)

    # Map each customer to their experiment group through area_id.
    customer_experiment = (
        customers[
            ["customer_id", "area_id"]
        ]
        .merge(
            experiment_assignment[
                ["area_id", "experiment_group"]
            ],
            on="area_id",
            how="left",
            validate="many_to_one",
        )
    )

    assert len(customer_experiment) == len(customers)

    assert (
        customer_experiment["experiment_group"]
        .notna()
        .all()
    )

    # Restrict transactions to the 28-day experiment period.
    experiment_transactions = transactions[
        transactions["transaction_date"].between(
            EXPERIMENT_START,
            EXPERIMENT_END,
        )
    ].copy()

    # Attach experiment group to each transaction.
    experiment_transactions = (
        experiment_transactions
        .merge(
            customer_experiment[
                ["customer_id", "experiment_group"]
            ],
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
    )

    assert (
        experiment_transactions["experiment_group"]
        .notna()
        .all()
    )

    # Contribution margin is derived from the raw transaction
    # economics. It is intentionally not stored in transactions.
    experiment_transactions["contribution_margin"] = (
        experiment_transactions["revenue"]
        - experiment_transactions["subsidy"]
        - experiment_transactions["discount"]
    )

    experiment_transactions["completed"] = (
        ~experiment_transactions["cancelled"]
    )

    # ------------------------------------------------------------
    # Customer-level experiment outcomes
    # ------------------------------------------------------------

    experiment_customers = (
        customer_experiment[
            customer_experiment["experiment_group"].isin(
                ["treatment", "control"]
            )
        ]
        .copy()
    )

    customer_outcomes = (
        experiment_transactions
        .groupby(
            "customer_id",
            as_index=False,
        )
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            completed_transactions=(
                "completed",
                "sum",
            ),
            contribution_margin=(
                "contribution_margin",
                "sum",
            ),
        )
    )

    experiment_customers = (
        experiment_customers
        .merge(
            customer_outcomes,
            on="customer_id",
            how="left",
        )
    )

    for column in [
        "transactions",
        "completed_transactions",
        "contribution_margin",
    ]:
        experiment_customers[column] = (
            experiment_customers[column]
            .fillna(0)
        )

    # ------------------------------------------------------------
    # Treatment vs control outcomes
    # ------------------------------------------------------------

    treatment_response_summary = (
        experiment_customers
        .groupby("experiment_group")
        .agg(
            customers=("customer_id", "count"),
            transactions=("transactions", "sum"),
            completed_transactions=(
                "completed_transactions",
                "sum",
            ),
            contribution_margin=(
                "contribution_margin",
                "sum",
            ),
        )
    )

    treatment_response_summary[
        "transactions_per_customer"
    ] = (
        treatment_response_summary["transactions"]
        / treatment_response_summary["customers"]
    )

    treatment_response_summary[
        "completed_transactions_per_customer"
    ] = (
        treatment_response_summary[
            "completed_transactions"
        ]
        / treatment_response_summary["customers"]
    )

    treatment_response_summary[
        "contribution_margin_per_customer"
    ] = (
        treatment_response_summary[
            "contribution_margin"
        ]
        / treatment_response_summary["customers"]
    )

    print("\nExperiment-period outcomes by group:")

    print(
        treatment_response_summary.round(4)
    )

    # ------------------------------------------------------------
    # Directional treatment response
    # ------------------------------------------------------------

    treatment = treatment_response_summary.loc[
        "treatment"
    ]

    control = treatment_response_summary.loc[
        "control"
    ]

    transaction_rate_ratio = (
        treatment["transactions_per_customer"]
        / control["transactions_per_customer"]
    )

    completed_rate_ratio = (
        treatment[
            "completed_transactions_per_customer"
        ]
        / control[
            "completed_transactions_per_customer"
        ]
    )

    contribution_margin_ratio = (
        treatment[
            "contribution_margin_per_customer"
        ]
        / control[
            "contribution_margin_per_customer"
        ]
    )

    print(
        "\nTreatment / Control ratios:"
    )

    print(
        f"Transactions per customer: "
        f"{transaction_rate_ratio:.4f}"
    )

    print(
        f"Completed transactions per customer: "
        f"{completed_rate_ratio:.4f}"
    )

    print(
        f"Contribution margin per customer: "
        f"{contribution_margin_ratio:.4f}"
    )

    # Treatment should generate stronger observed outcomes
    # because it receives higher marketing pressure.
    assert (
        treatment["transactions_per_customer"]
        > control["transactions_per_customer"]
    )

    assert (
        treatment[
            "completed_transactions_per_customer"
        ]
        > control[
            "completed_transactions_per_customer"
        ]
    )

    print(
        "\nContribution margin per customer is reported "
        "as an economic outcome diagnostic, not a directional "
        "mechanism assertion."
    )

    print(
        "\nTreatment response validation passed."
    )

     # ------------------------------------------------------------
    # 5. Treatment response diagnostic
    # ------------------------------------------------------------

    print("\n5. Treatment response diagnostic")
    print("-" * 60)

    # Entry date is stored in hidden_customer_characteristics,
    # consistent with the customer_data construction used by
    # generate_transactions().

    customer_entry_dates = (
        customers[
            ["customer_id"]
        ]
        .merge(
            hidden_customer_characteristics[
                ["customer_id", "entry_date"]
            ],
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
    )

    customer_entry_dates["entry_date"] = pd.to_datetime(
        customer_entry_dates["entry_date"]
    )

    pre_experiment_customers = (
        customer_experiment
        .merge(
            customer_entry_dates,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    assert (
        pre_experiment_customers["entry_date"]
        .notna()
        .all()
    )

    # Keep only customers who were already active when
    # the experiment began.
    pre_experiment_customers = (
        pre_experiment_customers[
            pre_experiment_customers["entry_date"]
            < EXPERIMENT_START
        ]
        .copy()
    )

    assert len(pre_experiment_customers) > 0

    # Count experiment-period outcomes for these customers.
    pre_experiment_outcomes = (
        experiment_transactions[
            experiment_transactions["customer_id"].isin(
                pre_experiment_customers["customer_id"]
            )
        ]
        .groupby("customer_id", as_index=False)
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            completed_transactions=(
                "completed",
                "sum",
            ),
            contribution_margin=(
                "contribution_margin",
                "sum",
            ),
        )
    )

    pre_experiment_customers = (
        pre_experiment_customers
        .merge(
            pre_experiment_outcomes,
            on="customer_id",
            how="left",
        )
    )

    for column in [
        "transactions",
        "completed_transactions",
        "contribution_margin",
    ]:
        pre_experiment_customers[column] = (
            pre_experiment_customers[column]
            .fillna(0)
        )

    # ------------------------------------------------------------
    # Group-level diagnostic
    # ------------------------------------------------------------

    pre_experiment_summary = (
        pre_experiment_customers
        .groupby("experiment_group")
        .agg(
            customers=("customer_id", "count"),
            transactions=("transactions", "sum"),
            completed_transactions=(
                "completed_transactions",
                "sum",
            ),
            contribution_margin=(
                "contribution_margin",
                "sum",
            ),
        )
    )

    pre_experiment_summary[
        "transactions_per_customer"
    ] = (
        pre_experiment_summary["transactions"]
        / pre_experiment_summary["customers"]
    )

    pre_experiment_summary[
        "completed_transactions_per_customer"
    ] = (
        pre_experiment_summary[
            "completed_transactions"
        ]
        / pre_experiment_summary["customers"]
    )

    pre_experiment_summary[
        "contribution_margin_per_customer"
    ] = (
        pre_experiment_summary[
            "contribution_margin"
        ]
        / pre_experiment_summary["customers"]
    )

    print(
        "\nExperiment-period outcomes among "
        "pre-existing customers:"
    )

    print(
        pre_experiment_summary.round(4)
    )

    treatment = pre_experiment_summary.loc[
        "treatment"
    ]

    control = pre_experiment_summary.loc[
        "control"
    ]

    print(
        "\nTreatment / Control ratios:"
    )

    print(
        f"Transactions per customer: "
        f"{(
            treatment["transactions_per_customer"]
            / control["transactions_per_customer"]
        ):.4f}"
    )

    print(
        f"Completed transactions per customer: "
        f"{(
            treatment["completed_transactions_per_customer"]
            / control["completed_transactions_per_customer"]
        ):.4f}"
    )

    print(
        f"Contribution margin per customer: "
        f"{(
            treatment["contribution_margin_per_customer"]
            / control["contribution_margin_per_customer"]
        ):.4f}"
    )

    print(
        "\nPre-existing customer treatment diagnostic completed."
    )

    # ------------------------------------------------------------
    # 6. Purchase probability diagnostic
    # ------------------------------------------------------------

    print("\n6. Purchase probability diagnostic")
    print("-" * 60)

    # Recreate the same customer-level structure used by
    # generate_transactions().
    customer_probability_data = (
        customers
        .merge(
            hidden_customer_characteristics,
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            channel_responsiveness,
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
    )

    assert len(customer_probability_data) == len(customers)

    # Area-level characteristics.
    area_lookup = (
        area_characteristics
        .set_index("area_id")
        .to_dict("index")
    )

    customer_area_purchase_propensity = np.array([
        area_lookup[area_id]["purchase_propensity"]
        for area_id in customer_probability_data["area_id"]
    ])

    customer_area_seasonality = np.array([
        area_lookup[area_id]["seasonality_sensitivity"]
        for area_id in customer_probability_data["area_id"]
    ])

    customer_area_marketing_response = np.array([
        area_lookup[area_id]["marketing_responsiveness"]
        for area_id in customer_probability_data["area_id"]
    ])

    # ------------------------------------------------------------
    # Use experiment-period average pressure by area
    # ------------------------------------------------------------

    experiment_area_pressure = (
        experiment_pressure
        .groupby("area_id")[pressure_columns]
        .mean()
        .reset_index()
    )

    customer_probability_data = (
        customer_probability_data
        .merge(
            experiment_area_pressure,
            on="area_id",
            how="left",
            validate="many_to_one",
        )
    )

    assert (
        customer_probability_data[pressure_columns]
        .notna()
        .all()
        .all()
    )

    # ------------------------------------------------------------
    # Reconstruct marketing response
    # ------------------------------------------------------------

    marketing_response = (
        customer_probability_data["Google_pressure"]
        * customer_probability_data["Google"]
        + customer_probability_data["Meta_pressure"]
        * customer_probability_data["Meta"]
        + customer_probability_data["TikTok_pressure"]
        * customer_probability_data["TikTok"]
        + customer_probability_data["CRM_pressure"]
        * customer_probability_data["CRM"]
    )

    # Exact marketing-effect formula from
    # generate_transactions().
    marketing_effect = (
        1.0
        + 0.18
        * np.log1p(
            marketing_response / 0.03
        )
        * customer_area_marketing_response
    )

    # July experiment seasonality.
    seasonality = (
        1.05
        ** customer_area_seasonality
    )

    # First-purchase scenario so that repeat behavior does not
    # influence this diagnostic.
    repeat_multiplier = 1.0

    # Exact base probability formula from
    # generate_transactions().
    base_probability = (
        0.00075
        * customer_probability_data[
            "purchase_propensity"
        ]
        / customer_probability_data[
            "purchase_propensity"
        ].mean()
    )

    purchase_probability = (
        base_probability
        * customer_area_purchase_propensity
        / customer_area_purchase_propensity.mean()
        * seasonality
        * marketing_effect
        * repeat_multiplier
    )

    purchase_probability = np.clip(
        purchase_probability,
        0.0,
        0.05,
    )

    customer_probability_data[
        "marketing_response"
    ] = marketing_response

    customer_probability_data[
        "marketing_effect"
    ] = marketing_effect

    customer_probability_data[
        "purchase_probability"
    ] = purchase_probability

    # Map experiment group to each customer through area_id.
    customer_probability_data = (
        customer_probability_data
        .merge(
            experiment_assignment[
                ["area_id", "experiment_group"]
            ],
            on="area_id",
            how="left",
            validate="many_to_one",
        )
    )

    assert (
        customer_probability_data["experiment_group"]
        .notna()
        .all()
    )
    # ------------------------------------------------------------
    # Treatment vs control
    # ------------------------------------------------------------

    probability_summary = (
        customer_probability_data
        .groupby("experiment_group")
        .agg(
            marketing_response=(
                "marketing_response",
                "mean",
            ),
            marketing_effect=(
                "marketing_effect",
                "mean",
            ),
            purchase_probability=(
                "purchase_probability",
                "mean",
            ),
        )
    )

    print(
        "\nModeled experiment-period probability components:"
    )

    print(
        probability_summary.round(6)
    )

    treatment = probability_summary.loc[
        "treatment"
    ]

    control = probability_summary.loc[
        "control"
    ]

    print(
        "\nTreatment / Control ratios:"
    )

    print(
        f"Marketing response: "
        f"{(
            treatment["marketing_response"]
            / control["marketing_response"]
        ):.4f}"
    )

    print(
        f"Marketing effect: "
        f"{(
            treatment["marketing_effect"]
            / control["marketing_effect"]
        ):.4f}"
    )

    print(
        f"Purchase probability: "
        f"{(
            treatment["purchase_probability"]
            / control["purchase_probability"]
        ):.4f}"
    )

    print(
        "\nPurchase probability diagnostic completed."
    )

    # ------------------------------------------------------------
    # 7. Counterfactual marketing-response diagnostic
    # ------------------------------------------------------------

    print("\n7. Counterfactual marketing-response diagnostic")
    print("-" * 60)

    # Use the same customers and area characteristics.
    # Only marketing pressure will change between scenarios.

    customer_counterfactual_data = (
        customers
        .merge(
            hidden_customer_characteristics,
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            channel_responsiveness,
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
    )

    assert len(customer_counterfactual_data) == len(customers)

    # Add area-level characteristics.
    area_lookup = (
        area_characteristics
        .set_index("area_id")
        .to_dict("index")
    )

    customer_counterfactual_data[
        "area_purchase_propensity"
    ] = [
        area_lookup[area_id]["purchase_propensity"]
        for area_id in customer_counterfactual_data["area_id"]
    ]

    customer_counterfactual_data[
        "area_marketing_responsiveness"
    ] = [
        area_lookup[area_id]["marketing_responsiveness"]
        for area_id in customer_counterfactual_data["area_id"]
    ]

    customer_counterfactual_data[
        "area_seasonality"
    ] = [
        area_lookup[area_id]["seasonality_sensitivity"]
        for area_id in customer_counterfactual_data["area_id"]
    ]

    # ------------------------------------------------------------
    # Average experiment-period pressure by group and area
    # ------------------------------------------------------------

    pressure_by_group = (
        experiment_pressure
        .groupby("experiment_group")[pressure_columns]
        .mean()
    )

    control_pressure = pressure_by_group.loc[
        "control"
    ]

    treatment_pressure = pressure_by_group.loc[
        "treatment"
    ]

    # ------------------------------------------------------------
    # Scenario 1: control-level pressure
    # ------------------------------------------------------------

    control_response = (
        control_pressure["Google_pressure"]
        * customer_counterfactual_data["Google"]
        + control_pressure["Meta_pressure"]
        * customer_counterfactual_data["Meta"]
        + control_pressure["TikTok_pressure"]
        * customer_counterfactual_data["TikTok"]
        + control_pressure["CRM_pressure"]
        * customer_counterfactual_data["CRM"]
    )

    control_effect = (
        1.0
        + 0.18
        * np.log1p(control_response / 0.03)
        * customer_counterfactual_data[
            "area_marketing_responsiveness"
        ]
    )

    # ------------------------------------------------------------
    # Scenario 2: treatment-level pressure
    # ------------------------------------------------------------

    treatment_response = (
        treatment_pressure["Google_pressure"]
        * customer_counterfactual_data["Google"]
        + treatment_pressure["Meta_pressure"]
        * customer_counterfactual_data["Meta"]
        + treatment_pressure["TikTok_pressure"]
        * customer_counterfactual_data["TikTok"]
        + treatment_pressure["CRM_pressure"]
        * customer_counterfactual_data["CRM"]
    )

    treatment_effect = (
        1.0
        + 0.18
        * np.log1p(treatment_response / 0.03)
        * customer_counterfactual_data[
            "area_marketing_responsiveness"
        ]
    )

    # ------------------------------------------------------------
    # Hold customer and area characteristics constant
    # ------------------------------------------------------------

    base_probability = (
        0.00075
        * customer_counterfactual_data["purchase_propensity"]
        / customer_counterfactual_data["purchase_propensity"].mean()
    )

    seasonality = (
        1.05
        ** customer_counterfactual_data[
            "area_seasonality"
        ]
    )

    area_purchase_factor = (
        customer_counterfactual_data[
            "area_purchase_propensity"
        ]
        / customer_counterfactual_data[
            "area_purchase_propensity"
        ].mean()
    )

    control_probability = (
        base_probability
        * area_purchase_factor
        * seasonality
        * control_effect
    )

    treatment_probability = (
        base_probability
        * area_purchase_factor
        * seasonality
        * treatment_effect
    )

    control_probability = np.clip(
        control_probability,
        0.0,
        0.05,
    )

    treatment_probability = np.clip(
        treatment_probability,
        0.0,
        0.05,
    )

    # ------------------------------------------------------------
    # Compare identical customers under both pressures
    # ------------------------------------------------------------

    counterfactual_summary = pd.DataFrame(
        {
            "control": [
                control_response.mean(),
                control_effect.mean(),
                control_probability.mean(),
            ],
            "treatment": [
                treatment_response.mean(),
                treatment_effect.mean(),
                treatment_probability.mean(),
            ],
        },
        index=[
            "marketing_response",
            "marketing_effect",
            "purchase_probability",
        ],
    )

    print(
        "\nCounterfactual modeled outcomes:"
    )

    print(
        counterfactual_summary.round(6)
    )

    print(
        "\nTreatment / Control ratios:"
    )

    print(
        f"Marketing response: "
        f"{(
            counterfactual_summary.loc[
                "marketing_response",
                "treatment",
            ]
            / counterfactual_summary.loc[
                "marketing_response",
                "control",
            ]
        ):.4f}"
    )

    print(
        f"Marketing effect: "
        f"{(
            counterfactual_summary.loc[
                "marketing_effect",
                "treatment",
            ]
            / counterfactual_summary.loc[
                "marketing_effect",
                "control",
            ]
        ):.4f}"
    )

    print(
        f"Purchase probability: "
        f"{(
            counterfactual_summary.loc[
                "purchase_probability",
                "treatment",
            ]
            / counterfactual_summary.loc[
                "purchase_probability",
                "control",
            ]
        ):.4f}"
    )

    print(
        "\nCounterfactual diagnostic completed."
    )

# === journey and attribution validation
def validate_touchpoint_structure(
    marketing_touchpoints,
    customers,
    hidden_customer_characteristics,
    marketing_performance,
):
    """Validate structural integrity of customer-level marketing touchpoints."""

    print("\n" + "=" * 70)
    print("DAY 4: TOUCHPOINT STRUCTURE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Required columns
    # --------------------------------------------------------
    required_columns = [
        "touchpoint_id",
        "customer_id",
        "touchpoint_timestamp",
        "channel",
        "campaign_id",
        "campaign_name",
    ]

    assert list(marketing_touchpoints.columns) == required_columns, (
        "Unexpected marketing_touchpoints schema."
    )

    # --------------------------------------------------------
    # 2. Basic integrity
    # --------------------------------------------------------
    assert len(marketing_touchpoints) > 0
    assert marketing_touchpoints[required_columns].notna().all().all()
    assert marketing_touchpoints["touchpoint_id"].is_unique

    assert pd.api.types.is_datetime64_any_dtype(
        marketing_touchpoints["touchpoint_timestamp"]
    )

    print(f"Touchpoints: {len(marketing_touchpoints):,}")
    print(
        f"Customers with touchpoints: "
        f"{marketing_touchpoints['customer_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # 3. Customer referential integrity
    # --------------------------------------------------------
    customer_ids = set(customers["customer_id"])

    assert marketing_touchpoints["customer_id"].isin(
        customer_ids
    ).all()

    customer_area_lookup = (
        customers
        .set_index("customer_id")["area_id"]
    )

    tp_customer_area = marketing_touchpoints["customer_id"].map(
        customer_area_lookup
    )

    assert tp_customer_area.notna().all()

    # --------------------------------------------------------
    # 4. Customer lifecycle timing
    # --------------------------------------------------------
    entry_date_lookup = (
        hidden_customer_characteristics
        .set_index("customer_id")["entry_date"]
        .pipe(pd.to_datetime)
    )

    tp_entry_dates = marketing_touchpoints["customer_id"].map(
        entry_date_lookup
    )

    tp_dates = (
        marketing_touchpoints["touchpoint_timestamp"]
        .dt.normalize()
    )

    assert tp_entry_dates.notna().all()
    assert (tp_dates >= tp_entry_dates).all()

    # --------------------------------------------------------
    # 5. Marketing campaign reference
    # --------------------------------------------------------
    marketing_reference = marketing_performance[
        [
            "campaign_id",
            "campaign_name",
            "channel",
            "date",
        ]
    ].copy()

    marketing_reference["date"] = (
        pd.to_datetime(marketing_reference["date"])
        .dt.normalize()
    )

    # One campaign_id must have one campaign name and channel.
    assert (
        marketing_reference
        .groupby("campaign_id")["campaign_name"]
        .nunique()
        .eq(1)
        .all()
    )

    assert (
        marketing_reference
        .groupby("campaign_id")["channel"]
        .nunique()
        .eq(1)
        .all()
    )

    campaign_reference = (
        marketing_reference
        .drop_duplicates("campaign_id")
        .set_index("campaign_id")
    )

    # --------------------------------------------------------
    # 6. Campaign referential integrity
    # --------------------------------------------------------
    tp_campaign_names = marketing_touchpoints["campaign_id"].map(
        campaign_reference["campaign_name"]
    )

    tp_campaign_channels = marketing_touchpoints["campaign_id"].map(
        campaign_reference["channel"]
    )

    assert tp_campaign_names.notna().all()
    assert tp_campaign_channels.notna().all()

    assert (
        marketing_touchpoints["campaign_name"]
        == tp_campaign_names
    ).all()

    assert (
        marketing_touchpoints["channel"]
        == tp_campaign_channels
    ).all()

    # --------------------------------------------------------
    # 7. Valid channels
    # --------------------------------------------------------
    expected_channels = {
        "Google",
        "Meta",
        "TikTok",
        "CRM",
    }

    assert set(
        marketing_touchpoints["channel"].unique()
    ).issubset(expected_channels)

    # --------------------------------------------------------
    # 8. Campaign area ↔ customer area
    # --------------------------------------------------------
    tp_area_ids = (
        marketing_touchpoints["campaign_name"]
        .str.extract(
            r"(AREA_\d{3})$",
            expand=False,
        )
    )

    assert tp_area_ids.notna().all()

    assert (
        tp_customer_area.to_numpy()
        == tp_area_ids.to_numpy()
    ).all()

    # --------------------------------------------------------
    # 9. Campaign + date must exist in marketing performance
    #
    # Use a compact set of valid campaign-date combinations
    # instead of merging against the full touchpoint table.
    # --------------------------------------------------------
    
    valid_campaign_dates = set(
        zip(
            marketing_reference["campaign_id"],
            marketing_reference["date"],
        )
    )

    touchpoint_campaign_dates = zip(
        marketing_touchpoints["campaign_id"],
        tp_dates,
    )

    assert all(
        key in valid_campaign_dates
        for key in touchpoint_campaign_dates
    )

    # --------------------------------------------------------
    # 10. Volume diagnostics
    # --------------------------------------------------------
    print("\nTouchpoints by channel:")
    print(
        marketing_touchpoints["channel"]
        .value_counts()
        .sort_index()
    )

    print("\nTouchpoints by area:")
    print(
        tp_area_ids
        .value_counts()
        .sort_index()
    )

    print("\nTouchpoint date range:")
    print(f"Min: {tp_dates.min()}")
    print(f"Max: {tp_dates.max()}")

    print("\nAll touchpoint structural checks passed.")

def validate_journey_behavior(
    marketing_touchpoints,
    customers,
    transactions,
    run_purchase_journey_diagnostic=False,
):
    """Validate journey behavior and optional purchase attribution readiness.
 
    Eligibility rule: a touchpoint is eligible for a purchase if it falls in
        [purchase_date - 14 days, purchase_date + 1 day)
    i.e. the left edge is inclusive and the whole purchase day is included,
    but the following day is not.
    """
 
    ATTRIBUTION_WINDOW_DAYS = 14
    JOURNEY_INACTIVITY_DAYS = 30
    window = pd.Timedelta(days=ATTRIBUTION_WINDOW_DAYS)
    one_day = pd.Timedelta(days=1)
 
    print("\n" + "=" * 70)
    print("DAY 4: JOURNEY BEHAVIOR VALIDATION")
    print("=" * 70)
 
    # --------------------------------------------------------
    # 1. Prepare touchpoints (only the columns we actually use)
    # --------------------------------------------------------
    touchpoints = marketing_touchpoints[
        ["touchpoint_id", "customer_id", "touchpoint_timestamp", "channel"]
    ].copy()
 
    touchpoints["touchpoint_timestamp"] = pd.to_datetime(
        touchpoints["touchpoint_timestamp"]
    ).astype("datetime64[ns]")
 
    touchpoints = touchpoints.sort_values(
        ["customer_id", "touchpoint_timestamp"]
    ).reset_index(drop=True)
 
    n_tp = len(touchpoints)
 
    # --------------------------------------------------------
    # 2. Prepare completed purchases
    # --------------------------------------------------------
    purchases = transactions.loc[
        ~transactions["cancelled"],
        ["transaction_id", "customer_id", "transaction_date"],
    ].copy()
 
    purchases["transaction_date"] = (
        pd.to_datetime(purchases["transaction_date"])
        .dt.normalize()
        .astype("datetime64[ns]")
    )
 
    purchases = purchases.sort_values(
        ["customer_id", "transaction_date"]
    ).reset_index(drop=True)
 
    # --------------------------------------------------------
    # 3. Derive journeys using 30-day inactivity (vectorised)
    # --------------------------------------------------------
    ts = touchpoints["touchpoint_timestamp"]
 
    gap_days = ts.diff().dt.total_seconds() / 86_400
 
    new_journey = (
        touchpoints["customer_id"].ne(touchpoints["customer_id"].shift())
        | gap_days.gt(JOURNEY_INACTIVITY_DAYS)
    ).to_numpy()
 
    journey_number = np.cumsum(new_journey)  # 1-based, unique across customers
    touchpoints["journey_number"] = journey_number
 
    starts = np.flatnonzero(new_journey)
    ends = np.append(starts[1:], n_tp)
 
    # --------------------------------------------------------
    # 4. Build journey summary without groupby / lambdas
    # --------------------------------------------------------
    ts_np = ts.to_numpy()
    touchpoint_count = ends - starts
 
    channel_codes, _ = pd.factorize(touchpoints["channel"])
    pairs = pd.DataFrame({"j": journey_number, "c": channel_codes})
    pairs = pairs[pairs["c"] >= 0].drop_duplicates()
    channel_count = np.bincount(
        pairs["j"].to_numpy() - 1, minlength=len(starts)
    )
 
    channels = touchpoints["channel"].astype(str).tolist()
    channel_path = [
        " → ".join(channels[a:b]) for a, b in zip(starts, ends)
    ]
 
    journey_summary = pd.DataFrame(
        {
            "customer_id": touchpoints["customer_id"].to_numpy()[starts],
            "journey_number": journey_number[starts],
            "journey_start": ts_np[starts],
            "journey_end": ts_np[ends - 1],
            "touchpoint_count": touchpoint_count,
            "channel_count": channel_count,
            "channel_path": channel_path,
        }
    )
 
    journey_summary["journey_duration_days"] = (
        journey_summary["journey_end"] - journey_summary["journey_start"]
    ).dt.total_seconds() / 86_400
 
    # --------------------------------------------------------
    # 4.1 Validate inactivity boundaries
    # --------------------------------------------------------
    within_gaps = gap_days.where(~new_journey)
    maximum_gap = within_gaps.max()
    maximum_gap = 0.0 if pd.isna(maximum_gap) else maximum_gap
 
    violating_journeys = touchpoints.loc[
        within_gaps.gt(JOURNEY_INACTIVITY_DAYS), "journey_number"
    ].nunique()
 
    print("\nJourney inactivity-boundary diagnostic:")
    print(f"Maximum within-journey gap: {maximum_gap:.2f} days")
    print(f"Journeys exceeding 30 days: {violating_journeys:,}")
 
    assert violating_journeys == 0, (
        "At least one journey contains a gap greater than "
        f"{JOURNEY_INACTIVITY_DAYS} days."
    )
 
    # --------------------------------------------------------
    # 5. Purchase eligibility window  [date - 14d, date + 1d)
    #
    # Touchpoints are sorted by (customer, time). We turn each
    # touchpoint into ONE sortable number:
    #
    #       key = customer_code * M + seconds_since_offset
    #
    # M is larger than any timestamp offset, so one customer's keys
    # can never overlap another customer's. A single searchsorted on
    # that key then gives, for every purchase, the exact slice of that
    # customer's eligible touchpoints:
    #
    #       left  = first touchpoint with time >= date - 14 days
    #       right = first touchpoint with time >= date + 1 day
    #
    # Eligible rows are [left, right). Whole seconds are exact here
    # because every boundary is a whole second (midnight based).
    # --------------------------------------------------------
    tp_codes, tp_unique_customers = pd.factorize(touchpoints["customer_id"])
    tp_codes = tp_codes.astype(np.int64)
 
    purchase_codes = tp_unique_customers.get_indexer(purchases["customer_id"])
    purchase_known = purchase_codes >= 0  # customer has any touchpoints
    purchase_codes = np.where(purchase_known, purchase_codes, 0).astype(
        np.int64
    )
 
    tp_sec = ts_np.astype("datetime64[s]").astype(np.int64)
    window_start_sec = (
        (purchases["transaction_date"] - window)
        .to_numpy()
        .astype("datetime64[s]")
        .astype(np.int64)
    )
    window_end_sec = (
        (purchases["transaction_date"] + one_day)
        .to_numpy()
        .astype("datetime64[s]")
        .astype(np.int64)
    )
 
    all_secs = np.concatenate([tp_sec, window_start_sec, window_end_sec])
    offset = all_secs.min() if all_secs.size else 0
    key_span = (all_secs.max() - offset + 1) if all_secs.size else 1
 
    assert (len(tp_unique_customers) + 1) * key_span < 2**62, (
        "Composite key would overflow; reduce the time span."
    )
 
    tp_key = tp_codes * key_span + (tp_sec - offset)
    assert (np.diff(tp_key) >= 0).all(), "Touchpoint keys are not sorted."
 
    left = np.searchsorted(
        tp_key,
        purchase_codes * key_span + (window_start_sec - offset),
        side="left",
    )
    right = np.searchsorted(
        tp_key,
        purchase_codes * key_span + (window_end_sec - offset),
        side="left",
    )
 
    eligible_count = np.where(purchase_known, right - left, 0)
    purchase_has_touchpoint = eligible_count > 0
 
    purchases["has_14d_touchpoint"] = purchase_has_touchpoint
 
    print("\nCompleted purchase attribution-window coverage:")
    print(f"Completed purchases: {len(purchases):,}")
    print(
        "With an eligible touchpoint: "
        f"{purchase_has_touchpoint.sum():,}"
    )
    print(
        "Without an eligible touchpoint: "
        f"{(~purchase_has_touchpoint).sum():,}"
    )
    print(
        "Coverage: "
        f"{purchase_has_touchpoint.mean():.2%}"
        if len(purchases)
        else "Coverage: N/A"
    )
 
    # --------------------------------------------------------
    # 6. Optional detailed purchase-to-journey mapping
    #
    # Uses the SAME left/right boundaries as Section 5, and checks
    # that every expanded touchpoint belongs to the purchase's own
    # customer.
    # --------------------------------------------------------
    purchase_journey_map = pd.DataFrame(
        columns=[
            "transaction_id",
            "customer_id",
            "journey_number",
            "purchase_date",
        ]
    )
 
    purchase_touchpoints = pd.DataFrame(
        columns=[
            "transaction_id",
            "customer_id",
            "purchase_date",
            "touchpoint_id",
            "journey_number",
            "touchpoint_timestamp",
            "days_before_purchase",
        ]
    )
 
    if run_purchase_journey_diagnostic:
 
        print("\nDetailed purchase-to-journey diagnostic:")
 
        if purchase_has_touchpoint.any():
            cov = purchases.loc[purchase_has_touchpoint].reset_index(
                drop=True
            )
            cov_left = left[purchase_has_touchpoint].astype(np.int64)
            counts = eligible_count[purchase_has_touchpoint].astype(np.int64)
 
            # Expand each purchase into its eligible touchpoint rows.
            purchase_rep = np.repeat(np.arange(len(cov)), counts)
            offsets = np.arange(counts.sum()) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            tp_idx = np.repeat(cov_left, counts) + offsets
 
            tp_rows = touchpoints.iloc[tp_idx]
            cov_rows = cov.iloc[purchase_rep]
 
            assert np.array_equal(
                tp_rows["customer_id"].to_numpy(),
                cov_rows["customer_id"].to_numpy(),
            ), "Expanded touchpoints include another customer's rows."
 
            purchase_touchpoints = pd.DataFrame(
                {
                    "transaction_id": cov_rows["transaction_id"].to_numpy(),
                    "customer_id": cov_rows["customer_id"].to_numpy(),
                    "purchase_date": cov_rows["transaction_date"].to_numpy(),
                    "touchpoint_id": tp_rows["touchpoint_id"].to_numpy(),
                    "journey_number": tp_rows["journey_number"].to_numpy(),
                    "touchpoint_timestamp": tp_rows[
                        "touchpoint_timestamp"
                    ].to_numpy(),
                }
            )
            # Note: purchase_date is midnight, so same-day touchpoints
            # give a small negative value (down to -1 day).
            purchase_touchpoints["days_before_purchase"] = (
                purchase_touchpoints["purchase_date"]
                - purchase_touchpoints["touchpoint_timestamp"]
            ).dt.total_seconds() / 86_400
 
        if not purchase_touchpoints.empty:
 
            jn = purchase_touchpoints.groupby("transaction_id")[
                "journey_number"
            ].agg(["min", "max"])
            spans_multiple = jn["min"].ne(jn["max"])
 
            assert not spans_multiple.any(), (
                "Eligible touchpoints for a purchase span "
                "multiple journeys."
            )
 
            purchase_journey_map = (
                purchase_touchpoints[
                    [
                        "transaction_id",
                        "customer_id",
                        "journey_number",
                        "purchase_date",
                    ]
                ]
                .drop_duplicates("transaction_id")
                .reset_index(drop=True)
            )
 
            print(
                "Purchases with eligible touchpoints: "
                f"{purchase_journey_map['transaction_id'].nunique():,}"
            )
 
            print(
                "Purchases spanning multiple journeys: "
                f"{spans_multiple.sum():,}"
            )
 
            purchases_per_journey = purchase_journey_map.groupby(
                ["customer_id", "journey_number"]
            )["transaction_id"].nunique()
 
            print(
                "Journeys associated with 2+ purchases: "
                f"{(purchases_per_journey > 1).sum():,}"
            )
 
            print("\nEligible touchpoints per purchase:")
            print(
                purchase_touchpoints.groupby("transaction_id")
                .size()
                .describe()
            )
 
            print("\nAttribution touchpoint timing:")
            print(purchase_touchpoints["days_before_purchase"].describe())
 
        else:
            print("No completed purchases have eligible touchpoints.")
 
    # --------------------------------------------------------
    # 7. Mark journeys with at least one covered purchase
    #
    # A purchase's journey is the journey of its last eligible
    # touchpoint (right - 1), using the same boundaries as above.
    # Journey numbers are unique across customers, so isin() is safe.
    # --------------------------------------------------------
    purchase_journey_ids = np.unique(
        journey_number[right[purchase_has_touchpoint] - 1]
    )
 
    journey_summary["has_purchase"] = journey_summary[
        "journey_number"
    ].isin(purchase_journey_ids)
 
    # --------------------------------------------------------
    # 8. Journey behavior diagnostics
    # --------------------------------------------------------
    journey_summary["journey_type"] = np.where(
        journey_summary["touchpoint_count"].eq(1),
        "single_touch",
        "multi_touch",
    )
 
    print("\nJourney behavioral diagnostics:")
 
    print("\nTouchpoints per journey by outcome:")
    print(
        journey_summary.groupby("has_purchase")["touchpoint_count"].agg(
            ["count", "mean", "median", "max"]
        )
    )
 
    print("\nJourney duration by outcome:")
    print(
        journey_summary.groupby("has_purchase")[
            "journey_duration_days"
        ].agg(["count", "mean", "median", "max"])
    )
 
    print("\nJourney type by outcome:")
    print(
        pd.crosstab(
            journey_summary["has_purchase"],
            journey_summary["journey_type"],
            normalize="index",
        )
    )
 
    print("\nChannels per journey by outcome:")
    print(
        journey_summary.groupby("has_purchase")["channel_count"].agg(
            ["mean", "median", "max"]
        )
    )
 
    purchase_journeys = journey_summary.loc[journey_summary["has_purchase"]]
 
    if not purchase_journeys.empty:
        multi_touch_share = (
            purchase_journeys["journey_type"].eq("multi_touch").mean()
        )
        print(
            "\nMulti-touch share among purchase-associated journeys: "
            f"{multi_touch_share:.2%}"
        )
 
    print(f"\nTotal journeys: {len(journey_summary):,}")
 
    print(
        "Customers with multiple journeys: "
        f"{(journey_summary.groupby('customer_id').size() > 1).sum():,}"
    )
 
    print("\nJourney type counts:")
    print(journey_summary["journey_type"].value_counts())
 
    print("\nTop channel paths:")
    print(journey_summary["channel_path"].value_counts().head(10))
 
    # --------------------------------------------------------
    # 9. Structural assertions
    # --------------------------------------------------------
    assert not journey_summary.empty
    assert journey_summary["touchpoint_count"].ge(1).all()
    assert journey_summary["journey_duration_days"].ge(0).all()
    assert (
        purchases["has_14d_touchpoint"].sum()
        + (~purchases["has_14d_touchpoint"]).sum()
        == len(purchases)
    )
 
    print("\nAll journey behavior checks passed.")
 
    return {
        "journey_summary": journey_summary,
        "purchase_journey_map": purchase_journey_map,
        "purchase_touchpoints": purchase_touchpoints,
        "purchase_coverage": purchases[
            [
                "transaction_id",
                "customer_id",
                "transaction_date",
                "has_14d_touchpoint",
            ]
        ],
    }





# ==== function generator ====
if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. Generate source data and simulation state
    # ---------------------------------------------------------
    print_memory("start")

    areas, area_characteristics = generate_areas()
    print_memory("after areas")

    campaign_configuration = generate_campaign_configuration(
        area_characteristics
    )
    print_memory("after campaign configuration")

    festival_campaign_configuration = (
        generate_festival_campaign_configuration(
            area_characteristics
        )
    )
    print_memory("after festival campaign configuration")

    experiment_assignment = generate_experiment_assignment(
        area_characteristics
    )
    print_memory("after experiment assignment")

    marketing_performance = generate_marketing_performance(
        campaign_configuration,
        festival_campaign_configuration,
        area_characteristics,
        experiment_assignment,
    )
    print_memory("after marketing performance")

    (
        customers,
        hidden_customer_characteristics,
        channel_responsiveness,
    ) = generate_customers(
        area_characteristics,
        marketing_performance,
    )
    print_memory("after customers")

    transactions = generate_transactions(
        customers,
        hidden_customer_characteristics,
        channel_responsiveness,
        area_characteristics,
        marketing_performance,
    )
    print_memory("after transactions")

    marketing_touchpoints = generate_marketing_touchpoints(
        customers,
        hidden_customer_characteristics,
        channel_responsiveness,
        marketing_performance,
    )
    print_memory("after marketing touchpoints")



    # ==== start validation function selection 
    # ============================================

    VALIDATION_DAY = 4

    if VALIDATION_DAY == 1:
        validate_day_1(
            area_characteristics,
            experiment_assignment,
            customers,
            hidden_customer_characteristics,
            marketing_performance,
            transactions,
        )

    elif VALIDATION_DAY == 2:
        validate_day_2(
            customers,
            hidden_customer_characteristics,
            transactions,
        )

    elif VALIDATION_DAY == 3:
        validate_day_3(
            area_characteristics,
            experiment_assignment,
            customers,
            hidden_customer_characteristics,
            channel_responsiveness,
            marketing_performance,
            transactions,
        )

    elif VALIDATION_DAY == 4:
        validate_journey_behavior(
            marketing_touchpoints,
            customers,
            transactions,
            run_purchase_journey_diagnostic=True,
        )

    # === integrity check for day 4 === 
    # journey_results = validate_journey_behavior(
    # marketing_touchpoints=marketing_touchpoints,
    # customers=customers,
    # transactions=transactions,
    # run_purchase_journey_diagnostic=True,
    # )

    # journey_map = journey_results["purchase_journey_map"].copy()
    # purchase_touchpoints = journey_results["purchase_touchpoints"].copy()
    # purchase_coverage = journey_results["purchase_coverage"].copy()

    # # Normalize transaction dates and calculate the exact attribution window.
    # purchase_coverage["transaction_date"] = pd.to_datetime(
    #     purchase_coverage["transaction_date"]
    # ).dt.normalize()

    # purchase_coverage["window_start"] = (
    #     purchase_coverage["transaction_date"] - pd.Timedelta(days=14)
    # )
    # purchase_coverage["window_end"] = (
    #     purchase_coverage["transaction_date"] + pd.Timedelta(days=1)
    # )

    # covered = purchase_coverage.loc[
    #     purchase_coverage["has_14d_touchpoint"]
    # ].copy()

    # # 1. Every covered purchase must appear in the journey map exactly once.
    # mapping_counts = journey_map.groupby("transaction_id").size()

    # missing_mappings = set(covered["transaction_id"]) - set(mapping_counts.index)
    # duplicate_mappings = mapping_counts[mapping_counts > 1]

    # print("Covered purchases:", len(covered))
    # print("Missing mappings:", len(missing_mappings))
    # print("Purchases mapped more than once:", len(duplicate_mappings))

    # assert not missing_mappings, "Some covered purchases have no journey mapping."
    # assert duplicate_mappings.empty, "Some purchases have multiple journey mappings."

    # # 2. Every recorded touchpoint must fall inside its purchase's window.
    # purchase_dates = covered[
    #     ["transaction_id", "customer_id", "window_start", "window_end"]
    # ]

    # detail = purchase_touchpoints.merge(
    #     purchase_dates,
    #     on=["transaction_id", "customer_id"],
    #     how="left",
    #     validate="many_to_one",
    # )

    # detail["touchpoint_timestamp"] = pd.to_datetime(
    #     detail["touchpoint_timestamp"]
    # )

    # outside_window = (
    #     detail["touchpoint_timestamp"].lt(detail["window_start"])
    #     | detail["touchpoint_timestamp"].ge(detail["window_end"])
    # )

    # print("Touchpoints outside attribution window:", int(outside_window.sum()))
    # assert not outside_window.any(), "Some mapped touchpoints fall outside the window."

    # # 3. All eligible touchpoints for a purchase must have one journey ID.
    # journey_counts = (
    #     detail.groupby("transaction_id")["journey_number"]
    #     .nunique()
    # )

    # multi_journey_purchases = journey_counts[journey_counts > 1]

    # print("Purchases spanning multiple journeys:", len(multi_journey_purchases))
    # assert multi_journey_purchases.empty, (
    #     "Some purchases have eligible touchpoints from multiple journeys."
    # )

    # # 4. The mapped journey ID must match the eligible touchpoints.
    # # Adjust "journey_number" below if your map uses a different column name.
    # journey_lookup = journey_map[
    #     ["transaction_id", "journey_number"]
    # ].drop_duplicates()

    # detail_check = detail.merge(
    #     journey_lookup,
    #     on="transaction_id",
    #     how="left",
    #     suffixes=("_touchpoint", "_mapped"),
    #     validate="many_to_one",
    # )

    # mismatches = detail_check.loc[
    #     detail_check["journey_number_touchpoint"]
    #     != detail_check["journey_number_mapped"]
    # ]

    # print("Touchpoints inconsistent with mapped journey:", len(mismatches))
    # assert mismatches.empty, (
    #     "The mapped journey does not match all eligible touchpoints."
    # )

    # print("\nAll purchase-to-journey integrity checks passed.")