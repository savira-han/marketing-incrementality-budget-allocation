from pathlib import Path

import numpy as np
import pandas as pd


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
                mean=np.log(1_000_000),
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

def generate_customers(area_characteristics):
    """Generate customers and their hidden behavioral characteristics."""

    area_ids = area_characteristics["area_id"].to_numpy()

    area_weights = (
        area_characteristics["customer_volume_potential"]
        * area_characteristics["baseline_demand"]
    )

    area_weights = area_weights / area_weights.sum()

    target_customers = RANDOM_GENERATOR.integers(
        TARGET_MIN_CUSTOMERS,
        TARGET_MAX_CUSTOMERS + 1,
    )

    customer_area_indices = RANDOM_GENERATOR.choice(
        len(area_ids),
        size=target_customers,
        p=area_weights,
    )

    customer_ids = [
        f"CUST_{i:06d}"
        for i in range(1, target_customers + 1)
    ]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "area_id": area_ids[customer_area_indices],
        }
    )

    entry_days = (
        HISTORICAL_END - HISTORICAL_START
    ).days

    entry_offsets = RANDOM_GENERATOR.integers(
        0,
        entry_days + 1,
        size=target_customers,
    )

    entry_dates = HISTORICAL_START + pd.to_timedelta(
        entry_offsets,
        unit="D",
    )

    hidden_customer_characteristics = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "entry_date": entry_dates,
            "purchase_propensity": RANDOM_GENERATOR.beta(
                a=2.5,
                b=35,
                size=target_customers,
            ),
            "aov_tendency": RANDOM_GENERATOR.lognormal(
                mean=np.log(1_000_000),
                sigma=0.35,
                size=target_customers,
            ),
            "repeat_purchase_tendency": RANDOM_GENERATOR.beta(
                a=2,
                b=5,
                size=target_customers,
            ),
            "price_sensitivity": RANDOM_GENERATOR.beta(
                a=2.5,
                b=4,
                size=target_customers,
            ),
        }
    )

    channel_responsiveness = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "Google": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=target_customers,
            ),
            "Meta": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=target_customers,
            ),
            "TikTok": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=target_customers,
            ),
            "CRM": RANDOM_GENERATOR.lognormal(
                mean=0.0,
                sigma=0.30,
                size=target_customers,
            ),
        }
    )

    return (
        customers,
        hidden_customer_characteristics,
        channel_responsiveness,
    )

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
    """Randomly assign areas to treatment/control within strata."""

    assignment_date = EXPERIMENT_START

    assignment_base = area_characteristics[
        [
            "area_id",
            "baseline_demand",
            "customer_volume_potential",
            "purchase_propensity",
        ]
    ].copy()

    # Standardize pre-experiment characteristics.
    for column in [
        "baseline_demand",
        "customer_volume_potential",
        "purchase_propensity",
    ]:
        mean = assignment_base[column].mean()
        std = assignment_base[column].std()

        assignment_base[f"{column}_z"] = (
            assignment_base[column] - mean
        ) / std

    # Composite pre-experiment score used only for stratification.
    assignment_base["stratification_score"] = (
        assignment_base["baseline_demand_z"]
        + assignment_base["customer_volume_potential_z"]
        + assignment_base["purchase_propensity_z"]
    ) / 3

    # Five strata with four areas each.
    assignment_base["stratum"] = pd.qcut(
        assignment_base["stratification_score"],
        q=5,
        labels=False,
    )

    assignment_records = []

    for stratum, stratum_data in assignment_base.groupby("stratum"):

        area_ids = stratum_data["area_id"].tolist()

        RANDOM_GENERATOR.shuffle(area_ids)

        for index, area_id in enumerate(area_ids):

            if index < len(area_ids) / 2:
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

    assignment_df = pd.DataFrame(assignment_records)

    # Validate stratified assignment before returning the raw experiment table.
    stratum_group_counts = (
        assignment_base
        .merge(
            assignment_df,
            on="area_id",
            how="left",
        )
        .groupby(["stratum", "experiment_group"])
        .size()
        .unstack(fill_value=0)
    )

    assert (stratum_group_counts["treatment"] == 2).all()
    assert (stratum_group_counts["control"] == 2).all()

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

    customer_area_baseline_demand = np.array(
        [
            area_lookup[area_id]["baseline_demand"]
            for area_id in customer_area_ids
        ]
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

    # Marketing pressure is expressed as clicks per customer
    # in each area. This allows areas with different customer
    # populations to be compared.
    for channel in ELIGIBLE_CHANNELS:
        daily_area_clicks[f"{channel}_pressure"] = (
            daily_area_clicks[channel]
            / daily_area_clicks["customer_count"]
        )

    # --------------------------------------------------------
    # Simulation state
    # --------------------------------------------------------

    purchase_count = np.zeros(
        len(customer_data),
        dtype=np.int16,
    )

    last_purchase_date = np.full(
        len(customer_data),
        np.datetime64("NaT","ns"),
        dtype="datetime64[ns]",
    )

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

    for simulation_date in all_dates:

        simulation_date_np = np.datetime64(
            simulation_date,
            "ns",
        )

        active_mask = (
            customer_entry_dates
            <= simulation_date_np
        )

        active_indices = np.flatnonzero(active_mask)

        if len(active_indices) == 0:
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

        google_pressure = np.array(
            [
                daily_marketing_lookup[area_id][
                    "Google_pressure"
                ]
                for area_id in customer_area_ids[
                    active_indices
                ]
            ]
        )

        meta_pressure = np.array(
            [
                daily_marketing_lookup[area_id][
                    "Meta_pressure"
                ]
                for area_id in customer_area_ids[
                    active_indices
                ]
            ]
        )

        tiktok_pressure = np.array(
            [
                daily_marketing_lookup[area_id][
                    "TikTok_pressure"
                ]
                for area_id in customer_area_ids[
                    active_indices
                ]
            ]
        )

        crm_pressure = np.array(
            [
                daily_marketing_lookup[area_id][
                    "CRM_pressure"
                ]
                for area_id in customer_area_ids[
                    active_indices
                ]
            ]
        )

        # ----------------------------------------------------
        # Customer-specific marketing response
        # ----------------------------------------------------

        marketing_response = (
            google_pressure
            * customer_google_response[
                active_indices
            ]
            + meta_pressure
            * customer_meta_response[
                active_indices
            ]
            + tiktok_pressure
            * customer_tiktok_response[
                active_indices
            ]
            + crm_pressure
            * customer_crm_response[
                active_indices
            ]
        )

        # Log transformation creates diminishing returns from
        # increasingly high marketing pressure.
        marketing_effect = (
            1.0
            + 0.18
            * np.log1p(
                marketing_response / 0.03
            )
            * customer_area_marketing_response[
                active_indices
            ]
        )

        # ----------------------------------------------------
        # Seasonality
        # ----------------------------------------------------

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

        seasonality = (
            month_multipliers[simulation_date.month]
            ** customer_area_seasonality[
                active_indices
            ]
        )

        # ----------------------------------------------------
        # Repeat purchase behavior
        # ----------------------------------------------------

        previous_purchase = (
            purchase_count[active_indices] > 0
        )

        repeat_multiplier = np.where(
            previous_purchase,
            1.0
            + 0.35
            * customer_repeat_tendency[
                active_indices
            ],
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
                active_indices
            ]
            / customer_purchase_propensity.mean()
        )

        purchase_probability = (
            base_probability
            * customer_area_purchase_propensity[
                active_indices
            ]
            / customer_area_purchase_propensity.mean()
            * seasonality
            * marketing_effect
            * repeat_multiplier
            * festival_multiplier
        )

        # Keep the probability in a realistic range.
        purchase_probability = np.clip(
            purchase_probability,
            0.0,
            0.05,
        )

        purchase_events = (
            RANDOM_GENERATOR.random(
                len(active_indices)
            )
            < purchase_probability
        )

        purchase_indices = active_indices[
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

        # Festival promotions create stronger discounting.
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

        # A small proportion of purchases are cancelled.
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

        # Cancelled transactions have no realized revenue or
        # variable promotional cost.
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
        # Store transaction records
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

            purchase_count[customer_index] += 1

            last_purchase_date[customer_index] = (
                simulation_date_np
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

if __name__ == "__main__":
    areas, area_characteristics = generate_areas()

    (
        customers,
        hidden_customer_characteristics,
        channel_responsiveness,
    ) = generate_customers(area_characteristics)

    campaign_configuration = generate_campaign_configuration(
        area_characteristics
    )

    festival_campaign_configuration = (
        generate_festival_campaign_configuration(
            area_characteristics
        )
    )

    experiment_assignment = generate_experiment_assignment(
        area_characteristics
    )

    print("\nExperiment assignment:")
    print(experiment_assignment.sort_values("area_id"))

    print("\nExperiment group counts:")
    print(
        experiment_assignment["experiment_group"]
        .value_counts()
    )

    print("\nAssignment by experiment group:")
    print(
        experiment_assignment
        .sort_values(["experiment_group", "area_id"])
        .to_string(index=False)
    )

    print("\nCampaign configuration:")
    print(campaign_configuration.head(10))

    print("\nCampaign counts by channel:")
    print(
        campaign_configuration["channel"]
        .value_counts()
        .sort_index()
    )

    print("\nFestival campaign configuration:")
    print(festival_campaign_configuration.head(10))

    print("\nFestival campaigns by channel:")
    print(
        festival_campaign_configuration["channel"]
        .value_counts()
        .sort_index()
    )

    assert len(experiment_assignment) == N_AREAS

    assert (
        experiment_assignment["area_id"].nunique()
        == N_AREAS
    )

    assert (
        experiment_assignment["experiment_group"]
        .value_counts()["treatment"]
        == N_TREATMENT_AREAS
    )

    assert (
        experiment_assignment["experiment_group"]
        .value_counts()["control"]
        == N_CONTROL_AREAS
    )

    assert (
        experiment_assignment["assignment_date"]
        .eq(EXPERIMENT_START)
        .all()
    )

    print("\nExperiment assignment validation passed.")

    marketing_performance = generate_marketing_performance(
        campaign_configuration,
        festival_campaign_configuration,
        area_characteristics,
        experiment_assignment,
    )

    print("\nMarketing performance shape:")
    print(marketing_performance.shape)

    print("\nMarketing performance date range:")
    print(
        marketing_performance["date"].min(),
        "to",
        marketing_performance["date"].max(),
    )

    validate_marketing_performance(
        marketing_performance,
        experiment_assignment,
    )

    transactions = generate_transactions(
        customers,
        hidden_customer_characteristics,
        channel_responsiveness,
        area_characteristics,
        marketing_performance,
    )

    print("\nTransactions:")
    print(transactions.head())

    print("\nTransaction dataset shape:")
    print(transactions.shape)

    marketing_touchpoints = generate_marketing_touchpoints(
        customers,
        hidden_customer_characteristics,
        channel_responsiveness,
        marketing_performance,
    )

    print("\nMarketing touchpoints:")
    print(marketing_touchpoints.head())

    print("\nMarketing touchpoint dataset shape:")
    print(marketing_touchpoints.shape)