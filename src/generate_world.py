"""
generate_world.py  -  Synthetic OTA marketing world (v5)

v5 changes (from the Day 3 audit; see docs/SIMULATION_RULES.md):
  * Customer behaviour fix: purchase rate, segment mix and a post-purchase COOLDOWN were
    re-tuned so that, among customers observed for 12+ months, buyers make about 1.6-2.0
    purchases and about 40-50% of buyers repeat.  New validation checks enforce this.
  * AOV_MEAN re-tuned so the iROAS pattern survives the lower purchase volume
    (only CRM stays profitable at the margin).
  * New truth columns: extra customers acquired because of the +25% spend, and their value.
  * Everything else (intent classes, in-market state, segments, organic touchpoints,
    imperfections, diff_pretrend variant) is unchanged and still awaits keep/drop/defer.

v4 recap: campaign intent classes (brand/retargeting reach customers who are already
in-market, so attributed credit >> causal effect), a latent in-market state, dormant /
casual / frequent customer segments, an Organic touchpoint source, richer raw-layer
imperfections, and WORLD_VARIANT=diff_pretrend (violated parallel trends stress test).

One coherent data-generating process for the Marketing Incrementality & Budget
Allocation project.  Order of construction (each step depends only on earlier ones):

    areas -> experiment_assignment -> campaigns -> marketing_performance
          -> customers (acquisition responds to spend) -> response calibration
          -> customer-level click exposure + touchpoints + transactions (one daily loop)
          -> simulation truth -> controlled imperfections -> validation

Key design decisions (all documented so they can be revisited):
  * Purchases respond to each customer's OWN adstocked click exposure, lagged one day,
    so a touchpoint always precedes the purchase it influences.
  * Latent click events are simulated at full scale (~25k/day, never materialised as a
    table).  `marketing_touchpoints` is a captured sample (TOUCHPOINT_CAPTURE_RATE).
  * Channel response = scale * area_responsiveness * Hill(adstock * customer_responsiveness).
    Hill slopes <= 1 => diminishing returns.  `scale` and `K` are CALIBRATED from target
    lift contributions and target mean saturation, not hand tuned.
  * Spend is seasonal (mildly) and has independent weekly area x channel noise, so
    response curves are identifiable.
  * Truth is computed by counterfactual inside the simulation loop, never re-derived.

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. Configuration
# ============================================================

MASTER_SEED = int(os.environ.get("WORLD_SEED", 42))
_PHASES = ["areas", "assignment", "marketing", "customers", "calibration",
           "touchpoints", "transactions", "imperfections", "inmarket"]
RNG = {name: np.random.default_rng(seq)
       for name, seq in zip(_PHASES, np.random.SeedSequence(MASTER_SEED).spawn(len(_PHASES)))}

# ---- timeline -------------------------------------------------------------
HISTORICAL_START = pd.Timestamp("2025-01-01")
HISTORICAL_END = pd.Timestamp("2026-06-30")
ASSIGNMENT_DATE = pd.Timestamp("2026-06-15")
EXPERIMENT_START = pd.Timestamp("2026-07-01")
EXPERIMENT_END = pd.Timestamp("2026-07-28")
POST_EXPERIMENT_START = pd.Timestamp("2026-07-29")
POST_EXPERIMENT_END = pd.Timestamp("2026-10-26")

ALL_DATES = pd.date_range(HISTORICAL_START, POST_EXPERIMENT_END, freq="D")
N_DAYS = len(ALL_DATES)
DATES_NS = ALL_DATES.to_numpy().astype("datetime64[ns]")

ATTRIBUTION_WINDOW_DAYS = 14          # used by validation; attribution itself is downstream
POST_EXPERIMENT_DAYS = 90

# ---- experiment -----------------------------------------------------------
N_AREAS = 20
N_TREATMENT_AREAS = 10
N_CONTROL_AREAS = 10
TREATMENT_SPEND_MULTIPLIER = 1.25

# ---- channels -------------------------------------------------------------
CHANNELS = ["Google", "Meta", "TikTok", "CRM"]
CH_IDX = {c: i for i, c in enumerate(CHANNELS)}
CHANNEL_MIX = {"Google": 0.40, "Meta": 0.30, "TikTok": 0.20, "CRM": 0.10}
MIX_ARR = np.array([CHANNEL_MIX[c] for c in CHANNELS])

CHANNEL_PARAMETERS = {
    # cpc/base_ctr: click economics.  adstock_decay: carry-over per day.
    # hill_slope (<=1 => concave).  target_mean_saturation: mean Hill value among recently
    # exposed customers at baseline exposure (drives K).  target_lift_contribution: mean relative
    # purchase-probability lift this channel contributes at baseline (drives scale).
    "Google": dict(cpc=2.20, base_ctr=0.035, adstock_decay=0.35, hill_slope=0.80,
                   target_mean_saturation=0.40, target_lift_contribution=0.115),
    "Meta":   dict(cpc=1.60, base_ctr=0.012, adstock_decay=0.50, hill_slope=0.90,
                   target_mean_saturation=0.35, target_lift_contribution=0.074),
    "TikTok": dict(cpc=1.10, base_ctr=0.009, adstock_decay=0.65, hill_slope=1.00,
                   target_mean_saturation=0.20, target_lift_contribution=0.047),
    "CRM":    dict(cpc=0.35, base_ctr=0.025, adstock_decay=0.20, hill_slope=1.00,
                   target_mean_saturation=0.55, target_lift_contribution=0.088),
}
DECAYS = np.array([CHANNEL_PARAMETERS[c]["adstock_decay"] for c in CHANNELS])

# (spend_weight, cpc_multiplier, ctr_multiplier, class) per campaign type
CAMPAIGN_TYPES = {
    "Google": {"BrandSearch": (0.80, 0.85, 1.40, "intent"), "GenericHotelSearch": (1.00, 1.05, 1.10, "discovery"),
               "DestinationSearch": (0.90, 0.95, 1.00, "discovery")},
    "Meta":   {"Prospecting": (1.10, 1.10, 0.85, "discovery"), "Retargeting": (0.80, 0.80, 1.20, "intent"),
               "HotelPromotion": (1.00, 1.00, 1.00, "discovery")},
    "TikTok": {"TravelDiscovery": (1.10, 1.05, 0.80, "discovery"), "DestinationContent": (0.90, 0.90, 0.85, "discovery"),
               "HotelPromotion": (1.00, 1.00, 1.00, "discovery")},
    "CRM":    {"Reengagement": (0.80, 0.80, 1.10, "discovery"), "RepeatBooking": (0.70, 0.70, 1.20, "intent"),
               "PromotionalOffer": (1.00, 1.00, 1.00, "discovery")},
}
FESTIVAL_SPEND_SHARE, FESTIVAL_CPC_MULT, FESTIVAL_CTR_MULT = 0.20, 1.10, 1.20

# ---- spend ----------------------------------------------------------------
BASE_DAILY_SPEND = 26_500              # calibrated so 2025 spend ~ $10M (validated)
SPEND_SEASONALITY_ELASTICITY = 0.5     # spend follows demand seasonality only partly
WEEKLY_BUDGET_SIGMA = 0.18             # exogenous area x channel x week variation

# ---- customers / behaviour ------------------------------------------------
SEED_CUSTOMERS = 400_000               # customers already existing at HISTORICAL_START (v5: was 200k)
ONGOING_ARRIVALS_PER_DAY = 340         # baseline new customers per day (all areas) (v5: was 170)
ACQUISITION_SPEND_ELASTICITY = 0.15    # new-customer arrivals ~ (trailing spend index)^0.15
BASE_DAILY_PURCHASE_PROB = 0.0007      # organic (no-marketing) daily purchase probability (v5: was 0.0022)
REPEAT_LIFT = 0.35                     # extra purchase rate after a first purchase (x repeat tendency)
COOLDOWN_DAYS = 40.0                   # v5: after a booking, purchase chance recovers as 1 - exp(-days/40)
MAX_DAILY_PURCHASE_PROB = 0.25

# v5 behaviour targets, measured on customers observed for at least 12 months
BEHAVIOR_MIN_HISTORY_DAYS = 365
TARGET_PURCHASES_PER_BUYER = (1.5, 2.2)     # band around the 1.6-2.0 target
TARGET_REPEAT_SHARE = (0.35, 0.55)          # band around the 40-50% target
TARGET_MEDIAN_GAP_DAYS = (30, 240)          # median days between a repeat buyer's purchases

MONTH_MULTIPLIERS = {1: 0.95, 2: 0.90, 3: 1.00, 4: 0.98, 5: 1.02, 6: 1.00,
                     7: 1.05, 8: 1.00, 9: 1.05, 10: 1.20, 11: 1.05, 12: 1.10}
DOW_MULTIPLIERS = np.array([0.97, 0.98, 0.99, 1.00, 1.03, 1.05, 0.98])  # Mon..Sun, mean 1
FESTIVAL_DEMAND_MULT = 1.30            # 10.10 festival: Oct 5-12
FESTIVAL_DISCOUNT_SHIFT, FESTIVAL_SUBSIDY_SHIFT = 0.04, 0.03

AOV_MEAN = 530.0                       # synthetic scale chosen so only CRM is profitable at the margin (v5: was 300)
REVENUE_NOISE_SIGMA = 0.15
DISC_A, DISC_B = 2.5, 5.5              # discount-rate Beta (before price-sensitivity factor)
SUBS_A, SUBS_B = 2.0, 8.0
RATE_CAP = 0.60
CANCEL_BASE, CANCEL_PSENS = 0.025, 0.025

# ---- v4: intent, in-market state, segments, organic, variant --------------
VARIANT = os.environ.get("WORLD_VARIANT", "baseline")      # baseline | diff_pretrend
PRETREND_SLOPE_PER_DAY = 0.010 / 30.0   # diff_pretrend: treatment areas drift +1%/month vs control
SEGMENT_NAMES = ["dormant", "casual", "frequent"]
SEGMENT_SHARES = [0.40, 0.45, 0.15]          # v5: was [0.55, 0.35, 0.10]
SEGMENT_MULT = [0.03, 1.0, 2.0]              # v5: was [0.03, 1.0, 2.5]
IN_MARKET_ENTER, IN_MARKET_EXIT, IN_MARKET_MULT = 0.006, 0.09, 8.0
IN_MARKET_STATIONARY = IN_MARKET_ENTER / (IN_MARKET_ENTER + IN_MARKET_EXIT)
CLASS_CLICK_BOOST = {"discovery": 1.5, "intent": 15.0}     # in-market click propensity multiplier
CLASS_CAUSAL_MULT = {"discovery": 1.0, "intent": 0.25}     # causal effect per spend share
ORGANIC_SESSIONS_PER_DAY, ORGANIC_CLICK_BOOST, ORGANIC_CAPTURE_RATE = 9_000, 10.0, 0.06
N_STREAMS = 2 * len(CHANNELS)                              # stream = 2*channel + (intent?1:0)
STREAM_NAMES = [f"{ch}|{cls}" for ch in CHANNELS for cls in ("discovery", "intent")]
DECAYS_S = np.repeat(np.array([CHANNEL_PARAMETERS[c]["adstock_decay"] for c in CHANNELS]), 2)

# ---- observability --------------------------------------------------------
TOUCHPOINT_CAPTURE_RATE = 0.15         # fraction of latent clicks logged as touchpoints (v5: was 0.10)
BROWSE_SIGMA = 1.1                     # clicks concentrate on in-market customers (heavy tail)
EXPOSED_X_MIN = 0.01                   # 'recently exposed' threshold used in calibration
TP_LOSS_RATES = {"Google": 0.03, "CRM": 0.01, "Meta": 0.08, "TikTok": 0.12}
LATE_ETL_RATE = 0.025
DUP_TXN_RATE, DUP_TP_RATE = 0.003, 0.010          # exact-duplicate rows (same primary key)
MISSING_UTM_RATE = 0.015                          # paid touchpoints losing campaign_id / name
ID_DRIFT_RATE = 0.015                             # touchpoint customer_id case / whitespace drift
TIKTOK_UTC_OFFSET_HOURS = -7                      # TikTok pixel logs UTC; warehouse is WIB (UTC+7)
PLATFORM_SPEND_BIAS = {"Google": 1.000, "Meta": 1.020, "TikTok": 0.970, "CRM": 1.000}
PLATFORM_CLICK_BIAS = {"Google": 1.040, "Meta": 1.000, "TikTok": 1.020, "CRM": 1.000}
N_NULL_AREA_SIZES = 2

# ---- calibration / truth windows -----------------------------------------
CAL_START, CAL_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-05-31")
CAL_WARMUP_DAYS = 30
CAL_SAMPLE_EVERY, CAL_SAMPLE_SIZE = 4, 30_000
REF_START, REF_END = pd.Timestamp("2026-04-15"), pd.Timestamp("2026-06-14")
CURVE_MULTIPLIERS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
MARGINAL_EXPOSURE_STEP = 0.10
CF_TAIL_DAYS = 21

# ---- output ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
OUTPUT_ROOT = Path(os.environ.get("WORLD_OUTPUT_ROOT", PROJECT_ROOT))
RAW_DATA_DIR = OUTPUT_ROOT / "data" / "raw"
CLEAN_REF_DIR = OUTPUT_ROOT / "data" / "processed" / "clean_reference"
TRUTH_DIR = OUTPUT_ROOT / "data" / "simulation_truth"
WRITE_CLEAN_REFERENCE = True


# ============================================================
# 2. Small helpers
# ============================================================

def ln1(rng, sigma, size=None):
    """Lognormal with mean exactly 1."""
    return rng.lognormal(-0.5 * sigma ** 2, sigma, size)


def idx_of(ts: pd.Timestamp) -> int:
    return int(ALL_DATES.get_loc(ts))


def hill(x, K, s):
    xs = np.power(x, s)
    return xs / (K ** s + xs)


def extract_area(series: pd.Series) -> pd.Series:
    return series.str.extract(r"(AREA_\d{3})$", expand=False)


FESTIVAL_DAY = np.asarray((ALL_DATES.month == 10) & (ALL_DATES.day >= 5) & (ALL_DATES.day <= 12))
MONTH_MULT_D = np.array([MONTH_MULTIPLIERS[m] for m in ALL_DATES.month])
DOW_D = DOW_MULTIPLIERS[ALL_DATES.dayofweek.to_numpy()]
SPEND_SEASON_D = MONTH_MULT_D ** SPEND_SEASONALITY_ELASTICITY
DEMAND_EVENT_D = np.where(FESTIVAL_DAY, FESTIVAL_DEMAND_MULT, 1.0)


# ============================================================
# 3. Areas, assignment, campaigns
# ============================================================

def generate_areas():
    rng = RNG["areas"]
    names = ["Jakarta", "Bandung", "Surabaya", "Yogyakarta", "Semarang", "Medan", "Denpasar",
             "Makassar", "Malang", "Palembang", "Tangerang", "Bekasi", "Depok", "Bogor",
             "Batam", "Balikpapan", "Solo", "Lombok", "Manado", "Padang"]
    assert len(names) == N_AREAS
    ids = [f"AREA_{i:03d}" for i in range(1, N_AREAS + 1)]
    areas = pd.DataFrame({
        "area_id": ids, "area_name": names,
        "area_size": pd.Series(rng.integers(40, 1_200, N_AREAS), dtype="Int64"),
    })
    chars = pd.DataFrame({
        "area_id": ids,
        "baseline_demand": ln1(rng, 0.35, N_AREAS),
        "customer_volume_potential": ln1(rng, 0.30, N_AREAS),
        "purchase_propensity": rng.beta(5, 25, N_AREAS),
        "aov_tendency": ln1(rng, 0.20, N_AREAS),                       # index, mean ~1
        "seasonality_sensitivity": rng.normal(1.0, 0.12, N_AREAS),
        "marketing_responsiveness": np.clip(rng.normal(1.0, 0.15, N_AREAS), 0.5, 1.5),
        "marketing_intensity": ln1(rng, 0.25, N_AREAS),                # spend per customer
    })
    return areas, chars


def area_weights(chars):
    """Customer-volume weights and spend weights (spend also scales with intensity)."""
    cw = (chars["customer_volume_potential"] * chars["baseline_demand"]).to_numpy()
    cw = cw / cw.sum()
    sw = cw * chars["marketing_intensity"].to_numpy()
    return cw, sw / sw.sum()


def generate_experiment_assignment(chars):
    """Stratified 10/10 randomisation.

    NOTE: stratification uses the hidden drivers of baseline volume / propensity / AOV
    (a proxy for observed baseline purchases and revenue).  Observed-data balance and
    pre-trend checks belong to Day 8 diagnostics.
    """
    rng = RNG["assignment"]
    cols = ["baseline_demand", "customer_volume_potential", "purchase_propensity", "aov_tendency"]
    base = chars[["area_id"] + cols].copy()
    z = (base[cols] - base[cols].mean()) / base[cols].std()
    base["score"] = z.mean(axis=1)
    base["stratum"] = pd.qcut(base["score"], q=5, labels=False)
    rows = []
    for _, grp in base.groupby("stratum"):
        ids = grp["area_id"].tolist()
        rng.shuffle(ids)
        for i, aid in enumerate(ids):
            rows.append({"experiment_id": "EXP_001", "area_id": aid,
                         "experiment_group": "treatment" if i < len(ids) / 2 else "control",
                         "assignment_date": ASSIGNMENT_DATE})
    return pd.DataFrame(rows).sort_values("area_id").reset_index(drop=True)


def generate_campaign_configuration(chars):
    rows, n = [], 1
    area_ids = chars["area_id"].tolist()
    for channel, types in CAMPAIGN_TYPES.items():
        tot_w = sum(v[0] for v in types.values())
        for ctype, (w, cpc_m, ctr_m, cls) in types.items():
            for aid in area_ids:
                rows.append(dict(campaign_id=f"CMP_{n:04d}", channel=channel, campaign_type=ctype,
                                 area_id=aid, campaign_name=f"{channel}_{ctype}_{aid}",
                                 share=w / tot_w, cpc_mult=cpc_m, ctr_mult=ctr_m, is_festival=False,
                                 campaign_class=cls, stream=2 * CH_IDX[channel] + int(cls == "intent")))
                n += 1
    for channel in CHANNELS:
        for aid in area_ids:
            rows.append(dict(campaign_id=f"CMP_{n:04d}", channel=channel, campaign_type="10.10_Festival",
                             area_id=aid, campaign_name=f"{channel}_10.10_{aid}",
                             share=FESTIVAL_SPEND_SHARE, cpc_mult=FESTIVAL_CPC_MULT,
                             ctr_mult=FESTIVAL_CTR_MULT, is_festival=True,
                             campaign_class="discovery", stream=2 * CH_IDX[channel]))
            n += 1
    return pd.DataFrame(rows)


# ============================================================
# 4. Marketing performance (vectorised: date x campaign matrices)
# ============================================================

def generate_marketing_performance(camp, chars, assignment):
    rng = RNG["marketing"]
    D, C = N_DAYS, len(camp)
    area_ids = chars["area_id"].tolist()
    A = len(area_ids)
    _, spend_w = area_weights(chars)

    area_idx = camp["area_id"].map({a: i for i, a in enumerate(area_ids)}).to_numpy()
    chan_idx = camp["channel"].map(CH_IDX).to_numpy()
    treat_area = assignment.set_index("area_id")["experiment_group"].eq("treatment")
    treat_c = camp["area_id"].map(treat_area).to_numpy(dtype=bool)

    week_idx = np.asarray((ALL_DATES - HISTORICAL_START).days // 7)
    exog = ln1(rng, WEEKLY_BUDGET_SIGMA, (A, len(CHANNELS), int(week_idx.max()) + 1))
    exog_mat = exog[area_idx[None, :], chan_idx[None, :], week_idx[:, None]]

    base = BASE_DAILY_SPEND * MIX_ARR[chan_idx] * camp["share"].to_numpy() * spend_w[area_idx]
    spend = base[None, :] * exog_mat * ln1(rng, 0.08, (D, C)) * SPEND_SEASON_D[:, None]

    exp_mask = np.asarray((ALL_DATES >= EXPERIMENT_START) & (ALL_DATES <= EXPERIMENT_END))
    spend = np.where(exp_mask[:, None] & treat_c[None, :], spend * TREATMENT_SPEND_MULTIPLIER, spend)

    active = np.ones((D, C), dtype=bool)
    fest_c = camp["is_festival"].to_numpy()
    active[:, fest_c] = FESTIVAL_DAY[:, None]
    spend = np.round(spend, 2) * active

    cpc = np.array([CHANNEL_PARAMETERS[c]["cpc"] for c in CHANNELS])[chan_idx] * camp["cpc_mult"].to_numpy()
    ctr = np.array([CHANNEL_PARAMETERS[c]["base_ctr"] for c in CHANNELS])[chan_idx] * camp["ctr_mult"].to_numpy()
    clicks = np.rint(spend / (cpc[None, :] * ln1(rng, 0.05, (D, C)))).astype(np.int64)
    impressions = np.rint(clicks / (ctr[None, :] * ln1(rng, 0.05, (D, C)))).astype(np.int64)

    d_i, c_i = np.nonzero(active)
    perf = pd.DataFrame({
        "date": ALL_DATES[d_i],
        "channel": camp["channel"].to_numpy()[c_i],
        "campaign_id": camp["campaign_id"].to_numpy()[c_i],
        "campaign_name": camp["campaign_name"].to_numpy()[c_i],
        "spend": spend[d_i, c_i],
        "impressions": impressions[d_i, c_i],
        "clicks": clicks[d_i, c_i],
    })
    return perf, clicks, spend


# ============================================================
# 5. Customers (acquisition responds to trailing area spend)
# ============================================================

@dataclass
class Population:
    customer_id: np.ndarray
    area_idx: np.ndarray
    entry_ns: np.ndarray
    purchase_propensity: np.ndarray
    aov_factor: np.ndarray
    repeat_tendency: np.ndarray
    price_sensitivity: np.ndarray
    channel_resp: np.ndarray            # n x 4
    browse_intensity: np.ndarray        # relative click propensity
    segment: np.ndarray                 # 0 dormant, 1 casual, 2 frequent

    @property
    def n(self):
        return len(self.customer_id)


def generate_customers(chars, camp, spend_mat, assignment):
    """Returns (population, acquisition).  `acquisition` holds the expected daily arrivals with the
    actual spend (`lam`) and in the no-extra-spend counterfactual (`lam_cf`), used for truth."""
    rng = RNG["customers"]
    A = len(chars)
    cust_w, _ = area_weights(chars)
    area_idx_c = camp["area_id"].map({a: i for i, a in enumerate(chars["area_id"])}).to_numpy()

    spend_area = np.zeros((N_DAYS, A))
    for a in range(A):
        spend_area[:, a] = spend_mat[:, area_idx_c == a].sum(axis=1)

    # counterfactual area spend: the same world without the experiment's extra 25%
    treat_area = assignment.set_index("area_id").loc[chars["area_id"], "experiment_group"].eq("treatment").to_numpy()
    exp_mask = np.asarray((ALL_DATES >= EXPERIMENT_START) & (ALL_DATES <= EXPERIMENT_END))
    spend_area_cf = spend_area.copy()
    spend_area_cf[np.ix_(exp_mask, treat_area)] /= TREATMENT_SPEND_MULTIPLIER

    trailing = pd.DataFrame(spend_area).rolling(7, min_periods=1).sum().shift(1)
    pre_mean = trailing.iloc[: idx_of(EXPERIMENT_START)].mean()
    sens = chars["seasonality_sensitivity"].to_numpy()

    def arrival_rate(spend_area_):
        tr = pd.DataFrame(spend_area_).rolling(7, min_periods=1).sum().shift(1)
        spend_index = (tr / pre_mean).fillna(1.0).to_numpy()
        return (ONGOING_ARRIVALS_PER_DAY * cust_w[None, :]
                * MONTH_MULT_D[:, None] ** sens[None, :]
                * spend_index ** ACQUISITION_SPEND_ELASTICITY)

    lam = arrival_rate(spend_area)
    lam_cf = arrival_rate(spend_area_cf)
    n_da = rng.poisson(lam)
    d_i, a_i = np.nonzero(n_da)
    rep = n_da[d_i, a_i]
    ongoing_area = np.repeat(a_i, rep)
    ongoing_entry = HISTORICAL_START + pd.to_timedelta(np.repeat(d_i, rep), unit="D")

    seed_area = rng.choice(A, size=SEED_CUSTOMERS, p=cust_w)
    seed_entry = HISTORICAL_START + pd.to_timedelta(rng.integers(-180, 0, SEED_CUSTOMERS), unit="D")

    area_idx = np.concatenate([seed_area, ongoing_area]).astype(np.int16)
    entry_ns = np.concatenate([seed_entry.to_numpy(), ongoing_entry.to_numpy()]).astype("datetime64[ns]")
    n = len(area_idx)

    order = np.lexsort((entry_ns, area_idx))           # area-major, entry-date-minor
    area_idx, entry_ns = area_idx[order], entry_ns[order]
    perm = rng.permutation(n)                          # IDs carry no area / date information
    customer_id = np.array([f"CUST_{p + 1:07d}" for p in perm], dtype=object)

    segment = rng.choice(3, size=n, p=SEGMENT_SHARES)
    purchase_propensity = np.array(SEGMENT_MULT)[segment] * ln1(rng, 0.35, n)
    browse = ln1(rng, BROWSE_SIGMA, n) * np.sqrt(purchase_propensity / purchase_propensity.mean())
    pop = Population(
        customer_id=customer_id, area_idx=area_idx, entry_ns=entry_ns,
        purchase_propensity=purchase_propensity,
        browse_intensity=browse, segment=segment,
        aov_factor=ln1(rng, 0.35, n),
        repeat_tendency=rng.beta(2, 5, n),
        price_sensitivity=rng.beta(2.5, 4, n),
        channel_resp=ln1(rng, 0.30, (n, len(CHANNELS))),
    )
    return pop, dict(lam=lam, lam_cf=lam_cf)


# ============================================================
# 6. Latent click allocation (shared by calibration and main pass)
# ============================================================

@dataclass
class ClickContext:
    offsets: np.ndarray
    sizes: np.ndarray
    active_n: np.ndarray          # days x areas
    groups: list                  # groups[a][stream] -> campaign column indices
    clicks_mat: np.ndarray        # days x campaigns
    organic_rate: np.ndarray      # latent organic sessions per day, per area


def build_click_context(pop, camp, chars, clicks_mat):
    A = len(chars)
    sizes = np.bincount(pop.area_idx, minlength=A)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    active_n = np.empty((N_DAYS, A), dtype=np.int64)
    for a in range(A):
        seg = pop.entry_ns[offsets[a]: offsets[a] + sizes[a]]
        active_n[:, a] = np.searchsorted(seg, DATES_NS, side="right")
    area_idx_c = camp["area_id"].map({a: i for i, a in enumerate(chars["area_id"])}).to_numpy()
    stream_c = camp["stream"].to_numpy()
    groups = [[np.flatnonzero((area_idx_c == a) & (stream_c == st)) for st in range(N_STREAMS)]
              for a in range(A)]
    cust_w, _ = area_weights(chars)
    return ClickContext(offsets, sizes, active_n, groups, clicks_mat, ORGANIC_SESSIONS_PER_DAY * cust_w)


def active_streams(ctx):
    return [st for st in range(N_STREAMS) if any(len(g[st]) for g in ctx.groups)]


def evolve_inmarket(state, rng):
    """Latent in-market episodes: people research, click intent ads, and buy soon after."""
    enter = rng.random(state.shape[0]) < IN_MARKET_ENTER
    stay = rng.random(state.shape[0]) >= IN_MARKET_EXIT
    return np.where(state, stay, enter)


def _pick(g, off, an, n, rng):
    """Weighted draw of n customers among the first `an` of an area segment (global cumsum g)."""
    base = g[off - 1] if off > 0 else 0.0
    u = base + rng.random(n) * (g[off + an - 1] - base)
    return np.minimum(np.searchsorted(g[off: off + an], u, side="right"), an - 1)


def allocate_day(ctx, d, n_cust, rng, inmarket, browse, capture=False):
    """Assign the day's latent clicks to active customers of the campaign's area.

    Intent campaigns (brand search, retargeting, repeat-booking CRM) reach in-market customers
    far more often than discovery campaigns do.  Returns per-customer click counts (n x streams)
    and, if capture, the logged paid touchpoints plus captured organic sessions."""
    def boosted(b):
        return np.cumsum(browse * (1.0 + (b - 1.0) * inmarket))
    g_cls = (boosted(CLASS_CLICK_BOOST["discovery"]), boosted(CLASS_CLICK_BOOST["intent"]))
    g_org = boosted(ORGANIC_CLICK_BOOST) if capture else None
    counts = np.zeros((n_cust, N_STREAMS))
    tp_c, tp_m, org_c, org_a = [], [], [], []
    for a in range(len(ctx.offsets)):
        an = ctx.active_n[d, a]
        if an == 0:
            continue
        off = ctx.offsets[a]
        for st in range(N_STREAMS):
            cids = ctx.groups[a][st]
            if len(cids) == 0:
                continue
            cl = ctx.clicks_mat[d, cids]
            n = int(cl.sum())
            if n == 0:
                continue
            picks = _pick(g_cls[st % 2], off, an, n, rng)
            counts[off: off + an, st] += np.bincount(picks, minlength=an)
            if capture:
                keep = rng.random(n) < TOUCHPOINT_CAPTURE_RATE
                if keep.any():
                    tp_c.append(off + picks[keep])
                    tp_m.append(np.repeat(cids, cl)[keep])
        if capture:
            k = int(rng.binomial(rng.poisson(ctx.organic_rate[a]), ORGANIC_CAPTURE_RATE))
            if k > 0:
                org_c.append(off + _pick(g_org, off, an, k, rng))
                org_a.append(np.full(k, a, dtype=np.int32))

    def cat(lst, dt):
        return np.concatenate(lst) if lst else np.empty(0, dt)
    return counts, cat(tp_c, np.int64), cat(tp_m, np.int64), cat(org_c, np.int64), cat(org_a, np.int32)


# ============================================================
# 7. Response calibration
# ============================================================

def stream_targets():
    """Target lift contribution per (channel, class): channel target split by spend share x causal effect."""
    out = {}
    for ch in CHANNELS:
        types = CAMPAIGN_TYPES[ch]
        tot_w = sum(v[0] for v in types.values())
        share = {"discovery": 0.0, "intent": 0.0}
        for _, (w, _, _, cls) in types.items():
            share[cls] += w / tot_w
        raw = {k: share[k] * CLASS_CAUSAL_MULT[k] for k in share}
        z = sum(raw.values())
        for cls in raw:
            out[(ch, cls)] = CHANNEL_PARAMETERS[ch]["target_lift_contribution"] * raw[cls] / z
    return out


def calibrate_response(pop, ctx, chars):
    """Solve Hill K (saturation among recently exposed) and scale per stream from real simulated
    exposure in a pre-experiment window.  Target lift contributions are PURCHASE-WEIGHTED (in-market
    customers buy ~8x more), so intent campaigns cannot hide a large causal effect behind their reach."""
    rng = RNG["calibration"]
    area_resp = chars["marketing_responsiveness"].to_numpy()
    targets = stream_targets()
    streams = active_streams(ctx)
    start, end = idx_of(CAL_START), idx_of(CAL_END)
    ad = np.zeros((pop.n, N_STREAMS))
    inmarket = rng.random(pop.n) < IN_MARKET_STATIONARY
    xs = {st: [] for st in streams}
    ars, wts = [], []
    for d in range(start - CAL_WARMUP_DAYS, end + 1):
        inmarket = evolve_inmarket(inmarket, rng)
        if d >= start and (d - start) % CAL_SAMPLE_EVERY == 0:
            act = np.flatnonzero(pop.entry_ns <= DATES_NS[d])
            samp = rng.choice(act, size=min(CAL_SAMPLE_SIZE, len(act)), replace=False)
            for st in streams:
                xs[st].append(ad[samp, st] * pop.channel_resp[samp, st // 2])
            ars.append(area_resp[pop.area_idx[samp]])
            wts.append(1.0 + (IN_MARKET_MULT - 1.0) * inmarket[samp])      # purchase-intensity weight
        counts = allocate_day(ctx, d, pop.n, rng, inmarket, pop.browse_intensity)[0]
        ad = ad * DECAYS_S + counts
    ar, wt = np.concatenate(ars), np.concatenate(wts)
    params = {}
    for st in streams:
        ch, cls = CHANNELS[st // 2], ("intent" if st % 2 else "discovery")
        cfg = CHANNEL_PARAMETERS[ch]
        slope = cfg["hill_slope"]
        target_sat = min(0.90, cfg["target_mean_saturation"] + (0.25 if cls == "intent" else 0.0))
        x = np.concatenate(xs[st])
        exposed = x > EXPOSED_X_MIN
        if exposed.sum() < 100:
            raise ValueError(f"too few exposed customers to calibrate stream {STREAM_NAMES[st]}")
        lo, hi = -12.0, 8.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if hill(x[exposed], np.exp(mid), slope).mean() > target_sat:
                lo = mid
            else:
                hi = mid
        K = float(np.exp(0.5 * (lo + hi)))
        h = hill(x, K, slope)
        params[STREAM_NAMES[st]] = dict(
            channel=ch, campaign_class=cls, K=K, slope=slope, decay=cfg["adstock_decay"],
            scale=float(targets[(ch, cls)] / ((wt * ar * h).sum() / wt.sum())), target_lift=float(targets[(ch, cls)]),
            mean_adstock_x=float(x.mean()), share_exposed=float(exposed.mean()),
            achieved_saturation_among_exposed=float(h[exposed].mean()))
    return params


# ============================================================
# 8. Behaviour simulation: exposure -> touchpoints -> purchases (+ counterfactual truth)
# ============================================================

def simulate_behavior(pop, ctx, camp, chars, assignment, cal, spend_mat):
    rng_a, rng_t, rng_s = RNG["touchpoints"], RNG["transactions"], RNG["inmarket"]
    A, n, D = len(chars), pop.n, N_DAYS
    area_ids_arr = chars["area_id"].to_numpy()
    area_resp = chars["marketing_responsiveness"].to_numpy()
    sens = chars["seasonality_sensitivity"].to_numpy()

    # ---- per-customer static quantities
    prop_norm = pop.purchase_propensity / pop.purchase_propensity.mean()
    a_prop = chars["purchase_propensity"].to_numpy()
    a_prop_norm = a_prop / a_prop.mean()
    a_aov = chars["aov_tendency"].to_numpy()
    rev_scale = AOV_MEAN * pop.aov_factor * (a_aov / a_aov.mean())[pop.area_idx]
    psens = pop.price_sensitivity
    rate_fac = 0.7 + 0.6 * psens
    p_cancel = CANCEL_BASE + CANCEL_PSENS * psens
    disc_mean, subs_mean = DISC_A / (DISC_A + DISC_B), SUBS_A / (SUBS_A + SUBS_B)
    cm_norm = rev_scale * (1 - disc_mean * rate_fac - subs_mean * rate_fac) * (1 - p_cancel)
    cm_fest = (rev_scale * (1 - (disc_mean + FESTIVAL_DISCOUNT_SHIFT) * rate_fac
                            - (subs_mean + FESTIVAL_SUBSIDY_SHIFT) * rate_fac) * (1 - p_cancel))
    season = MONTH_MULT_D[:, None] ** sens[None, :] * DOW_D[:, None]

    treat_area = assignment.set_index("area_id").loc[chars["area_id"], "experiment_group"].eq("treatment").to_numpy()
    treat_cust = treat_area[pop.area_idx]
    trend = np.ones((D, A))
    if VARIANT == "diff_pretrend":                    # violated parallel trends (stress-test variant)
        trend[:, treat_area] = 1.0 + PRETREND_SLOPE_PER_DAY * np.arange(D)[:, None]

    exp_s, exp_e = idx_of(EXPERIMENT_START), idx_of(EXPERIMENT_END)
    cf_end = min(exp_e + CF_TAIL_DAYS, D - 1)
    ref_s, ref_e = idx_of(REF_START), idx_of(REF_END)
    ACT = active_streams(ctx)
    K, S, SC = np.ones(N_STREAMS), np.ones(N_STREAMS), np.zeros(N_STREAMS)
    for st in ACT:
        q = cal[STREAM_NAMES[st]]
        K[st], S[st], SC[st] = q["K"], q["slope"], q["scale"]
    streams_of = [[st for st in (2 * c, 2 * c + 1) if st in ACT] for c in range(len(CHANNELS))]
    im_norm = 1.0 + (IN_MARKET_MULT - 1.0) * IN_MARKET_STATIONARY

    # ---- state
    ad = np.zeros((n, N_STREAMS))
    ad_cf = None
    pc = np.zeros(n, dtype=np.int16)
    last_d = np.full(n, -10_000, dtype=np.int32)       # day index of last purchase (v5 cooldown)
    inmarket = rng_s.random(n) < IN_MARKET_STATIONARY

    # ---- outputs
    t_day, t_cust, t_rev, t_disc, t_sub, t_canc = [], [], [], [], [], []
    tp_day, tp_cust, tp_camp = [], [], []
    org_day, org_cust, org_area = [], [], []
    acc = {k: np.zeros((D, N_STREAMS)) for k in ("inc_purch_s", "inc_cm_s", "marg_purch_s", "marg_cm_s")}
    acc.update({k: np.zeros((D, A)) for k in ("exp_purch", "exp_purch_nomkt", "exp_cm", "cf_purch", "cf_cm")})
    curve_purch = np.zeros((len(CHANNELS), len(CURVE_MULTIPLIERS)))
    curve_cm = np.zeros_like(curve_purch)
    curve_days, ref_lift_sum, ref_lift_n = 0, 0.0, 0
    n_purch = n_purch_im = 0

    def lifts(ad_state, act, ar_act, mult=1.0):
        out = np.zeros((N_STREAMS, len(act)))
        for st in ACT:
            x = ad_state[act, st] * pop.channel_resp[act, st // 2] * mult
            out[st] = SC[st] * ar_act * hill(x, K[st], S[st])
        return out

    for d in range(D):
        inmarket = evolve_inmarket(inmarket, rng_s)
        counts, c_hit, m_hit, oc_hit, oa_hit = allocate_day(ctx, d, n, rng_a, inmarket,
                                                            pop.browse_intensity, capture=True)
        if len(c_hit):
            tp_day.append(np.full(len(c_hit), d, dtype=np.int32))
            tp_cust.append(c_hit.astype(np.int32))
            tp_camp.append(m_hit.astype(np.int32))
        if len(oc_hit):
            org_day.append(np.full(len(oc_hit), d, dtype=np.int32))
            org_cust.append(oc_hit.astype(np.int32))
            org_area.append(oa_hit)

        act = np.flatnonzero(pop.entry_ns <= DATES_NS[d])
        if len(act):
            area_a = pop.area_idx[act]
            ar_act = area_resp[area_a]
            L = lifts(ad, act, ar_act)                     # exposure through d-1 (1-day lag)
            L_tot = L.sum(axis=0)
            repeat_mult = np.where(pc[act] > 0, 1.0 + REPEAT_LIFT * pop.repeat_tendency[act], 1.0)
            # v5 cooldown: right after a booking the chance is ~0 and recovers over ~COOLDOWN_DAYS
            # (customers who never bought have a huge 'days since' and so are unaffected)
            cooldown = 1.0 - np.exp(-(d - last_d[act]) / COOLDOWN_DAYS)
            im_f = (1.0 + (IN_MARKET_MULT - 1.0) * inmarket[act]) / im_norm
            P0 = (BASE_DAILY_PURCHASE_PROB * prop_norm[act] * a_prop_norm[area_a] * season[d, area_a]
                  * trend[d, area_a] * DEMAND_EVENT_D[d] * repeat_mult * cooldown * im_f)
            p = np.minimum(P0 * (1.0 + L_tot), MAX_DAILY_PURCHASE_PROB)
            cm = (cm_fest if FESTIVAL_DAY[d] else cm_norm)[act]

            inc = P0[None, :] * L
            acc["inc_purch_s"][d] = inc.sum(axis=1)
            acc["inc_cm_s"][d] = (inc * cm[None, :]).sum(axis=1)
            marg = P0[None, :] * (lifts(ad, act, ar_act, 1.0 + MARGINAL_EXPOSURE_STEP) - L)
            acc["marg_purch_s"][d] = marg.sum(axis=1)
            acc["marg_cm_s"][d] = (marg * cm[None, :]).sum(axis=1)
            acc["exp_purch"][d] = np.bincount(area_a, weights=p, minlength=A)
            acc["exp_purch_nomkt"][d] = np.bincount(area_a, weights=P0, minlength=A)
            acc["exp_cm"][d] = np.bincount(area_a, weights=p * cm, minlength=A)

            if exp_s <= d <= cf_end:                       # counterfactual: no extra 25% spend
                if d == exp_s:
                    ad_cf = ad.copy()
                diff = P0 * (L_tot - lifts(ad_cf, act, ar_act).sum(axis=0))
                acc["cf_purch"][d] = np.bincount(area_a, weights=diff, minlength=A)
                acc["cf_cm"][d] = np.bincount(area_a, weights=diff * cm, minlength=A)

            if ref_s <= d <= ref_e and (d - ref_s) % 3 == 0:   # spend-response curves by channel
                curve_days += 1
                for c in range(len(CHANNELS)):
                    for j, m in enumerate(CURVE_MULTIPLIERS):
                        if m <= 0:
                            continue
                        lm = sum(SC[st] * ar_act * hill(ad[act, st] * pop.channel_resp[act, c] * m, K[st], S[st])
                                 for st in streams_of[c])
                        w = P0 * lm
                        curve_purch[c, j] += w.sum()
                        curve_cm[c, j] += (w * cm).sum()
                ref_lift_sum += L_tot.sum()
                ref_lift_n += len(act)

            buy = rng_t.random(len(act)) < p
            idx = act[buy]
            if len(idx):
                m_ = len(idx)
                rev = rev_scale[idx] * ln1(rng_t, REVENUE_NOISE_SIGMA, m_)
                fd = FESTIVAL_DISCOUNT_SHIFT if FESTIVAL_DAY[d] else 0.0
                fs = FESTIVAL_SUBSIDY_SHIFT if FESTIVAL_DAY[d] else 0.0
                dr = np.clip((rng_t.beta(DISC_A, DISC_B, m_) + fd) * rate_fac[idx], 0, RATE_CAP)
                sr = np.clip((rng_t.beta(SUBS_A, SUBS_B, m_) + fs) * rate_fac[idx], 0, RATE_CAP)
                t_day.append(np.full(m_, d, dtype=np.int32))
                t_cust.append(idx.astype(np.int32))
                t_rev.append(np.round(rev, 2))
                t_disc.append(np.round(rev * dr, 2))
                t_sub.append(np.round(rev * sr, 2))
                t_canc.append(rng_t.random(m_) < p_cancel[idx])
                pc[idx] += 1
                last_d[idx] = d
                n_purch += m_
                n_purch_im += int(inmarket[idx].sum())

        ad = ad * DECAYS_S + counts                       # update AFTER purchase draw (touch precedes purchase)
        if ad_cf is not None and d >= exp_s and d <= cf_end:
            counts_cf = counts.copy()
            if d <= exp_e:
                counts_cf[treat_cust] /= TREATMENT_SPEND_MULTIPLIER
            ad_cf = ad_cf * DECAYS_S + counts_cf

    # ---- assemble tables
    day = np.concatenate(t_day)
    txn = pd.DataFrame({
        "customer_id": pop.customer_id[np.concatenate(t_cust)],
        "transaction_date": ALL_DATES[day],
        "revenue": np.concatenate(t_rev), "subsidy": np.concatenate(t_sub),
        "discount": np.concatenate(t_disc), "cancelled": np.concatenate(t_canc),
    })
    txn.insert(0, "transaction_id", [f"TXN_{i:08d}" for i in range(1, len(txn) + 1)])

    td, tc = np.concatenate(tp_day), np.concatenate(tp_camp)
    paid = pd.DataFrame({
        "customer_id": pop.customer_id[np.concatenate(tp_cust)],
        "touchpoint_timestamp": ALL_DATES[td] + pd.to_timedelta(rng_a.integers(0, 86_400, len(td)), unit="s"),
        "channel": camp["channel"].to_numpy()[tc],
        "campaign_id": camp["campaign_id"].to_numpy()[tc],
        "campaign_name": camp["campaign_name"].to_numpy()[tc],
    })
    od = np.concatenate(org_day)
    organic = pd.DataFrame({
        "customer_id": pop.customer_id[np.concatenate(org_cust)],
        "touchpoint_timestamp": ALL_DATES[od] + pd.to_timedelta(rng_a.integers(0, 86_400, len(od)), unit="s"),
        "channel": "Organic", "campaign_id": "ORGANIC",
        "campaign_name": "Organic_Direct_" + area_ids_arr[np.concatenate(org_area)],
    })
    tp = pd.concat([paid, organic], ignore_index=True).sort_values("touchpoint_timestamp", kind="stable") \
        .reset_index(drop=True)
    tp.insert(0, "touchpoint_id", [f"TP_{i:09d}" for i in range(1, len(tp) + 1)])

    acc.update(curve_purch=curve_purch, curve_cm=curve_cm, curve_days=curve_days,
               mean_total_lift_ref=ref_lift_sum / max(ref_lift_n, 1),
               inmarket_share_of_purchases=n_purch_im / max(n_purch, 1),
               cm_norm=cm_norm, rev_scale=rev_scale, prop_norm=prop_norm,
               a_prop_norm=a_prop_norm, area_resp=area_resp)
    return txn, tp, acc


# ============================================================
# 9. Simulation truth (counterfactual, not re-derived)
# ============================================================

def generate_simulation_truth(pop, areas, chars, assignment, camp, cal, acc, spend_mat, acq):
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    for k in ("inc_purch", "inc_cm", "marg_purch", "marg_cm"):         # streams -> channels
        acc[k] = acc[k + "_s"][:, 0::2] + acc[k + "_s"][:, 1::2]
    chan_idx_c = camp["channel"].map(CH_IDX).to_numpy()
    spend_ch = np.stack([spend_mat[:, chan_idx_c == c].sum(axis=1) for c in range(4)], axis=1)

    hist_e, exp_s, exp_e = idx_of(HISTORICAL_END), idx_of(EXPERIMENT_START), idx_of(EXPERIMENT_END)
    periods = {"history": slice(0, hist_e + 1), "experiment": slice(exp_s, exp_e + 1),
               "post_experiment": slice(exp_e + 1, N_DAYS), "all": slice(0, N_DAYS)}

    # 1. channel incrementality
    rows = []
    tot_purch = acc["exp_purch"].sum(axis=1)
    for pname, sl in periods.items():
        for c, ch in enumerate(CHANNELS):
            sp = spend_ch[sl, c].sum()
            inc_cm, inc_p = acc["inc_cm"][sl, c].sum(), acc["inc_purch"][sl, c].sum()
            marg_cm = acc["marg_cm"][sl, c].sum()
            rows.append(dict(
                period=pname, channel=ch, spend=sp,
                incremental_purchases=inc_p, incremental_contribution_margin=inc_cm,
                share_of_all_purchases=inc_p / tot_purch[sl].sum(),
                avg_iroas_cm=inc_cm / sp,
                marginal_iroas_cm_plus10pct=marg_cm / (MARGINAL_EXPOSURE_STEP * sp),
                marginal_to_avg_ratio=(marg_cm / (MARGINAL_EXPOSURE_STEP * sp)) / (inc_cm / sp)))
    chan_truth = pd.DataFrame(rows)

    # 1b. campaign-class (stream) incrementality: intent vs discovery within each channel
    stream_c = camp["stream"].to_numpy()
    rows_s = []
    for pname in ("history", "experiment", "all"):
        sl = periods[pname]
        for st in range(N_STREAMS):
            if not (stream_c == st).any():
                continue
            sp = spend_mat[sl][:, stream_c == st].sum()
            inc_cm, inc_p = acc["inc_cm_s"][sl, st].sum(), acc["inc_purch_s"][sl, st].sum()
            mg = acc["marg_cm_s"][sl, st].sum()
            q = cal[STREAM_NAMES[st]]
            rows_s.append(dict(
                period=pname, channel=CHANNELS[st // 2], campaign_class=q["campaign_class"], spend=sp,
                incremental_purchases=inc_p, incremental_contribution_margin=inc_cm,
                share_of_all_purchases=inc_p / tot_purch[sl].sum(), avg_iroas_cm=inc_cm / sp,
                marginal_iroas_cm_plus10pct=mg / (MARGINAL_EXPOSURE_STEP * sp),
                hill_K=q["K"], hill_slope=q["slope"], hill_scale=q["scale"], adstock_decay=q["decay"]))
    stream_truth = pd.DataFrame(rows_s)

    # 2. response curves in spend units (reference window, per day)
    ref_spend = spend_ch[idx_of(REF_START): idx_of(REF_END) + 1].mean(axis=0)
    cur = []
    for c, ch in enumerate(CHANNELS):
        spend_axis = np.array(CURVE_MULTIPLIERS) * ref_spend[c]
        inc_cm_pd = acc["curve_cm"][c] / acc["curve_days"]
        inc_p_pd = acc["curve_purch"][c] / acc["curve_days"]
        marginal = np.gradient(inc_cm_pd, spend_axis)
        for j, m in enumerate(CURVE_MULTIPLIERS):
            cur.append(dict(channel=ch, spend_multiplier=m, spend_per_day=spend_axis[j],
                            incremental_purchases_per_day=inc_p_pd[j],
                            incremental_cm_per_day=inc_cm_pd[j],
                            avg_cm_per_dollar=(inc_cm_pd[j] / spend_axis[j]) if m > 0 else np.nan,
                            marginal_cm_per_dollar=marginal[j]))
    curves = pd.DataFrame(cur)

    # 3. area effect (experiment treatment effect by counterfactual)
    win = slice(exp_s, exp_e + 1)
    tail = slice(exp_s, min(exp_e + CF_TAIL_DAYS, N_DAYS - 1) + 1)
    exp_p = acc["exp_purch"][win].sum(axis=0)
    eff_p, eff_cm = acc["cf_purch"][win].sum(axis=0), acc["cf_cm"][win].sum(axis=0)
    exp_cm = acc["exp_cm"][win].sum(axis=0)
    area_eff = assignment.merge(chars, on="area_id").assign(
        expected_purchases_experiment=exp_p,
        true_effect_purchases=eff_p,
        true_effect_purchases_pct=eff_p / np.maximum(exp_p - eff_p, 1e-9),
        true_effect_cm=eff_cm,
        true_effect_cm_pct=eff_cm / np.maximum(exp_cm - eff_cm, 1e-9),
        true_effect_purchases_incl_carryover=acc["cf_purch"][tail].sum(axis=0),
        true_effect_cm_incl_carryover=acc["cf_cm"][tail].sum(axis=0),
        organic_purchases_share_experiment=acc["exp_purch_nomkt"][win].sum(axis=0) / np.maximum(exp_p, 1e-9),
    )

    # 4. customer value (analytic, fresh-customer 365-day horizon, no realised noise)
    T = 365.0
    lam_base = (BASE_DAILY_PURCHASE_PROB * acc["prop_norm"] * acc["a_prop_norm"][pop.area_idx])
    mark_lift = acc["mean_total_lift_ref"] * acc["area_resp"][pop.area_idx]

    def expected_purchases(lam0):
        # first purchase at rate lam0; afterwards the rate is lam1, but each booking is followed by a
        # cooldown that costs about COOLDOWN_DAYS of exposure, so the effective repeat rate is
        # lam1 / (1 + lam1 * COOLDOWN_DAYS)
        lam1 = lam0 * (1.0 + REPEAT_LIFT * pop.repeat_tendency)
        lam_eff = lam1 / (1.0 + lam1 * COOLDOWN_DAYS)
        first = 1.0 - np.exp(-lam0 * T)
        return first + lam_eff * (T - first / lam0)

    e_no = expected_purchases(lam_base)
    e_mk = expected_purchases(lam_base * (1.0 + mark_lift))
    area_ids = chars["area_id"].to_numpy()
    cust_value = pd.DataFrame({
        "customer_id": pop.customer_id, "area_id": area_ids[pop.area_idx],
        "entry_date": pd.to_datetime(pop.entry_ns),
        "segment": np.array(SEGMENT_NAMES)[pop.segment],
        "purchase_propensity": pop.purchase_propensity, "aov_factor": pop.aov_factor,
        "browse_intensity": pop.browse_intensity,
        "repeat_purchase_tendency": pop.repeat_tendency, "price_sensitivity": pop.price_sensitivity,
        **{f"responsiveness_{ch}": pop.channel_resp[:, c] for c, ch in enumerate(CHANNELS)},
        "expected_cm_per_purchase": acc["cm_norm"],
        "expected_365d_purchases_no_marketing": e_no,
        "expected_365d_cm_no_marketing": e_no * acc["cm_norm"],
        "expected_365d_purchases_with_avg_marketing": e_mk,
        "expected_365d_cm_with_avg_marketing": e_mk * acc["cm_norm"],
    }).sort_values("customer_id").reset_index(drop=True)

    # 3b. v5: acquisition effect of the +25% spend (new customers arrive faster when spend is higher).
    # Kept separate from the purchase-lift truth so channel iROAS stays purely about purchases.
    extra = (acq["lam"] - acq["lam_cf"])[exp_s: exp_e + 8].sum(axis=0)     # spend affects arrivals for 7 more days
    new_c = cust_value[cust_value["entry_date"] >= HISTORICAL_START]
    cm_per_new = new_c.groupby("area_id")["expected_365d_cm_with_avg_marketing"].mean()
    area_eff["extra_customers_from_spend_increase"] = extra
    area_eff["expected_365d_cm_per_new_customer"] = area_eff["area_id"].map(cm_per_new).to_numpy()
    area_eff["true_effect_acquisition_cm_365d"] = (area_eff["extra_customers_from_spend_increase"]
                                                   * area_eff["expected_365d_cm_per_new_customer"])

    chan_truth.to_csv(TRUTH_DIR / "true_channel_incrementality.csv", index=False)
    stream_truth.to_csv(TRUTH_DIR / "true_campaign_class_incrementality.csv", index=False)
    curves.to_csv(TRUTH_DIR / "true_response_curves.csv", index=False)
    area_eff.to_csv(TRUTH_DIR / "true_area_effect.csv", index=False)
    cust_value.to_csv(TRUTH_DIR / "true_customer_value.csv", index=False)
    with open(TRUTH_DIR / "simulation_parameters.json", "w") as fh:
        json.dump({"calibrated_channel_parameters": cal,
                   "config": {"seed": MASTER_SEED, "base_daily_spend": BASE_DAILY_SPEND,
                              "base_daily_purchase_prob": BASE_DAILY_PURCHASE_PROB,
                              "cooldown_days": COOLDOWN_DAYS, "repeat_lift": REPEAT_LIFT,
                              "segment_shares": SEGMENT_SHARES, "segment_mult": SEGMENT_MULT,
                              "aov_mean": AOV_MEAN, "touchpoint_capture_rate": TOUCHPOINT_CAPTURE_RATE,
                              "acquisition_spend_elasticity": ACQUISITION_SPEND_ELASTICITY,
                              "treatment_spend_multiplier": TREATMENT_SPEND_MULTIPLIER,
                              "customer_count": int(pop.n), "variant": VARIANT,
                              "inmarket_share_of_purchases": acc["inmarket_share_of_purchases"]}}, fh, indent=2)
    return dict(channel=chan_truth, area=area_eff, curves=curves, stream=stream_truth,
                inmarket_share=acc["inmarket_share_of_purchases"])


# ============================================================
# 10. Controlled imperfections (raw layer)
# ============================================================

def apply_controlled_imperfections(tables):
    """Raw-layer data-quality issues, each logged in imperfection_log.json so cleaning can be audited:
    null area_size; channel-dependent touchpoint loss; missing UTM; customer_id drift; TikTok UTC
    timestamps; duplicate touchpoints / transactions; late ETL (`loaded_at`, an audit column outside
    the locked transactions schema); platform-vs-warehouse spend and click discrepancies."""
    rng = RNG["imperfections"]
    raw = {k: v.copy() for k, v in tables.items()}
    log = {}

    null_idx = rng.choice(len(raw["areas"]), size=N_NULL_AREA_SIZES, replace=False)
    raw["areas"].loc[null_idx, "area_size"] = pd.NA
    log["area_size_nulls"] = int(len(null_idx))

    tp = raw["marketing_touchpoints"]
    n0 = len(tp)
    loss = tp["channel"].map(TP_LOSS_RATES).fillna(0.05).to_numpy()
    tp = tp[rng.random(n0) >= loss].reset_index(drop=True)
    log["touchpoints_dropped"] = int(n0 - len(tp))

    paid = np.flatnonzero((tp["channel"] != "Organic").to_numpy())
    miss = rng.choice(paid, size=int(MISSING_UTM_RATE * len(paid)), replace=False)
    tp["campaign_id"] = tp["campaign_id"].astype(object)
    tp["campaign_name"] = tp["campaign_name"].astype(object)
    tp.loc[tp.index[miss], ["campaign_id", "campaign_name"]] = None
    log["touchpoints_missing_utm"] = int(len(miss))

    cust = tp["customer_id"].astype(object).to_numpy().copy()
    drift = rng.choice(len(tp), size=int(ID_DRIFT_RATE * len(tp)), replace=False)
    half = len(drift) // 2
    cust[drift[:half]] = [c.lower() for c in cust[drift[:half]]]
    cust[drift[half:]] = [c + " " for c in cust[drift[half:]]]
    tp["customer_id"] = cust
    log["touchpoint_customer_id_drift"] = int(len(drift))

    tt = (tp["channel"] == "TikTok").to_numpy()
    tp.loc[tt, "touchpoint_timestamp"] = tp.loc[tt, "touchpoint_timestamp"] + pd.Timedelta(hours=TIKTOK_UTC_OFFSET_HOURS)
    log["tiktok_rows_shifted_to_utc"] = int(tt.sum())

    dup = rng.choice(len(tp), size=int(DUP_TP_RATE * len(tp)), replace=False)
    tp = pd.concat([tp, tp.iloc[dup]], ignore_index=True)
    log["touchpoints_duplicated"] = int(len(dup))
    raw["marketing_touchpoints"] = tp

    txn = raw["transactions"]
    load = txn["transaction_date"] + pd.Timedelta(days=1) + pd.to_timedelta(rng.integers(2, 7, len(txn)), unit="h")
    late = rng.random(len(txn)) < LATE_ETL_RATE
    txn["loaded_at"] = load + pd.to_timedelta(np.where(late, rng.integers(24, 72, len(txn)), 0), unit="h")
    log["transactions_late_arriving"] = int(late.sum())
    dup_t = rng.choice(len(txn), size=int(DUP_TXN_RATE * len(txn)), replace=False)
    dups = txn.iloc[dup_t].copy()
    dups["loaded_at"] = dups["loaded_at"] + pd.Timedelta(hours=6)
    raw["transactions"] = pd.concat([txn, dups], ignore_index=True)
    log["transactions_duplicated"] = int(len(dup_t))

    mp = raw["marketing_performance"]
    sb = mp["channel"].map(PLATFORM_SPEND_BIAS).to_numpy() * ln1(rng, 0.01, len(mp))
    cb = mp["channel"].map(PLATFORM_CLICK_BIAS).to_numpy() * ln1(rng, 0.01, len(mp))
    mp["spend"] = np.round(mp["spend"].to_numpy() * sb, 2)
    mp["clicks"] = np.rint(mp["clicks"].to_numpy() * cb).astype(np.int64)
    raw["marketing_performance"] = mp
    log["platform_spend_bias"] = PLATFORM_SPEND_BIAS
    log["platform_click_bias"] = PLATFORM_CLICK_BIAS

    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRUTH_DIR / "imperfection_log.json", "w") as fh:
        json.dump(log, fh, indent=2)
    return raw


def write_tables(tables, folder: Path, compress=False):
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if compress:
            df.to_csv(folder / f"{name}.csv.gz", index=False, compression="gzip")
        else:
            df.to_csv(folder / f"{name}.csv", index=False)


# ============================================================
# 11. Validation (Phase 8) - assertions, not just prints
# ============================================================

def _ok(msg):
    print(f"  ✓ {msg}")


def behavior_metrics(txn, pop):
    """Purchase behaviour among customers observed for at least BEHAVIOR_MIN_HISTORY_DAYS.

    Late joiners have had less time to buy, so including them would understate repeat behaviour."""
    cutoff = POST_EXPERIMENT_END - pd.Timedelta(days=BEHAVIOR_MIN_HISTORY_DAYS)
    entry = pd.Series(pd.to_datetime(pop.entry_ns), index=pop.customer_id)
    mature_ids = entry.index[(entry <= cutoff).to_numpy()]
    n_per = txn.groupby("customer_id").size().reindex(mature_ids, fill_value=0)
    buyers = n_per[n_per > 0]
    t = txn[txn["customer_id"].isin(mature_ids)].sort_values(["customer_id", "transaction_date"])
    gaps = t.groupby("customer_id")["transaction_date"].diff().dt.days.dropna()
    return dict(n_mature=int(len(mature_ids)), buyer_share=float(len(buyers) / len(mature_ids)),
                purchases_per_buyer=float(buyers.mean()), repeat_share=float((buyers > 1).mean()),
                median_gap_days=float(gaps.median()) if len(gaps) else float("nan"))


def validate_clean_world(t, truth, cal, camp, pop):
    print("\n" + "=" * 68 + "\nPHASE 8 - VALIDATION OF CLEAN WORLD\n" + "=" * 68)
    areas, cust, mp, tp, txn, exp = (t[k] for k in
        ["areas", "customers", "marketing_performance", "marketing_touchpoints", "transactions",
         "experiment_assignment"])

    for k, df in t.items():
        assert len(df) > 0, f"{k} empty"
    _ok("all six tables populated: " + ", ".join(f"{k}={len(v):,}" for k, v in t.items()))

    assert areas["area_id"].is_unique and cust["customer_id"].is_unique
    assert txn["transaction_id"].is_unique and tp["touchpoint_id"].is_unique
    assert not mp.duplicated(["date", "campaign_id"]).any()
    assert not exp.duplicated(["experiment_id", "area_id"]).any()
    _ok("primary / composite keys unique")

    mp_area = extract_area(mp["campaign_name"])
    tp_area = extract_area(tp["campaign_name"])
    assert cust["area_id"].isin(areas["area_id"]).all() and exp["area_id"].isin(areas["area_id"]).all()
    assert txn["customer_id"].isin(cust["customer_id"]).all() and tp["customer_id"].isin(cust["customer_id"]).all()
    assert mp_area.isin(areas["area_id"]).all() and tp_area.isin(areas["area_id"]).all()
    paid_tp = tp["channel"] != "Organic"
    assert tp.loc[paid_tp, "campaign_id"].isin(mp["campaign_id"]).all()
    tp_cust_area = tp["customer_id"].map(cust.set_index("customer_id")["area_id"])
    assert (tp_cust_area == tp_area).all(), "touchpoint campaign area != customer area"
    _ok("foreign keys join; area parsed from campaign_name; touchpoint area == customer area")

    assert mp["date"].min() == HISTORICAL_START and mp["date"].max() == POST_EXPERIMENT_END
    assert txn["transaction_date"].between(HISTORICAL_START, POST_EXPERIMENT_END).all()
    assert tp["touchpoint_timestamp"].between(HISTORICAL_START, POST_EXPERIMENT_END + pd.Timedelta(days=1)).all()
    assert exp["assignment_date"].max() < EXPERIMENT_START
    _ok("date ranges correct; assignment precedes experiment")

    assert set(mp["channel"]) == set(CHANNELS) and set(tp["channel"]) == set(CHANNELS) | {"Organic"}
    _ok("all four paid channels present in performance and touchpoints; Organic present in touchpoints")

    g = exp["experiment_group"].value_counts()
    assert g["treatment"] == N_TREATMENT_AREAS and g["control"] == N_CONTROL_AREAS and len(exp) == N_AREAS
    _ok("assignment: 10 treatment / 10 control, one row per area")

    m = mp.assign(area_id=mp_area).merge(exp[["area_id", "experiment_group"]], on="area_id")
    pre = m[(m["date"] >= EXPERIMENT_START - pd.Timedelta(days=56)) & (m["date"] < EXPERIMENT_START)]
    ex = m[(m["date"] >= EXPERIMENT_START) & (m["date"] <= EXPERIMENT_END)]

    def daily(df, grp, ch=None):
        d = df[df["experiment_group"] == grp]
        d = d if ch is None else d[d["channel"] == ch]
        return d["spend"].sum() / d["date"].nunique()

    ratio = (daily(ex, "treatment") / daily(pre, "treatment")) / (daily(ex, "control") / daily(pre, "control"))
    assert 1.17 <= ratio <= 1.33, f"treatment spend ratio {ratio:.3f}"
    for ch in CHANNELS:
        r = (daily(ex, "treatment", ch) / daily(pre, "treatment", ch)) / (daily(ex, "control", ch) / daily(pre, "control", ch))
        assert 1.05 <= r <= 1.45, f"{ch} ratio {r:.3f}"
    tmix = ex[ex["experiment_group"] == "treatment"].groupby("channel")["spend"].sum()
    tmix = tmix / tmix.sum()
    assert all(abs(tmix[c] - CHANNEL_MIX[c]) < 0.03 for c in CHANNELS)
    _ok(f"treatment receives +25% spend (DiD ratio {ratio:.3f}); treatment mix "
        + str({c: round(float(tmix[c]), 3) for c in CHANNELS}))

    mix = mp.groupby("channel")["spend"].sum()
    mix = mix / mix.sum()
    assert all(abs(mix[c] - CHANNEL_MIX[c]) < 0.01 for c in CHANNELS), mix.to_dict()
    spend_2025 = mp.loc[mp["date"].dt.year == 2025, "spend"].sum()
    assert 9.0e6 <= spend_2025 <= 11.0e6, spend_2025
    _ok(f"4:3:2:1 mix holds ({ {c: round(float(mix[c]), 3) for c in CHANNELS} }); 2025 spend ${spend_2025 / 1e6:.2f}M")

    # ---- journeys: coverage, and the attribution-vs-incrementality gap
    codes = pd.Series(np.arange(len(cust)), index=cust["customer_id"])
    BIG = 10 ** 9
    tp_sec = ((tp["touchpoint_timestamp"] - HISTORICAL_START).dt.total_seconds()).to_numpy().astype(np.int64)
    raw_keys = codes.loc[tp["customer_id"]].to_numpy().astype(np.int64) * BIG + tp_sec
    order = np.argsort(raw_keys, kind="stable")
    tp_keys = raw_keys[order]
    cls_map = dict(zip(camp["campaign_id"], camp["campaign_class"]))
    tp_class = tp["campaign_id"].map(cls_map).fillna("organic").to_numpy()[order]
    hi = (codes.loc[txn["customer_id"]].to_numpy().astype(np.int64) * BIG
          + (txn["transaction_date"] - HISTORICAL_START).dt.days.to_numpy().astype(np.int64) * 86_400)
    lo = hi - ATTRIBUTION_WINDOW_DAYS * 86_400
    n_prior = np.searchsorted(tp_keys, hi, "left") - np.searchsorted(tp_keys, lo, "left")
    share_no_tp, share_multi = float((n_prior == 0).mean()), float((n_prior >= 2).mean())
    assert 0.05 < share_no_tp < 0.98 and share_multi > 0.005
    tp_per_cust = tp.groupby("customer_id").size()
    txn_per_cust = txn.groupby("customer_id").size()
    _ok(f"purchases with no logged touchpoint in prior 14d: {share_no_tp:.1%}; 2+ touchpoints: {share_multi:.1%}")

    # ---- v5 behaviour targets (customers observed for 12+ months)
    bm = behavior_metrics(txn, pop)
    buyers_all = len(txn_per_cust) / len(cust)
    assert 0.15 < buyers_all < 0.65, buyers_all
    assert TARGET_PURCHASES_PER_BUYER[0] <= bm["purchases_per_buyer"] <= TARGET_PURCHASES_PER_BUYER[1], bm
    assert TARGET_REPEAT_SHARE[0] <= bm["repeat_share"] <= TARGET_REPEAT_SHARE[1], bm
    assert TARGET_MEDIAN_GAP_DAYS[0] <= bm["median_gap_days"] <= TARGET_MEDIAN_GAP_DAYS[1], bm
    _ok(f"BEHAVIOUR (customers observed 12+ months, n={bm['n_mature']:,}): purchases per buyer "
        f"{bm['purchases_per_buyer']:.2f}; repeat buyers {bm['repeat_share']:.1%}; median days between "
        f"purchases {bm['median_gap_days']:.0f}; buyers {bm['buyer_share']:.1%} of customers")
    print(f"     all customers: customers with touchpoints {len(tp_per_cust) / len(cust):.1%}; buyers "
          f"{buyers_all:.1%}; purchases/buyer {len(txn) / len(txn_per_cust):.2f} "
          f"(lower than the 12+ month figure because many customers joined late)")

    pos = np.searchsorted(tp_keys, hi, "left") - 1
    valid = (pos >= 0) & (tp_keys[np.maximum(pos, 0)] >= lo)
    last = pd.Series(np.where(valid, tp_class[np.maximum(pos, 0)], "none")).value_counts(normalize=True)
    st_h = truth["stream"].query("period == 'history'")
    inc_intent = float(st_h.loc[st_h["campaign_class"] == "intent", "share_of_all_purchases"].sum())
    inc_disc = float(st_h.loc[st_h["campaign_class"] == "discovery", "share_of_all_purchases"].sum())
    lt_intent, lt_disc = float(last.get("intent", 0)), float(last.get("discovery", 0))
    assert lt_intent > 2.0 * inc_intent, (lt_intent, inc_intent)
    assert 0.20 <= truth["inmarket_share"] <= 0.60, truth["inmarket_share"]
    _ok(f"ATTRIBUTION GAP: intent campaigns get {lt_intent:.1%} of last-touch credit but cause only "
        f"{inc_intent:.1%} of purchases; discovery gets {lt_disc:.1%} vs {inc_disc:.1%} causal; "
        f"organic last-touch {float(last.get('organic', 0)):.1%}; in-market customers = "
        f"{truth['inmarket_share']:.0%} of purchases")

    assert (txn[["revenue", "subsidy", "discount"]] >= 0).all().all() and (txn["revenue"] > 0).all()
    ok = ~txn["cancelled"]
    cmr = (txn.loc[ok, "revenue"] - txn.loc[ok, "subsidy"] - txn.loc[ok, "discount"]).sum() / txn.loc[ok, "revenue"].sum()
    neg = float(((txn["revenue"] - txn["subsidy"] - txn["discount"]) < 0).mean())
    assert 0.30 <= cmr <= 0.70 and neg < 0.02
    _ok(f"contribution margin = {cmr:.1%} of revenue (non-cancelled); negative-CM bookings {neg:.2%}; "
        f"cancellation rate {txn['cancelled'].mean():.1%}")

    per_area = txn.assign(area_id=txn["customer_id"].map(cust.set_index("customer_id")["area_id"])) \
        .groupby("area_id").agg(n=("revenue", "size"), aov=("revenue", "mean"))
    ppc = per_area["n"] / cust.groupby("area_id").size()
    assert ppc.std() / ppc.mean() > 0.05 and per_area["aov"].std() / per_area["aov"].mean() > 0.05
    _ok(f"area heterogeneity: purchases/customer CV {ppc.std() / ppc.mean():.2f}, AOV CV "
        f"{per_area['aov'].std() / per_area['aov'].mean():.2f}")

    daily_tx = txn.groupby("transaction_date").size()
    fest = daily_tx[(daily_tx.index.month == 10) & (daily_tx.index.day.isin(range(5, 13))) & (daily_tx.index.year == 2025)].mean()
    ref = daily_tx[(daily_tx.index >= "2025-09-20") & (daily_tx.index <= "2025-09-30")].mean()
    assert fest / ref > 1.15, fest / ref
    _ok(f"10.10 festival demand lift vs late September: {fest / ref:.2f}x")

    ch_all = truth["channel"].query("period == 'history'")
    share = ch_all["share_of_all_purchases"].sum()
    assert 0.15 <= share <= 0.45, share
    assert ((ch_all["marginal_to_avg_ratio"] > 0.10) & (ch_all["marginal_to_avg_ratio"] < 0.95)).all(), \
        ch_all[["channel", "marginal_to_avg_ratio"]]
    cv = truth["curves"]
    for ch, gg in cv[cv["spend_multiplier"] >= 0.25].groupby("channel"):
        mg = gg["marginal_cm_per_dollar"].to_numpy()
        assert (mg > 0).all() and (np.diff(mg) <= 1e-9).all(), f"{ch} response not concave: {mg}"
    ae = truth["area"]
    tr = ae[ae["experiment_group"] == "treatment"]
    arm = tr["expected_purchases_experiment"].sum()
    eff_pct = tr["true_effect_purchases"].sum() / (arm - tr["true_effect_purchases"].sum())
    z = tr["true_effect_purchases"].sum() / np.sqrt(2 * arm)
    assert 0.015 <= eff_pct <= 0.08, eff_pct
    assert z >= 1.0, z
    assert (ae.loc[ae["experiment_group"] == "control", "true_effect_purchases"].abs() < 1e-6).all()
    _ok(f"marketing drives {share:.1%} of purchases; true +25%-spend effect {eff_pct:.2%} (Poisson z ~ {z:.1f}); "
        f"marginal/avg ratios " + str({r.channel: round(float(r.marginal_to_avg_ratio), 2) for r in ch_all.itertuples()}))
    cg = cust.merge(exp[["area_id", "experiment_group"]], on="area_id").set_index("customer_id")["experiment_group"]
    dd = txn.assign(g=txn["customer_id"].map(cg)).groupby(["transaction_date", "g"]).size().unstack()
    placebo = []
    for st in pd.date_range("2025-04-01", "2026-05-25", freq="28D"):      # experiment-like windows, no treatment
        ex_ = dd.loc[st: st + pd.Timedelta(days=27)].mean()
        pre_ = dd.loc[st - pd.Timedelta(days=57): st - pd.Timedelta(days=1)].mean()
        placebo.append((ex_["treatment"] / pre_["treatment"]) / (ex_["control"] / pre_["control"]) - 1)
    psd = float(np.std(placebo))
    assert eff_pct / psd > 0.5, (eff_pct, psd)
    print(f"     realistic power: naive DiD placebo SD {psd:.2%} vs true effect {eff_pct:.2%} "
          f"(effect/SD {eff_pct / psd:.1f}) -> a single naive DiD is noisy; covariate-adjusted estimators should do better")
    print("     channel iROAS (avg | marginal+10%): "
          + ", ".join(f"{r.channel} {r.avg_iroas_cm:.2f}|{r.marginal_iroas_cm_plus10pct:.2f}" for r in ch_all.itertuples()))
    print("     class iROAS   (avg | marginal+10%): "
          + ", ".join(f"{r.channel}/{r.campaign_class[:4]} {r.avg_iroas_cm:.2f}|{r.marginal_iroas_cm_plus10pct:.2f}"
                      for r in st_h.itertuples()))
    extra_c = float(tr["extra_customers_from_spend_increase"].sum())
    print(f"     acquisition effect of +25% spend: ~{extra_c:,.0f} extra customers "
          f"(about ${float(tr['true_effect_acquisition_cm_365d'].sum()):,.0f} expected 365-day CM; reported separately "
          f"from the purchase-lift truth)")

    total_rows = sum(len(v) for v in t.values())
    assert total_rows < 5_000_000
    est_mb = sum(v.memory_usage(deep=True).sum() for v in t.values()) / 1e6
    _ok(f"data size manageable: {total_rows:,} rows total, ~{est_mb:,.0f} MB in memory")


def validate_raw_exports(clean, raw):
    print("\nRAW-LAYER CHECKS (after imperfections)")
    assert raw["areas"]["area_size"].isna().sum() == N_NULL_AREA_SIZES
    tp, ctp = raw["marketing_touchpoints"], clean["marketing_touchpoints"]
    assert tp["touchpoint_id"].duplicated().sum() > 0 and raw["transactions"]["transaction_id"].duplicated().sum() > 0
    dedup = tp.drop_duplicates("touchpoint_id")
    kept = len(dedup) / len(ctp)
    assert 0.90 < kept < 0.97, kept
    assert raw["transactions"].drop_duplicates("transaction_id").shape[0] == len(clean["transactions"])
    unmatched = float((~dedup["customer_id"].isin(raw["customers"]["customer_id"])).mean())
    assert 0.005 < unmatched < 0.03, unmatched
    missing_utm = float(dedup.loc[dedup["channel"] != "Organic", "campaign_id"].isna().mean())
    assert 0.005 < missing_utm < 0.03, missing_utm
    txn = raw["transactions"]
    assert (txn["loaded_at"] > txn["transaction_date"]).all()
    late = float(((txn["loaded_at"] - txn["transaction_date"]) > pd.Timedelta(hours=48)).mean())
    assert 0.01 < late < 0.05, late
    m = clean["marketing_performance"].merge(raw["marketing_performance"], on=["date", "campaign_id"], suffixes=("_c", "_r"))
    sr = (m.groupby("channel_c")["spend_r"].sum() / m.groupby("channel_c")["spend_c"].sum()).round(3).to_dict()
    assert abs(sr["Meta"] - PLATFORM_SPEND_BIAS["Meta"]) < 0.01 and abs(sr["TikTok"] - PLATFORM_SPEND_BIAS["TikTok"]) < 0.01
    _ok(f"area_size nulls={N_NULL_AREA_SIZES}; touchpoints retained {kept:.1%}; customer_id drift {unmatched:.1%}; "
        f"missing UTM {missing_utm:.1%}; duplicates: {int(tp['touchpoint_id'].duplicated().sum()):,} touchpoints, "
        f"{int(txn['transaction_id'].duplicated().sum()):,} transactions; late ETL {late:.1%}")
    _ok(f"platform/warehouse spend ratio by channel: {sr}")


# ============================================================
# 12. Main
# ============================================================

def main():
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    log("Areas, assignment, campaigns")
    areas, chars = generate_areas()
    assignment = generate_experiment_assignment(chars)
    camp = generate_campaign_configuration(chars)

    log("Marketing performance")
    perf, clicks_mat, spend_mat = generate_marketing_performance(camp, chars, assignment)

    log("Customers (acquisition responds to spend)")
    pop, acq = generate_customers(chars, camp, spend_mat, assignment)
    log(f"  {pop.n:,} customers")
    ctx = build_click_context(pop, camp, chars, clicks_mat)

    log("Calibrating channel response curves")
    cal = calibrate_response(pop, ctx, chars)
    for ch, p in cal.items():
        print(f"     {ch:16s} K={p['K']:.4g} scale={p['scale']:.4f} target_lift={p['target_lift']:.4f} "
              f"exposed={p['share_exposed']:.0%} sat_exposed={p['achieved_saturation_among_exposed']:.2f}")

    log("Simulating exposure, touchpoints and transactions")
    txn, tp, acc = simulate_behavior(pop, ctx, camp, chars, assignment, cal, spend_mat)
    log(f"  {len(txn):,} transactions, {len(tp):,} touchpoints")

    log("Simulation truth")
    truth = generate_simulation_truth(pop, areas, chars, assignment, camp, cal, acc, spend_mat, acq)

    customers = pd.DataFrame({"customer_id": pop.customer_id,
                              "area_id": chars["area_id"].to_numpy()[pop.area_idx]}) \
        .sort_values("customer_id").reset_index(drop=True)
    clean = {"areas": areas, "experiment_assignment": assignment, "marketing_performance": perf,
             "customers": customers, "marketing_touchpoints": tp, "transactions": txn}

    log("Validation (clean world)")
    validate_clean_world(clean, truth, cal, camp, pop)

    log("Controlled imperfections + export")
    raw = apply_controlled_imperfections(clean)
    validate_raw_exports(clean, raw)
    write_tables(raw, RAW_DATA_DIR)
    if WRITE_CLEAN_REFERENCE:
        write_tables(clean, CLEAN_REF_DIR, compress=True)
    log(f"Done. Raw -> {RAW_DATA_DIR} | truth -> {TRUTH_DIR}")


if __name__ == "__main__":
    main()