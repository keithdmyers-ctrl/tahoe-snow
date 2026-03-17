#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Tahoe Snow ML
===============================================

Transforms raw SNOTEL + Open-Meteo data into training-ready feature matrices.

=== ML LEARNING NOTE: What is Feature Engineering? ===

Feature engineering is the process of creating INPUT VARIABLES (features)
that help your ML model learn patterns. It's often said that feature
engineering is where 80% of the value in ML comes from — a mediocre model
with great features will beat an advanced model with raw inputs.

Three types of features in our pipeline:

1. RAW FEATURES: Direct from APIs (temperature, precipitation, wind).
   These are your starting point but often not enough on their own.

2. DERIVED FEATURES: Physics-based calculations that encode domain
   knowledge (SLR, lapse rate, Froude number, orographic multiplier).
   These tell the model "here's how mountains affect weather" without
   making it learn basic atmospheric physics from scratch.

3. TEMPORAL FEATURES: Lagged values and trends (yesterday's snow depth,
   3-day pressure change, season of year). Weather is a time series — what
   happened yesterday strongly predicts today.

Think of features as "clues" you give the model. The better the clues,
the better the predictions. A human meteorologist uses all of these clues
when making a forecast — we're encoding that same knowledge numerically.

Usage:
    python feature_engineering.py                # Build all training data
    python feature_engineering.py --eda          # Run exploratory data analysis
    python feature_engineering.py --check        # Check feature completeness

Input:  data/historical/snotel_daily.parquet
        data/historical/era5_daily.parquet
        data/historical/archived_forecasts_daily.parquet (optional)

Output: data/historical/training_data.parquet
"""

import argparse
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tahoe_snow import SNOTEL_STATIONS
from resort_configs import RESORT_REGISTRY, get_active_resorts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historical")


# =========================================================================
# Station-to-Resort Mapping
# =========================================================================

def build_station_resort_map() -> dict:
    """
    Map each SNOTEL station to its nearest resort zone(s) for pairing
    forecast data with ground truth observations.

    === ML LEARNING NOTE: Label Alignment ===

    This is a critical step: we need to match each SNOTEL station's
    observations (our labels/targets) with the forecast data at the
    nearest resort zone (our features/inputs).

    The challenge: SNOTEL stations and resort zones aren't co-located.
    A SNOTEL station at 6,250 ft might be 5 miles from a resort whose
    base is at 6,540 ft. We need to decide which forecast point best
    represents each station's conditions.

    Our approach: use the resort_configs.py association (each resort
    lists its nearest SNOTEL stations). We trust the human-curated
    assignments since they account for topographic similarity, not
    just Euclidean distance.
    """
    station_to_resorts = {}
    snotel_id_to_name = {info["id"]: name for name, info in SNOTEL_STATIONS.items()}

    for resort_name, cfg in get_active_resorts().items():
        for sid in cfg.snotel_stations:
            station_name = snotel_id_to_name.get(sid)
            if station_name:
                if station_name not in station_to_resorts:
                    station_to_resorts[station_name] = []
                station_to_resorts[station_name].append({
                    "resort": resort_name,
                    "zones": cfg.zones,
                    "aspect": cfg.aspect,
                })

    return station_to_resorts


def find_nearest_zone(station_elev_ft: float, zones: list) -> str:
    """Find the zone (base/mid/peak) closest in elevation to the station."""
    best_zone = "base"
    best_diff = float("inf")
    for zone in zones:
        diff = abs(zone.elevation_ft - station_elev_ft)
        if diff < best_diff:
            best_diff = diff
            best_zone = zone.name
    return best_zone


# =========================================================================
# Derived Physics Features
#
# These delegate to tahoe_snow.py's authoritative implementations where
# possible, with thin adapters for signature differences. This eliminates
# train/serve skew: the same physics runs during training and production.
# =========================================================================

# Import authoritative physics from tahoe_snow.py
from tahoe_snow import (
    compute_slr as _ts_compute_slr,
    orographic_multiplier as _ts_orographic_multiplier,
)


def compute_slr(temp_f: float, wind_mph: float = 0, rh_pct: float = 80) -> float:
    """
    Compute Snow-to-Liquid Ratio (SLR). Delegates to tahoe_snow.py.

    === ML LEARNING NOTE: Domain Knowledge as Features ===

    SLR is how many inches of snow you get per inch of liquid water.
    Cold, calm conditions produce light fluffy snow (SLR ~15-20:1).
    Warm, windy conditions produce dense wet snow (SLR ~5-8:1).

    By computing SLR as a feature, we're telling the model "here's the
    physics relationship between temperature and snow density" instead of
    making it learn this from raw data. This is called "physics-informed
    feature engineering" and dramatically improves sample efficiency.

    The Tahoe region typically sees SLR between 8:1 (Sierra cement) and
    18:1 (cold storms from the north). This matters because a storm that
    drops 1" of liquid water could produce anywhere from 5" to 20" of snow.
    """
    # Delegate to tahoe_snow.py's authoritative Roebber (2003) implementation
    temp_c = (temp_f - 32) * 5 / 9
    return _ts_compute_slr(temp_c, wind_mph, rh_pct)


def compute_lapse_rate(temps_by_elev: list[tuple]) -> float:
    """
    Compute environmental lapse rate from temperature observations at
    multiple elevations.

    === ML LEARNING NOTE: Lapse Rate ===

    The lapse rate is how quickly temperature decreases with altitude.
    Standard atmosphere: ~6.5 C/km. But actual lapse rates vary:

    - Steep lapse rate (>7 C/km): Unstable atmosphere, good for
      convective enhancement of precipitation
    - Shallow lapse rate (<4 C/km): Stable, often means inversions
      where it's warmer on the mountain than in the valley
    - Negative lapse rate: Temperature INVERSION — traps cold air in
      valleys, rain at base but snow at peak

    This is a powerful feature because it tells the model about
    atmospheric stability, which controls precipitation type and amount.

    Args:
        temps_by_elev: List of (elevation_ft, temp_f) tuples.

    Returns:
        Lapse rate in C/km. Positive means temp decreases with altitude.
    """
    if len(temps_by_elev) < 2:
        return np.nan

    elevs = np.array([e for e, t in temps_by_elev if not np.isnan(t)])
    temps = np.array([t for e, t in temps_by_elev if not np.isnan(t)])

    if len(elevs) < 2:
        return np.nan

    # Convert to metric
    elevs_m = elevs * 0.3048
    temps_c = (temps - 32) * 5 / 9

    # Linear regression: temp = slope * elev + intercept
    elev_range = elevs_m.max() - elevs_m.min()
    if elev_range < 100:  # Need at least 100m elevation range
        return np.nan

    slope = np.polyfit(elevs_m, temps_c, 1)[0]
    # Lapse rate is -slope (positive means cooling with altitude)
    return round(-slope * 1000, 2)  # C per km


def compute_orographic_multiplier(
    elev_ft: float, wind_mph: float, wind_dir_deg: float,
    aspect_deg: float = 247.5,
) -> float:
    """
    Estimate orographic precipitation enhancement factor.

    === ML LEARNING NOTE: Orographic Enhancement ===

    Mountains force air upward, causing it to cool and condense moisture.
    The windward side of mountains gets MORE precipitation than flat
    terrain, while the leeward side gets LESS (rain shadow).

    For Tahoe, the optimal wind direction is WSW (247.5 degrees) — storms
    coming from the Pacific that hit the Sierra crest head-on produce the
    most orographic lift.

    This feature tells the model: "given the wind direction and mountain
    geometry, how much extra precipitation should we expect compared to
    what the model predicts at its coarse grid?"

    Key insight: NWP models run at 10-25 km grid spacing, so they can't
    resolve the actual Sierra Nevada terrain. They see a smoothed version
    of the mountains. The orographic multiplier corrects for this.
    """
    # Delegate to tahoe_snow.py's CAPE-aware implementation.
    # Note: tahoe_snow.py uses a hardcoded ideal=247.5 for Sierra Nevada.
    # The aspect_deg parameter from resort_configs is not used by the
    # authoritative function — the ideal orographic axis is constant.
    if np.isnan(wind_mph) or np.isnan(wind_dir_deg):
        return 1.0
    return _ts_orographic_multiplier(elev_ft, wind_mph, wind_dir_deg)


def compute_froude_number(wind_mph: float, stability_nv: float = 0.01,
                          barrier_m: float = 2000) -> float:
    """
    Compute Froude number for orographic flow regime.

    === ML LEARNING NOTE: Froude Number ===

    The Froude number determines whether wind flows OVER a mountain
    (Fr > 1, good for orographic precipitation) or AROUND it
    (Fr < 1, less precipitation on the mountain).

    Fr = wind_speed / (stability * barrier_height)

    - Fr > 1: "Supercritical" — air flows over the barrier, producing
      lots of uplift and precipitation
    - Fr < 1: "Subcritical" — air is blocked by the mountain, flows
      around it instead. Less precipitation on the mountain, but
      possibly more in gaps and passes
    - Fr ~ 1: Transitional — some flow over, some around

    This matters for Tahoe because the Sierra crest is ~2000m above
    the valley floor. In stable conditions with light winds, storms
    might not produce as much snow at high elevations as models predict.
    """
    if wind_mph <= 0 or stability_nv <= 0:
        return np.nan

    wind_mps = wind_mph * 0.44704
    fr = wind_mps / (stability_nv * barrier_m)
    return round(fr, 3)


def compute_wet_bulb_f(temp_f: float, rh_pct: float) -> float:
    """
    Estimate wet-bulb temperature (simplified Stull 2011 formula).

    === ML LEARNING NOTE: Wet Bulb Temperature ===

    Wet-bulb temperature determines whether precipitation falls as
    rain or snow. It's MORE accurate than regular temperature because
    evaporative cooling matters during precipitation.

    When snow falls through dry air, evaporation cools the air. So
    even if the air temperature is 36F (above freezing), if the air
    is dry enough, the wet-bulb temperature could be below 32F and
    the snow survives to the ground.

    Rule of thumb for Tahoe:
    - Wet bulb < 30F: All snow
    - Wet bulb 30-34F: Snow/rain boundary — elevation dependent
    - Wet bulb > 34F: Mostly rain at base, could still be snow at peak
    """
    temp_c = (temp_f - 32) * 5 / 9
    rh = max(1, min(100, rh_pct))

    # Stull (2011) approximation
    wb_c = temp_c * math.atan(0.151977 * (rh + 8.313659) ** 0.5) + \
           math.atan(temp_c + rh) - math.atan(rh - 1.676331) + \
           0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035

    return round(wb_c * 9 / 5 + 32, 1)


# =========================================================================
# Feature Building Pipeline
# =========================================================================

def compute_snotel_derived_features(snotel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived target variables to SNOTEL data.

    === ML LEARNING NOTE: Target Variables ===

    In supervised ML, you need TARGETS (labels) — the "right answers"
    the model learns to predict. For snow forecasting, our targets are:

    1. depth_change_in: Daily change in snow depth. Positive = new snow.
       Negative = melting/settling.

    2. swe_change_in: Daily change in Snow Water Equivalent. This is a
       better measure of new precipitation because it's not affected by
       compaction/settling of existing snow.

    3. effective_slr: The observed SLR = depth_change / swe_change.
       This tells us the actual snow density, which models often get wrong.

    We compute these per-station so we have labeled data for each
    SNOTEL site independently.
    """
    logger.info("Computing SNOTEL derived features...")
    df = snotel_df.copy()
    df = df.sort_values(["station_name", "date"])

    # Daily changes (group by station to avoid cross-station diffs)
    df["depth_change_in"] = df.groupby("station_name")["SNWD"].diff()
    df["swe_change_in"] = df.groupby("station_name")["WTEQ"].diff()

    # Effective SLR: only when both depth and SWE increased
    mask = (df["depth_change_in"] > 0) & (df["swe_change_in"] > 0.05)
    df["effective_slr"] = np.nan
    df.loc[mask, "effective_slr"] = (
        df.loc[mask, "depth_change_in"] / df.loc[mask, "swe_change_in"]
    )
    # Cap at physically reasonable range
    df.loc[df["effective_slr"] > 30, "effective_slr"] = np.nan
    df.loc[df["effective_slr"] < 2, "effective_slr"] = np.nan

    # Lagged features: what happened in the previous 1-3 days
    for lag in [1, 3, 7]:
        df[f"depth_lag{lag}d"] = df.groupby("station_name")["SNWD"].shift(lag)
        df[f"swe_lag{lag}d"] = df.groupby("station_name")["WTEQ"].shift(lag)
        df[f"tmax_lag{lag}d"] = df.groupby("station_name")["TMAX"].shift(lag)

    # Rolling features: trends over windows
    for window in [3, 7]:
        df[f"depth_change_{window}d"] = df.groupby("station_name")["SNWD"].diff(window)
        df[f"swe_change_{window}d"] = df.groupby("station_name")["WTEQ"].diff(window)
        df[f"tmax_avg_{window}d"] = df.groupby("station_name")["TMAX"].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    # Was there new snow? (binary classification target)
    df["new_snow"] = (df["depth_change_in"] > 0.5).astype(int)

    # Significant snowfall (>6 inches — "powder day")
    df["powder_day"] = (df["depth_change_in"] > 6).astype(int)

    # --- Improvement 2: Storm state features ---
    # Consecutive snow days: how many days in a row has it been snowing?
    # Resets to 0 after a non-snow day.
    def _consecutive_snow(group):
        is_snow = (group["depth_change_in"] > 0.5).astype(int)
        result = pd.Series(0, index=group.index, dtype=int)
        count = 0
        for i, val in is_snow.items():
            if val == 1:
                count += 1
            else:
                count = 0
            result[i] = count
        return result

    df["consecutive_snow_days"] = df.groupby("station_name", group_keys=False).apply(
        _consecutive_snow
    ).values

    # Storm total: cumulative depth change during current storm event
    # (resets after 2 consecutive dry days)
    def _storm_total(group):
        dc = group["depth_change_in"].fillna(0)
        result = pd.Series(0.0, index=group.index)
        total = 0.0
        dry_count = 0
        for i, val in dc.items():
            if val > 0.5:
                total += val
                dry_count = 0
            else:
                dry_count += 1
                if dry_count >= 2:
                    total = 0.0
            result[i] = total
        return result

    df["storm_total_in"] = df.groupby("station_name", group_keys=False).apply(
        _storm_total
    ).values

    # Settling pressure: existing deep snowpack + new water = compaction
    # A proxy for how much the existing pack will compress under new load
    df["settling_pressure"] = df["SNWD"].fillna(0) * df["swe_change_in"].clip(lower=0).fillna(0)

    # --- PREC-based precipitation (daily change in cumulative precip) ---
    # PREC is water-year cumulative precipitation in inches (liquid equivalent).
    # Daily delta = new precipitation. Unlike SWE-based detection, PREC captures
    # rain events too, making it the best single feature for total precipitation.
    df["prec_change_in"] = df.groupby("station_name")["PREC"].diff()
    # Clamp negatives (PREC resets at water year boundary Oct 1)
    df.loc[df["prec_change_in"] < 0, "prec_change_in"] = 0
    # 3-day and 7-day accumulated precipitation
    df["prec_3d_in"] = df.groupby("station_name")["PREC"].diff(3).clip(lower=0)
    df["prec_7d_in"] = df.groupby("station_name")["PREC"].diff(7).clip(lower=0)

    # Rain detection: precip fell but depth didn't increase = rain, not snow
    df["rain_event"] = (
        (df["prec_change_in"] > 0.1) & (df["depth_change_in"] <= 0)
    ).astype(int)

    # Observed density: SWE change / depth change (when both positive)
    # Lower density = fluffier snow = higher quality
    density_mask = (df["depth_change_in"] > 1) & (df["swe_change_in"] > 0.05)
    df["observed_density"] = np.nan
    df.loc[density_mask, "observed_density"] = (
        df.loc[density_mask, "swe_change_in"] / df.loc[density_mask, "depth_change_in"]
    )

    # Compaction rate: how fast is the existing snowpack settling?
    # Positive when depth decreases but SWE stays constant (= pure compaction)
    compact_mask = (df["depth_change_in"] < -0.5) & (df["swe_change_in"].abs() < 0.1)
    df["compaction_rate"] = 0.0
    df.loc[compact_mask, "compaction_rate"] = -df.loc[compact_mask, "depth_change_in"]

    # Diurnal temperature range (solar radiation proxy)
    df["diurnal_range_f"] = df["TMAX"] - df["TMIN"]

    # Freeze-thaw cycles: TMAX > 32 and TMIN < 28 (causes crust/melt-freeze)
    df["freeze_thaw"] = ((df["TMAX"] > 32) & (df["TMIN"] < 28)).astype(int)

    # Temperature asymmetry: how far above/below freezing
    df["tmax_above_freezing"] = (df["TMAX"] - 32).clip(lower=0)
    df["tmin_below_freezing"] = (32 - df["TMIN"]).clip(lower=0)

    # --- Improvement 5: Station-specific bias features ---
    # Target-encoded station mean (leave-one-out to avoid leakage)
    # Uses the historical average depth_change per station as a feature
    # so the model can learn station-specific offsets.
    station_means = df.groupby("station_name")["depth_change_in"].transform("mean")
    df["station_mean_depth_change"] = station_means

    # Station variability (some stations are noisier than others)
    station_stds = df.groupby("station_name")["depth_change_in"].transform("std")
    df["station_std_depth_change"] = station_stds

    logger.info("  Added %d derived columns to SNOTEL data", len(df.columns) - len(snotel_df.columns))
    return df


def compute_era5_derived_features(era5_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-derived features to ERA5/forecast data.

    These are the features that encode atmospheric physics knowledge.
    """
    logger.info("Computing ERA5 derived features...")
    df = era5_df.copy()
    df = df.sort_values(["resort", "zone", "date"])

    # SLR from temperature and wind
    df["slr"] = df.apply(
        lambda r: compute_slr(
            r.get("temperature_2m_mean", 32),
            r.get("wind_speed_10m_max", 0),
        ),
        axis=1,
    )

    # Orographic multiplier from wind direction and elevation
    df["orographic_mult"] = df.apply(
        lambda r: compute_orographic_multiplier(
            r.get("elev_ft", 7000),
            r.get("wind_speed_10m_max", 0),
            r.get("wind_direction_10m_dominant", 0),
            r.get("aspect_deg", 247.5),
        ),
        axis=1,
    )

    # Orographic-adjusted precipitation
    if "precipitation_sum" in df.columns:
        df["precip_orographic_in"] = df["precipitation_sum"] * df["orographic_mult"]

    # --- Improvement 1: Rain/snow discrimination ---
    # Rain fraction based on temperature: at warm temps, ERA5 "snowfall"
    # is often actually rain at base elevations. This feature lets the model
    # learn to discount the snowfall forecast when it's too warm.
    df["rain_fraction"] = df["temperature_2m_mean"].apply(
        lambda t: 0.0 if np.isnan(t) or t < 30 else min(1.0, (t - 30) / 6)
    )
    # Snow fraction is the complement
    df["snow_fraction"] = 1.0 - df["rain_fraction"]
    # Adjusted snowfall = snowfall * snow_fraction (what likely fell as snow)
    if "snowfall_sum" in df.columns:
        df["snowfall_adjusted_cm"] = df["snowfall_sum"] * df["snow_fraction"]

    # --- Met 1: Freezing level estimate ---
    # We don't have hourly freezing_level_height in our daily data, but we can
    # estimate it from the lapse rate and temperature. Freezing level ≈ the
    # elevation where temperature = 32F, extrapolated from the zone temperature.
    # freezing_level_ft = zone_elev + (temp_f - 32) / lapse_rate_f_per_ft
    # Using standard lapse rate 3.5F/1000ft as default
    df["est_freezing_level_ft"] = df.apply(
        lambda r: r.get("elev_ft", 7000) + (r.get("temperature_2m_mean", 32) - 32) / 3.5 * 1000
        if not np.isnan(r.get("temperature_2m_mean", np.nan)) else np.nan,
        axis=1,
    )
    # How far is the freezing level above/below the zone?
    df["freezing_level_margin_ft"] = df["est_freezing_level_ft"] - df["elev_ft"]
    # Positive = freezing level above zone (warmer, more rain risk)
    # Negative = freezing level below zone (colder, snow guaranteed)

    # --- Met 2: IVT proxy and AR detection ---
    # Integrated Vapor Transport (IVT) is the gold standard for AR detection,
    # but requires column-integrated humidity + wind from upper-air data.
    # Proxy: surface wind speed * (rain_sum / precipitation_sum) captures
    # "how much moisture is being transported." Strong wind + high rain
    # fraction = atmospheric river signature.
    precip = df["precipitation_sum"].clip(lower=0.01)  # avoid div by zero
    rain = df["rain_sum"].fillna(0).clip(lower=0)
    wind = df["wind_speed_10m_max"].fillna(0)
    df["moisture_transport_proxy"] = wind * (rain / precip)
    # Normalize: high values = lots of moisture moving fast
    df["moisture_transport_proxy"] = df["moisture_transport_proxy"].clip(upper=50)

    # AR flag: warm, wet, windy, from the south/southwest
    wind_dir = df["wind_direction_10m_dominant"].fillna(0)
    is_southerly = ((wind_dir >= 150) & (wind_dir <= 240)).astype(float)
    is_warm = (df["temperature_2m_mean"] > 34).astype(float)
    is_wet = (df["precipitation_sum"] > 0.3).astype(float)
    is_windy = (wind > 15).astype(float)
    # AR score: 0-4 (how many AR indicators are present)
    df["ar_score"] = is_southerly + is_warm + is_wet + is_windy
    # Binary AR flag (3+ indicators = likely AR)
    df["is_ar_event"] = (df["ar_score"] >= 3).astype(int)

    # Temperature trends
    df["temp_trend_3d"] = df.groupby(["resort", "zone"])["temperature_2m_mean"].diff(3)

    # Pressure features (if available)
    if "pressure_msl" in df.columns:
        df["pressure_change_24h"] = df.groupby(["resort", "zone"])["pressure_msl"].diff(1)
    elif "surface_pressure" in df.columns:
        df["pressure_change_24h"] = df.groupby(["resort", "zone"])["surface_pressure"].diff(1)

    # Calendar features
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["water_year_day"] = df["date"].apply(
        lambda d: (d - pd.Timestamp(f"{d.year if d.month >= 10 else d.year - 1}-10-01")).days
    )

    # Season (winter = Oct-Apr)
    df["is_winter"] = df["month"].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)

    # Water year (Oct-Sep)
    df["water_year"] = df["date"].apply(
        lambda d: d.year + 1 if d.month >= 10 else d.year
    )

    logger.info("  Added derived features to ERA5 data")
    return df


# =========================================================================
# Improvement 7: ENSO/PDO Climate Indices
# =========================================================================

def fetch_enso_oni() -> pd.DataFrame:
    """
    Fetch the Oceanic Nino Index (ONI) — the standard ENSO metric.

    ONI is a 3-month running mean of SST anomalies in the Nino 3.4 region.
    El Nino (ONI > 0.5): Tends toward warmer, wetter storms at Tahoe
    La Nina (ONI < -0.5): Tends toward colder, drier — but big La Nina
    years (2017, 2023) can produce enormous Tahoe snow from persistent
    northwesterly flow.

    Source: NOAA CPC via PSL
    """
    import requests
    try:
        # NOAA's ONI data in a simple text format
        resp = requests.get(
            "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("  Failed to fetch ONI data: HTTP %d", resp.status_code)
            return pd.DataFrame()

        lines = resp.text.strip().split("\n")
        rows = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 4:
                try:
                    year = int(parts[0])
                    # Season code (DJF, JFM, ..., NDJ)
                    season = parts[1]
                    oni = float(parts[-1])
                    # Map season to month (use middle month)
                    season_month = {
                        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
                        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
                        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
                    }
                    month = season_month.get(season)
                    if month:
                        rows.append({"year": year, "month": month, "oni": oni})
                except (ValueError, IndexError):
                    continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info("  Fetched ONI data: %d months (%d-%d)",
                    len(df), df["year"].min(), df["year"].max())
        return df

    except Exception as e:
        logger.warning("  Failed to fetch ONI: %s", e)
        return pd.DataFrame()


def add_climate_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add ENSO ONI index to training data."""
    logger.info("  Fetching climate indices (ENSO ONI)...")

    oni_df = fetch_enso_oni()
    if oni_df.empty:
        df["oni"] = np.nan
        df["enso_phase"] = 0  # 0=neutral
        return df

    # Create a lookup: (year, month) -> oni
    oni_lookup = {}
    for _, row in oni_df.iterrows():
        oni_lookup[(int(row["year"]), int(row["month"]))] = row["oni"]

    # Map ONI to training data
    df["oni"] = df["date"].apply(
        lambda d: oni_lookup.get((d.year, d.month), np.nan)
    )

    # Categorical ENSO phase
    df["enso_phase"] = df["oni"].apply(
        lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0) if not np.isnan(x) else 0
    )

    valid = df["oni"].notna().sum()
    logger.info("  ONI mapped for %d/%d rows (%.0f%%)", valid, len(df), 100*valid/len(df))
    return df


# =========================================================================
# Training Data Assembly
# =========================================================================

def build_training_data(snotel_df: pd.DataFrame,
                        era5_df: pd.DataFrame,
                        forecast_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge SNOTEL targets with ERA5/forecast features to create training data.

    === ML LEARNING NOTE: The Merge — Most Critical Step ===

    This is where we create the actual (feature, target) pairs that the
    ML model trains on. Every row in the output represents one day at one
    station, with:

    - FEATURES: What the weather model predicted (temperature, wind,
      precipitation from ERA5 or archived forecasts) + derived physics
      features (SLR, lapse rate, orographic multiplier)

    - TARGETS: What SNOTEL actually measured (depth change, SWE change,
      temperature)

    The model learns: "When the features look like THIS, the actual
    outcome tends to be THAT." Then for a new forecast, it predicts
    what the actual outcome will be and applies a correction.

    Key decisions here:
    1. We match by DATE (inner join — only keep days with BOTH forecast
       and observation data)
    2. We match by ELEVATION — each station is paired with the nearest
       resort zone at a similar elevation
    3. We only keep winter months (Oct-Apr) because summer data is
       irrelevant to snow forecasting
    """
    logger.info("Building training data...")

    # Step 1: Enrich both datasets
    snotel_enriched = compute_snotel_derived_features(snotel_df)
    era5_enriched = compute_era5_derived_features(era5_df)

    # Step 2: Map stations to nearest resort zones
    station_resort_map = build_station_resort_map()

    training_rows = []
    stations_matched = 0

    for station_name, station_info in SNOTEL_STATIONS.items():
        if station_name not in snotel_enriched["station_name"].unique():
            continue

        # Get this station's data
        station_data = snotel_enriched[snotel_enriched["station_name"] == station_name].copy()

        # Find the resort zone(s) this station maps to
        resort_mappings = station_resort_map.get(station_name, [])
        if not resort_mappings:
            logger.warning("  No resort mapping for station %s", station_name)
            continue

        for mapping in resort_mappings:
            resort_name = mapping["resort"]
            zones = mapping["zones"]
            nearest_zone = find_nearest_zone(station_info["elev_ft"], zones)

            # Get ERA5 data for this resort/zone
            era5_zone = era5_enriched[
                (era5_enriched["resort"] == resort_name) &
                (era5_enriched["zone"] == nearest_zone)
            ]

            if era5_zone.empty:
                logger.warning("  No ERA5 data for %s/%s", resort_name, nearest_zone)
                continue

            # Merge on date
            merged = pd.merge(
                station_data,
                era5_zone,
                on="date",
                how="inner",
                suffixes=("_snotel", "_era5"),
            )

            # Add metadata
            merged["resort"] = resort_name
            merged["matched_zone"] = nearest_zone
            merged["station_name"] = station_name
            merged["station_elev_ft"] = station_info["elev_ft"]
            merged["zone_elev_ft"] = era5_zone["elev_ft"].iloc[0] if len(era5_zone) > 0 else np.nan
            merged["elev_diff_ft"] = merged["station_elev_ft"] - merged["zone_elev_ft"]

            # Compute lapse rate from multi-zone temperatures
            # (use all zones of this resort on this date)
            training_rows.append(merged)
            stations_matched += 1

    if not training_rows:
        logger.error("No training data produced!")
        return pd.DataFrame()

    training_df = pd.concat(training_rows, ignore_index=True)

    # Step 3: Filter to winter months only (Oct-Apr)
    if "month" in training_df.columns:
        training_df = training_df[training_df["is_winter"] == 1].copy()

    # Step 4: Add lapse rate as a cross-zone feature
    training_df = _add_lapse_rate_feature(training_df, era5_enriched)

    # Step 5: Compute Froude number
    if "wind_speed_10m_max" in training_df.columns:
        training_df["froude_number"] = training_df["wind_speed_10m_max"].apply(
            lambda w: compute_froude_number(w) if not np.isnan(w) else np.nan
        )

    # Step 6: Compute wet bulb temperature
    if "temperature_2m_mean" in training_df.columns:
        training_df["wet_bulb_f"] = training_df.apply(
            lambda r: compute_wet_bulb_f(r["temperature_2m_mean"], 70)
            if not np.isnan(r.get("temperature_2m_mean", np.nan)) else np.nan,
            axis=1,
        )

    # Step 7: Add elevation band classification
    training_df["elevation_band"] = training_df["station_elev_ft"].apply(
        lambda e: "base" if e < 7000 else ("mid" if e < 8500 else "peak")
    )

    # Step 8: Improvement 7 — Add ENSO/PDO climate indices
    training_df = add_climate_indices(training_df)

    # Step 9: MLE 2 — Fix train/serve skew
    # For 2021+ rows where we have archived forecast data, OVERLAY the ECMWF
    # forecast values onto the core ERA5 weather features. This means the model
    # trains on what the forecast PREDICTED (same distribution as production),
    # not what actually happened (ERA5 reanalysis).
    # Pre-2021 rows keep ERA5 values (no archived forecasts available).
    if forecast_df is not None and not forecast_df.empty:
        logger.info("  Aligning training features with forecast data (fixing train/serve skew)...")
        # Use ECMWF as primary (best global model)
        ecmwf = forecast_df[forecast_df["model"] == "ecmwf_ifs025"].copy()
        if not ecmwf.empty:
            # Build a lookup: (resort, zone, date) -> forecast values
            ecmwf_lookup = {}
            for _, row in ecmwf.iterrows():
                key = (row["resort"], row["zone"], row["date"])
                ecmwf_lookup[key] = row

            n_aligned = 0
            for idx, trow in training_df.iterrows():
                key = (trow.get("resort"), trow.get("matched_zone"), trow["date"])
                if key in ecmwf_lookup:
                    frow = ecmwf_lookup[key]
                    # Overlay forecast values onto the ERA5-based core features
                    # These are the SAME column names the model sees in production
                    for col in ["temperature_2m_max", "temperature_2m_min",
                                "temperature_2m_mean", "precipitation_sum",
                                "snowfall_sum", "rain_sum",
                                "wind_speed_10m_max", "wind_gusts_10m_max",
                                "wind_direction_10m_dominant"]:
                        if col in frow.index and pd.notna(frow[col]):
                            training_df.at[idx, col] = frow[col]
                    n_aligned += 1

            pct = 100 * n_aligned / len(training_df)
            logger.info("  Aligned %d/%d rows (%.0f%%) with ECMWF forecast data",
                        n_aligned, len(training_df), pct)

            # Recompute derived features that depend on the overwritten values
            training_df["rain_fraction"] = training_df["temperature_2m_mean"].apply(
                lambda t: 0.0 if np.isnan(t) or t < 30 else min(1.0, (t - 30) / 6)
            )
            training_df["snow_fraction"] = 1.0 - training_df["rain_fraction"]
            training_df["snowfall_adjusted_cm"] = (
                training_df["snowfall_sum"] * training_df["snow_fraction"]
            )
            training_df["precip_orographic_in"] = (
                training_df["precipitation_sum"] * training_df["orographic_mult"]
            )

    logger.info("Training data: %d rows, %d columns, %d station-resort pairs",
                len(training_df), len(training_df.columns), stations_matched)

    return training_df


def _add_lapse_rate_feature(df: pd.DataFrame,
                            era5_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily lapse rate from multi-zone temperatures per resort."""
    logger.info("  Computing lapse rates from multi-zone temperatures...")

    # Build a lookup: (resort, date) -> lapse_rate
    lapse_rates = {}
    for resort in era5_df["resort"].unique():
        resort_data = era5_df[era5_df["resort"] == resort]
        for date in resort_data["date"].unique():
            day_data = resort_data[resort_data["date"] == date]
            if len(day_data) < 2:
                continue
            temps_by_elev = [
                (row["elev_ft"], row.get("temperature_2m_mean", np.nan))
                for _, row in day_data.iterrows()
            ]
            lr = compute_lapse_rate(temps_by_elev)
            if not np.isnan(lr):
                lapse_rates[(resort, date)] = lr

    # Map into training data
    df["lapse_rate_c_km"] = df.apply(
        lambda r: lapse_rates.get((r.get("resort", ""), r["date"]), np.nan),
        axis=1,
    )

    n_valid = df["lapse_rate_c_km"].notna().sum()
    logger.info("  Lapse rate computed for %d/%d rows (%.1f%%)",
                n_valid, len(df), 100 * n_valid / max(1, len(df)))

    return df


# =========================================================================
# Add Archived Forecast Features (when available)
# =========================================================================

def merge_forecast_features(training_df: pd.DataFrame,
                            forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge archived NWP forecast data as additional features.

    === ML LEARNING NOTE: Multi-Model Features ===

    Having forecasts from multiple NWP models (GFS, ECMWF, ICON) as
    features is powerful because:

    1. MODEL SPREAD: When models disagree, uncertainty is high. The ML
       model can learn "when GFS says 10\" and ECMWF says 3\", the truth
       is usually closer to ECMWF" (or vice versa, depending on conditions).

    2. MODEL-SPECIFIC BIASES: Each model has different terrain handling.
       GFS at 13km resolution sees a different Sierra Nevada than ECMWF
       at 9km. The ML model learns which model to trust in which conditions.

    3. ENSEMBLE INFORMATION: The spread across models is itself a feature.
       High model spread = high forecast uncertainty = the ML correction
       should be more conservative (closer to 1.0 = no correction).
    """
    if forecast_df is None or forecast_df.empty:
        logger.info("No archived forecast data to merge, using ERA5 only")
        return training_df

    logger.info("Merging archived forecast features...")

    # Pivot forecast data: one column per model per variable
    for model in forecast_df["model"].unique():
        model_short = model.replace("_seamless", "").replace("_ifs025", "")
        model_data = forecast_df[forecast_df["model"] == model]

        # Select key forecast columns to merge
        fcst_cols = {
            "temperature_2m_max": f"{model_short}_tmax_f",
            "temperature_2m_min": f"{model_short}_tmin_f",
            "temperature_2m_mean": f"{model_short}_tmean_f",
            "precipitation_sum": f"{model_short}_precip_in",
            "snowfall_sum": f"{model_short}_snow_cm",
            "wind_speed_10m_max": f"{model_short}_wind_max_mph",
            "wind_gusts_10m_max": f"{model_short}_gust_max_mph",
            "wind_direction_10m_dominant": f"{model_short}_wind_dir_deg",
        }

        # Rename columns for this model
        model_features = model_data[
            ["date", "resort", "zone"] +
            [c for c in fcst_cols.keys() if c in model_data.columns]
        ].rename(columns=fcst_cols)

        # Merge into training data
        training_df = pd.merge(
            training_df,
            model_features,
            left_on=["date", "resort", "matched_zone"],
            right_on=["date", "resort", "zone"],
            how="left",
            suffixes=("", f"_{model_short}"),
        )

        # Drop the duplicate "zone" column from forecast
        if "zone" in training_df.columns and f"zone_{model_short}" not in training_df.columns:
            # zone was from the merge key; already have matched_zone
            pass

    # Compute model spread features
    snow_cols = [c for c in training_df.columns if c.endswith("_snow_cm")]
    if len(snow_cols) >= 2:
        snow_matrix = training_df[snow_cols].values
        training_df["model_snow_spread"] = np.nanstd(snow_matrix, axis=1)
        training_df["model_snow_mean"] = np.nanmean(snow_matrix, axis=1)
        training_df["model_snow_max"] = np.nanmax(snow_matrix, axis=1)
        training_df["model_snow_min"] = np.nanmin(snow_matrix, axis=1)
        training_df["model_snow_range"] = (
            training_df["model_snow_max"] - training_df["model_snow_min"]
        )

    temp_cols = [c for c in training_df.columns if c.endswith("_tmax_f")]
    if len(temp_cols) >= 2:
        temp_matrix = training_df[temp_cols].values
        training_df["model_temp_spread"] = np.nanstd(temp_matrix, axis=1)
        training_df["model_temp_mean"] = np.nanmean(temp_matrix, axis=1)

    logger.info("  Added %d model-specific forecast columns", len(snow_cols) + len(temp_cols))
    return training_df


# =========================================================================
# Exploratory Data Analysis
# =========================================================================

def run_eda(training_df: pd.DataFrame):
    """
    Print exploratory data analysis summary.

    === ML LEARNING NOTE: EDA — Know Your Data Before Modeling ===

    EDA is the "look before you leap" step. Before training any model,
    you MUST understand:

    1. DISTRIBUTIONS: Are your features normally distributed? Skewed?
       Do you have outliers that could dominate the model?

    2. CORRELATIONS: Which features are strongly correlated with your
       target? Which features are correlated with EACH OTHER (multicollinearity)?

    3. MISSING DATA: How much data is missing? Is it random or systematic?
       (e.g., always missing during storms = problematic)

    4. CLASS BALANCE: For classification targets (did it snow? was it a
       powder day?), what's the ratio of positive to negative examples?
       Severe imbalance requires special handling.

    5. SEASONAL PATTERNS: Do biases change with season? (Usually yes for
       weather data — early season vs late season behave differently.)
    """
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    print(f"\n{'='*60}")
    print("TRAINING DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows: {len(training_df):,}")
    print(f"Columns: {len(training_df.columns)}")
    print(f"Date range: {training_df['date'].min().date()} to {training_df['date'].max().date()}")

    # Station coverage
    print(f"\n--- Stations ---")
    for station in sorted(training_df["station_name"].unique()):
        sdf = training_df[training_df["station_name"] == station]
        print(f"  {station:25s}: {len(sdf):5d} rows, "
              f"resort={sdf['resort'].iloc[0]}, "
              f"zone={sdf['matched_zone'].iloc[0]}, "
              f"elev={sdf['station_elev_ft'].iloc[0]}ft")

    # Target variable statistics
    print(f"\n--- Target Variables ---")
    targets = ["depth_change_in", "swe_change_in", "effective_slr",
                "TMAX", "TMIN", "new_snow", "powder_day"]
    for target in targets:
        if target in training_df.columns:
            s = training_df[target]
            valid = s.notna().sum()
            print(f"  {target:25s}: mean={s.mean():8.2f}, "
                  f"std={s.std():7.2f}, "
                  f"min={s.min():8.2f}, max={s.max():8.2f}, "
                  f"valid={valid}/{len(s)} ({100*valid/len(s):.0f}%)")

    # Feature completeness
    print(f"\n--- Feature Completeness ---")
    era5_features = [c for c in training_df.columns
                     if c.startswith("temperature_2m") or c.startswith("precipitation")
                     or c.startswith("wind_") or c.startswith("snow")]
    for feat in sorted(era5_features)[:15]:
        valid_pct = 100 * training_df[feat].notna().sum() / len(training_df)
        print(f"  {feat:40s}: {valid_pct:5.1f}% valid")

    # Derived features
    print(f"\n--- Derived Features ---")
    derived = ["slr", "orographic_mult", "lapse_rate_c_km", "froude_number",
               "wet_bulb_f", "temp_trend_3d"]
    for feat in derived:
        if feat in training_df.columns:
            s = training_df[feat]
            valid = s.notna().sum()
            print(f"  {feat:25s}: mean={s.mean():8.3f}, "
                  f"std={s.std():7.3f}, "
                  f"valid={valid}/{len(s)} ({100*valid/len(s):.0f}%)")

    # Snow day frequency
    print(f"\n--- Snow Event Statistics ---")
    if "new_snow" in training_df.columns:
        snow_days = training_df["new_snow"].sum()
        total_days = len(training_df)
        print(f"  Snow days (>0.5\" new):   {snow_days:5.0f}/{total_days} "
              f"({100*snow_days/total_days:.1f}%)")
    if "powder_day" in training_df.columns:
        powder_days = training_df["powder_day"].sum()
        total_days = len(training_df)
        print(f"  Powder days (>6\" new):   {powder_days:5.0f}/{total_days} "
              f"({100*powder_days/total_days:.1f}%)")

    # Seasonal breakdown
    print(f"\n--- By Month ---")
    monthly = training_df.groupby("month").agg(
        rows=("date", "count"),
        avg_depth_change=("depth_change_in", "mean"),
        avg_swe_change=("swe_change_in", "mean"),
    )
    for month, row in monthly.iterrows():
        month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(month)]
        print(f"  {month_name}: {row['rows']:5.0f} rows, "
              f"avg depth_change={row['avg_depth_change']:+.2f}\", "
              f"avg swe_change={row['avg_swe_change']:+.3f}\"")

    # Water year breakdown
    print(f"\n--- By Water Year ---")
    yearly = training_df.groupby("water_year").agg(
        rows=("date", "count"),
        total_snow_days=("new_snow", "sum"),
        total_powder_days=("powder_day", "sum"),
    )
    for year, row in yearly.iterrows():
        print(f"  {int(year)}: {row['rows']:5.0f} rows, "
              f"{row['total_snow_days']:3.0f} snow days, "
              f"{row['total_powder_days']:2.0f} powder days")

    # Correlation with targets
    print(f"\n--- Top Feature Correlations with depth_change_in ---")
    if "depth_change_in" in training_df.columns:
        numeric_cols = training_df.select_dtypes(include=[np.number]).columns
        correlations = training_df[numeric_cols].corr()["depth_change_in"].drop("depth_change_in", errors="ignore")
        correlations = correlations.dropna().abs().sort_values(ascending=False)
        for feat, corr in correlations.head(20).items():
            actual_corr = training_df[numeric_cols].corr()["depth_change_in"].get(feat, 0)
            print(f"  {feat:40s}: {actual_corr:+.3f}")

    print(f"\n{'='*60}")


# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature engineering for Tahoe Snow ML pipeline",
    )
    parser.add_argument("--eda", action="store_true",
                        help="Run exploratory data analysis only")
    parser.add_argument("--check", action="store_true",
                        help="Check feature completeness")
    args = parser.parse_args()

    # Load data
    snotel_path = os.path.join(DATA_DIR, "snotel_daily.parquet")
    era5_path = os.path.join(DATA_DIR, "era5_daily.parquet")
    forecast_path = os.path.join(DATA_DIR, "archived_forecasts_daily.parquet")

    if not os.path.exists(snotel_path):
        logger.error("SNOTEL data not found. Run: python historical_data_collector.py --snotel")
        sys.exit(1)
    if not os.path.exists(era5_path):
        logger.error("ERA5 data not found. Run: python historical_data_collector.py --era5")
        sys.exit(1)

    logger.info("Loading data...")
    snotel_df = pd.read_parquet(snotel_path)
    era5_df = pd.read_parquet(era5_path)
    logger.info("  SNOTEL: %d rows", len(snotel_df))
    logger.info("  ERA5:   %d rows", len(era5_df))

    forecast_df = None
    if os.path.exists(forecast_path):
        forecast_df = pd.read_parquet(forecast_path)
        logger.info("  Forecasts: %d rows", len(forecast_df))

    # Build training data
    training_df = build_training_data(snotel_df, era5_df, forecast_df)

    if training_df.empty:
        logger.error("No training data produced!")
        sys.exit(1)

    # Merge archived forecast features if available
    if forecast_df is not None:
        training_df = merge_forecast_features(training_df, forecast_df)

    # Save
    outpath = os.path.join(DATA_DIR, "training_data.parquet")
    training_df.to_parquet(outpath, index=False)
    logger.info("Saved training data: %d rows x %d cols to %s (%.1f MB)",
                len(training_df), len(training_df.columns), outpath,
                os.path.getsize(outpath) / 1e6)

    # CSV for easy inspection
    csv_path = os.path.join(DATA_DIR, "training_data.csv")
    training_df.to_csv(csv_path, index=False)

    # Always run EDA
    run_eda(training_df)


if __name__ == "__main__":
    main()
