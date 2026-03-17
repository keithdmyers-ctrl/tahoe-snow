#!/usr/bin/env python3
"""
Historical Data Collector for ML Training Pipeline
===================================================

Fetches and stores historical weather data for training snow forecast
correction models. Three data sources:

1. SNOTEL (ground truth): 10+ years of daily snow depth, SWE, temperature
   from 10 stations spanning 6,200-8,790 ft in the Tahoe basin.

2. Open-Meteo ERA5 Reanalysis (historical weather): What the atmosphere
   *actually did*, reconstructed from observations. Used as supplementary
   ground truth and for long-term bias analysis.

3. Open-Meteo Archived Forecasts (historical predictions): What the models
   *predicted* at the time. This is the key data for training "forecast
   minus observation = bias" correction models. Available from ~2021.

=== ML LEARNING NOTE: Why Three Data Sources? ===

In ML post-processing of weather forecasts, you need PAIRS of data:
  - What the model PREDICTED (the input/features)
  - What ACTUALLY HAPPENED (the target/label)

SNOTEL gives us ground truth. Archived forecasts give us what models predicted.
ERA5 reanalysis bridges the gap for years before archived forecasts exist
(pre-2021) — it's not a forecast, but it tells us what the atmosphere did at
grid scale, which helps us learn terrain-specific biases.

Think of it like grading a student's test:
  - Archived forecast = the student's answers
  - SNOTEL observation = the answer key
  - ERA5 reanalysis = a professor's reconstruction of what the right answer was

Usage:
    python historical_data_collector.py              # Collect all data
    python historical_data_collector.py --snotel     # SNOTEL only
    python historical_data_collector.py --era5       # ERA5 reanalysis only
    python historical_data_collector.py --forecasts  # Archived forecasts only
    python historical_data_collector.py --check      # Check existing data completeness
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resort_configs import RESORT_REGISTRY, get_active_resorts
from tahoe_snow import SNOTEL_STATIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# === Configuration ===

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historical")

# Date ranges
# ERA5 + SNOTEL: 10 years back from today
ERA5_START = "2016-01-01"
SNOTEL_START = "2016-01-01"

# Archived forecasts: available from ~late 2021
FORECAST_ARCHIVE_START = "2021-10-01"

# End date: yesterday (today's data may be incomplete)
END_DATE = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

# Open-Meteo rate limiting: free tier allows ~600 requests/hour
# We add delays and retry with exponential backoff on 429s and 500s
API_DELAY_SECONDS = 2.0  # 2s between requests (safe for free tier)
MAX_RETRIES = 6

# Open-Meteo variables
ERA5_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "snow_depth", "weather_code", "pressure_msl", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m",
]

ERA5_DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "snowfall_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
]

FORECAST_HOURLY_VARS = [
    "temperature_2m", "precipitation", "snowfall",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "relative_humidity_2m", "dew_point_2m", "pressure_msl",
    "cloud_cover", "snow_depth", "weather_code",
    # These are available in forecast API but not ERA5 archive:
    "freezing_level_height", "cape", "visibility",
]

# NWP models to fetch archived forecasts for
# NOTE: gfs_seamless/gfs025 return 500 on historical-forecast-api as of 2026-03.
# Using ncep_gfs013 (0.13° GFS, higher res) and gfs_hrrr (3km HRRR) instead.
FORECAST_MODELS = ["ncep_gfs013", "ecmwf_ifs025", "icon_seamless"]

# SNOTEL elements
SNOTEL_ELEMENTS = ["SNWD", "WTEQ", "TOBS", "TMAX", "TMIN", "PREC"]


def _api_get_with_retry(url: str, params: dict, label: str = "",
                        timeout: int = 120) -> dict | None:
    """
    GET request with exponential backoff retry on 429 (rate limit).

    === ML LEARNING NOTE: Robust Data Pipelines ===

    Real-world data collection ALWAYS hits API rate limits, timeouts, and
    transient errors. A robust pipeline retries with exponential backoff:
    wait 2s, then 4s, then 8s, etc. This is polite to the API provider
    and almost always succeeds within a few retries.

    The alternative — just failing and losing data — means your training
    dataset has gaps that could bias your model.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2 ** (attempt + 1), 90)
                logger.warning("    HTTP %d for %s, waiting %ds (attempt %d/%d)...",
                               resp.status_code, label, wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            wait = 2 ** (attempt + 1)
            logger.warning("    Timeout for %s, retrying in %ds...", label, wait)
            time.sleep(wait)
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("    Request failed for %s: %s, retrying in %ds...",
                               label, e, wait)
                time.sleep(wait)
            else:
                logger.error("    Failed after %d attempts for %s: %s",
                             MAX_RETRIES, label, e)
                return None
    return None


# =========================================================================
# SNOTEL Data Collection
# =========================================================================

def fetch_snotel_historical(station_name: str, station_info: dict,
                            start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical daily data for a single SNOTEL station.

    === ML LEARNING NOTE: Ground Truth Data ===

    SNOTEL (SNOwpack TELemetry) stations are automated weather stations
    maintained by the USDA Natural Resources Conservation Service (NRCS).
    They measure snow depth, snow water equivalent (SWE), temperature, and
    precipitation — exactly what we need as "ground truth" labels for
    training our ML model.

    Key variables:
      SNWD = Snow Depth (inches) — how deep is the snowpack?
      WTEQ = Snow Water Equivalent (inches) — how much water is in the snow?
      TOBS = Observed Temperature (F) — point-in-time temp
      TMAX/TMIN = Daily max/min temperature
      PREC = Accumulated Precipitation (inches, water year cumulative)

    Why SWE matters more than depth: Snow compresses over time (settling),
    so depth alone doesn't tell you how much new snow fell. But SWE
    (the amount of water if you melted the snowpack) only increases when
    new precipitation falls. Daily SWE change = new precipitation.

    Args:
        station_name: Human-readable station name.
        station_info: Dict with "id", "state", "elev_ft", "lat", "lon".
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with columns: date, station_name, station_id, elev_ft,
        lat, lon, SNWD, WTEQ, TOBS, TMAX, TMIN, PREC.
    """
    station_id = station_info["id"]
    state = station_info["state"]
    triplet = f"{station_id}:{state}:SNTL"

    logger.info("  Fetching SNOTEL %s (%s) from %s to %s",
                station_name, triplet, start_date, end_date)

    try:
        resp = requests.get(
            "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
            params={
                "stationTriplets": triplet,
                "elements": ",".join(SNOTEL_ELEMENTS),
                "beginDate": start_date,
                "endDate": end_date,
                "duration": "DAILY",
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error("  Failed to fetch SNOTEL %s: %s", station_name, e)
        return pd.DataFrame()

    if not data:
        logger.warning("  No data returned for %s", station_name)
        return pd.DataFrame()

    # Parse NRCS AWDB response format
    # Response is: [{"stationTriplet": ..., "data": [{"stationElement": {"elementCode": ...}, "values": [...]}]}]
    # The outer list has one entry per station; "data" contains elements.
    element_data = {}

    # Handle both formats: the response is a list with one station object
    station_obj = data[0] if isinstance(data, list) else data
    elements_list = station_obj.get("data", data)  # fallback to top-level list

    for elem_block in elements_list:
        station_elem = elem_block.get("stationElement", {})
        element_name = station_elem.get("elementCode", "") or station_elem.get("elementCd", "")
        values = elem_block.get("values", [])
        if not element_name or not values:
            continue

        for v in values:
            date_str = str(v.get("date", ""))[:10]  # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"
            value = v.get("value")
            if date_str:
                if date_str not in element_data:
                    element_data[date_str] = {}
                element_data[date_str][element_name] = value

    if not element_data:
        logger.warning("  No parseable data for %s", station_name)
        return pd.DataFrame()

    # Build DataFrame
    rows = []
    for date_str, elements in sorted(element_data.items()):
        row = {
            "date": date_str,
            "station_name": station_name,
            "station_id": station_id,
            "elev_ft": station_info["elev_ft"],
            "lat": station_info["lat"],
            "lon": station_info["lon"],
        }
        for elem in SNOTEL_ELEMENTS:
            row[elem] = elements.get(elem)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Basic quality control
    df = _snotel_quality_control(df, station_name)

    logger.info("  Got %d days for %s (%.1f%% coverage)",
                len(df), station_name,
                100 * len(df) / max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days))
    return df


def _snotel_quality_control(df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """
    Apply quality control filters to SNOTEL data.

    === ML LEARNING NOTE: Data Quality ===

    Raw sensor data is NEVER clean. SNOTEL stations have known issues:

    1. Snow bridging: Snow forms a bridge over the depth sensor, so depth
       reads as constant even as new snow falls. Detectable when SNWD is
       flat but WTEQ increases (water is being added but depth doesn't change).

    2. Sensor failures: Values of exactly 0 during winter, or huge jumps
       (depth changes 50+ inches in one day). These are clearly artifacts.

    3. Missing data: Gaps from equipment failure or telemetry issues.

    We don't delete bad data — we flag it with NaN so the ML model can
    handle it (XGBoost natively handles missing values, which is one reason
    we chose it).

    This step is CRITICAL. Training on bad data teaches the model wrong
    patterns. "Garbage in, garbage out" is the most fundamental ML rule.
    """
    n_before = len(df)

    # Flag impossible values as NaN
    if "SNWD" in df.columns:
        # Snow depth can't be negative or > 500 inches
        df.loc[(df["SNWD"] < 0) | (df["SNWD"] > 500), "SNWD"] = np.nan

    if "WTEQ" in df.columns:
        # SWE can't be negative or > 200 inches
        df.loc[(df["WTEQ"] < 0) | (df["WTEQ"] > 200), "WTEQ"] = np.nan

    if "TOBS" in df.columns:
        # Temperature outside -60F to 120F is sensor failure
        df.loc[(df["TOBS"] < -60) | (df["TOBS"] > 120), "TOBS"] = np.nan

    if "TMAX" in df.columns:
        df.loc[(df["TMAX"] < -60) | (df["TMAX"] > 120), "TMAX"] = np.nan

    if "TMIN" in df.columns:
        df.loc[(df["TMIN"] < -60) | (df["TMIN"] > 120), "TMIN"] = np.nan
        # TMIN should be <= TMAX
        if "TMAX" in df.columns:
            bad_minmax = df["TMIN"] > df["TMAX"] + 5  # 5F tolerance for timing
            df.loc[bad_minmax, ["TMIN", "TMAX"]] = np.nan

    # Flag snow bridging: SNWD flat for 5+ days while WTEQ increases
    if "SNWD" in df.columns and "WTEQ" in df.columns:
        snwd_diff = df["SNWD"].diff()
        wteq_diff = df["WTEQ"].diff()
        # Rolling 5-day check: depth hasn't changed but water has increased
        snwd_flat = snwd_diff.rolling(5).std() < 0.1  # Nearly constant depth
        wteq_up = wteq_diff.rolling(5).sum() > 0.5    # SWE increased > 0.5"
        bridging = snwd_flat & wteq_up
        n_bridging = bridging.sum()
        if n_bridging > 0:
            logger.info("    Flagged %d days of possible snow bridging at %s",
                        n_bridging, station_name)
            df.loc[bridging, "SNWD"] = np.nan  # Keep WTEQ, flag depth

    # Flag single-day depth jumps > 48 inches (physically unlikely)
    if "SNWD" in df.columns:
        depth_jump = df["SNWD"].diff().abs()
        big_jumps = depth_jump > 48
        n_jumps = big_jumps.sum()
        if n_jumps > 0:
            logger.info("    Flagged %d days with >48\" depth jumps at %s",
                        n_jumps, station_name)
            df.loc[big_jumps, "SNWD"] = np.nan

    return df


def collect_all_snotel(start_date: str = SNOTEL_START,
                       end_date: str = END_DATE) -> pd.DataFrame:
    """Fetch historical data for all 10 SNOTEL stations and combine."""
    logger.info("=" * 60)
    logger.info("COLLECTING SNOTEL DATA (%s to %s)", start_date, end_date)
    logger.info("=" * 60)

    all_dfs = []
    for name, info in SNOTEL_STATIONS.items():
        df = fetch_snotel_historical(name, info, start_date, end_date)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)  # Be polite to NRCS API

    if not all_dfs:
        logger.error("No SNOTEL data collected!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save
    outpath = os.path.join(DATA_DIR, "snotel_daily.parquet")
    combined.to_parquet(outpath, index=False)
    logger.info("Saved %d rows to %s (%.1f MB)",
                len(combined), outpath,
                os.path.getsize(outpath) / 1e6)

    # Also save CSV for easy inspection
    csv_path = os.path.join(DATA_DIR, "snotel_daily.csv")
    combined.to_csv(csv_path, index=False)
    logger.info("Also saved CSV to %s", csv_path)

    return combined


# =========================================================================
# Open-Meteo ERA5 Reanalysis Collection
# =========================================================================

def _get_all_zone_coords() -> list[dict]:
    """
    Get lat/lon/elev for every resort zone (base/mid/peak) across all
    active resorts. These are the points where we'll fetch weather data.

    === ML LEARNING NOTE: Spatial Sampling Strategy ===

    We fetch data at each resort zone's exact coordinates because weather
    varies dramatically with elevation and position relative to the Sierra
    crest. A temperature reading at 6,500 ft (Heavenly base) tells you
    something very different than 10,000 ft (Heavenly peak).

    By sampling at 3 elevations x 5 resorts = 15 points, we capture the
    elevation gradient that the ML model needs to learn. This is what makes
    our approach "hyperlocal" compared to a model like PEAKS that uses a
    uniform 3km grid.
    """
    coords = []
    for name, cfg in get_active_resorts().items():
        for zone in cfg.zones:
            coords.append({
                "resort": name,
                "zone": zone.name,
                "label": zone.label,
                "lat": zone.lat,
                "lon": zone.lon,
                "elev_ft": zone.elevation_ft,
                "aspect_deg": zone.aspect_deg,
                "slope_angle_deg": zone.slope_angle_deg,
            })
    return coords


def fetch_era5_for_point(lat: float, lon: float, label: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch ERA5 reanalysis data for a single lat/lon point.

    === ML LEARNING NOTE: Reanalysis vs Forecast ===

    ERA5 is a "reanalysis" — ECMWF takes all available observations
    (satellites, weather stations, radiosondes, buoys) and runs their
    model to produce a physically consistent reconstruction of what the
    atmosphere looked like. It's the closest thing to "truth" at grid scale.

    This is NOT a forecast. ERA5 for January 15, 2020 was computed AFTER
    January 15 using all available observations. It's what happened, not
    what was predicted.

    We use daily aggregates (not hourly) to match SNOTEL's daily reporting
    cadence. This simplifies alignment: one ERA5 row per day per point,
    one SNOTEL row per day per station.
    """
    logger.info("    ERA5 for %s (%.4f, %.4f)...", label, lat, lon)

    # Open-Meteo can handle large date ranges in a single call
    # but for 10 years of daily data, it's fine as one request
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(ERA5_DAILY_VARS),
        "timezone": "America/Los_Angeles",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }

    data = _api_get_with_retry(
        "https://archive-api.open-meteo.com/v1/archive",
        params, label=label,
    )
    if data is None:
        return pd.DataFrame()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        logger.warning("    No ERA5 data returned for %s", label)
        return pd.DataFrame()

    # Build DataFrame from response
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    for var in ERA5_DAILY_VARS:
        df[var] = daily.get(var, [None] * len(dates))

    logger.info("    Got %d days of ERA5 for %s", len(df), label)
    return df


def collect_all_era5(start_date: str = ERA5_START,
                     end_date: str = END_DATE) -> pd.DataFrame:
    """
    Fetch ERA5 reanalysis for all resort zone coordinates.

    This produces one big DataFrame with columns:
    resort, zone, lat, lon, elev_ft, date, <weather variables>
    """
    logger.info("=" * 60)
    logger.info("COLLECTING ERA5 REANALYSIS DATA (%s to %s)", start_date, end_date)
    logger.info("=" * 60)

    coords = _get_all_zone_coords()
    logger.info("Fetching data for %d zone coordinates", len(coords))

    all_dfs = []
    for i, coord in enumerate(coords):
        label = f"{coord['resort']}/{coord['zone']} ({coord['label']})"
        logger.info("  [%d/%d] %s", i + 1, len(coords), label)

        df = fetch_era5_for_point(
            coord["lat"], coord["lon"], label, start_date, end_date
        )

        if not df.empty:
            # Add location metadata columns
            df["resort"] = coord["resort"]
            df["zone"] = coord["zone"]
            df["lat"] = coord["lat"]
            df["lon"] = coord["lon"]
            df["elev_ft"] = coord["elev_ft"]
            df["aspect_deg"] = coord["aspect_deg"]
            df["slope_angle_deg"] = coord["slope_angle_deg"]
            all_dfs.append(df)

        time.sleep(API_DELAY_SECONDS)

    if not all_dfs:
        logger.error("No ERA5 data collected!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    outpath = os.path.join(DATA_DIR, "era5_daily.parquet")
    combined.to_parquet(outpath, index=False)
    logger.info("Saved %d rows to %s (%.1f MB)",
                len(combined), outpath,
                os.path.getsize(outpath) / 1e6)

    csv_path = os.path.join(DATA_DIR, "era5_daily.csv")
    combined.to_csv(csv_path, index=False)
    logger.info("Also saved CSV to %s", csv_path)

    return combined


# =========================================================================
# Open-Meteo Archived Forecast Collection
# =========================================================================

def fetch_archived_forecasts_for_point(
    lat: float, lon: float, label: str,
    model: str, start_date: str, end_date: str,
) -> pd.DataFrame:
    """
    Fetch archived NWP model forecasts for a single point and model.

    === ML LEARNING NOTE: The Core Training Data ===

    THIS is the most important data source for ML post-processing.

    The key insight: to train a model that corrects forecast errors, you
    need to know what the forecast SAID at the time — not what actually
    happened. That's what archived forecasts give you.

    Example training pair:
      Feature: "On Jan 15, 2023, GFS predicted 6\" of snow at Heavenly Peak"
      Label:   "SNOTEL measured 4\" of new snow that day"
      Lesson:  "GFS overestimated by 50% in these conditions"

    After seeing thousands of these pairs across different weather regimes,
    elevations, and seasons, the model learns WHEN and HOW MUCH each NWP
    model is biased — and can correct future forecasts.

    We fetch daily aggregates (sum of precip/snow, max/min temp) to match
    SNOTEL's daily cadence. In practice we chunk the date range into yearly
    segments because the API has response size limits.
    """
    logger.info("    Archived %s for %s (%.4f, %.4f)...", model, label, lat, lon)

    # Chunk into 6-month segments to keep response sizes small and reduce
    # server load (the historical-forecast-api is less robust than the archive API)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    all_dfs = []

    current = start
    while current < end:
        chunk_end = min(current + pd.DateOffset(months=6) - pd.DateOffset(days=1), end)
        chunk_start_str = current.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start_str,
            "end_date": chunk_end_str,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                     "precipitation_sum,snowfall_sum,rain_sum,"
                     "wind_speed_10m_max,wind_gusts_10m_max,"
                     "wind_direction_10m_dominant",
            "models": model,
            "timezone": "America/Los_Angeles",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
        }

        data = _api_get_with_retry(
            "https://historical-forecast-api.open-meteo.com/v1/forecast",
            params,
            label=f"{model}/{label} ({chunk_start_str} to {chunk_end_str})",
        )
        if data is None:
            current = chunk_end + pd.DateOffset(days=1)
            time.sleep(API_DELAY_SECONDS)
            continue

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if dates:
            df = pd.DataFrame({"date": pd.to_datetime(dates)})
            for var in daily:
                if var != "time":
                    df[var] = daily[var]
            all_dfs.append(df)

        current = chunk_end + pd.DateOffset(days=1)
        time.sleep(API_DELAY_SECONDS)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    # Drop duplicate dates (from overlapping chunks)
    combined = combined.drop_duplicates(subset=["date"], keep="first")
    logger.info("    Got %d days of %s for %s", len(combined), model, label)
    return combined


def collect_all_archived_forecasts(
    start_date: str = FORECAST_ARCHIVE_START,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    Fetch archived GFS/ECMWF/ICON forecasts for all resort zone coordinates.

    === ML LEARNING NOTE: Multi-Model Approach ===

    We fetch forecasts from 3 different NWP models (GFS, ECMWF, ICON)
    because each model has different biases:

    - GFS (American): Tends to overpredict precipitation at high elevations,
      but good at synoptic-scale pattern placement
    - ECMWF (European): Generally best skill globally, but can underpredict
      in complex terrain due to coarser orography
    - ICON (German): Strong in short-range, different orographic representation

    By training on all three models' forecasts, the ML correction model
    learns model-specific biases. This is exactly what the NWS MOS system
    has done since the 1970s — except we use gradient-boosted trees instead
    of linear regression, capturing nonlinear interactions.
    """
    logger.info("=" * 60)
    logger.info("COLLECTING ARCHIVED FORECASTS (%s to %s)", start_date, end_date)
    logger.info("Models: %s", ", ".join(FORECAST_MODELS))
    logger.info("=" * 60)

    coords = _get_all_zone_coords()
    total_requests = len(coords) * len(FORECAST_MODELS)
    logger.info("Fetching data for %d zone x %d models = %d series",
                len(coords), len(FORECAST_MODELS), total_requests)

    all_dfs = []
    request_num = 0

    for model in FORECAST_MODELS:
        logger.info("--- Model: %s ---", model)
        for i, coord in enumerate(coords):
            request_num += 1
            label = f"{coord['resort']}/{coord['zone']}"
            logger.info("  [%d/%d] %s (%s)", request_num, total_requests, label, model)

            df = fetch_archived_forecasts_for_point(
                coord["lat"], coord["lon"], label,
                model, start_date, end_date,
            )

            if not df.empty:
                df["resort"] = coord["resort"]
                df["zone"] = coord["zone"]
                df["model"] = model
                df["lat"] = coord["lat"]
                df["lon"] = coord["lon"]
                df["elev_ft"] = coord["elev_ft"]
                df["aspect_deg"] = coord["aspect_deg"]
                df["slope_angle_deg"] = coord["slope_angle_deg"]
                all_dfs.append(df)

    if not all_dfs:
        logger.error("No archived forecast data collected!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    outpath = os.path.join(DATA_DIR, "archived_forecasts_daily.parquet")
    combined.to_parquet(outpath, index=False)
    logger.info("Saved %d rows to %s (%.1f MB)",
                len(combined), outpath,
                os.path.getsize(outpath) / 1e6)

    csv_path = os.path.join(DATA_DIR, "archived_forecasts_daily.csv")
    combined.to_csv(csv_path, index=False)
    logger.info("Also saved CSV to %s", csv_path)

    return combined


# =========================================================================
# Data Completeness Check
# =========================================================================

def check_data_completeness():
    """
    Report on completeness of collected data.

    === ML LEARNING NOTE: Why Completeness Matters ===

    ML models learn from patterns in data. If your training data has
    systematic gaps — e.g., always missing data during storms because
    sensors fail in bad weather — the model learns a biased view of
    reality. It's like studying for a test but only reading the easy
    chapters.

    This check ensures we know WHERE our gaps are so we can:
    1. Fill small gaps (1-2 days) via interpolation
    2. Exclude stations/periods with too many gaps
    3. Understand if our model has blind spots
    """
    logger.info("=" * 60)
    logger.info("DATA COMPLETENESS CHECK")
    logger.info("=" * 60)

    # Check SNOTEL
    snotel_path = os.path.join(DATA_DIR, "snotel_daily.parquet")
    if os.path.exists(snotel_path):
        df = pd.read_parquet(snotel_path)
        logger.info("\nSNOTEL: %d total rows", len(df))
        for station in df["station_name"].unique():
            sdf = df[df["station_name"] == station]
            date_range = (sdf["date"].max() - sdf["date"].min()).days + 1
            coverage = len(sdf) / max(1, date_range) * 100
            snwd_valid = sdf["SNWD"].notna().sum() / len(sdf) * 100
            wteq_valid = sdf["WTEQ"].notna().sum() / len(sdf) * 100
            logger.info("  %-20s: %5d days, %5.1f%% coverage, SNWD valid: %5.1f%%, WTEQ valid: %5.1f%%",
                        station, len(sdf), coverage, snwd_valid, wteq_valid)
    else:
        logger.warning("  SNOTEL data not found at %s", snotel_path)

    # Check ERA5
    era5_path = os.path.join(DATA_DIR, "era5_daily.parquet")
    if os.path.exists(era5_path):
        df = pd.read_parquet(era5_path)
        logger.info("\nERA5 Reanalysis: %d total rows", len(df))
        for resort in sorted(df["resort"].unique()):
            rdf = df[df["resort"] == resort]
            zones = rdf["zone"].unique()
            logger.info("  %-20s: %d rows across zones: %s",
                        resort, len(rdf), ", ".join(sorted(zones)))
    else:
        logger.warning("  ERA5 data not found at %s", era5_path)

    # Check archived forecasts
    fcst_path = os.path.join(DATA_DIR, "archived_forecasts_daily.parquet")
    if os.path.exists(fcst_path):
        df = pd.read_parquet(fcst_path)
        logger.info("\nArchived Forecasts: %d total rows", len(df))
        for model in sorted(df["model"].unique()):
            mdf = df[df["model"] == model]
            n_points = mdf.groupby(["resort", "zone"]).ngroups
            date_range = f"{mdf['date'].min().date()} to {mdf['date'].max().date()}"
            logger.info("  %-20s: %d rows, %d resort/zone combos, %s",
                        model, len(mdf), n_points, date_range)
    else:
        logger.warning("  Archived forecasts not found at %s", fcst_path)


# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect historical weather data for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python historical_data_collector.py              # Collect everything
  python historical_data_collector.py --snotel     # SNOTEL only (fastest)
  python historical_data_collector.py --era5       # ERA5 reanalysis only
  python historical_data_collector.py --forecasts  # Archived model forecasts
  python historical_data_collector.py --check      # Check data completeness
        """,
    )
    parser.add_argument("--snotel", action="store_true",
                        help="Collect SNOTEL ground truth data only")
    parser.add_argument("--era5", action="store_true",
                        help="Collect ERA5 reanalysis data only")
    parser.add_argument("--forecasts", action="store_true",
                        help="Collect archived NWP forecast data only")
    parser.add_argument("--check", action="store_true",
                        help="Check data completeness without fetching")
    parser.add_argument("--start", type=str, default=None,
                        help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="Override end date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.check:
        check_data_completeness()
        return

    # If no specific source selected, collect everything
    collect_all = not (args.snotel or args.era5 or args.forecasts)
    start = args.start
    end = args.end or END_DATE

    t0 = time.time()

    if args.snotel or collect_all:
        collect_all_snotel(start or SNOTEL_START, end)

    if args.era5 or collect_all:
        collect_all_era5(start or ERA5_START, end)

    if args.forecasts or collect_all:
        collect_all_archived_forecasts(start or FORECAST_ARCHIVE_START, end)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("DATA COLLECTION COMPLETE in %.1f minutes", elapsed / 60)
    logger.info("=" * 60)

    # Always run completeness check at the end
    check_data_completeness()


if __name__ == "__main__":
    main()
