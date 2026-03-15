#!/usr/bin/env python3
"""
Tahoe Snow Conditions Analyzer — Full Feature Build

Features:
  P0: Multi-day forecast (15-day), hourly timeline (48h), powder alerts
  P1: Day/night accumulation, multi-model comparison, historical snow, wind chill
  P2: AI-style summary, avalanche conditions
  P3: HRRR model data, RWIS road weather, radar nowcast, CSSL enhanced
  P4: Terrain-aware downscaling, snow settling (Kojima 1967), precip phase
      probability (logistic transition), lake effect parameterization

Data sources (all free, public, no API keys except where noted):
  - Open-Meteo (GFS, ECMWF, ICON models — 15-day hourly forecasts)
  - NWS API (current observations, forecast, hourly forecast, gridpoints)
  - SNOTEL/NRCS (snowpack depth, SWE, temperature — current + historical)
  - CSSL/CDEC (Central Sierra Snow Lab — hourly snow, wind at Donner Summit)
  - Avalanche.org (Sierra Avalanche Center danger ratings)
  - NWS Reno WFO (Area Forecast Discussion)
  - HRRR via Herbie (optional, --hrrr flag — requires herbie-data package)
  - RWIS road weather via Synoptic API (requires SYNOPTIC_TOKEN env var)
  - Open-Meteo ensemble forecasts (GFS 30-member, ECMWF 50-member)
  - Radar nowcast via Open-Meteo current conditions

Physics models:
  - SLR: Roebber (2003) temperature-dependent with wind/humidity corrections
  - Wet-bulb: Stull (2011) approximation for precip type classification
  - Precip phase: Logistic transition probability (snow/mix/rain)
  - Terrain: Aspect-based diurnal temperature correction (N/S/E/W facing)
  - Settling: Kojima (1967) exponential compaction model
  - Lake effect: Tahoe fetch-based snowfall enhancement (NW wind + cold air)
  - Orographic: Aspect + elevation + wind + CAPE enhancement factor

Usage:
  python tahoe_snow.py              # full report
  python tahoe_snow.py --json       # JSON output
  python tahoe_snow.py --hrrr       # include HRRR model data
  python tahoe_snow.py --compact    # shorter report (no hourly timeline)
"""

import json
import logging
import os
import re
import sys
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from resort_configs import get_active_resorts_legacy, RESORT_REGISTRY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resort Presets
# ---------------------------------------------------------------------------

RESORTS = get_active_resorts_legacy()

SNOTEL_STATIONS = {
    "Tahoe City Cross":  {"id": "809", "state": "CA", "elev_ft": 6230},
    "Fallen Leaf":       {"id": "473", "state": "CA", "elev_ft": 6250},
    "Hagan's Meadow":    {"id": "518", "state": "CA", "elev_ft": 8200},
    "Independence Lake": {"id": "539", "state": "CA", "elev_ft": 7000},
    "Independence Camp": {"id": "540", "state": "CA", "elev_ft": 6900},
    "Rubicon #2":        {"id": "724", "state": "CA", "elev_ft": 6700},
    "Squaw Valley GC":   {"id": "784", "state": "CA", "elev_ft": 6200},
    "Ward Creek #3":     {"id": "848", "state": "NV", "elev_ft": 6600},
    "Mt Rose Ski Area":  {"id": "652", "state": "NV", "elev_ft": 8790},
    "CSS Lab":           {"id": "428", "state": "CA", "elev_ft": 6890},
}

DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
    "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
    "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
    "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gfs_hrrr"]
MODEL_LABELS = {"gfs_seamless": "GFS", "ecmwf_ifs025": "ECMWF", "icon_seamless": "ICON", "gfs_hrrr": "HRRR"}

# Water year starts Oct 1
def water_year_start() -> str:
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-10-01"


# ---------------------------------------------------------------------------
# Snow Physics
# ---------------------------------------------------------------------------

def compute_slr(temp_c: float, wind_mph: float = 0, rh_pct: float | None = None) -> float:
    """Temperature-dependent snow-to-liquid ratio with wind/humidity corrections.

    Base: Roebber (2003) temperature-dependent SLR.
    Wind correction: Strong winds break snow crystal dendrites, reducing SLR
    (Judson & Doesken 2000). >30 mph reduces SLR 10-25%.
    Humidity correction: Low RH (<50%) allows sublimation during fall,
    slightly increasing density (lower SLR). High RH preserves crystal
    structure (higher SLR).
    """
    # Base Roebber temperature curve
    if temp_c <= -18:
        slr = 20.0 + min(5.0, (-18 - temp_c) * 0.5)
    elif temp_c <= -12:
        slr = 15.0 + (-12 - temp_c) * (5.0 / 6.0)
    elif temp_c <= -6:
        slr = 12.0 + (-6 - temp_c) * (3.0 / 6.0)
    elif temp_c <= -1:
        slr = 8.0 + (-1 - temp_c) * (4.0 / 5.0)
    elif temp_c <= 0:
        slr = 5.0 + (0 - temp_c) * 3.0
    else:
        slr = max(1.0, 5.0 - temp_c * 2.0)

    # Wind correction: strong wind breaks dendrites (Judson & Doesken 2000)
    if wind_mph > 15:
        wind_factor = max(0.75, 1.0 - (wind_mph - 15) / 100.0)
        slr *= wind_factor

    # Humidity correction: dry air reduces SLR via sublimation
    if rh_pct is not None:
        if rh_pct < 50:
            slr *= 0.90 + (rh_pct / 50.0) * 0.10  # 0.90-1.0 factor
        elif rh_pct > 85:
            slr *= 1.0 + min(0.05, (rh_pct - 85) / 300.0)  # up to +5%

    return slr


def wind_chill_f(temp_f: float, wind_mph: float) -> float:
    """NWS wind chill formula. Valid for T<=50F, V>=3mph."""
    if temp_f > 50 or wind_mph < 3:
        return temp_f
    wc = (35.74 + 0.6215 * temp_f
          - 35.75 * (wind_mph ** 0.16)
          + 0.4275 * temp_f * (wind_mph ** 0.16))
    return round(wc, 1)


def orographic_multiplier(elev_ft: float, wind_mph: float, wind_dir: float,
                          cape_jkg: float = 0) -> float:
    """Orographic precipitation enhancement factor.

    Combines:
    - Aspect alignment (ideal: WSW at 247.5° for Sierra Nevada)
    - Elevation factor (higher = more orographic lift)
    - Wind speed factor (stronger flow = more forced ascent)
    - CAPE coupling (convective instability enhances orographic precip)
      Reference: Kirshbaum & Smith (2008) — CAPE-terrain interaction
    """
    ideal = 247.5
    diff = abs(wind_dir - ideal)
    if diff > 180:
        diff = 360 - diff
    d = max(0.3, 1.0 - (diff / 180.0) * 0.7)
    e = 1.0 + max(0.0, (elev_ft - 6225) / (10000 - 6225)) * 0.5
    w = 1.0 + min(0.3, wind_mph / 100.0)

    # CAPE coupling: convective instability amplifies orographic lift
    # CAPE > 100 J/kg rare in winter but common during transitional storms
    cape = cape_jkg or 0
    c = 1.0 + min(0.4, cape / 500.0) if cape > 50 else 1.0

    return d * e * w * c


def compute_lapse_rate(snotel: dict, synoptic: dict | None = None,
                       sounding: dict | None = None) -> float | None:
    """Compute actual lapse rate (C/km) from multi-source observations.

    Data fusion priority:
    1. Fresh radiosonde sounding (<6h) — measured free atmosphere profile (best)
    2. SNOTEL + Synoptic surface stations — wide elevation spread (good)
    3. SNOTEL only — baseline (adequate)

    Returns the observed lapse rate (positive = temp decreases with altitude),
    or None if insufficient data. Typical range: 4-9 C/km.
    Inversions (negative lapse) are common in Tahoe — this detects them.
    """
    # Check for fresh sounding first (best source — free atmosphere, not surface)
    sounding_lapse = None
    sounding_fresh = False
    if sounding and "error" not in sounding and sounding.get("lapse_rate_c_km") is not None:
        sounding_lapse = sounding["lapse_rate_c_km"]
        # Check freshness
        if sounding.get("time"):
            try:
                st = datetime.fromisoformat(sounding["time"])
                age_h = (datetime.now(timezone.utc) - st).total_seconds() / 3600
                sounding_fresh = age_h <= 6
            except (ValueError, TypeError):
                pass

    # Collect surface station observations
    points = []  # (elev_m, temp_c)
    for name, st_info in SNOTEL_STATIONS.items():
        data = snotel.get(name, {})
        temp_f = data.get("temp_f")
        if temp_f is not None and "error" not in data:
            elev_m = st_info["elev_ft"] * 0.3048
            temp_c = (temp_f - 32) * 5 / 9
            points.append((elev_m, temp_c))

    # Incorporate Synoptic/MesoWest stations for better elevation spread
    if synoptic and "error" not in synoptic:
        for stn in synoptic.get("stations", []):
            temp_f = stn.get("temp_f")
            elev_ft = stn.get("elev_ft")
            if temp_f is not None and elev_ft is not None and elev_ft > 4000:
                elev_m = elev_ft * 0.3048
                temp_c = (temp_f - 32) * 5 / 9
                points.append((elev_m, temp_c))

    # Compute surface-based lapse rate
    surface_lapse = None
    if len(points) >= 3:
        elevs = np.array([p[0] for p in points])
        temps = np.array([p[1] for p in points])
        slope, _ = np.polyfit(elevs, temps, 1)
        surface_lapse = -slope * 1000.0
        surface_lapse = max(-3.0, min(12.0, surface_lapse))

    # Fusion: blend sounding and surface observations
    if sounding_fresh and sounding_lapse is not None:
        if surface_lapse is not None:
            # Fresh sounding + surface: 60% sounding (free air), 40% surface
            # Surface stations are affected by cold-air pooling; sounding is not
            blended = sounding_lapse * 0.6 + surface_lapse * 0.4
        else:
            blended = sounding_lapse
    elif sounding_lapse is not None and surface_lapse is not None:
        # Aging sounding + surface: 40% sounding, 60% surface
        blended = sounding_lapse * 0.4 + surface_lapse * 0.6
    elif surface_lapse is not None:
        blended = surface_lapse
    elif sounding_lapse is not None:
        blended = sounding_lapse
    else:
        return None

    blended = max(-3.0, min(12.0, blended))
    return round(blended, 2)


def estimate_temp_c(base_c: float, base_m: float, target_m: float,
                    saturated: bool = True, observed_lapse_rate: float | None = None) -> float:
    """Estimate temperature at target elevation using best available lapse rate."""
    if observed_lapse_rate is not None:
        lapse = observed_lapse_rate / 1000.0
    else:
        lapse = 5.5 / 1000.0 if saturated else 6.5 / 1000.0
    return base_c - ((target_m - base_m) * lapse)


def snow_quality_str(slr: float) -> str:
    if slr >= 18: return "Blower pow (cold smoke)"
    elif slr >= 14: return "Light dry powder"
    elif slr >= 11: return "Classic powder"
    elif slr >= 8: return "Packable powder"
    elif slr >= 5: return "Sierra cement"
    else: return "Wet / slushy"


def wind_dir_str(deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[round(deg / 22.5) % 16]


def wet_bulb_temp_c(temp_c: float, rh_pct: float | None) -> float:
    """Approximate wet-bulb temperature using Stull (2011) formula.

    More accurate than dry-bulb for precipitation type classification because
    evaporative cooling of falling hydrometeors determines phase at ground level.
    Valid for RH 5-99%, T -20 to 50°C. Accuracy ~0.3°C.
    """
    if rh_pct is None or rh_pct < 5:
        return temp_c  # Fall back to dry-bulb if no humidity
    rh = max(5, min(99, rh_pct))
    t = temp_c
    tw = (t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
          + math.atan(t + rh)
          - math.atan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
          - 4.686035)
    return tw


def precip_type(temp_c: float, has_precip: bool,
                rh_pct: float | None = None) -> str:
    """Classify precipitation type using wet-bulb temperature.

    Uses wet-bulb temperature (Stull 2011) when humidity is available,
    which accounts for evaporative cooling of falling precipitation.
    This is the method used by NWS and research meteorologists.
    Falls back to dry-bulb thresholds when humidity is unavailable.
    """
    if not has_precip:
        return "None"
    # Use wet-bulb if humidity available (meteorologically correct)
    t_wb = wet_bulb_temp_c(temp_c, rh_pct) if rh_pct is not None else temp_c
    if t_wb <= -1.0:
        return "Snow"
    elif t_wb <= 1.5:
        return "Mix"
    else:
        return "Rain"


def precip_phase_probability(wet_bulb_c: float, elevation_m: float,
                              freezing_level_m: float | None) -> dict:
    """Return probability of snow/mix/rain using logistic transition.

    Instead of hard cutoffs (Snow/Mix/Rain), returns a continuous probability
    distribution across precipitation phases. Uses logistic sigmoid centered
    at 0.5C wet-bulb with width 1.5C, plus elevation corrections relative
    to the freezing level.

    Args:
        wet_bulb_c: Wet-bulb temperature in Celsius
        elevation_m: Station elevation in meters
        freezing_level_m: Freezing level height in meters (or None)

    Returns:
        dict with p_snow, p_mix, p_rain (0-1), and dominant_type string
    """
    # Logistic sigmoid for snow probability
    # Center at 0.5C, width 1.5C — transition zone roughly -1C to 2C
    k = 1.0 / 1.5  # steepness
    center = 0.5
    # p_snow decreases as wet_bulb increases
    logistic = 1.0 / (1.0 + math.exp(k * (wet_bulb_c - center)))

    # Base probabilities from logistic
    if wet_bulb_c < -4.0:
        p_snow = 0.95
        p_rain = 0.01
    elif wet_bulb_c > 5.0:
        p_snow = 0.01
        p_rain = 0.95
    else:
        p_snow = logistic * 0.95
        p_rain = (1.0 - logistic) * 0.95

    # Elevation correction relative to freezing level
    if freezing_level_m is not None:
        elev_diff = elevation_m - freezing_level_m
        if elev_diff > 300:
            # Well above freezing level — boost snow
            p_snow = min(1.0, p_snow + 0.1)
            p_rain = max(0.0, p_rain - 0.1)
        elif elev_diff < -300:
            # Well below freezing level — boost rain
            p_rain = min(1.0, p_rain + 0.1)
            p_snow = max(0.0, p_snow - 0.1)

    # Mix gets the remainder
    p_mix = max(0.0, 1.0 - p_snow - p_rain)

    # Normalize to sum to 1.0
    total = p_snow + p_mix + p_rain
    if total > 0:
        p_snow /= total
        p_mix /= total
        p_rain /= total

    # Determine dominant type
    if p_snow >= p_mix and p_snow >= p_rain:
        dominant = "Snow"
    elif p_rain >= p_snow and p_rain >= p_mix:
        dominant = "Rain"
    else:
        dominant = "Mix"

    return {
        "p_snow": round(p_snow, 3),
        "p_mix": round(p_mix, 3),
        "p_rain": round(p_rain, 3),
        "dominant_type": dominant,
    }


def terrain_adjusted_temperature(base_temp_c: float, elevation_m: float,
                                  aspect_degrees: float, hour_of_day: int,
                                  cloud_cover_pct: float = 50.0) -> float:
    """Apply diurnal temperature correction based on slope aspect.

    Terrain aspect significantly affects micro-climate temperatures:
    - South-facing slopes receive more solar radiation (warmer days)
    - North-facing slopes retain cold air and snow (cooler days)
    - East/west aspects shift the diurnal temperature peak

    Corrections are scaled by (1 - cloud_cover/100) since overcast
    conditions eliminate aspect-driven solar heating differences.

    Args:
        base_temp_c: Base temperature at this elevation (C)
        elevation_m: Elevation in meters (unused, reserved for future)
        aspect_degrees: Slope aspect in degrees (0=N, 90=E, 180=S, 270=W)
        hour_of_day: Hour of day (0-23, local time)
        cloud_cover_pct: Cloud cover percentage (0-100)

    Returns:
        Adjusted temperature in Celsius
    """
    # Normalize aspect to 0-360
    aspect = aspect_degrees % 360
    cloud_factor = max(0.0, 1.0 - cloud_cover_pct / 100.0)

    # Determine if it's daytime (6am-6pm) and solar peak (noon-3pm)
    is_day = 6 <= hour_of_day < 18
    is_morning = 6 <= hour_of_day < 12
    is_afternoon = 12 <= hour_of_day < 18
    is_peak_solar = 12 <= hour_of_day < 15

    correction = 0.0

    # South-facing (135-225 degrees)
    if 135 <= aspect <= 225:
        if is_peak_solar:
            correction = 2.0
        elif is_day:
            correction = 1.0
        else:
            correction = -0.5
    # North-facing (315-360 or 0-45 degrees)
    elif aspect >= 315 or aspect <= 45:
        if is_day:
            correction = -1.0
        else:
            correction = 0.5  # radiative cooling trapped in valleys
    # East-facing (45-135 degrees)
    elif 45 < aspect < 135:
        if is_morning:
            correction = 1.0
        elif is_afternoon:
            correction = -0.5
        else:
            correction = 0.0
    # West-facing (225-315 degrees)
    elif 225 < aspect < 315:
        if is_morning:
            correction = -0.5
        elif is_afternoon:
            correction = 1.5
        else:
            correction = 0.0

    # Scale by cloud factor — clouds eliminate aspect effects
    correction *= cloud_factor

    return base_temp_c + correction


def settled_snow_depth(fresh_snow_inches: float, hours_since_fall: float,
                       temp_f: float, wind_mph: float) -> float:
    """Estimate settled snow depth using Kojima (1967) exponential decay.

    Fresh snow compacts under its own weight. The rate depends on:
    - Temperature: warmer snow settles faster (metamorphism + melt)
    - Wind: wind packs and densifies surface snow
    - Time: exponential decay — fastest settling in first hours

    Args:
        fresh_snow_inches: Initial fresh snowfall in inches
        hours_since_fall: Hours since the snow fell
        temp_f: Current temperature in Fahrenheit
        wind_mph: Wind speed in mph

    Returns:
        Estimated remaining snow depth in inches
    """
    if fresh_snow_inches <= 0 or hours_since_fall <= 0:
        return fresh_snow_inches

    # Compaction rate: base 0.08/day, increases with warmth
    compaction_rate = 0.08 + 0.005 * max(0, temp_f - 15)

    # Wind factor: wind packing increases compaction
    wind_factor = 1.0 + 0.02 * min(wind_mph, 30)

    # Exponential decay (Kojima 1967)
    remaining_fraction = math.exp(-compaction_rate * wind_factor * hours_since_fall / 24.0)

    # Clamp: can't settle more than 40% in first 24h typically
    remaining_fraction = max(0.60, remaining_fraction)

    return round(fresh_snow_inches * remaining_fraction, 1)


def lake_effect_enhancement(wind_dir_deg: float, wind_speed_mph: float,
                             air_temp_c: float,
                             lake_temp_c: float = 4.5) -> float:
    """Compute Lake Tahoe lake-effect snowfall enhancement factor.

    Lake Tahoe (mean winter surface ~4.5C/39F) can enhance snowfall on
    its east shore when cold NW-N winds blow across the lake's ~12-mile
    fetch. This is a modest effect compared to Great Lakes lake-effect
    but measurable in cold outbreaks.

    Conditions for lake effect:
    - Wind direction: NW to N (300-360, 0-30) — fetch across the lake
    - Lake-air temperature difference > 10C
    - Wind speed 10-30 mph (calm=no flux, strong=mechanical mixing)

    Returns:
        Enhancement factor (1.0 = no effect, up to 1.3 = 30% more snow)
    """
    # Temperature difference factor (requires >10C lake-air diff)
    temp_diff = lake_temp_c - air_temp_c
    temp_factor = min(1.0, max(0.0, (temp_diff - 10.0) / 10.0))

    # Direction factor: cosine alignment to optimal NNW fetch (345 degrees)
    # Full width at half max ~40 degrees
    optimal_dir = 345.0
    dir_diff = abs(wind_dir_deg - optimal_dir)
    if dir_diff > 180:
        dir_diff = 360 - dir_diff
    # Gaussian-like: sigma ~40 degrees
    dir_factor = math.exp(-(dir_diff ** 2) / (2 * 40.0 ** 2))

    # Wind speed factor: bell curve centered at 20 mph, sigma 8 mph
    speed_factor = math.exp(-((wind_speed_mph - 20.0) ** 2) / (2 * 8.0 ** 2))

    # Combined enhancement: up to 30% more snow
    enhancement = 1.0 + 0.3 * temp_factor * dir_factor * speed_factor

    return round(enhancement, 3)


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_nws_observations(lat: float, lon: float) -> dict:
    """Current conditions from nearest NWS station."""
    headers = {"User-Agent": "TahoeSnowStation/1.0 (keith@local)"}
    try:
        resp = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
                            headers=headers, timeout=10)
        if resp.status_code != 200:
            return {}
        obs_url = resp.json().get("properties", {}).get("observationStations", "")
        if not obs_url:
            return {}
        stations = requests.get(obs_url, headers=headers, timeout=10).json().get("features", [])
        for station in stations[:3]:
            sid = station["properties"]["stationIdentifier"]
            obs = requests.get(f"https://api.weather.gov/stations/{sid}/observations/latest",
                               headers=headers, timeout=10)
            if obs.status_code == 200:
                p = obs.json().get("properties", {})
                tc = p.get("temperature", {}).get("value")
                if tc is not None:
                    ws = (p.get("windSpeed", {}).get("value") or 0) * 0.621371
                    wg = (p.get("windGust", {}).get("value") or 0) * 0.621371
                    wd = p.get("windDirection", {}).get("value") or 0
                    tf = round(tc * 9 / 5 + 32, 1)
                    return {
                        "station": sid,
                        "conditions": p.get("textDescription", ""),
                        "temp_f": tf, "temp_c": round(tc, 1),
                        "humidity_pct": round(p.get("relativeHumidity", {}).get("value") or 0),
                        "wind_mph": round(ws, 1), "wind_gust_mph": round(wg, 1),
                        "wind_dir_deg": wd, "wind_dir": wind_dir_str(wd),
                        "feels_like_f": wind_chill_f(tf, ws),
                        "visibility_mi": round((p.get("visibility", {}).get("value") or 0) / 1609.34, 1),
                        "barometer_inhg": round((p.get("barometricPressure", {}).get("value") or 0) / 3386.39, 2),
                        "timestamp": p.get("timestamp", ""),
                    }
        return {}
    except Exception as e:
        logger.warning("fetch_nws_observations failed: %s", e)
        return {}


def fetch_nws_forecast(lat: float, lon: float) -> dict:
    """7-day periods + 48h hourly from NWS."""
    headers = {"User-Agent": "TahoeSnowStation/1.0 (keith@local)"}
    try:
        resp = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
                            headers=headers, timeout=10)
        if resp.status_code != 200:
            return {}
        props = resp.json().get("properties")
        if not props:
            return {}
        result = {}
        fr = requests.get(props["forecast"], headers=headers, timeout=10)
        if fr.status_code == 200:
            result["periods"] = fr.json().get("properties", {}).get("periods", [])
        hr = requests.get(props["forecastHourly"], headers=headers, timeout=10)
        if hr.status_code == 200:
            result["hourly"] = hr.json().get("properties", {}).get("periods", [])[:156]  # up to 6.5 days
        return result
    except Exception as e:
        logger.warning("fetch_nws_forecast failed: %s", e)
        return {}


def fetch_open_meteo(lat: float, lon: float) -> dict:
    """Multi-model 16-day hourly forecast from Open-Meteo (GFS, ECMWF, ICON)."""
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join([
                "temperature_2m", "precipitation", "snowfall", "snow_depth",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "freezing_level_height", "weather_code", "cloud_cover",
                "relative_humidity_2m", "dew_point_2m",
                "apparent_temperature", "pressure_msl", "visibility", "cape",
            ]),
            "daily": "sunrise,sunset",
            "models": ",".join(MODELS),
            "forecast_days": 16,
            "timezone": "America/Los_Angeles",
        }
        resp = requests.get("https://api.open-meteo.com/v1/forecast",
                            params=params, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def fetch_open_meteo_multi(locations: dict) -> dict:
    """
    Fetch Open-Meteo for multiple lat/lon points in one API call.
    Open-Meteo supports comma-separated lat/lon for multi-point queries.
    Returns dict keyed by location name.
    """
    names = list(locations.keys())
    lats = ",".join(str(locations[n]["lat"]) for n in names)
    lons = ",".join(str(locations[n]["lon"]) for n in names)
    try:
        params = {
            "latitude": lats, "longitude": lons,
            "hourly": ",".join([
                "temperature_2m", "precipitation", "snowfall", "snow_depth",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "freezing_level_height", "weather_code", "cloud_cover",
                "relative_humidity_2m", "dew_point_2m",
                "apparent_temperature", "pressure_msl", "visibility", "cape",
            ]),
            "daily": "sunrise,sunset",
            "models": ",".join(MODELS),
            "forecast_days": 16,
            "timezone": "America/Los_Angeles",
        }
        resp = requests.get("https://api.open-meteo.com/v1/forecast",
                            params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # Multi-point returns a list; single point returns a dict
            if isinstance(data, list):
                return {names[i]: data[i] for i in range(min(len(names), len(data)))}
            else:
                return {names[0]: data}
        return {n: {"error": f"HTTP {resp.status_code}"} for n in names}
    except Exception as e:
        return {n: {"error": str(e)} for n in names}


def fetch_snotel_current() -> dict:
    """Current SNOTEL readings."""
    results = {}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    for name, st in SNOTEL_STATIONS.items():
        try:
            resp = requests.get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
                params={"stationTriplets": f"{st['id']}:{st['state']}:SNTL",
                         "elements": "SNWD,WTEQ,TOBS", "beginDate": yesterday,
                         "endDate": today, "duration": "DAILY"}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                sd = {"name": name, "elev_ft": st["elev_ft"]}
                if data:
                    elements = data[0].get("data", data) if isinstance(data[0], dict) and "data" in data[0] else data
                    for el in elements:
                        code = el.get("stationElement", {}).get("elementCode", "")
                        vals = el.get("values", [])
                        if vals:
                            v = vals[-1].get("value")
                            if code == "SNWD": sd["snow_depth_in"] = v
                            elif code == "WTEQ": sd["swe_in"] = v
                            elif code == "TOBS": sd["temp_f"] = v
                results[name] = sd
            else:
                results[name] = {"name": name, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            results[name] = {"name": name, "error": str(e)}
    return results


def fetch_snotel_history(station_id: str, state: str, days: int = 10) -> list:
    """Fetch N days of SNOTEL history for one station."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        resp = requests.get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
            params={"stationTriplets": f"{station_id}:{state}:SNTL",
                     "elements": "SNWD,WTEQ", "beginDate": start,
                     "endDate": today, "duration": "DAILY"}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                result = {}
                elements = data[0].get("data", data) if isinstance(data[0], dict) and "data" in data[0] else data
                for el in elements:
                    code = el.get("stationElement", {}).get("elementCode", "")
                    result[code] = [(v["date"], v["value"]) for v in el.get("values", []) if v.get("value") is not None]
                return result
        return {}
    except Exception as e:
        logger.warning("fetch_snotel_history(%s) failed: %s", station_id, e)
        return {}


def fetch_snotel_season(station_id: str, state: str) -> dict:
    """Fetch season stats for one station (water year to date)."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        resp = requests.get("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
            params={"stationTriplets": f"{station_id}:{state}:SNTL",
                     "elements": "SNWD,WTEQ", "beginDate": water_year_start(),
                     "endDate": today, "duration": "DAILY"}, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                stats = {}
                elements = data[0].get("data", data) if isinstance(data[0], dict) and "data" in data[0] else data
                for el in elements:
                    code = el.get("stationElement", {}).get("elementCode", "")
                    vals = [v["value"] for v in el.get("values", []) if v.get("value") is not None]
                    if vals:
                        stats[code] = {"current": vals[-1], "peak": max(vals), "days": len(vals)}
                return stats
        return {}
    except Exception as e:
        logger.warning("fetch_snotel_season(%s) failed: %s", station_id, e)
        return {}


def fetch_nws_gridpoints(lat: float, lon: float) -> dict:
    """Fetch raw NWS gridpoint forecast data — snowfall amounts, snow levels, QPF.

    Unlike /forecast (text periods) or /forecastHourly (simplified hourly),
    the raw gridpoints endpoint has forecaster-edited numerical grids:
      - snowfallAmount: 6-hour snow accumulation (mm → inches)
      - snowLevel: snow level height (m → ft)
      - quantitativePrecipitation: liquid equivalent QPF (mm → inches)
      - probabilityOfPrecipitation: precip probability (%)
      - iceAccumulation: ice accumulation (mm → inches)
    """
    headers = {"User-Agent": "TahoeSnowStation/1.0 (keith@local)"}
    try:
        resp = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
                            headers=headers, timeout=10)
        if resp.status_code != 200:
            return {"error": f"points HTTP {resp.status_code}"}
        props = resp.json().get("properties", {})
        wfo = props.get("gridId", "")
        grid_x = props.get("gridX")
        grid_y = props.get("gridY")
        if not wfo or grid_x is None:
            return {"error": "No grid coordinates"}

        grid_resp = requests.get(
            f"https://api.weather.gov/gridpoints/{wfo}/{grid_x},{grid_y}",
            headers=headers, timeout=15)
        if grid_resp.status_code != 200:
            return {"error": f"gridpoints HTTP {grid_resp.status_code}"}

        grid_props = grid_resp.json().get("properties", {})
        result = {"wfo": wfo, "gridX": grid_x, "gridY": grid_y}

        for field in ("snowfallAmount", "snowLevel", "quantitativePrecipitation",
                       "probabilityOfPrecipitation", "iceAccumulation"):
            field_data = grid_props.get(field, {})
            uom = field_data.get("uom", "")
            values = field_data.get("values", [])
            parsed = []
            for v in values:
                val = v.get("value")
                if val is None:
                    continue
                valid_time = v.get("validTime", "")
                parts = valid_time.split("/")
                start = parts[0] if parts else ""
                duration_str = parts[1] if len(parts) > 1 else "PT1H"
                # Parse duration to hours
                dur_match = re.match(r"PT?(\d+)H", duration_str)
                dur_hours = int(dur_match.group(1)) if dur_match else 1

                # Convert units
                if field == "snowLevel":
                    converted = round(val * 3.28084)  # m → ft
                elif field == "probabilityOfPrecipitation":
                    converted = val  # already %
                else:
                    converted = round(val / 25.4, 2)  # mm → inches

                parsed.append({
                    "start": start,
                    "hours": dur_hours,
                    "value": converted,
                })
            result[field] = parsed

        return result
    except Exception as e:
        return {"error": str(e)}


def fetch_nbm(lat: float, lon: float) -> dict:
    """Fetch National Blend of Models (NBM) via Open-Meteo.

    NBM is NWS's bias-corrected multi-model consensus — typically more accurate
    than any single model for US locations. Provides temp, precip, and
    precipitation probability.
    """
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join([
                "temperature_2m", "precipitation", "precipitation_probability",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            ]),
            "models": "ncep_nbm_conus",
            "forecast_days": 7,
            "timezone": "America/Los_Angeles",
        }
        resp = requests.get("https://api.open-meteo.com/v1/forecast",
                            params=params, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# Well-known Weather.com public API key (used by WU web widgets)
_WU_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"


def fetch_pws_nearby(lat: float, lon: float, max_stations: int = 5) -> list[dict]:
    """Fetch current observations from nearby Weather Underground personal stations.

    Averages multiple PWS readings for better ground truth than a single
    NWS airport station. Free, no API key registration needed.
    """
    try:
        # Step 1: Find nearby stations
        resp = requests.get("https://api.weather.com/v3/location/near", params={
            "geocode": f"{lat},{lon}",
            "product": "pws",
            "format": "json",
            "apiKey": _WU_API_KEY,
        }, timeout=10)
        if resp.status_code != 200:
            return []

        loc = resp.json().get("location", {})
        station_ids = loc.get("stationId", [])[:max_stations]
        if not station_ids:
            return []

        # Step 2: Fetch observations from each station
        results = []
        for sid in station_ids:
            try:
                obs_resp = requests.get(
                    "https://api.weather.com/v2/pws/observations/current",
                    params={
                        "stationId": sid,
                        "format": "json",
                        "units": "e",  # imperial
                        "apiKey": _WU_API_KEY,
                    }, timeout=8)
                if obs_resp.status_code == 200:
                    obs_data = obs_resp.json().get("observations", [{}])[0]
                    imp = obs_data.get("imperial", {})
                    results.append({
                        "station_id": sid,
                        "temp_f": imp.get("temp"),
                        "humidity_pct": obs_data.get("humidity"),
                        "wind_mph": imp.get("windSpeed"),
                        "wind_gust_mph": imp.get("windGust"),
                        "pressure_inhg": imp.get("pressure"),
                        "precip_rate_in": imp.get("precipRate"),
                        "precip_total_in": imp.get("precipTotal"),
                        "solar_radiation": obs_data.get("solarRadiation"),
                        "uv": obs_data.get("uv"),
                        "timestamp": obs_data.get("obsTimeLocal"),
                    })
            except Exception as e:
                logger.debug("PWS station %s fetch failed: %s", sid, e)
                continue
        return results
    except Exception as e:
        logger.warning("fetch_pws_nearby failed: %s", e)
        return []


def aggregate_pws(stations: list[dict]) -> dict:
    """Aggregate multiple PWS readings into a consensus observation.

    Uses median for temperature (robust to outlier stations) and
    averages for other metrics.
    """
    if not stations:
        return {}

    def _median(vals):
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return None
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    temps = [s["temp_f"] for s in stations]
    return {
        "temp_f": _median(temps),
        "humidity_pct": _median([s.get("humidity_pct") for s in stations]),
        "wind_mph": _mean([s.get("wind_mph") for s in stations]),
        "pressure_inhg": _median([s.get("pressure_inhg") for s in stations]),
        "precip_rate_in": _mean([s.get("precip_rate_in") for s in stations]),
        "precip_total_in": max((s.get("precip_total_in") or 0 for s in stations), default=0),
        "stations_used": len(stations),
        "is_raining": any((s.get("precip_rate_in") or 0) > 0 for s in stations),
    }


def fetch_synoptic_stations(lat: float, lon: float, radius_miles: int = 30) -> dict:
    """Fetch nearby weather station observations from Synoptic/MesoWest API.

    Accesses 170,000+ stations: RAWS, DOT road sensors, ski area stations,
    NWS/FAA, and private mesonets. Free tier: 5K requests/month.

    Requires SYNOPTIC_TOKEN env var (register free at synopticdata.com).
    Returns empty result gracefully if token not set.
    """
    token = os.environ.get("SYNOPTIC_TOKEN", "")
    if not token:
        return {"stations": [], "note": "SYNOPTIC_TOKEN not set"}
    try:
        resp = requests.get("https://api.synopticdata.com/v2/stations/latest", params={
            "token": token,
            "radius": f"{lat},{lon},{radius_miles}",
            "vars": "air_temp,snow_depth,wind_speed,wind_gust,wind_direction,"
                    "relative_humidity,dew_point_temperature,pressure",
            "units": "english",
            "status": "active",
            "limit": "25",
        }, timeout=15)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}

        data = resp.json()
        if data.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
            return {"error": data.get("SUMMARY", {}).get("RESPONSE_MESSAGE", "API error")}

        stations = []
        for stn in data.get("STATION", []):
            obs = stn.get("OBSERVATIONS", {})
            elev_m = stn.get("ELEVATION")
            elev_ft = round(float(elev_m) * 3.28084) if elev_m else None

            def _val(key):
                v = obs.get(key)
                if isinstance(v, dict):
                    return v.get("value")
                return v

            stations.append({
                "id": stn.get("STID", ""),
                "name": stn.get("NAME", ""),
                "network": stn.get("MNET_SHORTNAME", ""),
                "lat": float(stn.get("LATITUDE", 0)),
                "lon": float(stn.get("LONGITUDE", 0)),
                "elev_ft": elev_ft,
                "distance_mi": round(float(stn.get("DISTANCE", 0)), 1) if stn.get("DISTANCE") else None,
                "temp_f": _val("air_temp_value_1"),
                "wind_mph": _val("wind_speed_value_1"),
                "wind_gust_mph": _val("wind_gust_value_1"),
                "wind_dir_deg": _val("wind_direction_value_1"),
                "humidity_pct": _val("relative_humidity_value_1"),
                "snow_depth_in": _val("snow_depth_value_1"),
                "pressure_inhg": _val("pressure_value_1"),
            })

        return {"stations": stations, "count": len(stations)}
    except Exception as e:
        return {"error": f"Synoptic API error: {type(e).__name__}"}


def fetch_cssl_snow() -> dict:
    """Fetch Central Sierra Snow Lab data from CDEC (California Data Exchange Center).

    Station CSL — UC Berkeley's snow lab at Donner Summit (~6890ft).
    More granular than SNOTEL for real-time storm tracking.
    """
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")

        result = {}
        # Sensor 3 = Snow Depth (daily)
        resp = requests.get(
            "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet",
            params={
                "Stations": "CSL",
                "SensorNums": "3",
                "dur_code": "D",
                "Start": start,
                "End": today,
            }, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            valid = [d for d in data if d.get("value") is not None and d["value"] != -9999]
            if valid:
                result["snow_depth_in"] = valid[-1]["value"]
                result["snow_depth_date"] = valid[-1].get("obsDate", "")
                # Calculate 24h change
                if len(valid) >= 2:
                    delta = valid[-1]["value"] - valid[-2]["value"]
                    result["snow_24h_change_in"] = round(delta, 1)

        # Sensor 18 = Snow Water Content (daily)
        resp2 = requests.get(
            "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet",
            params={
                "Stations": "CSL",
                "SensorNums": "18",
                "dur_code": "D",
                "Start": start,
                "End": today,
            }, timeout=15)
        if resp2.status_code == 200:
            data2 = resp2.json()
            valid2 = [d for d in data2 if d.get("value") is not None and d["value"] != -9999]
            if valid2:
                result["swe_in"] = valid2[-1]["value"]

        # Sensor 3 (Hourly) — Snow Depth at hourly resolution for storm tracking
        try:
            resp_h = requests.get(
                "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet",
                params={
                    "Stations": "CSL",
                    "SensorNums": "3",
                    "dur_code": "H",
                    "Start": start,
                    "End": today,
                }, timeout=15)
            if resp_h.status_code == 200:
                data_h = resp_h.json()
                valid_h = [d for d in data_h if d.get("value") is not None and d["value"] != -9999]
                if valid_h:
                    result["hourly_snow_depth"] = [
                        {"time": d.get("obsDate", ""), "depth_in": d["value"]}
                        for d in valid_h[-48:]  # last 48 hours
                    ]
                    # Calculate hourly new snow accumulation
                    hourly_new = []
                    for i in range(1, len(valid_h)):
                        delta = valid_h[i]["value"] - valid_h[i-1]["value"]
                        if delta > 0:
                            hourly_new.append({
                                "time": valid_h[i].get("obsDate", ""),
                                "new_snow_in": round(delta, 1),
                            })
                    result["hourly_new_snow"] = hourly_new[-24:]  # last 24 entries
        except Exception as e:
            logger.debug("CSSL hourly snow data unavailable: %s", e)

        # Sensor 9 = Wind Speed, Sensor 10 = Wind Direction at Donner Summit
        for sensor_num, key in [("9", "wind_mph"), ("10", "wind_dir_deg")]:
            try:
                resp_w = requests.get(
                    "https://cdec.water.ca.gov/dynamicapp/req/JSONDataServlet",
                    params={
                        "Stations": "CSL",
                        "SensorNums": sensor_num,
                        "dur_code": "H",
                        "Start": (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%d"),
                        "End": today,
                    }, timeout=15)
                if resp_w.status_code == 200:
                    data_w = resp_w.json()
                    valid_w = [d for d in data_w if d.get("value") is not None and d["value"] != -9999]
                    if valid_w:
                        result[key] = valid_w[-1]["value"]
            except Exception as e:
                logger.debug("CSSL sensor %s fetch failed: %s", sensor_num, e)

        result["source"] = "CDEC/CSSL"
        result["elev_ft"] = 6890
        return result
    except Exception as e:
        return {"error": str(e)}


def fetch_nws_alerts(lat: float, lon: float) -> list[dict]:
    """Fetch active NWS alerts (watches, warnings, advisories) for a location."""
    headers = {"User-Agent": "TahoeSnowStation/1.0 (keith@local)"}
    try:
        resp = requests.get("https://api.weather.gov/alerts/active", params={
            "point": f"{lat},{lon}",
            "status": "actual",
        }, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        alerts = []
        for f in resp.json().get("features", []):
            p = f.get("properties", {})
            alerts.append({
                "event": p.get("event", ""),
                "severity": p.get("severity", ""),
                "urgency": p.get("urgency", ""),
                "headline": p.get("headline", ""),
                "description": (p.get("description", "") or "")[:500],
                "onset": p.get("onset", ""),
                "expires": p.get("expires", ""),
            })
        return alerts
    except Exception as e:
        logger.warning("fetch_nws_alerts failed: %s", e)
        return []


def fetch_sounding(station: str = "REV") -> dict:
    """Fetch latest upper-air sounding from Iowa State Mesonet.

    Reno (REV) soundings provide the actual measured atmosphere profile
    over Tahoe — real lapse rate, freezing level, moisture, and inversions.
    Launched twice daily: 00Z and 12Z.

    Returns dict with profile levels and derived snow/freeze levels.
    """
    try:
        now = datetime.now(timezone.utc)
        # Find most recent sounding time (00Z or 12Z)
        if now.hour >= 12:
            sounding_time = now.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            sounding_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # If it's very recent, the sounding may not be posted yet — try previous
        if (now - sounding_time).total_seconds() < 3600:
            sounding_time -= timedelta(hours=12)

        ts = sounding_time.strftime("%Y%m%d%H%M")
        resp = requests.get("https://mesonet.agron.iastate.edu/json/raob.py",
                            params={"ts": ts, "station": station}, timeout=15)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}

        profiles = resp.json().get("profiles", [])
        if not profiles:
            return {"error": "no_data"}

        levels = profiles[0].get("profile", [])
        if not levels:
            return {"error": "empty_profile"}

        # Extract key levels in the Tahoe elevation range
        profile = []
        freezing_level_m = None
        snow_level_m = None
        prev_temp = None
        prev_hght = None

        for lev in levels:
            h = lev.get("hght")
            t = lev.get("tmpc")
            dp = lev.get("dwpc")
            if h is None or t is None:
                continue
            profile.append({
                "hght_m": h,
                "hght_ft": round(h * 3.28084),
                "temp_c": t,
                "dewpoint_c": dp,
                "dp_depression": round(t - dp, 1) if dp is not None else None,
            })

            # Find freezing level (0°C crossing)
            if prev_temp is not None and prev_hght is not None:
                if prev_temp > 0 and t <= 0 and freezing_level_m is None:
                    frac = prev_temp / (prev_temp - t) if prev_temp != t else 0
                    freezing_level_m = prev_hght + frac * (h - prev_hght)
                # Snow level (~1°C wet bulb crossing, approximate as 1°C)
                if prev_temp > 1 and t <= 1 and snow_level_m is None:
                    frac = (prev_temp - 1) / (prev_temp - t) if prev_temp != t else 0
                    snow_level_m = prev_hght + frac * (h - prev_hght)

            prev_temp = t
            prev_hght = h

        # Compute observed lapse rate from profile in Tahoe range (1500-3500m)
        tahoe_levels = [p for p in profile if 1500 <= p["hght_m"] <= 3500]
        lapse_rate = None
        if len(tahoe_levels) >= 3:
            h_low, t_low = tahoe_levels[0]["hght_m"], tahoe_levels[0]["temp_c"]
            h_high, t_high = tahoe_levels[-1]["hght_m"], tahoe_levels[-1]["temp_c"]
            if h_high > h_low:
                lapse_rate = round(-(t_high - t_low) / ((h_high - h_low) / 1000), 2)

        return {
            "station": station,
            "time": sounding_time.isoformat(),
            "levels": len(profile),
            "freezing_level_ft": round(freezing_level_m * 3.28084) if freezing_level_m else None,
            "snow_level_ft": round(snow_level_m * 3.28084) if snow_level_m else None,
            "lapse_rate_c_km": lapse_rate,
            "profile_summary": [p for p in profile if 1500 <= p["hght_m"] <= 3500][::3],  # every 3rd level
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_climate_normals(lat: float, lon: float) -> dict:
    """Fetch 30-year climate normals from Open-Meteo for current month.

    Returns average high, low, and precipitation for the current month
    based on 1991-2020 climate data. Used to show anomalies.
    """
    try:
        now = datetime.now()
        month = now.month
        # Fetch just the current month across the 30-year baseline
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": "1991-01-01", "end_date": "2020-12-31",
            "models": "EC_Earth3P_HR",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        }
        resp = requests.get("https://climate-api.open-meteo.com/v1/climate",
                            params=params, timeout=20)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}

        daily = resp.json().get("daily", {})
        times = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])

        # Filter to current month
        month_highs = []
        month_lows = []
        month_precip = []
        for i, t in enumerate(times):
            if f"-{month:02d}-" in t:
                if i < len(highs) and highs[i] is not None:
                    month_highs.append(highs[i])
                if i < len(lows) and lows[i] is not None:
                    month_lows.append(lows[i])
                if i < len(precip) and precip[i] is not None:
                    month_precip.append(precip[i])

        if not month_highs:
            return {"error": "no_data"}

        avg_high_c = sum(month_highs) / len(month_highs)
        avg_low_c = sum(month_lows) / len(month_lows)
        # month_precip has daily values for ~30 days/yr * 30 yrs.
        # Average monthly total = sum / num_years
        num_years = 30
        avg_monthly_precip_mm = sum(month_precip) / num_years

        return {
            "month": now.strftime("%B"),
            "avg_high_f": round(avg_high_c * 9/5 + 32),
            "avg_low_f": round(avg_low_c * 9/5 + 32),
            "avg_precip_in_month": round(avg_monthly_precip_mm / 25.4, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_ensemble(lat: float, lon: float) -> dict:
    """Fetch ensemble forecasts for probabilistic snow/temp predictions.

    Queries GFS (31 members) and ECMWF (51 members) ensembles via Open-Meteo.
    Aggregates per-member hourly data into daily snowfall percentiles
    (p10/p25/p50/p75/p90) — gives true uncertainty ranges rather than
    comparing different deterministic models.
    """
    result = {"models": {}}
    for model, label in [("gfs_seamless", "GFS"), ("ecmwf_ifs025", "ECMWF")]:
        try:
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation,snowfall",
                "models": model,
                "forecast_days": 7,
                "timezone": "America/Los_Angeles",
            }
            resp = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble",
                                params=params, timeout=30)
            if resp.status_code != 200:
                result["models"][label] = {"error": f"HTTP {resp.status_code}"}
                continue

            data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                result["models"][label] = {"error": "No time data"}
                continue

            # Find per-member keys for snowfall and temperature
            snow_keys = sorted(k for k in hourly if k.startswith("snowfall_member"))
            temp_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
            if not snow_keys:
                result["models"][label] = {"error": "No member data"}
                continue

            n_members = len(snow_keys)

            # Aggregate hourly → daily per member
            daily_snow = {}   # date → [member_total, ...]
            daily_temps = {}  # date → {member_idx: [temps], ...}
            for i, t in enumerate(times):
                date = t[:10]
                if date not in daily_snow:
                    daily_snow[date] = [0.0] * n_members
                    daily_temps[date] = {j: [] for j in range(n_members)}
                for j, mk in enumerate(snow_keys):
                    vals = hourly.get(mk, [])
                    v = vals[i] if i < len(vals) and vals[i] is not None else 0.0
                    daily_snow[date][j] += v  # cm
                for j, tk in enumerate(temp_keys):
                    if j >= n_members:
                        break
                    vals = hourly.get(tk, [])
                    v = vals[i] if i < len(vals) and vals[i] is not None else None
                    if v is not None:
                        daily_temps[date][j].append(v)

            def _pct(arr, p):
                if not arr:
                    return 0
                idx = round(p / 100 * (len(arr) - 1))
                return arr[min(idx, len(arr) - 1)]

            daily_out = []
            for date in sorted(daily_snow.keys()):
                snow_in = sorted(round(v / 2.54, 1) for v in daily_snow[date])
                temp_highs = sorted(
                    round(max(daily_temps[date][j]) * 9/5 + 32, 1)
                    for j in range(min(n_members, len(temp_keys)))
                    if daily_temps[date][j]
                )
                entry = {
                    "date": date,
                    "snow_p10": _pct(snow_in, 10),
                    "snow_p25": _pct(snow_in, 25),
                    "snow_p50": _pct(snow_in, 50),
                    "snow_p75": _pct(snow_in, 75),
                    "snow_p90": _pct(snow_in, 90),
                    "snow_max": snow_in[-1] if snow_in else 0,
                    "members": n_members,
                }
                if temp_highs:
                    entry["temp_p10_f"] = _pct(temp_highs, 10)
                    entry["temp_p50_f"] = _pct(temp_highs, 50)
                    entry["temp_p90_f"] = _pct(temp_highs, 90)
                daily_out.append(entry)

            result["models"][label] = daily_out
        except Exception as e:
            result["models"][label] = {"error": str(e)}

    return result


def fetch_caltrans_chains() -> list[dict]:
    """Fetch Caltrans chain control status for District 3 (Tahoe area).

    Returns a list of active chain controls on I-80 and US-50 corridors.
    Data updates every 5 minutes from Caltrans CWWP.
    """
    try:
        resp = requests.get("https://cwwp2.dot.ca.gov/data/d3/cc/ccStatusD03.json",
                            timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        controls = []
        for item in data:
            loc = item.get("Location", {})
            route = loc.get("route", "")
            # Filter to I-80 and US-50 (main Tahoe corridors)
            if route not in ("I-80", "US-50", "80", "50"):
                continue
            status_desc = item.get("StatusDescription", "")
            if not status_desc or "no chain" in status_desc.lower():
                continue
            controls.append({
                "route": route,
                "status": status_desc,
                "location": loc.get("locationDescription", ""),
                "lat": loc.get("latitude"),
                "lon": loc.get("longitude"),
                "elevation": loc.get("elevation"),
                "timestamp": item.get("Timestamp", ""),
            })
        return controls
    except Exception as e:
        logger.warning("fetch_chain_controls failed: %s", e)
        return []


def fetch_lift_status(resort: str = "heavenly") -> dict:
    """Fetch lift status from Liftie.info API.

    Args:
        resort: Resort slug — heavenly, northstar, or kirkwood.
    Returns dict with open/closed counts and per-lift status.
    """
    try:
        resp = requests.get(f"https://liftie.info/api/resort/{resort}",
                            headers={"User-Agent": "TahoeSnowStation/1.0"},
                            timeout=10)
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}"}
        data = resp.json()
        lifts = data.get("lifts", {}).get("status", {})
        open_lifts = [name for name, status in lifts.items() if status == "open"]
        closed_lifts = [name for name, status in lifts.items() if status == "closed"]
        hold_lifts = [name for name, status in lifts.items() if status == "hold"]
        return {
            "resort": resort,
            "open": len(open_lifts),
            "closed": len(closed_lifts),
            "hold": len(hold_lifts),
            "total": len(lifts),
            "open_names": open_lifts,
            "hold_names": hold_lifts,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_all_lift_status() -> dict:
    """Fetch lift status for all active Tahoe resorts."""
    results = {}
    slugs = [
        (name.lower(), cfg.liftie_slug)
        for name, cfg in RESORT_REGISTRY.items()
        if cfg.enabled and cfg.liftie_slug
    ]
    for name, slug in slugs:
        results[name] = fetch_lift_status(slug)
    return results


def fetch_avalanche() -> dict:
    """Fetch Sierra Avalanche Center danger rating."""
    try:
        resp = requests.get("https://api.avalanche.org/v2/public/products/map-layer",
                            headers={"User-Agent": "TahoeSnowStation/1.0"}, timeout=10)
        if resp.status_code == 200:
            for f in resp.json().get("features", []):
                p = f.get("properties", {})
                if p.get("center_id") == "SAC":
                    danger = p.get("danger_level")
                    danger_labels = {0: "No Rating", 1: "Low", 2: "Moderate",
                                     3: "Considerable", 4: "High", 5: "Extreme"}
                    return {
                        "zone": p.get("name", "Central Sierra Nevada"),
                        "danger_level": danger,
                        "danger_label": danger_labels.get(danger, "Unknown"),
                        "travel_advice": p.get("travel_advice", ""),
                        "start": p.get("start_date", ""),
                        "end": p.get("end_date", ""),
                        "link": p.get("link", ""),
                    }
        return {"error": "Could not fetch"}
    except Exception as e:
        return {"error": str(e)}


def fetch_forecast_discussion() -> str:
    """Fetch latest NWS Reno AFD text."""
    headers = {"User-Agent": "TahoeSnowStation/1.0 (keith@local)"}
    try:
        resp = requests.get("https://api.weather.gov/products/types/AFD/locations/REV",
                            headers=headers, timeout=10)
        if resp.status_code == 200:
            products = resp.json().get("@graph", [])
            if products:
                pid = products[0].get("@id") or products[0].get("id", "")
                url = pid if pid.startswith("http") else f"https://api.weather.gov/products/{pid}"
                ar = requests.get(url, headers=headers, timeout=10)
                if ar.status_code == 200:
                    return ar.json().get("productText", "")[:4000]
        return ""
    except Exception as e:
        logger.warning("fetch_forecast_discussion failed: %s", e)
        return ""


def fetch_hrrr(lat: float, lon: float) -> dict:
    """Fetch HRRR model data via Herbie for 0-18h lead time.

    HRRR (High-Resolution Rapid Refresh) is NOAA's 3km convection-allowing
    model, updated hourly. Best model for 0-18h precipitation timing and type.

    Variables fetched:
    - TMP:2m — 2-meter temperature (K -> C)
    - UGRD+VGRD:10m — 10-meter wind components (m/s -> mph + direction)
    - APCP:surface — accumulated precipitation (mm)
    - CSNOW:surface — categorical snow flag (0/1)
    - CRAIN:surface — categorical rain flag (0/1)
    - REFC:entire atmosphere — composite reflectivity (dBZ)
    - CAPE:surface — convective available potential energy (J/kg)

    For each resort coordinate, finds nearest HRRR grid point.
    Returns dict matching existing model format for BMA integration.
    """
    try:
        from herbie import Herbie
        now = datetime.now(timezone.utc)
        # Use run from 2 hours ago (latest available usually delayed ~90min)
        mt = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
        results = {"model_run": mt.isoformat(), "forecasts": [], "source": "herbie"}

        # Try multiple model runs if the latest isn't available
        for run_offset in [2, 3, 4, 5]:
            mt = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=run_offset)
            try:
                # Test if this run exists
                H_test = Herbie(mt.strftime("%Y-%m-%d %H:00"), model="hrrr",
                                product="sfc", fxx=1, verbose=False)
                results["model_run"] = mt.isoformat()
                break
            except Exception:
                continue
        else:
            return {"error": "No recent HRRR run available"}

        for fxx in range(1, 19):
            try:
                H = Herbie(mt.strftime("%Y-%m-%d %H:00"), model="hrrr",
                           product="sfc", fxx=fxx, verbose=False)
                valid_time = mt + timedelta(hours=fxx)
                hd = {"fxx": fxx, "valid": valid_time.isoformat()}

                # Herbie uses longitude in 0-360 for some GRIB projections
                # Try both conventions
                lon_360 = lon + 360 if lon < 0 else lon

                # Temperature (TMP:2m) — Kelvin to Celsius
                try:
                    ds = H.xarray(":TMP:2 m above ground", verbose=False)
                    var_names = [v for v in ds.data_vars]
                    if var_names:
                        val = float(ds[var_names[0]].sel(
                            latitude=lat, longitude=lon_360, method="nearest").values)
                        hd["temp_c"] = round(val - 273.15, 1)
                        hd["temp_f"] = round(hd["temp_c"] * 9/5 + 32, 1)
                except Exception:
                    pass

                # Accumulated Precipitation (APCP:surface)
                try:
                    ds = H.xarray(":APCP:surface", verbose=False)
                    var_names = [v for v in ds.data_vars if "apcp" in v.lower() or "tp" in v.lower()]
                    if var_names:
                        hd["precip_mm"] = round(float(ds[var_names[0]].sel(
                            latitude=lat, longitude=lon_360, method="nearest").values), 2)
                except Exception:
                    pass

                # Wind (UGRD + VGRD at 10m)
                try:
                    du = H.xarray(":UGRD:10 m above ground", verbose=False)
                    dv = H.xarray(":VGRD:10 m above ground", verbose=False)
                    u_names = [v for v in du.data_vars if "u" in v.lower()]
                    v_names = [v for v in dv.data_vars if "v" in v.lower()]
                    if u_names and v_names:
                        u = float(du[u_names[0]].sel(
                            latitude=lat, longitude=lon_360, method="nearest").values)
                        v = float(dv[v_names[0]].sel(
                            latitude=lat, longitude=lon_360, method="nearest").values)
                        hd["wind_mph"] = round(math.sqrt(u**2 + v**2) * 2.237, 1)
                        hd["wind_dir"] = round((math.degrees(math.atan2(-u, -v)) + 360) % 360)
                except Exception:
                    pass

                # Categorical Snow (CSNOW:surface) — 0 or 1
                try:
                    ds = H.xarray(":CSNOW:surface", verbose=False)
                    var_names = [v for v in ds.data_vars]
                    if var_names:
                        hd["is_snow"] = int(float(ds[var_names[0]].sel(
                            latitude=lat, longitude=lon_360, method="nearest").values))
                except Exception:
                    pass

                # Categorical Rain (CRAIN:surface) — 0 or 1
                try:
                    ds = H.xarray(":CRAIN:surface", verbose=False)
                    var_names = [v for v in ds.data_vars]
                    if var_names:
                        hd["is_rain"] = int(float(ds[var_names[0]].sel(
                            latitude=lat, longitude=lon_360, method="nearest").values))
                except Exception:
                    pass

                # Composite Reflectivity (REFC)
                try:
                    ds = H.xarray(":REFC:entire atmosphere", verbose=False)
                    var_names = [v for v in ds.data_vars]
                    if var_names:
                        hd["reflectivity_dbz"] = round(float(ds[var_names[0]].sel(
                            latitude=lat, longitude=lon_360, method="nearest").values), 1)
                except Exception:
                    pass

                # CAPE (surface)
                try:
                    ds = H.xarray(":CAPE:surface", verbose=False)
                    var_names = [v for v in ds.data_vars]
                    if var_names:
                        hd["cape_jkg"] = round(float(ds[var_names[0]].sel(
                            latitude=lat, longitude=lon_360, method="nearest").values))
                except Exception:
                    pass

                # Derive snowfall from precip + temp + precip type flags
                if "precip_mm" in hd and "temp_c" in hd:
                    tc = hd["temp_c"]
                    pmm = hd["precip_mm"]
                    is_snow = hd.get("is_snow", 0)
                    is_rain = hd.get("is_rain", 0)
                    if is_snow and not is_rain and pmm > 0:
                        slr = compute_slr(tc, wind_mph=hd.get("wind_mph", 0))
                        hd["snowfall_cm"] = round(pmm / 10.0 * slr, 1)
                    elif is_snow and is_rain and pmm > 0:
                        slr = compute_slr(tc, wind_mph=hd.get("wind_mph", 0))
                        hd["snowfall_cm"] = round(pmm / 10.0 * slr * 0.5, 1)
                    else:
                        hd["snowfall_cm"] = 0

                results["forecasts"].append(hd)
            except Exception as e:
                results["forecasts"].append({"fxx": fxx, "error": str(e)})
        return results
    except ImportError:
        return {"error": "Herbie not installed — pip install herbie-data"}
    except Exception as e:
        return {"error": str(e)}


def fetch_rwis_stations(lat: float = 39.17, lon: float = -120.145) -> list[dict]:
    """Fetch RWIS (Road Weather Information System) data via Synoptic/MesoWest.

    Targets Caltrans road weather stations on I-80 (Donner Summit, Kingvale)
    and US-50 (Echo Summit). These stations report pavement temperature,
    visibility, and precipitation type — key indicators for chain controls
    before official announcements.

    Uses the same Synoptic API as fetch_synoptic_stations but filters
    specifically for DOT/RWIS network stations on the two main Tahoe corridors.

    Returns:
        List of station dicts with road weather data.
    """
    token = os.environ.get("SYNOPTIC_TOKEN", "")
    if not token:
        return []
    try:
        # Fetch DOT/RWIS-class stations within 40 miles of Tahoe
        resp = requests.get("https://api.synopticdata.com/v2/stations/latest", params={
            "token": token,
            "radius": f"{lat},{lon},40",
            "vars": "air_temp,road_temp,road_surface_condition,visibility,"
                    "wind_speed,wind_direction,precip_accum,road_subsurface_tmp",
            "network": "1,2,71,96,162",  # ASOS, RAWS, Caltrans DOT, MADIS, RWIS
            "units": "english",
            "status": "active",
            "limit": "30",
        }, timeout=15)
        if resp.status_code != 200:
            return []

        data = resp.json()
        if data.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
            return []

        # Known RWIS station IDs or name patterns for I-80 and US-50
        i80_keywords = ["donner", "kingvale", "soda springs", "truckee", "i-80",
                        "i80", "norden", "boreal", "blue canyon", "emigrant"]
        us50_keywords = ["echo summit", "echo pass", "meyers", "us-50", "us50",
                         "twin bridges", "strawberry", "kyburz", "pollock"]
        corridor_keywords = i80_keywords + us50_keywords

        stations = []
        for stn in data.get("STATION", []):
            name = (stn.get("NAME", "") or "").lower()
            stid = (stn.get("STID", "") or "").lower()
            network = (stn.get("MNET_SHORTNAME", "") or "").lower()

            # Filter for corridor stations (DOT/RWIS or near corridor)
            is_dot = "dot" in network or "rwis" in network or "caltrans" in network
            is_corridor = any(kw in name or kw in stid for kw in corridor_keywords)

            if not (is_dot or is_corridor):
                continue

            obs = stn.get("OBSERVATIONS", {})
            elev_m = stn.get("ELEVATION")
            elev_ft = round(float(elev_m) * 3.28084) if elev_m else None

            def _val(key):
                v = obs.get(key)
                if isinstance(v, dict):
                    return v.get("value")
                return v

            station_data = {
                "id": stn.get("STID", ""),
                "name": stn.get("NAME", ""),
                "network": stn.get("MNET_SHORTNAME", ""),
                "lat": float(stn.get("LATITUDE", 0)),
                "lon": float(stn.get("LONGITUDE", 0)),
                "elev_ft": elev_ft,
                "temp_f": _val("air_temp_value_1"),
                "pavement_temp_f": _val("road_temp_value_1"),
                "road_condition": _val("road_surface_condition_value_1"),
                "visibility_mi": _val("visibility_value_1"),
                "wind_mph": _val("wind_speed_value_1"),
                "wind_dir_deg": _val("wind_direction_value_1"),
                "precip_accum_in": _val("precip_accum_value_1"),
            }

            # Determine corridor
            if any(kw in name or kw in stid for kw in i80_keywords):
                station_data["corridor"] = "I-80"
            elif any(kw in name or kw in stid for kw in us50_keywords):
                station_data["corridor"] = "US-50"
            else:
                station_data["corridor"] = "unknown"

            stations.append(station_data)

        return stations
    except Exception as e:
        logger.warning("fetch_rwis_stations failed: %s", e)
        return []


def fetch_radar_nowcast(lat: float, lon: float,
                         upstream_lat: float | None = None,
                         upstream_lon: float | None = None) -> dict:
    """Fetch current precipitation status for radar-like nowcasting.

    Uses Open-Meteo's current-weather API for real-time conditions, then
    compares local conditions with upstream stations to estimate if
    precipitation is approaching.

    For Oakland: upstream is Pacific coast / west
    For Tahoe: upstream is Sacramento Valley / west

    Args:
        lat, lon: Location to check
        upstream_lat, upstream_lon: Optional upstream check point (west of target)

    Returns:
        Dict with precip_now, precip_approaching, estimated_arrival_minutes, etc.
    """
    result = {
        "precip_now": False,
        "precip_approaching": False,
        "estimated_arrival_minutes": None,
        "current_precip_mm": 0.0,
        "current_weather_code": None,
        "radar_source": "open-meteo",
    }

    try:
        # Fetch current conditions at target location
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat,
            "longitude": lon,
            "current": "precipitation,rain,showers,snowfall,weather_code,is_day",
            "timezone": "America/Los_Angeles",
        }, timeout=10)
        if resp.status_code == 200:
            current = resp.json().get("current", {})
            precip = (current.get("precipitation") or 0)
            rain = (current.get("rain") or 0)
            snow = (current.get("snowfall") or 0)
            result["precip_now"] = (precip + rain + snow) > 0.1
            result["current_precip_mm"] = round(precip, 2)
            result["current_weather_code"] = current.get("weather_code")

        # Check upstream location if provided
        if upstream_lat is not None and upstream_lon is not None:
            resp2 = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": upstream_lat,
                "longitude": upstream_lon,
                "current": "precipitation,rain,showers,snowfall,weather_code",
                "hourly": "precipitation",
                "forecast_days": 1,
                "timezone": "America/Los_Angeles",
            }, timeout=10)
            if resp2.status_code == 200:
                up_current = resp2.json().get("current", {})
                up_precip = (up_current.get("precipitation") or 0)
                up_rain = (up_current.get("rain") or 0)
                up_snow = (up_current.get("snowfall") or 0)
                upstream_wet = (up_precip + up_rain + up_snow) > 0.1

                if upstream_wet and not result["precip_now"]:
                    result["precip_approaching"] = True
                    # Rough estimate: typical storm speed ~30mph
                    # Distance approximation between upstream and target
                    dlat = abs(lat - upstream_lat)
                    dlon = abs(lon - upstream_lon)
                    dist_deg = math.sqrt(dlat**2 + dlon**2)
                    dist_miles = dist_deg * 69.0  # rough lat-degree to miles
                    storm_speed_mph = 30
                    result["estimated_arrival_minutes"] = round(
                        dist_miles / storm_speed_mph * 60
                    )

    except Exception as e:
        logger.warning("fetch_radar_nowcast failed: %s", e)

    return result


def fetch_webcam_conditions() -> list[dict]:
    """Placeholder for future webcam image analysis via vision API.

    Defines the interface for webcam-based condition detection at key
    Tahoe locations. Requires a vision model API (e.g., Claude vision)
    to analyze images for snow conditions, visibility, and precipitation.

    This is a Tier 6 feature — returns None for all condition fields
    until a vision model is integrated.

    Returns:
        List of webcam station dicts with None condition fields.
    """
    webcams = [
        {
            "station": "Heavenly Base (California Lodge)",
            "url": "https://www.skiheavenly.com/the-mountain/webcams/california-lodge.aspx",
            "conditions": None,
            "visibility": None,
            "snowing": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "station": "Heavenly Gondola Top",
            "url": "https://www.skiheavenly.com/the-mountain/webcams/gondola-top.aspx",
            "conditions": None,
            "visibility": None,
            "snowing": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "station": "Kirkwood Base",
            "url": "https://www.kirkwood.com/the-mountain/webcams.aspx",
            "conditions": None,
            "visibility": None,
            "snowing": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "station": "I-80 Donner Summit (Caltrans)",
            "url": "https://cwwp2.dot.ca.gov/vm/loc/d3/hwy80atdonnerpass.htm",
            "conditions": None,
            "visibility": None,
            "snowing": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ]
    # Future: pass each URL to a vision model API for analysis
    # Example: conditions = vision_model.analyze(url, prompt="Describe snow conditions")
    return webcams


# ---------------------------------------------------------------------------
# Multi-Model Processing
# ---------------------------------------------------------------------------

def parse_open_meteo(om: dict, elev_ft: int, observed_lapse_rate: float | None = None) -> dict:
    """
    Parse Open-Meteo multi-model response into structured per-model,
    per-hour forecasts with elevation-adjusted snow physics applied.
    """
    if "error" in om or "hourly" not in om:
        return {"error": om.get("error", "No data")}

    hourly = om["hourly"]
    times = hourly.get("time", [])
    base_elev_m = om.get("elevation", 1900)  # Open-Meteo returns model grid elevation
    target_m = elev_ft * 0.3048

    parsed = {"times": times, "models": {}}

    for model in MODELS:
        label = MODEL_LABELS[model]
        temps = hourly.get(f"temperature_2m_{model}", [])
        precip = hourly.get(f"precipitation_{model}", [])
        snow = hourly.get(f"snowfall_{model}", [])
        sdepth = hourly.get(f"snow_depth_{model}", [])
        wspd = hourly.get(f"wind_speed_10m_{model}", [])
        wdir = hourly.get(f"wind_direction_10m_{model}", [])
        wgust = hourly.get(f"wind_gusts_10m_{model}", [])
        freezing = hourly.get(f"freezing_level_height_{model}", [])
        wcode = hourly.get(f"weather_code_{model}", [])
        cloud = hourly.get(f"cloud_cover_{model}", [])
        rh = hourly.get(f"relative_humidity_2m_{model}", [])
        dewpt = hourly.get(f"dew_point_2m_{model}", [])
        apparent = hourly.get(f"apparent_temperature_{model}", [])
        psl = hourly.get(f"pressure_msl_{model}", [])
        vis = hourly.get(f"visibility_{model}", [])
        cape_data = hourly.get(f"cape_{model}", [])

        hours = []
        for i in range(len(times)):
            tc = temps[i] if i < len(temps) and temps[i] is not None else None
            # Adjust temperature to target elevation
            if tc is not None:
                tc = estimate_temp_c(tc, base_elev_m, target_m,
                                     observed_lapse_rate=observed_lapse_rate)
            tf = round(tc * 9/5 + 32, 1) if tc is not None else None
            ws = wspd[i] if i < len(wspd) and wspd[i] is not None else 0
            wd = wdir[i] if i < len(wdir) and wdir[i] is not None else 0
            wg = wgust[i] if i < len(wgust) and wgust[i] is not None else 0
            pr = precip[i] if i < len(precip) and precip[i] is not None else 0
            sf = snow[i] if i < len(snow) and snow[i] is not None else 0
            sd = sdepth[i] if i < len(sdepth) and sdepth[i] is not None else None
            hm = rh[i] if i < len(rh) and rh[i] is not None else None

            # Compute SLR-adjusted snowfall at target elevation
            ca = cape_data[i] if i < len(cape_data) and cape_data[i] is not None else 0
            slr = compute_slr(tc, wind_mph=ws * 0.621371, rh_pct=hm) if tc is not None else 10
            oro = orographic_multiplier(elev_ft, ws * 0.621371, wd, cape_jkg=ca)
            adj_precip_in = (pr / 25.4) * oro  # mm -> inches, orographic adjusted
            pt = precip_type(tc, pr > 0.1, rh_pct=hm) if tc is not None else "None"

            if pt == "Snow":
                snow_in = adj_precip_in * slr
            elif pt == "Mix":
                snow_in = adj_precip_in * slr * 0.5
            else:
                snow_in = 0

            fl = wind_chill_f(tf, ws * 0.621371) if tf is not None else None

            # Extract new optional fields (may be None if model doesn't provide them)
            fz = freezing[i] if i < len(freezing) and freezing[i] is not None else None
            wc = wcode[i] if i < len(wcode) and wcode[i] is not None else None
            cc = cloud[i] if i < len(cloud) and cloud[i] is not None else None
            # hm and ca already extracted above for wet-bulb precip type / CAPE coupling
            dp = dewpt[i] if i < len(dewpt) and dewpt[i] is not None else None
            ps = psl[i] if i < len(psl) and psl[i] is not None else None
            vi = vis[i] if i < len(vis) and vis[i] is not None else None

            hours.append({
                "time": times[i],
                "temp_f": tf, "temp_c": round(tc, 1) if tc is not None else None,
                "feels_like_f": fl,
                "precip_in": round(adj_precip_in, 3),
                "snowfall_in": round(snow_in, 1),
                "snow_depth_m": sd,
                "wind_mph": round(ws * 0.621371, 1),
                "wind_gust_mph": round(wg * 0.621371, 1),
                "wind_dir": wind_dir_str(wd),
                "wind_dir_deg": wd,
                "precip_type": pt,
                "slr": round(slr, 1),
                "snow_quality": snow_quality_str(slr),
                "freezing_level_ft": round(fz * 3.28084) if fz is not None else None,
                "weather_code": int(wc) if wc is not None else None,
                "cloud_cover_pct": round(cc) if cc is not None else None,
                "humidity_pct": round(hm) if hm is not None else None,
                "dewpoint_f": round(dp * 9/5 + 32, 1) if dp is not None else None,
                "pressure_hpa": round(ps, 1) if ps is not None else None,
                "visibility_mi": round(vi / 1609.34, 1) if vi is not None else None,
                "cape_jkg": round(ca) if ca is not None else None,
            })

        parsed["models"][label] = hours

    return parsed


def aggregate_daily(hours: list) -> list:
    """Aggregate hourly data into day/night 12h buckets."""
    if not hours:
        return []

    buckets = []
    current_date = None
    day_hours = []   # 6am-6pm
    night_hours = [] # 6pm-6am

    for h in hours:
        try:
            dt = datetime.fromisoformat(h["time"])
        except (ValueError, TypeError):
            continue

        date_str = dt.strftime("%Y-%m-%d")
        hour = dt.hour

        if current_date is None:
            current_date = date_str

        if date_str != current_date:
            # Emit day bucket for previous date
            if day_hours:
                buckets.append(_summarize_bucket(current_date, "Day", day_hours))
            if night_hours:
                buckets.append(_summarize_bucket(current_date, "Night", night_hours))
            day_hours = []
            night_hours = []
            current_date = date_str

        if 6 <= hour < 18:
            day_hours.append(h)
        else:
            night_hours.append(h)

    # Final
    if day_hours:
        buckets.append(_summarize_bucket(current_date, "Day", day_hours))
    if night_hours:
        buckets.append(_summarize_bucket(current_date, "Night", night_hours))

    return buckets


def _summarize_bucket(date: str, period: str, hours: list) -> dict:
    temps = [h["temp_f"] for h in hours if h["temp_f"] is not None]
    feels = [h["feels_like_f"] for h in hours if h["feels_like_f"] is not None]
    winds = [h["wind_mph"] for h in hours if h["wind_mph"] is not None]
    gusts = [h["wind_gust_mph"] for h in hours if h["wind_gust_mph"] is not None]
    snow = sum(h["snowfall_in"] for h in hours)
    liquid = sum(h["precip_in"] for h in hours)
    ptypes = [h["precip_type"] for h in hours if h["precip_type"] != "None"]

    if "Snow" in ptypes and "Rain" in ptypes:
        dom_type = "Mix"
    elif "Snow" in ptypes:
        dom_type = "Snow"
    elif "Rain" in ptypes:
        dom_type = "Rain"
    elif "Mix" in ptypes:
        dom_type = "Mix"
    else:
        dom_type = "None"

    return {
        "date": date,
        "period": period,
        "temp_high_f": round(max(temps), 1) if temps else None,
        "temp_low_f": round(min(temps), 1) if temps else None,
        "feels_like_low_f": round(min(feels), 1) if feels else None,
        "wind_avg_mph": round(np.mean(winds), 1) if winds else 0,
        "wind_max_gust_mph": round(max(gusts), 1) if gusts else 0,
        "snow_in": round(snow, 1),
        "liquid_in": round(liquid, 2),
        "precip_type": dom_type,
    }


def multi_model_spread(parsed: dict) -> list:
    """Compute daily spread across models for agreement/confidence."""
    if "error" in parsed:
        return []

    models = parsed["models"]
    model_names = list(models.keys())
    if not model_names:
        return []

    # Get number of hours and daily aggregate each model
    dailies = {}
    for mname in model_names:
        dailies[mname] = aggregate_daily(models[mname])

    # Merge by date
    all_dates = set()
    for mname in model_names:
        for b in dailies[mname]:
            all_dates.add(b["date"])

    spread = []
    for date in sorted(all_dates):
        day_data = {"date": date, "models": {}}
        for mname in model_names:
            # Sum snow for the full day (day + night)
            day_snow = sum(b["snow_in"] for b in dailies[mname] if b["date"] == date)
            day_liquid = sum(b["liquid_in"] for b in dailies[mname] if b["date"] == date)
            temps = [b["temp_high_f"] for b in dailies[mname] if b["date"] == date and b["temp_high_f"] is not None]
            day_data["models"][mname] = {
                "snow_in": round(day_snow, 1),
                "liquid_in": round(day_liquid, 2),
                "temp_high_f": max(temps) if temps else None,
            }

        # Compute spread — snow and temperature
        snows = [day_data["models"][m]["snow_in"] for m in model_names if m in day_data["models"]]
        temps = [day_data["models"][m]["temp_high_f"] for m in model_names
                 if m in day_data["models"] and day_data["models"][m]["temp_high_f"] is not None]
        day_data["snow_range"] = [min(snows), max(snows)] if snows else [0, 0]
        day_data["snow_spread"] = round(max(snows) - min(snows), 1) if snows else 0
        day_data["temp_spread_f"] = round(max(temps) - min(temps), 1) if len(temps) >= 2 else None
        n_models_with_data = len([m for m in model_names if m in day_data["models"]])
        day_data["models_available"] = n_models_with_data

        # Confidence: snow spread (in), temp spread (F), model count
        n_models = n_models_with_data
        ts = day_data["temp_spread_f"] or 0
        ss = day_data["snow_spread"]
        if n_models <= 1:
            day_data["confidence"] = "Low"
        elif n_models == 2 and ts > 10:
            day_data["confidence"] = "Low"
        elif ss >= 5 or ts >= 15:
            day_data["confidence"] = "Low"
        elif ss >= 2 or ts >= 8 or n_models == 2:
            day_data["confidence"] = "Medium"
        else:
            day_data["confidence"] = "High"

        spread.append(day_data)

    return spread


def skill_weighted_blend(parsed: dict, model_weights: dict | None = None) -> list:
    """Blend multi-model forecasts using skill-based weights.

    Instead of showing a single deterministic model (GFS), this produces
    a weighted average across all available models. Models with better
    verified skill get higher weight.

    This is a simplified form of Bayesian Model Averaging (BMA), which is
    the standard post-processing technique at operational weather centers
    (Raftery et al. 2005).

    Args:
        parsed: Output from parse_open_meteo() containing per-model hourly data
        model_weights: Dict of {model_label: weight}, e.g. {"GFS": 0.30, "ECMWF": 0.35}
                       Weights are normalized to sum to 1.0.
                       If None, uses equal weights.

    Returns:
        List of blended hourly forecasts (same format as single-model hours).
    """
    if "error" in parsed or "models" not in parsed:
        return []

    models = parsed["models"]
    available = [m for m in models if models[m]]
    if not available:
        return []

    # If only one model available, return it directly
    if len(available) == 1:
        return models[available[0]]

    # Normalize weights to available models
    if model_weights:
        raw_w = {m: model_weights.get(m, 0.1) for m in available}
    else:
        raw_w = {m: 1.0 for m in available}
    total_w = sum(raw_w.values())
    weights = {m: w / total_w for m, w in raw_w.items()}

    # Determine common time range (min length across models)
    min_len = min(len(models[m]) for m in available)
    if min_len == 0:
        return []

    blended = []
    numeric_fields = [
        "temp_f", "temp_c", "feels_like_f", "precip_in", "snowfall_in",
        "wind_mph", "wind_gust_mph", "wind_dir_deg", "slr",
        "freezing_level_ft", "cloud_cover_pct", "humidity_pct",
        "dewpoint_f", "pressure_hpa", "visibility_mi", "cape_jkg",
    ]

    for i in range(min_len):
        hour = {"time": models[available[0]][i]["time"]}

        for field in numeric_fields:
            vals = []
            w_list = []
            for m in available:
                v = models[m][i].get(field)
                if v is not None:
                    vals.append(v)
                    w_list.append(weights[m])
            if vals:
                # Weighted average
                w_total = sum(w_list)
                if w_total > 0:
                    blended_val = sum(v * w for v, w in zip(vals, w_list)) / w_total
                else:
                    blended_val = sum(vals) / len(vals)
                # Round appropriately
                if field in ("temp_f", "feels_like_f", "dewpoint_f"):
                    hour[field] = round(blended_val, 1)
                elif field in ("precip_in",):
                    hour[field] = round(blended_val, 3)
                elif field in ("snowfall_in", "slr", "wind_mph", "wind_gust_mph"):
                    hour[field] = round(blended_val, 1)
                elif field == "temp_c":
                    hour[field] = round(blended_val, 1)
                else:
                    hour[field] = round(blended_val, 1) if blended_val is not None else None
            else:
                hour[field] = None

        # Wind direction needs circular averaging
        wd_vals = []
        wd_weights = []
        for m in available:
            wd = models[m][i].get("wind_dir_deg")
            if wd is not None:
                wd_vals.append(wd)
                wd_weights.append(weights[m])
        if wd_vals:
            sin_sum = sum(math.sin(math.radians(d)) * w for d, w in zip(wd_vals, wd_weights))
            cos_sum = sum(math.cos(math.radians(d)) * w for d, w in zip(wd_vals, wd_weights))
            hour["wind_dir_deg"] = round(math.degrees(math.atan2(sin_sum, cos_sum)) % 360)
            hour["wind_dir"] = wind_dir_str(hour["wind_dir_deg"])
        else:
            hour["wind_dir"] = "N"

        # Precipitation type from blended wet-bulb temp
        tc = hour.get("temp_c")
        rh = hour.get("humidity_pct")
        pr = hour.get("precip_in", 0) or 0
        hour["precip_type"] = precip_type(tc, pr > 0.1, rh_pct=rh) if tc is not None else "None"
        hour["snow_quality"] = snow_quality_str(hour.get("slr") or 10)

        # Snow depth: use max across models (conservative for planning)
        sd_vals = [models[m][i].get("snow_depth_m") for m in available
                   if models[m][i].get("snow_depth_m") is not None]
        hour["snow_depth_m"] = max(sd_vals) if sd_vals else None

        # Weather code: use mode (most common)
        wc_vals = [models[m][i].get("weather_code") for m in available
                   if models[m][i].get("weather_code") is not None]
        if wc_vals:
            hour["weather_code"] = max(set(wc_vals), key=wc_vals.count)
        else:
            hour["weather_code"] = None

        # Model spread for this hour (useful for confidence)
        snow_vals = [models[m][i].get("snowfall_in", 0) for m in available]
        hour["_model_spread_snow"] = round(max(snow_vals) - min(snow_vals), 1) if len(snow_vals) > 1 else 0

        blended.append(hour)

    return blended


def validate_snow_level(computed_snow_level_ft: int | None,
                        sounding: dict,
                        model_freezing_levels: list[float | None]) -> dict:
    """Validate computed snow level against observed sounding and model data.

    Cross-references:
      1. Our computed snow level (from lapse rate extrapolation)
      2. Radiosonde observed snow level (ground truth — twice daily)
      3. Model-predicted freezing levels (GFS, ECMWF, ICON)

    Returns validated snow level and confidence assessment.
    """
    result = {
        "computed_ft": computed_snow_level_ft,
        "sounding_ft": None,
        "model_median_ft": None,
        "validated_ft": computed_snow_level_ft,
        "confidence": "low",
        "method": "computed",
    }

    # Sounding-observed snow level (best ground truth, but only 2x daily)
    sounding_sl = sounding.get("snow_level_ft") if sounding and "error" not in sounding else None
    result["sounding_ft"] = sounding_sl

    # Model freezing levels (continuous, but model-dependent)
    valid_fl = [fl for fl in model_freezing_levels if fl is not None]
    model_median = None
    if valid_fl:
        valid_fl_sorted = sorted(valid_fl)
        n = len(valid_fl_sorted)
        model_median = valid_fl_sorted[n // 2] if n % 2 else (valid_fl_sorted[n // 2 - 1] + valid_fl_sorted[n // 2]) / 2
        result["model_median_ft"] = round(model_median)

    # Validation logic: prefer sounding, fall back to model consensus
    if sounding_sl is not None:
        # Sounding is ground truth — use it, but check for staleness
        sounding_age_h = None
        if sounding.get("time"):
            try:
                st = datetime.fromisoformat(sounding["time"])
                sounding_age_h = (datetime.now(timezone.utc) - st).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        if sounding_age_h is not None and sounding_age_h <= 6:
            # Fresh sounding — high confidence, use it directly
            result["validated_ft"] = sounding_sl
            result["confidence"] = "high"
            result["method"] = "sounding"
        elif sounding_age_h is not None and sounding_age_h <= 12:
            # Aging sounding — blend with model consensus
            if model_median is not None:
                result["validated_ft"] = round((sounding_sl * 0.6 + model_median * 0.4))
                result["confidence"] = "medium"
                result["method"] = "sounding+model_blend"
            else:
                result["validated_ft"] = sounding_sl
                result["confidence"] = "medium"
                result["method"] = "sounding_aged"
        else:
            # Stale sounding — model consensus preferred
            if model_median is not None:
                result["validated_ft"] = round(model_median)
                result["confidence"] = "medium"
                result["method"] = "model_consensus"
    elif model_median is not None:
        # No sounding — use model consensus
        result["validated_ft"] = round(model_median)
        # Confidence based on model agreement
        if valid_fl and len(valid_fl) >= 3:
            spread = max(valid_fl) - min(valid_fl)
            if spread < 1000:
                result["confidence"] = "high"
            elif spread < 2500:
                result["confidence"] = "medium"
            else:
                result["confidence"] = "low"
        else:
            result["confidence"] = "medium"
        result["method"] = "model_consensus"

    # Cross-check: flag large discrepancies between computed and validated
    if computed_snow_level_ft and result["validated_ft"]:
        discrepancy = abs(computed_snow_level_ft - result["validated_ft"])
        result["discrepancy_ft"] = discrepancy
        if discrepancy > 2000:
            result["warning"] = "Large discrepancy between computed and validated snow level"

    return result


def blend_nws_grids(nws_grids: dict, model_snow_24h: float,
                    model_snow_7d: float) -> dict:
    """Blend NWS forecaster-edited gridpoint snow amounts with model data.

    NWS gridpoint data represents human-edited forecasts from NWS meteorologists.
    They have access to mesoscale models, radar, and local knowledge that
    automated models miss. Blending these with raw model output gives better
    accuracy than either alone.

    Weight: NWS grids 40%, model output 60%
    (Models have higher temporal resolution; NWS has human expertise)
    """
    if not nws_grids or "error" in nws_grids:
        return {"snow_24h": model_snow_24h, "snow_7d": model_snow_7d,
                "method": "model_only"}

    now = datetime.now(timezone.utc)
    nws_snow_24h = 0.0
    nws_snow_7d = 0.0

    snowfall_data = nws_grids.get("snowfallAmount", [])
    for entry in snowfall_data:
        try:
            start = datetime.fromisoformat(entry["start"])
            hours_ahead = (start - now).total_seconds() / 3600
            if hours_ahead < 0:
                continue
            val = entry.get("value", 0) or 0
            if hours_ahead < 24:
                nws_snow_24h += val
            if hours_ahead < 168:
                nws_snow_7d += val
        except (ValueError, TypeError, KeyError):
            continue

    # Blend: 40% NWS human-edited, 60% model
    if nws_snow_24h > 0 or nws_snow_7d > 0:
        blended_24h = round(model_snow_24h * 0.6 + nws_snow_24h * 0.4, 1)
        blended_7d = round(model_snow_7d * 0.6 + nws_snow_7d * 0.4, 1)
        return {
            "snow_24h": blended_24h,
            "snow_7d": blended_7d,
            "nws_24h": round(nws_snow_24h, 1),
            "nws_7d": round(nws_snow_7d, 1),
            "model_24h": model_snow_24h,
            "model_7d": model_snow_7d,
            "method": "nws_model_blend",
        }

    return {"snow_24h": model_snow_24h, "snow_7d": model_snow_7d,
            "method": "model_only"}


def calibrated_confidence(model_spread: list, ensemble: dict | None = None) -> list:
    """Compute calibrated confidence scores using ensemble spread.

    Instead of heuristic thresholds (spread > 5" = Low), uses the ensemble
    interquartile range (IQR) as a statistically grounded measure of
    forecast uncertainty.

    Ensemble spread is a better uncertainty metric than deterministic model
    disagreement because it samples the actual probability distribution
    of possible outcomes.
    """
    if not model_spread:
        return model_spread

    # Extract ensemble daily data if available
    ensemble_daily = {}
    if ensemble and "models" in ensemble:
        for model_label, daily_list in ensemble["models"].items():
            if isinstance(daily_list, list):
                for day in daily_list:
                    date = day.get("date", "")
                    if date not in ensemble_daily:
                        ensemble_daily[date] = day
                    else:
                        # Merge: take wider spread (min for low percentiles, max for high)
                        existing = ensemble_daily[date]
                        for k in ("snow_p10", "snow_p25"):
                            if k in day:
                                existing[k] = min(existing.get(k, day[k]), day[k])
                        for k in ("snow_p50",):
                            if k in day:
                                existing[k] = (existing.get(k, day[k]) + day[k]) / 2
                        for k in ("snow_p75", "snow_p90"):
                            if k in day:
                                existing[k] = max(existing.get(k, day[k]), day[k])

    for day in model_spread:
        date = day["date"]
        ens = ensemble_daily.get(date)

        if ens:
            # Use ensemble IQR for calibrated confidence
            iqr = (ens.get("snow_p75", 0) or 0) - (ens.get("snow_p25", 0) or 0)
            p90_p10 = (ens.get("snow_p90", 0) or 0) - (ens.get("snow_p10", 0) or 0)

            if iqr < 1.0 and p90_p10 < 3.0:
                day["confidence"] = "High"
            elif iqr < 3.0 and p90_p10 < 8.0:
                day["confidence"] = "Medium"
            else:
                day["confidence"] = "Low"

            day["ensemble_iqr_in"] = round(iqr, 1)
            day["ensemble_range_in"] = round(p90_p10, 1)
            day["confidence_method"] = "ensemble_calibrated"

            # Add ensemble median as reference
            day["ensemble_median_in"] = ens.get("snow_p50", 0)
        else:
            # Fall back to deterministic spread (original method)
            day["confidence_method"] = "deterministic_spread"

    return model_spread


# ---------------------------------------------------------------------------
# "Should I Go?" Decision Engine
# ---------------------------------------------------------------------------

STORM_ARCHIVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   ".storm_archive.json")


def _snow_score(snow_24h: float, snow_72h: float) -> int:
    """Score snow forecast on 0-100 scale.
    Weighs 24h at 60%, 72h at 40% to reward imminent powder."""
    def _curve(inches):
        if inches <= 0:
            return 0
        if inches <= 6:
            return int(inches * 50 / 6)
        if inches <= 12:
            return int(50 + (inches - 6) * 30 / 6)
        if inches <= 18:
            return int(80 + (inches - 12) * 20 / 6)
        return 100
    return int(_curve(snow_24h) * 0.6 + _curve(snow_72h) * 0.4)


def _quality_score(slr: float | None) -> int:
    """Score snow quality by SLR. Higher SLR = lighter drier powder."""
    if slr is None:
        return 50  # neutral default
    if slr >= 15:
        return 100
    if slr >= 12:
        return 80
    if slr >= 8:
        return 50
    return 20


def _lift_score(lifts_data: dict, resort_key: str) -> int:
    """Score lift availability for a resort."""
    if not lifts_data:
        return 50  # neutral when unknown
    key = resort_key.lower()
    d = lifts_data.get(key, {})
    if not d or d.get("error") or not d.get("total"):
        return 50
    frac = d["open"] / d["total"]
    if frac >= 1.0:
        return 100
    if frac >= 0.75:
        return 80
    if frac >= 0.50:
        return 50
    return 20


def _avalanche_score(avy: dict) -> int:
    """Inverse avalanche danger score."""
    if not avy or "danger_level" not in avy:
        return 80  # neutral when unavailable
    level = avy.get("danger_level", 0) or 0
    return {0: 80, 1: 100, 2: 80, 3: 40, 4: 10, 5: 0}.get(level, 50)


def _chain_score(chains: list) -> int:
    """Score chain controls. No chains = 100, more restrictions = lower."""
    if not chains:
        return 100
    # Count active restrictions
    n = len(chains)
    # Check for R3 (chains required all vehicles)
    for c in chains:
        status = (c.get("status") or "").upper()
        if "R3" in status or "ALL VEHICLES" in status.upper():
            return 10
        if "R2" in status or "4WD" in status.upper():
            return 40
    if n >= 3:
        return 30
    if n >= 2:
        return 50
    return 60


def _crowd_factor() -> float:
    """Weekend/holiday crowd multiplier."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    weekday = now.weekday()
    # Major ski holidays (approx)
    month_day = (now.month, now.day)
    holiday_windows = [
        (12, 24), (12, 25), (12, 26), (12, 27), (12, 28), (12, 29), (12, 30), (12, 31),
        (1, 1), (1, 2),
        (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19),  # Presidents week
    ]
    if month_day in holiday_windows:
        return 0.75
    if weekday >= 5:  # Sat/Sun
        return 0.85
    return 1.0


def _model_agreement_bonus(model_spread: list) -> int:
    """Bonus/penalty based on model agreement for near-term forecast."""
    if not model_spread:
        return 0
    # Look at first 3 days
    spreads = [d.get("snow_spread", 0) for d in model_spread[:3]]
    if not spreads:
        return 0
    avg_spread = sum(spreads) / len(spreads)
    if avg_spread < 1.5:
        return 10  # high agreement bonus
    if avg_spread > 5.0:
        return -10  # high disagreement penalty
    return 0


def compute_ski_decision(analysis: dict) -> dict:
    """Synthesize all data sources into a go/no-go ski decision.

    Computes per-resort scores and picks the best option.
    Also scans next 5 days to find the optimal ski day.

    Returns:
        {
            "score": int (0-100),
            "label": str,
            "factors": {"snow": int, "quality": int, "lifts": int,
                        "avalanche": int, "chains": int,
                        "crowd": float, "agreement": int},
            "best_resort": str,
            "best_day": str,
            "reasoning": str,
        }
    """
    if not analysis or "resorts" not in analysis:
        return {
            "score": 0, "label": "Stay home -- save your gas money",
            "factors": {}, "best_resort": None, "best_day": None,
            "reasoning": "Insufficient data to compute decision.",
        }

    avy = analysis.get("avalanche", {})
    chains = analysis.get("chains", [])
    lifts = analysis.get("lifts", {})

    avy_s = _avalanche_score(avy)
    chain_s = _chain_score(chains or [])
    crowd = _crowd_factor()

    best_score = -1
    best_resort = None
    best_factors = {}

    for rn, rd in analysis.get("resorts", {}).items():
        peak = rd.get("zones", {}).get("peak", {})
        if not peak or "error" in peak:
            continue

        snow_24h = peak.get("snow_24h", 0) or 0
        snow_72h = 0
        spread = peak.get("model_spread", [])
        for d in spread[:3]:
            for mname, md in d.get("models", {}).items():
                s = md.get("snow_in", 0) or 0
                if s > snow_72h:
                    snow_72h = s
                    break
            # Use blended/GFS 72h from timeline
        timeline = peak.get("timeline_48h", [])
        if len(timeline) >= 48:
            snow_72h = max(snow_72h,
                           sum(h.get("snowfall_in", 0) for h in timeline[:48]))
        # Also check 72h from model_spread
        spread_72h = sum(
            max((md.get("snow_in", 0) for md in d.get("models", {}).values()), default=0)
            for d in spread[:3]
        )
        snow_72h = max(snow_72h, spread_72h)

        snap = peak.get("current", {})
        slr = snap.get("slr")
        snow_s = _snow_score(snow_24h, snow_72h)
        qual_s = _quality_score(slr)
        lift_s = _lift_score(lifts, rn)
        agreement = _model_agreement_bonus(spread)

        raw = (snow_s * 0.35 + qual_s * 0.20 + lift_s * 0.15 +
               avy_s * 0.15 + chain_s * 0.10 + agreement * 0.05)
        final = int(raw * crowd)
        final = max(0, min(100, final))

        if final > best_score:
            best_score = final
            best_resort = rn
            best_factors = {
                "snow": snow_s, "quality": qual_s, "lifts": lift_s,
                "avalanche": avy_s, "chains": chain_s,
                "crowd": crowd, "agreement": agreement,
            }

    # Find best day in next 5 days by scanning model_spread
    best_day = None
    best_day_score = -1
    if best_resort:
        peak = analysis["resorts"][best_resort]["zones"].get("peak", {})
        spread = peak.get("model_spread", [])
        for d in spread[:5]:
            day_snow = max(
                (md.get("snow_in", 0) for md in d.get("models", {}).values()),
                default=0,
            )
            day_score = _snow_score(day_snow, day_snow)
            if day_score > best_day_score:
                best_day_score = day_score
                best_day = d.get("date", "")

    # Format best_day as readable string
    best_day_str = None
    if best_day:
        try:
            dt = datetime.strptime(best_day, "%Y-%m-%d")
            best_day_str = dt.strftime("%A")
        except (ValueError, TypeError):
            best_day_str = best_day

    # Decision label
    if best_score >= 90:
        label = "Epic day -- drop everything and go"
    elif best_score >= 75:
        label = "Strong go -- conditions are excellent"
    elif best_score >= 60:
        label = "Worth it -- good conditions"
    elif best_score >= 45:
        label = "Marginal -- check again tomorrow"
    elif best_score >= 30:
        label = "Probably skip -- conditions are subpar"
    else:
        label = "Stay home -- save your gas money"

    # Build reasoning
    reasons = []
    snow_s = best_factors.get("snow", 0)
    if snow_s >= 80:
        reasons.append("significant snowfall expected")
    elif snow_s >= 50:
        reasons.append("moderate snow in forecast")
    elif snow_s > 0:
        reasons.append("light snow possible")
    else:
        reasons.append("no new snow expected")

    qual_s = best_factors.get("quality", 50)
    if qual_s >= 80:
        reasons.append("powder quality looks excellent")
    elif qual_s <= 30:
        reasons.append("heavy/wet snow expected")

    avy_s_val = best_factors.get("avalanche", 80)
    if avy_s_val <= 40:
        reasons.append("elevated avalanche danger")

    chain_s_val = best_factors.get("chains", 100)
    if chain_s_val <= 40:
        reasons.append("significant chain controls active")

    if crowd < 0.85:
        reasons.append("expect holiday crowds")
    elif crowd < 1.0:
        reasons.append("weekend crowds likely")

    reasoning = "; ".join(reasons).capitalize() + "."

    return {
        "score": best_score if best_score >= 0 else 0,
        "label": label,
        "factors": best_factors,
        "best_resort": best_resort,
        "best_day": best_day_str,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Storm Timeline Narrative
# ---------------------------------------------------------------------------

def generate_storm_narrative(analysis: dict) -> str:
    """Generate a meteorologist-style storm briefing from forecast data.

    Analyzes the next 7 days across all resorts and produces a 3-6 sentence
    natural-language briefing covering:
    - Next system timing
    - Snow level evolution
    - Peak snowfall timing and rates
    - Wind impacts
    - Transition/clearing timing
    - Model confidence
    """
    if not analysis or "resorts" not in analysis:
        return "Insufficient data for storm narrative."

    # Collect data from the best resort (Heavenly preferred, or first available)
    resort_name = "Heavenly"
    if resort_name not in analysis["resorts"]:
        resort_name = next(iter(analysis["resorts"]), None)
    if not resort_name:
        return "No resort data available."

    rd = analysis["resorts"][resort_name]
    peak = rd.get("zones", {}).get("peak", {})
    if not peak or "error" in peak:
        return "No peak zone data available for narrative."

    timeline = peak.get("timeline_48h", [])
    spread = peak.get("model_spread", [])

    # --- Analyze next 7 days from model_spread ---
    storm_days = []  # days with meaningful snow
    dry_days = []
    total_7d_snow = 0
    for d in spread[:7]:
        models = d.get("models", {})
        day_snows = [md.get("snow_in", 0) or 0 for md in models.values()]
        avg_snow = sum(day_snows) / len(day_snows) if day_snows else 0
        max_snow = max(day_snows) if day_snows else 0
        date = d.get("date", "")
        total_7d_snow += avg_snow
        if avg_snow >= 1.0:
            storm_days.append({
                "date": date,
                "avg_snow": round(avg_snow, 1),
                "max_snow": round(max_snow, 1),
                "spread": d.get("snow_spread", 0),
                "confidence": d.get("confidence", "Medium"),
            })
        else:
            dry_days.append(date)

    # --- Analyze timeline for hourly details ---
    peak_rate = 0  # inches per hour
    peak_rate_hour = None
    heavy_snow_start = None
    heavy_snow_end = None
    max_wind = 0
    max_wind_hour = None
    snow_levels = []
    freezing_levels = []

    for h in timeline:
        snow_in = h.get("snowfall_in", 0) or 0
        wind = h.get("wind_mph", 0) or 0
        gust = h.get("wind_gust_mph", 0) or 0
        fl = h.get("freezing_level_ft")
        time_str = h.get("time", "")

        if snow_in > peak_rate:
            peak_rate = snow_in
            peak_rate_hour = time_str

        if snow_in >= 0.5 and heavy_snow_start is None:
            heavy_snow_start = time_str
        if snow_in >= 0.5:
            heavy_snow_end = time_str

        effective_wind = max(wind, gust)
        if effective_wind > max_wind:
            max_wind = effective_wind
            max_wind_hour = time_str

        if fl is not None:
            snow_levels.append((time_str, fl))
        freezing = h.get("freezing_level_ft")
        if freezing is not None:
            freezing_levels.append((time_str, freezing))

    # --- Build narrative ---
    sentences = []

    # 1. System timing
    if not storm_days:
        sentences.append("Dry pattern continues through the 7-day forecast period with "
                         "no significant precipitation expected across the Tahoe basin.")
    elif len(storm_days) == 1:
        sd = storm_days[0]
        try:
            dt = datetime.strptime(sd["date"], "%Y-%m-%d")
            day_name = dt.strftime("%A")
        except (ValueError, TypeError):
            day_name = sd["date"]
        sentences.append(f"A weather system is expected to bring snow to the region on "
                         f"{day_name}, with {sd['avg_snow']}-{sd['max_snow']}\" forecast "
                         f"at upper elevations.")
    else:
        first = storm_days[0]
        last = storm_days[-1]
        try:
            first_dt = datetime.strptime(first["date"], "%Y-%m-%d")
            last_dt = datetime.strptime(last["date"], "%Y-%m-%d")
            first_name = first_dt.strftime("%A")
            last_name = last_dt.strftime("%A")
        except (ValueError, TypeError):
            first_name = first["date"]
            last_name = last["date"]
        sentences.append(f"An active pattern brings snow from {first_name} through "
                         f"{last_name}, with {round(total_7d_snow)}\" total expected "
                         f"at upper elevations over the period.")

    # 2. Snow level evolution
    if snow_levels and len(snow_levels) >= 2:
        first_sl = snow_levels[0][1]
        last_sl = snow_levels[-1][1]
        min_sl = min(sl for _, sl in snow_levels)
        max_sl = max(sl for _, sl in snow_levels)
        if abs(max_sl - min_sl) > 500:
            # Find when min occurs
            min_time = None
            for t, sl in snow_levels:
                if sl == min_sl:
                    min_time = t
                    break
            min_label = ""
            if min_time:
                try:
                    dt = datetime.fromisoformat(min_time)
                    min_label = f" by {dt.strftime('%A %I%p').strip('0').lower()}"
                except (ValueError, TypeError):
                    pass
            if first_sl > last_sl:
                sentences.append(f"Snow levels start at {int(first_sl)}' dropping to "
                                 f"{int(min_sl)}'{min_label}.")
            else:
                sentences.append(f"Snow levels rise from {int(first_sl)}' to "
                                 f"{int(max_sl)}' through the period.")

    # 3. Peak snowfall timing
    if peak_rate >= 0.3 and peak_rate_hour:
        try:
            dt = datetime.fromisoformat(peak_rate_hour)
            time_label = dt.strftime("%A overnight" if dt.hour < 6 or dt.hour >= 22
                                     else "%A %I%p").strip("0").lower()
        except (ValueError, TypeError):
            time_label = "during the storm"
        sentences.append(f"Heaviest accumulation expected {time_label}, "
                         f"{round(peak_rate, 1)}-{round(peak_rate * 1.5, 1)}\"/hr "
                         f"above 8000'.")

    # 4. Wind impacts
    if max_wind >= 30:
        wind_desc = "gusty" if max_wind < 45 else "strong"
        sentences.append(f"{wind_desc.capitalize()} ridgetop winds "
                         f"{int(max_wind - 10)}-{int(max_wind)} mph may impact "
                         f"upper mountain operations.")

    # 5. Clearing timing
    if storm_days and dry_days:
        # Find first dry day after last storm day
        last_storm_date = storm_days[-1]["date"]
        clearing_day = None
        for dd in sorted(dry_days):
            if dd > last_storm_date:
                clearing_day = dd
                break
        if clearing_day:
            try:
                dt = datetime.strptime(clearing_day, "%Y-%m-%d")
                day_name = dt.strftime("%A")
                sentences.append(f"Clearing expected {day_name} with cold overnight "
                                 f"temperatures preserving snow quality.")
            except (ValueError, TypeError):
                pass

    # 6. Model confidence
    if storm_days:
        confidences = [d["confidence"] for d in storm_days]
        low_count = confidences.count("Low")
        high_count = confidences.count("High")
        avg_spread = sum(d["spread"] for d in storm_days) / len(storm_days)
        if high_count == len(storm_days):
            sentences.append("Models are in excellent agreement on timing and amounts.")
        elif low_count >= len(storm_days) // 2:
            sentences.append(f"Models show significant disagreement with "
                             f"{round(avg_spread)}\" spread between solutions -- "
                             f"amounts could vary substantially.")
        else:
            sentences.append(f"Models in reasonable agreement on timing; "
                             f"amounts range {storm_days[0]['avg_snow']}-"
                             f"{storm_days[0]['max_snow']}\" showing moderate uncertainty.")

    return " ".join(sentences) if sentences else "No significant weather expected in the extended forecast."


# ---------------------------------------------------------------------------
# Storm Archive
# ---------------------------------------------------------------------------

def log_storm_event(analysis: dict) -> dict | None:
    """Log a completed storm event to the archive.

    Called when pressure-based storm detection transitions from active to inactive.
    Saves storm totals, duration, model performance, and conditions.

    Returns the logged event or None if not applicable.
    """
    storm = analysis.get("storm", {})
    if not storm:
        return None

    event = {
        "start_date": storm.get("storm_start"),
        "end_date": datetime.now(timezone.utc).isoformat(),
        "duration_hours": storm.get("duration_hours"),
        "total_precip_in": 0,
        "peak_snow_rate": 0,
        "max_wind": 0,
        "snow_level_range": [None, None],
        "resort_totals": {},
        "snotel_totals": storm.get("station_totals", {}),
        "model_performance": {},
        "conditions_summary": "",
    }

    # Gather resort peak snow totals
    for rn, rd in analysis.get("resorts", {}).items():
        peak = rd.get("zones", {}).get("peak", {})
        event["resort_totals"][rn] = round(peak.get("snow_24h", 0), 1)

    # Total precip from best SNOTEL
    snotel_totals = storm.get("station_totals", {})
    if snotel_totals:
        event["total_precip_in"] = round(max(snotel_totals.values(), default=0), 1)

    # Peak snow rate and wind from timeline
    for rn, rd in analysis.get("resorts", {}).items():
        peak = rd.get("zones", {}).get("peak", {})
        for h in peak.get("timeline_48h", []):
            sr = h.get("snowfall_in", 0) or 0
            if sr > event["peak_snow_rate"]:
                event["peak_snow_rate"] = round(sr, 1)
            w = max(h.get("wind_mph", 0) or 0, h.get("wind_gust_mph", 0) or 0)
            if w > event["max_wind"]:
                event["max_wind"] = round(w)
            fl = h.get("freezing_level_ft")
            if fl is not None:
                if event["snow_level_range"][0] is None or fl < event["snow_level_range"][0]:
                    event["snow_level_range"][0] = int(fl)
                if event["snow_level_range"][1] is None or fl > event["snow_level_range"][1]:
                    event["snow_level_range"][1] = int(fl)
        break  # Only need one resort for timeline

    # Build conditions summary
    parts = []
    if event["total_precip_in"] > 0:
        parts.append(f"{event['total_precip_in']}\" total")
    if event["duration_hours"]:
        parts.append(f"{event['duration_hours']}h duration")
    if event["max_wind"] > 30:
        parts.append(f"gusts to {event['max_wind']} mph")
    event["conditions_summary"] = ", ".join(parts) if parts else "Minor event"

    # Model performance: compare forecast vs observed for each model
    for rn, rd in analysis.get("resorts", {}).items():
        peak = rd.get("zones", {}).get("peak", {})
        spread = peak.get("model_spread", [])
        if spread:
            for mname, md in spread[0].get("models", {}).items():
                forecast_snow = md.get("snow_in", 0)
                event["model_performance"][mname] = {
                    "forecast": round(forecast_snow, 1),
                    "actual": event["total_precip_in"],
                    "error": round(abs(forecast_snow - event["total_precip_in"]), 1),
                }
        break

    # Save to archive
    archive = get_storm_history()
    archive.append(event)
    # Keep last 50 events
    archive = archive[-50:]
    try:
        tmp = STORM_ARCHIVE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(archive, f, indent=2)
        os.replace(tmp, STORM_ARCHIVE_FILE)
    except Exception as e:
        logger.warning("Failed to save storm archive: %s", e)

    return event


def get_storm_history() -> list:
    """Load storm archive from disk. Returns list of past storm events."""
    try:
        with open(STORM_ARCHIVE_FILE) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# Summary Generator (rule-based, no LLM needed)
# ---------------------------------------------------------------------------

def generate_summary(analysis: dict) -> str:
    """Generate a natural-language AI-style summary from the analysis data."""
    lines = []
    cur = analysis.get("current_conditions", {})
    obs = cur.get("observation", {})

    # Current snapshot
    if obs:
        lines.append(f"Currently {obs.get('conditions', 'clear').lower()} at lake level with "
                      f"temperatures at {obs.get('temp_f', '?')}F (feels like {obs.get('feels_like_f', '?')}F).")
    elif cur.get("lake_level_temp_f"):
        lines.append(f"Lake level temperature is {cur['lake_level_temp_f']}F.")

    # Snow level
    sl = cur.get("snow_level_ft")
    if sl is not None and sl > 0:
        lines.append(f"Snow level is at {sl}ft.")
    elif sl == 0:
        lines.append("Snow level is at the valley floor — any precip will fall as snow.")

    # Active precip?
    if cur.get("precipitation_active"):
        lines.append("Precipitation is active in the forecast period.")

    # Next-day outlook
    comp = analysis.get("comparison", {}).get("resorts", {})
    max_snow = 0
    best_resort = None
    for rn, rc in comp.items():
        ps = rc.get("peak_24h_snow_in", 0)
        if ps > max_snow:
            max_snow = ps
            best_resort = rn

    if max_snow > 0:
        # Include ensemble range if available
        hero = analysis.get("hero_stats", {})
        snow_range = hero.get("snow_24h_range")
        if snow_range and snow_range.get("p10") is not None:
            lines.append(f"Next 24h: {best_resort} leads with {max_snow}\" expected "
                         f"({snow_range['p10']}-{snow_range['p90']}\" range).")
        else:
            lines.append(f"Next 24h: {best_resort} leads with {max_snow}\" expected at the summit.")
    else:
        lines.append("No significant snowfall expected in the next 24 hours.")

    # Multi-day outlook
    spread = analysis.get("multi_model_spread_peak", [])
    storm_days = [d for d in spread if d.get("snow_range", (0, 0))[1] > 2]
    if storm_days:
        dates = [d["date"] for d in storm_days[:3]]
        lines.append(f"Models show potential snow on: {', '.join(dates)}.")
        low_conf = [d for d in storm_days if d.get("confidence") == "Low"]
        if low_conf:
            lines.append("Model agreement is low for some storm days — uncertainty remains.")
    else:
        lines.append("Extended forecast (15-day) shows dry conditions across all models.")

    # Snowpack
    snotel = analysis.get("snotel_current", {})
    depths = [(n, d.get("snow_depth_in")) for n, d in snotel.items()
              if d.get("snow_depth_in") is not None and d.get("snow_depth_in") > 0]
    if depths:
        best = max(depths, key=lambda x: x[1])
        lines.append(f"Deepest snowpack: {best[0]} at {best[1]}\".")

    # Avalanche
    avy = analysis.get("avalanche", {})
    if avy and "danger_label" in avy:
        lines.append(f"Avalanche danger: {avy['danger_label']} ({avy.get('travel_advice', '')}).")

    # Season context from history
    season = analysis.get("season_stats", {})
    if season:
        peaks = [(n, s.get("SNWD", {}).get("peak", 0)) for n, s in season.items()]
        if peaks:
            best_p = max(peaks, key=lambda x: x[1])
            lines.append(f"Season peak snowpack: {best_p[0]} hit {best_p[1]}\" this winter.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Core Analysis
# ---------------------------------------------------------------------------

def analyze_all(obs: dict, nws: dict, om: dict, snotel: dict,
                afd_text: str, avy: dict, hrrr: dict,
                nws_grids: dict | None = None,
                sounding: dict | None = None,
                ensemble: dict | None = None,
                synoptic: dict | None = None,
                rwis: list | None = None,
                radar_nowcast: dict | None = None,
                cssl: dict | None = None) -> dict:
    now = datetime.now(timezone.utc)

    # --- Load model skill weights and bias corrections ---
    try:
        from forecast_verification import get_model_weights, get_bias_corrections
        model_weights = get_model_weights()
        bias_corrections = get_bias_corrections()
    except Exception as e:
        logger.debug("Forecast verification weights unavailable: %s", e)
        model_weights = None
        bias_corrections = {}

    # Compute observed lapse rate from multi-source observations
    # Fuses: sounding (free air), SNOTEL (surface), Synoptic (surface)
    observed_lapse_rate = compute_lapse_rate(snotel, synoptic=synoptic,
                                             sounding=sounding)

    # Base conditions from observation or NWS hourly
    base_temp_f = None
    base_wind_mph = 0.0
    base_wind_dir = 0.0

    if obs:
        base_temp_f = obs.get("temp_f")
        base_wind_mph = obs.get("wind_mph", 0)
        base_wind_dir = obs.get("wind_dir_deg", 0)
    else:
        hourly = nws.get("hourly", [])
        if hourly:
            base_temp_f = hourly[0].get("temperature")
            try:
                base_wind_mph = float(hourly[0].get("windSpeed", "0 mph").split()[0])
            except Exception:
                pass
            base_wind_dir = DIR_MAP.get(hourly[0].get("windDirection", "W"), 270)

    base_temp_c = (base_temp_f - 32) * 5 / 9 if base_temp_f else 0.0
    base_elev_m = 6225 * 0.3048

    # Snow level
    profile = [(ft * 0.3048, estimate_temp_c(base_temp_c, base_elev_m, ft * 0.3048,
                                              observed_lapse_rate=observed_lapse_rate))
               for ft in range(5000, 11500, 250)]
    threshold = 1.0
    snow_level_m = None
    for i in range(len(profile) - 1):
        a1, t1 = profile[i]
        a2, t2 = profile[i+1]
        if (t1 >= threshold) != (t2 >= threshold):
            frac = (threshold - t1) / (t2 - t1)
            snow_level_m = a1 + frac * (a2 - a1)
            break
    if snow_level_m is None and profile[0][1] < threshold:
        snow_level_m = 0.0
    snow_level_ft = int(snow_level_m / 0.3048) if snow_level_m is not None else None

    # Freezing level
    if base_temp_c <= 0:
        freeze_ft = 0
    else:
        freeze_ft = int((base_elev_m + base_temp_c / (5.5/1000)) / 0.3048)

    # Parse Open-Meteo for each resort zone using per-resort grid data
    # Each resort gets its own Open-Meteo grid point (base location) for
    # accurate local weather — resorts are 15-35 miles apart.
    resort_data = {}
    for rn, resort in RESORTS.items():
        # Use the resort's base location for the Open-Meteo grid point
        resort_om = om.get(rn) if isinstance(om, dict) and rn in om else om
        zones = {}
        for zk in ("base", "mid", "peak"):
            loc = resort[zk]
            parsed = parse_open_meteo(resort_om, loc["elev_ft"], observed_lapse_rate=observed_lapse_rate)
            zones[zk] = {
                "label": loc["label"],
                "elev_ft": loc["elev_ft"],
                "parsed": parsed,
            }
        # Nearest SNOTEL
        ns = [snotel[s] for s in resort.get("nearest_snotel", [])
              if s in snotel and "error" not in snotel[s]]
        resort_data[rn] = {"zones": zones, "aspect": resort["aspect"], "nearest_snotel": ns}

    # Validate snow level against sounding and model data
    model_freezing_levels = []
    # Extract freezing levels from all available model first-hour data
    for rn, rd in resort_data.items():
        for zk, zd in rd["zones"].items():
            p = zd["parsed"]
            if "error" not in p:
                for mname, mhours in p["models"].items():
                    if mhours:
                        fl = mhours[0].get("freezing_level_ft")
                        if fl is not None:
                            model_freezing_levels.append(fl)
        break  # Only need one resort's models for freezing level

    snow_level_validation = validate_snow_level(
        snow_level_ft, sounding or {}, model_freezing_levels
    )
    validated_snow_level_ft = snow_level_validation.get("validated_ft", snow_level_ft)

    # Current conditions block
    current = {
        "timestamp": now.isoformat(),
        "observation": obs,
        "lake_level_temp_f": base_temp_f,
        "wind_mph": round(base_wind_mph, 1),
        "wind_dir": wind_dir_str(base_wind_dir),
        "snow_level_ft": validated_snow_level_ft,
        "freezing_level_ft": freeze_ft,
        "precipitation_active": False,
        "snow_level_validation": snow_level_validation,
    }

    # Determine if precip is active from hourly
    hourly = nws.get("hourly", [])
    for p in hourly[:3]:
        prob = p.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
        if prob > 50:
            current["precipitation_active"] = True
            break

    # Build per-resort output with skill-weighted multi-model blend
    resorts_out = {}
    for rn, rd in resort_data.items():
        zones_out = {}
        for zk, zd in rd["zones"].items():
            p = zd["parsed"]
            if "error" not in p:
                # Skill-weighted blend of all models (BMA-style)
                blended = skill_weighted_blend(p, model_weights)
                # Fall back to GFS if blend fails (copy to avoid mutating model data)
                if blended:
                    primary = blended
                else:
                    primary = [dict(h) for h in p["models"].get("GFS", [])]

                # Apply bias corrections to blended forecast
                # Only snowfall (multiplicative). Temp bias is daily-level, not per-hour.
                if bias_corrections and primary:
                    corr = (bias_corrections.get("blend")
                            or bias_corrections.get("gfs")
                            or next(iter(bias_corrections.values()), {}))
                    if "snow_in" in corr:
                        snow_corr = corr["snow_in"]
                        if abs(snow_corr) > 0.1:
                            factor = max(0.5, min(1.5, 1.0 - snow_corr / 10.0))
                            for h in primary:
                                if h.get("snowfall_in", 0) > 0:
                                    h["snowfall_in"] = round(h["snowfall_in"] * factor, 1)

                # Current zone snapshot (first available hour)
                snap = primary[0] if primary else {}

                # Apply terrain-adjusted temperature to snapshot
                zone_cfg = RESORTS[rn][zk]
                aspect_deg = zone_cfg.get("aspect_deg")
                if snap and aspect_deg is not None:
                    snap_temp_c = snap.get("temp_c")
                    snap_cc = snap.get("cloud_cover_pct", 50) or 50
                    if snap_temp_c is not None:
                        try:
                            snap_time = datetime.fromisoformat(snap.get("time", ""))
                            hour_of_day = snap_time.hour
                        except (ValueError, TypeError):
                            hour_of_day = now.hour
                        adj_temp_c = terrain_adjusted_temperature(
                            snap_temp_c, zone_cfg["elev_ft"] * 0.3048,
                            aspect_deg, hour_of_day, snap_cc
                        )
                        snap["temp_c_terrain_adj"] = round(adj_temp_c, 1)
                        snap["temp_f_terrain_adj"] = round(adj_temp_c * 9/5 + 32, 1)

                # Compute precip phase probability for current conditions
                if snap and snap.get("temp_c") is not None:
                    wb = wet_bulb_temp_c(
                        snap.get("temp_c_terrain_adj", snap["temp_c"]),
                        snap.get("humidity_pct")
                    )
                    fl_m = (snap.get("freezing_level_ft") or 0) * 0.3048 if snap.get("freezing_level_ft") else None
                    snap["precip_phase"] = precip_phase_probability(
                        wb, zone_cfg["elev_ft"] * 0.3048, fl_m
                    )

                # Apply lake effect enhancement for east-shore resorts
                is_east_shore = RESORTS[rn].get("east_shore", False)
                lake_effect_factor = 1.0
                if is_east_shore and snap:
                    wind_dir = snap.get("wind_dir_deg", 0) or 0
                    wind_spd = snap.get("wind_mph", 0) or 0
                    air_tc = snap.get("temp_c_terrain_adj", snap.get("temp_c", 0)) or 0
                    lake_effect_factor = lake_effect_enhancement(
                        wind_dir, wind_spd, air_tc
                    )
                    snap["lake_effect_factor"] = lake_effect_factor

                # 48h hourly timeline
                timeline_48h = primary[:48]

                # Day/night buckets
                buckets = aggregate_daily(primary)

                # 24h snow total (forecast) from blended model
                snow_24h = sum(h["snowfall_in"] for h in primary[:24])

                # Apply lake effect enhancement to snow totals
                if lake_effect_factor > 1.0:
                    snow_24h *= lake_effect_factor

                # 7-day forecast snow total from blended model
                snow_7d_forecast = sum(h["snowfall_in"] for h in primary[:168])
                if lake_effect_factor > 1.0:
                    snow_7d_forecast *= lake_effect_factor

                # Apply snow settling to 24h total for "settled depth" estimate
                avg_temp_f = snap.get("temp_f", 32) or 32
                avg_wind = snap.get("wind_mph", 5) or 5
                settled_24h = settled_snow_depth(snow_24h, 12.0, avg_temp_f, avg_wind)

                # Blend with NWS forecaster-edited gridpoint data
                nws_blend = blend_nws_grids(
                    nws_grids, snow_24h, snow_7d_forecast
                ) if nws_grids else {"snow_24h": snow_24h,
                                      "snow_7d": snow_7d_forecast,
                                      "method": "model_only"}

                # Multi-model spread (daily)
                model_spread = multi_model_spread(p)

                # Calibrate confidence using ensemble spread
                model_spread = calibrated_confidence(model_spread, ensemble)

                zones_out[zk] = {
                    "label": zd["label"],
                    "elev_ft": zd["elev_ft"],
                    "current": snap,
                    "timeline_48h": timeline_48h,
                    "day_night_buckets": buckets,
                    "snow_24h": round(nws_blend["snow_24h"], 1),
                    "snow_24h_settled": round(settled_24h, 1),
                    "snow_7d_forecast": round(nws_blend["snow_7d"], 1),
                    "model_spread": model_spread,
                    "nws_blend": nws_blend,
                    "blend_method": "skill_weighted" if blended else "gfs_fallback",
                    "model_weights_used": model_weights or {},
                    "lake_effect_factor": round(lake_effect_factor, 3),
                }
            else:
                zones_out[zk] = {"label": zd["label"], "elev_ft": zd["elev_ft"], "error": p["error"]}

        resorts_out[rn] = {
            "zones": zones_out,
            "aspect": rd["aspect"],
            "nearest_snotel": rd["nearest_snotel"],
        }

    # Comparison
    comparison = {}
    for rn, rd in resorts_out.items():
        base_z = rd["zones"].get("base", {})
        mid_z = rd["zones"].get("mid", {})
        peak_z = rd["zones"].get("peak", {})
        bc = base_z.get("current", {})
        pc = peak_z.get("current", {})
        sn = rd.get("nearest_snotel", [])
        max_depth = max((s.get("snow_depth_in", 0) or 0 for s in sn), default=0)
        max_swe = max((s.get("swe_in", 0) or 0 for s in sn), default=0)

        comparison[rn] = {
            "base_elev_ft": base_z.get("elev_ft", 0),
            "peak_elev_ft": peak_z.get("elev_ft", 0),
            "vert_ft": peak_z.get("elev_ft", 0) - base_z.get("elev_ft", 0),
            "base_temp_f": bc.get("temp_f"),
            "peak_temp_f": pc.get("temp_f"),
            "base_feels_like_f": bc.get("feels_like_f"),
            "peak_feels_like_f": pc.get("feels_like_f"),
            "base_precip_type": bc.get("precip_type", "None"),
            "peak_precip_type": pc.get("precip_type", "None"),
            "base_24h_snow_in": base_z.get("snow_24h", 0),
            "mid_24h_snow_in": mid_z.get("snow_24h", 0),
            "peak_24h_snow_in": peak_z.get("snow_24h", 0),
            "peak_7d_forecast_in": peak_z.get("snow_7d_forecast", 0),
            "peak_slr": pc.get("slr"),
            "peak_quality": pc.get("snow_quality", "N/A"),
            "snowpack_depth_in": max_depth,
            "snowpack_swe_in": max_swe,
            "aspect": rd["aspect"],
        }

    # Multi-model spread for peak of highest resort (for summary)
    best_peak = None
    for rn in RESORTS:
        pk = resorts_out.get(rn, {}).get("zones", {}).get("peak", {})
        if "model_spread" in pk:
            best_peak = pk["model_spread"]
            break

    # Historical + Season SNOTEL — include display stations + all resort-linked stations
    hist_stations = {"Mt Rose Ski Area", "CSS Lab", "Independence Lake"}
    for rn, resort in RESORTS.items():
        for sname in resort.get("nearest_snotel", []):
            hist_stations.add(sname)
    snotel_history = {}
    season_stats = {}
    for sname in hist_stations:
        if sname in SNOTEL_STATIONS:
            st = SNOTEL_STATIONS[sname]
            snotel_history[sname] = fetch_snotel_history(st["id"], st["state"], days=10)
            season_stats[sname] = fetch_snotel_season(st["id"], st["state"])

    # AFD section extraction
    afd_snippet = ""
    if afd_text:
        upper = afd_text.upper()
        for hdr in [".DISCUSSION", ".SHORT TERM", ".NEAR TERM", ".SYNOPSIS", ".KEY MESSAGES"]:
            idx = upper.find(hdr)
            if idx >= 0:
                end = afd_text.find("\n&&", idx + 1)
                if end < 0: end = afd_text.find("\n.", idx + len(hdr))
                if end < 0: end = min(idx + 1500, len(afd_text))
                afd_snippet = afd_text[idx:end].strip()
                break
        if not afd_snippet:
            afd_snippet = afd_text[:1500].strip()

    # Include lapse rate info in current conditions
    if observed_lapse_rate is not None:
        current["observed_lapse_rate_c_km"] = observed_lapse_rate

    # Extract sunrise/sunset from Open-Meteo daily data
    solar = {}
    # om may be a dict keyed by resort name (from fetch_open_meteo_multi) or a
    # plain Open-Meteo response dict. Try to find daily sunrise/sunset from any
    # available source.
    _om_sources = list(om.values()) if isinstance(om, dict) and all(isinstance(v, dict) for v in om.values()) else [om]
    for _om_src in _om_sources:
        if not isinstance(_om_src, dict) or "error" in _om_src:
            continue
        daily = _om_src.get("daily", {})
        sr_list = daily.get("sunrise", [])
        ss_list = daily.get("sunset", [])
        if sr_list and ss_list:
            try:
                sr_dt = datetime.fromisoformat(sr_list[0])
                ss_dt = datetime.fromisoformat(ss_list[0])
                daylight_sec = (ss_dt - sr_dt).total_seconds()
                daylight_hours = round(daylight_sec / 3600, 2)
                solar = {
                    "sunrise": sr_dt.strftime("%-I:%M %p"),
                    "sunset": ss_dt.strftime("%-I:%M %p"),
                    "sunrise_iso": sr_list[0],
                    "sunset_iso": ss_list[0],
                    "daylight_hours": daylight_hours,
                }
            except (ValueError, TypeError, IndexError):
                pass
            break

    result = {
        "generated": now.isoformat(),
        "current_conditions": current,
        "resorts": resorts_out,
        "comparison": {"resorts": comparison},
        "snotel_current": snotel,
        "snotel_history": snotel_history,
        "season_stats": season_stats,
        "avalanche": avy,
        "forecaster_discussion": afd_snippet or "Not available",
        "multi_model_spread_peak": best_peak or [],
        "model_weights": model_weights or {},
        "bias_corrections": bias_corrections,
        "ensemble": ensemble,
        "sounding": sounding,
        "rwis": rwis or [],
        "radar_nowcast": radar_nowcast or {},
        "cssl": cssl or {},
        "hrrr": hrrr or {},
        "solar": solar,
    }

    # Rankings
    ranked_snow = sorted(comparison.items(), key=lambda x: x[1]["peak_24h_snow_in"], reverse=True)
    ranked_depth = sorted(comparison.items(), key=lambda x: x[1]["snowpack_depth_in"], reverse=True)
    ranked_slr = sorted(comparison.items(), key=lambda x: x[1].get("peak_slr") or 0, reverse=True)
    result["comparison"]["rankings"] = {
        "most_new_snow": ranked_snow[0][0] if ranked_snow else None,
        "deepest_snowpack": ranked_depth[0][0] if ranked_depth else None,
        "best_powder_quality": ranked_slr[0][0] if ranked_slr else None,
    }

    # Historical snowfall from SNOTEL (sum of positive daily depth changes)
    hist_7d_snow = {}
    hist_24h_snow = {}
    for sname, hist in snotel_history.items():
        snwd = hist.get("SNWD", [])
        if len(snwd) >= 2:
            # 7-day: sum positive depth increases over last 7 days
            recent = snwd[-8:] if len(snwd) >= 8 else snwd
            total_new = 0.0
            for i in range(1, len(recent)):
                delta = (recent[i][1] or 0) - (recent[i-1][1] or 0)
                if delta > 0:
                    total_new += delta
            hist_7d_snow[sname] = round(total_new, 1)

            # 24h: last day's depth increase (last 2 readings)
            delta_24h = (snwd[-1][1] or 0) - (snwd[-2][1] or 0)
            hist_24h_snow[sname] = round(max(delta_24h, 0), 1)

    result_hist_7d = max(hist_7d_snow.values()) if hist_7d_snow else 0
    result_hist_7d_station = max(hist_7d_snow, key=hist_7d_snow.get) if hist_7d_snow else ""
    result_hist_24h = max(hist_24h_snow.values()) if hist_24h_snow else 0
    result_hist_24h_station = max(hist_24h_snow, key=hist_24h_snow.get) if hist_24h_snow else ""

    # Per-resort snowfall summary
    for rn, resort in RESORTS.items():
        rd = resorts_out.get(rn, {})
        peak_z = rd.get("zones", {}).get("peak", {})
        # Forecast from peak zone
        fcast_24h = peak_z.get("snow_24h", 0)
        fcast_7d = peak_z.get("snow_7d_forecast", 0)
        # Historical from nearest SNOTEL (best station)
        resort_snotel_names = resort.get("nearest_snotel", [])
        best_hist_24h = 0
        best_hist_24h_station = ""
        best_hist_7d = 0
        best_hist_7d_station = ""
        for sn in resort_snotel_names:
            if sn in hist_24h_snow and hist_24h_snow[sn] > best_hist_24h:
                best_hist_24h = hist_24h_snow[sn]
                best_hist_24h_station = sn
            if sn in hist_7d_snow and hist_7d_snow[sn] > best_hist_7d:
                best_hist_7d = hist_7d_snow[sn]
                best_hist_7d_station = sn
        rd["snowfall_summary"] = {
            "hist_24h_in": best_hist_24h,
            "hist_24h_station": best_hist_24h_station,
            "hist_7d_in": best_hist_7d,
            "hist_7d_station": best_hist_7d_station,
            "forecast_24h_in": fcast_24h,
            "forecast_7d_in": fcast_7d,
        }

    # Hero stats — the 4 most important numbers for at-a-glance scanning
    hero_temp_f = current.get("observation", {}).get("temp_f") or current.get("lake_level_temp_f")
    if hero_temp_f is None:
        for rn in RESORTS:
            snap = resorts_out.get(rn, {}).get("zones", {}).get("mid", {}).get("current", {})
            if snap.get("temp_f") is not None:
                hero_temp_f = snap["temp_f"]
                break

    # Reuse ranked data from above instead of re-iterating
    ranked_snotel = sorted(snotel.items(), key=lambda x: x[1].get("snow_depth_in", 0) or 0, reverse=True)
    hero_snowpack = (ranked_snotel[0][1].get("snow_depth_in", 0) or 0) if ranked_snotel else 0
    hero_snowpack_station = ranked_snotel[0][0] if ranked_snotel else ""

    hero_24h = round(ranked_snow[0][1]["peak_24h_snow_in"], 1) if ranked_snow else 0
    hero_24h_resort = ranked_snow[0][0] if ranked_snow else ""

    hero_72h = 0
    hero_72h_resort = ""
    hero_7d_forecast = 0
    hero_7d_forecast_resort = ""
    for rn in RESORTS:
        peak_z = resorts_out.get(rn, {}).get("zones", {}).get("peak", {})
        # Use blended day_night_buckets for 72h total (already skill-weighted + NWS-blended)
        buckets = peak_z.get("day_night_buckets", [])
        # Sum snow from first 3 days of buckets (each day has Day+Night)
        bucket_dates = sorted(set(b["date"] for b in buckets))
        total_3d = sum(b["snow_in"] for b in buckets if b["date"] in bucket_dates[:3])
        if total_3d > hero_72h:
            hero_72h = round(total_3d, 1)
            hero_72h_resort = rn
        # Use the zone's already-blended snow_7d_forecast (skill-weighted + NWS gridpoint blended)
        total_7d = peak_z.get("snow_7d_forecast", 0)
        if total_7d > hero_7d_forecast:
            hero_7d_forecast = round(total_7d, 1)
            hero_7d_forecast_resort = rn

    # Extract ensemble uncertainty ranges for hero stats
    ens_24h_range = None
    ens_7d_range = None
    if ensemble and "models" in ensemble:
        # Aggregate ensemble percentiles across first day and first 7 days
        all_daily = []
        for model_label, daily_list in ensemble["models"].items():
            if isinstance(daily_list, list):
                all_daily = daily_list  # Use whichever has data
                break
        if all_daily:
            # 24h range from first day
            if len(all_daily) >= 1:
                d = all_daily[0]
                ens_24h_range = {
                    "p10": d.get("snow_p10", 0),
                    "p50": d.get("snow_p50", 0),
                    "p90": d.get("snow_p90", 0),
                }
            # 7d range from sum of daily percentiles
            if len(all_daily) >= 2:
                p10_sum = sum(d.get("snow_p10", 0) or 0 for d in all_daily[:7])
                p50_sum = sum(d.get("snow_p50", 0) or 0 for d in all_daily[:7])
                p90_sum = sum(d.get("snow_p90", 0) or 0 for d in all_daily[:7])
                ens_7d_range = {
                    "p10": round(p10_sum, 1),
                    "p50": round(p50_sum, 1),
                    "p90": round(p90_sum, 1),
                }

    result["hero_stats"] = {
        "temp_f": hero_temp_f,
        "snowpack_in": hero_snowpack,
        "snowpack_station": hero_snowpack_station,
        "snow_24h_in": hero_24h,
        "snow_24h_resort": hero_24h_resort,
        "snow_24h_range": ens_24h_range,
        "snow_72h_in": hero_72h,
        "snow_72h_resort": hero_72h_resort,
        "snow_24h_hist_in": result_hist_24h,
        "snow_24h_hist_station": result_hist_24h_station,
        "snow_7d_hist_in": result_hist_7d,
        "snow_7d_hist_station": result_hist_7d_station,
        "snow_7d_forecast_in": hero_7d_forecast,
        "snow_7d_forecast_resort": hero_7d_forecast_resort,
        "snow_7d_range": ens_7d_range,
    }
    result["hist_7d_snow"] = hist_7d_snow

    # Generate summary
    result["summary"] = generate_summary(result)

    # Storm narrative (meteorologist-style briefing)
    result["storm_narrative"] = generate_storm_narrative(result)

    return result


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

W = 78  # report width

def _f(val, suf="", default="N/A"):
    if val is None: return default
    return f"{val}{suf}"

def _bar(n, max_n=20, char="*"):
    """Simple ASCII bar chart."""
    if n <= 0: return ""
    scaled = min(int(n / max_n * 20), 40) if max_n > 0 else 0
    return char * max(1, scaled) if n > 0 else ""


def format_report(a: dict, compact: bool = False) -> str:
    L = []

    # Header
    L.append("=" * W)
    L.append("  LAKE TAHOE SNOW CONDITIONS REPORT")
    try:
        dt = datetime.fromisoformat(a["generated"])
        L.append(f"  {dt.strftime('%A, %B %d %Y  %I:%M %p %Z')}")
    except Exception: L.append(f"  {a['generated']}")
    L.append("=" * W)

    # ── Summary ──
    summary = a.get("summary", "")
    if summary:
        L.append("")
        L.append("SUMMARY")
        L.append("-" * W)
        # Word-wrap at ~74 chars
        words = summary.split()
        line = "  "
        for w in words:
            if len(line) + len(w) + 1 > W - 2:
                L.append(line)
                line = "  " + w
            else:
                line += " " + w if line.strip() else "  " + w
        if line.strip(): L.append(line)

    # ── Current Conditions ──
    cur = a["current_conditions"]
    obs = cur.get("observation", {})
    L.append("")
    L.append("CURRENT CONDITIONS")
    L.append("-" * W)
    if obs:
        L.append(f"  Station:      {obs.get('station', '?')} | {obs.get('conditions', '')}")
        L.append(f"  Temperature:  {obs.get('temp_f', '?')}F  (feels like {obs.get('feels_like_f', '?')}F)")
        L.append(f"  Humidity:     {obs.get('humidity_pct', 'N/A')}%")
        gust = f"  gusting {obs['wind_gust_mph']} mph" if obs.get('wind_gust_mph', 0) > 0 else ""
        L.append(f"  Wind:         {obs['wind_dir']} at {obs['wind_mph']} mph{gust}")
        if obs.get('visibility_mi', 0) > 0:
            L.append(f"  Visibility:   {obs['visibility_mi']} mi")
        if obs.get('barometer_inhg', 0) > 0:
            L.append(f"  Barometer:    {obs['barometer_inhg']}\" Hg")
    else:
        L.append(f"  Temperature:  {_f(cur.get('lake_level_temp_f'), 'F')}")
        L.append(f"  Wind:         {cur.get('wind_dir', 'N/A')} at {cur.get('wind_mph', 0)} mph")

    L.append(f"  Snow Level:   {_f(cur.get('snow_level_ft'), ' ft')}")
    L.append(f"  Freeze Level: {_f(cur.get('freezing_level_ft'), ' ft')}")

    # ── Avalanche ──
    avy = a.get("avalanche", {})
    if avy and "danger_label" in avy:
        L.append("")
        L.append("AVALANCHE CONDITIONS (Sierra Avalanche Center)")
        L.append("-" * W)
        L.append(f"  Danger:  {avy['danger_label']} ({avy.get('danger_level', '?')}/5)")
        if avy.get("travel_advice"):
            L.append(f"  Advice:  {avy['travel_advice']}")
        if avy.get("link"):
            L.append(f"  Detail:  {avy['link']}")

    # ── Per-Resort ──
    for rn in RESORTS:
        rd = a.get("resorts", {}).get(rn, {})
        L.append("")
        L.append("=" * W)
        L.append(f"  {rn.upper()}  (aspect: {rd.get('aspect', '?')})")
        L.append("=" * W)

        for zk in ("base", "mid", "peak"):
            zd = rd.get("zones", {}).get(zk, {})
            if not zd: continue
            snap = zd.get("current", {})
            L.append("")
            L.append(f"  [{zk.upper()}] {zd.get('label', '?')}  ({zd.get('elev_ft', '?')} ft)")
            L.append(f"  {'.' * (W - 4)}")
            L.append(f"    Temp:        {_f(snap.get('temp_f'), 'F')}  "
                      f"(feels like {_f(snap.get('feels_like_f'), 'F')})")
            L.append(f"    Precip:      {snap.get('precip_type', 'None')}")
            L.append(f"    SLR:         {snap.get('slr', '?')}:1  |  {snap.get('snow_quality', 'N/A')}")
            L.append(f"    24h Snow:    {zd.get('snow_24h', 0)}\"")
            L.append(f"    Wind:        {snap.get('wind_dir', '?')} {snap.get('wind_mph', 0)} mph  "
                      f"(gusts {snap.get('wind_gust_mph', 0)} mph)")

        # Day/night accumulation table for peak zone
        peak_z = rd.get("zones", {}).get("peak", {})
        buckets = peak_z.get("day_night_buckets", [])
        if buckets:
            L.append("")
            L.append(f"  {rn} Peak — Day/Night Snow Accumulation (GFS)")
            L.append(f"  {'Date':12s} {'Period':6s} {'Snow':>6s} {'Liquid':>7s} {'Hi/Lo F':>8s} "
                      f"{'Feels':>6s} {'Wind':>8s} {'Type':>5s}")
            L.append(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*8} {'-'*5}")
            for b in buckets[:20]:  # 10 days of day+night
                hi = _f(b.get("temp_high_f"), "")
                lo = _f(b.get("temp_low_f"), "")
                fl = _f(b.get("feels_like_low_f"), "")
                wg = f"{b.get('wind_avg_mph', 0)}/{b.get('wind_max_gust_mph', 0)}"
                L.append(f"  {b['date']:12s} {b['period']:6s} {b['snow_in']:5.1f}\" "
                          f"{b['liquid_in']:6.2f}\" {hi:>4s}/{lo:<3s} {fl:>5s} "
                          f"{wg:>8s} {b['precip_type']:>5s}")

        # Hourly timeline (48h) for peak
        if not compact:
            timeline = peak_z.get("timeline_48h", [])
            if timeline:
                L.append("")
                L.append(f"  {rn} Peak — 48h Hourly Timeline (GFS)")
                L.append(f"  {'Hour':16s} {'Temp':>5s} {'Feel':>5s} {'Snow':>5s} "
                          f"{'Wind':>6s} {'Gust':>5s} {'Dir':>4s} {'Type':>5s}")
                L.append(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*4} {'-'*5}")
                for h in timeline:
                    try:
                        tdt = datetime.fromisoformat(h["time"])
                        tstr = tdt.strftime("%a %m/%d %I%p")
                    except Exception: tstr = h.get("time", "?")[:16]
                    L.append(f"  {tstr:16s} {_f(h.get('temp_f'), ''):>5s} "
                              f"{_f(h.get('feels_like_f'), ''):>5s} "
                              f"{h.get('snowfall_in', 0):4.1f}\" "
                              f"{h.get('wind_mph', 0):5.1f} {h.get('wind_gust_mph', 0):5.1f} "
                              f"{h.get('wind_dir', ''):>4s} {h.get('precip_type', ''):>5s}")

        # Multi-model spread for peak
        spread = peak_z.get("model_spread", [])
        if spread:
            L.append("")
            L.append(f"  {rn} Peak — Multi-Model Snow Forecast (daily)")
            L.append(f"  {'Date':12s} {'GFS':>6s} {'ECMWF':>6s} {'ICON':>6s} "
                      f"{'Spread':>7s} {'Conf':>6s}  {'Agreement':16s}")
            L.append(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6}  {'-'*16}")
            for d in spread[:15]:
                models = d.get("models", {})
                gfs_s = models.get("GFS", {}).get("snow_in", 0)
                ecmwf_s = models.get("ECMWF", {}).get("snow_in", 0)
                icon_s = models.get("ICON", {}).get("snow_in", 0)
                sr = d.get("snow_range", (0, 0))
                conf = d.get("confidence", "?")
                bar = _bar(max(gfs_s, ecmwf_s, icon_s), 15)
                L.append(f"  {d['date']:12s} {gfs_s:5.1f}\" {ecmwf_s:5.1f}\" {icon_s:5.1f}\" "
                          f"{d.get('snow_spread', 0):6.1f}\" {conf:>6s}  {bar}")

        # Nearby SNOTEL
        ns = rd.get("nearest_snotel", [])
        if ns:
            L.append("")
            L.append(f"  Nearby SNOTEL:")
            for s in ns:
                L.append(f"    {s.get('name', '?')} ({s.get('elev_ft', '?')}ft): "
                          f"Depth={_f(s.get('snow_depth_in'), '\"')} | "
                          f"SWE={_f(s.get('swe_in'), '\"')} | "
                          f"Temp={_f(s.get('temp_f'), 'F')}")

    # ── Mountain Comparison ──
    comp = a.get("comparison", {}).get("resorts", {})
    rankings = a.get("comparison", {}).get("rankings", {})
    L.append("")
    L.append("=" * W)
    L.append("  MOUNTAIN COMPARISON")
    L.append("=" * W)

    cw = 22
    hdr = f"  {'':18s}"
    for rn in RESORTS: hdr += f" {rn:>{cw}s}"
    L.append(hdr)
    L.append("  " + "-" * (18 + len(RESORTS) * (cw + 1)))

    def _row(label, key, suf=""):
        r = f"  {label:18s}"
        for rn in RESORTS:
            v = comp.get(rn, {}).get(key)
            r += f" {_f(v, suf):>{cw}s}"
        return r

    L.append(_row("Base Elev", "base_elev_ft", " ft"))
    L.append(_row("Peak Elev", "peak_elev_ft", " ft"))
    L.append(_row("Vert Drop", "vert_ft", " ft"))
    L.append(_row("Base Temp", "base_temp_f", "F"))
    L.append(_row("Peak Temp", "peak_temp_f", "F"))
    L.append(_row("Base Feels Like", "base_feels_like_f", "F"))
    L.append(_row("Peak Feels Like", "peak_feels_like_f", "F"))
    L.append(_row("Base Precip", "base_precip_type"))
    L.append(_row("Peak Precip", "peak_precip_type"))
    L.append(_row("Base 24h Snow", "base_24h_snow_in", "\""))
    L.append(_row("Mid 24h Snow", "mid_24h_snow_in", "\""))
    L.append(_row("Peak 24h Snow", "peak_24h_snow_in", "\""))
    L.append(_row("Peak SLR", "peak_slr", ":1"))
    L.append(_row("Peak Quality", "peak_quality"))
    L.append(_row("Snowpack Depth", "snowpack_depth_in", "\""))
    L.append(_row("Snowpack SWE", "snowpack_swe_in", "\""))
    L.append(_row("Aspect", "aspect"))

    L.append("")
    L.append(f"  Best new snow:       {rankings.get('most_new_snow', 'N/A')}")
    L.append(f"  Best powder quality: {rankings.get('best_powder_quality', 'N/A')}")
    L.append(f"  Deepest snowpack:    {rankings.get('deepest_snowpack', 'N/A')}")

    # ── Historical SNOTEL ──
    hist = a.get("snotel_history", {})
    season = a.get("season_stats", {})
    if hist or season:
        L.append("")
        L.append("-" * W)
        L.append("SNOWPACK HISTORY & SEASON STATS")
        L.append("-" * W)

        for sname in hist:
            h = hist[sname]
            s = season.get(sname, {})
            L.append(f"  {sname} ({SNOTEL_STATIONS.get(sname, {}).get('elev_ft', '?')}ft)")

            # 10-day depth trend
            depths = h.get("SNWD", [])
            if depths:
                L.append("    10-day depth:")
                trend = "    "
                for date, val in depths:
                    d = date.split("-")
                    trend += f"{d[1]}/{d[2]}:{val}\"  "
                L.append(trend)

            # Season stats
            snwd = s.get("SNWD", {})
            wteq = s.get("WTEQ", {})
            if snwd:
                L.append(f"    Season: current={snwd.get('current', '?')}\"  "
                          f"peak={snwd.get('peak', '?')}\"  "
                          f"({snwd.get('days', '?')} days tracked)")
            if wteq:
                L.append(f"    SWE:    current={wteq.get('current', '?')}\"  "
                          f"peak={wteq.get('peak', '?')}\"")
            L.append("")

    # ── All SNOTEL ──
    L.append("-" * W)
    L.append("ALL SNOTEL STATIONS (current)")
    L.append("-" * W)
    for name, data in sorted(a.get("snotel_current", {}).items(),
                              key=lambda x: x[1].get("elev_ft", 0)):
        if "error" in data:
            L.append(f"  {name}: Error - {data['error']}")
        else:
            L.append(f"  {name:22s} ({data.get('elev_ft', '?')}ft): "
                      f"Depth={_f(data.get('snow_depth_in'), '\"'):>5s} | "
                      f"SWE={_f(data.get('swe_in'), '\"'):>6s} | "
                      f"Temp={_f(data.get('temp_f'), 'F'):>6s}")

    # ── Forecaster Discussion ──
    L.append("")
    L.append("-" * W)
    L.append("NWS FORECASTER DISCUSSION (Reno WFO)")
    L.append("-" * W)
    afd = a.get("forecaster_discussion", "Not available")
    if afd and afd != "Not available":
        for line in afd.split("\n"):
            L.append(f"  {line}")
    else:
        L.append("  Not available")

    # Footer
    L.append("")
    L.append("=" * W)
    L.append("Blend: Skill-weighted GFS+ECMWF+ICON+NWS grids+HRRR | Ensemble: GFS-31+ECMWF-51")
    L.append("Physics: Wet-bulb precip (Stull 2011), SLR (Roebber 2003), BMA, terrain, settling")
    L.append("=" * W)

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    use_hrrr = "--hrrr" in sys.argv
    json_out = "--json" in sys.argv
    compact = "--compact" in sys.argv

    print("Fetching data sources...")

    print("  [1/6] Current observations...")
    obs = fetch_nws_observations(39.1700, -120.1450)

    print("  [2/6] NWS forecast...")
    nws = fetch_nws_forecast(39.1700, -120.1450)

    print("  [3/6] Multi-model forecast (GFS/ECMWF/ICON, 16 days, per-resort)...")
    resort_points = {rn: {"lat": r["base"]["lat"], "lon": r["base"]["lon"]}
                     for rn, r in RESORTS.items()}
    om = fetch_open_meteo_multi(resort_points)

    print("  [4/6] SNOTEL snowpack (10 stations)...")
    snotel = fetch_snotel_current()

    print("  [5/6] Avalanche conditions + forecast discussion...")
    avy = fetch_avalanche()
    afd = fetch_forecast_discussion()

    hrrr = {}
    if use_hrrr:
        print("  [6/6] HRRR model data...")
        hrrr = fetch_hrrr(39.17, -120.145)
    else:
        print("  [6/6] Skipping HRRR (use --hrrr to enable)")

    print("  [+] NWS gridpoint data (forecaster-edited snow amounts)...")
    nws_grids = fetch_nws_gridpoints(38.9280, -119.9070)  # Heavenly peak

    print("  [+] Reno upper-air sounding (observed atmosphere profile)...")
    sounding = fetch_sounding("REV")

    print("  [+] Ensemble forecasts (GFS 31-member, ECMWF 51-member)...")
    ensemble = fetch_ensemble(38.93, -119.94)  # Heavenly area

    print("  [+] Synoptic/MesoWest stations (high-elevation obs)...")
    synoptic = fetch_synoptic_stations(39.17, -120.145, radius_miles=30)

    print("  [+] RWIS road weather stations (I-80, US-50)...")
    rwis = fetch_rwis_stations(39.17, -120.145)

    print("  [+] Radar nowcast (precipitation tracking)...")
    # For Tahoe, upstream is Sacramento Valley (~100mi west)
    radar_nowcast = fetch_radar_nowcast(39.17, -120.145,
                                         upstream_lat=38.58, upstream_lon=-121.49)

    print("  [+] CSSL enhanced (hourly snow, wind at Donner Summit)...")
    cssl = fetch_cssl_snow()

    print("\nAnalyzing conditions + building forecasts...")
    print("  Using skill-weighted multi-model blend (BMA-style)")
    print("  + terrain downscaling, snow settling, precip phase probability")
    print("  + lake effect enhancement for east-shore resorts")
    analysis = analyze_all(obs, nws, om, snotel, afd, avy, hrrr,
                           nws_grids=nws_grids, sounding=sounding,
                           ensemble=ensemble, synoptic=synoptic,
                           rwis=rwis, radar_nowcast=radar_nowcast,
                           cssl=cssl)

    if json_out:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(format_report(analysis, compact=compact))


if __name__ == "__main__":
    main()
