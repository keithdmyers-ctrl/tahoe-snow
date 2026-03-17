#!/usr/bin/env python3
"""
Sierra Outdoor Conditions Engine
=================================

Activity-specific conditions for: climbing, biking, hiking, fishing,
paddling, camping, cycling, photography, and backcountry skiing.

Built on the weather data already flowing through tahoe_snow.py,
plus targeted data from USGS stream gauges and Open-Meteo soil/AQI.

No ML models — this is physics, astronomy, and domain-expert decision
logic synthesized into actionable go/no-go recommendations.
"""

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# 1. Aspect-Solar Model
# =========================================================================

def compute_solar_exposure(lat: float, lon: float, aspect_deg: float,
                           slope_deg: float = 25.0,
                           date: datetime = None) -> dict:
    """
    Compute when a slope/wall with a given aspect is in sun vs shade.

    Uses simplified solar geometry: solar azimuth + altitude by hour,
    compared against slope normal vector.

    Args:
        lat: Latitude (degrees N)
        lon: Longitude (degrees W, negative)
        aspect_deg: Slope faces this direction (0=N, 90=E, 180=S, 270=W)
        slope_deg: Slope steepness (degrees from horizontal)
        date: Date to compute for (default: today)

    Returns:
        Dict with hourly sun/shade status and summary.
    """
    if date is None:
        date = datetime.now(timezone.utc)

    day_of_year = date.timetuple().tm_yday
    lat_rad = math.radians(lat)

    # Solar declination (Cooper equation)
    decl = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    decl_rad = math.radians(decl)

    # Compute solar position for each hour (local solar time)
    # Longitude correction: Tahoe is ~120W, Pacific timezone center is 120W
    hourly = []
    sun_start = None
    sun_end = None

    for hour in range(5, 21):  # 5 AM to 8 PM local
        # Hour angle (15 degrees per hour, solar noon = 0)
        hour_angle = math.radians(15 * (hour - 12))

        # Solar altitude (elevation angle above horizon)
        sin_alt = (math.sin(lat_rad) * math.sin(decl_rad) +
                   math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle))
        solar_alt_deg = math.degrees(math.asin(max(-1, min(1, sin_alt))))

        if solar_alt_deg <= 0:
            hourly.append({"hour": hour, "status": "dark", "solar_alt": 0})
            continue

        # Solar azimuth
        cos_az = ((math.sin(decl_rad) - math.sin(lat_rad) * sin_alt) /
                  (math.cos(lat_rad) * math.cos(math.asin(sin_alt))))
        cos_az = max(-1, min(1, cos_az))
        solar_az_deg = math.degrees(math.acos(cos_az))
        if hour_angle > 0:
            solar_az_deg = 360 - solar_az_deg

        # Is the slope in sun? Compare solar azimuth to slope aspect.
        # Slope gets sun when solar azimuth is within ~90° of slope's
        # facing direction AND sun is high enough to clear the slope.
        az_diff = abs(solar_az_deg - aspect_deg)
        if az_diff > 180:
            az_diff = 360 - az_diff

        # Sun hits this aspect when the azimuth difference < 90° + some
        # margin based on solar altitude
        in_sun = az_diff < (90 + solar_alt_deg * 0.5)

        # Steep north-facing slopes lose sun earlier
        if aspect_deg < 45 or aspect_deg > 315:  # N-facing
            in_sun = in_sun and solar_alt_deg > (slope_deg * 0.6)

        status = "sun" if in_sun else "shade"
        if in_sun:
            if sun_start is None:
                sun_start = hour
            sun_end = hour

        hourly.append({
            "hour": hour,
            "status": status,
            "solar_alt": round(solar_alt_deg, 1),
            "solar_az": round(solar_az_deg, 1),
        })

    # Summary
    sun_hours = sum(1 for h in hourly if h["status"] == "sun")
    month = date.month

    # Golden hour (photography)
    sunrise_h = next((h["hour"] for h in hourly if h["solar_alt"] > 0), 6)
    sunset_h = next((h["hour"] for h in reversed(hourly) if h["solar_alt"] > 0), 19)

    return {
        "aspect_deg": aspect_deg,
        "aspect_name": _deg_to_compass(aspect_deg),
        "sun_hours": sun_hours,
        "sun_start": f"{sun_start}:00" if sun_start else "No direct sun",
        "sun_end": f"{sun_end}:00" if sun_end else None,
        "golden_hour_am": f"{sunrise_h}:00-{sunrise_h+1}:00",
        "golden_hour_pm": f"{sunset_h-1}:00-{sunset_h}:00",
        "hourly": hourly,
        "summary": (
            f"{'S' if aspect_deg > 135 and aspect_deg < 225 else 'N' if (aspect_deg < 45 or aspect_deg > 315) else 'E' if aspect_deg < 135 else 'W'}-facing: "
            f"{sun_hours}h of sun, "
            f"{'in sun ' + (sun_start and f'{sun_start}:00-{sun_end}:00' or 'none') or 'shade all day'}"
        ),
        # Climbing-specific: wall temperature estimate
        "wall_temp_note": (
            "Wall heats up in sun — good friction in shade, greasy in afternoon sun"
            if sun_hours > 4 and month in [6, 7, 8]
            else "Cool wall — best friction conditions" if sun_hours < 3
            else "Mixed sun/shade"
        ),
    }


def _deg_to_compass(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((deg + 22.5) % 360 / 45)]


# =========================================================================
# 2. Trail Conditions (Soil Moisture / Mud / Dust / Ice)
# =========================================================================

def compute_trail_conditions(temp_f: float, temp_min_f: float,
                             precip_last_48h: float,
                             wind_mph: float = 0,
                             month: int = None) -> dict:
    """
    Estimate trail surface conditions from weather data.

    No soil moisture API needed — we can infer conditions from recent
    precipitation, temperature (freeze-thaw), and wind (dust).
    """
    if month is None:
        month = datetime.now(timezone.utc).month

    conditions = []
    warnings = []
    riding_quality = 5  # 1-5 scale for MTB

    # Freeze-thaw cycle: frozen overnight, thawing midday = worst mud
    if temp_min_f is not None and temp_f is not None:
        if temp_min_f < 28 and temp_f > 38:
            conditions.append("Freeze-thaw cycle active")
            warnings.append("Morning: icy/frozen. Afternoon: soft/muddy. Best 10AM-noon when thawed but firm.")
            riding_quality = min(riding_quality, 2)
        elif temp_min_f < 28 and temp_f < 34:
            conditions.append("Frozen ground")
            warnings.append("Trails frozen and firm — great for hiking, slippery on rock.")
            riding_quality = min(riding_quality, 3)

    # Recent precipitation
    if precip_last_48h > 0.5:
        conditions.append(f"Recent precip: {precip_last_48h:.1f}\" in 48h")
        if temp_f is not None and temp_f > 40:
            warnings.append("Trails likely muddy. Avoid to prevent trail damage.")
            riding_quality = min(riding_quality, 1)
        elif temp_f is not None and temp_f > 32:
            warnings.append("Wet trails. Expect mud in shaded sections.")
            riding_quality = min(riding_quality, 2)
        else:
            warnings.append("Snow on trails. Expect postholing above 7000ft.")
            riding_quality = min(riding_quality, 1)
    elif precip_last_48h > 0.1:
        conditions.append("Light recent precipitation")
        warnings.append("Some damp spots. Mostly rideable.")
        riding_quality = min(riding_quality, 4)

    # Dust (dry + windy)
    if precip_last_48h < 0.05 and month in [7, 8, 9]:
        conditions.append("Dry conditions")
        if wind_mph > 15:
            warnings.append("Dusty trails with wind. Eye protection recommended.")
            riding_quality = min(riding_quality, 3)
        else:
            conditions.append("Good dust — hero dirt conditions")
            riding_quality = min(riding_quality, 5)

    # Seasonal snow coverage estimate
    if month in [11, 12, 1, 2, 3]:
        snow_line_ft = (5500 + ((temp_f if temp_f is not None else 32) - 20) * 80) if temp_f is not None else 6500
        conditions.append(f"Estimated snow line: ~{snow_line_ft:.0f}ft")
        warnings.append(f"Expect snow above {snow_line_ft:.0f}ft. Trails above this impassable for bikes.")
    elif month in [4, 5]:
        conditions.append("Spring transition — variable snow coverage")
        warnings.append("Higher trails may still have snow. Check recent trip reports.")
    elif month == 6:
        conditions.append("Early summer — most trails snow-free below 8500ft")

    # No data
    if not conditions:
        conditions.append("Dry, firm trails expected")

    labels = {5: "Hero Dirt", 4: "Good", 3: "Fair", 2: "Poor (muddy/icy)", 1: "Closed/Impassable"}

    return {
        "conditions": conditions,
        "warnings": warnings,
        "mtb_quality": riding_quality,
        "mtb_label": labels.get(riding_quality, "Unknown"),
        "summary": f"Trail: {labels.get(riding_quality, '?')}. {'; '.join(warnings[:2]) if warnings else 'Good conditions.'}",
    }


# =========================================================================
# 3. Stream Flow (USGS Gauges)
# =========================================================================

# Key USGS gauges for Tahoe area fishing/paddling
TAHOE_STREAM_GAUGES = {
    "Truckee River at Tahoe City": {"id": "10337500", "activity": ["fish", "paddle"]},
    "Truckee River at Reno": {"id": "10348000", "activity": ["fish"]},
    "Upper Truckee River": {"id": "10336610", "activity": ["fish"]},
    "Blackwood Creek": {"id": "10336660", "activity": ["fish"]},
    "Ward Creek": {"id": "10336676", "activity": ["fish"]},
    "General Creek": {"id": "10336645", "activity": ["fish"]},
}


def fetch_stream_conditions() -> dict:
    """
    Fetch real-time stream flow data from USGS Water Services API.

    Returns flow rate (cfs), water temperature where available, and
    fishing/paddling condition assessment.
    """
    results = {}

    for name, info in TAHOE_STREAM_GAUGES.items():
        try:
            # USGS instantaneous values API
            resp = requests.get(
                "https://waterservices.usgs.gov/nwis/iv/",
                params={
                    "sites": info["id"],
                    "parameterCd": "00060,00010",  # 00060=discharge CFS, 00010=water temp C
                    "format": "json",
                    "siteStatus": "active",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            ts_list = data.get("value", {}).get("timeSeries", [])

            flow_cfs = None
            water_temp_c = None

            for ts in ts_list:
                param = ts.get("variable", {}).get("variableCode", [{}])[0].get("value", "")
                values = ts.get("values", [{}])[0].get("value", [])
                if values:
                    latest = values[-1].get("value")
                    if param == "00060" and latest:
                        flow_cfs = float(latest)
                    elif param == "00010" and latest:
                        water_temp_c = float(latest)

            if flow_cfs is not None:
                water_temp_f = round(water_temp_c * 9 / 5 + 32, 1) if water_temp_c else None

                # Fishing assessment
                fishing = _assess_fishing_flow(name, flow_cfs, water_temp_f)

                results[name] = {
                    "gauge_id": info["id"],
                    "flow_cfs": round(flow_cfs, 1),
                    "water_temp_f": water_temp_f,
                    "water_temp_c": round(water_temp_c, 1) if water_temp_c else None,
                    "fishing": fishing,
                    "activities": info["activity"],
                }

        except Exception as e:
            logger.debug("Failed to fetch USGS gauge %s: %s", name, e)

    return results


def _assess_fishing_flow(stream_name: str, flow_cfs: float,
                         water_temp_f: float = None) -> dict:
    """
    Assess fishing conditions based on flow and water temperature.

    Trout are most active at 50-65°F water temp, moderate flows.
    """
    conditions = []
    rating = 3  # 1-5

    # Flow assessment (general Tahoe streams)
    if flow_cfs > 500:
        conditions.append("High water — dangerous wading")
        rating = min(rating, 1)
    elif flow_cfs > 200:
        conditions.append("Elevated flow — fish in eddies and banks")
        rating = min(rating, 3)
    elif flow_cfs > 50:
        conditions.append("Good wadeable flow")
        rating = min(rating, 5)
    elif flow_cfs > 10:
        conditions.append("Low flow — fish concentrated in pools")
        rating = min(rating, 4)
    else:
        conditions.append("Very low flow — limited fishing")
        rating = min(rating, 2)

    # Temperature assessment
    if water_temp_f is not None:
        if water_temp_f > 68:
            conditions.append(f"Water {water_temp_f:.0f}°F — too warm, trout stressed. Fish early morning only.")
            rating = min(rating, 1)
        elif water_temp_f > 60:
            conditions.append(f"Water {water_temp_f:.0f}°F — warm, fish are active but handle carefully")
            rating = min(rating, 3)
        elif water_temp_f > 50:
            conditions.append(f"Water {water_temp_f:.0f}°F — ideal trout temperature")
            rating = min(rating, 5)
        elif water_temp_f > 40:
            conditions.append(f"Water {water_temp_f:.0f}°F — cool, slower fishing, nymphs best")
            rating = min(rating, 3)
        else:
            conditions.append(f"Water {water_temp_f:.0f}°F — cold, very slow fishing")
            rating = min(rating, 2)

    labels = {5: "Excellent", 4: "Good", 3: "Fair", 2: "Poor", 1: "Avoid"}

    return {
        "rating": rating,
        "label": labels.get(rating, "Unknown"),
        "conditions": conditions,
    }


# =========================================================================
# 4. Directional Wind by Route
# =========================================================================

# Known Sierra cycling/driving routes and their bearings
SIERRA_ROUTES = {
    "Kingsbury Grade (SR-207)": {"bearing": 270, "type": "climb", "elev_gain_ft": 2600},
    "Spooner Summit (US-50 E)": {"bearing": 90, "type": "climb", "elev_gain_ft": 1000},
    "Echo Summit (US-50 W)": {"bearing": 270, "type": "climb", "elev_gain_ft": 1500},
    "Donner Pass (I-80)": {"bearing": 270, "type": "drive", "elev_gain_ft": 2000},
    "Mt Rose Highway (SR-431)": {"bearing": 45, "type": "climb", "elev_gain_ft": 3000},
    "West Shore (SR-89)": {"bearing": 0, "type": "flat", "elev_gain_ft": 200},
    "East Shore Trail": {"bearing": 30, "type": "bike_path", "elev_gain_ft": 100},
    "Lake Tahoe Loop": {"bearing": None, "type": "loop", "elev_gain_ft": 500},  # All directions
}


def compute_route_wind(wind_dir_deg: float, wind_mph: float,
                       route_name: str = None) -> dict:
    """
    Compute headwind/tailwind/crosswind for known routes.

    Returns wind components for all routes or a specific one.
    """
    if np.isnan(wind_dir_deg) or np.isnan(wind_mph):
        return {"available": False}

    results = {}
    routes = {route_name: SIERRA_ROUTES[route_name]} if route_name and route_name in SIERRA_ROUTES else SIERRA_ROUTES

    for name, info in routes.items():
        bearing = info["bearing"]
        if bearing is None:
            # Loop route — average over all directions
            results[name] = {
                "headwind_mph": round(wind_mph * 0.5, 1),  # Average effect
                "crosswind_mph": round(wind_mph * 0.5, 1),
                "assessment": f"Variable: ~{wind_mph*0.5:.0f} mph average resistance",
            }
            continue

        # Angle between wind direction and route bearing
        # Wind FROM a direction means it opposes travel toward that direction
        angle = math.radians(wind_dir_deg - bearing)

        # Headwind component (positive = headwind, negative = tailwind)
        headwind = wind_mph * math.cos(angle)
        crosswind = abs(wind_mph * math.sin(angle))

        if headwind > 5:
            assessment = f"{headwind:.0f} mph headwind (uphill direction)"
        elif headwind < -5:
            assessment = f"{-headwind:.0f} mph tailwind (uphill direction)"
        else:
            assessment = "Minimal headwind/tailwind"

        if crosswind > 15:
            assessment += f" + {crosswind:.0f} mph crosswind — gusty!"

        results[name] = {
            "headwind_mph": round(headwind, 1),
            "tailwind_mph": round(-headwind, 1),
            "crosswind_mph": round(crosswind, 1),
            "wind_dir_deg": wind_dir_deg,
            "route_bearing": bearing,
            "route_type": info["type"],
            "assessment": assessment,
        }

    return {"available": True, "routes": results}


# =========================================================================
# 5. Pressure Trend for Fishing
# =========================================================================

def compute_fishing_pressure(pressure_hpa: float = None,
                             pressure_trend: float = None) -> dict:
    """
    Assess fishing conditions from barometric pressure.

    Fish behavior correlates with pressure changes:
    - Falling pressure (pre-storm): Fish feed aggressively (best fishing)
    - Steady high: Normal activity
    - Rising (post-storm): Initially slow, improving
    - Rapid drop: Fish go deep, very slow
    """
    if pressure_hpa is None and pressure_trend is None:
        return {"available": False, "detail": "No pressure data"}

    if pressure_trend is not None:
        if pressure_trend < -3:
            activity = "Excellent"
            detail = "Rapidly falling pressure — fish are feeding aggressively before the storm. Best fishing of the week."
            rating = 5
        elif pressure_trend < -1:
            activity = "Good"
            detail = "Slowly falling pressure — fish are active. Good window."
            rating = 4
        elif abs(pressure_trend) <= 1:
            activity = "Fair"
            detail = "Steady pressure — normal fish activity."
            rating = 3
        elif pressure_trend > 3:
            activity = "Poor"
            detail = "Rapidly rising pressure (post-storm) — fish sluggish. Try deeper water."
            rating = 2
        else:
            activity = "Fair-Good"
            detail = "Slowly rising — fish activity recovering."
            rating = 3
    else:
        # Just pressure, no trend
        if pressure_hpa and pressure_hpa < 1005:
            activity = "Good"
            detail = "Low pressure system — fish tend to be more active."
            rating = 4
        else:
            activity = "Fair"
            detail = "Normal pressure."
            rating = 3

    return {
        "available": True,
        "activity": activity,
        "rating": rating,
        "detail": detail,
        "pressure_hpa": pressure_hpa,
        "pressure_trend_hpa": pressure_trend,
        "best_technique": (
            "Topwater/streamers — aggressive fish" if rating >= 4
            else "Nymphs/subsurface — fish holding deeper" if rating <= 2
            else "Standard presentations"
        ),
    }


# =========================================================================
# 6. Smoke/AQI Forecast
# =========================================================================

def fetch_aqi_forecast(lat: float = 39.17, lon: float = -120.145) -> dict:
    """
    Fetch AQI forecast for next 48 hours from Open-Meteo Air Quality API.

    Returns hourly AQI forecast with smoke clearing prediction.
    """
    try:
        resp = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "us_aqi,pm2_5,pm10",
                "forecast_days": 3,
                "timezone": "America/Los_Angeles",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return {"available": False, "reason": f"HTTP {resp.status_code}"}

        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        aqi_values = hourly.get("us_aqi", [])
        pm25_values = hourly.get("pm2_5", [])

        if not aqi_values:
            return {"available": False, "reason": "No AQI data returned"}

        # Current
        current_aqi = aqi_values[0] if aqi_values else None

        # Find when AQI drops below thresholds
        good_hour = None  # AQI < 50
        moderate_hour = None  # AQI < 100
        for i, aqi in enumerate(aqi_values):
            if aqi is not None:
                if aqi < 100 and moderate_hour is None:
                    moderate_hour = i
                if aqi < 50 and good_hour is None:
                    good_hour = i

        # 24h and 48h outlook
        aqi_24h = aqi_values[:24]
        aqi_48h = aqi_values[:48]
        max_24h = max((v for v in aqi_24h if v is not None), default=0)
        max_48h = max((v for v in aqi_48h if v is not None), default=0)
        avg_24h = np.nanmean([v for v in aqi_24h if v is not None]) if aqi_24h else 0

        # Trend
        if len(aqi_values) >= 24:
            first_12 = np.nanmean([v for v in aqi_values[:12] if v is not None])
            last_12 = np.nanmean([v for v in aqi_values[12:24] if v is not None])
            if last_12 < first_12 * 0.7:
                trend = "Improving"
                detail = "Air quality expected to improve over the next 12-24 hours."
            elif last_12 > first_12 * 1.3:
                trend = "Worsening"
                detail = "Smoke expected to increase. Consider rescheduling outdoor plans."
            else:
                trend = "Steady"
                detail = "Air quality expected to remain similar."
        else:
            trend = "Unknown"
            detail = "Insufficient forecast data."

        clearing_msg = None
        if current_aqi and current_aqi > 100:
            if moderate_hour and moderate_hour < 24:
                clearing_msg = f"AQI drops below 100 in ~{moderate_hour}h"
            elif moderate_hour:
                clearing_msg = f"AQI drops below 100 in ~{moderate_hour}h (tomorrow)"
            else:
                clearing_msg = "No clearing expected in the next 48 hours"

        return {
            "available": True,
            "current_aqi": current_aqi,
            "max_24h": max_24h,
            "avg_24h": round(avg_24h),
            "trend": trend,
            "trend_detail": detail,
            "clearing": clearing_msg,
            "hours_to_good": good_hour,
            "hours_to_moderate": moderate_hour,
        }

    except Exception as e:
        logger.debug("AQI forecast fetch failed: %s", e)
        return {"available": False, "reason": str(e)}


# =========================================================================
# Master Outdoor Conditions Builder
# =========================================================================

def compute_all_outdoor_conditions(analysis: dict) -> dict:
    """
    Compute all outdoor activity conditions from the analysis dict.

    This is the main entry point — call after analyze_all() to add
    outdoor conditions to the analysis.

    Returns a dict with keys for each activity type.
    """
    conditions = analysis.get("current_conditions", {})
    obs = conditions.get("observation", {})
    resorts = analysis.get("resorts", {})
    now = datetime.now(timezone.utc)
    month = now.month

    # Get representative weather from first available zone
    temp_f = None
    temp_min_f = None
    wind_mph = 0
    wind_dir = 0
    gusts = 0
    precip_48h = 0
    pressure = None
    pressure_trend = None

    for rd in resorts.values():
        for zd in rd.get("zones", {}).values():
            cur = zd.get("current", {})
            if cur.get("temp_f") is not None:
                temp_f = temp_f or cur.get("temp_f")
                wind_mph = wind_mph or cur.get("wind_mph", 0) or 0
                wind_dir = wind_dir or cur.get("wind_dir")
                gusts = gusts or cur.get("wind_gust_mph", 0) or 0
                pressure = pressure or cur.get("pressure_hpa")
            tl = zd.get("timeline_48h", [])
            if tl:
                precip_48h = max(precip_48h, sum(h.get("precip_in", 0) or 0 for h in tl[:48]))
                # Estimate overnight low from timeline (lowest temp in next 24h)
                temps = [h.get("temp_f") or h.get("temperature_2m") for h in tl[:24] if h.get("temp_f") or h.get("temperature_2m")]
                if temps and temp_min_f is None:
                    temp_min_f = min(temps)
            # Keep searching until we have all conditions
            if temp_f is not None and wind_mph:
                break
        if temp_f is not None and wind_mph:
            break

    # Convert wind direction string to degrees
    if isinstance(wind_dir, str):
        dir_map = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
        wind_dir = dir_map.get(wind_dir, 0)

    result = {}

    # 1. Solar exposure for key aspects
    result["solar"] = {}
    for aspect, label in [(0, "North"), (90, "East"), (180, "South"), (270, "West")]:
        result["solar"][label] = compute_solar_exposure(
            39.1, -120.1, aspect, slope_deg=30, date=now
        )

    # 2. Trail conditions
    result["trail"] = compute_trail_conditions(
        temp_f or 55, temp_min_f, precip_48h, wind_mph, month
    )

    # 3. Stream flow (network call — may fail)
    result["streams"] = fetch_stream_conditions()

    # 4. Route wind
    result["route_wind"] = compute_route_wind(
        wind_dir or 0, wind_mph or 0
    )

    # 5. Fishing pressure
    result["fishing_pressure"] = compute_fishing_pressure(
        pressure, pressure_trend
    )

    # 6. AQI forecast (network call — may fail)
    result["aqi_forecast"] = fetch_aqi_forecast()

    return result


# =========================================================================
# Activity Decision Cards (UX layer)
# =========================================================================

def _apply_aqi_override(signal: str, score: int, aqi_val,
                        threshold_caution: int = 100,
                        threshold_nogo: int = 150) -> tuple:
    """Apply consistent AQI overrides to any aerobic activity.

    Thresholds are configurable because high-exertion activities (cycling,
    MTB) justify a lower caution threshold than low-exertion ones (photo).

    Returns (signal, score) after applying overrides.
    """
    if aqi_val is None:
        return signal, score
    if aqi_val > threshold_nogo:
        return "no-go", min(score, 2)
    if aqi_val > threshold_caution:
        return ("caution" if signal == "go" else signal), min(score, 4)
    return signal, score


def _ensure_timing(timing_list: list, fallback: str = "Any time — conditions are ideal") -> list:
    """Guarantee timing is a non-empty list of time-window strings."""
    if timing_list and any(t.strip() for t in timing_list if isinstance(t, str)):
        return [t for t in timing_list if isinstance(t, str) and t.strip()]
    return [fallback]


def compute_activity_decisions(analysis: dict) -> list:
    """
    Produce a standardized decision card for each outdoor activity.

    Each card has:
      - activity: name
      - icon: emoji
      - signal: "go" | "caution" | "no-go"
      - score: 0-10 (calibrated: 0-2 no-go, 3-4 poor, 5-6 caution,
                      7-8 good, 9-10 excellent)
      - headline: one-line summary
      - metrics: list of {label, value, status?} (3-5 key numbers)
      - timing: list of time-window strings ("when to go")
      - warnings: list of advisory strings (separate from timing)
      - detail: expanded explanation
      - season: "winter" | "summer" | "year-round" (for frontend filtering)

    Cards are sorted by score descending (best activities first).
    """
    resorts = analysis.get("resorts", {})
    ml = analysis  # ML corrections are already in the analysis dict
    outdoor = analysis.get("outdoor", {})
    now = datetime.now(timezone.utc)
    month = now.month

    # Gather representative conditions
    temp_f, wind_mph, gusts, precip_24h = _gather_conditions(resorts)
    aqi = analysis.get("current_conditions", {}).get("observation", {}).get("aqi", {})
    aqi_val = aqi.get("us_aqi") if isinstance(aqi, dict) else None

    cards = []

    # --- SKI ---
    pa = ml.get("ml_powder_alert", {})
    ranking = ml.get("ml_resort_ranking", [])
    top_resort = ranking[0] if ranking else {}
    ski_level = pa.get("level", 0)
    ski_signal = "go" if ski_level >= 3 else ("caution" if ski_level >= 2 else "no-go")
    # Score: level 5 -> 10, level 4 -> 8, level 3 -> 7, level 2 -> 5, level 1 -> 3, level 0 -> 1
    ski_score = max(1, min(10, ski_level * 2)) if ski_level > 0 else 1
    off_season = month in [6, 7, 8, 9]
    if off_season and ski_level < 2:
        ski_signal = "no-go"
        ski_score = 0  # Hide-worthy in summer

    cards.append({
        "activity": "Ski",
        "icon": "⛷️",
        "signal": ski_signal,
        "score": ski_score,
        "headline": pa.get("message", "Check conditions"),
        "metrics": [
            {"label": "Alert", "value": pa.get("label", "—"),
             "status": "good" if ski_level >= 3 else ("warning" if ski_level >= 2 else "danger")},
            {"label": "Best Resort", "value": top_resort.get("resort", "—")},
            {"label": "Quality", "value": f"{pa.get('best_quality_rating', 0)}/5",
             "status": "good" if pa.get("best_quality_rating", 0) >= 4 else "warning"},
        ],
        "timing": _ensure_timing([], "Check resort opening hours"),
        "warnings": top_resort.get("reasons", []),
        "detail": pa.get("message", ""),
        "season": "winter",
    })

    # --- BACKCOUNTRY SKI ---
    avy = analysis.get("avalanche", {})
    avy_danger = avy.get("danger_level", 0) if not avy.get("error") else None
    avy_label = avy.get("danger_label", "Unknown")
    snotel = analysis.get("snotel_current", {})
    # Average SWE across stations as snowpack proxy
    swe_values = [s.get("swe_in", 0) for s in snotel.values()
                  if isinstance(s, dict) and s.get("swe_in") is not None]
    avg_swe = sum(swe_values) / len(swe_values) if swe_values else 0

    bc_score = 5  # Default: caution
    bc_signal = "caution"
    bc_warnings = []
    bc_timing = []

    if avy_danger is not None:
        if avy_danger >= 4:  # High or Extreme
            bc_score = 1
            bc_signal = "no-go"
            bc_warnings.append(f"Avalanche danger: {avy_label} — avoid backcountry")
        elif avy_danger == 3:  # Considerable
            bc_score = 3
            bc_signal = "caution"
            bc_warnings.append(f"Avalanche danger: {avy_label} — careful route selection required")
            bc_timing.append("Avoid steep (>30 deg) windloaded aspects")
        elif avy_danger == 2:  # Moderate
            bc_score = 7
            bc_signal = "go"
            bc_warnings.append(f"Avalanche danger: {avy_label} — normal caution")
            bc_timing.append("Best: early morning before solar warming")
        elif avy_danger <= 1:  # Low or No Rating
            bc_score = 8
            bc_signal = "go"
            bc_timing.append("Any time — low avalanche hazard")
    else:
        bc_warnings.append("No avalanche forecast available — check sierraavalanchecenter.org")

    if off_season and avg_swe < 5:
        bc_score = 0
        bc_signal = "no-go"
        bc_warnings.append("No snowpack for backcountry travel")

    # Snow quality from powder alert
    snow_quality = pa.get("best_quality_rating", 0)
    if snow_quality >= 4:
        bc_score = min(10, bc_score + 1)

    cards.append({
        "activity": "Backcountry Ski",
        "icon": "🎿",
        "signal": bc_signal,
        "score": bc_score,
        "headline": bc_warnings[0] if bc_warnings else "Check avalanche forecast",
        "metrics": [
            {"label": "Avy Danger", "value": avy_label,
             "status": "danger" if (avy_danger or 0) >= 3 else ("warning" if (avy_danger or 0) >= 2 else "good")},
            {"label": "Snowpack", "value": f"{avg_swe:.0f}\" SWE" if avg_swe else "—",
             "status": "good" if avg_swe > 20 else ("warning" if avg_swe > 5 else "danger")},
            {"label": "Quality", "value": f"{snow_quality}/5" if snow_quality else "—"},
        ],
        "timing": _ensure_timing(bc_timing, "Check sierraavalanchecenter.org"),
        "warnings": bc_warnings,
        "detail": avy.get("travel_advice", ""),
        "season": "winter",
    })

    # --- HIKE ---
    outdoor_data = ml.get("ml_outdoor", {})
    thunder = outdoor_data.get("thunderstorm", {})
    heat = outdoor_data.get("heat_safety", {})
    smoke = outdoor_data.get("smoke_aqi", {})
    rec = outdoor_data.get("recommendation", {})
    hike_window = outdoor_data.get("best_window", {})

    hike_signal = {"GO": "go", "GO (with awareness)": "caution",
                   "CAUTION": "caution", "NO-GO": "no-go"}.get(rec.get("verdict", ""), "caution")
    hike_score = {"go": 8, "caution": 5, "no-go": 2}.get(hike_signal, 5)
    # Perfect score only if signal is already "go" AND no hazards
    if (hike_signal == "go"
            and thunder.get("risk") == "Very Low"
            and heat.get("risk") in ["Low", "None"]
            and (aqi_val is None or aqi_val < 50)
            and precip_24h < 0.1):
        hike_score = 10

    # Apply consistent AQI override (hike = aerobic, 100/150 thresholds)
    hike_signal, hike_score = _apply_aqi_override(hike_signal, hike_score, aqi_val, 100, 150)

    hike_warnings = []
    turn_around = thunder.get("turn_around_time")
    if turn_around:
        hike_warnings.append(f"Be below treeline by {turn_around}")
    if heat.get("risk") in ["High", "Extreme"]:
        hike_warnings.append(heat.get("water_recommendation", "Carry extra water"))

    cards.append({
        "activity": "Hike",
        "icon": "🥾",
        "signal": hike_signal,
        "score": hike_score,
        "headline": rec.get("summary", "Check conditions"),
        "metrics": [
            {"label": "Storms", "value": thunder.get("risk", "?"),
             "status": "good" if thunder.get("risk") in ["Very Low", "Low"] else "warning" if thunder.get("risk") == "Moderate" else "danger"},
            {"label": "AQI", "value": str(aqi_val or "—"),
             "status": "good" if (aqi_val or 0) < 50 else "warning" if (aqi_val or 0) < 100 else "danger"},
            {"label": "Heat", "value": heat.get("risk", "?"),
             "status": "good" if heat.get("risk") in ["Low", "None"] else "warning" if heat.get("risk") == "Moderate" else "danger"},
            {"label": "UV", "value": outdoor_data.get("uv_exposure", {}).get("risk", "?")},
        ],
        "timing": _ensure_timing(hike_window.get("best_windows", [])),
        "warnings": hike_warnings,
        "detail": hike_window.get("advice", ""),
        "season": "year-round",
    })

    # --- MTB ---
    trail = outdoor.get("trail", {})
    mtb_quality = trail.get("mtb_quality", 3)
    mtb_signal = "go" if mtb_quality >= 4 else ("caution" if mtb_quality >= 3 else "no-go")
    mtb_score = mtb_quality * 2
    # Apply consistent AQI override (MTB = high exertion, 100/150)
    mtb_signal, mtb_score = _apply_aqi_override(mtb_signal, mtb_score, aqi_val, 100, 150)

    mtb_timing = []
    if mtb_quality <= 2 and temp_f and temp_f > 32:
        mtb_timing.append("Wait 24-48h for trails to dry")
    elif mtb_quality >= 4:
        mtb_timing.append("Any time — trails are prime")

    cards.append({
        "activity": "Mountain Bike",
        "icon": "🚵",
        "signal": mtb_signal,
        "score": mtb_score,
        "headline": trail.get("summary", "Check trail conditions"),
        "metrics": [
            {"label": "Trail", "value": trail.get("mtb_label", "?"),
             "status": "good" if mtb_quality >= 4 else "warning" if mtb_quality >= 3 else "danger"},
            {"label": "AQI", "value": str(aqi_val or "—"),
             "status": "good" if (aqi_val or 0) < 50 else "warning" if (aqi_val or 0) < 100 else "danger"},
            {"label": "Temp", "value": f"{temp_f:.0f}°F" if temp_f else "—"},
        ],
        "timing": _ensure_timing(mtb_timing),
        "warnings": trail.get("warnings", [])[:3],
        "detail": "; ".join(trail.get("conditions", [])),
        "season": "summer",
    })

    # --- FISH ---
    fp = outdoor.get("fishing_pressure", {})
    streams = outdoor.get("streams", {})
    fish_rating = fp.get("rating", 3) if fp.get("available") else 3
    fish_signal = "go" if fish_rating >= 4 else ("caution" if fish_rating >= 3 else "no-go")
    fish_score = fish_rating * 2
    best_stream = None
    best_stream_detail = None
    for sname, sdata in streams.items():
        fish_info = sdata.get("fishing", {})
        if fish_info.get("rating", 0) >= 4:
            best_stream = sname
            best_stream_detail = fish_info.get("conditions", [])
            break

    fish_timing = []
    if fish_rating >= 4:
        fish_timing.append("Now — fish are feeding actively")
    elif fp.get("pressure_trend_hpa") and fp.get("pressure_trend_hpa") < -1:
        fish_timing.append("Next 6-12h — fish feed aggressively before storms")
    else:
        fish_timing.append("Early morning or late evening")

    cards.append({
        "activity": "Fish",
        "icon": "🎣",
        "signal": fish_signal,
        "score": fish_score,
        "headline": fp.get("detail", "Check conditions"),
        "metrics": [
            {"label": "Pressure", "value": fp.get("activity", "?"),
             "status": "good" if fish_rating >= 4 else "warning" if fish_rating >= 3 else "danger"},
            {"label": "Technique", "value": fp.get("best_technique", "—")},
            {"label": "Best Stream", "value": best_stream or "—"},
        ],
        "timing": _ensure_timing(fish_timing),
        "warnings": (best_stream_detail or [])[:2],
        "detail": fp.get("detail", ""),
        "season": "year-round",
    })

    # --- CLIMB ---
    solar = outdoor.get("solar", {})
    south_wall = solar.get("South", {})
    north_wall = solar.get("North", {})
    climb_score = 7
    climb_signal = "go"
    climb_warnings = []
    climb_timing = []
    if precip_24h > 0.1:
        climb_score = 3
        climb_signal = "caution"
        climb_warnings.append("Recent rain — rock may be wet/seeping")
        climb_timing.append("Wait 24-48h for rock to dry")
    if wind_mph and wind_mph > 25:
        climb_score = min(climb_score, 4)
        climb_signal = "caution"
        climb_warnings.append(f"Windy: {wind_mph:.0f} mph")
    if temp_f and temp_f > 90:
        climb_warnings.append("Hot — south/west faces will be greasy")
        climb_timing.append("Climb north-facing in shade, or go early/late")
    elif temp_f and temp_f < 40:
        climb_warnings.append(f"Cold ({temp_f:.0f}°F) — fingers will be numb")
    if not climb_warnings:
        climb_warnings.append("Good climbing conditions")
    if not climb_timing:
        sun_note = south_wall.get("wall_temp_note", "")
        if "shade" in sun_note.lower():
            climb_timing.append("Shade walls best — cool friction")
        else:
            climb_timing.append("Any time — good friction conditions")

    cards.append({
        "activity": "Climb",
        "icon": "🧗",
        "signal": climb_signal,
        "score": climb_score,
        "headline": climb_warnings[0],
        "metrics": [
            {"label": "Temp", "value": f"{temp_f:.0f}°F" if temp_f else "—",
             "status": "good" if temp_f and 45 <= temp_f <= 80 else "warning"},
            {"label": "Wind", "value": f"{wind_mph:.0f} mph" if wind_mph else "—",
             "status": "good" if (wind_mph or 0) < 15 else "warning" if (wind_mph or 0) < 25 else "danger"},
            {"label": "S Face", "value": f"{south_wall.get('sun_hours', '?')}h sun"},
            {"label": "N Face", "value": f"{north_wall.get('sun_hours', '?')}h sun"},
        ],
        "timing": _ensure_timing(climb_timing),
        "warnings": climb_warnings,
        "detail": "; ".join(climb_warnings),
        "season": "year-round",
    })

    # --- PADDLE ---
    paddle_score = 7
    paddle_signal = "go"
    paddle_warnings = []
    if wind_mph and wind_mph > 15:
        paddle_score = max(2, 7 - int(wind_mph / 5))
        paddle_signal = "caution" if wind_mph < 25 else "no-go"
        paddle_warnings.append(f"Wind {wind_mph:.0f} mph — choppy conditions")
    if gusts and gusts > 25:
        paddle_warnings.append(f"Gusts to {gusts:.0f} mph — stay near shore")
    if temp_f and temp_f < 50:
        paddle_score = min(paddle_score, 4)
        paddle_warnings.append(f"Cold ({temp_f:.0f}°F) — wear layers, cold water immersion risk")
    # AQI: paddling is lower exertion than cycling, higher threshold
    paddle_signal, paddle_score = _apply_aqi_override(paddle_signal, paddle_score, aqi_val, 150, 200)

    cards.append({
        "activity": "Paddle",
        "icon": "🛶",
        "signal": paddle_signal,
        "score": paddle_score,
        "headline": f"{'Calm water' if (wind_mph or 0) < 10 else f'Wind {wind_mph:.0f} mph — choppy'}" if wind_mph is not None else "Check wind",
        "metrics": [
            {"label": "Wind", "value": f"{wind_mph:.0f} mph" if wind_mph else "—",
             "status": "good" if (wind_mph or 0) < 10 else "warning" if (wind_mph or 0) < 20 else "danger"},
            {"label": "Gusts", "value": f"{gusts:.0f} mph" if gusts else "—"},
            {"label": "Temp", "value": f"{temp_f:.0f}°F" if temp_f else "—"},
        ],
        "timing": _ensure_timing(["Before 11 AM — Tahoe thermals build 15-20 mph SW wind by 2 PM"]),
        "warnings": paddle_warnings,
        "detail": "Lake Tahoe afternoon thermals typically build 15-20 mph SW wind by 2 PM.",
        "season": "summer",
    })

    # --- ROAD CYCLE ---
    route_wind = outdoor.get("route_wind", {})
    cycle_routes = route_wind.get("routes", {})
    best_route = None
    best_tailwind = -999
    for rname, rinfo in cycle_routes.items():
        if rinfo.get("route_type") in ["climb", "flat", "bike_path"]:
            tw = rinfo.get("tailwind_mph", 0)
            if tw > best_tailwind:
                best_tailwind = tw
                best_route = rname

    cycle_score = 7
    cycle_signal = "go"
    cycle_warnings = []
    # AQI: cycling is high exertion, lower threshold (100/150)
    cycle_signal, cycle_score = _apply_aqi_override(cycle_signal, cycle_score, aqi_val, 100, 150)
    if aqi_val and aqi_val > 100:
        cycle_warnings.append(f"Poor air quality (AQI {aqi_val}) — high breathing rate amplifies exposure")
    if temp_f and temp_f > 95:
        cycle_score = min(cycle_score, 3)
        cycle_signal = "caution" if cycle_signal == "go" else cycle_signal
        cycle_warnings.append(f"Hot ({temp_f:.0f}°F) — risk of heat illness on climbs")

    cycle_timing = []
    if best_tailwind > 5:
        cycle_timing.append(f"Ride {best_route} — {best_tailwind:.0f} mph tailwind on the climb")
    else:
        cycle_timing.append("Early AM (calm winds) or match wind direction")

    cards.append({
        "activity": "Road Cycle",
        "icon": "🚴",
        "signal": cycle_signal,
        "score": cycle_score,
        "headline": f"Best route: {best_route}" if best_route else "Check wind and routes",
        "metrics": [
            {"label": "Best Route", "value": best_route or "—"},
            {"label": "Tailwind", "value": f"{best_tailwind:.0f} mph" if best_tailwind > 0 else "—",
             "status": "good" if best_tailwind > 5 else "warning"},
            {"label": "AQI", "value": str(aqi_val or "—"),
             "status": "good" if (aqi_val or 0) < 50 else "warning" if (aqi_val or 0) < 100 else "danger"},
        ],
        "timing": _ensure_timing(cycle_timing),
        "warnings": cycle_warnings,
        "detail": "; ".join(rinfo.get("assessment", "") for rinfo in cycle_routes.values() if rinfo.get("route_type") == "climb")[:200],
        "season": "year-round",
    })

    # --- CAMP ---
    water = outdoor_data.get("water_availability", {})
    camp_score = 7
    camp_signal = "go"
    camp_warnings = []
    camp_timing = []
    if aqi_val and aqi_val > 200:
        camp_score = min(camp_score, 2)
        camp_signal = "no-go"
        camp_warnings.append(f"Hazardous air quality (AQI {aqi_val}) — do not camp")
    elif aqi_val and aqi_val > 150:
        camp_score = min(camp_score, 4)
        camp_signal = "caution"
        camp_warnings.append(f"Unhealthy air (AQI {aqi_val}) — limit outdoor time")
    elif aqi_val and aqi_val > 100:
        camp_score = min(camp_score, 5)
        camp_warnings.append(f"Smoky (AQI {aqi_val}) — sensitive groups affected")
    if precip_24h > 0.3:
        camp_warnings.append("Rain expected — waterproof your setup")
        camp_timing.append("Set up camp before afternoon showers")
    if temp_f and temp_f < 35:
        camp_warnings.append(f"Cold: {temp_f:.0f}°F — 20°F bag recommended")
    if not camp_warnings:
        camp_warnings.append("Great camping weather")
    if not camp_timing:
        camp_timing.append("Any time — weather looks good")

    cards.append({
        "activity": "Camp",
        "icon": "⛺",
        "signal": camp_signal,
        "score": camp_score,
        "headline": camp_warnings[0] if camp_warnings else "Great camping weather",
        "metrics": [
            {"label": "Low Temp", "value": f"{(temp_f or 50) - 15:.0f}°F" if temp_f else "—",
             "status": "good" if (temp_f or 50) - 15 > 40 else "warning" if (temp_f or 50) - 15 > 25 else "danger"},
            {"label": "AQI", "value": str(aqi_val or "—"),
             "status": "good" if (aqi_val or 0) < 50 else "warning" if (aqi_val or 0) < 100 else "danger"},
            {"label": "Water", "value": water.get("status", "?")},
        ],
        "timing": _ensure_timing(camp_timing),
        "warnings": camp_warnings,
        "detail": water.get("detail", ""),
        "season": "year-round",
    })

    # --- PHOTO ---
    photo_score = 6
    photo_signal = "go"
    south_solar = solar.get("South", {})
    golden_am = south_solar.get("golden_hour_am", "?")
    golden_pm = south_solar.get("golden_hour_pm", "?")
    cloud_note = "Clear skies" if (precip_24h or 0) < 0.01 else "Clouds — dramatic light possible"
    photo_warnings = []
    if aqi_val and aqi_val > 50:
        cloud_note = f"Hazy (AQI {aqi_val}) — warm tones, reduced contrast"
        if aqi_val > 100:
            photo_warnings.append("Heavy haze may obscure mountain views")
        if aqi_val > 150:
            photo_score = min(photo_score, 3)
            photo_signal = "caution"
    if precip_24h > 0.01:
        photo_score = min(10, photo_score + 2)  # Clouds boost photo score
        photo_warnings.append("Clouds can create dramatic light — stay ready")

    cards.append({
        "activity": "Photo",
        "icon": "📷",
        "signal": photo_signal,
        "score": photo_score,
        "headline": cloud_note,
        "metrics": [
            {"label": "Golden AM", "value": golden_am},
            {"label": "Golden PM", "value": golden_pm},
            {"label": "Sky", "value": cloud_note[:20]},
        ],
        "timing": _ensure_timing([f"Golden hour: {golden_am} and {golden_pm}"]),
        "warnings": photo_warnings,
        "detail": "Best light at golden hour. Smoke can create dramatic sunsets.",
        "season": "year-round",
    })

    # Sort by score descending (green activities first)
    cards.sort(key=lambda c: c["score"], reverse=True)

    # Assign rank
    for i, card in enumerate(cards):
        card["rank"] = i + 1

    return cards


def _gather_conditions(resorts: dict):
    """Extract representative weather from resort zones.

    Iterates through all resorts and zones until valid data is found,
    rather than only checking the first zone of the first resort.
    """
    temp_f = None
    wind_mph = None
    gusts = None
    precip_24h = 0

    for rd in resorts.values():
        for zd in rd.get("zones", {}).values():
            cur = zd.get("current", {})
            if cur.get("temp_f") is not None:
                temp_f = temp_f or cur.get("temp_f")
                wind_mph = wind_mph or cur.get("wind_mph", 0)
                gusts = gusts or cur.get("wind_gust_mph", 0)
            tl = zd.get("timeline_48h", [])
            if tl:
                precip_24h = max(precip_24h, sum(h.get("precip_in", 0) or 0 for h in tl[:24]))
            # Stop once we have all basic conditions
            if temp_f is not None and wind_mph is not None:
                break
        if temp_f is not None and wind_mph is not None:
            break

    return temp_f, wind_mph, gusts, precip_24h
