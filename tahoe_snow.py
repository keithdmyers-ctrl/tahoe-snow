#!/usr/bin/env python3
"""
Tahoe Snow Conditions Analyzer — Full Feature Build

Features:
  P0: Multi-day forecast (15-day), hourly timeline (48h), powder alerts
  P1: Day/night accumulation, multi-model comparison, historical snow, wind chill
  P2: AI-style summary, avalanche conditions

Data sources (all free, public, no API keys):
  - Open-Meteo (GFS, ECMWF, ICON models — 15-day hourly forecasts)
  - NWS API (current observations, forecast, hourly forecast)
  - SNOTEL/NRCS (snowpack depth, SWE, temperature — current + historical)
  - Avalanche.org (Sierra Avalanche Center danger ratings)
  - NWS Reno WFO (Area Forecast Discussion)
  - HRRR via Herbie (optional, --hrrr flag)

Usage:
  python tahoe_snow.py              # full report
  python tahoe_snow.py --json       # JSON output
  python tahoe_snow.py --hrrr       # include HRRR model data
  python tahoe_snow.py --compact    # shorter report (no hourly timeline)
"""

import json
import sys
import math
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Resort Presets
# ---------------------------------------------------------------------------

RESORTS = {
    "Heavenly": {
        "base": {"label": "California Lodge", "lat": 38.9353, "lon": -119.9406, "elev_ft": 6540},
        "mid":  {"label": "Sky Deck / Tamarack", "lat": 38.9310, "lon": -119.9250, "elev_ft": 8500},
        "peak": {"label": "Monument Peak", "lat": 38.9280, "lon": -119.9070, "elev_ft": 10067},
        "aspect": "NE",
        "nearest_snotel": ["Fallen Leaf", "Hagan's Meadow"],
    },
    "Northstar": {
        "base": {"label": "Village", "lat": 39.2744, "lon": -120.1210, "elev_ft": 6330},
        "mid":  {"label": "Vista Express", "lat": 39.2680, "lon": -120.1150, "elev_ft": 7600},
        "peak": {"label": "Mt Pluto", "lat": 39.2630, "lon": -120.1100, "elev_ft": 8610},
        "aspect": "SW",
        "nearest_snotel": ["Independence Lake", "Independence Camp", "Tahoe City Cross"],
    },
    "Kirkwood": {
        "base": {"label": "Lodge", "lat": 38.6850, "lon": -120.0650, "elev_ft": 7800},
        "mid":  {"label": "Sunrise / Solitude", "lat": 38.6820, "lon": -120.0720, "elev_ft": 8800},
        "peak": {"label": "Thimble Peak", "lat": 38.6790, "lon": -120.0780, "elev_ft": 9800},
        "aspect": "N",
        "nearest_snotel": ["CSS Lab", "Hagan's Meadow"],
    },
}

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

MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
MODEL_LABELS = {"gfs_seamless": "GFS", "ecmwf_ifs025": "ECMWF", "icon_seamless": "ICON"}

# Water year starts Oct 1
def water_year_start() -> str:
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-10-01"


# ---------------------------------------------------------------------------
# Snow Physics
# ---------------------------------------------------------------------------

def compute_slr(temp_c: float) -> float:
    """Temperature-dependent snow-to-liquid ratio (Roebber 2003)."""
    if temp_c <= -18:
        return 20.0 + min(5.0, (-18 - temp_c) * 0.5)
    elif temp_c <= -12:
        return 15.0 + (-12 - temp_c) * (5.0 / 6.0)
    elif temp_c <= -6:
        return 12.0 + (-6 - temp_c) * (3.0 / 6.0)
    elif temp_c <= -1:
        return 8.0 + (-1 - temp_c) * (4.0 / 5.0)
    elif temp_c <= 0:
        return 5.0 + (0 - temp_c) * 3.0
    else:
        return max(1.0, 5.0 - temp_c * 2.0)


def wind_chill_f(temp_f: float, wind_mph: float) -> float:
    """NWS wind chill formula. Valid for T<=50F, V>=3mph."""
    if temp_f > 50 or wind_mph < 3:
        return temp_f
    wc = (35.74 + 0.6215 * temp_f
          - 35.75 * (wind_mph ** 0.16)
          + 0.4275 * temp_f * (wind_mph ** 0.16))
    return round(wc, 1)


def orographic_multiplier(elev_ft: float, wind_mph: float, wind_dir: float) -> float:
    ideal = 247.5
    diff = abs(wind_dir - ideal)
    if diff > 180:
        diff = 360 - diff
    d = max(0.3, 1.0 - (diff / 180.0) * 0.7)
    e = 1.0 + max(0.0, (elev_ft - 6225) / (10000 - 6225)) * 0.5
    w = 1.0 + min(0.3, wind_mph / 100.0)
    return d * e * w


def estimate_temp_c(base_c: float, base_m: float, target_m: float,
                    saturated: bool = True) -> float:
    # Moist adiabatic: ~5.5C/km (during precip). Standard: ~6.5C/km (dry/avg).
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


def precip_type(temp_c: float, has_precip: bool) -> str:
    if not has_precip:
        return "None"
    if temp_c <= -2:
        return "Snow"
    elif temp_c <= 1:
        return "Mix"
    else:
        return "Rain"


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
        obs_url = resp.json()["properties"].get("observationStations", "")
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
    except Exception:
        return {}


def fetch_nws_forecast(lat: float, lon: float) -> dict:
    """7-day periods + 48h hourly from NWS."""
    headers = {"User-Agent": "TahoeSnowStation/1.0 (keith@local)"}
    try:
        resp = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
                            headers=headers, timeout=10)
        if resp.status_code != 200:
            return {}
        props = resp.json()["properties"]
        result = {}
        fr = requests.get(props["forecast"], headers=headers, timeout=10)
        if fr.status_code == 200:
            result["periods"] = fr.json()["properties"]["periods"]
        hr = requests.get(props["forecastHourly"], headers=headers, timeout=10)
        if hr.status_code == 200:
            result["hourly"] = hr.json()["properties"]["periods"][:156]  # up to 6.5 days
        return result
    except Exception:
        return {}


def fetch_open_meteo(lat: float, lon: float) -> dict:
    """Multi-model 16-day hourly forecast from Open-Meteo (GFS, ECMWF, ICON)."""
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join([
                "temperature_2m", "precipitation", "snowfall", "snow_depth",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            ]),
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
            ]),
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
                return {names[i]: data[i] for i in range(len(names))}
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
    except Exception:
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
    except Exception:
        return {}


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
    except Exception:
        return ""


def fetch_hrrr(lat: float, lon: float) -> dict:
    """Optional HRRR model data via Herbie."""
    try:
        from herbie import Herbie
        now = datetime.now(timezone.utc)
        mt = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
        results = {"model_run": mt.isoformat(), "forecasts": []}
        for fxx in range(1, 19):
            try:
                H = Herbie(mt.strftime("%Y-%m-%d %H:00"), model="hrrr",
                           product="sfc", fxx=fxx, verbose=False)
                hd = {"fxx": fxx, "valid": (mt + timedelta(hours=fxx)).isoformat()}
                try:
                    ds = H.xarray(":TMP:2 m above ground", verbose=False)
                    hd["temp_c"] = float(ds["t2m"].sel(latitude=lat, longitude=lon+360, method="nearest").values) - 273.15
                except Exception: pass
                try:
                    ds = H.xarray(":APCP:surface", verbose=False)
                    vn = [v for v in ds.data_vars if "apcp" in v.lower() or "tp" in v.lower()]
                    if vn: hd["precip_mm"] = float(ds[vn[0]].sel(latitude=lat, longitude=lon+360, method="nearest").values)
                except Exception: pass
                try:
                    du = H.xarray(":UGRD:10 m above ground", verbose=False)
                    dv = H.xarray(":VGRD:10 m above ground", verbose=False)
                    un = [v for v in du.data_vars if "u" in v.lower()]
                    vn = [v for v in dv.data_vars if "v" in v.lower()]
                    if un and vn:
                        u = float(du[un[0]].sel(latitude=lat, longitude=lon+360, method="nearest").values)
                        v = float(dv[vn[0]].sel(latitude=lat, longitude=lon+360, method="nearest").values)
                        hd["wind_mph"] = math.sqrt(u**2 + v**2) * 2.237
                        hd["wind_dir"] = (math.degrees(math.atan2(-u, -v)) + 360) % 360
                except Exception: pass
                results["forecasts"].append(hd)
            except Exception as e:
                results["forecasts"].append({"fxx": fxx, "error": str(e)})
        return results
    except ImportError:
        return {"error": "Herbie not installed"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Multi-Model Processing
# ---------------------------------------------------------------------------

def parse_open_meteo(om: dict, elev_ft: int) -> dict:
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

        hours = []
        for i in range(len(times)):
            tc = temps[i] if i < len(temps) and temps[i] is not None else None
            # Adjust temperature to target elevation
            if tc is not None:
                tc = estimate_temp_c(tc, base_elev_m, target_m)
            tf = round(tc * 9/5 + 32, 1) if tc is not None else None
            ws = wspd[i] if i < len(wspd) and wspd[i] is not None else 0
            wd = wdir[i] if i < len(wdir) and wdir[i] is not None else 0
            wg = wgust[i] if i < len(wgust) and wgust[i] is not None else 0
            pr = precip[i] if i < len(precip) and precip[i] is not None else 0
            sf = snow[i] if i < len(snow) and snow[i] is not None else 0
            sd = sdepth[i] if i < len(sdepth) and sdepth[i] is not None else None

            # Compute SLR-adjusted snowfall at target elevation
            slr = compute_slr(tc) if tc is not None else 10
            oro = orographic_multiplier(elev_ft, ws * 0.621371, wd)
            adj_precip_in = (pr / 25.4) * oro  # mm -> inches, orographic adjusted
            pt = precip_type(tc, pr > 0.1) if tc is not None else "None"

            if pt == "Snow":
                snow_in = adj_precip_in * slr
            elif pt == "Mix":
                snow_in = adj_precip_in * slr * 0.5
            else:
                snow_in = 0

            fl = wind_chill_f(tf, ws * 0.621371) if tf is not None else None

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
        day_data["snow_range"] = (min(snows), max(snows)) if snows else (0, 0)
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


# ---------------------------------------------------------------------------
# Summary Generator (rule-based, no LLM needed)
# ---------------------------------------------------------------------------

def generate_summary(analysis: dict) -> str:
    """Generate a natural-language AI-style summary from the analysis data."""
    lines = []
    cur = analysis["current_conditions"]
    obs = cur.get("observation", {})

    # Current snapshot
    if obs:
        lines.append(f"Currently {obs.get('conditions', 'clear').lower()} at lake level with "
                      f"temperatures at {obs['temp_f']}F (feels like {obs['feels_like_f']}F).")
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
                afd_text: str, avy: dict, hrrr: dict) -> dict:
    now = datetime.now(timezone.utc)

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
            try: base_wind_mph = float(hourly[0].get("windSpeed", "0 mph").split()[0])
            except: pass
            base_wind_dir = DIR_MAP.get(hourly[0].get("windDirection", "W"), 270)

    base_temp_c = (base_temp_f - 32) * 5 / 9 if base_temp_f else 0.0
    base_elev_m = 6225 * 0.3048

    # Snow level
    profile = [(ft * 0.3048, estimate_temp_c(base_temp_c, base_elev_m, ft * 0.3048))
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
            parsed = parse_open_meteo(resort_om, loc["elev_ft"])
            zones[zk] = {
                "label": loc["label"],
                "elev_ft": loc["elev_ft"],
                "parsed": parsed,
            }
        # Nearest SNOTEL
        ns = [snotel[s] for s in resort.get("nearest_snotel", [])
              if s in snotel and "error" not in snotel[s]]
        resort_data[rn] = {"zones": zones, "aspect": resort["aspect"], "nearest_snotel": ns}

    # Current conditions block
    current = {
        "timestamp": now.isoformat(),
        "observation": obs,
        "lake_level_temp_f": base_temp_f,
        "wind_mph": round(base_wind_mph, 1),
        "wind_dir": wind_dir_str(base_wind_dir),
        "snow_level_ft": snow_level_ft,
        "freezing_level_ft": freeze_ft,
        "precipitation_active": False,
    }

    # Determine if precip is active from hourly
    hourly = nws.get("hourly", [])
    for p in hourly[:3]:
        prob = p.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
        if prob > 50:
            current["precipitation_active"] = True
            break

    # Build per-resort output with multi-model
    resorts_out = {}
    for rn, rd in resort_data.items():
        zones_out = {}
        for zk, zd in rd["zones"].items():
            p = zd["parsed"]
            if "error" not in p:
                # Use GFS as primary, but include all
                gfs = p["models"].get("GFS", [])

                # Current zone snapshot (first available hour)
                snap = gfs[0] if gfs else {}

                # 48h hourly timeline
                timeline_48h = gfs[:48]

                # Day/night buckets
                buckets = aggregate_daily(gfs)

                # 24h snow total (forecast)
                snow_24h = sum(h["snowfall_in"] for h in gfs[:24])

                # 7-day forecast snow total
                snow_7d_forecast = sum(h["snowfall_in"] for h in gfs[:168])

                # Multi-model spread (daily)
                model_spread = multi_model_spread(p)

                zones_out[zk] = {
                    "label": zd["label"],
                    "elev_ft": zd["elev_ft"],
                    "current": snap,
                    "timeline_48h": timeline_48h,
                    "day_night_buckets": buckets,
                    "snow_24h": round(snow_24h, 1),
                    "snow_7d_forecast": round(snow_7d_forecast, 1),
                    "model_spread": model_spread,
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

    # Historical + Season SNOTEL for select stations
    hist_stations = ["Mt Rose Ski Area", "CSS Lab", "Independence Lake"]
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
        spread = resorts_out.get(rn, {}).get("zones", {}).get("peak", {}).get("model_spread", [])
        total_3d = sum((d.get("models", {}).get("GFS", {}).get("snow_in", 0) for d in spread[:3]), 0)
        if total_3d > hero_72h:
            hero_72h = round(total_3d, 1)
            hero_72h_resort = rn
        total_7d = sum((d.get("models", {}).get("GFS", {}).get("snow_in", 0) for d in spread[:7]), 0)
        if total_7d > hero_7d_forecast:
            hero_7d_forecast = round(total_7d, 1)
            hero_7d_forecast_resort = rn

    result["hero_stats"] = {
        "temp_f": hero_temp_f,
        "snowpack_in": hero_snowpack,
        "snowpack_station": hero_snowpack_station,
        "snow_24h_in": hero_24h,
        "snow_24h_resort": hero_24h_resort,
        "snow_72h_in": hero_72h,
        "snow_72h_resort": hero_72h_resort,
        "snow_24h_hist_in": result_hist_24h,
        "snow_24h_hist_station": result_hist_24h_station,
        "snow_7d_hist_in": result_hist_7d,
        "snow_7d_hist_station": result_hist_7d_station,
        "snow_7d_forecast_in": hero_7d_forecast,
        "snow_7d_forecast_resort": hero_7d_forecast_resort,
    }
    result["hist_7d_snow"] = hist_7d_snow

    # Generate summary
    result["summary"] = generate_summary(result)

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
    except: L.append(f"  {a['generated']}")
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
        L.append(f"  Temperature:  {obs['temp_f']}F  (feels like {obs['feels_like_f']}F)")
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
                    except: tstr = h.get("time", "?")[:16]
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
    L.append("Models: GFS + ECMWF + ICON via Open-Meteo | Obs: SNOTEL + NWS")
    L.append("Physics: SLR (Roebber 2003), NWS wind chill, moist adiabatic lapse rate")
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

    print("\nAnalyzing conditions + building forecasts...")
    analysis = analyze_all(obs, nws, om, snotel, afd, avy, hrrr)

    if json_out:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(format_report(analysis, compact=compact))


if __name__ == "__main__":
    main()
