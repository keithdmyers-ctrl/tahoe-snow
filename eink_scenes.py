#!/usr/bin/env python3
"""
E-Ink Scene Manager -- Inky Impression 7.3" (2025 Edition)

Button-driven scene switching between Oakland local weather and Heavenly/Tahoe.

Buttons (GPIO on Inky Impression 7.3"):
  A (GPIO 5):  Show Oakland scene
  B (GPIO 6):  Show Heavenly scene
  C (GPIO 16): Force refresh current scene
  D (GPIO 24): Show Heavenly Detail scene

Runs as a daemon. Also supports cron-triggered refresh.

Usage:
  python eink_scenes.py                # start button listener daemon
  python eink_scenes.py --scene oakland   # render Oakland scene once
  python eink_scenes.py --scene heavenly  # render Heavenly scene once
  python eink_scenes.py --preview         # save PNG instead of display
  python eink_scenes.py --refresh         # refresh whatever scene is active
"""

import json
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import fetch_tahoe_analysis, fetch_oakland_data
from tahoe_snow import aggregate_pws
from sensors import read_all as read_sensors
from pressure_forecast import get_forecast as get_pressure_forecast, predict_rain_timing
from forecast_verification import log_daily_verification, get_bias_corrections
from eink_renderer import render_template, send_to_display, DISPLAY_W, DISPLAY_H

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOME = {"label": "Oakland", "lat": 37.8024, "lon": -122.1828, "elev_ft": 450}

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scene_state.json")
PREVIEW_DIR = os.path.dirname(os.path.abspath(__file__))

# Inky Impression 7.3" button GPIOs
BTN_A = 5   # Oakland
BTN_B = 6   # Heavenly
BTN_C = 16  # Refresh
BTN_D = 24  # Heavenly Detail

# Weather condition -> emoji mapping for NWS shortForecast text
CONDITION_ICONS = {
    "sunny": "\u2600\ufe0f", "clear": "\u2600\ufe0f", "mostly sunny": "\U0001f324",
    "partly sunny": "\u26c5", "partly cloudy": "\u26c5", "mostly cloudy": "\U0001f325",
    "cloudy": "\u2601\ufe0f", "overcast": "\u2601\ufe0f",
    "rain": "\U0001f327", "showers": "\U0001f327", "drizzle": "\U0001f327",
    "thunderstorm": "\u26c8", "thunder": "\u26c8",
    "snow": "\u2744\ufe0f", "flurries": "\U0001f328", "blizzard": "\u2744\ufe0f",
    "fog": "\U0001f32b", "haze": "\U0001f32b", "mist": "\U0001f32b",
    "wind": "\U0001f4a8", "breezy": "\U0001f4a8",
}


def _condition_icon(text: str) -> str:
    """Map NWS condition text to an emoji."""
    t = text.lower()
    for key, icon in CONDITION_ICONS.items():
        if key in t:
            return icon
    return "\U0001f321"


def _t(val):
    """Format temp to int string."""
    if val is None:
        return "--"
    try:
        return str(int(round(float(val))))
    except (ValueError, TypeError):
        return "--"


def _snow_str(val):
    if val is None:
        return '--'
    if val >= 0.5:
        return f'{val:.0f}"'
    elif val > 0:
        return f'{val:.1f}"'
    return '--'


# ---------------------------------------------------------------------------
# Data fetching (thread-safe cache)
# ---------------------------------------------------------------------------
_cache = {"data": None, "oakland": None, "sensors": None, "timestamp": 0}
_cache_lock = threading.Lock()
CACHE_TTL = 900  # 15 min


def fetch_all(force=False):
    """Fetch all data via shared pipeline, cached with lock protection."""
    now = time.time()
    with _cache_lock:
        if not force and _cache["data"] and (now - _cache["timestamp"]) < CACHE_TTL:
            return _cache

    print("Fetching sensor data...")
    sensors = read_sensors()

    print("Fetching Oakland weather...")
    oakland = fetch_oakland_data()

    print("Fetching Tahoe data + analysis...")
    analysis = fetch_tahoe_analysis()

    # Build a cache dict that the scene builders expect
    new_cache = {
        "data": analysis,
        "home_obs": oakland["home_obs"],
        "home_fc": oakland["home_fc"],
        "home_om": oakland["home_om"],
        "home_nbm": oakland["home_nbm"],
        "home_pws": oakland["home_pws"],
        "home_alerts": oakland["home_alerts"],
        "cssl": analysis.get("cssl"),
        "tahoe_alerts": analysis.get("alerts", []),
        "sounding": analysis.get("sounding"),
        "home_normals": None,  # Oakland normals not fetched here; use pipeline if needed
        "storm": analysis.get("storm"),
        "chains": analysis.get("chains"),
        "lifts": analysis.get("lifts"),
        "rwis": analysis.get("rwis", []),
        "sensors": sensors,
        "timestamp": time.time(),
    }

    # Daily forecast verification logging (best-effort)
    try:
        log_daily_verification(oakland["home_obs"], oakland["home_fc"], analysis)
    except Exception:
        pass  # verification is best-effort

    with _cache_lock:
        _cache.update(new_cache)

    return new_cache


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------
def build_oakland_context(cache) -> dict:
    """Build template context for Oakland scene."""
    analysis = cache["data"]
    home_obs = cache["home_obs"] or {}
    home_fc = cache["home_fc"] or {}
    sensors = cache["sensors"] or {}
    indoor = sensors.get("indoor", {})
    outdoor = sensors.get("outdoor", {})

    try:
        dt = datetime.fromisoformat(analysis["generated"])
        ts = dt.strftime("%a %b %d  %I:%M%p")
    except Exception:
        ts = ""

    # Today's high/low from NWS periods
    periods = home_fc.get("periods", [])
    today_hi = "--"
    today_lo = "--"
    for p in periods:
        if p.get("isDaytime", True) and today_hi == "--":
            today_hi = _t(p.get("temperature"))
        if not p.get("isDaytime", True) and today_lo == "--":
            today_lo = _t(p.get("temperature"))
        if today_hi != "--" and today_lo != "--":
            break

    # Hourly forecast (next 16 hours)
    hourly_raw = home_fc.get("hourly", [])[:16]
    hourly = []
    for h in hourly_raw:
        start = h.get("startTime", "")
        try:
            hdt = datetime.fromisoformat(start)
            time_label = hdt.strftime("%-I%p").lower()
        except Exception:
            time_label = ""
        short = h.get("shortForecast", "")
        icon = _condition_icon(short)
        temp = _t(h.get("temperature"))
        precip = h.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
        hourly.append({
            "time": time_label, "icon": icon, "temp": temp,
            "precip_pct": int(precip),
        })

    # 5-day forecast
    forecast = []
    seen_days = set()
    for p in periods:
        if not p.get("isDaytime", True):
            continue
        name = p.get("name", "")[:3]
        if name in seen_days:
            continue
        seen_days.add(name)
        short = p.get("shortForecast", "")
        icon = _condition_icon(short)
        hi = _t(p.get("temperature"))
        # Find matching night
        lo = "--"
        for np_ in periods:
            if not np_.get("isDaytime", True):
                np_start = np_.get("startTime", "")[:10]
                p_start = p.get("startTime", "")[:10]
                if np_start == p_start:
                    lo = _t(np_.get("temperature"))
                    break
        forecast.append({
            "name": name, "icon": icon, "hi": hi, "lo": lo,
            "short": short[:22],
        })
        if len(forecast) >= 5:
            break

    # Wind
    wind_parts = []
    wd = home_obs.get("wind_dir", "")
    ws = home_obs.get("wind_mph", 0)
    if ws:
        wind_parts.append(f"{wd} {ws:.0f}mph")
    wg = home_obs.get("wind_gust_mph", 0)
    if wg:
        wind_parts.append(f"gusts {wg:.0f}")
    wind = "  ".join(wind_parts)

    # Pressure forecast from BME280 history
    outdoor_humidity = outdoor.get("humidity_pct")
    pf = get_pressure_forecast(current_humidity=outdoor_humidity)

    # PWS consensus -- better ground truth than single NWS airport station
    pws_raw = cache.get("home_pws") or []
    pws = aggregate_pws(pws_raw) if pws_raw else {}

    # Combined rain timing prediction (NWS hourly + NBM + Open-Meteo + BME280)
    nws_hourly_full = home_fc.get("hourly", [])[:48]
    home_om = cache.get("home_om") or {}
    home_nbm = cache.get("home_nbm") or {}
    rain_timing = predict_rain_timing(nws_hourly_full, pf,
                                       open_meteo=home_om, nbm=home_nbm,
                                       pws_is_raining=pws.get("is_raining", False))

    # Use PWS consensus as fallback/cross-check for outdoor temp
    outdoor_temp = outdoor.get("temp_f")
    outdoor_stale = outdoor.get("stale", False)
    if (outdoor_stale or outdoor_temp is None) and pws.get("temp_f") is not None:
        outdoor_temp = pws["temp_f"]
        outdoor_stale = False  # PWS data is fresh

    # NWS alerts for Oakland area
    home_alerts = cache.get("home_alerts") or []
    alerts = [{"event": a["event"], "severity": a["severity"],
               "headline": a.get("headline", "")[:80]}
              for a in home_alerts[:3]]  # max 3 alerts

    # Climate normals anomaly
    normals = cache.get("home_normals") or {}
    temp_anomaly = None
    if normals.get("avg_high_f") and today_hi != "--":
        try:
            temp_anomaly = int(today_hi) - normals["avg_high_f"]
        except (ValueError, TypeError):
            pass

    # Solar data (sunrise/sunset from Open-Meteo daily)
    solar = {}
    home_om_data = cache.get("home_om") or {}
    if "daily" in home_om_data:
        daily = home_om_data["daily"]
        sr_list = daily.get("sunrise", [])
        ss_list = daily.get("sunset", [])
        if sr_list and ss_list:
            try:
                sr_dt = datetime.fromisoformat(sr_list[0])
                ss_dt = datetime.fromisoformat(ss_list[0])
                daylight_sec = (ss_dt - sr_dt).total_seconds()
                solar = {
                    "sunrise": sr_dt.strftime("%-I:%M %p"),
                    "sunset": ss_dt.strftime("%-I:%M %p"),
                    "daylight_hours": round(daylight_sec / 3600, 1),
                }
            except (ValueError, TypeError, IndexError):
                pass

    # Visibility from NWS observations
    visibility_mi = home_obs.get("visibility_mi")

    return {
        "location": HOME["label"].upper(),
        "timestamp": ts,
        "indoor_temp": _t(indoor.get("temp_f")),
        "indoor_humidity": indoor.get("humidity_pct"),
        "outdoor_temp": _t(outdoor_temp),
        "outdoor_humidity": outdoor_humidity,
        "outdoor_stale": outdoor_stale,
        "outdoor_pressure": outdoor.get("pressure_hpa"),
        "pressure_forecast": pf,
        "rain_timing": rain_timing,
        "today_high": today_hi,
        "today_low": today_lo,
        "conditions": home_obs.get("conditions", ""),
        "wind": wind,
        "nws_temp": _t(home_obs.get("temp_f")),
        "hourly": hourly,
        "forecast": forecast,
        "alerts": alerts,
        "temp_anomaly": temp_anomaly,
        "normals": normals,
        "solar": solar,
        "visibility_mi": visibility_mi,
        "scene_hint": "Active: Oakland",
    }


def build_heavenly_context(cache) -> dict:
    """Build template context for Heavenly scene."""
    analysis = cache["data"]
    cur = analysis["current_conditions"]
    tahoe_obs = cur.get("observation", {})
    avy = analysis.get("avalanche", {})
    snotel = analysis.get("snotel_current", {})

    try:
        dt = datetime.fromisoformat(analysis["generated"])
        ts = dt.strftime("%a %b %d  %I:%M%p")
    except Exception:
        ts = ""

    # Tahoe current
    tahoe_temp = _t(tahoe_obs.get("temp_f", cur.get("lake_level_temp_f")))
    tahoe_feels = _t(tahoe_obs.get("feels_like_f"))
    tahoe_cond = tahoe_obs.get("conditions", "")
    t_wd = tahoe_obs.get("wind_dir", cur.get("wind_dir", ""))
    t_ws = tahoe_obs.get("wind_mph", cur.get("wind_mph", 0))
    tahoe_wind = f"{t_wd} {t_ws}mph" if t_ws else ""

    sl = cur.get("snow_level_ft")
    fl = cur.get("freezing_level_ft")
    snow_level = f"{sl}ft" if sl is not None else "N/A"
    freeze_level = f"{fl}ft" if fl else "N/A"

    avy_label = avy.get("danger_label", "N/A") if avy else "N/A"
    avy_level = avy.get("danger_level", 0) if avy else 0
    avy_classes = {0: "", 1: "avy-low", 2: "avy-mod", 3: "avy-con", 4: "avy-high", 5: "avy-ext"}
    avy_class = avy_classes.get(avy_level, "")

    # Heavenly zones
    heavenly = analysis.get("resorts", {}).get("Heavenly", {})
    zones = []
    for zone_key, zone_name in [("peak", "Peak"), ("mid", "Mid"), ("base", "Base")]:
        zd = heavenly.get("zones", {}).get(zone_key, {})
        snap = zd.get("current", {})
        snow24 = zd.get("snow_24h", 0)
        zones.append({
            "name": zone_name,
            "elev": zd.get("elev_ft", "?"),
            "temp": _t(snap.get("temp_f")),
            "feels": _t(snap.get("feels_like_f")),
            "snow": snow24,
            "snow_str": _snow_str(snow24),
            "wind_dir": snap.get("wind_dir", ""),
            "wind_mph": f"{snap.get('wind_mph', 0):.0f}" if snap.get("wind_mph") else "",
            "gust": snap.get("wind_gust_mph", 0),
            "quality": (snap.get("snow_quality", "")[:12]) if snow24 > 0 else "",
            "precip_type": snap.get("precip_type", "None"),
        })

    # 5-day snow forecast (peak zone)
    peak = heavenly.get("zones", {}).get("peak", {})
    buckets = peak.get("day_night_buckets", [])
    daily = {}
    for b in buckets:
        d = b["date"]
        if d not in daily:
            daily[d] = {"snow": 0, "hi": None, "lo": None, "conditions": []}
        daily[d]["snow"] += b["snow_in"]
        if b.get("temp_high_f") is not None:
            if daily[d]["hi"] is None or b["temp_high_f"] > daily[d]["hi"]:
                daily[d]["hi"] = b["temp_high_f"]
        if b.get("temp_low_f") is not None:
            if daily[d]["lo"] is None or b["temp_low_f"] < daily[d]["lo"]:
                daily[d]["lo"] = b["temp_low_f"]
        cond = b.get("conditions", "")
        if cond and cond not in daily[d]["conditions"]:
            daily[d]["conditions"].append(cond)

    snow_days = []
    for date in sorted(daily.keys())[:7]:
        dd = daily[date]
        try:
            ddt = datetime.strptime(date, "%Y-%m-%d")
            name = ddt.strftime("%a")
        except Exception:
            name = date[-5:]
        short = dd["conditions"][0][:16] if dd["conditions"] else ""
        icon = _condition_icon(short) if short else "\U0001f321"
        snow_days.append({
            "name": name,
            "snow": dd["snow"],
            "snow_str": _snow_str(dd["snow"]),
            "hi": _t(dd["hi"]),
            "lo": _t(dd["lo"]),
            "short": short,
            "icon": icon,
        })

    # SNOTEL stations (top by depth) + CSSL
    stations = [(n, d) for n, d in snotel.items()
                if "error" not in d and d.get("snow_depth_in") is not None
                and d.get("snow_depth_in") > 0]
    # Include CSSL (Central Sierra Snow Lab) if available
    cssl = cache.get("cssl") or {}
    if cssl.get("snow_depth_in") is not None and "error" not in cssl:
        stations.append(("CSSL", {"snow_depth_in": cssl["snow_depth_in"]}))
    stations.sort(key=lambda x: x[1]["snow_depth_in"], reverse=True)
    snotel_list = [{"name": n[:12], "depth": f"{d['snow_depth_in']:.0f}"}
                   for n, d in stations[:6]]

    # NWS alerts for Tahoe area
    tahoe_alerts = cache.get("tahoe_alerts") or []
    alerts = [{"event": a["event"], "severity": a["severity"],
               "headline": a.get("headline", "")[:80]}
              for a in tahoe_alerts[:3]]

    # Sounding-derived data (override modeled snow/freeze levels with observed)
    sounding = cache.get("sounding") or {}
    if sounding.get("freezing_level_ft") and "error" not in sounding:
        freeze_level = f"{sounding['freezing_level_ft']}ft"
    if sounding.get("snow_level_ft") and "error" not in sounding:
        snow_level = f"{sounding['snow_level_ft']}ft"
    sounding_lapse = sounding.get("lapse_rate_c_km")

    # Storm totals
    storm = cache.get("storm") or {}

    # Chain controls
    chains = cache.get("chains") or []

    # Lift status (Heavenly)
    lifts_data = (cache.get("lifts") or {}).get("heavenly", {})
    lifts_open = lifts_data.get("open", 0) if "error" not in lifts_data else None
    lifts_total = lifts_data.get("total", 0) if "error" not in lifts_data else None

    # RWIS road temp -- pick the first station with pavement data for compact display
    rwis = cache.get("rwis") or []
    road_temp_f = None
    for r in rwis:
        if r.get("pavement_temp_f") is not None:
            road_temp_f = round(r["pavement_temp_f"])
            break

    return {
        "timestamp": ts,
        "tahoe_temp": tahoe_temp,
        "tahoe_feels": tahoe_feels,
        "tahoe_conditions": tahoe_cond,
        "tahoe_wind": tahoe_wind,
        "snow_level": snow_level,
        "freeze_level": freeze_level,
        "avy_label": avy_label,
        "avy_class": avy_class,
        "zones": zones,
        "snow_days": snow_days,
        "snotel_stations": snotel_list,
        "alerts": alerts,
        "storm": storm,
        "chains": chains,
        "lifts_open": lifts_open,
        "lifts_total": lifts_total,
        "sounding_lapse": sounding_lapse,
        "road_temp_f": road_temp_f,
        "scene_hint": "Active: Heavenly",
    }


PRECIP_ICONS = {"Snow": "\u2744\ufe0f", "Mix": "\U0001f328", "Rain": "\U0001f327", "None": ""}
PRECIP_CLASSES = {"Snow": "snow-type", "Mix": "mix-type", "Rain": "rain-type", "None": "none-type"}
ZONE_ICONS = {"Peak": "\u25b2", "Mid": "\u25a0", "Base": "\u25cf"}


def build_detail_context(cache) -> dict:
    """Build template context for the mountain detail scene."""
    analysis = cache["data"]
    cur = analysis["current_conditions"]
    tahoe_obs = cur.get("observation", {})
    avy = analysis.get("avalanche", {})
    snotel = analysis.get("snotel_current", {})

    try:
        dt = datetime.fromisoformat(analysis["generated"])
        ts = dt.strftime("%a %b %d  %I:%M%p")
    except Exception:
        ts = ""

    sl = cur.get("snow_level_ft")
    fl = cur.get("freezing_level_ft")
    snow_level = f"{sl}ft" if sl is not None else "N/A"
    freeze_level = f"{fl}ft" if fl else "N/A"

    avy_label = avy.get("danger_label", "N/A") if avy else "N/A"
    avy_level = avy.get("danger_level", 0) if avy else 0
    avy_classes = {0: "", 1: "avy-low", 2: "avy-mod", 3: "avy-con", 4: "avy-high", 5: "avy-ext"}
    avy_class = avy_classes.get(avy_level, "")

    heavenly = analysis.get("resorts", {}).get("Heavenly", {})

    # Build zones in peak -> mid -> base order (high to low elevation)
    zones = []
    for zone_key, zone_name in [("peak", "Peak"), ("mid", "Mid"), ("base", "Base")]:
        zd = heavenly.get("zones", {}).get(zone_key, {})
        snap = zd.get("current", {})
        snow24 = zd.get("snow_24h", 0)
        timeline = zd.get("timeline_48h", [])
        precip_type = snap.get("precip_type", "None")

        # Build 12-hour mini timeline
        hourly = []
        max_snow = max((h.get("snowfall_in", 0) for h in timeline[:12]), default=1) or 1
        for h in timeline[:12]:
            t = h.get("time", "")
            try:
                hdt = datetime.fromisoformat(t)
                time_label = hdt.strftime("%-I%p").lower()
            except Exception:
                time_label = ""
            snow_in = h.get("snowfall_in", 0)
            pt = h.get("precip_type", "None")
            bar_class = "sn" if pt == "Snow" else "mx" if pt == "Mix" else "rn"
            bar_pct = min(int((snow_in / max_snow) * 100), 100) if snow_in > 0 else 0
            hourly.append({
                "time": time_label,
                "temp": _t(h.get("temp_f")),
                "snow": snow_in,
                "snow_label": f'{snow_in:.0f}"' if snow_in >= 0.5 else "",
                "bar_class": bar_class,
                "bar_pct": max(bar_pct, 10) if snow_in > 0 else 0,
            })

        zones.append({
            "name": zone_name,
            "icon": ZONE_ICONS.get(zone_name, "\u00b7"),
            "label": zd.get("label", ""),
            "elev": zd.get("elev_ft", "?"),
            "temp": _t(snap.get("temp_f")),
            "feels": _t(snap.get("feels_like_f")),
            "snow_24h": snow24,
            "snow_str": _snow_str(snow24),
            "wind_dir": snap.get("wind_dir", ""),
            "wind_mph": f"{snap.get('wind_mph', 0):.0f}" if snap.get("wind_mph") else "",
            "gust": snap.get("wind_gust_mph", 0),
            "quality": (snap.get("snow_quality", "")[:12]) if snow24 > 0 else "",
            "precip_type": precip_type,
            "precip_icon": PRECIP_ICONS.get(precip_type, "\u2014"),
            "precip_class": PRECIP_CLASSES.get(precip_type, "none-type"),
            "hourly": hourly,
        })

    # SNOTEL stations
    stations = [(n, d) for n, d in snotel.items()
                if "error" not in d and d.get("snow_depth_in") is not None
                and d.get("snow_depth_in") > 0]
    stations.sort(key=lambda x: x[1]["snow_depth_in"], reverse=True)
    snotel_list = [{"name": n[:12], "depth": f"{d['snow_depth_in']:.0f}"}
                   for n, d in stations[:6]]

    # Sounding-derived data (override modeled levels with observed)
    sounding = cache.get("sounding") or {}
    if sounding.get("freezing_level_ft") and "error" not in sounding:
        freeze_level = f"{sounding['freezing_level_ft']}ft"
    if sounding.get("snow_level_ft") and "error" not in sounding:
        snow_level = f"{sounding['snow_level_ft']}ft"

    # Storm totals
    storm = cache.get("storm") or {}

    # Tahoe alerts
    tahoe_alerts = cache.get("tahoe_alerts") or []
    alerts = [{"event": a["event"], "severity": a["severity"],
               "headline": a.get("headline", "")[:80]}
              for a in tahoe_alerts[:3]]

    return {
        "timestamp": ts,
        "snow_level": snow_level,
        "freeze_level": freeze_level,
        "avy_label": avy_label,
        "avy_class": avy_class,
        "zones": zones,
        "snotel_stations": snotel_list,
        "alerts": alerts,
        "storm": storm,
        "scene_hint": "Active: Detail",
    }


# ---------------------------------------------------------------------------
# Scene rendering
# ---------------------------------------------------------------------------
def render_scene(scene: str, preview=False, force_refresh=False):
    """Render a scene to display or preview file."""
    cache = fetch_all(force=force_refresh)

    if scene == "oakland":
        ctx = build_oakland_context(cache)
        template = "oakland.html"
    elif scene == "heavenly":
        ctx = build_heavenly_context(cache)
        template = "heavenly.html"
    elif scene == "detail":
        ctx = build_detail_context(cache)
        template = "detail.html"
    else:
        raise ValueError(f"Unknown scene: {scene}")

    print(f"Rendering {scene} scene...")
    img = render_template(template, ctx)

    if preview:
        path = os.path.join(PREVIEW_DIR, f"preview_{scene}.png")
        img.save(path)
        print(f"Preview saved to {path}")
    else:
        if not send_to_display(img):
            path = os.path.join(PREVIEW_DIR, f"preview_{scene}.png")
            img.save(path)
            print(f"No display found. Preview saved to {path}")
        else:
            print("Display updated.")

    # Save active scene state
    save_state(scene)
    return img


def save_state(scene: str):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"scene": scene, "updated": time.time()}, f)
    os.replace(tmp, STATE_FILE)


def load_state() -> str:
    try:
        with open(STATE_FILE) as f:
            return json.load(f).get("scene", "oakland")
    except (FileNotFoundError, json.JSONDecodeError):
        return "oakland"


# ---------------------------------------------------------------------------
# Button listener
# ---------------------------------------------------------------------------
def start_button_listener():
    """Listen for button presses and switch scenes. Blocks forever."""
    try:
        import gpiod
        from gpiod.line import Bias, Direction, Edge
    except ImportError:
        print("gpiod not available -- button listener requires Raspberry Pi.")
        print("Run with --scene oakland or --scene heavenly instead.")
        return

    # Pi 5 uses gpiochip4, Pi 3/4 use gpiochip0
    chip_path = "/dev/gpiochip4" if os.path.exists("/dev/gpiochip4") else "/dev/gpiochip0"
    chip = gpiod.Chip(chip_path)
    print(f"Using GPIO chip: {chip_path}")
    line_config = {
        BTN_A: gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP,
                                  edge_detection=Edge.FALLING, debounce_period=None),
        BTN_B: gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP,
                                  edge_detection=Edge.FALLING, debounce_period=None),
        BTN_C: gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP,
                                  edge_detection=Edge.FALLING, debounce_period=None),
        BTN_D: gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP,
                                  edge_detection=Edge.FALLING, debounce_period=None),
    }

    request = chip.request_lines(consumer="tahoe-snow-scenes", config=line_config)
    print("Button listener active. Press A=Oakland, B=Heavenly, C=Refresh, D=Detail")

    # Render current scene on startup
    current = load_state()
    render_scene(current)

    while True:
        for event in request.read_edge_events():
            pin = event.line_offset
            print(f"Button press: GPIO {pin}")

            if pin == BTN_A:
                render_scene("oakland")
            elif pin == BTN_B:
                render_scene("heavenly")
            elif pin == BTN_C:
                current = load_state()
                render_scene(current, force_refresh=True)
            elif pin == BTN_D:
                render_scene("detail")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    preview = "--preview" in sys.argv

    if "--scene" in sys.argv:
        idx = sys.argv.index("--scene")
        if idx + 1 < len(sys.argv):
            scene = sys.argv[idx + 1]
            render_scene(scene, preview=preview)
            return

    if "--refresh" in sys.argv:
        scene = load_state()
        render_scene(scene, preview=preview, force_refresh=True)
        return

    if "--listen" in sys.argv:
        start_button_listener()
        return

    # Default: start button listener daemon
    start_button_listener()


if __name__ == "__main__":
    main()
