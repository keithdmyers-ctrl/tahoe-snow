#!/usr/bin/env python3
"""
Barometric pressure trend analysis and weather forecasting.

Uses a Zambretti-inspired algorithm — the same approach used by commercial
weather stations (Davis Vantage, Oregon Scientific, La Crosse) — adapted
for BME280 sensor data from the outdoor ESP32.

The Zambretti Forecaster (1920, Negretti & Zambra) predicts 12-hour weather
from three inputs: absolute pressure, 3-hour pressure trend, and optionally
wind direction / season. It achieves ~90% accuracy in temperate climates.

This implementation:
  - Tracks pressure history in a JSON file (rolling 24h window)
  - Computes 1h and 3h pressure change rates
  - Classifies trend as rising/steady/falling with intensity
  - Produces a Zambretti-class weather forecast + rain probability
  - Combines pressure trend with humidity for better rain prediction

Usage:
  from pressure_forecast import get_forecast, record_pressure
  record_pressure(1018.4)  # call on each sensor reading
  fc = get_forecast()      # returns forecast dict
"""

import json
import math
import os
from datetime import datetime, timezone, timedelta

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".pressure_history.json")
MAX_HISTORY_HOURS = 24
# 2706 Kingsland Ave, Oakland hills — ~450ft / 137m
STATION_ELEVATION_M = 137


def _load_history() -> list[dict]:
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_history(history: list[dict]):
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f)
    os.replace(tmp, HISTORY_FILE)


def _sea_level_pressure(station_hpa: float, elevation_m: float) -> float:
    """Convert station pressure to mean sea level pressure (QNH).
    Standard barometric formula used by weather services."""
    if elevation_m == 0:
        return station_hpa
    return station_hpa * (1 - 0.0065 * elevation_m / 288.15) ** -5.255


def record_pressure(pressure_hpa: float, humidity_pct: float | None = None):
    """Record a pressure reading to history. Call on each sensor update."""
    now = datetime.now(timezone.utc)
    history = _load_history()

    slp = _sea_level_pressure(pressure_hpa, STATION_ELEVATION_M)

    history.append({
        "t": now.isoformat(),
        "hpa": round(slp, 2),
        "raw": round(pressure_hpa, 2),
        "rh": round(humidity_pct, 1) if humidity_pct is not None else None,
    })

    # Trim to MAX_HISTORY_HOURS
    cutoff = now - timedelta(hours=MAX_HISTORY_HOURS)
    history = [h for h in history if datetime.fromisoformat(h["t"]) > cutoff]

    _save_history(history)


def _get_pressure_at_offset(history: list[dict], hours_ago: float) -> float | None:
    """Find the pressure reading closest to `hours_ago` hours before now."""
    if not history:
        return None
    now = datetime.fromisoformat(history[-1]["t"])
    target = now - timedelta(hours=hours_ago)
    # Find closest reading within 15-min tolerance
    best = None
    best_delta = timedelta(minutes=15)
    for h in history:
        dt = abs(datetime.fromisoformat(h["t"]) - target)
        if dt < best_delta:
            best = h["hpa"]
            best_delta = dt
    return best


def _classify_trend(delta_3h: float) -> tuple[str, str]:
    """Classify 3-hour pressure change.

    Commercial stations (Davis, La Crosse) use these thresholds:
      Rapidly falling:  < -6.0 hPa/3h
      Falling:          < -1.6 hPa/3h
      Slowly falling:   < -0.5 hPa/3h
      Steady:           -0.5 to +0.5 hPa/3h
      Slowly rising:    > +0.5 hPa/3h
      Rising:           > +1.6 hPa/3h
      Rapidly rising:   > +6.0 hPa/3h

    Returns (direction, intensity).
    """
    if delta_3h <= -6.0:
        return "falling", "rapidly"
    elif delta_3h <= -1.6:
        return "falling", "falling"
    elif delta_3h <= -0.5:
        return "falling", "slowly"
    elif delta_3h < 0.5:
        return "steady", "steady"
    elif delta_3h < 1.6:
        return "rising", "slowly"
    elif delta_3h < 6.0:
        return "rising", "rising"
    else:
        return "rising", "rapidly"


# Zambretti forecast lookup tables
# Each table maps a Z-number to (short_text, rain_probability_pct)
_ZAMBRETTI_FALLING = [
    ("Settled Fine",              5),   # 0
    ("Fine Weather",             10),   # 1
    ("Fine, Becoming Less Settled", 15),  # 2
    ("Fairly Fine, Showery Later", 30),   # 3
    ("Showery, Becoming More Unsettled", 50),  # 4
    ("Unsettled, Rain Later",    65),   # 5
    ("Rain at Times, Worse Later", 75),   # 6
    ("Rain at Times, Becoming Very Unsettled", 80),  # 7
    ("Very Unsettled, Rain",     85),   # 8
    ("Stormy, Much Rain",        95),   # 9
]

_ZAMBRETTI_STEADY = [
    ("Settled Fine",              5),   # 10
    ("Fine Weather",             10),   # 11
    ("Fine, Possibly Showers",   20),   # 12
    ("Fairly Fine, Showers Likely", 35),  # 13
    ("Showery, Bright Intervals", 45),   # 14
    ("Changeable, Some Rain",    55),   # 15
    ("Unsettled, Rain at Times", 65),   # 16
    ("Rain at Frequent Intervals", 75),  # 17
    ("Very Unsettled, Rain",     85),   # 18
    ("Stormy, Much Rain",        95),   # 19
]

_ZAMBRETTI_RISING = [
    ("Settled Fine",              5),   # 20
    ("Fine Weather",              5),   # 21
    ("Becoming Fine",            10),   # 22
    ("Fairly Fine, Improving",   10),   # 23
    ("Fairly Fine, Possibly Showers", 20),  # 24
    ("Showery Early, Improving", 25),   # 25
    ("Changeable, Mending",      30),   # 26
    ("Rather Unsettled, Clearing Later", 40),  # 27
    ("Unsettled, Probably Improving", 45),  # 28
    ("Unsettled, Short Fine Intervals", 55),  # 29
    ("Very Unsettled, Finer at Times", 60),  # 30
    ("Stormy, Possibly Improving", 65),  # 31
    ("Stormy, Much Rain",        80),   # 32
]


def _zambretti_number(pressure_hpa: float, trend: str, month: int) -> int:
    """Calculate Zambretti Z-number from sea-level pressure and trend.

    Uses the standard Zambretti equations from Negretti & Zambra (1920),
    as implemented by commercial weather stations.
    """
    p = pressure_hpa

    # Season adjustment: winter = more pessimistic, summer = more optimistic
    # Northern hemisphere: winter = Nov-Feb, summer = May-Aug
    if month in (11, 12, 1, 2):
        season_adj = 1  # shift toward worse weather
    elif month in (5, 6, 7, 8):
        season_adj = -1  # shift toward better weather
    else:
        season_adj = 0

    if trend == "falling":
        # Z = 127 - 0.12 * P  (produces values in range 0-9)
        z = int(round(127 - 0.12 * p))
        z = max(0, min(9, z + season_adj))
        return z
    elif trend == "rising":
        # Z = 185 - 0.16 * P  (produces values in range 20-32)
        z = int(round(185 - 0.16 * p))
        z = max(20, min(32, z - season_adj))
        return z
    else:
        # Steady: Z = 144 - 0.13 * P  (produces values in range 10-19)
        z = int(round(144 - 0.13 * p))
        z = max(10, min(19, z))
        return z


def _lookup_forecast(z: int) -> tuple[str, int]:
    """Look up forecast text and rain probability from Z-number."""
    if z <= 9:
        return _ZAMBRETTI_FALLING[z]
    elif z <= 19:
        return _ZAMBRETTI_STEADY[z - 10]
    else:
        idx = min(z - 20, len(_ZAMBRETTI_RISING) - 1)
        return _ZAMBRETTI_RISING[idx]


def get_forecast(current_humidity: float | None = None) -> dict:
    """Generate a weather forecast from pressure history.

    Returns:
        {
            "pressure_hpa": 1018.4,       # current sea-level pressure
            "trend_1h": -0.8,             # 1-hour change in hPa
            "trend_3h": -2.1,             # 3-hour change in hPa
            "trend_dir": "falling",       # rising / steady / falling
            "trend_label": "Falling",     # human-readable trend
            "trend_arrow": "v",           # single char arrow for e-ink
            "forecast": "Showery, Becoming More Unsettled",
            "rain_pct": 50,               # rain probability 0-100
            "confidence": "good",         # good / fair / low (based on data age)
            "data_hours": 3.2,            # hours of history available
        }
    """
    history = _load_history()
    if not history:
        return {"error": "no_data", "forecast": "Waiting for data",
                "rain_pct": None, "confidence": "none"}

    now_entry = history[-1]
    current_p = now_entry["hpa"]

    # How much history do we have?
    first_t = datetime.fromisoformat(history[0]["t"])
    last_t = datetime.fromisoformat(now_entry["t"])
    data_hours = (last_t - first_t).total_seconds() / 3600

    # Get pressure deltas
    p_1h_ago = _get_pressure_at_offset(history, 1.0)
    p_3h_ago = _get_pressure_at_offset(history, 3.0)

    trend_1h = (current_p - p_1h_ago) if p_1h_ago is not None else None
    trend_3h = (current_p - p_3h_ago) if p_3h_ago is not None else None

    # Determine confidence
    if data_hours >= 3.0 and trend_3h is not None:
        confidence = "good"
        delta_for_trend = trend_3h
    elif data_hours >= 1.0 and trend_1h is not None:
        confidence = "fair"
        # Extrapolate 1h trend to 3h equivalent for Zambretti
        delta_for_trend = trend_1h * 3.0
    else:
        confidence = "low"
        delta_for_trend = 0.0  # treat as steady

    # Classify trend
    trend_dir, intensity = _classify_trend(delta_for_trend)

    # Arrow characters for e-ink display
    arrows = {
        ("rising", "rapidly"): "^^",
        ("rising", "rising"): "^",
        ("rising", "slowly"): "^",
        ("steady", "steady"): "-",
        ("falling", "slowly"): "v",
        ("falling", "falling"): "v",
        ("falling", "rapidly"): "vv",
    }
    trend_arrow = arrows.get((trend_dir, intensity), "-")

    # Human-readable trend label
    labels = {
        ("rising", "rapidly"): "Rising Fast",
        ("rising", "rising"): "Rising",
        ("rising", "slowly"): "Rising Slowly",
        ("steady", "steady"): "Steady",
        ("falling", "slowly"): "Falling Slowly",
        ("falling", "falling"): "Falling",
        ("falling", "rapidly"): "Falling Fast",
    }
    trend_label = labels.get((trend_dir, intensity), "Steady")

    # Zambretti forecast
    month = datetime.now().month
    z = _zambretti_number(current_p, trend_dir, month)
    forecast_text, rain_pct = _lookup_forecast(z)

    # Humidity adjustment: high humidity + falling pressure = more rain confidence
    if current_humidity is not None:
        if current_humidity > 85 and trend_dir == "falling":
            rain_pct = min(rain_pct + 15, 99)
        elif current_humidity > 80 and trend_dir == "falling":
            rain_pct = min(rain_pct + 10, 99)
        elif current_humidity < 40 and trend_dir != "falling":
            rain_pct = max(rain_pct - 10, 0)

    return {
        "pressure_hpa": current_p,
        "trend_1h": round(trend_1h, 2) if trend_1h is not None else None,
        "trend_3h": round(trend_3h, 2) if trend_3h is not None else None,
        "trend_dir": trend_dir,
        "trend_label": trend_label,
        "trend_arrow": trend_arrow,
        "forecast": forecast_text,
        "rain_pct": rain_pct,
        "confidence": confidence,
        "data_hours": round(data_hours, 1),
    }


def _parse_open_meteo_precip(om: dict) -> list[float]:
    """Extract hourly precipitation totals (mm) from Open-Meteo multi-model data.
    Returns a list of per-hour precip amounts averaged across available models."""
    if not om or "error" in om:
        return []
    # Open-Meteo returns per-model hourly data
    model_precips = []
    for key, val in om.items():
        if not isinstance(val, dict):
            continue
        hourly = val.get("hourly", om.get("hourly"))
        if hourly and "precipitation" in hourly:
            model_precips.append(hourly["precipitation"])
    # Also check top-level hourly (single-model response)
    if not model_precips and "hourly" in om and "precipitation" in om["hourly"]:
        model_precips.append(om["hourly"]["precipitation"])
    if not model_precips:
        return []
    # Average across models, truncate to min length
    min_len = min(len(p) for p in model_precips)
    averaged = []
    for i in range(min_len):
        vals = [p[i] for p in model_precips if p[i] is not None]
        averaged.append(sum(vals) / len(vals) if vals else 0)
    return averaged


def _count_models_with_precip(om: dict, hour_range: int = 24) -> tuple[int, int]:
    """Count how many Open-Meteo models show precipitation in the next N hours.
    Returns (models_with_precip, total_models)."""
    if not om or "error" in om:
        return 0, 0
    models_total = 0
    models_wet = 0
    for key, val in om.items():
        if not isinstance(val, dict):
            continue
        hourly = val.get("hourly")
        if hourly and "precipitation" in hourly:
            models_total += 1
            precip = hourly["precipitation"][:hour_range]
            if any((p or 0) > 0.1 for p in precip):
                models_wet += 1
    # Single-model fallback
    if models_total == 0 and "hourly" in om and "precipitation" in om["hourly"]:
        models_total = 1
        precip = om["hourly"]["precipitation"][:hour_range]
        if any((p or 0) > 0.1 for p in precip):
            models_wet = 1
    return models_wet, models_total


def predict_rain_timing(nws_hourly: list[dict], pressure_fc: dict,
                        open_meteo: dict | None = None,
                        nbm: dict | None = None,
                        pws_is_raining: bool = False) -> dict:
    """Combine all available data sources to predict rain timing.

    Data sources fused:
      1. NWS hourly precip probability (best timing — radar/model-based)
      2. NWS hourly dewpoint depression (moisture approaching — dp close to temp)
      3. Open-Meteo multi-model precip (GFS, ECMWF, ICON, HRRR model agreement)
      4. NBM (National Blend of Models) precip probability — bias-corrected consensus
      5. BME280 pressure trend (local early warning)
      6. BME280 humidity (current local moisture)
      7. PWS ground truth (nearby personal weather stations — is it raining now?)

    Returns dict with will_rain, rain_start, hours_until, combined_pct, summary, etc.
    """
    if not nws_hourly:
        return _rain_result_none(pressure_fc)

    now = datetime.now(timezone.utc)

    # --- 1. Scan NWS hourly for precipitation windows ---
    rain_hours = []
    dewpoint_approaching = False
    for h in nws_hourly[:48]:
        pop = h.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
        rh = h.get("relativeHumidity", {}).get("value") or 0
        dp_c = h.get("dewpoint", {}).get("value")  # Celsius
        temp_f = h.get("temperature")
        start = h.get("startTime", "")
        short = h.get("shortForecast", "").lower()

        rain_keywords = ("rain", "shower", "drizzle", "thunderstorm", "precip")
        text_says_rain = any(k in short for k in rain_keywords)

        # Dewpoint depression: when dewpoint approaches air temp, precip imminent
        dp_depression = None
        if dp_c is not None and temp_f is not None:
            temp_c = (temp_f - 32) * 5 / 9
            dp_depression = temp_c - dp_c
            if dp_depression < 3.0 and not dewpoint_approaching:
                dewpoint_approaching = True

        rain_hours.append({
            "start": start,
            "pop": pop,
            "rh": rh,
            "dewpoint_c": dp_c,
            "dp_depression": dp_depression,
            "text_rain": text_says_rain,
            "short": h.get("shortForecast", ""),
        })

    # --- 2. Find rain onset from NWS ---
    POP_ONSET_THRESHOLD = 20
    POP_LIKELY_THRESHOLD = 50

    nws_onset = None
    nws_peak_pct = 0
    nws_onset_idx = None
    for i, rh in enumerate(rain_hours):
        pop = rh["pop"]
        if pop > nws_peak_pct:
            nws_peak_pct = pop
        if nws_onset is None and (pop >= POP_ONSET_THRESHOLD or rh["text_rain"]):
            nws_onset = rh
            nws_onset_idx = i

    # Find rain end
    nws_end = None
    if nws_onset_idx is not None:
        for i in range(nws_onset_idx + 1, len(rain_hours)):
            if rain_hours[i]["pop"] < POP_ONSET_THRESHOLD and not rain_hours[i]["text_rain"]:
                nws_end = rain_hours[i]
                break

    # --- 3. Open-Meteo model agreement ---
    om = open_meteo or {}
    models_wet, models_total = _count_models_with_precip(om, 24)
    model_agreement = models_wet / models_total if models_total > 0 else 0

    # --- 3b. NBM precipitation probability (bias-corrected consensus) ---
    nbm_max_pop = 0
    nbm_data = nbm or {}
    if "hourly" in nbm_data:
        nbm_pop = nbm_data["hourly"].get("precipitation_probability", [])
        if nbm_pop:
            nbm_max_pop = max((p or 0) for p in nbm_pop[:24])

    # --- 4. Score pressure-based rain likelihood ---
    baro_rain = False
    baro_timing_hours = None
    trend_dir = pressure_fc.get("trend_dir", "steady")
    trend_3h = pressure_fc.get("trend_3h")
    baro_rain_pct = pressure_fc.get("rain_pct", 0) or 0

    if trend_dir == "falling" and baro_rain_pct >= 40:
        baro_rain = True
        if trend_3h is not None:
            rate = abs(trend_3h)
            if rate >= 6.0:
                baro_timing_hours = 2
            elif rate >= 3.0:
                baro_timing_hours = 4
            elif rate >= 1.6:
                baro_timing_hours = 8
            else:
                baro_timing_hours = 12

    # --- 5. Weighted probability fusion ---
    #
    # Each source contributes a rain probability (0-100) and a weight.
    # Weights reflect known forecast skill:
    #   - NBM: highest — it's NWS's own bias-corrected blend of 31 models
    #   - NWS POP: high — radar-informed, but single-model
    #   - Open-Meteo agreement: medium — raw multi-model consensus
    #   - Barometer: low — local early warning, not calibrated probability
    #   - Dewpoint: modifier — physical signal, not a probability source
    #   - PWS: override — ground truth (it's either raining or not)
    #
    # Sources only contribute if they have data. Final probability is
    # weighted average of contributing sources, then modified by
    # dewpoint signal and PWS ground truth.

    sources = []  # list of (probability, weight)

    # NBM — bias-corrected consensus (weight: 0.35)
    if nbm_max_pop > 0 or nbm_data:
        sources.append((nbm_max_pop, 0.35))

    # NWS hourly POP — radar-informed (weight: 0.30)
    if nws_peak_pct > 0 or nws_onset is not None:
        sources.append((nws_peak_pct, 0.30))

    # Open-Meteo model agreement — raw multi-model (weight: 0.20)
    # Convert agreement fraction to probability-like score
    if models_total > 0:
        # Scale: 0 models agree = 5%, all agree = 80%
        om_pct = model_agreement * 75 + 5
        sources.append((om_pct, 0.20))

    # Barometer/Zambretti — local early warning (weight: 0.15)
    if baro_rain_pct > 0 or baro_rain:
        sources.append((baro_rain_pct, 0.15))

    # Compute weighted average
    if not sources or all(s[0] == 0 for s in sources):
        # No source shows rain
        if not pws_is_raining:
            return _rain_result_none(pressure_fc)
        # PWS override below will handle this

    total_weight = sum(w for _, w in sources)
    if total_weight > 0:
        combined_pct = sum(p * w for p, w in sources) / total_weight
    else:
        combined_pct = 0

    # Dewpoint modifier: moisture approaching boosts confidence (+8%)
    if dewpoint_approaching and combined_pct > 0:
        combined_pct = combined_pct * 1.08

    # Clamp to 1-99 range
    combined_pct = max(1, min(99, round(combined_pct)))

    # Recalibrate using verification data (Platt-style)
    try:
        from forecast_verification import recalibrate_pop
        combined_pct = recalibrate_pop(combined_pct)
    except Exception:
        pass  # Use uncalibrated if verification module unavailable

    # Determine if rain is predicted (threshold: 15%)
    RAIN_THRESHOLD = 15
    will_rain = combined_pct >= RAIN_THRESHOLD

    # PWS ground truth override: if stations report rain NOW, it's raining
    if pws_is_raining:
        combined_pct = max(combined_pct, 90)
        will_rain = True

    if not will_rain:
        return _rain_result_none(pressure_fc)

    # --- 6. Timing from best available source ---
    # NWS hourly provides best timing; barometer as fallback
    nws_rain = nws_onset is not None and nws_peak_pct >= POP_ONSET_THRESHOLD
    if nws_onset:
        try:
            onset_dt = datetime.fromisoformat(nws_onset["start"])
            hours_until = max(0, (onset_dt - now).total_seconds() / 3600)
        except (ValueError, TypeError):
            hours_until = None
            onset_dt = None
    else:
        onset_dt = None
        hours_until = baro_timing_hours

    # If barometer suggests rain sooner than NWS, blend timing
    if nws_rain and baro_rain and baro_timing_hours and hours_until:
        if baro_timing_hours < hours_until:
            hours_until = (hours_until + baro_timing_hours) / 2

    # Confidence from source count and agreement
    n_sources_active = sum(1 for p, _ in sources if p >= RAIN_THRESHOLD)
    if n_sources_active >= 3:
        confidence = "high"
    elif n_sources_active >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    if pws_is_raining:
        confidence = "high"

    # Source agreement label
    if nws_rain and baro_rain:
        agreement = "both"
    elif nws_rain:
        agreement = "nws_only"
    elif baro_rain:
        agreement = "baro_only"
    else:
        agreement = "models_only"

    # Duration
    duration_hours = None
    if nws_onset_idx is not None and nws_end:
        try:
            end_dt = datetime.fromisoformat(nws_end["start"])
            start_dt = datetime.fromisoformat(nws_onset["start"])
            duration_hours = (end_dt - start_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

    # Format time strings
    rain_start_str = _format_rain_time(onset_dt or
                                        (now + timedelta(hours=hours_until) if hours_until else None))
    rain_end_str = None
    if nws_end:
        try:
            rain_end_str = _format_rain_time(datetime.fromisoformat(nws_end["start"]))
        except (ValueError, TypeError):
            pass

    summary = _build_summary(agreement, rain_start_str, hours_until,
                             combined_pct, confidence, duration_hours)

    # Build weight breakdown for debugging/display
    weight_breakdown = {s_name: {"pct": round(s_pct), "weight": s_wt}
                        for s_name, s_pct, s_wt in [
                            ("nbm", nbm_max_pop, 0.35),
                            ("nws", nws_peak_pct, 0.30),
                            ("open_meteo", round(model_agreement * 75 + 5) if models_total > 0 else 0, 0.20),
                            ("barometer", baro_rain_pct, 0.15),
                        ] if s_pct > 0 or s_name in ("nbm", "nws")}

    return {
        "will_rain": True,
        "rain_start": rain_start_str,
        "rain_start_dt": onset_dt.isoformat() if onset_dt else None,
        "hours_until": round(hours_until, 1) if hours_until is not None else None,
        "rain_end": rain_end_str,
        "duration_hours": round(duration_hours) if duration_hours else None,
        "peak_pct": nws_peak_pct,
        "combined_pct": combined_pct,
        "confidence": confidence,
        "summary": summary,
        "source_agreement": agreement,
        "sources": weight_breakdown,
        "models_agree": f"{models_wet}/{models_total}" if models_total > 0 else None,
        "nbm_max_pop": nbm_max_pop,
        "dewpoint_signal": dewpoint_approaching,
        "pws_raining": pws_is_raining,
    }


def _rain_result_none(pressure_fc: dict) -> dict:
    trend = pressure_fc.get("trend_label", "Steady")
    return {
        "will_rain": False,
        "rain_start": None,
        "rain_start_dt": None,
        "hours_until": None,
        "rain_end": None,
        "duration_hours": None,
        "peak_pct": 0,
        "combined_pct": 0,
        "confidence": "high",
        "summary": "No rain expected",
        "source_agreement": "neither",
    }


def _format_rain_time(dt) -> str | None:
    """Format a datetime to a compact time string like '6pm' or 'Tue 2am'."""
    if dt is None:
        return None
    try:
        # Convert to local time for display
        from zoneinfo import ZoneInfo
        local = dt.astimezone(ZoneInfo("America/Los_Angeles"))
        now_local = datetime.now(ZoneInfo("America/Los_Angeles"))

        time_str = local.strftime("%-I%p").lower()

        # If more than 18 hours away, prepend day name
        delta = (local - now_local).total_seconds() / 3600
        if delta > 18:
            time_str = local.strftime("%a ") + time_str
        return time_str
    except Exception:
        return None


def _build_summary(agreement: str, rain_start: str | None, hours_until: float | None,
                   combined_pct: int, confidence: str, duration_hours: int | None) -> str:
    """Build a compact one-line summary for display."""
    if not rain_start:
        if agreement == "baro_only":
            if hours_until and hours_until <= 6:
                return f"Rain possible within {int(hours_until)}h"
            return "Rain possible (pressure dropping)"
        return "No rain expected"

    # Confidence word
    if combined_pct >= 70:
        word = "likely"
    elif combined_pct >= 40:
        word = "possible"
    else:
        word = "slight chance"

    # Core message
    if hours_until is not None and hours_until < 1:
        msg = f"Rain {word} now"
    elif hours_until is not None and hours_until <= 2:
        msg = f"Rain {word} within {int(round(hours_until))}h"
    else:
        msg = f"Rain {word} by {rain_start}"

    # Add duration if known and non-trivial
    if duration_hours and duration_hours >= 2:
        msg += f" ({int(duration_hours)}h)"

    return msg


STORM_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".storm_state.json")


def get_storm_total(snotel_current: dict = None) -> dict:
    """Track cumulative snowfall during multi-day storms.

    Detects storm start/end from barometric pressure trend reversals:
    - Storm starts when pressure drops below 1013 hPa and is falling
    - Storm ends when pressure rises above 1013 hPa and has been rising for 6+ hours

    Accumulates snowfall from SNOTEL depth changes during the storm window.
    Returns storm total, duration, and current status.
    """
    # Load state
    try:
        with open(STORM_STATE_FILE) as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {"in_storm": False, "storm_start": None, "storm_start_depth": {},
                 "peak_snow": {}}

    history = _load_history()
    if not history:
        return {"in_storm": False, "storm_total_in": 0}

    current_p = history[-1]["hpa"]
    now = datetime.fromisoformat(history[-1]["t"])

    # Get recent pressure trend
    p_6h = _get_pressure_at_offset(history, 6.0)
    trend_6h = (current_p - p_6h) if p_6h is not None else 0

    # Storm detection thresholds
    STORM_PRESSURE = 1013.0  # hPa — below this suggests active weather
    RISING_THRESHOLD = 1.0   # hPa/6h — must be rising this much to end storm

    was_in_storm = state.get("in_storm", False)

    if not was_in_storm:
        # Check if storm is starting
        if current_p < STORM_PRESSURE and trend_6h is not None and trend_6h < -1.0:
            state["in_storm"] = True
            state["storm_start"] = now.isoformat()
            # Record current SNOTEL depths as baseline
            if snotel_current:
                state["storm_start_depth"] = {
                    n: d.get("snow_depth_in", 0) or 0
                    for n, d in snotel_current.items()
                    if "error" not in d and d.get("snow_depth_in") is not None
                }
            state["peak_snow"] = {}
    else:
        # Check if storm is ending
        if current_p > STORM_PRESSURE and trend_6h is not None and trend_6h > RISING_THRESHOLD:
            state["in_storm"] = False

    # Calculate storm total from SNOTEL depth changes
    storm_total = 0
    station_totals = {}
    if state.get("in_storm") and snotel_current and state.get("storm_start_depth"):
        for name, depth_start in state["storm_start_depth"].items():
            current_data = snotel_current.get(name, {})
            if "error" in current_data:
                continue
            current_depth = current_data.get("snow_depth_in", 0) or 0
            delta = max(0, current_depth - depth_start)
            station_totals[name] = round(delta, 1)
            # Track peak (depth can decrease from settling)
            prev_peak = state.get("peak_snow", {}).get(name, 0)
            if delta > prev_peak:
                state.setdefault("peak_snow", {})[name] = delta
            storm_total = max(storm_total, state.get("peak_snow", {}).get(name, 0))

    # Calculate storm duration
    duration_hours = None
    if state.get("in_storm") and state.get("storm_start"):
        try:
            start = datetime.fromisoformat(state["storm_start"])
            duration_hours = round((now - start).total_seconds() / 3600)
        except (ValueError, TypeError):
            pass

    # Save state (atomic write to prevent corruption)
    try:
        tmp_file = STORM_STATE_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(state, f)
        os.replace(tmp_file, STORM_STATE_FILE)
    except Exception:
        pass

    return {
        "in_storm": state.get("in_storm", False),
        "storm_start": state.get("storm_start"),
        "storm_total_in": round(storm_total, 1),
        "duration_hours": duration_hours,
        "station_totals": station_totals,
    }


if __name__ == "__main__":
    fc = get_forecast()
    print(json.dumps(fc, indent=2))
