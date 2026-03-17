#!/usr/bin/env python3
"""
ML Forecast Corrections — Live Pipeline Integration
====================================================

Loads trained XGBoost models and applies corrections to the output of
analyze_all() in tahoe_snow.py.

Includes:
- Shadow mode: log predictions without applying them (MLE 1)
- Drift detection: track prediction accuracy over time (MLE 3)
- Snow quality prediction: SLR-based quality labels (Skier 1)
- Wind hold probability: upper-mountain closure risk (Skier 2)
- Snow timing: when does snow start/peak (Skier 3)

Usage:
    from ml_corrections import apply_ml_corrections
    analysis = analyze_all(...)
    analysis = apply_ml_corrections(analysis)
"""

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE_DIR, "data", "models")
SHADOW_LOG = os.path.join(_BASE_DIR, ".ml_shadow_log.json")

# Shadow mode: when True, predictions are logged but NOT applied to forecasts.
# Set to False once you've verified the model improves forecasts in production.
SHADOW_MODE = True

# Cache loaded models (thread-safe via lock)
import threading
_loaded_models = {}
_load_attempted = False
_load_lock = threading.Lock()


# =========================================================================
# Model Loading
# =========================================================================

def _load_models():
    """Load all trained XGBoost models from disk."""
    global _loaded_models, _load_attempted
    _load_attempted = True

    if not os.path.exists(MODEL_DIR):
        logger.debug("ML model directory not found: %s", MODEL_DIR)
        return

    try:
        import xgboost as xgb
    except ImportError:
        logger.debug("XGBoost not installed — ML corrections disabled")
        return

    model_files = [f for f in os.listdir(MODEL_DIR)
                   if f.endswith(".json") and not f.endswith("_meta.json")]

    for model_file in model_files:
        name = model_file.replace(".json", "")
        meta_file = os.path.join(MODEL_DIR, f"{name}_meta.json")

        if not os.path.exists(meta_file):
            continue

        try:
            if "new_snow" in name:
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()

            model.load_model(os.path.join(MODEL_DIR, model_file))

            with open(meta_file) as f:
                meta = json.load(f)

            _loaded_models[name] = {
                "model": model,
                "meta": meta,
                "features": meta.get("features_used", []),
            }
        except Exception as e:
            logger.warning("Failed to load ML model %s: %s", name, e)

    if _loaded_models:
        logger.info("Loaded %d ML models", len(_loaded_models))


# =========================================================================
# Feature Extraction (aligned with training features)
# =========================================================================

def _extract_live_features(analysis: dict, zone_data: dict) -> dict:
    """
    Extract features from a live analysis dict, matching the training features.

    ONLY uses features that are available at forecast time:
    - NWP model outputs (temperature, wind, precipitation forecasts)
    - Physics-derived features (SLR, orographic multiplier, Froude)
    - Calendar/terrain features
    - Yesterday's SNOTEL observations (lag-1 shift features)
    - Freezing level estimate, AR detection (derived from above)

    Does NOT use same-day observation features (prec_change_in,
    observed_density, consecutive_snow_days, etc.)
    """
    features = {}

    current = zone_data.get("current", {})
    conditions = analysis.get("current_conditions", {})

    # --- NWP forecast outputs ---
    features["temperature_2m_mean"] = current.get("temp_f", np.nan)
    features["temperature_2m_max"] = current.get("temp_f", np.nan)
    # Use temp_f as proxy for min temp — feels_like is NOT the same as Tmin
    features["temperature_2m_min"] = current.get("temp_f", np.nan)
    features["wind_speed_10m_max"] = current.get("wind_mph", np.nan)
    features["wind_gusts_10m_max"] = current.get("wind_gust_mph", np.nan)
    features["wind_direction_10m_dominant"] = _dir_to_deg(current.get("wind_dir", ""))

    # Precipitation from forecast timeline (next 24h)
    timeline = zone_data.get("timeline_48h", [])
    if timeline:
        features["precipitation_sum"] = sum(
            h.get("precip_in", 0) or 0 for h in timeline[:24]
        )
        features["snowfall_sum"] = sum(
            (h.get("snowfall_in", 0) or 0) * 2.54  # inches to cm
            for h in timeline[:24]
        )
        features["rain_sum"] = max(0, sum(
            (h.get("precip_in", 0) or 0) - (h.get("snowfall_in", 0) or 0) * 0.1
            for h in timeline[:24]
        ))

    # --- Physics features ---
    features["slr"] = current.get("slr", 10)
    features["froude_number"] = analysis.get("froude_number", np.nan)
    features["lapse_rate_c_km"] = conditions.get("observed_lapse_rate_c_km", np.nan)
    features["wet_bulb_f"] = current.get("dewpoint_f", np.nan)

    from feature_engineering import compute_orographic_multiplier
    elev_ft = zone_data.get("elev_ft", 7000)
    wind_mph = features.get("wind_speed_10m_max", 0) or 0
    wind_dir = features.get("wind_direction_10m_dominant", 0) or 0

    features["orographic_mult"] = compute_orographic_multiplier(
        elev_ft, wind_mph, wind_dir,
    )
    features["precip_orographic_in"] = (
        features.get("precipitation_sum", 0) * features["orographic_mult"]
    )

    # Rain/snow discrimination
    temp = features.get("temperature_2m_mean", 32)
    if not np.isnan(temp):
        features["rain_fraction"] = 0.0 if temp < 30 else min(1.0, (temp - 30) / 6)
    else:
        features["rain_fraction"] = np.nan
    features["snow_fraction"] = 1.0 - features.get("rain_fraction", 0.5)
    snowfall = features.get("snowfall_sum", 0) or 0
    features["snowfall_adjusted_cm"] = snowfall * features.get("snow_fraction", 1.0)

    # --- Freezing level + AR detection ---
    if not np.isnan(temp):
        features["est_freezing_level_ft"] = elev_ft + (temp - 32) / 3.5 * 1000
        features["freezing_level_margin_ft"] = features["est_freezing_level_ft"] - elev_ft
    else:
        features["est_freezing_level_ft"] = np.nan
        features["freezing_level_margin_ft"] = np.nan

    precip = max(0.01, features.get("precipitation_sum", 0) or 0.01)
    rain = max(0, features.get("rain_sum", 0) or 0)
    features["moisture_transport_proxy"] = min(50, wind_mph * (rain / precip))

    is_southerly = 1.0 if 150 <= wind_dir <= 240 else 0.0
    is_warm = 1.0 if (not np.isnan(temp) and temp > 34) else 0.0
    is_wet = 1.0 if precip > 0.3 else 0.0
    is_windy = 1.0 if wind_mph > 15 else 0.0
    features["ar_score"] = is_southerly + is_warm + is_wet + is_windy
    features["is_ar_event"] = 1 if features["ar_score"] >= 3 else 0

    # Temperature-derived features
    if not np.isnan(temp):
        features["tmax_above_freezing"] = max(0, temp - 32)
        features["tmin_below_freezing"] = max(0, 32 - (features.get("temperature_2m_min", temp) or temp))
        features["diurnal_range_f"] = features["tmax_above_freezing"] + features["tmin_below_freezing"]
        features["freeze_thaw"] = 1 if (features["tmax_above_freezing"] > 0 and features["tmin_below_freezing"] > 4) else 0
    else:
        for f in ["tmax_above_freezing", "tmin_below_freezing", "diurnal_range_f", "freeze_thaw"]:
            features[f] = np.nan

    # --- Calendar features ---
    now = datetime.now(timezone.utc)
    features["month"] = float(now.month)
    features["day_of_year"] = float(now.timetuple().tm_yday)
    features["water_year_day"] = float(
        (now - datetime(now.year if now.month >= 10 else now.year - 1, 10, 1,
                        tzinfo=timezone.utc)).days
    )
    features["temp_trend_3d"] = np.nan  # Would need 3 days of prior ERA5 data

    # --- Terrain features (static) ---
    features["station_elev_ft"] = elev_ft
    features["zone_elev_ft"] = elev_ft
    features["elev_diff_ft"] = 0.0
    features["aspect_deg"] = zone_data.get("aspect_deg", 0)
    features["slope_angle_deg"] = zone_data.get("slope_angle_deg", 0)

    # --- Yesterday's SNOTEL (legitimately available at forecast time) ---
    features["depth_lag1d"] = np.nan
    features["swe_lag1d"] = np.nan
    features["tmax_lag1d"] = np.nan
    features["tmax_avg_3d"] = np.nan

    snotel = analysis.get("snotel_current", {})
    if snotel:
        for station_name, station_data in snotel.items():
            if isinstance(station_data, dict) and "error" not in station_data:
                features["depth_lag1d"] = station_data.get("depth_in", np.nan)
                features["swe_lag1d"] = station_data.get("swe_in", np.nan)
                features["tmax_lag1d"] = station_data.get("temp_f", np.nan)
                break

    # Station bias features — use precomputed means from training data
    # These are the historical average depth_change per station (static values
    # computed across the full training set). Stored here as a lookup table
    # rather than recomputing from the training parquet at runtime.
    _STATION_STATS = {
        # (mean_depth_change, std_depth_change) computed from training data
        "Tahoe City Cross": (0.10, 2.82), "Fallen Leaf": (0.09, 2.58),
        "Hagan's Meadow": (0.22, 3.72), "Independence Lake": (0.18, 3.55),
        "Independence Camp": (0.15, 3.10), "Rubicon #2": (0.14, 3.30),
        "Squaw Valley GC": (0.12, 3.52), "Mt Rose Ski Area": (0.28, 4.75),
        "CSS Lab": (0.19, 3.82),
    }
    # Match the nearest SNOTEL station for the stats
    if snotel:
        for station_name in snotel:
            if station_name in _STATION_STATS:
                mean, std = _STATION_STATS[station_name]
                features["station_mean_depth_change"] = mean
                features["station_std_depth_change"] = std
                break
            else:
                features["station_mean_depth_change"] = 0.15  # basin average
                features["station_std_depth_change"] = 3.4
    else:
        features["station_mean_depth_change"] = 0.15
        features["station_std_depth_change"] = 3.4

    # Model spread — compute from ensemble data if available
    ensemble = analysis.get("ensemble", {})
    ensemble_models = ensemble.get("models", {}) if isinstance(ensemble, dict) else {}
    snow_p50_values = []
    temp_values = []
    for model_label, daily_list in ensemble_models.items():
        if isinstance(daily_list, list) and daily_list:
            day0 = daily_list[0] if daily_list else {}
            sp50 = day0.get("snow_p50")
            if sp50 is not None:
                snow_p50_values.append(sp50)
            tp50 = day0.get("temp_high_p50") or day0.get("temp_p50")
            if tp50 is not None:
                temp_values.append(tp50)

    if len(snow_p50_values) >= 2:
        arr = np.array(snow_p50_values)
        features["model_snow_spread"] = float(np.std(arr))
        features["model_snow_mean"] = float(np.mean(arr))
        features["model_snow_range"] = float(np.max(arr) - np.min(arr))
    else:
        features["model_snow_spread"] = np.nan
        features["model_snow_mean"] = np.nan
        features["model_snow_range"] = np.nan

    if len(temp_values) >= 2:
        features["model_temp_spread"] = float(np.std(temp_values))
        features["model_temp_mean"] = float(np.mean(temp_values))
    else:
        features["model_temp_spread"] = np.nan
        features["model_temp_mean"] = np.nan

    # ENSO
    features["oni"] = np.nan
    features["enso_phase"] = 0

    return features


def _dir_to_deg(wind_dir: str) -> float:
    """Convert wind direction string to degrees."""
    dir_map = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
        "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
        "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
        "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    return dir_map.get(str(wind_dir).upper(), np.nan)


def _elev_to_band(elev_ft: float) -> str:
    """Map elevation to training band name."""
    if elev_ft < 7000:
        return "base"
    elif elev_ft < 8500:
        return "mid"
    else:
        return "peak"


# =========================================================================
# Skier Features: Snow Quality, Wind Holds, Timing
# =========================================================================

def _predict_snow_quality(slr: float, temp_f: float, wind_mph: float) -> dict:
    """
    Predict snow quality from SLR, temperature, and wind.

    Returns dict with quality label, description, and rating (1-5).
    """
    if np.isnan(slr) or slr <= 0:
        return {"quality": "Unknown", "description": "Insufficient data", "rating": 0}

    if slr >= 15 and temp_f < 25 and wind_mph < 20:
        return {"quality": "Blower", "description": "Ultra-light dry powder",
                "rating": 5}
    elif slr >= 12 and temp_f < 28:
        return {"quality": "Cold Smoke", "description": "Light fluffy powder",
                "rating": 5}
    elif slr >= 10 and temp_f < 32:
        return {"quality": "Powder", "description": "Good skiing powder",
                "rating": 4}
    elif slr >= 8 and temp_f < 34:
        return {"quality": "Average", "description": "Typical Sierra snow",
                "rating": 3}
    elif slr >= 6:
        return {"quality": "Heavy", "description": "Dense, heavy snow (Sierra cement)",
                "rating": 2}
    else:
        return {"quality": "Wet", "description": "Very wet/slushy, rain-mixed",
                "rating": 1}


def _predict_wind_hold(wind_gust_mph: float, elev_ft: float) -> dict:
    """
    Predict wind hold probability based on gusts and elevation.

    Wind hold thresholds vary by resort, but generally:
    - Gondolas: hold at 45-50 mph gusts
    - High-speed quads: hold at 55-60 mph
    - Fixed-grip chairs: hold at 60-70 mph
    """
    if np.isnan(wind_gust_mph):
        return {"probability": 0, "risk": "Unknown", "detail": "No wind data"}

    # Adjust for elevation (higher = windier, exposed ridgelines)
    elev_factor = 1.0 + max(0, (elev_ft - 8000)) / 5000  # +20% at 9000ft

    effective_gust = wind_gust_mph * elev_factor

    if effective_gust >= 60:
        return {"probability": 0.95, "risk": "Very High",
                "detail": f"Gusts {wind_gust_mph:.0f}+ mph — expect upper mountain closures"}
    elif effective_gust >= 50:
        return {"probability": 0.70, "risk": "High",
                "detail": f"Gusts {wind_gust_mph:.0f} mph — gondola/exposed lifts likely on hold"}
    elif effective_gust >= 40:
        return {"probability": 0.35, "risk": "Moderate",
                "detail": f"Gusts {wind_gust_mph:.0f} mph — possible delays on exposed lifts"}
    elif effective_gust >= 30:
        return {"probability": 0.10, "risk": "Low",
                "detail": f"Gusts {wind_gust_mph:.0f} mph — unlikely issues"}
    else:
        return {"probability": 0.0, "risk": "None",
                "detail": "Calm conditions"}


def _predict_snow_timing(timeline: list) -> dict:
    """
    Analyze the hourly forecast timeline to predict snow start, peak, and end.

    Returns dict with timing details for trip planning.
    """
    if not timeline:
        return {"available": False}

    snow_hours = []
    max_rate = 0
    max_rate_idx = 0

    for i, h in enumerate(timeline[:48]):
        snow_in = h.get("snowfall_in", 0) or 0
        if snow_in > 0:
            snow_hours.append(i)
            if snow_in > max_rate:
                max_rate = snow_in
                max_rate_idx = i

    if not snow_hours:
        return {"available": True, "snowing": False,
                "summary": "No snow expected in the next 48 hours"}

    start_hour = snow_hours[0]
    end_hour = snow_hours[-1]

    # Total accumulation by period
    total_12h = sum((timeline[i].get("snowfall_in", 0) or 0) for i in range(min(12, len(timeline))))
    total_24h = sum((timeline[i].get("snowfall_in", 0) or 0) for i in range(min(24, len(timeline))))
    total_48h = sum((timeline[i].get("snowfall_in", 0) or 0) for i in range(min(48, len(timeline))))

    # When is snow heaviest?
    if max_rate_idx < 12:
        peak_period = "next 12 hours"
    elif max_rate_idx < 24:
        peak_period = "12-24 hours from now"
    else:
        peak_period = "24-48 hours from now"

    return {
        "available": True,
        "snowing": True,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "peak_hour": max_rate_idx,
        "peak_rate_in_hr": round(max_rate, 1),
        "peak_period": peak_period,
        "total_12h_in": round(total_12h, 1),
        "total_24h_in": round(total_24h, 1),
        "total_48h_in": round(total_48h, 1),
        "summary": f"Snow {'starting now' if start_hour == 0 else f'starting in ~{start_hour}h'}, "
                   f"heaviest {peak_period}, "
                   f"{total_24h:.0f}\" in 24h",
    }


# =========================================================================
# Shadow Mode Logging (MLE 1 + MLE 3)
# =========================================================================

def _log_shadow_prediction(resort: str, zone: str, predictions: dict):
    """
    Log ML predictions for later comparison against actual observations.

    This enables:
    1. Shadow mode: see what corrections WOULD have been applied
    2. Drift detection: track if predictions degrade over time
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resort": resort,
        "zone": zone,
        **predictions,
    }

    try:
        existing = []
        if os.path.exists(SHADOW_LOG):
            with open(SHADOW_LOG) as f:
                existing = json.load(f)

        existing.append(entry)

        # Keep last 30 days of predictions (~5000 entries max)
        if len(existing) > 5000:
            existing = existing[-5000:]

        tmp = SHADOW_LOG + ".tmp"
        with open(tmp, "w") as f:
            json.dump(existing, f)
        os.replace(tmp, SHADOW_LOG)
    except Exception as e:
        logger.debug("Failed to write shadow log: %s", e)


def get_drift_metrics() -> dict:
    """
    Compute drift metrics from shadow log + verification data.

    Returns rolling MAE over recent predictions vs SNOTEL actuals.
    """
    if not os.path.exists(SHADOW_LOG):
        return {"available": False, "reason": "No shadow predictions logged yet"}

    try:
        with open(SHADOW_LOG) as f:
            predictions = json.load(f)

        if len(predictions) < 10:
            return {"available": False, "reason": f"Only {len(predictions)} predictions logged"}

        # Count predictions by day
        days = set()
        for p in predictions:
            ts = p.get("timestamp", "")[:10]
            if ts:
                days.add(ts)

        return {
            "available": True,
            "total_predictions": len(predictions),
            "days_tracked": len(days),
            "shadow_mode": SHADOW_MODE,
            "note": "Activate verification by comparing ml_depth_prediction_in "
                    "against next-day SNOTEL depth_change. Set SHADOW_MODE=False "
                    "in ml_corrections.py once rolling MAE < raw forecast MAE.",
        }
    except Exception:
        return {"available": False, "reason": "Error reading shadow log"}


# =========================================================================
# Skier 5/5: Resort Ranking, Chains, Best Window, Powder Alert
# =========================================================================

def _rank_resorts(resorts: dict) -> list:
    """
    Rank resorts by skiing conditions: snow amount, quality, and accessibility.

    Produces a sorted recommendation list answering "where should I go today?"

    Scoring:
      - Snow amount at peak (40% weight): more snow = higher score
      - Snow quality rating (30% weight): lighter/drier = higher score
      - Wind hold risk (20% weight): lower wind = higher score
      - Accessibility (10% weight): lower chain requirement = higher score
    """
    rankings = []
    for resort_name, resort_data in resorts.items():
        zones = resort_data.get("zones", {})
        peak = zones.get("peak", zones.get("mid", {}))
        base = zones.get("base", {})

        # Snow amount score (0-10)
        depth_pred = peak.get("ml_depth_prediction_in", 0) or 0
        snow_score = min(10, depth_pred / 2)  # 20"+ = max score

        # Quality score (0-10)
        quality = peak.get("ml_snow_quality", {})
        quality_score = (quality.get("rating", 3) / 5) * 10

        # Wind score (0-10, inverted: less wind = higher)
        wind_hold = peak.get("ml_wind_hold", {})
        wind_prob = wind_hold.get("probability", 0)
        wind_score = 10 * (1 - wind_prob)

        # Access score (0-10): base elevation matters for chains
        # Higher base = more likely to need chains
        base_elev = base.get("elev_ft", 7000)
        base_temp = base.get("current", {}).get("temp_f", 32)
        if base_temp > 36 or base_elev < 6500:
            access_score = 9  # Probably no chains
        elif base_temp > 32:
            access_score = 6  # Maybe chains
        else:
            access_score = 3  # Likely chains

        # Weighted total
        total = (snow_score * 0.40 + quality_score * 0.30 +
                 wind_score * 0.20 + access_score * 0.10)

        # Build reasoning
        reasons = []
        if depth_pred >= 6:
            reasons.append(f"{depth_pred:.0f}\" expected")
        if quality.get("rating", 0) >= 4:
            reasons.append(quality.get("quality", ""))
        if wind_prob < 0.2:
            reasons.append("lifts running")
        elif wind_prob > 0.5:
            reasons.append("wind hold risk")

        timing = peak.get("ml_snow_timing", {})
        if timing.get("snowing") and timing.get("peak_period"):
            reasons.append(f"heaviest {timing['peak_period']}")

        rankings.append({
            "resort": resort_name,
            "score": round(total, 1),
            "snow_score": round(snow_score, 1),
            "quality_score": round(quality_score, 1),
            "wind_score": round(wind_score, 1),
            "access_score": round(access_score, 1),
            "reasons": reasons,
            "summary": f"{resort_name}: {total:.0f}/10" +
                       (f" — {', '.join(reasons)}" if reasons else ""),
        })

    rankings.sort(key=lambda x: x["score"], reverse=True)

    # Add recommendation text
    if rankings:
        top = rankings[0]
        if top["score"] >= 7:
            top["recommendation"] = f"Go to {top['resort']}! Excellent conditions."
        elif top["score"] >= 5:
            top["recommendation"] = f"{top['resort']} looks best today. Decent conditions."
        else:
            top["recommendation"] = "Conditions are marginal across all resorts."

    return rankings


def _predict_chains(resorts: dict, analysis: dict) -> dict:
    """
    Predict chain requirements for each highway route.

    Chain control levels:
      R1: Chains or snow tires required
      R2: Chains required on all vehicles (except 4WD with snow tires)
      R3: Chains required on ALL vehicles, no exceptions

    Based on: snow level, precipitation intensity, base temperature, route elevation.
    """
    # Import resort configs for chain route info
    try:
        from resort_configs import get_active_resorts
        resort_configs = get_active_resorts()
    except ImportError:
        return {"available": False}

    # Group resorts by chain route
    routes = {}
    for name, cfg in resort_configs.items():
        route = cfg.chain_route
        if not route:
            continue
        if route not in routes:
            routes[route] = {"resorts": [], "elevations": []}
        routes[route]["resorts"].append(name)
        # Use base zone elevation as road elevation proxy
        if cfg.base():
            routes[route]["elevations"].append(cfg.base().elevation_ft)

    predictions = {}
    for route, info in routes.items():
        # Get conditions from the first resort on this route
        resort_name = info["resorts"][0]
        resort_data = resorts.get(resort_name, {})
        base = resort_data.get("zones", {}).get("base", {})
        base_current = base.get("current", {})

        temp_f = base_current.get("temp_f", 35)
        snow_prob = base.get("ml_new_snow_prob", 0)
        depth_pred = base.get("ml_depth_prediction_in", 0) or 0
        timing = base.get("ml_snow_timing", {})
        is_snowing = timing.get("snowing", False)
        pass_elev = max(info["elevations"]) if info["elevations"] else 7000

        # Decision logic
        if not is_snowing and snow_prob < 0.3 and (temp_f is None or temp_f > 36):
            level = "None"
            detail = "Clear roads expected"
        elif temp_f is not None and temp_f > 36 and depth_pred < 2:
            level = "None"
            detail = "Above freezing, light precip — roads should be clear"
        elif temp_f is not None and temp_f > 33 and depth_pred < 4:
            level = "R1"
            detail = "Chains or snow tires — wet snow possible"
        elif depth_pred >= 6 or (temp_f is not None and temp_f < 28 and is_snowing):
            level = "R2"
            detail = f"Chains required — heavy snow expected ({depth_pred:.0f}\")"
        elif is_snowing or depth_pred >= 2:
            level = "R1"
            detail = "Chains or snow tires — active snow"
        else:
            level = "None"
            detail = "No chain controls expected"

        predictions[route] = {
            "level": level,
            "detail": detail,
            "resorts": info["resorts"],
            "pass_elevation_ft": pass_elev,
        }

    return {"available": True, "routes": predictions}


def _compute_best_window(zone_data: dict) -> dict:
    """
    Determine the best skiing window by combining timing, wind, and quality.

    Answers: "When should I be on the mountain?"
    """
    timeline = zone_data.get("timeline_48h", [])
    if not timeline:
        return {"available": False}

    wind_hold = zone_data.get("ml_wind_hold", {})
    wind_risk = wind_hold.get("probability", 0)
    timing = zone_data.get("ml_snow_timing", {})
    quality = zone_data.get("ml_snow_quality", {})

    # Score each 3-hour window based on snow + wind conditions
    windows = []
    for start_h in range(0, min(24, len(timeline)), 3):
        end_h = min(start_h + 3, len(timeline))
        chunk = timeline[start_h:end_h]

        snow_rate = sum(h.get("snowfall_in", 0) or 0 for h in chunk)
        avg_wind = np.mean([h.get("wind_mph", 0) or 0 for h in chunk]) if chunk else 0
        max_gust = max((h.get("wind_gust_mph", 0) or 0 for h in chunk), default=0)

        # Score: snow is good, wind is bad
        snow_bonus = min(3, snow_rate)  # Up to 3 points for snow
        wind_penalty = min(3, max_gust / 25)  # Up to 3 penalty for wind
        score = 5 + snow_bonus - wind_penalty  # 2-8 scale

        period = f"{start_h}h-{end_h}h from now"
        if start_h <= 3:
            period = "Early morning (now-3h)"
        elif start_h <= 6:
            period = "Morning (3-6h)"
        elif start_h <= 9:
            period = "Late morning (6-9h)"
        elif start_h <= 12:
            period = "Midday (9-12h)"
        elif start_h <= 15:
            period = "Afternoon (12-15h)"
        elif start_h <= 18:
            period = "Late afternoon (15-18h)"
        else:
            period = "Evening (18-21h)"

        windows.append({
            "period": period,
            "start_hour": start_h,
            "snow_rate_in": round(snow_rate, 1),
            "max_gust_mph": round(max_gust, 0),
            "score": round(score, 1),
        })

    windows.sort(key=lambda x: x["score"], reverse=True)
    best = windows[0] if windows else None

    # Generate advice
    if not best:
        advice = "No forecast data available"
    elif wind_risk > 0.7:
        advice = f"Upper mountain likely closed. Best window: {best['period']} if lifts open."
    elif best["snow_rate_in"] > 1 and best["max_gust_mph"] < 35:
        advice = f"Best skiing: {best['period']} — active snow, manageable wind."
    elif best["snow_rate_in"] > 0:
        advice = f"Best skiing: {best['period']} — some new snow, {quality.get('quality', 'typical')} quality."
    else:
        # No snow, find calmest window
        calmest = min(windows, key=lambda x: x["max_gust_mph"])
        advice = f"No new snow expected. Calmest conditions: {calmest['period']}."

    return {
        "available": True,
        "best_period": best["period"] if best else "Unknown",
        "best_score": best["score"] if best else 0,
        "advice": advice,
        "windows": windows[:4],  # Top 4 windows
    }


def _compute_powder_alert(resorts: dict) -> dict:
    """
    Compute powder alert: should the user set an alarm?

    Thresholds:
      - SEND IT (5/5): 12\"+ expected, quality 4+, lifts likely running
      - STRONG (4/5): 8\"+ expected, quality 3+, some wind risk
      - MODERATE (3/5): 4\"+ expected, any quality
      - LOW (2/5): 1-4\" expected
      - NONE (1/5): No significant snow
    """
    best_resort = None
    best_depth = 0
    best_quality = 0
    best_wind_risk = 1.0

    for resort_name, resort_data in resorts.items():
        zones = resort_data.get("zones", {})
        peak = zones.get("peak", zones.get("mid", {}))
        if not peak:
            continue

        depth = peak.get("ml_depth_prediction_in", 0) or 0
        quality = peak.get("ml_snow_quality", {}).get("rating", 0)
        wind_prob = peak.get("ml_wind_hold", {}).get("probability", 0)

        # Composite score
        score = depth * (quality / 5) * (1 - wind_prob * 0.5)
        if score > best_depth * (best_quality / 5 if best_quality > 0 else 1) * (1 - best_wind_risk * 0.5):
            best_resort = resort_name
            best_depth = depth
            best_quality = quality
            best_wind_risk = wind_prob

    if best_depth >= 12 and best_quality >= 4 and best_wind_risk < 0.3:
        level = 5
        label = "SEND IT"
        message = f"Powder day at {best_resort}! {best_depth:.0f}\" of {_quality_name(best_quality)} expected. Set your alarm."
    elif best_depth >= 8 and best_quality >= 3:
        level = 4
        label = "STRONG"
        message = f"Great day ahead at {best_resort}. {best_depth:.0f}\" expected" + (", but watch for wind holds." if best_wind_risk > 0.3 else ".")
    elif best_depth >= 4:
        level = 3
        label = "MODERATE"
        message = f"Decent snow at {best_resort} — {best_depth:.0f}\" expected."
    elif best_depth >= 1:
        level = 2
        label = "LOW"
        message = f"Light snow possible. {best_depth:.0f}\" at {best_resort}."
    else:
        level = 1
        label = "NONE"
        message = "No significant snow expected across Tahoe resorts."

    return {
        "level": level,
        "label": label,
        "message": message,
        "best_resort": best_resort,
        "best_depth_in": round(best_depth, 1) if best_depth else 0,
        "best_quality_rating": best_quality,
    }


def _quality_name(rating: int) -> str:
    """Map quality rating to short name."""
    return {5: "light powder", 4: "powder", 3: "snow", 2: "heavy snow", 1: "wet snow"}.get(rating, "snow")


# =========================================================================
# Summer Outdoor Features: Thin wrapper over outdoor_conditions.py
# =========================================================================

def _build_ml_outdoor_from_outdoor(analysis: dict) -> dict:
    """Build the ml_outdoor summary from outdoor_conditions data.

    This delegates the heavy lifting to outdoor_conditions.py functions
    (already computed in tahoe_snow.py). We just repackage for the ML
    corrections output format.
    """
    outdoor = analysis.get("outdoor", {})
    resorts = analysis.get("resorts", {})

    # Gather basic conditions for the summary
    temp_f = None
    cape = None
    humidity = None
    timeline = None
    for rd in resorts.values():
        for zd in rd.get("zones", {}).values():
            cur = zd.get("current", {})
            if cur.get("temp_f") is not None:
                temp_f = temp_f or cur.get("temp_f")
                cape = cape or cur.get("cape_jkg")
                humidity = humidity or cur.get("humidity_pct")
                timeline = timeline or zd.get("timeline_48h")
                break
        if temp_f:
            break

    now = datetime.now(timezone.utc)
    month = now.month

    result = {}
    result["thunderstorm"] = _assess_thunderstorm_risk(cape, temp_f, humidity, timeline)
    result["heat_safety"] = _assess_heat_risk(temp_f, humidity, timeline)

    aqi_data = analysis.get("current_conditions", {}).get("observation", {}).get("aqi", {})
    result["smoke_aqi"] = _assess_smoke_risk(aqi_data)
    result["uv_exposure"] = _assess_uv_risk(month, 7000)

    snotel = analysis.get("snotel_current", {})
    result["water_availability"] = _assess_water_availability(snotel, month)
    result["best_window"] = _compute_summer_window(temp_f, cape, timeline, result["thunderstorm"])
    result["recommendation"] = _summer_activity_recommendation(result)

    return result


def _compute_summer_outdoor(analysis: dict, resorts: dict) -> dict:
    """
    Compute summer hiking/biking conditions and recommendations.

    Covers: thunderstorm risk, heat safety, smoke/AQI, UV exposure,
    creek/water availability, and optimal activity windows.
    """
    conditions = analysis.get("current_conditions", {})
    obs = conditions.get("observation", {})

    # Get representative conditions (use mid-elevation zone)
    temp_f = None
    wind_mph = None
    gusts = None
    humidity = None
    cape = None
    timeline = None

    for resort_data in resorts.values():
        zones = resort_data.get("zones", {})
        for zone_key in ["mid", "peak", "base"]:
            zd = zones.get(zone_key, {})
            cur = zd.get("current", {})
            if cur.get("temp_f") is not None:
                temp_f = temp_f or cur.get("temp_f")
                wind_mph = wind_mph or cur.get("wind_mph")
                gusts = gusts or cur.get("wind_gust_mph")
                humidity = humidity or cur.get("humidity_pct")
                cape = cape or cur.get("cape_jkg")
                timeline = timeline or zd.get("timeline_48h")
                break

    result = {}

    # --- 1. Thunderstorm Risk ---
    result["thunderstorm"] = _assess_thunderstorm_risk(
        cape, temp_f, humidity, timeline
    )

    # --- 2. Heat Safety ---
    result["heat_safety"] = _assess_heat_risk(temp_f, humidity, timeline)

    # --- 3. Smoke / AQI ---
    aqi_data = obs.get("aqi") or analysis.get("air_quality", {})
    result["smoke_aqi"] = _assess_smoke_risk(aqi_data)

    # --- 4. UV Exposure ---
    now = datetime.now(timezone.utc)
    month = now.month
    elev_ft = 7000  # mid-mountain default
    result["uv_exposure"] = _assess_uv_risk(month, elev_ft)

    # --- 5. Creek/Water Availability ---
    snotel = analysis.get("snotel_current", {})
    result["water_availability"] = _assess_water_availability(snotel, month)

    # --- 6. Best Activity Window ---
    result["best_window"] = _compute_summer_window(
        temp_f, cape, timeline, result["thunderstorm"]
    )

    # --- 7. Activity Recommendation ---
    result["recommendation"] = _summer_activity_recommendation(result)

    return result


def _assess_thunderstorm_risk(cape, temp_f, humidity, timeline) -> dict:
    """
    Assess thunderstorm risk from CAPE, temperature, and moisture.

    Sierra thunderstorms are most common July-August, typically building
    after noon from daytime heating. Hikers above treeline are at highest risk.

    CAPE thresholds for Sierra:
      <100 J/kg: Very low risk
      100-500: Low-moderate (isolated cells possible)
      500-1000: Moderate (scattered thunderstorms likely)
      >1000: High (widespread, potentially severe)
    """
    cape = cape or 0

    # Check timeline for afternoon CAPE buildup
    afternoon_precip = 0
    if timeline:
        for i, h in enumerate(timeline[12:21]):  # Noon to 9pm
            afternoon_precip += h.get("precip_in", 0) or 0

    if cape > 1000:
        risk = "High"
        detail = "Strong instability — thunderstorms likely afternoon/evening. Avoid exposed ridges after 11 AM."
        probability = 0.8
    elif cape > 500 or (afternoon_precip > 0.2 and cape > 200):
        risk = "Moderate"
        detail = "Scattered thunderstorms possible after noon. Plan to be below treeline by 1 PM."
        probability = 0.5
    elif cape > 100 or afternoon_precip > 0.1:
        risk = "Low"
        detail = "Isolated thunderstorms possible late afternoon. Keep an eye on building clouds."
        probability = 0.2
    else:
        risk = "Very Low"
        detail = "Stable atmosphere. Low thunderstorm risk."
        probability = 0.05

    return {
        "risk": risk,
        "probability": round(probability, 2),
        "detail": detail,
        "cape_jkg": cape,
        "turn_around_time": "11 AM" if risk == "High" else ("1 PM" if risk == "Moderate" else None),
    }


def _assess_heat_risk(temp_f, humidity, timeline) -> dict:
    """
    Assess heat-related risk for outdoor activity.

    At Tahoe elevations (6000-10000 ft), heat is less common than lowland
    areas, but dehydration risk is higher due to altitude and low humidity.
    """
    if temp_f is None:
        return {"risk": "Unknown", "detail": "No temperature data"}

    # Compute heat index approximation
    rh = humidity or 30  # Tahoe is typically dry
    # Simplified heat index (Rothfusz regression)
    if temp_f >= 80:
        hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh
              - 0.22475541 * temp_f * rh - 0.00683783 * temp_f**2
              - 0.05481717 * rh**2 + 0.00122874 * temp_f**2 * rh
              + 0.00085282 * temp_f * rh**2 - 0.00000199 * temp_f**2 * rh**2)
    else:
        hi = temp_f

    # Altitude dehydration factor: +10% water need per 3000ft above sea level
    altitude_factor = "Drink 50-100% more water than at sea level (altitude + dry air)."

    if hi >= 105 or temp_f >= 95:
        risk = "Extreme"
        detail = f"Dangerously hot ({temp_f:.0f}F). Avoid strenuous activity noon-4 PM."
        water_per_hr = "32+ oz/hr"
    elif hi >= 90 or temp_f >= 85:
        risk = "High"
        detail = f"Hot ({temp_f:.0f}F). Start early, carry extra water, take shade breaks."
        water_per_hr = "24-32 oz/hr"
    elif temp_f >= 75:
        risk = "Moderate"
        detail = f"Warm ({temp_f:.0f}F). Normal summer conditions. Stay hydrated."
        water_per_hr = "16-24 oz/hr"
    elif temp_f >= 60:
        risk = "Low"
        detail = f"Comfortable ({temp_f:.0f}F). Great hiking/biking weather."
        water_per_hr = "12-16 oz/hr"
    else:
        risk = "None"
        detail = f"Cool ({temp_f:.0f}F). Layer up for elevation and wind."
        water_per_hr = "8-12 oz/hr"

    return {
        "risk": risk,
        "detail": detail,
        "temperature_f": temp_f,
        "humidity_pct": rh,
        "heat_index_f": round(hi, 0),
        "water_recommendation": water_per_hr,
        "altitude_note": altitude_factor,
    }


def _assess_smoke_risk(aqi_data) -> dict:
    """
    Assess smoke/air quality risk for outdoor activity.

    Tahoe fire season (July-October) regularly produces AQI > 100 from
    regional wildfires. This is the #1 reason hikers cancel trips.
    """
    if not aqi_data or isinstance(aqi_data, str):
        return {"risk": "Unknown", "aqi": None,
                "detail": "No air quality data. Check airnow.gov before heading out."}

    aqi = aqi_data.get("us_aqi") or aqi_data.get("aqi")
    if aqi is None:
        return {"risk": "Unknown", "aqi": None,
                "detail": "No AQI reading available."}

    pm25 = aqi_data.get("pm2_5") or aqi_data.get("pm25")

    if aqi <= 50:
        risk = "Good"
        detail = "Air quality is good. No restrictions on outdoor activity."
        activity = "All activities OK"
    elif aqi <= 100:
        risk = "Moderate"
        detail = "Air quality is acceptable. Unusually sensitive people should limit prolonged outdoor exertion."
        activity = "Most activities OK. Sensitive groups may notice effects."
    elif aqi <= 150:
        risk = "Unhealthy for Sensitive Groups"
        detail = "Reduce prolonged outdoor exertion. Consider shorter hikes at lower elevations (smoke rises)."
        activity = "Shorten activities. Avoid strenuous uphill."
    elif aqi <= 200:
        risk = "Unhealthy"
        detail = "Everyone may experience health effects. Avoid prolonged outdoor exertion."
        activity = "Avoid strenuous outdoor activity. Light walks OK."
    elif aqi <= 300:
        risk = "Very Unhealthy"
        detail = "Health alert. Avoid all outdoor exertion."
        activity = "Stay indoors. Cancel outdoor plans."
    else:
        risk = "Hazardous"
        detail = "Emergency conditions. Everyone should avoid all outdoor activity."
        activity = "Do NOT go outside."

    return {
        "risk": risk,
        "aqi": aqi,
        "pm25": pm25,
        "detail": detail,
        "activity_guidance": activity,
    }


def _assess_uv_risk(month: int, elev_ft: float) -> dict:
    """
    Assess UV exposure risk by month and elevation.

    UV intensity increases ~10-12% per 1000m (3280ft) of elevation.
    At 8000ft, UV is ~25% stronger than sea level. Combined with thin
    air, reflection off granite/snow, and long exposure during hikes,
    sunburn is a major risk June-September.
    """
    # Base UV index by month for 39°N latitude (Tahoe)
    base_uv = {1: 2, 2: 3, 3: 5, 4: 7, 5: 9, 6: 11,
               7: 11, 8: 10, 9: 8, 10: 5, 11: 3, 12: 2}
    uv = base_uv.get(month, 6)

    # Altitude correction: +12% per 1000m
    altitude_boost = 1 + (elev_ft * 0.3048 / 1000) * 0.12
    effective_uv = uv * altitude_boost

    # Time to burn for fair skin (SPF 1 equivalent)
    if effective_uv > 0:
        burn_minutes = round(200 / effective_uv)
    else:
        burn_minutes = 999

    if effective_uv >= 11:
        risk = "Extreme"
        detail = f"UV index ~{effective_uv:.0f} at elevation. Apply SPF 50+ every 90 min. Wear hat + sunglasses."
    elif effective_uv >= 8:
        risk = "Very High"
        detail = f"UV index ~{effective_uv:.0f}. Sunburn in {burn_minutes} min unprotected. SPF 30+ required."
    elif effective_uv >= 6:
        risk = "High"
        detail = f"UV index ~{effective_uv:.0f}. Wear sunscreen and a hat."
    elif effective_uv >= 3:
        risk = "Moderate"
        detail = f"UV index ~{effective_uv:.0f}. Sunscreen recommended for extended exposure."
    else:
        risk = "Low"
        detail = "Low UV risk."

    return {
        "risk": risk,
        "effective_uv_index": round(effective_uv, 1),
        "base_uv_index": uv,
        "altitude_boost_pct": round((altitude_boost - 1) * 100),
        "burn_time_minutes": burn_minutes,
        "detail": detail,
    }


def _assess_water_availability(snotel: dict, month: int) -> dict:
    """
    Estimate creek water availability from snowpack and season.

    Hikers need to know if backcountry water sources (creeks, springs)
    are flowing. This is driven by snowmelt: high SWE = flowing creeks.
    By late September, many seasonal creeks are dry.
    """
    # Get current SWE from nearest SNOTEL station
    swe_values = []
    for station_name, data in snotel.items():
        if isinstance(data, dict) and "error" not in data:
            swe = data.get("swe_in") or data.get("WTEQ")
            if swe is not None:
                swe_values.append(swe)

    avg_swe = np.mean(swe_values) if swe_values else None

    if month in [11, 12, 1, 2, 3]:
        return {"status": "Frozen/Snowy",
                "detail": "Water sources frozen or snow-covered. Bring all water.",
                "carry_water": True}

    if avg_swe is not None and avg_swe > 10:
        return {"status": "Excellent",
                "detail": f"Snowpack SWE: {avg_swe:.1f}\". Creeks and springs should be flowing well.",
                "carry_water": True,  # Always carry water in mountains
                "creek_flow": "High"}
    elif avg_swe is not None and avg_swe > 3:
        return {"status": "Good",
                "detail": f"Snowpack SWE: {avg_swe:.1f}\". Most creeks still flowing. Some seasonal sources drying.",
                "carry_water": True,
                "creek_flow": "Moderate"}
    elif month in [6, 7]:
        return {"status": "Good",
                "detail": "Early-mid summer. Most water sources should be flowing from snowmelt.",
                "carry_water": True,
                "creek_flow": "Moderate-High"}
    elif month in [8, 9]:
        return {"status": "Low",
                "detail": "Late summer. Many seasonal creeks are dry. Carry all water you need.",
                "carry_water": True,
                "creek_flow": "Low"}
    elif month == 10:
        return {"status": "Very Low",
                "detail": "Fall. Most seasonal water sources dry. Only perennial streams flowing.",
                "carry_water": True,
                "creek_flow": "Very Low"}
    else:
        return {"status": "Unknown",
                "detail": "Carry extra water. Check recent trip reports for water source status.",
                "carry_water": True}


def _compute_summer_window(temp_f, cape, timeline, thunderstorm) -> dict:
    """
    Compute the best hiking/biking window considering heat and thunderstorms.

    Sierra summer pattern: cool mornings, hot midday, thunderstorms 2-5 PM.
    Best hiking: start at dawn, summit by noon, down by 2 PM.
    Best biking: early morning or late afternoon (heat dependent).
    """
    t_risk = thunderstorm.get("risk", "Low")
    turn_around = thunderstorm.get("turn_around_time")

    if temp_f is not None and temp_f > 85:
        if t_risk in ["High", "Moderate"]:
            advice = f"Start at dawn (5-6 AM). Heat + thunderstorm risk. Be off exposed terrain by {turn_around or '1 PM'}. Resume after 5 PM if clear."
            windows = ["5-11 AM (best)", "After 5 PM (if storms clear)"]
        else:
            advice = "Start early (6-7 AM) to avoid afternoon heat. Best conditions before 11 AM."
            windows = ["6-11 AM (best)", "After 5 PM (cooler)"]
    elif t_risk in ["High", "Moderate"]:
        advice = f"Start by 7 AM. Be below treeline by {turn_around or '1 PM'}. Thunderstorm risk peaks 2-5 PM."
        windows = ["7 AM - Noon (best)", "After storms clear (~6 PM)"]
    elif temp_f is not None and temp_f < 50:
        advice = "Cool conditions. Layer up. Best conditions 10 AM - 3 PM when warmest."
        windows = ["10 AM - 3 PM (warmest)"]
    else:
        advice = "Great conditions all day. No significant heat or storm concerns."
        windows = ["All day (6 AM - 6 PM)"]

    return {
        "advice": advice,
        "best_windows": windows,
        "thunderstorm_constraint": turn_around,
        "heat_constraint": "Start early" if (temp_f or 70) > 85 else None,
    }


def _summer_activity_recommendation(conditions: dict) -> dict:
    """
    Synthesize all conditions into a single go/no-go activity recommendation.
    """
    thunder = conditions.get("thunderstorm", {})
    heat = conditions.get("heat_safety", {})
    smoke = conditions.get("smoke_aqi", {})
    water = conditions.get("water_availability", {})
    window = conditions.get("best_window", {})

    blockers = []
    warnings = []
    positives = []

    # Check blockers (no-go conditions)
    if smoke.get("aqi") and smoke["aqi"] > 200:
        blockers.append(f"Air quality is {smoke['risk']} (AQI {smoke['aqi']})")
    if heat.get("risk") == "Extreme":
        blockers.append(f"Extreme heat ({heat.get('temperature_f', '?')}F)")

    # Check warnings
    if thunder.get("risk") in ["High", "Moderate"]:
        warnings.append(f"Thunderstorm risk: {thunder['risk']}")
    if smoke.get("aqi") and 100 < smoke["aqi"] <= 200:
        warnings.append(f"Smoke: AQI {smoke['aqi']}")
    if heat.get("risk") in ["High"]:
        warnings.append(f"Heat: {heat.get('temperature_f', '?')}F")

    # Check positives
    if smoke.get("risk") in ["Good", "Moderate"]:
        positives.append("Clean air")
    if heat.get("risk") in ["Low", "None", "Moderate"]:
        positives.append("Comfortable temps")
    if thunder.get("risk") in ["Very Low", "Low"]:
        positives.append("No storm risk")
    if water.get("creek_flow") in ["High", "Moderate-High"]:
        positives.append("Water sources flowing")

    if blockers:
        verdict = "NO-GO"
        color = "red"
        summary = f"Conditions not safe: {'; '.join(blockers)}"
    elif len(warnings) >= 2:
        verdict = "CAUTION"
        color = "yellow"
        summary = f"Proceed with caution: {'; '.join(warnings)}. {window.get('advice', '')}"
    elif warnings:
        verdict = "GO (with awareness)"
        color = "yellow-green"
        summary = f"Good to go, but watch for: {warnings[0]}. {window.get('advice', '')}"
    else:
        verdict = "GO"
        color = "green"
        summary = f"Excellent conditions! {', '.join(positives)}."

    return {
        "verdict": verdict,
        "color": color,
        "summary": summary,
        "blockers": blockers,
        "warnings": warnings,
        "positives": positives,
    }


# =========================================================================
# Main Correction Pipeline
# =========================================================================

def apply_ml_corrections(analysis: dict) -> dict:
    """
    Apply trained ML corrections to an analyze_all() output.

    In SHADOW_MODE (default): predictions are logged but NOT applied.
    When SHADOW_MODE=False: predictions are applied to the forecast.

    Adds to each zone:
      - ml_depth_prediction_in: predicted depth change (always logged)
      - ml_new_snow_prob: probability of new snow (always logged)
      - ml_snow_quality: quality label + rating (always added)
      - ml_wind_hold: wind hold probability + risk (always added)
      - ml_snow_timing: when snow starts/peaks (always added)
      - ml_corrected: whether corrections were actually applied
      - ml_correction_factor: multiplicative correction (only if not shadow)

    Adds to top level:
      - ml_status: dict with model info, shadow mode flag, drift metrics
    """
    global _load_attempted
    with _load_lock:
        if not _load_attempted:
            _load_models()

    if not _loaded_models:
        analysis["ml_status"] = {"active": False, "reason": "No trained models available"}
        return analysis

    corrections_applied = 0
    resorts = analysis.get("resorts", {})

    for resort_name, resort_data in resorts.items():
        zones = resort_data.get("zones", {})
        for zone_key, zone_data in zones.items():
            elev_ft = zone_data.get("elev_ft", 0)
            band = _elev_to_band(elev_ft)
            current = zone_data.get("current", {})

            # Extract forecast-time features
            features = _extract_live_features(analysis, zone_data)

            predictions = {}

            # --- Depth change prediction ---
            depth_model_key = f"depth_change_in_{band}"
            if depth_model_key not in _loaded_models:
                depth_model_key = "depth_change_in_all"

            if depth_model_key in _loaded_models:
                entry = _loaded_models[depth_model_key]
                model_features = entry["features"]

                feature_array = np.array([
                    features.get(f, np.nan) for f in model_features
                ]).reshape(1, -1)

                try:
                    predicted_depth = float(entry["model"].predict(feature_array)[0])
                    predictions["ml_depth_prediction_in"] = round(predicted_depth, 1)
                    zone_data["ml_depth_prediction_in"] = round(predicted_depth, 1)

                    # Apply correction only if NOT in shadow mode
                    if not SHADOW_MODE:
                        existing_snow_24h = zone_data.get("snow_24h", 0) or 0
                        if existing_snow_24h > 0 and predicted_depth > 0:
                            correction = predicted_depth / max(0.1, existing_snow_24h)
                            correction = max(0.3, min(2.0, correction))
                            zone_data["ml_correction_factor"] = round(correction, 3)
                        else:
                            zone_data["ml_correction_factor"] = 1.0
                        zone_data["ml_corrected"] = True
                        corrections_applied += 1
                    else:
                        zone_data["ml_corrected"] = False
                        zone_data["ml_correction_factor"] = 1.0  # No correction in shadow

                except Exception as e:
                    logger.warning("ML depth prediction failed for %s/%s: %s",
                                   resort_name, zone_key, e)
                    zone_data["ml_corrected"] = False

            # --- New snow probability ---
            snow_model_key = f"new_snow_{band}"
            if snow_model_key not in _loaded_models:
                snow_model_key = "new_snow_all"

            if snow_model_key in _loaded_models:
                entry = _loaded_models[snow_model_key]
                model_features = entry["features"]
                feature_array = np.array([
                    features.get(f, np.nan) for f in model_features
                ]).reshape(1, -1)
                try:
                    prob = float(entry["model"].predict_proba(feature_array)[0, 1])
                    predictions["ml_new_snow_prob"] = round(prob, 3)
                    zone_data["ml_new_snow_prob"] = round(prob, 3)
                except Exception:
                    pass

            # --- Skier 1: Snow quality ---
            slr = features.get("slr", 10)
            temp = features.get("temperature_2m_mean", 32)
            wind = features.get("wind_speed_10m_max", 0)
            zone_data["ml_snow_quality"] = _predict_snow_quality(slr, temp, wind or 0)

            # --- Skier 2: Wind hold probability ---
            gusts = features.get("wind_gusts_10m_max", 0) or 0
            zone_data["ml_wind_hold"] = _predict_wind_hold(gusts, elev_ft)

            # --- Skier 3: Snow timing ---
            timeline = zone_data.get("timeline_48h", [])
            zone_data["ml_snow_timing"] = _predict_snow_timing(timeline)

            # --- Shadow logging ---
            if predictions:
                _log_shadow_prediction(resort_name, zone_key, predictions)

    # =====================================================================
    # Cross-Resort Decision Engine (Skier 5/5 features)
    # =====================================================================

    # --- 1. Resort Recommendation: "Which resort should I go to?" ---
    analysis["ml_resort_ranking"] = _rank_resorts(resorts)

    # --- 2. Chain Requirement Prediction ---
    analysis["ml_chain_forecast"] = _predict_chains(resorts, analysis)

    # --- 3. Best Skiing Window ---
    for resort_name, resort_data in resorts.items():
        zones = resort_data.get("zones", {})
        peak = zones.get("peak", {})
        mid = zones.get("mid", {})
        best_zone = peak if peak else mid
        if best_zone:
            resort_data["ml_best_window"] = _compute_best_window(best_zone)

    # --- 4. Powder Alert ---
    analysis["ml_powder_alert"] = _compute_powder_alert(resorts)

    # --- 5. Summer Outdoor Conditions (Hiker/Biker) ---
    # Delegate to outdoor_conditions.py (already called from tahoe_snow.py).
    # If analysis["outdoor"] doesn't exist yet, compute it here as fallback.
    if "outdoor" not in analysis:
        try:
            from outdoor_conditions import compute_all_outdoor_conditions
            analysis["outdoor"] = compute_all_outdoor_conditions(analysis)
        except Exception:
            pass
    # Build the ml_outdoor summary from the outdoor data
    analysis["ml_outdoor"] = _build_ml_outdoor_from_outdoor(analysis)

    # Add ML status
    analysis["ml_status"] = {
        "active": True,
        "shadow_mode": SHADOW_MODE,
        "corrections_applied": corrections_applied,
        "models_loaded": len(_loaded_models),
        "drift": get_drift_metrics(),
    }

    return analysis


def get_ml_status() -> dict:
    """Return current ML pipeline status for dashboard display."""
    global _load_attempted
    with _load_lock:
        if not _load_attempted:
            _load_models()

    if not _loaded_models:
        return {"active": False, "models_loaded": 0}

    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "active": True,
        "shadow_mode": SHADOW_MODE,
        "models_loaded": len(_loaded_models),
        "model_names": list(_loaded_models.keys()),
        "trained_at": summary.get("trained_at", "unknown"),
        "drift": get_drift_metrics(),
    }
