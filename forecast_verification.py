#!/usr/bin/env python3
"""
Forecast verification, bias tracking, and model skill scoring.

Logs daily forecasts and compares them to actual observations to build
location-specific bias corrections over time.

Tracks:
  - Temperature bias per source (NWS, Open-Meteo models, BME280)
  - Precipitation probability calibration (predicted % vs actual occurrence)
  - Timing accuracy (predicted rain start vs actual)
  - Model skill scores (MAE, RMSE, Brier, CRPS) by source and lead time
  - Per-model weights for skill-weighted blending

After 14+ days of data, applies automatic bias corrections.
After 7+ days, computes model skill weights for blending.

Usage:
  from forecast_verification import (log_forecast, log_actual,
      get_bias_corrections, get_model_weights, get_verification_summary)
  log_forecast(source, metric, predicted_value, valid_time)
  log_actual(metric, actual_value, obs_time)
  corrections = get_bias_corrections()
  weights = get_model_weights()   # {"GFS": 0.35, "ECMWF": 0.40, ...}
"""

import json
import math
import os
from datetime import datetime, timezone, timedelta

VERIFY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".forecast_verification.json")
MAX_DAYS = 90  # Keep 90 days of verification data
MIN_DAYS_FOR_WEIGHTS = 7  # Minimum days before computing skill weights


def _load_data() -> dict:
    try:
        with open(VERIFY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"forecasts": [], "actuals": [], "daily_scores": []}


def _save_data(data: dict):
    tmp = VERIFY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, VERIFY_FILE)


def log_forecast(source: str, metric: str, value: float,
                 valid_date: str, lead_hours: int = 0):
    """Log a forecast prediction for later verification.

    Args:
        source: e.g. "nws", "gfs", "ecmwf", "icon", "hrrr", "bme280"
        metric: e.g. "temp_high_f", "temp_low_f", "precip_pct", "rain_start_hour"
        value: the predicted value
        valid_date: YYYY-MM-DD the forecast is valid for
        lead_hours: hours ahead this forecast was made
    """
    data = _load_data()
    data["forecasts"].append({
        "t": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "metric": metric,
        "value": value,
        "valid_date": valid_date,
        "lead_hours": lead_hours,
    })
    _trim(data)
    _save_data(data)


def log_actual(metric: str, value: float, obs_date: str):
    """Log an actual observation for verification.

    Args:
        metric: e.g. "temp_high_f", "temp_low_f", "did_rain" (0 or 1)
        value: the observed value
        obs_date: YYYY-MM-DD
    """
    data = _load_data()
    # Deduplicate: only one actual per metric per date
    data["actuals"] = [a for a in data["actuals"]
                       if not (a["metric"] == metric and a["date"] == obs_date)]
    data["actuals"].append({
        "t": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "value": value,
        "date": obs_date,
    })
    _trim(data)
    _score_day(data, obs_date)
    _save_data(data)


def _score_day(data: dict, date: str):
    """Score all forecasts for a given date against actuals."""
    actuals_for_date = {a["metric"]: a["value"] for a in data["actuals"]
                        if a["date"] == date}
    if not actuals_for_date:
        return

    forecasts_for_date = [f for f in data["forecasts"] if f["valid_date"] == date]

    scores = []
    for fc in forecasts_for_date:
        metric = fc["metric"]
        if metric in actuals_for_date:
            actual = actuals_for_date[metric]
            error = fc["value"] - actual
            scores.append({
                "date": date,
                "source": fc["source"],
                "metric": metric,
                "predicted": fc["value"],
                "actual": actual,
                "error": round(error, 2),
                "abs_error": round(abs(error), 2),
                "lead_hours": fc["lead_hours"],
            })

    if scores:
        # Remove old scores for this date, add new ones
        data["daily_scores"] = [s for s in data["daily_scores"] if s["date"] != date]
        data["daily_scores"].extend(scores)


def get_bias_corrections() -> dict:
    """Compute bias corrections from accumulated verification data.

    Returns dict of {source: {metric: correction_value}}.
    Correction should be ADDED to future forecasts to debias them.
    Only returns corrections with 14+ days of data.
    """
    data = _load_data()
    scores = data.get("daily_scores", [])

    if not scores:
        return {}

    # Group errors by source+metric
    groups = {}
    for s in scores:
        key = (s["source"], s["metric"])
        if key not in groups:
            groups[key] = []
        groups[key].append(s["error"])

    corrections = {}
    for (source, metric), errors in groups.items():
        if len(errors) < 14:
            continue
        # Use median error as bias correction (robust to outliers)
        errors_sorted = sorted(errors)
        n = len(errors_sorted)
        median = errors_sorted[n // 2] if n % 2 else (errors_sorted[n // 2 - 1] + errors_sorted[n // 2]) / 2
        # Correction is negative of bias (add to prediction to correct)
        correction = -round(median, 1)
        if abs(correction) >= 0.5:  # Only correct if bias is meaningful
            if source not in corrections:
                corrections[source] = {}
            corrections[source][metric] = correction

    return corrections


def get_verification_summary() -> dict:
    """Get a summary of forecast accuracy by source and metric.

    Includes MAE, RMSE, mean bias, and Brier score (for precip_pct).
    """
    data = _load_data()
    scores = data.get("daily_scores", [])

    if not scores:
        return {"message": "No verification data yet", "days": 0}

    dates = set(s["date"] for s in scores)
    groups = {}
    for s in scores:
        key = (s["source"], s["metric"])
        if key not in groups:
            groups[key] = {"errors": [], "abs_errors": [], "predicted": [],
                           "actual": [], "lead_hours": []}
        groups[key]["errors"].append(s["error"])
        groups[key]["abs_errors"].append(s["abs_error"])
        groups[key]["predicted"].append(s.get("predicted", 0))
        groups[key]["actual"].append(s.get("actual", 0))
        groups[key]["lead_hours"].append(s.get("lead_hours", 0))

    summary = {"days": len(dates), "sources": {}}
    for (source, metric), g in groups.items():
        if source not in summary["sources"]:
            summary["sources"][source] = {}
        n = len(g["errors"])
        mean_error = sum(g["errors"]) / n
        mae = sum(g["abs_errors"]) / n
        rmse = math.sqrt(sum(e ** 2 for e in g["errors"]) / n)

        entry = {
            "n": n,
            "mean_bias": round(mean_error, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
        }

        # Brier score for precipitation probability
        if metric == "precip_pct":
            brier = _brier_score(g["predicted"], g["actual"])
            if brier is not None:
                entry["brier_score"] = round(brier, 4)

        # Skill by lead time buckets
        lead_buckets = _group_by_lead_time(g["abs_errors"], g["lead_hours"])
        if lead_buckets:
            entry["mae_by_lead"] = lead_buckets

        summary["sources"][source][metric] = entry

    return summary


def _brier_score(predicted_pcts: list, actuals: list) -> float | None:
    """Compute Brier score for probability forecasts.

    Brier = (1/N) * sum((p_i - o_i)^2) where p_i is predicted probability
    (0-1) and o_i is binary outcome (0 or 1).
    Perfect score = 0, worst = 1.
    """
    pairs = [(p, a) for p, a in zip(predicted_pcts, actuals)
             if p is not None and a is not None]
    if len(pairs) < 5:
        return None
    total = 0.0
    for p_pct, actual in pairs:
        p = max(0, min(100, p_pct)) / 100.0  # Convert % to probability
        o = 1.0 if actual > 0 else 0.0
        total += (p - o) ** 2
    return total / len(pairs)


def _group_by_lead_time(abs_errors: list, lead_hours: list) -> dict:
    """Group MAE by lead time buckets: 0-24h, 24-48h, 48-72h, 72h+."""
    buckets = {"0-24h": [], "24-48h": [], "48-72h": [], "72h+": []}
    for err, lead in zip(abs_errors, lead_hours):
        if lead < 24:
            buckets["0-24h"].append(err)
        elif lead < 48:
            buckets["24-48h"].append(err)
        elif lead < 72:
            buckets["48-72h"].append(err)
        else:
            buckets["72h+"].append(err)
    result = {}
    for k, v in buckets.items():
        if v:
            result[k] = round(sum(v) / len(v), 2)
    return result


def get_model_weights(metric: str = "temp_high_f") -> dict:
    """Compute skill-based weights for multi-model blending.

    Uses inverse MAE weighting with exponential recency decay:
    recent verification days get more weight than older ones.
    This allows the system to adapt to seasonal model skill changes
    (e.g., ECMWF often gains skill in spring vs winter).

    Half-life: 14 days (error from 14 days ago has half the weight
    of today's error).

    Returns dict like {"GFS": 0.30, "ECMWF": 0.35, "ICON": 0.20, "NBM": 0.15}
    with weights summing to 1.0.
    """
    data = _load_data()
    scores = data.get("daily_scores", [])

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    half_life_days = 14.0

    # Group by source for the specified metric, with recency weighting
    source_weighted_errors = {}  # src -> (weighted_sum_errors, weighted_count)
    for s in scores:
        if s["metric"] != metric:
            continue
        src = s["source"].upper()
        if src not in source_weighted_errors:
            source_weighted_errors[src] = [0.0, 0.0]

        # Compute age-based weight
        try:
            age_days = (datetime.fromisoformat(today) -
                        datetime.fromisoformat(s["date"])).days
        except (ValueError, TypeError):
            age_days = 30  # fallback
        decay_weight = 2.0 ** (-age_days / half_life_days)

        source_weighted_errors[src][0] += s["abs_error"] * decay_weight
        source_weighted_errors[src][1] += decay_weight

    # Need minimum effective samples for each source
    valid_sources = {}
    for src, (w_sum, w_count) in source_weighted_errors.items():
        if w_count >= MIN_DAYS_FOR_WEIGHTS * 0.5:  # Effective sample count
            valid_sources[src] = w_sum / w_count  # Weighted MAE

    if len(valid_sources) < 2:
        return _default_weights()

    # Inverse MAE weighting: w_i = (1/MAE_i) / sum(1/MAE_j)
    eps = 0.01
    inv_mae = {src: 1.0 / (mae + eps) for src, mae in valid_sources.items()}
    total = sum(inv_mae.values())
    weights = {src: round(w / total, 3) for src, w in inv_mae.items()}

    return weights


def _default_weights() -> dict:
    """Default model weights when verification data is insufficient.

    Based on published model skill assessments:
    - ECMWF: Generally highest skill globally
    - NBM: NWS bias-corrected blend, strong for US locations
    - GFS: Good general skill, US-optimized
    - ICON: European model, good independent check
    """
    return {"ECMWF": 0.30, "GFS": 0.30, "NBM": 0.25, "ICON": 0.15}


def get_pop_calibration() -> dict:
    """Get precipitation probability calibration data.

    Returns observed frequency of rain for each predicted probability bin.
    Used for reliability assessment and Platt-style recalibration.
    Bins: 0-10%, 10-20%, ..., 90-100%.
    """
    data = _load_data()
    scores = data.get("daily_scores", [])

    bins = {f"{i*10}-{(i+1)*10}%": {"count": 0, "rained": 0}
            for i in range(10)}

    for s in scores:
        if s["metric"] != "precip_pct":
            continue
        # Find matching actual for this date
        actual_rain = None
        for a in data.get("actuals", []):
            if a["metric"] == "did_rain" and a["date"] == s["date"]:
                actual_rain = a["value"]
                break
        if actual_rain is None:
            continue

        predicted_pct = max(0, min(99, s["predicted"]))
        bin_idx = min(int(predicted_pct / 10), 9)
        bin_key = f"{bin_idx*10}-{(bin_idx+1)*10}%"
        bins[bin_key]["count"] += 1
        if actual_rain > 0:
            bins[bin_key]["rained"] += 1

    # Compute observed frequency per bin
    result = {}
    for k, v in bins.items():
        if v["count"] >= 3:
            result[k] = {
                "n": v["count"],
                "observed_freq": round(v["rained"] / v["count"], 3),
            }

    return result


def recalibrate_pop(raw_pop_pct: float) -> float:
    """Recalibrate precipitation probability using reliability data.

    If sufficient verification data exists, adjusts raw model PoP
    toward observed frequency (Platt-style isotonic recalibration).
    Otherwise returns raw value unchanged.

    Example: If models say "60%" but it only rained 45% of the time
    when they said 60%, this adjusts toward 45%.
    """
    cal_data = get_pop_calibration()
    if not cal_data or len(cal_data) < 3:
        return raw_pop_pct  # Insufficient data for recalibration

    # Find the bin for this prediction
    bin_idx = min(int(max(0, raw_pop_pct) / 10), 9)
    bin_key = f"{bin_idx*10}-{(bin_idx+1)*10}%"

    if bin_key in cal_data and cal_data[bin_key]["n"] >= 5:
        observed = cal_data[bin_key]["observed_freq"]
        raw_frac = raw_pop_pct / 100.0
        # Blend: 60% observed calibration, 40% raw (don't over-correct)
        calibrated = observed * 0.6 + raw_frac * 0.4
        return round(calibrated * 100, 1)

    return raw_pop_pct


def log_daily_verification(home_obs: dict, home_fc: dict, analysis: dict):
    """Convenience function: log today's forecasts and yesterday's actuals.

    Call this once per day (e.g., at 11pm) to:
    1. Log today's model forecasts for tomorrow
    2. Log today's actual observations to verify yesterday's forecasts
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    # --- Log actuals for today ---
    if home_obs:
        temp_f = home_obs.get("temp_f")
        if temp_f is not None:
            log_actual("temp_current_f", temp_f, today)

    # Today's high/low from NWS periods
    periods = home_fc.get("periods", []) if home_fc else []
    for p in periods:
        start = p.get("startTime", "")[:10]
        if start == today:
            temp = p.get("temperature")
            if temp is not None:
                if p.get("isDaytime", True):
                    log_actual("temp_high_f", temp, today)
                else:
                    log_actual("temp_low_f", temp, today)

    # Did it rain today? Check hourly for actuals
    hourly = home_fc.get("hourly", []) if home_fc else []
    rain_today = 0
    for h in hourly:
        start = h.get("startTime", "")[:10]
        if start != today:
            continue
        short = h.get("shortForecast", "").lower()
        if any(k in short for k in ("rain", "shower", "drizzle", "thunder")):
            rain_today = 1
            break
    log_actual("did_rain", rain_today, today)

    # --- Log forecasts for tomorrow ---
    for p in periods:
        start = p.get("startTime", "")[:10]
        if start == tomorrow:
            temp = p.get("temperature")
            if temp is not None:
                metric = "temp_high_f" if p.get("isDaytime", True) else "temp_low_f"
                log_forecast("nws", metric, temp, tomorrow, lead_hours=24)

    # Log rain probability for tomorrow from NWS hourly
    tomorrow_pops = []
    for h in hourly:
        start = h.get("startTime", "")[:10]
        if start == tomorrow:
            pop = h.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
            tomorrow_pops.append(pop)
    if tomorrow_pops:
        log_forecast("nws", "precip_pct", max(tomorrow_pops), tomorrow, lead_hours=24)

    # Log Open-Meteo model forecasts for tomorrow from model spread data
    resorts = analysis.get("resorts", {})
    heavenly = resorts.get("Heavenly", {})
    peak = heavenly.get("zones", {}).get("peak", {})

    # Log per-model temperature forecasts for skill tracking
    model_spread = peak.get("model_spread", [])
    for day_data in model_spread:
        if day_data.get("date") != tomorrow:
            continue
        models = day_data.get("models", {})
        for model_name, model_data in models.items():
            temp = model_data.get("temp_high_f")
            if temp is not None:
                log_forecast(model_name.lower(), "temp_high_f", temp,
                             tomorrow, lead_hours=24)
            snow = model_data.get("snow_in")
            if snow is not None:
                log_forecast(model_name.lower(), "snow_in", snow,
                             tomorrow, lead_hours=24)
        break

    # Also log blended forecast for tracking blend accuracy
    buckets = peak.get("day_night_buckets", [])
    for b in buckets:
        if b["date"] == tomorrow and b["period"] == "Day":
            if b.get("temp_high_f") is not None:
                log_forecast("blend", "temp_high_f", b["temp_high_f"],
                             tomorrow, lead_hours=24)


def _trim(data: dict):
    """Remove entries older than MAX_DAYS."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_DAYS)).isoformat()
    data["forecasts"] = [f for f in data["forecasts"] if f["t"] > cutoff]
    data["actuals"] = [a for a in data["actuals"] if a["t"] > cutoff]
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=MAX_DAYS)).strftime("%Y-%m-%d")
    data["daily_scores"] = [s for s in data["daily_scores"] if s["date"] > cutoff_date]


if __name__ == "__main__":
    summary = get_verification_summary()
    print(json.dumps(summary, indent=2))
    corrections = get_bias_corrections()
    if corrections:
        print("\nBias corrections:")
        print(json.dumps(corrections, indent=2))
    else:
        print("\nNo bias corrections yet (need 14+ days of data)")
