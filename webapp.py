#!/usr/bin/env python3
"""
Tahoe Snow Web App -- Flask backend.

Serves the dashboard UI and provides a JSON API for live data.
Data is cached in memory with a configurable TTL to avoid hammering APIs.

Usage:
  .venv/bin/python3 webapp.py                 # dev server on :5000
  .venv/bin/python3 webapp.py --port 8080     # custom port
"""

import logging
import sys
import os
import time
import threading
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import fetch_tahoe_analysis, fetch_oakland_data
from tahoe_snow import log_storm_event
from forecast_verification import log_daily_verification, get_verification_summary
try:
    from outdoor_conditions import compute_activity_decisions
except ImportError:
    compute_activity_decisions = None

logger = logging.getLogger(__name__)

app = Flask(__name__)


def _build_oakland_summary(oakland: dict) -> dict:
    """Build a compact Oakland weather summary for the web API from raw data."""
    obs = oakland.get("home_obs", {})
    fc = oakland.get("home_fc", {})
    om = oakland.get("home_om", {})
    aqi = oakland.get("home_aqi", {})

    summary = {
        "current": {},
        "forecast": [],
        "aqi": {},
        "uv": {},
        "hourly_rain": [],
    }

    # Current conditions from NWS observation
    if obs and not obs.get("error"):
        summary["current"] = {
            "temp_f": obs.get("temp_f"),
            "feels_like_f": obs.get("feels_like_f"),
            "conditions": obs.get("conditions", ""),
            "humidity_pct": obs.get("humidity_pct"),
            "wind_mph": obs.get("wind_mph", 0),
            "wind_gust_mph": obs.get("wind_gust_mph", 0),
            "wind_dir": obs.get("wind_dir", ""),
            "visibility_mi": obs.get("visibility_mi"),
            "barometer_inhg": obs.get("barometer_inhg"),
            "station": obs.get("station", ""),
        }

    # NWS 7-day forecast periods (today/tonight + next 6)
    periods = fc.get("periods", [])
    for p in periods[:8]:
        summary["forecast"].append({
            "name": p.get("name", ""),
            "temp": p.get("temperature"),
            "unit": p.get("temperatureUnit", "F"),
            "short": p.get("shortForecast", ""),
            "detail": p.get("detailedForecast", ""),
            "wind_speed": p.get("windSpeed", ""),
            "wind_dir": p.get("windDirection", ""),
            "is_daytime": p.get("isDaytime", True),
            "rain_pct": p.get("probabilityOfPrecipitation", {}).get("value"),
        })

    # AQI
    if aqi and not aqi.get("error"):
        summary["aqi"] = {
            "aqi": aqi.get("aqi"),
            "category": aqi.get("category", ""),
            "pm2_5": aqi.get("pm2_5"),
        }

    # UV and solar from Open-Meteo
    if om and not om.get("error"):
        daily = om.get("daily", {})
        uv_max_list = daily.get("uv_index_max", [])
        sunrise_list = daily.get("sunrise", [])
        sunset_list = daily.get("sunset", [])
        if uv_max_list:
            summary["uv"] = {"max": uv_max_list[0]}
        if sunrise_list and sunset_list:
            # Extract just time portion
            try:
                sr = sunrise_list[0].split("T")[1] if "T" in str(sunrise_list[0]) else sunrise_list[0]
                ss = sunset_list[0].split("T")[1] if "T" in str(sunset_list[0]) else sunset_list[0]
                summary["sunrise"] = sr
                summary["sunset"] = ss
            except (IndexError, AttributeError):
                pass

        # Hourly rain probability for next 24h from Open-Meteo
        hourly = om.get("hourly", {})
        times = hourly.get("time", [])
        precip = hourly.get("precipitation", [])
        for i in range(min(24, len(times), len(precip))):
            summary["hourly_rain"].append({
                "time": times[i],
                "precip_mm": precip[i] if precip[i] is not None else 0,
            })

    return summary


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

CACHE_TTL = 900  # 15 minutes
_cache = {"data": None, "timestamp": 0, "loading": False}
_lock = threading.Lock()
_prev_storm_state = False  # Track storm transitions for archiving


def get_analysis(force: bool = False) -> dict:
    """Return cached analysis, refreshing if stale."""
    global _prev_storm_state

    now = time.time()
    with _lock:
        if _cache["data"] and not force and (now - _cache["timestamp"]) < CACHE_TTL:
            return _cache["data"]
        if _cache["loading"]:
            return _cache["data"] or {"error": "Loading..."}
        _cache["loading"] = True

    try:
        analysis = fetch_tahoe_analysis()

        # Daily forecast verification logging (best-effort)
        # Also include Oakland data in the API response for the web dashboard
        try:
            oakland = fetch_oakland_data()
            log_daily_verification(oakland["home_obs"], oakland["home_fc"], analysis)
            # Expose Oakland data for the web UI
            analysis["oakland"] = _build_oakland_summary(oakland)
        except Exception:
            pass  # verification is best-effort, never break main flow

        # Storm archive: log when storm ends (transition from active to inactive)
        storm = analysis.get("storm")
        current_storm = storm.get("in_storm", False) if storm else False

        # Thread safety: update _prev_storm_state inside the lock
        with _lock:
            prev = _prev_storm_state
            _prev_storm_state = current_storm

        if prev and not current_storm:
            log_storm_event(analysis)

        with _lock:
            _cache["data"] = analysis
            _cache["timestamp"] = time.time()
            _cache["loading"] = False
        return analysis
    except Exception as e:
        logger.exception("Error fetching weather data")
        with _lock:
            _cache["loading"] = False
            cached = _cache["data"]
        return cached or {"error": "Weather data is temporarily unavailable. Please try refreshing in a few minutes."}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    data = get_analysis()
    return jsonify(data)


@app.route("/api/verification")
def api_verification():
    """Return forecast verification summary as JSON."""
    try:
        summary = get_verification_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error("Verification summary failed: %s", e)
        return jsonify({"error": "Forecast verification data is temporarily unavailable."}), 500


@app.route("/api/decision")
def api_decision():
    """Return just the ski decision data."""
    data = get_analysis()
    if data.get("error"):
        return jsonify({"error": data["error"]}), 503
    return jsonify({
        "decision": data.get("decision", {}),
        "storm_narrative": data.get("storm_narrative", ""),
        "storm_history": data.get("storm_history", []),
    })


@app.route("/api/activities")
def api_activities():
    """Return activity decision cards for the outdoor dashboard.

    Each card has: activity, icon, signal (go/caution/no-go), score,
    headline, metrics, timing, detail. Sorted by score descending.
    """
    if compute_activity_decisions is None:
        return jsonify({"error": "Activity engine not available"}), 503
    data = get_analysis()
    if data.get("error"):
        return jsonify({"error": data["error"]}), 503
    try:
        cards = compute_activity_decisions(data)
        return jsonify({"activities": cards, "generated": data.get("generated", "")})
    except Exception as e:
        logger.error("Activity decisions failed: %s", e)
        return jsonify({"error": "Activity recommendations are temporarily unavailable. Please try again."}), 500


@app.route('/sw.js')
def service_worker():
    resp = app.send_static_file('sw.js')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Service-Worker-Allowed'] = '/'
    return resp

@app.route('/clear-cache')
def clear_cache():
    """Emergency cache clear — serves a page that unregisters SW and clears caches."""
    return '''<!DOCTYPE html><html><body style="background:#0a1420;color:white;font-family:system-ui;padding:40px;text-align:center">
    <h2>Clearing cache...</h2><p id="status">Working...</p>
    <script>
    (async () => {
        const s = document.getElementById('status');
        try {
            const regs = await navigator.serviceWorker.getRegistrations();
            for (const r of regs) { await r.unregister(); }
            s.textContent = 'Unregistered ' + regs.length + ' service worker(s). ';
            const keys = await caches.keys();
            for (const k of keys) { await caches.delete(k); }
            s.textContent += 'Deleted ' + keys.length + ' cache(s). Redirecting...';
            setTimeout(() => window.location.href = '/', 1500);
        } catch(e) { s.textContent = 'Error: ' + e.message; }
    })();
    </script></body></html>''', 200, {'Content-Type': 'text/html'}


@app.route('/manifest.json')
def manifest():
    return app.send_static_file('manifest.json')


@app.route("/api/refresh")
def api_refresh():
    with _lock:
        age = time.time() - _cache["timestamp"]
    if age < 60:
        return jsonify({"status": "throttled", "retry_after": round(60 - age)}), 429
    data = get_analysis(force=True)
    return jsonify({"status": "ok", "generated": data.get("generated", "")})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = 5000
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    # Pre-fetch data in background so first page load is fast
    threading.Thread(target=get_analysis, daemon=True).start()

    print(f"Starting Tahoe Snow web app on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
