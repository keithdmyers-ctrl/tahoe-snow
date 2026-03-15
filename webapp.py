#!/usr/bin/env python3
"""
Tahoe Snow Web App — Flask backend.

Serves the dashboard UI and provides a JSON API for live data.
Data is cached in memory with a configurable TTL to avoid hammering APIs.

Usage:
  .venv/bin/python3 webapp.py                 # dev server on :5000
  .venv/bin/python3 webapp.py --port 8080     # custom port
"""

import json
import sys
import os
import time
import threading
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tahoe_snow import (
    fetch_nws_observations, fetch_nws_forecast, fetch_nws_gridpoints,
    fetch_open_meteo_multi,
    fetch_nbm, fetch_pws_nearby, aggregate_pws,
    fetch_synoptic_stations, fetch_cssl_snow,
    fetch_nws_alerts, fetch_sounding, fetch_climate_normals,
    fetch_ensemble, fetch_caltrans_chains, fetch_all_lift_status,
    fetch_snotel_current, fetch_snotel_history, fetch_snotel_season,
    fetch_avalanche, fetch_forecast_discussion,
    analyze_all, RESORTS, SNOTEL_STATIONS,
)
from pressure_forecast import get_storm_total
from forecast_verification import log_daily_verification, get_verification_summary

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

CACHE_TTL = 900  # 15 minutes
_cache = {"data": None, "timestamp": 0, "loading": False}
_lock = threading.Lock()


def get_analysis(force: bool = False) -> dict:
    """Return cached analysis, refreshing if stale."""
    now = time.time()
    with _lock:
        if _cache["data"] and not force and (now - _cache["timestamp"]) < CACHE_TTL:
            return _cache["data"]
        if _cache["loading"]:
            return _cache["data"] or {"error": "Loading..."}
        _cache["loading"] = True

    try:
        obs = fetch_nws_observations(39.17, -120.145)
        nws = fetch_nws_forecast(39.17, -120.145)
        resort_points = {rn: {"lat": r["base"]["lat"], "lon": r["base"]["lon"]}
                         for rn, r in RESORTS.items()}
        om = fetch_open_meteo_multi(resort_points)
        snotel = fetch_snotel_current()
        avy = fetch_avalanche()
        afd = fetch_forecast_discussion()

        # Additional data sources
        tahoe_nbm = fetch_nbm(39.17, -120.145)
        nws_grids = fetch_nws_gridpoints(39.17, -120.145)
        cssl = fetch_cssl_snow()
        tahoe_alerts = fetch_nws_alerts(39.17, -120.145)
        sounding = fetch_sounding("REV")
        normals = fetch_climate_normals(39.17, -120.145)
        ensemble = fetch_ensemble(39.17, -120.145)
        synoptic = fetch_synoptic_stations(39.17, -120.145, radius_miles=30)
        storm = get_storm_total(snotel)
        chains = fetch_caltrans_chains()
        lifts = fetch_all_lift_status()

        # Pass all data sources to analyze_all for full pipeline integration:
        # NWS grids, sounding, ensemble, and Synoptic stations
        analysis = analyze_all(obs, nws, om, snotel, afd, avy, {},
                               nws_grids=nws_grids if "error" not in nws_grids else None,
                               sounding=sounding if "error" not in sounding else None,
                               ensemble=ensemble if ensemble.get("models") else None,
                               synoptic=synoptic if "error" not in synoptic else None)

        # Attach extra data to analysis for the web UI
        analysis["nbm"] = tahoe_nbm if "error" not in tahoe_nbm else None
        analysis["nws_grids"] = nws_grids if "error" not in nws_grids else None
        analysis["cssl"] = cssl if "error" not in cssl else None
        analysis["alerts"] = tahoe_alerts or []
        if not analysis.get("sounding"):
            analysis["sounding"] = sounding if "error" not in sounding else None
        analysis["normals"] = normals if "error" not in normals else None
        if not analysis.get("ensemble"):
            analysis["ensemble"] = ensemble if ensemble.get("models") else None
        analysis["synoptic"] = synoptic if "error" not in synoptic else None
        analysis["storm"] = storm
        analysis["chains"] = chains
        analysis["lifts"] = lifts

        # Daily forecast verification logging (best-effort)
        try:
            home_obs = fetch_nws_observations(37.8024, -122.1828)
            home_fc = fetch_nws_forecast(37.8024, -122.1828)
            log_daily_verification(home_obs, home_fc, analysis)
        except Exception:
            pass  # verification is best-effort, never break main flow

        with _lock:
            _cache["data"] = analysis
            _cache["timestamp"] = time.time()
            _cache["loading"] = False
        return analysis
    except Exception as e:
        import logging
        logging.exception("Error fetching weather data")
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
        return jsonify({"error": str(e)}), 500


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
