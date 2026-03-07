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

sys.path.insert(0, os.path.dirname(__file__))
from tahoe_snow import (
    fetch_nws_observations, fetch_nws_forecast, fetch_open_meteo_multi,
    fetch_nbm, fetch_pws_nearby, aggregate_pws, fetch_cssl_snow,
    fetch_nws_alerts, fetch_sounding, fetch_climate_normals,
    fetch_caltrans_chains, fetch_all_lift_status,
    fetch_snotel_current, fetch_snotel_history, fetch_snotel_season,
    fetch_avalanche, fetch_forecast_discussion,
    analyze_all, RESORTS, SNOTEL_STATIONS,
)
from pressure_forecast import get_storm_total

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
        cssl = fetch_cssl_snow()
        tahoe_alerts = fetch_nws_alerts(39.17, -120.145)
        sounding = fetch_sounding("REV")
        normals = fetch_climate_normals(39.17, -120.145)
        storm = get_storm_total(snotel)
        chains = fetch_caltrans_chains()
        lifts = fetch_all_lift_status()

        analysis = analyze_all(obs, nws, om, snotel, afd, avy, {})

        # Attach extra data to analysis for the web UI
        analysis["nbm"] = tahoe_nbm if "error" not in tahoe_nbm else None
        analysis["cssl"] = cssl if "error" not in cssl else None
        analysis["alerts"] = tahoe_alerts or []
        analysis["sounding"] = sounding if "error" not in sounding else None
        analysis["normals"] = normals if "error" not in normals else None
        analysis["storm"] = storm
        analysis["chains"] = chains
        analysis["lifts"] = lifts

        with _lock:
            _cache["data"] = analysis
            _cache["timestamp"] = time.time()
            _cache["loading"] = False
        return analysis
    except Exception as e:
        with _lock:
            _cache["loading"] = False
        return _cache["data"] or {"error": str(e)}


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


@app.route("/api/refresh")
def api_refresh():
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
