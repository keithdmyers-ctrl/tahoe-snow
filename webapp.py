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

logger = logging.getLogger(__name__)

app = Flask(__name__)

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
        try:
            oakland = fetch_oakland_data()
            log_daily_verification(oakland["home_obs"], oakland["home_fc"], analysis)
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
        return jsonify({"error": str(e)}), 500


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
