#!/usr/bin/env python3
"""
Tahoe Snow Powder Alert System

Run via cron to check conditions and send notifications when thresholds are met.

Setup:
  1. Edit alerts_config.json with your thresholds and notification preferences
  2. Add to crontab: */30 * * * * cd /home/keith/projects/tahoe-snow && .venv/bin/python3 alerts.py

Notification methods:
  - Desktop notification (notify-send, works on GNOME/Wayland)
  - Webhook (POST to any URL — Discord, Slack, ntfy.sh, etc.)
  - Log file (always written)
"""

import json
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Import from main module
sys.path.insert(0, os.path.dirname(__file__))
from tahoe_snow import (fetch_nws_observations, fetch_nws_forecast,
                         fetch_open_meteo, fetch_nws_gridpoints,
                         fetch_snotel_current, fetch_avalanche,
                         fetch_forecast_discussion, fetch_sounding,
                         fetch_ensemble, fetch_synoptic_stations,
                         analyze_all, RESORTS)

import requests

CONFIG_FILE = Path(__file__).parent / "alerts_config.json"
STATE_FILE = Path(__file__).parent / ".alerts_state.json"
LOG_FILE = Path(__file__).parent / "alerts.log"

DEFAULT_CONFIG = {
    "enabled": True,
    "check_interval_minutes": 30,
    "resorts": {
        "Heavenly": {"enabled": True},
        "Northstar": {"enabled": True},
        "Kirkwood": {"enabled": True},
    },
    "thresholds": {
        "snow_24h_inches": 6,        # alert when 24h snow >= this
        "snow_48h_inches": 12,       # alert when 48h snow >= this
        "powder_quality_slr_min": 12, # alert for quality powder (SLR >= this)
        "avalanche_danger_min": 3,    # alert when avy danger >= this (Considerable)
    },
    "notifications": {
        "desktop": True,
        "webhook_url": "",           # set to ntfy/Discord/Slack webhook
        "webhook_method": "POST",
        "log": True,
    },
    "cooldown_hours": 6,  # don't re-alert for same condition within this window
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    # Create default
    tmp = str(CONFIG_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    os.replace(tmp, CONFIG_FILE)
    print(f"Created default config at {CONFIG_FILE}")
    return DEFAULT_CONFIG


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_alerts": {}}


def save_state(state: dict):
    tmp = str(STATE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def log(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def send_desktop(title: str, body: str):
    """Send GNOME desktop notification."""
    try:
        subprocess.run(["notify-send", "--urgency=critical",
                        f"--icon=weather-snow", title, body],
                       timeout=5, check=False)
    except FileNotFoundError:
        pass


def send_webhook(url: str, title: str, body: str):
    """Send to a webhook (ntfy.sh, Discord, Slack, etc.)."""
    if not url:
        return
    try:
        if "ntfy" in url:
            requests.post(url, data=body,
                          headers={"Title": title, "Priority": "high",
                                   "Tags": "snowflake"},
                          timeout=10)
        elif "discord" in url:
            requests.post(url, json={"content": f"**{title}**\n{body}"}, timeout=10)
        elif "slack" in url:
            requests.post(url, json={"text": f"*{title}*\n{body}"}, timeout=10)
        else:
            requests.post(url, json={"title": title, "body": body}, timeout=10)
    except Exception as e:
        log(f"Webhook error: {e}")


def check_cooldown(state: dict, alert_key: str, cooldown_hours: int) -> bool:
    """Return True if we should suppress this alert (still in cooldown)."""
    last = state.get("last_alerts", {}).get(alert_key)
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
        now = datetime.now(timezone.utc)
        return (now - last_dt).total_seconds() < cooldown_hours * 3600
    except Exception:
        return False


def mark_alerted(state: dict, alert_key: str):
    if "last_alerts" not in state:
        state["last_alerts"] = {}
    state["last_alerts"][alert_key] = datetime.now(timezone.utc).isoformat()


def main():
    config = load_config()
    if not config.get("enabled"):
        return

    state = load_state()
    thresholds = config.get("thresholds", {})
    notif = config.get("notifications", {})
    cooldown = config.get("cooldown_hours", 6)

    alerts = []

    # Fetch data and run full blended pipeline via analyze_all()
    log("Checking conditions...")
    tahoe_lat, tahoe_lon = 39.17, -120.145
    obs = fetch_nws_observations(tahoe_lat, tahoe_lon)
    nws = fetch_nws_forecast(tahoe_lat, tahoe_lon)
    om = fetch_open_meteo(tahoe_lat, tahoe_lon)
    snotel = fetch_snotel_current()
    avy = fetch_avalanche()
    afd = fetch_forecast_discussion()

    # Optional enrichment sources (best-effort)
    try:
        nws_grids = fetch_nws_gridpoints(tahoe_lat, tahoe_lon)
    except Exception:
        nws_grids = {}
    try:
        sounding = fetch_sounding("REV")
    except Exception:
        sounding = {}
    try:
        ensemble = fetch_ensemble(tahoe_lat, tahoe_lon)
    except Exception:
        ensemble = {}
    try:
        synoptic = fetch_synoptic_stations(tahoe_lat, tahoe_lon, radius_miles=30)
    except Exception:
        synoptic = {}

    analysis = analyze_all(obs, nws, om, snotel, afd, avy, {},
                           nws_grids=nws_grids if "error" not in nws_grids else None,
                           sounding=sounding if "error" not in sounding else None,
                           ensemble=ensemble if ensemble.get("models") else None,
                           synoptic=synoptic if "error" not in synoptic else None)

    for resort_name, resort_cfg in config.get("resorts", {}).items():
        if not resort_cfg.get("enabled"):
            continue
        if resort_name not in RESORTS:
            continue

        # Use blended data from analyze_all() instead of raw single-model output
        resort_data = analysis.get("resorts", {}).get(resort_name, {})
        peak_z = resort_data.get("zones", {}).get("peak", {})
        timeline = peak_z.get("timeline_48h", [])

        if not timeline:
            log(f"No blended data for {resort_name}")
            continue

        # 24h and 48h snow from blended timeline
        snow_24h = sum(h.get("snowfall_in", 0) for h in timeline[:24])
        snow_48h = sum(h.get("snowfall_in", 0) for h in timeline[:48])

        # Average SLR during snow hours
        snow_hours = [h for h in timeline[:48] if h.get("snowfall_in", 0) > 0]
        avg_slr = (sum(h.get("slr", 0) for h in snow_hours) / len(snow_hours)) if snow_hours else 0

        # Check 24h threshold
        thresh_24 = thresholds.get("snow_24h_inches", 6)
        if snow_24h >= thresh_24:
            key = f"{resort_name}_24h_snow"
            if not check_cooldown(state, key, cooldown):
                alerts.append({
                    "resort": resort_name,
                    "type": "24h Snow Alert",
                    "message": f"{resort_name} peak: {snow_24h:.0f}\" expected in next 24h!",
                })
                mark_alerted(state, key)

        # Check 48h threshold
        thresh_48 = thresholds.get("snow_48h_inches", 12)
        if snow_48h >= thresh_48:
            key = f"{resort_name}_48h_snow"
            if not check_cooldown(state, key, cooldown):
                alerts.append({
                    "resort": resort_name,
                    "type": "48h Snow Alert",
                    "message": f"{resort_name} peak: {snow_48h:.0f}\" expected in next 48h!",
                })
                mark_alerted(state, key)

        # Quality powder check
        slr_min = thresholds.get("powder_quality_slr_min", 12)
        if avg_slr >= slr_min and snow_24h >= 3:
            key = f"{resort_name}_powder_quality"
            if not check_cooldown(state, key, cooldown):
                quality = "light dry powder" if avg_slr >= 14 else "classic powder"
                alerts.append({
                    "resort": resort_name,
                    "type": "Powder Quality Alert",
                    "message": f"{resort_name}: {snow_24h:.0f}\" of {quality} (SLR {avg_slr:.0f}:1) incoming!",
                })
                mark_alerted(state, key)

    # Avalanche check
    avy_min = thresholds.get("avalanche_danger_min", 3)
    if avy and avy.get("danger_level", 0) >= avy_min:
        key = "avalanche_danger"
        if not check_cooldown(state, key, cooldown):
            alerts.append({
                "type": "Avalanche Warning",
                "message": f"Sierra Nevada avalanche danger: {avy['danger_label']}. {avy.get('travel_advice', '')}",
            })
            mark_alerted(state, key)

    # Send alerts
    if alerts:
        for alert in alerts:
            title = f"Tahoe Snow: {alert['type']}"
            body = alert["message"]
            log(f"ALERT: {body}")
            if notif.get("desktop"):
                send_desktop(title, body)
            if notif.get("webhook_url"):
                send_webhook(notif["webhook_url"], title, body)
    else:
        log("No alerts triggered.")

    save_state(state)


if __name__ == "__main__":
    main()
