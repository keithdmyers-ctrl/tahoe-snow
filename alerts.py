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
from tahoe_snow import (fetch_open_meteo, fetch_snotel_current, fetch_avalanche,
                         parse_open_meteo, aggregate_daily, RESORTS, compute_slr)

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
    with open(CONFIG_FILE, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
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
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


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

    # Fetch data
    log("Checking conditions...")
    om = fetch_open_meteo(39.17, -120.145)
    snotel = fetch_snotel_current()
    avy = fetch_avalanche()

    for resort_name, resort_cfg in config.get("resorts", {}).items():
        if not resort_cfg.get("enabled"):
            continue
        if resort_name not in RESORTS:
            continue

        resort = RESORTS[resort_name]
        peak = resort["peak"]

        # Parse forecast for peak
        parsed = parse_open_meteo(om, peak["elev_ft"])
        if "error" in parsed:
            log(f"Error parsing {resort_name}: {parsed['error']}")
            continue

        gfs = parsed["models"].get("GFS", [])
        if not gfs:
            continue

        # 24h snow
        snow_24h = sum(h["snowfall_in"] for h in gfs[:24])
        snow_48h = sum(h["snowfall_in"] for h in gfs[:48])

        # Best SLR during snow
        snow_hours = [h for h in gfs[:48] if h["snowfall_in"] > 0]
        avg_slr = (sum(h["slr"] for h in snow_hours) / len(snow_hours)) if snow_hours else 0

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
