#!/usr/bin/env python3
"""
Local sensor reading — dual ESP32 architecture.

Both indoor and outdoor ESP32s POST to sensor_server.py, which saves
readings to sensor_data.json. This module reads that file.

Usage:
  from sensors import read_indoor, read_outdoor, read_all
  indoor = read_indoor()    # {"temp_f": 68.2, "humidity_pct": 47, ...}
  outdoor = read_outdoor()  # {"temp_f": 54.3, ...} or {} if stale/missing
  both = read_all()         # {"indoor": {...}, "outdoor": {...}}
"""

import json
import os
from datetime import datetime, timezone

SENSOR_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_data.json")
MAX_AGE_SEC = 900  # Consider data stale after 15 minutes


def _load() -> dict:
    try:
        with open(SENSOR_DATA_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _check_stale(reading: dict) -> dict:
    """Mark reading as stale if older than MAX_AGE_SEC."""
    updated = reading.get("updated")
    if updated:
        try:
            ts = datetime.fromisoformat(updated)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > MAX_AGE_SEC:
                reading["stale"] = True
        except (ValueError, TypeError):
            pass
    return reading


def read_indoor() -> dict:
    """Read latest indoor ESP32 data from sensor_data.json."""
    data = _load()
    indoor = data.get("indoor", {})
    return _check_stale(indoor)


def read_outdoor() -> dict:
    """Read latest outdoor ESP32 data from sensor_data.json."""
    data = _load()
    outdoor = data.get("outdoor", {})
    return _check_stale(outdoor)


def read_all() -> dict:
    """Read both sensors. Returns {"indoor": {...}, "outdoor": {...}}."""
    return {
        "indoor": read_indoor(),
        "outdoor": read_outdoor(),
    }


if __name__ == "__main__":
    result = read_all()
    print(json.dumps(result, indent=2))
