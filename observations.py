#!/usr/bin/env python3
"""
Crowdsourced Observation System for Tahoe Snow

Allows recording and querying ground-truth weather and snow observations
from field reporters (ski patrol, backcountry users, road crews).

Used for:
- Verification of model forecasts against actual conditions
- Real-time snow depth / quality updates between SNOTEL readings
- Road condition reports (chains, closures, visibility)
- Ground-truth training data for ML pipeline

Storage: JSON file-backed with atomic writes, FIFO eviction at 10k entries.
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBSERVATIONS_FILE = os.path.join(_BASE_DIR, ".observations.json")
MAX_OBSERVATIONS = 10000


class ObsType(Enum):
    """Types of field observations."""
    SNOW_DEPTH = "snow_depth"
    SNOW_QUALITY = "snow_quality"
    CONDITIONS = "conditions"
    ROAD = "road"
    CUSTOM = "custom"


@dataclass
class Observation:
    """A single ground-truth observation from a field reporter."""
    timestamp: str  # ISO 8601 UTC
    observer_id: str  # Unique identifier for the reporter
    lat: float
    lon: float
    elevation_ft: float
    location_name: str  # Human-readable location (e.g., "Heavenly Peak")
    obs_type: str  # ObsType value string
    value: float  # Numeric value (inches for depth, 1-5 for quality, etc.)
    unit: str  # "inches", "rating", "boolean", etc.
    notes: str = ""
    photo_url: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        """Deserialize from dict."""
        return cls(
            timestamp=d["timestamp"],
            observer_id=d["observer_id"],
            lat=d["lat"],
            lon=d["lon"],
            elevation_ft=d["elevation_ft"],
            location_name=d["location_name"],
            obs_type=d["obs_type"],
            value=d["value"],
            unit=d["unit"],
            notes=d.get("notes", ""),
            photo_url=d.get("photo_url"),
        )


class ObservationValidator:
    """Validates observations for range, duplicates, and outliers."""

    # Range limits by observation type
    RANGES = {
        ObsType.SNOW_DEPTH.value: (0, 200),       # inches
        ObsType.SNOW_QUALITY.value: (1, 5),        # 1-5 rating
        ObsType.CONDITIONS.value: (-40, 120),      # temp F
        ObsType.ROAD.value: (0, 4),                # 0=clear, 1=chains, 2=R2, 3=closed, 4=unknown
        ObsType.CUSTOM.value: (-1e6, 1e6),         # arbitrary
    }

    def validate(self, obs: Observation, existing: list) -> tuple:
        """Validate an observation.

        Args:
            obs: Observation to validate.
            existing: List of existing observation dicts for duplicate check.

        Returns:
            Tuple of (is_valid: bool, errors: list[str], warnings: list[str]).
        """
        errors = []
        warnings = []

        # Range check
        range_err = self._check_range(obs)
        if range_err:
            errors.append(range_err)

        # Coordinate sanity (Tahoe area roughly)
        if not (37.5 <= obs.lat <= 40.5):
            errors.append(f"Latitude {obs.lat} outside valid range (37.5-40.5)")
        if not (-121.5 <= obs.lon <= -119.0):
            errors.append(f"Longitude {obs.lon} outside valid range (-121.5 to -119.0)")

        # Elevation sanity
        if not (0 <= obs.elevation_ft <= 15000):
            errors.append(f"Elevation {obs.elevation_ft}ft outside valid range (0-15000)")

        # Duplicate check
        dup = self._check_duplicate(obs, existing)
        if dup:
            errors.append(dup)

        # Observer ID required
        if not obs.observer_id or not obs.observer_id.strip():
            errors.append("observer_id is required")

        # Timestamp validity
        try:
            datetime.fromisoformat(obs.timestamp)
        except (ValueError, TypeError):
            errors.append(f"Invalid timestamp format: {obs.timestamp}")

        return (len(errors) == 0, errors, warnings)

    def check_outlier(self, obs: Observation, forecast_value: Optional[float] = None,
                      recent_obs: Optional[list] = None) -> Optional[str]:
        """Flag observations that are outliers relative to forecast or recent obs.

        Args:
            obs: Observation to check.
            forecast_value: Current model forecast for comparison.
            recent_obs: Recent observations of the same type nearby.

        Returns:
            Warning string if outlier detected, None otherwise.
        """
        values = []
        if recent_obs:
            values = [o.get("value", 0) for o in recent_obs
                      if o.get("obs_type") == obs.obs_type]

        if forecast_value is not None:
            values.append(forecast_value)

        if len(values) < 3:
            return None

        import numpy as np
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std > 0 and abs(obs.value - mean) > 3 * std:
            return (
                f"Outlier: value {obs.value} is >3 sigma from mean {mean:.1f} "
                f"(std={std:.1f})"
            )
        return None

    def _check_range(self, obs: Observation) -> Optional[str]:
        """Check if observation value is within expected range."""
        limits = self.RANGES.get(obs.obs_type)
        if limits is None:
            return None
        low, high = limits
        if not (low <= obs.value <= high):
            return (
                f"Value {obs.value} outside range [{low}, {high}] "
                f"for type {obs.obs_type}"
            )
        return None

    def _check_duplicate(self, obs: Observation, existing: list) -> Optional[str]:
        """Check for duplicate observation (same observer+location+type within 1 hour)."""
        try:
            obs_time = datetime.fromisoformat(obs.timestamp)
        except (ValueError, TypeError):
            return None

        for e in existing:
            if (e.get("observer_id") == obs.observer_id and
                    e.get("obs_type") == obs.obs_type and
                    e.get("location_name") == obs.location_name):
                try:
                    e_time = datetime.fromisoformat(e["timestamp"])
                    if abs((obs_time - e_time).total_seconds()) < 3600:
                        return (
                            f"Duplicate: same observer/type/location within 1 hour "
                            f"(existing at {e['timestamp']})"
                        )
                except (ValueError, TypeError, KeyError):
                    continue
        return None


class ObservationStore:
    """JSON file-backed observation storage with spatial queries."""

    def __init__(self, filepath: str = OBSERVATIONS_FILE):
        self._filepath = filepath
        self._validator = ObservationValidator()

    def _load(self) -> list:
        """Load observations from disk."""
        if not os.path.exists(self._filepath):
            return []
        try:
            with open(self._filepath) as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get("observations", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, observations: list):
        """Atomically save observations to disk."""
        tmp = self._filepath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(observations, f, indent=2)
        os.replace(tmp, self._filepath)

    def add(self, obs: Observation) -> dict:
        """Validate and store a new observation.

        Args:
            obs: Observation to add.

        Returns:
            Dict with "success" bool, "errors" list, and "warnings" list.
        """
        existing = self._load()
        is_valid, errors, warnings = self._validator.validate(obs, existing)

        if not is_valid:
            return {"success": False, "errors": errors, "warnings": warnings}

        # Check for outlier (warning only, doesn't block)
        outlier_warn = self._validator.check_outlier(obs, recent_obs=existing[-50:])
        if outlier_warn:
            warnings.append(outlier_warn)

        # Add observation
        existing.append(obs.to_dict())

        # FIFO eviction
        if len(existing) > MAX_OBSERVATIONS:
            existing = existing[-MAX_OBSERVATIONS:]

        self._save(existing)
        return {"success": True, "errors": [], "warnings": warnings}

    def get_recent(self, hours: int = 24, obs_type: Optional[str] = None,
                   location: Optional[str] = None) -> list:
        """Get recent observations, optionally filtered.

        Args:
            hours: How many hours back to query.
            obs_type: Filter by ObsType value (e.g., "snow_depth").
            location: Filter by location_name substring (case-insensitive).

        Returns:
            List of observation dicts, newest first.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        observations = self._load()

        results = []
        for o in observations:
            ts = o.get("timestamp", "")
            if ts < cutoff:
                continue
            if obs_type and o.get("obs_type") != obs_type:
                continue
            if location and location.lower() not in o.get("location_name", "").lower():
                continue
            results.append(o)

        # Newest first
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    def get_near(self, lat: float, lon: float, radius_miles: float = 5,
                 hours: int = 24) -> list:
        """Get observations near a point within a time window.

        Uses Haversine distance for spatial filtering.

        Args:
            lat: Center latitude.
            lon: Center longitude.
            radius_miles: Search radius in miles.
            hours: How many hours back to query.

        Returns:
            List of observation dicts with added "distance_mi" field.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        observations = self._load()

        results = []
        for o in observations:
            ts = o.get("timestamp", "")
            if ts < cutoff:
                continue

            o_lat = o.get("lat")
            o_lon = o.get("lon")
            if o_lat is None or o_lon is None:
                continue

            dist = _haversine_miles(lat, lon, o_lat, o_lon)
            if dist <= radius_miles:
                result = dict(o)
                result["distance_mi"] = round(dist, 2)
                results.append(result)

        results.sort(key=lambda x: x.get("distance_mi", 999))
        return results

    def get_summary(self) -> dict:
        """Get counts by type and recent activity summary.

        Returns:
            Dict with counts_by_type, total_count, recent_24h, oldest, newest.
        """
        observations = self._load()
        if not observations:
            return {
                "total_count": 0,
                "counts_by_type": {},
                "recent_24h": 0,
                "oldest": None,
                "newest": None,
            }

        cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

        counts_by_type = {}
        recent_24h = 0
        timestamps = []

        for o in observations:
            otype = o.get("obs_type", "unknown")
            counts_by_type[otype] = counts_by_type.get(otype, 0) + 1

            ts = o.get("timestamp", "")
            if ts:
                timestamps.append(ts)
                if ts >= cutoff_24h:
                    recent_24h += 1

        timestamps.sort()

        return {
            "total_count": len(observations),
            "counts_by_type": counts_by_type,
            "recent_24h": recent_24h,
            "oldest": timestamps[0] if timestamps else None,
            "newest": timestamps[-1] if timestamps else None,
        }


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in miles."""
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# === Flask routes for webapp.py ===
# POST /api/observation — submit new observation
#   Body: {"observer_id": "...", "lat": ..., "lon": ..., "elevation_ft": ...,
#          "location_name": "...", "obs_type": "snow_depth", "value": 24,
#          "unit": "inches", "notes": "Fresh powder at summit"}
#   Response: {"success": true, "warnings": [...]}
#
# GET /api/observations — get recent observations
#   Query params: hours=24, obs_type=snow_depth, location=Heavenly
#   Response: [{"timestamp": "...", "observer_id": "...", ...}, ...]
#
# GET /api/observations/summary — observation counts and activity
#   Response: {"total_count": 150, "counts_by_type": {...}, "recent_24h": 12}
#
# Example integration in webapp.py:
#
#   from observations import ObservationStore, Observation, ObsType
#
#   obs_store = ObservationStore()
#
#   @app.route("/api/observation", methods=["POST"])
#   def api_add_observation():
#       data = request.get_json()
#       if not data:
#           return jsonify({"success": False, "errors": ["No JSON body"]}), 400
#       obs = Observation(
#           timestamp=datetime.now(timezone.utc).isoformat(),
#           observer_id=data.get("observer_id", ""),
#           lat=data.get("lat", 0),
#           lon=data.get("lon", 0),
#           elevation_ft=data.get("elevation_ft", 0),
#           location_name=data.get("location_name", ""),
#           obs_type=data.get("obs_type", "custom"),
#           value=data.get("value", 0),
#           unit=data.get("unit", ""),
#           notes=data.get("notes", ""),
#           photo_url=data.get("photo_url"),
#       )
#       result = obs_store.add(obs)
#       status = 200 if result["success"] else 400
#       return jsonify(result), status
#
#   @app.route("/api/observations")
#   def api_get_observations():
#       hours = request.args.get("hours", 24, type=int)
#       obs_type = request.args.get("obs_type")
#       location = request.args.get("location")
#       return jsonify(obs_store.get_recent(hours, obs_type, location))
#
#   @app.route("/api/observations/summary")
#   def api_observations_summary():
#       return jsonify(obs_store.get_summary())


# === Integration with tahoe_snow.py ===
# To use observations in analysis, add to analyze_all():
#
#   from observations import ObservationStore
#   obs_store = ObservationStore()
#   recent_obs = obs_store.get_recent(hours=6)
#   # Override model values with ground truth where available
#   for obs in recent_obs:
#       if obs.obs_type == ObsType.SNOW_DEPTH:
#           # Find nearest zone and update current snow depth
