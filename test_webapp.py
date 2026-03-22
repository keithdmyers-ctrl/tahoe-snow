#!/usr/bin/env python3
"""Tests for webapp.py — Flask routes, caching, API responses."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from webapp import app, _build_oakland_summary, _cache, _lock, CACHE_TTL


@pytest.fixture
def client():
    """Flask test client with fresh cache."""
    app.config["TESTING"] = True
    # Reset cache
    with _lock:
        _cache["data"] = None
        _cache["timestamp"] = 0
        _cache["loading"] = False
    with app.test_client() as c:
        yield c


# =========================================================================
# _build_oakland_summary
# =========================================================================

class TestBuildOaklandSummary:
    def test_empty_input(self):
        result = _build_oakland_summary({})
        assert result["current"] == {}
        assert result["forecast"] == []
        assert result["aqi"] == {}

    def test_observation_data(self):
        oakland = {
            "home_obs": {
                "temp_f": 65,
                "feels_like_f": 63,
                "conditions": "Partly Cloudy",
                "humidity_pct": 55,
                "wind_mph": 8,
                "wind_gust_mph": 12,
                "wind_dir": "W",
                "visibility_mi": 10,
                "barometer_inhg": 30.1,
                "station": "KOAK",
            }
        }
        result = _build_oakland_summary(oakland)
        assert result["current"]["temp_f"] == 65
        assert result["current"]["conditions"] == "Partly Cloudy"
        assert result["current"]["station"] == "KOAK"

    def test_observation_with_error_key(self):
        oakland = {"home_obs": {"error": "API down"}}
        result = _build_oakland_summary(oakland)
        assert result["current"] == {}

    def test_forecast_periods(self):
        oakland = {
            "home_fc": {
                "periods": [
                    {
                        "name": "Today",
                        "temperature": 72,
                        "temperatureUnit": "F",
                        "shortForecast": "Sunny",
                        "detailedForecast": "Clear skies all day",
                        "windSpeed": "10 mph",
                        "windDirection": "W",
                        "isDaytime": True,
                        "probabilityOfPrecipitation": {"value": 5},
                    },
                    {
                        "name": "Tonight",
                        "temperature": 55,
                        "temperatureUnit": "F",
                        "shortForecast": "Clear",
                        "detailedForecast": "Clear tonight",
                        "windSpeed": "5 mph",
                        "windDirection": "NW",
                        "isDaytime": False,
                        "probabilityOfPrecipitation": {"value": 0},
                    },
                ]
            }
        }
        result = _build_oakland_summary(oakland)
        assert len(result["forecast"]) == 2
        assert result["forecast"][0]["name"] == "Today"
        assert result["forecast"][0]["temp"] == 72
        assert result["forecast"][1]["is_daytime"] is False

    def test_forecast_truncated_to_8(self):
        oakland = {"home_fc": {"periods": [{"name": f"P{i}"} for i in range(15)]}}
        result = _build_oakland_summary(oakland)
        assert len(result["forecast"]) == 8

    def test_aqi_data(self):
        oakland = {"home_aqi": {"aqi": 42, "category": "Good", "pm2_5": 8.5}}
        result = _build_oakland_summary(oakland)
        assert result["aqi"]["aqi"] == 42
        assert result["aqi"]["category"] == "Good"

    def test_aqi_with_error(self):
        oakland = {"home_aqi": {"error": "API down"}}
        result = _build_oakland_summary(oakland)
        assert result["aqi"] == {}

    def test_uv_and_sunrise_sunset(self):
        oakland = {
            "home_om": {
                "daily": {
                    "uv_index_max": [8.5],
                    "sunrise": ["2025-03-20T06:45"],
                    "sunset": ["2025-03-20T19:15"],
                }
            }
        }
        result = _build_oakland_summary(oakland)
        assert result["uv"]["max"] == 8.5
        assert result["sunrise"] == "06:45"
        assert result["sunset"] == "19:15"

    def test_hourly_rain(self):
        oakland = {
            "home_om": {
                "hourly": {
                    "time": [f"2025-03-20T{h:02d}:00" for h in range(24)],
                    "precipitation": [0.0] * 12 + [0.5] * 12,
                }
            }
        }
        result = _build_oakland_summary(oakland)
        assert len(result["hourly_rain"]) == 24
        assert result["hourly_rain"][0]["precip_mm"] == 0.0
        assert result["hourly_rain"][12]["precip_mm"] == 0.5

    def test_hourly_rain_with_none_values(self):
        oakland = {
            "home_om": {
                "hourly": {
                    "time": ["2025-03-20T00:00"],
                    "precipitation": [None],
                }
            }
        }
        result = _build_oakland_summary(oakland)
        assert result["hourly_rain"][0]["precip_mm"] == 0


# =========================================================================
# API Routes
# =========================================================================

class TestApiRoutes:
    def test_api_data(self, client):
        mock_analysis = {
            "decision": {"go": True},
            "generated": "2025-03-20T12:00:00Z",
        }
        with patch("webapp.get_analysis", return_value=mock_analysis):
            resp = client.get("/api/data")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["decision"]["go"] is True

    def test_api_decision(self, client):
        mock_analysis = {
            "decision": {"go": True, "reason": "Powder day"},
            "storm_narrative": "Big storm hitting tonight",
            "storm_history": [{"date": "2025-03-19"}],
        }
        with patch("webapp.get_analysis", return_value=mock_analysis):
            resp = client.get("/api/decision")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["decision"]["go"] is True
        assert "Big storm" in data["storm_narrative"]

    def test_api_decision_error(self, client):
        with patch("webapp.get_analysis", return_value={"error": "Data unavailable"}):
            resp = client.get("/api/decision")
        assert resp.status_code == 503

    def test_api_verification(self, client):
        mock_summary = {"model_scores": {"GFS": 0.8}}
        with patch("webapp.get_verification_summary", return_value=mock_summary):
            resp = client.get("/api/verification")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model_scores" in data

    def test_api_verification_error(self, client):
        with patch("webapp.get_verification_summary", side_effect=Exception("DB error")):
            resp = client.get("/api/verification")
        assert resp.status_code == 500
        assert "error" in resp.get_json()

    def test_api_activities_no_engine(self, client):
        with patch("webapp.compute_activity_decisions", None):
            resp = client.get("/api/activities")
        assert resp.status_code == 503

    def test_api_activities_success(self, client):
        mock_analysis = {"generated": "2025-03-20T12:00:00Z"}
        mock_cards = [{"activity": "Ski", "score": 9}]
        with patch("webapp.get_analysis", return_value=mock_analysis), \
             patch("webapp.compute_activity_decisions", return_value=mock_cards):
            resp = client.get("/api/activities")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["activities"]) == 1
        assert data["activities"][0]["activity"] == "Ski"

    def test_api_activities_error_in_analysis(self, client):
        with patch("webapp.get_analysis", return_value={"error": "Loading..."}):
            resp = client.get("/api/activities")
        assert resp.status_code == 503

    def test_clear_cache_returns_html(self, client):
        resp = client.get("/clear-cache")
        assert resp.status_code == 200
        assert b"Clearing cache" in resp.data


# =========================================================================
# Cache + Refresh
# =========================================================================

class TestCacheAndRefresh:
    def test_refresh_throttled(self, client):
        # Pre-populate cache with recent timestamp
        with _lock:
            _cache["data"] = {"decision": {}}
            _cache["timestamp"] = time.time()
            _cache["loading"] = False
        resp = client.get("/api/refresh")
        assert resp.status_code == 429
        data = resp.get_json()
        assert data["status"] == "throttled"
        assert "retry_after" in data

    def test_refresh_allowed_after_cooldown(self, client):
        with _lock:
            _cache["data"] = {"decision": {}}
            _cache["timestamp"] = time.time() - 120  # 2 min ago
            _cache["loading"] = False
        mock_analysis = {"generated": "fresh"}
        with patch("webapp.fetch_tahoe_analysis", return_value=mock_analysis), \
             patch("webapp.fetch_oakland_data", side_effect=Exception("skip")):
            resp = client.get("/api/refresh")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_get_analysis_returns_cached(self, client):
        """Cached data is returned without re-fetching."""
        with _lock:
            _cache["data"] = {"cached": True}
            _cache["timestamp"] = time.time()
            _cache["loading"] = False
        with patch("webapp.fetch_tahoe_analysis") as mock_fetch:
            resp = client.get("/api/data")
        mock_fetch.assert_not_called()
        assert resp.get_json()["cached"] is True

    def test_get_analysis_fetches_when_stale(self, client):
        """Stale cache triggers a fresh fetch."""
        with _lock:
            _cache["data"] = {"stale": True}
            _cache["timestamp"] = time.time() - CACHE_TTL - 10
            _cache["loading"] = False
        fresh = {"fresh": True}
        with patch("webapp.fetch_tahoe_analysis", return_value=fresh), \
             patch("webapp.fetch_oakland_data", side_effect=Exception("skip")):
            resp = client.get("/api/data")
        data = resp.get_json()
        assert data.get("fresh") is True
