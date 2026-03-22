#!/usr/bin/env python3
"""Integration tests — verify components work together correctly."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from webapp import app, _build_oakland_summary, _cache, _lock


# =========================================================================
# Webapp + data_pipeline integration
# =========================================================================

class TestWebappIntegration:
    """Test the webapp correctly transforms analysis data to API responses."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        with _lock:
            _cache["data"] = None
            _cache["timestamp"] = 0
            _cache["loading"] = False
        yield
        with _lock:
            _cache["data"] = None
            _cache["timestamp"] = 0
            _cache["loading"] = False

    def test_api_data_structure(self):
        """API /api/data returns valid JSON with expected top-level keys."""
        mock = {
            "decision": {"go": True, "reason": "Fresh powder"},
            "storm_narrative": "Storm narrative text",
            "resorts": {"Heavenly": {"zones": {}}},
            "generated": "2025-03-20T12:00:00Z",
        }
        with app.test_client() as client:
            with patch("webapp.get_analysis", return_value=mock):
                resp = client.get("/api/data")
        data = resp.get_json()
        assert resp.status_code == 200
        assert "decision" in data
        assert "resorts" in data

    def test_api_decision_with_storm(self):
        """Decision endpoint correctly extracts storm data."""
        mock = {
            "decision": {"go": True},
            "storm_narrative": "Epic storm",
            "storm_history": [{"date": "2025-03-19", "total": 15}],
        }
        with app.test_client() as client:
            with patch("webapp.get_analysis", return_value=mock):
                resp = client.get("/api/decision")
        data = resp.get_json()
        assert data["storm_narrative"] == "Epic storm"
        assert len(data["storm_history"]) == 1

    def test_concurrent_cache_access(self):
        """Multiple concurrent requests don't corrupt the cache."""
        import threading
        errors = []

        def fetch():
            try:
                with app.test_client() as client:
                    with patch("webapp.fetch_tahoe_analysis", return_value={"ok": True}), \
                         patch("webapp.fetch_oakland_data", side_effect=Exception):
                        resp = client.get("/api/data")
                        data = resp.get_json()
                        if resp.status_code != 200:
                            errors.append(f"Status {resp.status_code}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=fetch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, f"Concurrent access errors: {errors}"


# =========================================================================
# Oakland summary integration
# =========================================================================

class TestOaklandSummaryIntegration:
    def test_full_oakland_response(self):
        """Full Oakland data transforms correctly through the summary builder."""
        oakland = {
            "home_obs": {
                "temp_f": 68, "feels_like_f": 66, "conditions": "Partly Cloudy",
                "humidity_pct": 55, "wind_mph": 8, "wind_gust_mph": 15,
                "wind_dir": "W", "visibility_mi": 10, "barometer_inhg": 30.1,
                "station": "KOAK",
            },
            "home_fc": {
                "periods": [
                    {"name": "Today", "temperature": 72, "temperatureUnit": "F",
                     "shortForecast": "Sunny", "detailedForecast": "Clear all day",
                     "windSpeed": "10 mph", "windDirection": "W",
                     "isDaytime": True,
                     "probabilityOfPrecipitation": {"value": 5}},
                ]
            },
            "home_aqi": {"aqi": 35, "category": "Good", "pm2_5": 6.2},
            "home_om": {
                "daily": {
                    "uv_index_max": [7.5],
                    "sunrise": ["2025-03-20T06:30"],
                    "sunset": ["2025-03-20T19:20"],
                },
                "hourly": {
                    "time": [f"2025-03-20T{h:02d}:00" for h in range(24)],
                    "precipitation": [0.0] * 24,
                },
            },
        }
        result = _build_oakland_summary(oakland)
        # All sections populated
        assert result["current"]["temp_f"] == 68
        assert len(result["forecast"]) == 1
        assert result["aqi"]["aqi"] == 35
        assert result["uv"]["max"] == 7.5
        assert result["sunrise"] == "06:30"
        assert len(result["hourly_rain"]) == 24


# =========================================================================
# ML corrections integration
# =========================================================================

class TestMLCorrectionsIntegration:
    def test_snow_quality_and_wind_hold_consistent(self):
        """Snow quality and wind hold predictions should be internally consistent."""
        from ml_corrections import _predict_snow_quality, _predict_wind_hold

        # Blower powder + calm → best conditions
        quality = _predict_snow_quality(16, 20, 10)
        wind = _predict_wind_hold(10, 8000)
        assert quality["rating"] == 5
        assert wind["probability"] == 0.0

        # Wet snow + extreme wind → worst conditions
        quality = _predict_snow_quality(4, 38, 60)
        wind = _predict_wind_hold(65, 9000)
        assert quality["rating"] == 1
        assert wind["probability"] > 0.5

    def test_resort_ranking_consistent_with_components(self):
        """Resort with better conditions should rank higher."""
        from ml_corrections import _rank_resorts

        resorts = {
            "GoodResort": {
                "zones": {
                    "peak": {
                        "ml_depth_prediction_in": 18,
                        "ml_snow_quality": {"quality": "Blower", "rating": 5},
                        "ml_wind_hold": {"probability": 0.0},
                        "ml_snow_timing": {"snowing": True, "peak_period": "now"},
                    },
                    "base": {"elev_ft": 6200, "current": {"temp_f": 35}},
                },
            },
            "BadResort": {
                "zones": {
                    "peak": {
                        "ml_depth_prediction_in": 1,
                        "ml_snow_quality": {"quality": "Wet", "rating": 1},
                        "ml_wind_hold": {"probability": 0.9},
                        "ml_snow_timing": {},
                    },
                    "base": {"elev_ft": 7500, "current": {"temp_f": 28}},
                },
            },
        }
        rankings = _rank_resorts(resorts)
        assert rankings[0]["resort"] == "GoodResort"
        assert rankings[0]["score"] > rankings[1]["score"]

    def test_powder_alert_matches_conditions(self):
        """Powder alert level should match the depth and quality."""
        from ml_corrections import _compute_powder_alert

        # SEND IT: 15"+, quality 5, low wind
        resorts = {
            "TestResort": {
                "zones": {
                    "peak": {
                        "ml_depth_prediction_in": 15,
                        "ml_snow_quality": {"rating": 5},
                        "ml_wind_hold": {"probability": 0.1},
                    }
                }
            }
        }
        alert = _compute_powder_alert(resorts)
        assert alert["level"] == 5
        assert alert["label"] == "SEND IT"

    def test_feature_extraction_with_complete_data(self):
        """Feature extraction produces all expected keys with real-looking data."""
        from ml_corrections import _extract_live_features

        analysis = {
            "current_conditions": {"observed_lapse_rate_c_km": -6.5},
            "froude_number": 1.2,
            "snotel_current": {
                "Fallen Leaf": {"depth_in": 55, "swe_in": 22, "temp_f": 28}
            },
            "ensemble": {
                "models": {
                    "GFS": [{"snow_p50": 6, "temp_high_p50": 30}],
                    "ECMWF": [{"snow_p50": 8, "temp_high_p50": 28}],
                }
            },
        }
        zone_data = {
            "current": {
                "temp_f": 25, "wind_mph": 20, "wind_gust_mph": 35,
                "wind_dir": "SW", "slr": 14, "dewpoint_f": 15,
            },
            "timeline_48h": [
                {"precip_in": 0.2, "snowfall_in": 1.0} for _ in range(48)
            ],
            "elev_ft": 8800,
        }
        features = _extract_live_features(analysis, zone_data)

        # All critical features present
        assert features["temperature_2m_mean"] == 25
        assert features["wind_speed_10m_max"] == 20
        assert features["slr"] == 14
        assert features["rain_fraction"] == 0.0  # Cold temp
        assert features["snow_fraction"] == 1.0
        assert features["is_ar_event"] == 0 or features["is_ar_event"] == 1
        assert features["depth_lag1d"] == 55
        assert features["station_mean_depth_change"] == 0.09  # Fallen Leaf stats
        assert features["model_snow_spread"] > 0
