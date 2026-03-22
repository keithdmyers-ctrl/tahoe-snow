#!/usr/bin/env python3
"""Tests for ml_corrections.py — snow quality, wind holds, timing, ranking, alerts."""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from ml_corrections import (
    _compute_best_window,
    _compute_powder_alert,
    _dir_to_deg,
    _elev_to_band,
    _extract_live_features,
    _log_shadow_prediction,
    _predict_chains,
    _predict_snow_quality,
    _predict_snow_timing,
    _predict_wind_hold,
    _quality_name,
    _rank_resorts,
    _build_ml_outdoor_from_outdoor,
    get_drift_metrics,
    SHADOW_LOG,
)


# =========================================================================
# _dir_to_deg
# =========================================================================

class TestDirToDeg:
    def test_cardinal_directions(self):
        assert _dir_to_deg("N") == 0
        assert _dir_to_deg("E") == 90
        assert _dir_to_deg("S") == 180
        assert _dir_to_deg("W") == 270

    def test_intercardinal_directions(self):
        assert _dir_to_deg("NE") == 45
        assert _dir_to_deg("SE") == 135
        assert _dir_to_deg("SW") == 225
        assert _dir_to_deg("NW") == 315

    def test_16_point_directions(self):
        assert _dir_to_deg("NNE") == 22.5
        assert _dir_to_deg("SSW") == 202.5
        assert _dir_to_deg("WNW") == 292.5

    def test_case_insensitive(self):
        assert _dir_to_deg("n") == 0
        assert _dir_to_deg("ne") == 45
        assert _dir_to_deg("ssw") == 202.5

    def test_invalid_returns_nan(self):
        assert np.isnan(_dir_to_deg(""))
        assert np.isnan(_dir_to_deg("INVALID"))
        assert np.isnan(_dir_to_deg("X"))

    def test_numeric_passthrough(self):
        # Non-string input gets str() then lookup — should return nan
        assert np.isnan(_dir_to_deg(180))


# =========================================================================
# _elev_to_band
# =========================================================================

class TestElevToBand:
    def test_base(self):
        assert _elev_to_band(6000) == "base"
        assert _elev_to_band(6999) == "base"

    def test_mid(self):
        assert _elev_to_band(7000) == "mid"
        assert _elev_to_band(8499) == "mid"

    def test_peak(self):
        assert _elev_to_band(8500) == "peak"
        assert _elev_to_band(10000) == "peak"

    def test_boundary_values(self):
        assert _elev_to_band(7000) == "mid"  # exactly 7000
        assert _elev_to_band(8500) == "peak"  # exactly 8500


# =========================================================================
# _predict_snow_quality
# =========================================================================

class TestSnowQuality:
    def test_blower_powder(self):
        result = _predict_snow_quality(slr=16, temp_f=20, wind_mph=10)
        assert result["quality"] == "Blower"
        assert result["rating"] == 5

    def test_cold_smoke(self):
        result = _predict_snow_quality(slr=13, temp_f=26, wind_mph=25)
        assert result["quality"] == "Cold Smoke"
        assert result["rating"] == 5

    def test_powder(self):
        result = _predict_snow_quality(slr=10, temp_f=30, wind_mph=10)
        assert result["quality"] == "Powder"
        assert result["rating"] == 4

    def test_average(self):
        result = _predict_snow_quality(slr=8, temp_f=33, wind_mph=10)
        assert result["quality"] == "Average"
        assert result["rating"] == 3

    def test_heavy(self):
        result = _predict_snow_quality(slr=6, temp_f=35, wind_mph=5)
        assert result["quality"] == "Heavy"
        assert result["rating"] == 2

    def test_wet(self):
        result = _predict_snow_quality(slr=4, temp_f=38, wind_mph=5)
        assert result["quality"] == "Wet"
        assert result["rating"] == 1

    def test_unknown_nan_slr(self):
        result = _predict_snow_quality(slr=np.nan, temp_f=30, wind_mph=10)
        assert result["quality"] == "Unknown"
        assert result["rating"] == 0

    def test_unknown_zero_slr(self):
        result = _predict_snow_quality(slr=0, temp_f=30, wind_mph=10)
        assert result["quality"] == "Unknown"
        assert result["rating"] == 0

    def test_blower_blocked_by_high_wind(self):
        # SLR 15 but wind 25 blocks "Blower" threshold
        result = _predict_snow_quality(slr=15, temp_f=20, wind_mph=25)
        assert result["quality"] == "Cold Smoke"  # Falls to next tier


# =========================================================================
# _predict_wind_hold
# =========================================================================

class TestWindHold:
    def test_calm(self):
        result = _predict_wind_hold(wind_gust_mph=10, elev_ft=7000)
        assert result["probability"] == 0.0
        assert result["risk"] == "None"

    def test_low_risk(self):
        result = _predict_wind_hold(wind_gust_mph=35, elev_ft=7000)
        assert result["probability"] == 0.10
        assert result["risk"] == "Low"

    def test_moderate_risk(self):
        result = _predict_wind_hold(wind_gust_mph=45, elev_ft=7000)
        assert result["probability"] == 0.35
        assert result["risk"] == "Moderate"

    def test_high_risk(self):
        result = _predict_wind_hold(wind_gust_mph=55, elev_ft=7000)
        assert result["probability"] == 0.70
        assert result["risk"] == "High"

    def test_very_high_risk(self):
        result = _predict_wind_hold(wind_gust_mph=65, elev_ft=7000)
        assert result["probability"] == 0.95
        assert result["risk"] == "Very High"

    def test_elevation_factor_increases_risk(self):
        # Same gusts, higher elevation → higher effective gust → higher risk
        low = _predict_wind_hold(wind_gust_mph=45, elev_ft=7000)
        high = _predict_wind_hold(wind_gust_mph=45, elev_ft=9500)
        assert high["probability"] >= low["probability"]

    def test_nan_gusts(self):
        result = _predict_wind_hold(wind_gust_mph=np.nan, elev_ft=8000)
        assert result["probability"] == 0
        assert result["risk"] == "Unknown"


# =========================================================================
# _predict_snow_timing
# =========================================================================

class TestSnowTiming:
    def test_no_timeline(self):
        result = _predict_snow_timing([])
        assert result["available"] is False

    def test_no_snow(self):
        timeline = [{"snowfall_in": 0} for _ in range(48)]
        result = _predict_snow_timing(timeline)
        assert result["snowing"] is False
        assert "No snow" in result["summary"]

    def test_snow_starting_now(self):
        timeline = [{"snowfall_in": 1.0}] + [{"snowfall_in": 0.5}] * 11 + [{"snowfall_in": 0}] * 36
        result = _predict_snow_timing(timeline)
        assert result["snowing"] is True
        assert result["start_hour"] == 0
        assert "starting now" in result["summary"]
        assert result["total_24h_in"] > 0

    def test_snow_delayed(self):
        timeline = [{"snowfall_in": 0}] * 6 + [{"snowfall_in": 2.0}] * 6 + [{"snowfall_in": 0}] * 36
        result = _predict_snow_timing(timeline)
        assert result["start_hour"] == 6
        assert "starting in ~6h" in result["summary"]

    def test_peak_period_identification(self):
        # Heaviest snow 0-12h
        timeline = [{"snowfall_in": 3.0}] * 5 + [{"snowfall_in": 0}] * 43
        result = _predict_snow_timing(timeline)
        assert "next 12 hours" in result["summary"]

    def test_peak_mid_range(self):
        # Heaviest snow 12-24h
        timeline = [{"snowfall_in": 0.1}] * 12 + [{"snowfall_in": 3.0}] * 5 + [{"snowfall_in": 0}] * 31
        result = _predict_snow_timing(timeline)
        assert "12-24 hours" in result["summary"]

    def test_accumulation_totals(self):
        timeline = [{"snowfall_in": 1.0}] * 48
        result = _predict_snow_timing(timeline)
        assert result["total_12h_in"] == 12.0
        assert result["total_24h_in"] == 24.0
        assert result["total_48h_in"] == 48.0

    def test_short_timeline(self):
        """Timeline shorter than 48h should not crash."""
        timeline = [{"snowfall_in": 0.5}] * 6
        result = _predict_snow_timing(timeline)
        assert result["snowing"] is True
        assert result["total_12h_in"] == 3.0  # Only 6h of data

    def test_none_snowfall_values(self):
        """None snowfall values should be treated as 0."""
        timeline = [{"snowfall_in": None}] * 24 + [{"snowfall_in": 1.0}] * 24
        result = _predict_snow_timing(timeline)
        assert result["start_hour"] == 24

    def test_missing_snowfall_key(self):
        """Missing snowfall_in key should be treated as 0."""
        timeline = [{}] * 48
        result = _predict_snow_timing(timeline)
        assert result["snowing"] is False

    def test_late_peak(self):
        """Peak in 24-48h range."""
        timeline = ([{"snowfall_in": 0}] * 24 +
                    [{"snowfall_in": 0.5}] * 5 +
                    [{"snowfall_in": 3.0}] * 1 +
                    [{"snowfall_in": 0.5}] * 18)
        result = _predict_snow_timing(timeline)
        assert "24-48 hours" in result["summary"]


# =========================================================================
# _rank_resorts
# =========================================================================

class TestRankResorts:
    def _make_resort(self, depth=0, quality_rating=3, wind_prob=0, base_temp=35, base_elev=6200):
        return {
            "zones": {
                "peak": {
                    "ml_depth_prediction_in": depth,
                    "ml_snow_quality": {"quality": "Powder", "rating": quality_rating},
                    "ml_wind_hold": {"probability": wind_prob},
                    "ml_snow_timing": {},
                },
                "base": {
                    "elev_ft": base_elev,
                    "current": {"temp_f": base_temp},
                },
            }
        }

    def test_empty_resorts(self):
        result = _rank_resorts({})
        assert result == []

    def test_single_resort(self):
        resorts = {"Heavenly": self._make_resort(depth=10, quality_rating=4)}
        result = _rank_resorts(resorts)
        assert len(result) == 1
        assert result[0]["resort"] == "Heavenly"
        assert result[0]["score"] > 0

    def test_ranking_order(self):
        resorts = {
            "Good": self._make_resort(depth=15, quality_rating=5, wind_prob=0),
            "Bad": self._make_resort(depth=2, quality_rating=1, wind_prob=0.8),
        }
        result = _rank_resorts(resorts)
        assert result[0]["resort"] == "Good"
        assert result[1]["resort"] == "Bad"
        assert result[0]["score"] > result[1]["score"]

    def test_recommendation_excellent(self):
        resorts = {"Palisades": self._make_resort(depth=20, quality_rating=5, wind_prob=0)}
        result = _rank_resorts(resorts)
        assert "Excellent" in result[0].get("recommendation", "")

    def test_recommendation_marginal(self):
        resorts = {"MtRose": self._make_resort(depth=0, quality_rating=1, wind_prob=0.9)}
        result = _rank_resorts(resorts)
        assert "marginal" in result[0].get("recommendation", "")

    def test_score_components_present(self):
        resorts = {"Northstar": self._make_resort(depth=8)}
        result = _rank_resorts(resorts)
        r = result[0]
        for key in ["snow_score", "quality_score", "wind_score", "access_score", "score"]:
            assert key in r

    def test_missing_quality_data_no_inflation(self):
        """Resort with missing ml_snow_quality should not get inflated score."""
        resorts = {
            "HasData": self._make_resort(depth=10, quality_rating=4),
            "NoData": {
                "zones": {
                    "peak": {
                        "ml_depth_prediction_in": 10,
                        # ml_snow_quality is MISSING
                        "ml_wind_hold": {"probability": 0},
                        "ml_snow_timing": {},
                    },
                    "base": {"elev_ft": 6200, "current": {"temp_f": 35}},
                },
            },
        }
        result = _rank_resorts(resorts)
        has_data = [r for r in result if r["resort"] == "HasData"][0]
        no_data = [r for r in result if r["resort"] == "NoData"][0]
        # Resort with actual quality data should score higher
        assert has_data["quality_score"] > no_data["quality_score"]
        assert no_data["quality_score"] == 0  # No data → 0 quality score

    def test_missing_zones_still_ranks(self):
        """Resort with empty zones should still be ranked (no crash)."""
        resorts = {"EmptyResort": {"zones": {}}}
        result = _rank_resorts(resorts)
        assert len(result) == 1
        assert result[0]["score"] >= 0

    def test_only_mid_zone(self):
        """Resort with only mid zone (no peak) should still rank."""
        resorts = {
            "MidOnly": {
                "zones": {
                    "mid": {
                        "ml_depth_prediction_in": 8,
                        "ml_snow_quality": {"quality": "Powder", "rating": 4},
                        "ml_wind_hold": {"probability": 0.1},
                        "ml_snow_timing": {},
                    },
                    "base": {"elev_ft": 6500, "current": {"temp_f": 30}},
                },
            },
        }
        result = _rank_resorts(resorts)
        assert len(result) == 1
        assert result[0]["score"] > 0

    def test_wind_reduces_score(self):
        """High wind should significantly reduce resort score."""
        calm = self._make_resort(depth=10, quality_rating=4, wind_prob=0.0)
        windy = self._make_resort(depth=10, quality_rating=4, wind_prob=0.9)
        resorts = {"Calm": calm, "Windy": windy}
        result = _rank_resorts(resorts)
        calm_r = [r for r in result if r["resort"] == "Calm"][0]
        windy_r = [r for r in result if r["resort"] == "Windy"][0]
        assert calm_r["wind_score"] > windy_r["wind_score"]
        assert calm_r["score"] > windy_r["score"]

    def test_access_score_warm_base(self):
        """Warm base temp → high access score (no chains)."""
        resorts = {"Warm": self._make_resort(base_temp=40, base_elev=6200)}
        result = _rank_resorts(resorts)
        assert result[0]["access_score"] == 9

    def test_access_score_cold_base(self):
        """Cold base temp → low access score (chains likely)."""
        resorts = {"Cold": self._make_resort(base_temp=25, base_elev=7500)}
        result = _rank_resorts(resorts)
        assert result[0]["access_score"] == 3


# =========================================================================
# _compute_powder_alert
# =========================================================================

class TestPowderAlert:
    def _make_resorts(self, depth=0, quality=3, wind_prob=0):
        return {
            "TestResort": {
                "zones": {
                    "peak": {
                        "ml_depth_prediction_in": depth,
                        "ml_snow_quality": {"rating": quality},
                        "ml_wind_hold": {"probability": wind_prob},
                    }
                }
            }
        }

    def test_send_it(self):
        result = _compute_powder_alert(self._make_resorts(depth=15, quality=5, wind_prob=0.1))
        assert result["level"] == 5
        assert result["label"] == "SEND IT"

    def test_strong(self):
        result = _compute_powder_alert(self._make_resorts(depth=10, quality=4, wind_prob=0.5))
        assert result["level"] == 4
        assert result["label"] == "STRONG"

    def test_moderate(self):
        result = _compute_powder_alert(self._make_resorts(depth=5, quality=2, wind_prob=0))
        assert result["level"] == 3
        assert result["label"] == "MODERATE"

    def test_low(self):
        result = _compute_powder_alert(self._make_resorts(depth=2, quality=2, wind_prob=0))
        assert result["level"] == 2
        assert result["label"] == "LOW"

    def test_none(self):
        result = _compute_powder_alert(self._make_resorts(depth=0, quality=0, wind_prob=0))
        assert result["level"] == 1
        assert result["label"] == "NONE"

    def test_empty_resorts(self):
        result = _compute_powder_alert({})
        assert result["level"] == 1
        assert result["label"] == "NONE"

    def test_all_resorts_no_peak(self):
        """Resorts with only base zone (no peak/mid) should not crash."""
        resorts = {
            "NoPeak": {
                "zones": {
                    "base": {
                        "ml_depth_prediction_in": 5,
                        "ml_snow_quality": {"rating": 3},
                        "ml_wind_hold": {"probability": 0},
                    }
                }
            }
        }
        result = _compute_powder_alert(resorts)
        # Should still produce a valid alert (base is skipped, peak/mid is empty)
        assert result["level"] >= 1

    def test_none_message_does_not_contain_none_resort(self):
        """NONE alert should not reference a resort name."""
        result = _compute_powder_alert(self._make_resorts(depth=0, quality=0))
        assert "None" not in result["message"] or result["best_resort"] is None

    def test_send_it_includes_resort_name(self):
        result = _compute_powder_alert(self._make_resorts(depth=15, quality=5, wind_prob=0.1))
        assert "TestResort" in result["message"]
        assert result["best_resort"] == "TestResort"

    def test_multiple_resorts_picks_best(self):
        resorts = {
            "GoodResort": {
                "zones": {"peak": {
                    "ml_depth_prediction_in": 18,
                    "ml_snow_quality": {"rating": 5},
                    "ml_wind_hold": {"probability": 0},
                }}
            },
            "BadResort": {
                "zones": {"peak": {
                    "ml_depth_prediction_in": 2,
                    "ml_snow_quality": {"rating": 1},
                    "ml_wind_hold": {"probability": 0.8},
                }}
            },
        }
        result = _compute_powder_alert(resorts)
        assert result["best_resort"] == "GoodResort"


# =========================================================================
# _compute_best_window
# =========================================================================

class TestBestWindow:
    def test_empty_timeline(self):
        result = _compute_best_window({"timeline_48h": []})
        assert result["available"] is False

    def test_no_timeline_key(self):
        result = _compute_best_window({})
        assert result["available"] is False

    def test_calm_no_snow(self):
        zone = {
            "timeline_48h": [
                {"snowfall_in": 0, "wind_mph": 5, "wind_gust_mph": 10}
                for _ in range(24)
            ],
            "ml_wind_hold": {"probability": 0},
            "ml_snow_timing": {},
            "ml_snow_quality": {"quality": "Powder"},
        }
        result = _compute_best_window(zone)
        assert result["available"] is True
        assert "No new snow" in result["advice"]
        assert len(result["windows"]) <= 4

    def test_snowy_with_low_wind(self):
        zone = {
            "timeline_48h": [
                {"snowfall_in": 1.5, "wind_mph": 10, "wind_gust_mph": 15}
                for _ in range(24)
            ],
            "ml_wind_hold": {"probability": 0.1},
            "ml_snow_timing": {},
            "ml_snow_quality": {"quality": "Powder"},
        }
        result = _compute_best_window(zone)
        assert result["available"] is True
        assert result["best_score"] > 5  # Snow bonus pushes above baseline

    def test_high_wind_risk(self):
        zone = {
            "timeline_48h": [
                {"snowfall_in": 2.0, "wind_mph": 50, "wind_gust_mph": 70}
                for _ in range(24)
            ],
            "ml_wind_hold": {"probability": 0.9},
            "ml_snow_timing": {},
            "ml_snow_quality": {},
        }
        result = _compute_best_window(zone)
        assert "closed" in result["advice"].lower() or "wind" in result["advice"].lower()


# =========================================================================
# _quality_name
# =========================================================================

class TestQualityName:
    def test_all_ratings(self):
        assert _quality_name(5) == "light powder"
        assert _quality_name(4) == "powder"
        assert _quality_name(3) == "snow"
        assert _quality_name(2) == "heavy snow"
        assert _quality_name(1) == "wet snow"

    def test_unknown_rating(self):
        assert _quality_name(0) == "snow"
        assert _quality_name(99) == "snow"


# =========================================================================
# _extract_live_features
# =========================================================================

class TestExtractLiveFeatures:
    def test_basic_extraction(self):
        analysis = {
            "current_conditions": {},
            "snotel_current": {},
        }
        zone_data = {
            "current": {
                "temp_f": 28,
                "wind_mph": 15,
                "wind_gust_mph": 25,
                "wind_dir": "SW",
                "slr": 12,
                "dewpoint_f": 20,
            },
            "timeline_48h": [
                {"precip_in": 0.1, "snowfall_in": 0.5} for _ in range(24)
            ],
            "elev_ft": 8500,
        }
        features = _extract_live_features(analysis, zone_data)
        assert features["temperature_2m_mean"] == 28
        assert features["wind_speed_10m_max"] == 15
        assert features["slr"] == 12
        assert features["station_elev_ft"] == 8500
        assert "month" in features
        assert "day_of_year" in features

    def test_rain_fraction_cold(self):
        analysis = {"current_conditions": {}, "snotel_current": {}}
        zone_data = {
            "current": {"temp_f": 20, "wind_mph": 0, "wind_dir": "N"},
            "timeline_48h": [],
            "elev_ft": 8000,
        }
        features = _extract_live_features(analysis, zone_data)
        assert features["rain_fraction"] == 0.0
        assert features["snow_fraction"] == 1.0

    def test_rain_fraction_warm(self):
        analysis = {"current_conditions": {}, "snotel_current": {}}
        zone_data = {
            "current": {"temp_f": 40, "wind_mph": 0, "wind_dir": "N"},
            "timeline_48h": [],
            "elev_ft": 8000,
        }
        features = _extract_live_features(analysis, zone_data)
        assert features["rain_fraction"] == 1.0
        assert features["snow_fraction"] == 0.0

    def test_nan_temp_handling(self):
        analysis = {"current_conditions": {}, "snotel_current": {}}
        zone_data = {
            "current": {"wind_mph": 0, "wind_dir": "N"},
            "timeline_48h": [],
            "elev_ft": 8000,
        }
        features = _extract_live_features(analysis, zone_data)
        # temp_f is nan, rain_fraction should be nan but snow_fraction defaults
        assert features["snow_fraction"] == 0.5  # fallback when rain_fraction is nan

    def test_snotel_lag_features(self):
        analysis = {
            "current_conditions": {},
            "snotel_current": {
                "Fallen Leaf": {"depth_in": 50, "swe_in": 20, "temp_f": 25},
            },
        }
        zone_data = {
            "current": {"temp_f": 30, "wind_mph": 10, "wind_dir": "W"},
            "timeline_48h": [],
            "elev_ft": 7500,
        }
        features = _extract_live_features(analysis, zone_data)
        assert features["depth_lag1d"] == 50
        assert features["swe_lag1d"] == 20
        assert features["station_mean_depth_change"] == 0.09  # Fallen Leaf stats

    def test_ar_detection(self):
        analysis = {"current_conditions": {}, "snotel_current": {}}
        zone_data = {
            "current": {"temp_f": 40, "wind_mph": 25, "wind_dir": "S"},
            "timeline_48h": [{"precip_in": 0.5, "snowfall_in": 0} for _ in range(24)],
            "elev_ft": 7000,
        }
        features = _extract_live_features(analysis, zone_data)
        # Southerly wind, warm, wet, windy → AR score should be high
        assert features["ar_score"] >= 3
        assert features["is_ar_event"] == 1


# =========================================================================
# Shadow logging
# =========================================================================

class TestShadowLogging:
    def test_log_creates_file(self, tmp_path):
        log_path = str(tmp_path / "shadow.json")
        with patch("ml_corrections.SHADOW_LOG", log_path):
            _log_shadow_prediction("Heavenly", "peak", {"ml_depth_prediction_in": 5.2})
        with open(log_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["resort"] == "Heavenly"
        assert data[0]["ml_depth_prediction_in"] == 5.2

    def test_log_appends(self, tmp_path):
        log_path = str(tmp_path / "shadow.json")
        with patch("ml_corrections.SHADOW_LOG", log_path):
            _log_shadow_prediction("A", "peak", {"val": 1})
            _log_shadow_prediction("B", "mid", {"val": 2})
        with open(log_path) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_log_truncates_at_5000(self, tmp_path):
        log_path = str(tmp_path / "shadow.json")
        # Pre-populate with 5000 entries
        with open(log_path, "w") as f:
            json.dump([{"i": i} for i in range(5000)], f)
        with patch("ml_corrections.SHADOW_LOG", log_path):
            _log_shadow_prediction("New", "peak", {"val": 1})
        with open(log_path) as f:
            data = json.load(f)
        assert len(data) == 5000  # Truncated to last 5000

    def test_log_handles_corrupted_json(self, tmp_path):
        log_path = str(tmp_path / "shadow.json")
        with open(log_path, "w") as f:
            json.dump({"not": "a list"}, f)  # Corrupted: dict instead of list
        with patch("ml_corrections.SHADOW_LOG", log_path):
            _log_shadow_prediction("Test", "peak", {"val": 1})
        with open(log_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1


# =========================================================================
# get_drift_metrics
# =========================================================================

class TestDriftMetrics:
    def test_no_log_file(self, tmp_path):
        with patch("ml_corrections.SHADOW_LOG", str(tmp_path / "nonexistent.json")):
            result = get_drift_metrics()
        assert result["available"] is False

    def test_few_predictions(self, tmp_path):
        log_path = str(tmp_path / "shadow.json")
        with open(log_path, "w") as f:
            json.dump([{"timestamp": "2025-01-01T00:00:00"} for _ in range(5)], f)
        with patch("ml_corrections.SHADOW_LOG", log_path):
            result = get_drift_metrics()
        assert result["available"] is False
        assert "Only 5" in result["reason"]

    def test_sufficient_predictions(self, tmp_path):
        log_path = str(tmp_path / "shadow.json")
        preds = [{"timestamp": f"2025-01-{i+1:02d}T00:00:00"} for i in range(15)]
        with open(log_path, "w") as f:
            json.dump(preds, f)
        with patch("ml_corrections.SHADOW_LOG", log_path):
            result = get_drift_metrics()
        assert result["available"] is True
        assert result["total_predictions"] == 15
        assert result["days_tracked"] == 15


# =========================================================================
# _predict_chains
# =========================================================================

class TestPredictChains:
    def test_clear_roads(self):
        """All resorts warm with no snow → no chains on any route."""
        from resort_configs import get_active_resorts
        configs = get_active_resorts()
        resorts = {}
        for name in configs:
            resorts[name] = {
                "zones": {
                    "base": {
                        "current": {"temp_f": 45},
                        "ml_new_snow_prob": 0.0,
                        "ml_depth_prediction_in": 0,
                        "ml_snow_timing": {"snowing": False},
                    }
                }
            }
        result = _predict_chains(resorts, {})
        assert result.get("available", False) is True
        for route_data in result.get("routes", {}).values():
            assert route_data["level"] == "None"

    def test_heavy_snow_r2(self):
        """All resorts cold with heavy snow → R2 chains."""
        from resort_configs import get_active_resorts
        configs = get_active_resorts()
        resorts = {}
        for name in configs:
            resorts[name] = {
                "zones": {
                    "base": {
                        "current": {"temp_f": 22},
                        "ml_new_snow_prob": 0.95,
                        "ml_depth_prediction_in": 12,
                        "ml_snow_timing": {"snowing": True},
                    }
                }
            }
        result = _predict_chains(resorts, {})
        assert result.get("available", False) is True
        for route_data in result.get("routes", {}).values():
            assert route_data["level"] in ("R2", "R3")


# =========================================================================
# _build_ml_outdoor_from_outdoor
# =========================================================================

class TestBuildMlOutdoor:
    def test_empty_analysis(self):
        result = _build_ml_outdoor_from_outdoor({"resorts": {}})
        assert "thunderstorm" in result
        assert "heat_safety" in result
        assert "uv_exposure" in result

    def test_with_resort_data(self):
        analysis = {
            "outdoor": {},
            "resorts": {
                "Heavenly": {
                    "zones": {
                        "mid": {
                            "current": {
                                "temp_f": 75, "cape_jkg": 100, "humidity_pct": 30
                            },
                            "timeline_48h": [{"temp_f": 75}] * 24,
                        }
                    }
                }
            },
            "current_conditions": {"observation": {"aqi": {}}},
            "snotel_current": {},
        }
        result = _build_ml_outdoor_from_outdoor(analysis)
        assert result["thunderstorm"] is not None
        assert result["heat_safety"] is not None
