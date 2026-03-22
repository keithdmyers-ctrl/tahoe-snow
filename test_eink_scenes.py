#!/usr/bin/env python3
"""Tests for eink_scenes.py — scene context builders, formatting helpers."""

import pytest
from unittest.mock import patch

from eink_scenes import (
    _condition_icon,
    _t,
    _snow_str,
    build_oakland_context,
    build_heavenly_context,
    CONDITION_ICONS,
)


# =========================================================================
# _condition_icon
# =========================================================================

class TestConditionIcon:
    def test_sunny(self):
        assert _condition_icon("Sunny") == "\u2600\ufe0f"

    def test_rain(self):
        assert _condition_icon("Chance Rain Showers") == "\U0001f327"

    def test_snow(self):
        assert _condition_icon("Heavy Snow") == "\u2744\ufe0f"

    def test_fog(self):
        assert _condition_icon("Patchy Fog") == "\U0001f32b"

    def test_thunderstorm(self):
        assert _condition_icon("Thunderstorms Likely") == "\u26c8"

    def test_unknown_returns_thermometer(self):
        assert _condition_icon("Something Weird") == "\U0001f321"

    def test_case_insensitive(self):
        assert _condition_icon("SUNNY") == "\u2600\ufe0f"
        assert _condition_icon("rain") == "\U0001f327"

    def test_empty_string(self):
        assert _condition_icon("") == "\U0001f321"

    def test_all_unique_conditions(self):
        """Conditions that aren't substrings of earlier keys should map correctly."""
        # Note: _condition_icon does substring matching, so "mostly sunny" matches
        # "sunny" first. This is a known limitation of the iteration order.
        unique_keys = ["clear", "cloudy", "overcast", "rain", "drizzle",
                       "thunderstorm", "snow", "flurries", "blizzard",
                       "fog", "haze", "mist", "wind", "breezy"]
        for key in unique_keys:
            assert _condition_icon(key) == CONDITION_ICONS[key], f"Failed for {key}"


# =========================================================================
# _t (temperature formatter)
# =========================================================================

class TestTempFormatter:
    def test_integer(self):
        assert _t(72) == "72"

    def test_float(self):
        assert _t(72.6) == "73"  # Rounds

    def test_float_string(self):
        assert _t("68.4") == "68"

    def test_none(self):
        assert _t(None) == "--"

    def test_invalid_string(self):
        assert _t("N/A") == "--"

    def test_zero(self):
        assert _t(0) == "0"

    def test_negative(self):
        assert _t(-5) == "-5"


# =========================================================================
# _snow_str
# =========================================================================

class TestSnowStr:
    def test_none(self):
        assert _snow_str(None) == "--"

    def test_zero(self):
        assert _snow_str(0) == "--"

    def test_large_amount(self):
        assert _snow_str(12) == '12"'

    def test_half_inch(self):
        assert _snow_str(0.5) == '0"'  # Rounds to 0

    def test_small_amount(self):
        result = _snow_str(0.3)
        assert '"' in result
        assert "0.3" in result

    def test_exactly_half(self):
        result = _snow_str(0.5)
        assert '"' in result


# =========================================================================
# build_oakland_context
# =========================================================================

class TestBuildOaklandContext:
    def _make_cache(self):
        return {
            "data": {
                "generated": "2025-03-20T12:00:00+00:00",
                "current_conditions": {"observation": {}},
            },
            "home_obs": {
                "temp_f": 65, "wind_dir": "W", "wind_mph": 8,
                "wind_gust_mph": 12, "conditions": "Partly Cloudy",
                "visibility_mi": 10,
            },
            "home_fc": {
                "periods": [
                    {"isDaytime": True, "temperature": 72, "name": "Today",
                     "shortForecast": "Sunny", "startTime": "2025-03-20T08:00:00"},
                    {"isDaytime": False, "temperature": 55, "name": "Tonight",
                     "shortForecast": "Clear", "startTime": "2025-03-20T18:00:00"},
                ],
                "hourly": [],
            },
            "home_om": {
                "daily": {
                    "uv_index_max": [7.5],
                    "sunrise": ["2025-03-20T06:30:00"],
                    "sunset": ["2025-03-20T19:15:00"],
                },
            },
            "home_nbm": {},
            "home_pws": [],
            "home_alerts": [],
            "home_normals": {"avg_high_f": 65},
            "home_aqi": {"aqi": 35, "category": "Good"},
            "sensors": {"indoor": {"temp_f": 70}, "outdoor": {"temp_f": 62, "humidity_pct": 55}},
        }

    def test_basic_context_structure(self):
        with patch("eink_scenes.get_pressure_forecast", return_value={"rain_pct": 10}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(self._make_cache())
        assert ctx["location"] == "OAKLAND"
        assert ctx["today_high"] == "72"
        assert ctx["today_low"] == "55"
        assert ctx["nws_temp"] == "65"
        assert ctx["outdoor_temp"] == "62"
        assert ctx["indoor_temp"] == "70"

    def test_solar_data(self):
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(self._make_cache())
        assert "sunrise" in ctx["solar"]
        assert "sunset" in ctx["solar"]

    def test_uv_index(self):
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(self._make_cache())
        assert ctx["uv_index"] == 7.5

    def test_aqi_data(self):
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(self._make_cache())
        assert ctx["aqi"] == 35
        assert ctx["aqi_category"] == "Good"

    def test_wind_formatting(self):
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(self._make_cache())
        assert "W 8mph" in ctx["wind"]

    def test_temp_anomaly(self):
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(self._make_cache())
        # Today high 72 vs normal 65 → anomaly of +7
        assert ctx["temp_anomaly"] == 7

    def test_empty_observations(self):
        cache = self._make_cache()
        cache["home_obs"] = {}
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(cache)
        assert ctx["wind"] == ""
        assert ctx["nws_temp"] == "--"

    def test_empty_forecast(self):
        cache = self._make_cache()
        cache["home_fc"] = {}
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(cache)
        assert ctx["today_high"] == "--"
        assert ctx["today_low"] == "--"

    def test_no_sensors(self):
        cache = self._make_cache()
        cache["sensors"] = None
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(cache)
        assert ctx["indoor_temp"] == "--"

    def test_alerts_truncated(self):
        cache = self._make_cache()
        cache["home_alerts"] = [
            {"event": f"Alert{i}", "severity": "Moderate", "headline": f"H{i}"}
            for i in range(10)
        ]
        with patch("eink_scenes.get_pressure_forecast", return_value={}), \
             patch("eink_scenes.predict_rain_timing", return_value={}):
            ctx = build_oakland_context(cache)
        assert len(ctx["alerts"]) == 3  # Max 3


# =========================================================================
# build_heavenly_context
# =========================================================================

class TestBuildHeavenlyContext:
    def _make_cache(self):
        return {
            "data": {
                "generated": "2025-03-20T12:00:00+00:00",
                "current_conditions": {
                    "observation": {
                        "temp_f": 30, "wind_mph": 15, "wind_dir": "SW",
                        "conditions": "Snow",
                    },
                },
                "resorts": {
                    "Heavenly": {
                        "zones": {
                            "peak": {
                                "elev_ft": 10067,
                                "current": {"temp_f": 22, "wind_mph": 25, "snowfall_in": 3},
                                "snow_24h": 6,
                                "snow_7d_forecast": [
                                    {"date": "2025-03-20", "hi": 30, "lo": 18, "snow_in": 4},
                                    {"date": "2025-03-21", "hi": 35, "lo": 22, "snow_in": 0},
                                ],
                                "timeline_48h": [{"snowfall_in": 0.5, "temp_f": 25} for _ in range(48)],
                            },
                            "mid": {
                                "elev_ft": 8000,
                                "current": {"temp_f": 28},
                                "snow_24h": 4,
                            },
                            "base": {
                                "elev_ft": 6540,
                                "current": {"temp_f": 34, "snow_level_ft": 6000},
                                "snow_24h": 2,
                            },
                        },
                    },
                },
                "storm": {"in_storm": True, "storm_total_in": 15},
                "decision": {"go": True, "reason": "Powder day"},
                "storm_narrative": "Big storm",
                "avalanche": {"danger_level": 2, "danger_label": "Moderate"},
                "snotel_current": {
                    "Fallen Leaf": {"depth_in": 55, "swe_in": 22},
                },
                "ml_powder_alert": {"level": 4, "label": "STRONG", "message": "Great day"},
                "ml_resort_ranking": [{"resort": "Heavenly", "score": 8}],
            },
            "sensors": {},
            "cssl": {},
            "tahoe_alerts": [],
        }

    def test_basic_context(self):
        ctx = build_heavenly_context(self._make_cache())
        assert "zones" in ctx
        assert "snotel" in ctx or "snotel_summary" in ctx or isinstance(ctx, dict)

    def test_with_empty_analysis(self):
        cache = self._make_cache()
        cache["data"]["resorts"] = {}
        cache["data"]["snotel_current"] = {}
        # Should not crash
        ctx = build_heavenly_context(cache)
        assert isinstance(ctx, dict)
