#!/usr/bin/env python3
"""Tests for pressure_forecast.py — Zambretti barometric prediction, trend classification."""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from pressure_forecast import (
    _classify_trend,
    _get_pressure_at_offset,
    _lookup_forecast,
    _parse_open_meteo_precip,
    _sea_level_pressure,
    _zambretti_number,
    _count_models_with_precip,
    get_forecast,
    record_pressure,
)


# =========================================================================
# _sea_level_pressure
# =========================================================================

class TestSeaLevelPressure:
    def test_zero_elevation(self):
        assert _sea_level_pressure(1013.25, 0) == 1013.25

    def test_correction_positive(self):
        """Higher elevation → station pressure is lower → SLP should be higher."""
        slp = _sea_level_pressure(1000, 137)
        assert slp > 1000

    def test_known_correction(self):
        """At ~137m, correction should be about +16 hPa."""
        slp = _sea_level_pressure(1000, 137)
        assert 1015 < slp < 1018


# =========================================================================
# _classify_trend
# =========================================================================

class TestClassifyTrend:
    def test_rapidly_falling(self):
        assert _classify_trend(-7.0) == ("falling", "rapidly")

    def test_falling(self):
        assert _classify_trend(-3.0) == ("falling", "falling")

    def test_slowly_falling(self):
        assert _classify_trend(-1.0) == ("falling", "slowly")

    def test_steady(self):
        assert _classify_trend(0.0) == ("steady", "steady")
        assert _classify_trend(0.4) == ("steady", "steady")
        assert _classify_trend(-0.4) == ("steady", "steady")

    def test_slowly_rising(self):
        assert _classify_trend(1.0) == ("rising", "slowly")

    def test_rising(self):
        assert _classify_trend(3.0) == ("rising", "rising")

    def test_rapidly_rising(self):
        assert _classify_trend(7.0) == ("rising", "rapidly")

    def test_boundary_values(self):
        assert _classify_trend(-6.0)[1] == "rapidly"
        assert _classify_trend(-1.6)[1] == "falling"
        assert _classify_trend(-0.5)[1] == "slowly"
        assert _classify_trend(0.5)[0] == "rising"
        assert _classify_trend(1.6)[0] == "rising"
        assert _classify_trend(6.0)[1] == "rapidly"


# =========================================================================
# _zambretti_number
# =========================================================================

class TestZambrettiNumber:
    def test_falling_high_pressure(self):
        """High pressure falling → low Z number (still fine)."""
        z = _zambretti_number(1030, "falling", 6)
        assert 0 <= z <= 9

    def test_falling_low_pressure(self):
        """Low pressure falling → high Z number (stormy)."""
        z = _zambretti_number(990, "falling", 6)
        assert 5 <= z <= 9

    def test_rising_low_pressure(self):
        z = _zambretti_number(990, "rising", 6)
        assert 20 <= z <= 32

    def test_steady(self):
        z = _zambretti_number(1013, "steady", 6)
        assert 10 <= z <= 19

    def test_winter_pessimistic(self):
        """Winter should push toward worse weather."""
        summer_z = _zambretti_number(1015, "falling", 7)
        winter_z = _zambretti_number(1015, "falling", 1)
        assert winter_z >= summer_z

    def test_rising_season_effect(self):
        """Rising pressure: winter season_adj=1, summer=-1.
        For rising, z = base - season_adj, so winter gets lower z (better)."""
        summer_z = _zambretti_number(1015, "rising", 7)
        winter_z = _zambretti_number(1015, "rising", 1)
        # Both valid Z numbers in rising range
        assert 20 <= summer_z <= 32
        assert 20 <= winter_z <= 32


# =========================================================================
# _lookup_forecast
# =========================================================================

class TestLookupForecast:
    def test_falling_range(self):
        for z in range(10):
            text, rain = _lookup_forecast(z)
            assert isinstance(text, str)
            assert 0 <= rain <= 100

    def test_steady_range(self):
        for z in range(10, 20):
            text, rain = _lookup_forecast(z)
            assert isinstance(text, str)

    def test_rising_range(self):
        for z in range(20, 33):
            text, rain = _lookup_forecast(z)
            assert isinstance(text, str)

    def test_rain_increases_with_worse_weather(self):
        _, rain_fine = _lookup_forecast(0)  # Fine
        _, rain_storm = _lookup_forecast(9)  # Stormy
        assert rain_storm > rain_fine


# =========================================================================
# _get_pressure_at_offset
# =========================================================================

class TestGetPressureAtOffset:
    def test_empty_history(self):
        assert _get_pressure_at_offset([], 1.0) is None

    def test_exact_match(self):
        now = datetime.now(timezone.utc)
        history = [
            {"t": (now - timedelta(hours=1)).isoformat(), "hpa": 1015.0},
            {"t": now.isoformat(), "hpa": 1013.0},
        ]
        result = _get_pressure_at_offset(history, 1.0)
        assert result == 1015.0

    def test_no_match_within_tolerance(self):
        now = datetime.now(timezone.utc)
        history = [
            {"t": (now - timedelta(hours=5)).isoformat(), "hpa": 1015.0},
            {"t": now.isoformat(), "hpa": 1013.0},
        ]
        # Looking for 1h ago — 5h ago is outside 15-min tolerance
        result = _get_pressure_at_offset(history, 1.0)
        assert result is None


# =========================================================================
# get_forecast (integration)
# =========================================================================

class TestGetForecast:
    def test_no_history(self):
        with patch("pressure_forecast._load_history", return_value=[]):
            result = get_forecast()
        assert result["confidence"] == "none"
        assert "error" in result

    def test_with_3h_history(self):
        now = datetime.now(timezone.utc)
        history = [
            {"t": (now - timedelta(hours=3)).isoformat(), "hpa": 1020.0, "rh": 50},
            {"t": (now - timedelta(hours=2)).isoformat(), "hpa": 1018.0, "rh": 55},
            {"t": (now - timedelta(hours=1)).isoformat(), "hpa": 1016.0, "rh": 60},
            {"t": now.isoformat(), "hpa": 1014.0, "rh": 65},
        ]
        with patch("pressure_forecast._load_history", return_value=history):
            result = get_forecast()
        assert result["confidence"] == "good"
        assert result["trend_dir"] == "falling"
        assert result["pressure_hpa"] == 1014.0
        assert result["rain_pct"] is not None
        assert isinstance(result["forecast"], str)

    def test_humidity_adjustment(self):
        now = datetime.now(timezone.utc)
        history = [
            {"t": (now - timedelta(hours=3)).isoformat(), "hpa": 1020.0},
            {"t": now.isoformat(), "hpa": 1014.0},
        ]
        with patch("pressure_forecast._load_history", return_value=history):
            without_humidity = get_forecast()
            with_high_humidity = get_forecast(current_humidity=90)
        # High humidity + falling pressure → more rain
        assert with_high_humidity["rain_pct"] >= without_humidity["rain_pct"]

    def test_low_humidity_reduces_rain(self):
        now = datetime.now(timezone.utc)
        history = [
            {"t": (now - timedelta(hours=3)).isoformat(), "hpa": 1013.0},
            {"t": now.isoformat(), "hpa": 1013.0},  # steady
        ]
        with patch("pressure_forecast._load_history", return_value=history):
            without_humidity = get_forecast()
            with_low_humidity = get_forecast(current_humidity=30)
        assert with_low_humidity["rain_pct"] <= without_humidity["rain_pct"]

    def test_1h_data_fair_confidence(self):
        now = datetime.now(timezone.utc)
        history = [
            {"t": (now - timedelta(hours=1)).isoformat(), "hpa": 1015.0},
            {"t": now.isoformat(), "hpa": 1013.0},
        ]
        with patch("pressure_forecast._load_history", return_value=history):
            result = get_forecast()
        assert result["confidence"] == "fair"


# =========================================================================
# _parse_open_meteo_precip
# =========================================================================

class TestParseOpenMeteoPrecip:
    def test_empty_dict(self):
        assert _parse_open_meteo_precip({}) == []

    def test_error_key(self):
        assert _parse_open_meteo_precip({"error": "timeout"}) == []

    def test_single_model(self):
        om = {"hourly": {"precipitation": [0.0, 0.5, 1.0]}}
        result = _parse_open_meteo_precip(om)
        assert result == [0.0, 0.5, 1.0]

    def test_multi_model_average(self):
        om = {
            "gfs": {"hourly": {"precipitation": [0.0, 2.0, 4.0]}},
            "ecmwf": {"hourly": {"precipitation": [0.0, 0.0, 2.0]}},
        }
        result = _parse_open_meteo_precip(om)
        assert len(result) == 3
        assert result[1] == pytest.approx(1.0)  # avg of 2.0 and 0.0
        assert result[2] == pytest.approx(3.0)  # avg of 4.0 and 2.0

    def test_none_values_skipped(self):
        om = {"hourly": {"precipitation": [1.0, None, 3.0]}}
        result = _parse_open_meteo_precip(om)
        assert result[0] == 1.0
        assert result[1] == 0  # None treated as no data, vals empty → 0
        assert result[2] == 3.0


# =========================================================================
# _count_models_with_precip
# =========================================================================

class TestCountModelsWithPrecip:
    def test_empty(self):
        assert _count_models_with_precip({}) == (0, 0)

    def test_error(self):
        assert _count_models_with_precip({"error": "fail"}) == (0, 0)

    def test_none_input(self):
        assert _count_models_with_precip(None) == (0, 0)


# =========================================================================
# record_pressure
# =========================================================================

class TestRecordPressure:
    def test_records_and_computes_slp(self, tmp_path):
        path = str(tmp_path / "pressure.json")
        with patch("pressure_forecast.HISTORY_FILE", path):
            record_pressure(1000.0, humidity_pct=50)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["raw"] == 1000.0
        assert data[0]["hpa"] > 1000.0  # SLP corrected
        assert data[0]["rh"] == 50.0

    def test_trims_old_entries(self, tmp_path):
        path = str(tmp_path / "pressure.json")
        now = datetime.now(timezone.utc)
        old = [{"t": (now - timedelta(hours=30)).isoformat(), "hpa": 1013, "raw": 1000, "rh": None}]
        with open(path, "w") as f:
            json.dump(old, f)
        with patch("pressure_forecast.HISTORY_FILE", path):
            record_pressure(1000.0)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1  # Old entry trimmed
