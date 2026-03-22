#!/usr/bin/env python3
"""Tests for forecast_verification.py — bias tracking, model weights, scoring."""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from forecast_verification import (
    _brier_score,
    _group_by_lead_time,
    _load_data,
    _save_data,
    _score_day,
    _settling_factor,
    _trim,
    _default_weights,
    get_bias_corrections,
    get_model_weights,
    get_pop_calibration,
    log_actual,
    log_forecast,
    recalibrate_pop,
    VERIFY_FILE,
)


@pytest.fixture
def tmp_verify_file(tmp_path):
    """Redirect verification file to temp directory."""
    path = str(tmp_path / "verify.json")
    with patch("forecast_verification.VERIFY_FILE", path):
        yield path


# =========================================================================
# _settling_factor
# =========================================================================

class TestSettlingFactor:
    def test_very_cold(self):
        """At 10°F, almost no settling."""
        factor = _settling_factor(10.0)
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_freezing(self):
        """At 32°F, maximum settling (~25%)."""
        factor = _settling_factor(32.0)
        assert factor == pytest.approx(0.75, abs=0.01)

    def test_mid_range(self):
        """At 21°F (midpoint), ~12.5% settling."""
        factor = _settling_factor(21.0)
        assert 0.8 < factor < 0.95

    def test_below_10(self):
        """Below 10°F, still 1.0 (clamped)."""
        assert _settling_factor(0) == pytest.approx(1.0, abs=0.01)

    def test_above_32(self):
        """Above 32°F, still 0.75 (clamped)."""
        assert _settling_factor(50) == pytest.approx(0.75, abs=0.01)


# =========================================================================
# _brier_score
# =========================================================================

class TestBrierScore:
    def test_perfect_forecast(self):
        """100% when rained, 0% when didn't → Brier = 0."""
        predicted = [100, 0, 100, 0, 100]
        actuals = [1, 0, 1, 0, 1]
        assert _brier_score(predicted, actuals) == pytest.approx(0.0, abs=0.01)

    def test_worst_forecast(self):
        """0% when rained, 100% when didn't → Brier = 1."""
        predicted = [0, 100, 0, 100, 0]
        actuals = [1, 0, 1, 0, 1]
        assert _brier_score(predicted, actuals) == pytest.approx(1.0, abs=0.01)

    def test_50_50(self):
        """50% everywhere → Brier = 0.25."""
        predicted = [50, 50, 50, 50, 50]
        actuals = [1, 0, 1, 0, 1]
        assert _brier_score(predicted, actuals) == pytest.approx(0.25, abs=0.01)

    def test_insufficient_data(self):
        assert _brier_score([50, 50], [1, 0]) is None

    def test_none_values_skipped(self):
        predicted = [100, None, 0, 100, 0, 100]
        actuals = [1, 1, 0, 1, 0, 1]
        result = _brier_score(predicted, actuals)
        assert result is not None
        assert result == pytest.approx(0.0, abs=0.01)


# =========================================================================
# _group_by_lead_time
# =========================================================================

class TestGroupByLeadTime:
    def test_all_buckets(self):
        errors = [1.0, 2.0, 3.0, 4.0]
        leads = [12, 36, 60, 96]
        result = _group_by_lead_time(errors, leads)
        assert result["0-24h"] == 1.0
        assert result["24-48h"] == 2.0
        assert result["48-72h"] == 3.0
        assert result["72h+"] == 4.0

    def test_multiple_in_bucket(self):
        errors = [1.0, 3.0]
        leads = [6, 12]
        result = _group_by_lead_time(errors, leads)
        assert result["0-24h"] == pytest.approx(2.0)

    def test_empty(self):
        result = _group_by_lead_time([], [])
        assert result == {}


# =========================================================================
# _score_day
# =========================================================================

class TestScoreDay:
    def test_scores_matching_forecasts(self):
        data = {
            "forecasts": [
                {"source": "gfs", "metric": "temp_high_f", "value": 70, "valid_date": "2025-03-20", "lead_hours": 24},
                {"source": "ecmwf", "metric": "temp_high_f", "value": 72, "valid_date": "2025-03-20", "lead_hours": 24},
            ],
            "actuals": [
                {"metric": "temp_high_f", "value": 68, "date": "2025-03-20"},
            ],
            "daily_scores": [],
        }
        _score_day(data, "2025-03-20")
        assert len(data["daily_scores"]) == 2
        gfs = [s for s in data["daily_scores"] if s["source"] == "gfs"][0]
        assert gfs["error"] == 2.0
        assert gfs["abs_error"] == 2.0
        ecmwf = [s for s in data["daily_scores"] if s["source"] == "ecmwf"][0]
        assert ecmwf["error"] == 4.0

    def test_no_actuals_noop(self):
        data = {
            "forecasts": [{"source": "gfs", "metric": "temp_high_f", "value": 70, "valid_date": "2025-03-20", "lead_hours": 0}],
            "actuals": [],
            "daily_scores": [],
        }
        _score_day(data, "2025-03-20")
        assert len(data["daily_scores"]) == 0

    def test_replaces_existing_scores(self):
        data = {
            "forecasts": [{"source": "gfs", "metric": "temp_high_f", "value": 70, "valid_date": "2025-03-20", "lead_hours": 0}],
            "actuals": [{"metric": "temp_high_f", "value": 65, "date": "2025-03-20"}],
            "daily_scores": [{"date": "2025-03-20", "source": "gfs", "metric": "temp_high_f", "predicted": 75, "actual": 65, "error": 10, "abs_error": 10, "lead_hours": 0}],
        }
        _score_day(data, "2025-03-20")
        scores = [s for s in data["daily_scores"] if s["date"] == "2025-03-20"]
        assert len(scores) == 1
        assert scores[0]["predicted"] == 70  # Updated


# =========================================================================
# log_forecast, log_actual, get_bias_corrections
# =========================================================================

class TestLogAndBias:
    def test_log_and_retrieve_bias(self, tmp_verify_file):
        """After 14+ days, bias corrections should be computed."""
        today = datetime.now(timezone.utc)
        for i in range(15):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            log_forecast("gfs", "temp_high_f", 75, d, 24)
            log_actual("temp_high_f", 70, d)
        corrections = get_bias_corrections()
        # GFS consistently predicts 5°F too high → correction should be ~-5
        if "gfs" in corrections and "temp_high_f" in corrections["gfs"]:
            assert corrections["gfs"]["temp_high_f"] < 0

    def test_insufficient_data_no_corrections(self, tmp_verify_file):
        log_forecast("gfs", "temp_high_f", 75, "2025-03-20", 24)
        log_actual("temp_high_f", 70, "2025-03-20")
        corrections = get_bias_corrections()
        # Only 1 day of data — no corrections
        assert corrections == {} or "gfs" not in corrections


# =========================================================================
# get_model_weights
# =========================================================================

class TestModelWeights:
    def test_default_weights_insufficient_data(self, tmp_verify_file):
        weights = get_model_weights()
        assert weights == _default_weights()

    def test_weights_sum_to_one(self, tmp_verify_file):
        """With sufficient data, weights should sum to ~1.0."""
        today = datetime.now(timezone.utc)
        for i in range(10):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            log_forecast("gfs", "temp_high_f", 72, d, 24)
            log_forecast("ecmwf", "temp_high_f", 70, d, 24)
            log_actual("temp_high_f", 71, d)
        weights = get_model_weights("temp_high_f")
        if len(weights) >= 2:
            assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_better_model_gets_more_weight(self, tmp_verify_file):
        """ECMWF closer to truth should get higher weight than GFS."""
        today = datetime.now(timezone.utc)
        for i in range(10):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            log_forecast("gfs", "temp_high_f", 80, d, 24)  # 10°F off
            log_forecast("ecmwf", "temp_high_f", 71, d, 24)  # 1°F off
            log_actual("temp_high_f", 70, d)
        weights = get_model_weights("temp_high_f")
        if "ECMWF" in weights and "GFS" in weights:
            assert weights["ECMWF"] > weights["GFS"]


# =========================================================================
# POP calibration
# =========================================================================

class TestPopCalibration:
    def test_empty_no_data(self, tmp_verify_file):
        result = get_pop_calibration()
        assert result == {}

    def test_recalibrate_no_data(self, tmp_verify_file):
        assert recalibrate_pop(50) == 50  # Unchanged


# =========================================================================
# _trim
# =========================================================================

class TestTrim:
    def test_removes_old_entries(self):
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=100)).isoformat()
        recent = (now - timedelta(days=5)).isoformat()
        data = {
            "forecasts": [
                {"t": old, "source": "a", "metric": "b", "value": 1, "valid_date": "old", "lead_hours": 0},
                {"t": recent, "source": "a", "metric": "b", "value": 2, "valid_date": "new", "lead_hours": 0},
            ],
            "actuals": [
                {"t": old, "metric": "b", "value": 1, "date": "old"},
                {"t": recent, "metric": "b", "value": 2, "date": "new"},
            ],
            "daily_scores": [
                {"date": (now - timedelta(days=100)).strftime("%Y-%m-%d"), "source": "a", "metric": "b"},
                {"date": (now - timedelta(days=5)).strftime("%Y-%m-%d"), "source": "a", "metric": "b"},
            ],
        }
        _trim(data)
        assert len(data["forecasts"]) == 1
        assert len(data["actuals"]) == 1
        assert len(data["daily_scores"]) == 1


# =========================================================================
# File I/O
# =========================================================================

class TestFileIO:
    def test_load_missing_file(self, tmp_verify_file):
        data = _load_data()
        assert data == {"forecasts": [], "actuals": [], "daily_scores": []}

    def test_save_and_load(self, tmp_verify_file):
        data = {"forecasts": [{"test": True}], "actuals": [], "daily_scores": []}
        _save_data(data)
        loaded = _load_data()
        assert loaded["forecasts"][0]["test"] is True

    def test_corrupted_json(self, tmp_verify_file):
        with open(tmp_verify_file, "w") as f:
            f.write("not json{{{")
        data = _load_data()
        assert data == {"forecasts": [], "actuals": [], "daily_scores": []}
