#!/usr/bin/env python3
"""Tests for data_pipeline.py — helpers, error checking, coordinate config."""

import pytest

from data_pipeline import (
    _check_error,
    _safe_fetch,
    TAHOE,
    OAKLAND,
    SOUNDING_STATION,
)


# =========================================================================
# _safe_fetch
# =========================================================================

class TestSafeFetch:
    def test_successful_call(self):
        result = _safe_fetch(lambda: {"data": 42})
        assert result == {"data": 42}

    def test_exception_returns_default(self):
        def bad():
            raise ConnectionError("timeout")
        result = _safe_fetch(bad)
        assert result == {}

    def test_custom_default(self):
        def bad():
            raise ValueError
        result = _safe_fetch(bad, default=[])
        assert result == []

    def test_passes_args(self):
        def add(a, b):
            return a + b
        assert _safe_fetch(add, 3, 4) == 7

    def test_passes_kwargs(self):
        def greet(name="World"):
            return f"Hello {name}"
        assert _safe_fetch(greet, name="Keith") == "Hello Keith"


# =========================================================================
# _check_error
# =========================================================================

class TestCheckError:
    def test_clean_dict(self):
        data = {"temp_f": 65, "conditions": "Clear"}
        assert _check_error(data) == data

    def test_error_dict(self):
        data = {"error": "API timeout", "status": 503}
        assert _check_error(data) is None

    def test_error_with_custom_default(self):
        data = {"error": "bad request"}
        assert _check_error(data, default={}) == {}

    def test_non_dict(self):
        assert _check_error([1, 2, 3]) == [1, 2, 3]
        assert _check_error("string") == "string"
        assert _check_error(None) is None

    def test_empty_dict(self):
        assert _check_error({}) == {}


# =========================================================================
# Configuration constants
# =========================================================================

class TestConfig:
    def test_tahoe_coordinates(self):
        assert 39.0 < TAHOE["lat"] < 39.5
        assert -120.5 < TAHOE["lon"] < -119.5

    def test_oakland_coordinates(self):
        assert 37.5 < OAKLAND["lat"] < 38.0
        assert -122.5 < OAKLAND["lon"] < -121.5

    def test_sounding_station(self):
        assert SOUNDING_STATION == "REV"  # Reno upper air
