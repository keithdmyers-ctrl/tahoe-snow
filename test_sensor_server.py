#!/usr/bin/env python3
"""Tests for sensor_server.py — HTTP endpoints, rate limiting, data persistence."""

import json
import time
from unittest.mock import patch

import pytest

from sensor_server import (
    _check_rate_limit,
    _last_post_by_ip,
    _rate_limit_lock,
    load_data,
    save_data,
)


# =========================================================================
# Rate limiting
# =========================================================================

class TestRateLimit:
    def setup_method(self):
        """Clear rate limit state between tests."""
        with _rate_limit_lock:
            _last_post_by_ip.clear()

    def test_first_request_allowed(self):
        assert _check_rate_limit("192.168.1.1") is True

    def test_rapid_requests_blocked(self):
        _check_rate_limit("192.168.1.2")
        assert _check_rate_limit("192.168.1.2") is False

    def test_different_ips_independent(self):
        _check_rate_limit("192.168.1.3")
        assert _check_rate_limit("192.168.1.4") is True

    def test_allowed_after_cooldown(self):
        _check_rate_limit("192.168.1.5")
        # Simulate time passing
        with _rate_limit_lock:
            _last_post_by_ip["192.168.1.5"] = time.monotonic() - 5
        assert _check_rate_limit("192.168.1.5") is True


# =========================================================================
# Data persistence
# =========================================================================

class TestDataPersistence:
    def test_load_missing_file(self, tmp_path):
        with patch("sensor_server.DATA_FILE", str(tmp_path / "missing.json")):
            data = load_data()
        assert data == {}

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "sensor.json")
        with patch("sensor_server.DATA_FILE", path):
            save_data({"indoor": {"temp_f": 72}})
            loaded = load_data()
        assert loaded["indoor"]["temp_f"] == 72

    def test_corrupted_json(self, tmp_path):
        path = tmp_path / "sensor.json"
        path.write_text("not json{{{")
        with patch("sensor_server.DATA_FILE", str(path)):
            data = load_data()
        assert data == {}

    def test_save_overwrites(self, tmp_path):
        path = str(tmp_path / "sensor.json")
        with patch("sensor_server.DATA_FILE", path):
            save_data({"v": 1})
            save_data({"v": 2})
            loaded = load_data()
        assert loaded["v"] == 2
