#!/usr/bin/env python3
"""Tests for alerts.py — powder alerts, cooldown, notifications, config."""

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from alerts import (
    check_cooldown,
    load_config,
    load_state,
    log,
    mark_alerted,
    save_state,
    send_webhook,
    DEFAULT_CONFIG,
)


# =========================================================================
# load_config / load_state / save_state
# =========================================================================

class TestConfig:
    def test_load_creates_default(self, tmp_path):
        path = tmp_path / "config.json"
        with patch("alerts.CONFIG_FILE", path):
            config = load_config()
        assert config["enabled"] is True
        assert "thresholds" in config
        assert path.exists()

    def test_load_existing(self, tmp_path):
        path = tmp_path / "config.json"
        custom = {"enabled": False, "thresholds": {"snow_24h_inches": 10}}
        with open(path, "w") as f:
            json.dump(custom, f)
        with patch("alerts.CONFIG_FILE", path):
            config = load_config()
        assert config["enabled"] is False
        assert config["thresholds"]["snow_24h_inches"] == 10


class TestState:
    def test_load_empty(self, tmp_path):
        path = tmp_path / "state.json"
        with patch("alerts.STATE_FILE", path):
            state = load_state()
        assert state == {"last_alerts": {}}

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "state.json"
        state = {"last_alerts": {"test_key": "2025-03-20T12:00:00+00:00"}}
        with patch("alerts.STATE_FILE", path):
            save_state(state)
            loaded = load_state()
        assert loaded["last_alerts"]["test_key"] == "2025-03-20T12:00:00+00:00"

    def test_load_corrupted(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("not json{{{")
        with patch("alerts.STATE_FILE", path):
            state = load_state()
        assert state == {"last_alerts": {}}


# =========================================================================
# Cooldown logic
# =========================================================================

class TestCooldown:
    def test_no_previous_alert(self):
        state = {"last_alerts": {}}
        assert check_cooldown(state, "resort_snow", 6) is False

    def test_within_cooldown(self):
        now = datetime.now(timezone.utc)
        state = {"last_alerts": {"resort_snow": now.isoformat()}}
        assert check_cooldown(state, "resort_snow", 6) is True

    def test_after_cooldown(self):
        old = datetime.now(timezone.utc) - timedelta(hours=7)
        state = {"last_alerts": {"resort_snow": old.isoformat()}}
        assert check_cooldown(state, "resort_snow", 6) is False

    def test_invalid_timestamp(self):
        state = {"last_alerts": {"key": "not-a-date"}}
        assert check_cooldown(state, "key", 6) is False

    def test_mark_alerted(self):
        state = {"last_alerts": {}}
        mark_alerted(state, "test_key")
        assert "test_key" in state["last_alerts"]
        # Verify it's a valid ISO timestamp
        dt = datetime.fromisoformat(state["last_alerts"]["test_key"])
        assert dt.tzinfo is not None


# =========================================================================
# send_webhook
# =========================================================================

class TestSendWebhook:
    def test_empty_url_noop(self):
        # Should not raise
        send_webhook("", "Test", "Body")

    @patch("alerts.requests.post")
    def test_ntfy(self, mock_post):
        send_webhook("https://ntfy.sh/tahoe", "Powder!", "12 inches")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "ntfy" in args[0]
        assert kwargs["headers"]["Title"] == "Powder!"

    @patch("alerts.requests.post")
    def test_discord(self, mock_post):
        send_webhook("https://discord.com/api/webhooks/123", "Alert", "Snow")
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "**Alert**" in kwargs["json"]["content"]

    @patch("alerts.requests.post")
    def test_slack(self, mock_post):
        send_webhook("https://hooks.slack.com/services/x/y/z", "Alert", "Snow")
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert "*Alert*" in kwargs["json"]["text"]

    @patch("alerts.requests.post", side_effect=Exception("timeout"))
    def test_webhook_error_handled(self, mock_post, tmp_path):
        log_file = tmp_path / "alerts.log"
        with patch("alerts.LOG_FILE", log_file):
            send_webhook("https://example.com/hook", "Test", "Body")
        # Should not raise


# =========================================================================
# log
# =========================================================================

class TestLog:
    def test_writes_to_file(self, tmp_path):
        log_file = tmp_path / "alerts.log"
        with patch("alerts.LOG_FILE", log_file):
            log("Test message")
        content = log_file.read_text()
        assert "Test message" in content

    def test_appends(self, tmp_path):
        log_file = tmp_path / "alerts.log"
        with patch("alerts.LOG_FILE", log_file):
            log("First")
            log("Second")
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2


# =========================================================================
# DEFAULT_CONFIG structure
# =========================================================================

class TestDefaultConfig:
    def test_has_required_keys(self):
        assert "enabled" in DEFAULT_CONFIG
        assert "thresholds" in DEFAULT_CONFIG
        assert "notifications" in DEFAULT_CONFIG
        assert "cooldown_hours" in DEFAULT_CONFIG

    def test_thresholds(self):
        t = DEFAULT_CONFIG["thresholds"]
        assert t["snow_24h_inches"] > 0
        assert t["snow_48h_inches"] > t["snow_24h_inches"]
        assert t["powder_quality_slr_min"] >= 10
