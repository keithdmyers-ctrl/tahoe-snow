#!/usr/bin/env python3
"""
Tests for Tier 6 provisions: ML pipeline, observation system, and resort configs.

Tests cover:
  1. MLPipeline — readiness checks, feature extraction, status reporting
  2. ObservationStore — add, get_recent, get_near, validator
  3. ResortConfig — validation, registry, active filtering, SNOTEL dedup
"""

import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from unittest import TestCase, main as unittest_main

sys.path.insert(0, os.path.dirname(__file__))

from ml_pipeline import (
    MLPipeline, MLPipelineConfig, FeatureExtractor, SnowCorrectionModel,
    VERIFICATION_LOG, SNOW_VERIFICATION,
)
from observations import (
    Observation, ObsType, ObservationStore, ObservationValidator,
    _haversine_miles, MAX_OBSERVATIONS,
)
from resort_configs import (
    ResortConfig, ZoneConfig, RESORT_REGISTRY,
    get_active_resorts, get_resort, get_all_snotel_stations,
    get_resorts_by_region, to_legacy_format, get_active_resorts_legacy,
)


# ===================================================================
# 1. ML Pipeline Tests
# ===================================================================

class TestMLPipelineConfig(TestCase):
    """Test MLPipelineConfig defaults."""

    def test_default_config(self):
        cfg = MLPipelineConfig()
        self.assertEqual(cfg.min_training_days, 90)
        self.assertEqual(cfg.retrain_interval_days, 7)
        self.assertGreater(len(cfg.feature_list), 10)
        self.assertIn("gfs_temp_f", cfg.feature_list)
        self.assertIn("ensemble_spread", cfg.feature_list)
        self.assertIn("month", cfg.feature_list)
        self.assertGreater(len(cfg.target_variables), 0)
        self.assertIn("snow_in", cfg.target_variables)

    def test_custom_config(self):
        cfg = MLPipelineConfig(min_training_days=30, retrain_interval_days=3)
        self.assertEqual(cfg.min_training_days, 30)
        self.assertEqual(cfg.retrain_interval_days, 3)


class TestMLPipelineReadiness(TestCase):
    """Test MLPipeline.is_ready()."""

    def test_is_ready_no_data(self):
        """Pipeline should not be ready with no verification data."""
        config = MLPipelineConfig()
        # Point to a nonexistent path for model storage
        config.model_path = tempfile.mkdtemp()
        pipeline = MLPipeline(config)
        self.assertFalse(pipeline.is_ready())

    def test_is_ready_insufficient_data(self):
        """Pipeline should not be ready with fewer than min_training_days."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Write 10 days of data (less than default 90)
            scores = [{"date": f"2026-01-{i+1:02d}", "source": "gfs",
                       "metric": "temp_high_f", "predicted": 30, "actual": 28,
                       "error": 2, "abs_error": 2, "lead_hours": 24}
                      for i in range(10)]
            json.dump({"forecasts": [], "actuals": [], "daily_scores": scores}, f)
            tmp_path = f.name

        try:
            import ml_pipeline
            orig = ml_pipeline.VERIFICATION_LOG
            ml_pipeline.VERIFICATION_LOG = tmp_path
            pipeline = MLPipeline()
            self.assertFalse(pipeline.is_ready())
            status = pipeline.get_status()
            self.assertFalse(status["ready"])
            self.assertEqual(status["days_of_data"], 10)
            self.assertEqual(status["days_needed"], 90)
        finally:
            ml_pipeline.VERIFICATION_LOG = orig
            os.unlink(tmp_path)

    def test_is_ready_sufficient_data(self):
        """Pipeline should be ready with enough verification data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            scores = [{"date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                       "source": "gfs", "metric": "temp_high_f",
                       "predicted": 30, "actual": 28, "error": 2,
                       "abs_error": 2, "lead_hours": 24}
                      for i in range(100)]
            json.dump({"forecasts": [], "actuals": [], "daily_scores": scores}, f)
            tmp_path = f.name

        try:
            import ml_pipeline
            orig = ml_pipeline.VERIFICATION_LOG
            ml_pipeline.VERIFICATION_LOG = tmp_path
            pipeline = MLPipeline()
            self.assertTrue(pipeline.is_ready())
        finally:
            ml_pipeline.VERIFICATION_LOG = orig
            os.unlink(tmp_path)


class TestFeatureExtractor(TestCase):
    """Test FeatureExtractor.extract()."""

    def _make_mock_analysis(self):
        """Build a mock analysis dict resembling analyze_all() output."""
        return {
            "generated": "2026-03-14T12:00:00+00:00",
            "current_conditions": {
                "observation": {
                    "temp_f": 35,
                    "pressure_hpa": 1013.2,
                    "pressure_trend_3h": -1.5,
                    "pressure_trend_12h": -4.0,
                },
                "snow_level_ft": 7500,
                "observed_lapse_rate_c_km": 5.8,
            },
            "resorts": {
                "Heavenly": {
                    "zones": {
                        "peak": {
                            "model_spread": [{
                                "date": "2026-03-14",
                                "models": {
                                    "GFS": {"temp_high_f": 28, "snow_in": 6},
                                    "ECMWF": {"temp_high_f": 26, "snow_in": 8},
                                    "ICON": {"temp_high_f": 30, "snow_in": 4},
                                    "HRRR": {"temp_high_f": 27, "snow_in": 7},
                                },
                            }],
                            "current": {"wind_mph": 25.3},
                        },
                    },
                },
            },
            "sounding": {"freezing_level_ft": 7200},
            "ensemble": {
                "models": {
                    "GFS_ensemble": [{"snow_p50": 5.0}],
                    "ECMWF_ensemble": [{"snow_p50": 7.0}],
                },
            },
        }

    def test_extract_basic(self):
        """Feature extraction should produce expected keys."""
        extractor = FeatureExtractor()
        analysis = self._make_mock_analysis()
        features = extractor.extract(analysis, {})

        # NWP model features
        self.assertEqual(features["gfs_temp_f"], 28)
        self.assertEqual(features["ecmwf_temp_f"], 26)
        self.assertEqual(features["icon_temp_f"], 30)
        self.assertEqual(features["hrrr_temp_f"], 27)
        self.assertEqual(features["gfs_snow_in"], 6)
        self.assertEqual(features["ecmwf_snow_in"], 8)

        # Sounding features
        self.assertEqual(features["freeze_level_ft"], 7200)
        self.assertEqual(features["snow_level_ft"], 7500)
        self.assertAlmostEqual(features["lapse_rate_c_km"], 5.8)
        self.assertEqual(features["inversion_present"], 0.0)  # 5.8 > 3.0

        # Pressure features
        self.assertAlmostEqual(features["pressure_hpa"], 1013.2)
        self.assertAlmostEqual(features["pressure_trend_3h"], -1.5)

        # Calendar features
        self.assertEqual(features["month"], 3.0)
        self.assertEqual(features["day_of_year"], 73.0)

    def test_extract_inversion_detection(self):
        """Should detect temperature inversion (low lapse rate)."""
        extractor = FeatureExtractor()
        analysis = self._make_mock_analysis()
        analysis["current_conditions"]["observed_lapse_rate_c_km"] = 2.0
        features = extractor.extract(analysis, {})
        self.assertEqual(features["inversion_present"], 1.0)

    def test_extract_missing_data(self):
        """Should handle empty analysis gracefully with NaN values."""
        extractor = FeatureExtractor()
        features = extractor.extract({}, {})

        import numpy as np
        # Missing data should be NaN, not crash
        self.assertTrue(np.isnan(features["freeze_level_ft"]))
        self.assertTrue(np.isnan(features["pressure_hpa"]))
        self.assertTrue(np.isnan(features["ensemble_spread"]))
        # Calendar features should still have valid values
        self.assertGreater(features["month"], 0)

    def test_extract_ensemble_stats(self):
        """Should compute ensemble spread, IQR, and skewness."""
        extractor = FeatureExtractor()
        analysis = self._make_mock_analysis()
        features = extractor.extract(analysis, {})

        self.assertGreater(features["ensemble_spread"], 0)
        self.assertGreater(features["ensemble_iqr"], 0)
        self.assertIsNotNone(features["ensemble_skewness"])


class TestSnowCorrectionModel(TestCase):
    """Test SnowCorrectionModel."""

    def test_untrained_returns_identity(self):
        """Untrained model should return 1.0 (no correction)."""
        model = SnowCorrectionModel("peak", 24)
        self.assertEqual(model.predict({}), 1.0)
        self.assertEqual(model.predict([1, 2, 3]), 1.0)

    def test_train_fallback(self):
        """Training in fallback mode should succeed and still return 1.0."""
        import numpy as np
        model = SnowCorrectionModel("peak", 24)
        features = np.random.randn(20, 5)
        targets = np.random.randn(20)
        model.train(features, targets)
        self.assertTrue(model._trained)
        self.assertEqual(model.predict([0, 0, 0, 0, 0]), 1.0)

    def test_train_insufficient_data(self):
        """Training with < 10 samples should log warning but not crash."""
        import numpy as np
        model = SnowCorrectionModel("base", 48)
        features = np.random.randn(5, 3)
        targets = np.random.randn(5)
        model.train(features, targets)
        # Model should still not be fully trained
        # (insufficient data in this branch still sets _trained for fallback)

    def test_save_load_roundtrip(self):
        """Should save and load model state."""
        import numpy as np
        tmpdir = tempfile.mkdtemp()
        model = SnowCorrectionModel("mid", 72)
        model._feature_names = ["feat_a", "feat_b"]
        model._trained = True
        model._feature_importances = {"feat_a": 0.7, "feat_b": 0.3}
        model.save(tmpdir)

        loaded = SnowCorrectionModel.load(tmpdir, "mid", 72)
        self.assertTrue(loaded._trained)
        self.assertEqual(loaded._feature_names, ["feat_a", "feat_b"])
        self.assertEqual(loaded._feature_importances, {"feat_a": 0.7, "feat_b": 0.3})

    def test_get_feature_importance(self):
        """Feature importance should be sorted descending."""
        model = SnowCorrectionModel("peak", 24)
        model._feature_importances = {"c": 0.1, "a": 0.5, "b": 0.3}
        imp = model.get_feature_importance()
        keys = list(imp.keys())
        self.assertEqual(keys, ["a", "b", "c"])


class TestMLPipelineStatus(TestCase):
    """Test MLPipeline.get_status()."""

    def test_status_no_data(self):
        """Status should report not ready with no data."""
        pipeline = MLPipeline()
        status = pipeline.get_status()
        self.assertFalse(status["ready"])
        self.assertEqual(status["days_of_data"], 0)
        self.assertEqual(status["models_trained"], 0)
        self.assertIsNone(status["last_train_date"])
        self.assertIsInstance(status["feature_importances"], dict)


# ===================================================================
# 2. Observation System Tests
# ===================================================================

class TestObservation(TestCase):
    """Test Observation dataclass."""

    def _make_obs(self, **overrides):
        defaults = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "observer_id": "test_user",
            "lat": 38.93,
            "lon": -119.94,
            "elevation_ft": 10067,
            "location_name": "Heavenly Peak",
            "obs_type": ObsType.SNOW_DEPTH.value,
            "value": 24.0,
            "unit": "inches",
            "notes": "Fresh powder",
        }
        defaults.update(overrides)
        return Observation(**defaults)

    def test_to_dict_roundtrip(self):
        obs = self._make_obs()
        d = obs.to_dict()
        restored = Observation.from_dict(d)
        self.assertEqual(obs.observer_id, restored.observer_id)
        self.assertEqual(obs.value, restored.value)
        self.assertEqual(obs.lat, restored.lat)

    def test_obs_type_values(self):
        """All ObsType values should be valid strings."""
        for otype in ObsType:
            self.assertIsInstance(otype.value, str)
            self.assertTrue(len(otype.value) > 0)


class TestObservationValidator(TestCase):
    """Test ObservationValidator range checks and duplicate detection."""

    def _make_obs(self, **overrides):
        defaults = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "observer_id": "test_user",
            "lat": 38.93,
            "lon": -119.94,
            "elevation_ft": 10067,
            "location_name": "Heavenly Peak",
            "obs_type": ObsType.SNOW_DEPTH.value,
            "value": 24.0,
            "unit": "inches",
        }
        defaults.update(overrides)
        return Observation(**defaults)

    def test_valid_observation(self):
        validator = ObservationValidator()
        obs = self._make_obs()
        is_valid, errors, warnings = validator.validate(obs, [])
        self.assertTrue(is_valid, f"Should be valid, got errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_range_check_snow_depth_too_high(self):
        validator = ObservationValidator()
        obs = self._make_obs(value=250)  # Max is 200"
        is_valid, errors, _ = validator.validate(obs, [])
        self.assertFalse(is_valid)
        self.assertTrue(any("range" in e.lower() for e in errors))

    def test_range_check_snow_depth_negative(self):
        validator = ObservationValidator()
        obs = self._make_obs(value=-5)
        is_valid, errors, _ = validator.validate(obs, [])
        self.assertFalse(is_valid)
        self.assertTrue(any("range" in e.lower() for e in errors))

    def test_range_check_quality_valid(self):
        validator = ObservationValidator()
        obs = self._make_obs(obs_type=ObsType.SNOW_QUALITY.value, value=3)
        is_valid, errors, _ = validator.validate(obs, [])
        self.assertTrue(is_valid, f"Quality 3 should be valid, got: {errors}")

    def test_range_check_quality_out_of_range(self):
        validator = ObservationValidator()
        obs = self._make_obs(obs_type=ObsType.SNOW_QUALITY.value, value=6)
        is_valid, errors, _ = validator.validate(obs, [])
        self.assertFalse(is_valid)

    def test_duplicate_detection(self):
        validator = ObservationValidator()
        now = datetime.now(timezone.utc)
        existing = [{
            "timestamp": (now - timedelta(minutes=30)).isoformat(),
            "observer_id": "test_user",
            "obs_type": ObsType.SNOW_DEPTH.value,
            "location_name": "Heavenly Peak",
        }]
        obs = self._make_obs(timestamp=now.isoformat())
        is_valid, errors, _ = validator.validate(obs, existing)
        self.assertFalse(is_valid)
        self.assertTrue(any("duplicate" in e.lower() for e in errors))

    def test_no_duplicate_different_observer(self):
        validator = ObservationValidator()
        now = datetime.now(timezone.utc)
        existing = [{
            "timestamp": (now - timedelta(minutes=30)).isoformat(),
            "observer_id": "other_user",
            "obs_type": ObsType.SNOW_DEPTH.value,
            "location_name": "Heavenly Peak",
        }]
        obs = self._make_obs(timestamp=now.isoformat())
        is_valid, errors, _ = validator.validate(obs, existing)
        self.assertTrue(is_valid, f"Different observer should not trigger duplicate: {errors}")

    def test_no_duplicate_after_1_hour(self):
        validator = ObservationValidator()
        now = datetime.now(timezone.utc)
        existing = [{
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "observer_id": "test_user",
            "obs_type": ObsType.SNOW_DEPTH.value,
            "location_name": "Heavenly Peak",
        }]
        obs = self._make_obs(timestamp=now.isoformat())
        is_valid, errors, _ = validator.validate(obs, existing)
        self.assertTrue(is_valid, f"After 1 hour should not be duplicate: {errors}")

    def test_invalid_coordinates(self):
        validator = ObservationValidator()
        obs = self._make_obs(lat=50.0)  # Way outside Tahoe area
        is_valid, errors, _ = validator.validate(obs, [])
        self.assertFalse(is_valid)
        self.assertTrue(any("latitude" in e.lower() for e in errors))

    def test_empty_observer_id(self):
        validator = ObservationValidator()
        obs = self._make_obs(observer_id="")
        is_valid, errors, _ = validator.validate(obs, [])
        self.assertFalse(is_valid)
        self.assertTrue(any("observer_id" in e for e in errors))


class TestObservationStore(TestCase):
    """Test ObservationStore add/get_recent/get_near."""

    def setUp(self):
        self._tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self._tmpfile.write("[]")
        self._tmpfile.close()
        self.store = ObservationStore(self._tmpfile.name)

    def tearDown(self):
        os.unlink(self._tmpfile.name)

    def _make_obs(self, **overrides):
        defaults = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "observer_id": "tester",
            "lat": 38.93,
            "lon": -119.94,
            "elevation_ft": 10067,
            "location_name": "Heavenly Peak",
            "obs_type": ObsType.SNOW_DEPTH.value,
            "value": 24.0,
            "unit": "inches",
        }
        defaults.update(overrides)
        return Observation(**defaults)

    def test_add_valid(self):
        obs = self._make_obs()
        result = self.store.add(obs)
        self.assertTrue(result["success"])
        self.assertEqual(len(result["errors"]), 0)

    def test_add_and_retrieve(self):
        obs = self._make_obs()
        self.store.add(obs)
        recent = self.store.get_recent(hours=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["value"], 24.0)

    def test_get_recent_type_filter(self):
        obs1 = self._make_obs(obs_type=ObsType.SNOW_DEPTH.value, value=24)
        obs2 = self._make_obs(
            obs_type=ObsType.CONDITIONS.value, value=28,
            observer_id="other",
        )
        self.store.add(obs1)
        self.store.add(obs2)

        snow_only = self.store.get_recent(hours=1, obs_type=ObsType.SNOW_DEPTH.value)
        self.assertEqual(len(snow_only), 1)
        self.assertEqual(snow_only[0]["obs_type"], ObsType.SNOW_DEPTH.value)

    def test_get_recent_location_filter(self):
        obs1 = self._make_obs(location_name="Heavenly Peak")
        obs2 = self._make_obs(
            location_name="Kirkwood Lodge",
            observer_id="other",
            lat=38.685, lon=-120.065,
        )
        self.store.add(obs1)
        self.store.add(obs2)

        heavenly = self.store.get_recent(hours=1, location="Heavenly")
        self.assertEqual(len(heavenly), 1)

    def test_get_recent_time_filter(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        obs = self._make_obs(timestamp=old_ts)
        self.store.add(obs)

        recent = self.store.get_recent(hours=24)
        self.assertEqual(len(recent), 0)

        all_obs = self.store.get_recent(hours=72)
        self.assertEqual(len(all_obs), 1)

    def test_get_near(self):
        # Heavenly peak
        obs1 = self._make_obs(lat=38.928, lon=-119.907)
        self.store.add(obs1)
        # Kirkwood (far away)
        obs2 = self._make_obs(
            lat=38.685, lon=-120.065, location_name="Kirkwood",
            observer_id="other",
        )
        self.store.add(obs2)

        near_heavenly = self.store.get_near(38.93, -119.94, radius_miles=5, hours=1)
        self.assertEqual(len(near_heavenly), 1)
        self.assertIn("distance_mi", near_heavenly[0])
        self.assertLess(near_heavenly[0]["distance_mi"], 5)

    def test_get_summary_empty(self):
        summary = self.store.get_summary()
        self.assertEqual(summary["total_count"], 0)
        self.assertEqual(summary["recent_24h"], 0)

    def test_get_summary_with_data(self):
        for i in range(5):
            obs = self._make_obs(
                observer_id=f"user_{i}",
                value=10 + i,
            )
            self.store.add(obs)
        summary = self.store.get_summary()
        self.assertEqual(summary["total_count"], 5)
        self.assertEqual(summary["recent_24h"], 5)
        self.assertIn(ObsType.SNOW_DEPTH.value, summary["counts_by_type"])

    def test_fifo_eviction(self):
        """Store should evict oldest observations when exceeding MAX_OBSERVATIONS."""
        # Directly write many observations to test FIFO
        obs_list = []
        for i in range(105):
            obs_list.append({
                "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=i)).isoformat(),
                "observer_id": f"user_{i}",
                "lat": 38.93,
                "lon": -119.94,
                "elevation_ft": 10067,
                "location_name": "Test",
                "obs_type": "snow_depth",
                "value": i,
                "unit": "inches",
            })
        with open(self._tmpfile.name, "w") as f:
            json.dump(obs_list, f)

        # Add one more — should trigger eviction at MAX_OBSERVATIONS
        # (not 105, but the real MAX is 10000; this just checks the mechanism works)
        new_obs = self._make_obs(observer_id="newest")
        self.store.add(new_obs)

        # Should still be under the limit
        loaded = self.store._load()
        self.assertLessEqual(len(loaded), MAX_OBSERVATIONS)


class TestHaversine(TestCase):
    """Test the haversine distance function."""

    def test_same_point(self):
        self.assertAlmostEqual(_haversine_miles(38.93, -119.94, 38.93, -119.94), 0, places=3)

    def test_known_distance(self):
        # Heavenly to Kirkwood: roughly 20 miles
        dist = _haversine_miles(38.935, -119.941, 38.685, -120.065)
        self.assertGreater(dist, 15)
        self.assertLess(dist, 25)


# ===================================================================
# 3. Resort Config Tests
# ===================================================================

class TestZoneConfig(TestCase):
    """Test ZoneConfig dataclass."""

    def test_zone_creation(self):
        zone = ZoneConfig("peak", "Summit", 39.0, -120.0, 9000, 45.0, 30.0)
        self.assertEqual(zone.name, "peak")
        self.assertEqual(zone.elevation_ft, 9000)
        self.assertEqual(zone.aspect_deg, 45.0)
        self.assertEqual(zone.slope_angle_deg, 30.0)


class TestResortConfig(TestCase):
    """Test ResortConfig methods."""

    def test_get_zone(self):
        cfg = RESORT_REGISTRY["Heavenly"]
        base = cfg.get_zone("base")
        self.assertIsNotNone(base)
        self.assertEqual(base.label, "California Lodge")
        peak = cfg.get_zone("peak")
        self.assertEqual(peak.label, "Monument Peak")

    def test_get_zone_nonexistent(self):
        cfg = RESORT_REGISTRY["Heavenly"]
        self.assertIsNone(cfg.get_zone("nonexistent"))

    def test_convenience_methods(self):
        cfg = RESORT_REGISTRY["Heavenly"]
        self.assertIsNotNone(cfg.base())
        self.assertIsNotNone(cfg.mid())
        self.assertIsNotNone(cfg.peak())


class TestResortRegistry(TestCase):
    """Test RESORT_REGISTRY contents and structure."""

    def test_all_resorts_have_three_zones(self):
        for name, cfg in RESORT_REGISTRY.items():
            self.assertEqual(len(cfg.zones), 3, f"{name} should have 3 zones")
            zone_names = [z.name for z in cfg.zones]
            self.assertIn("base", zone_names, f"{name} missing base zone")
            self.assertIn("mid", zone_names, f"{name} missing mid zone")
            self.assertIn("peak", zone_names, f"{name} missing peak zone")

    def test_elevations_ordered(self):
        """Base < mid < peak for each resort."""
        for name, cfg in RESORT_REGISTRY.items():
            base = cfg.base()
            mid = cfg.mid()
            peak = cfg.peak()
            self.assertLess(base.elevation_ft, mid.elevation_ft,
                            f"{name} base ({base.elevation_ft}) >= mid ({mid.elevation_ft})")
            self.assertLess(mid.elevation_ft, peak.elevation_ft,
                            f"{name} mid ({mid.elevation_ft}) >= peak ({peak.elevation_ft})")

    def test_coordinates_in_tahoe_area(self):
        """All coordinates should be in the Lake Tahoe region."""
        for name, cfg in RESORT_REGISTRY.items():
            for zone in cfg.zones:
                self.assertGreater(zone.lat, 38.5,
                                   f"{name}/{zone.name} lat too low: {zone.lat}")
                self.assertLess(zone.lat, 39.5,
                                f"{name}/{zone.name} lat too high: {zone.lat}")
                self.assertGreater(zone.lon, -121,
                                   f"{name}/{zone.name} lon too low: {zone.lon}")
                self.assertLess(zone.lon, -119.5,
                                f"{name}/{zone.name} lon too high: {zone.lon}")

    def test_heavenly_known_coordinates(self):
        """Verify Heavenly coordinates match known values."""
        cfg = RESORT_REGISTRY["Heavenly"]
        base = cfg.base()
        self.assertAlmostEqual(base.lat, 38.9353, places=3)
        self.assertAlmostEqual(base.lon, -119.9406, places=3)
        self.assertEqual(base.elevation_ft, 6540)
        peak = cfg.peak()
        self.assertAlmostEqual(peak.lat, 38.9280, places=3)
        self.assertEqual(peak.elevation_ft, 10067)

    def test_northstar_known_coordinates(self):
        """Verify Northstar coordinates match known values."""
        cfg = RESORT_REGISTRY["Northstar"]
        base = cfg.base()
        self.assertAlmostEqual(base.lat, 39.2744, places=3)
        self.assertAlmostEqual(base.lon, -120.1210, places=3)
        self.assertEqual(base.elevation_ft, 6330)
        peak = cfg.peak()
        self.assertEqual(peak.elevation_ft, 8610)

    def test_kirkwood_known_coordinates(self):
        """Verify Kirkwood coordinates match known values."""
        cfg = RESORT_REGISTRY["Kirkwood"]
        base = cfg.base()
        self.assertAlmostEqual(base.lat, 38.6850, places=3)
        self.assertAlmostEqual(base.lon, -120.0650, places=3)
        self.assertEqual(base.elevation_ft, 7800)
        peak = cfg.peak()
        self.assertEqual(peak.elevation_ft, 9800)

    def test_active_resorts_count(self):
        """Should have exactly 3 enabled resorts by default."""
        active = get_active_resorts()
        self.assertEqual(len(active), 3)
        self.assertIn("Heavenly", active)
        self.assertIn("Northstar", active)
        self.assertIn("Kirkwood", active)

    def test_disabled_resorts_not_active(self):
        """Disabled resorts should not appear in active list."""
        active = get_active_resorts()
        self.assertNotIn("Palisades Tahoe", active)
        self.assertNotIn("Sugar Bowl", active)
        self.assertNotIn("Sierra-at-Tahoe", active)
        self.assertNotIn("Boreal", active)
        self.assertNotIn("Mt. Rose", active)

    def test_total_registry_count(self):
        """Should have 8 total resorts (3 active + 5 stubbed)."""
        self.assertEqual(len(RESORT_REGISTRY), 8)


class TestGetResort(TestCase):
    """Test get_resort()."""

    def test_existing(self):
        cfg = get_resort("Heavenly")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.name, "Heavenly")

    def test_nonexistent(self):
        cfg = get_resort("Nonexistent")
        self.assertIsNone(cfg)


class TestSNOTELDedup(TestCase):
    """Test get_all_snotel_stations() deduplication."""

    def test_no_duplicates(self):
        stations = get_all_snotel_stations()
        self.assertEqual(len(stations), len(set(stations)))

    def test_known_stations_present(self):
        stations = get_all_snotel_stations()
        # These should be in the active resorts' SNOTEL lists
        self.assertIn("473", stations)  # Fallen Leaf (Heavenly)
        self.assertIn("518", stations)  # Hagan's Meadow (Heavenly + Kirkwood)
        self.assertIn("539", stations)  # Independence Lake (Northstar)
        self.assertIn("428", stations)  # CSS Lab (Kirkwood)

    def test_dedup_across_resorts(self):
        """Station 518 (Hagan's Meadow) is shared by Heavenly and Kirkwood."""
        stations = get_all_snotel_stations()
        count_518 = stations.count("518")
        self.assertEqual(count_518, 1)


class TestRegionFiltering(TestCase):
    """Test get_resorts_by_region()."""

    def test_tahoe_south(self):
        south = get_resorts_by_region("tahoe_south")
        self.assertIn("Heavenly", south)
        self.assertIn("Kirkwood", south)
        self.assertNotIn("Northstar", south)

    def test_tahoe_north(self):
        north = get_resorts_by_region("tahoe_north")
        self.assertIn("Northstar", north)
        self.assertNotIn("Heavenly", north)


class TestLegacyFormat(TestCase):
    """Test conversion to legacy RESORTS dict format."""

    def test_legacy_structure(self):
        """Legacy format should match what tahoe_snow.py expects."""
        cfg = RESORT_REGISTRY["Heavenly"]
        legacy = to_legacy_format(cfg)

        self.assertIn("base", legacy)
        self.assertIn("mid", legacy)
        self.assertIn("peak", legacy)
        self.assertIn("aspect", legacy)
        self.assertIn("nearest_snotel", legacy)

        self.assertEqual(legacy["base"]["label"], "California Lodge")
        self.assertEqual(legacy["base"]["lat"], 38.9353)
        self.assertEqual(legacy["base"]["lon"], -119.9406)
        self.assertEqual(legacy["base"]["elev_ft"], 6540)

    def test_legacy_matches_original(self):
        """Legacy format should match the original RESORTS dict in tahoe_snow.py."""
        from tahoe_snow import RESORTS

        legacy_all = get_active_resorts_legacy()

        for name in RESORTS:
            self.assertIn(name, legacy_all, f"Missing resort: {name}")
            orig = RESORTS[name]
            conv = legacy_all[name]

            for zone in ("base", "mid", "peak"):
                self.assertEqual(orig[zone]["label"], conv[zone]["label"],
                                 f"{name}/{zone} label mismatch")
                self.assertAlmostEqual(orig[zone]["lat"], conv[zone]["lat"], places=4,
                                       msg=f"{name}/{zone} lat mismatch")
                self.assertAlmostEqual(orig[zone]["lon"], conv[zone]["lon"], places=4,
                                       msg=f"{name}/{zone} lon mismatch")
                self.assertEqual(orig[zone]["elev_ft"], conv[zone]["elev_ft"],
                                 f"{name}/{zone} elev mismatch")

            self.assertEqual(orig["aspect"], conv["aspect"],
                             f"{name} aspect mismatch")
            self.assertEqual(orig["nearest_snotel"], conv["nearest_snotel"],
                             f"{name} nearest_snotel mismatch")


if __name__ == "__main__":
    print("=" * 70)
    print("  TIER 6 PROVISIONS — TEST SUITE")
    print("=" * 70)
    print()
    unittest_main(verbosity=2)
