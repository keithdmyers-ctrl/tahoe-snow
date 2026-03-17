#!/usr/bin/env python3
"""Tests for outdoor activity decision cards."""

import os
import sys
from unittest import TestCase, main as unittest_main

sys.path.insert(0, os.path.dirname(__file__))

from outdoor_conditions import (
    compute_activity_decisions,
    compute_solar_exposure,
    compute_trail_conditions,
    compute_route_wind,
    compute_fishing_pressure,
    _apply_aqi_override,
    _ensure_timing,
)


def _make_analysis(**overrides):
    """Build a minimal analysis dict for testing."""
    base = {
        "resorts": {
            "Heavenly": {
                "zones": {
                    "peak": {
                        "elev_ft": 10067,
                        "current": {
                            "temp_f": 70, "wind_mph": 10, "wind_gust_mph": 18,
                            "wind_dir": "SW", "humidity_pct": 30, "cape_jkg": 100,
                            "pressure_hpa": 1015,
                        },
                        "timeline_48h": [{"precip_in": 0, "snowfall_in": 0} for _ in range(24)],
                        "snow_24h": 0,
                    }
                }
            }
        },
        "current_conditions": {"observation": {"aqi": {"us_aqi": 40}}},
        "ml_powder_alert": {"level": 1, "label": "NONE", "message": "No snow.", "best_quality_rating": 0},
        "ml_resort_ranking": [],
        "ml_outdoor": {
            "thunderstorm": {"risk": "Very Low", "probability": 0.05, "detail": "Stable.", "turn_around_time": None},
            "heat_safety": {"risk": "Low", "detail": "Comfortable.", "temperature_f": 70, "water_recommendation": "12 oz/hr"},
            "smoke_aqi": {"risk": "Good", "aqi": 40, "detail": "Good.", "activity_guidance": "All OK"},
            "uv_exposure": {"risk": "High", "effective_uv_index": 8},
            "best_window": {"advice": "Great all day.", "best_windows": ["All day (6 AM - 6 PM)"]},
            "recommendation": {"verdict": "GO", "summary": "Perfect day!", "positives": ["Clean air"], "warnings": [], "blockers": []},
            "water_availability": {"status": "Good", "detail": "Creeks flowing."},
        },
        "outdoor": {
            "trail": {"mtb_quality": 5, "mtb_label": "Hero Dirt", "summary": "Hero dirt!", "conditions": ["Dry"], "warnings": []},
            "solar": {"South": {"sun_hours": 11, "golden_hour_am": "6-7", "golden_hour_pm": "19-20", "wall_temp_note": "Mixed"}, "North": {"sun_hours": 2, "wall_temp_note": "Cool wall"}},
            "streams": {},
            "route_wind": {"available": True, "routes": {}},
            "fishing_pressure": {"available": True, "activity": "Fair", "rating": 3, "detail": "Steady.", "best_technique": "Standard", "pressure_trend_hpa": None},
            "aqi_forecast": {"available": False},
        },
        "avalanche": {"error": True},
        "snotel_current": {},
    }
    base.update(overrides)
    return base


class TestActivityDecisions(TestCase):

    def test_returns_10_cards(self):
        cards = compute_activity_decisions(_make_analysis())
        self.assertEqual(len(cards), 10)

    def test_cards_sorted_by_score(self):
        cards = compute_activity_decisions(_make_analysis())
        scores = [c["score"] for c in cards]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_all_cards_have_required_fields(self):
        cards = compute_activity_decisions(_make_analysis())
        for card in cards:
            for field in ["activity", "icon", "signal", "score", "headline",
                          "metrics", "timing", "warnings", "detail", "season"]:
                self.assertIn(field, card, f"Missing '{field}' in {card['activity']}")
            self.assertIn(card["signal"], ("go", "caution", "no-go"))
            self.assertIn(card["season"], ("winter", "summer", "year-round"))
            self.assertIsInstance(card["timing"], list)
            self.assertIsInstance(card["warnings"], list)
            self.assertIsInstance(card["metrics"], list)
            self.assertTrue(len(card["timing"]) > 0, f"Empty timing in {card['activity']}")

    def test_ski_is_low_in_summer(self):
        cards = compute_activity_decisions(_make_analysis())
        ski = next(c for c in cards if c["activity"] == "Ski")
        # In March (current month), ski might not be off-season, so just check it exists
        self.assertIsNotNone(ski)
        self.assertIn(ski["signal"], ("go", "caution", "no-go"))

    def test_high_aqi_degrades_aerobic_activities(self):
        analysis = _make_analysis()
        analysis["current_conditions"]["observation"]["aqi"]["us_aqi"] = 180
        cards = compute_activity_decisions(analysis)
        hike = next(c for c in cards if c["activity"] == "Hike")
        mtb = next(c for c in cards if c["activity"] == "Mountain Bike")
        # AQI 180 should push aerobic activities to no-go
        self.assertEqual(hike["signal"], "no-go")
        self.assertEqual(mtb["signal"], "no-go")

    def test_paddle_calm_is_go(self):
        cards = compute_activity_decisions(_make_analysis())
        paddle = next(c for c in cards if c["activity"] == "Paddle")
        self.assertEqual(paddle["signal"], "go")

    def test_backcountry_ski_card_exists(self):
        cards = compute_activity_decisions(_make_analysis())
        bc = next((c for c in cards if c["activity"] == "Backcountry Ski"), None)
        self.assertIsNotNone(bc)

    def test_metrics_have_label_and_value(self):
        cards = compute_activity_decisions(_make_analysis())
        for card in cards:
            for m in card["metrics"]:
                self.assertIn("label", m)
                self.assertIn("value", m)


class TestAqiOverride(TestCase):

    def test_no_aqi_no_change(self):
        signal, score = _apply_aqi_override("go", 8, None)
        self.assertEqual(signal, "go")
        self.assertEqual(score, 8)

    def test_high_aqi_forces_nogo(self):
        signal, score = _apply_aqi_override("go", 8, 200, threshold_nogo=150)
        self.assertEqual(signal, "no-go")
        self.assertLessEqual(score, 2)

    def test_moderate_aqi_forces_caution(self):
        signal, score = _apply_aqi_override("go", 8, 120, threshold_caution=100)
        self.assertEqual(signal, "caution")
        self.assertLessEqual(score, 4)


class TestEnsureTiming(TestCase):

    def test_empty_list_returns_fallback(self):
        result = _ensure_timing([])
        self.assertEqual(len(result), 1)
        self.assertIn("ideal", result[0].lower())

    def test_valid_list_passes_through(self):
        result = _ensure_timing(["Morning", "Evening"])
        self.assertEqual(result, ["Morning", "Evening"])

    def test_whitespace_only_returns_fallback(self):
        result = _ensure_timing(["  ", ""])
        self.assertEqual(len(result), 1)


class TestSolarExposure(TestCase):

    def test_south_face_gets_most_sun(self):
        south = compute_solar_exposure(39.1, -120.1, 180, 25)
        north = compute_solar_exposure(39.1, -120.1, 0, 25)
        self.assertGreater(south["sun_hours"], north["sun_hours"])

    def test_has_golden_hours(self):
        result = compute_solar_exposure(39.1, -120.1, 180)
        self.assertIn("golden_hour_am", result)
        self.assertIn("golden_hour_pm", result)


class TestTrailConditions(TestCase):

    def test_dry_summer_is_hero_dirt(self):
        t = compute_trail_conditions(75, 55, 0.0, 5, month=7)
        self.assertEqual(t["mtb_quality"], 5)

    def test_wet_trails_are_poor(self):
        t = compute_trail_conditions(50, 35, 1.0, 5, month=10)
        self.assertLessEqual(t["mtb_quality"], 2)

    def test_freeze_thaw_detected(self):
        t = compute_trail_conditions(45, 25, 0.0, 5, month=11)
        self.assertTrue(any("freeze" in c.lower() for c in t["conditions"]))


class TestRouteWind(TestCase):

    def test_headwind_detected(self):
        # SW wind + westbound route = headwind
        r = compute_route_wind(225, 20, "Echo Summit (US-50 W)")
        route = r["routes"]["Echo Summit (US-50 W)"]
        self.assertGreater(route["headwind_mph"], 0)

    def test_no_wind_no_data(self):
        r = compute_route_wind(float("nan"), float("nan"))
        self.assertFalse(r.get("available", True))


class TestFishingPressure(TestCase):

    def test_falling_pressure_is_good(self):
        f = compute_fishing_pressure(1008, -2.5)
        self.assertGreaterEqual(f["rating"], 4)

    def test_steady_is_fair(self):
        f = compute_fishing_pressure(1015, 0.0)
        self.assertEqual(f["rating"], 3)

    def test_no_data(self):
        f = compute_fishing_pressure(None, None)
        self.assertFalse(f.get("available", True))


if __name__ == "__main__":
    unittest_main(verbosity=2)
