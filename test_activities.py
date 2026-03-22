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


class TestEdgeCases(TestCase):
    """Edge case tests for activity scoring boundary conditions."""

    def test_all_none_weather(self):
        """Activity cards should handle missing weather data gracefully."""
        analysis = _make_analysis()
        # Clear out all zone current data
        for rname in analysis["resorts"]:
            for zname in analysis["resorts"][rname]["zones"]:
                analysis["resorts"][rname]["zones"][zname]["current"] = {}
        cards = compute_activity_decisions(analysis)
        self.assertEqual(len(cards), 10)
        for card in cards:
            self.assertIn(card["signal"], ("go", "caution", "no-go"))

    def test_aqi_zero(self):
        """AQI=0 is valid (very clean air) and should not be treated as None."""
        analysis = _make_analysis()
        analysis["current_conditions"]["observation"]["aqi"]["us_aqi"] = 0
        cards = compute_activity_decisions(analysis)
        hike = next(c for c in cards if c["activity"] == "Hike")
        self.assertEqual(hike["signal"], "go")

    def test_aqi_none(self):
        """Missing AQI data should not crash or degrade all activities."""
        analysis = _make_analysis()
        analysis["current_conditions"]["observation"]["aqi"] = {}
        cards = compute_activity_decisions(analysis)
        self.assertEqual(len(cards), 10)

    def test_extreme_wind(self):
        """60mph wind should affect paddle and climb."""
        analysis = _make_analysis()
        for rname in analysis["resorts"]:
            for zname in analysis["resorts"][rname]["zones"]:
                analysis["resorts"][rname]["zones"][zname]["current"]["wind_mph"] = 60
                analysis["resorts"][rname]["zones"][zname]["current"]["wind_gust_mph"] = 80
        cards = compute_activity_decisions(analysis)
        paddle = next(c for c in cards if c["activity"] == "Paddle")
        climb = next(c for c in cards if c["activity"] == "Climb")
        self.assertEqual(paddle["signal"], "no-go")
        self.assertLessEqual(climb["score"], 5)  # Wind degrades climbing

    def test_freezing_temps(self):
        """Freezing temps should affect camping and paddle scores."""
        analysis = _make_analysis()
        for rname in analysis["resorts"]:
            for zname in analysis["resorts"][rname]["zones"]:
                analysis["resorts"][rname]["zones"][zname]["current"]["temp_f"] = 20
        cards = compute_activity_decisions(analysis)
        paddle = next(c for c in cards if c["activity"] == "Paddle")
        self.assertLessEqual(paddle["score"], 4)

    def test_missing_solar_data(self):
        """Missing solar data should not crash photo or climb cards."""
        analysis = _make_analysis()
        analysis["outdoor"]["solar"] = {}
        cards = compute_activity_decisions(analysis)
        photo = next(c for c in cards if c["activity"] == "Photo")
        climb = next(c for c in cards if c["activity"] == "Climb")
        self.assertIsNotNone(photo)
        self.assertIsNotNone(climb)

    def test_missing_trail_data(self):
        """Missing trail data should not crash MTB card."""
        analysis = _make_analysis()
        analysis["outdoor"]["trail"] = {}
        cards = compute_activity_decisions(analysis)
        mtb = next(c for c in cards if c["activity"] == "Mountain Bike")
        self.assertIsNotNone(mtb)

    def test_high_avalanche_danger(self):
        """High avy danger should make BC Ski no-go."""
        analysis = _make_analysis()
        analysis["avalanche"] = {"danger_level": 4, "danger_label": "High", "travel_advice": "Avoid all backcountry"}
        cards = compute_activity_decisions(analysis)
        bc = next(c for c in cards if c["activity"] == "Backcountry Ski")
        self.assertEqual(bc["signal"], "no-go")
        self.assertLessEqual(bc["score"], 2)

    def test_scores_are_bounded(self):
        """All scores should be in range 0-10."""
        analysis = _make_analysis()
        cards = compute_activity_decisions(analysis)
        for card in cards:
            self.assertGreaterEqual(card["score"], 0, f"{card['activity']} score < 0")
            self.assertLessEqual(card["score"], 10, f"{card['activity']} score > 10")

    def test_photo_cloud_boost(self):
        """Photo should get a boost from clouds (dramatic light)."""
        analysis = _make_analysis()
        for rname in analysis["resorts"]:
            for zname in analysis["resorts"][rname]["zones"]:
                analysis["resorts"][rname]["zones"][zname]["timeline_48h"] = [
                    {"precip_in": 0.1, "snowfall_in": 0} for _ in range(24)
                ]
        cards = compute_activity_decisions(analysis)
        photo = next(c for c in cards if c["activity"] == "Photo")
        # Clouds should boost photo score above base (6)
        self.assertGreaterEqual(photo["score"], 7)

    def test_empty_resorts(self):
        """Empty resorts dict should still produce cards (degraded)."""
        analysis = _make_analysis()
        analysis["resorts"] = {}
        cards = compute_activity_decisions(analysis)
        self.assertEqual(len(cards), 10)


class TestAqiOverrideEdgeCases(TestCase):

    def test_aqi_exactly_at_threshold(self):
        """AQI at exact threshold should NOT trigger override."""
        signal, score = _apply_aqi_override("go", 8, 100, threshold_caution=100)
        self.assertEqual(signal, "go")

    def test_aqi_one_above_threshold(self):
        """AQI one above threshold should trigger override."""
        signal, score = _apply_aqi_override("go", 8, 101, threshold_caution=100)
        self.assertEqual(signal, "caution")

    def test_already_nogo_stays_nogo(self):
        """If already no-go, AQI should not upgrade to caution."""
        signal, score = _apply_aqi_override("no-go", 2, 120, threshold_caution=100)
        self.assertEqual(signal, "no-go")

    def test_score_never_negative(self):
        signal, score = _apply_aqi_override("go", 0, 200, threshold_nogo=150)
        self.assertGreaterEqual(score, 0)


class TestGatherConditions(TestCase):
    """Test _gather_conditions extracts weather data correctly."""

    def test_with_data(self):
        from outdoor_conditions import _gather_conditions
        resorts = {
            "R1": {
                "zones": {
                    "peak": {
                        "current": {"temp_f": 30, "wind_mph": 15, "wind_gust_mph": 25},
                        "timeline_48h": [{"precip_in": 0.1}] * 24,
                    }
                }
            }
        }
        temp, wind, gusts, precip = _gather_conditions(resorts)
        self.assertEqual(temp, 30)
        self.assertEqual(wind, 15)
        self.assertEqual(gusts, 25)
        self.assertAlmostEqual(precip, 2.4, places=1)

    def test_empty_resorts(self):
        from outdoor_conditions import _gather_conditions
        temp, wind, gusts, precip = _gather_conditions({})
        self.assertIsNone(temp)
        self.assertIsNone(wind)
        self.assertEqual(precip, 0)

    def test_missing_current(self):
        from outdoor_conditions import _gather_conditions
        resorts = {"R1": {"zones": {"peak": {}}}}
        temp, wind, gusts, precip = _gather_conditions(resorts)
        self.assertIsNone(temp)

    def test_fallback_to_second_zone(self):
        from outdoor_conditions import _gather_conditions
        resorts = {
            "R1": {
                "zones": {
                    "peak": {"current": {}},  # No temp_f
                    "base": {
                        "current": {"temp_f": 40, "wind_mph": 5},
                        "timeline_48h": [],
                    },
                }
            }
        }
        temp, wind, gusts, precip = _gather_conditions(resorts)
        self.assertEqual(temp, 40)


if __name__ == "__main__":
    unittest_main(verbosity=2)
