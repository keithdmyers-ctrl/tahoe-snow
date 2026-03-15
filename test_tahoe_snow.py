#!/usr/bin/env python3
"""
Comprehensive test suite for Tahoe Snow Conditions Analyzer.

Tests cover:
  1. Snow physics calculations (SLR, wind chill, orographic, lapse rate)
  2. Data fetching (all APIs - live integration tests)
  3. Data parsing and analysis pipeline
  4. Output formatting
  5. Alerts system
  6. Edge cases and error handling
"""

import json
import sys
import os
import math
from datetime import datetime, timedelta, timezone
from unittest import TestCase, main as unittest_main

sys.path.insert(0, os.path.dirname(__file__))

from tahoe_snow import (
    compute_slr, wind_chill_f, orographic_multiplier, estimate_temp_c,
    snow_quality_str, wind_dir_str, precip_type, water_year_start,
    precip_phase_probability, terrain_adjusted_temperature,
    settled_snow_depth, lake_effect_enhancement,
    fetch_nws_observations, fetch_nws_forecast, fetch_open_meteo,
    fetch_snotel_current, fetch_snotel_history, fetch_snotel_season,
    fetch_avalanche, fetch_forecast_discussion,
    fetch_radar_nowcast, fetch_webcam_conditions,
    parse_open_meteo, aggregate_daily, multi_model_spread,
    generate_summary, analyze_all, format_report,
    RESORTS, SNOTEL_STATIONS, MODELS, MODEL_LABELS,
)


# ===================================================================
# 1. Snow Physics Unit Tests
# ===================================================================

class TestSnowPhysics(TestCase):
    """Test all snow physics calculations."""

    def test_slr_very_cold(self):
        """Very cold temps should produce high SLR (fluffy powder)."""
        slr = compute_slr(-20)
        self.assertGreaterEqual(slr, 20)
        self.assertLessEqual(slr, 25)

    def test_slr_cold(self):
        """Cold temps: classic powder SLR range."""
        slr = compute_slr(-15)
        self.assertGreaterEqual(slr, 15)
        self.assertLessEqual(slr, 20)

    def test_slr_moderate(self):
        """Moderate temps: good powder."""
        slr = compute_slr(-9)
        self.assertGreaterEqual(slr, 12)
        self.assertLessEqual(slr, 15)

    def test_slr_near_freezing(self):
        """Near freezing: packable snow."""
        slr = compute_slr(-3)
        self.assertGreaterEqual(slr, 8)
        self.assertLessEqual(slr, 12)

    def test_slr_just_below_zero(self):
        """Just below freezing: heavy wet snow."""
        slr = compute_slr(-0.5)
        self.assertGreaterEqual(slr, 5)
        self.assertLessEqual(slr, 8)

    def test_slr_above_freezing(self):
        """Above freezing: rain/slush."""
        slr = compute_slr(2)
        self.assertGreaterEqual(slr, 1)
        self.assertLess(slr, 5)

    def test_slr_monotonic_decrease(self):
        """SLR should generally decrease as temp increases."""
        temps = [-25, -18, -12, -6, -1, 0, 2]
        slrs = [compute_slr(t) for t in temps]
        for i in range(len(slrs) - 1):
            self.assertGreaterEqual(slrs[i], slrs[i+1],
                f"SLR should decrease: {temps[i]}C={slrs[i]} vs {temps[i+1]}C={slrs[i+1]}")

    def test_slr_never_negative(self):
        """SLR should never be negative or zero."""
        for t in range(-30, 20):
            self.assertGreater(compute_slr(t), 0, f"SLR negative at {t}C")

    def test_wind_chill_cold_windy(self):
        """Wind chill should be lower than actual temp in cold+wind."""
        wc = wind_chill_f(20, 25)
        self.assertLess(wc, 20)
        self.assertGreater(wc, -10)

    def test_wind_chill_warm(self):
        """Above 50F, wind chill = actual temp."""
        self.assertEqual(wind_chill_f(55, 30), 55)

    def test_wind_chill_calm(self):
        """Below 3mph, wind chill = actual temp."""
        self.assertEqual(wind_chill_f(20, 1), 20)

    def test_wind_chill_known_value(self):
        """NWS table: 0F + 15mph wind = -19F wind chill."""
        wc = wind_chill_f(0, 15)
        self.assertAlmostEqual(wc, -19, delta=2)

    def test_orographic_wsw_high(self):
        """WSW wind at high elevation = maximum enhancement."""
        oro = orographic_multiplier(9000, 30, 247.5)
        self.assertGreater(oro, 1.2)

    def test_orographic_east_low(self):
        """East wind at low elevation = minimal enhancement."""
        oro = orographic_multiplier(6500, 10, 90)
        self.assertLess(oro, 0.8)

    def test_orographic_always_positive(self):
        """Orographic factor should never be zero or negative."""
        for elev in range(5000, 11000, 500):
            for wdir in range(0, 360, 45):
                for wspd in [0, 10, 30, 60]:
                    self.assertGreater(orographic_multiplier(elev, wspd, wdir), 0)

    def test_lapse_rate(self):
        """Temperature should decrease with altitude."""
        base_c = 5.0
        base_m = 1900
        high_temp = estimate_temp_c(base_c, base_m, 3000)
        self.assertLess(high_temp, base_c)
        # ~5.5C per 1000m
        expected = base_c - (1100 * 5.5 / 1000)
        self.assertAlmostEqual(high_temp, expected, places=1)

    def test_snow_quality_labels(self):
        """Quality labels should match SLR ranges."""
        self.assertIn("cold smoke", snow_quality_str(20).lower())
        self.assertIn("powder", snow_quality_str(15).lower())
        self.assertIn("classic", snow_quality_str(12).lower())
        self.assertIn("packable", snow_quality_str(9).lower())
        self.assertIn("cement", snow_quality_str(6).lower())
        self.assertIn("wet", snow_quality_str(3).lower())

    def test_wind_direction_cardinal(self):
        """Wind direction strings should be correct."""
        self.assertEqual(wind_dir_str(0), "N")
        self.assertEqual(wind_dir_str(90), "E")
        self.assertEqual(wind_dir_str(180), "S")
        self.assertEqual(wind_dir_str(270), "W")

    def test_precip_type_logic(self):
        """Precip type should match temperature."""
        self.assertEqual(precip_type(-5, True), "Snow")
        self.assertEqual(precip_type(5, True), "Rain")
        self.assertEqual(precip_type(0, True), "Mix")
        self.assertEqual(precip_type(-10, False), "None")

    def test_water_year_start(self):
        """Water year start should be Oct 1."""
        ws = water_year_start()
        self.assertTrue(ws.endswith("-10-01"))


# ===================================================================
# 2. Configuration Validation
# ===================================================================

class TestConfiguration(TestCase):
    """Verify resort presets and station data are valid."""

    def test_all_resorts_have_zones(self):
        for name, resort in RESORTS.items():
            for zone in ("base", "mid", "peak"):
                self.assertIn(zone, resort, f"{name} missing {zone}")
                self.assertIn("lat", resort[zone])
                self.assertIn("lon", resort[zone])
                self.assertIn("elev_ft", resort[zone])
                self.assertIn("label", resort[zone])

    def test_resort_elevations_ordered(self):
        """Base < mid < peak for each resort."""
        for name, resort in RESORTS.items():
            self.assertLess(resort["base"]["elev_ft"], resort["mid"]["elev_ft"],
                            f"{name} base >= mid")
            self.assertLess(resort["mid"]["elev_ft"], resort["peak"]["elev_ft"],
                            f"{name} mid >= peak")

    def test_resort_coords_in_tahoe(self):
        """All coords should be in the Lake Tahoe area."""
        for name, resort in RESORTS.items():
            for zone in ("base", "mid", "peak"):
                lat = resort[zone]["lat"]
                lon = resort[zone]["lon"]
                self.assertGreater(lat, 38.5, f"{name}.{zone} lat too low")
                self.assertLess(lat, 39.5, f"{name}.{zone} lat too high")
                self.assertGreater(lon, -121, f"{name}.{zone} lon too low")
                self.assertLess(lon, -119.5, f"{name}.{zone} lon too high")

    def test_snotel_stations_valid(self):
        for name, st in SNOTEL_STATIONS.items():
            self.assertIn("id", st)
            self.assertIn("state", st)
            self.assertIn("elev_ft", st)
            self.assertIn(st["state"], ("CA", "NV"))

    def test_nearest_snotel_exist(self):
        """Each resort's nearest_snotel should reference real stations."""
        for name, resort in RESORTS.items():
            for sname in resort.get("nearest_snotel", []):
                self.assertIn(sname, SNOTEL_STATIONS,
                              f"{name} references unknown SNOTEL: {sname}")

    def test_model_config(self):
        self.assertEqual(len(MODELS), 4)  # GFS, ECMWF, ICON, HRRR
        for m in MODELS:
            self.assertIn(m, MODEL_LABELS)


# ===================================================================
# 3. Live API Integration Tests
# ===================================================================

class TestAPIs(TestCase):
    """Test all external API endpoints return valid data."""

    def test_nws_observations(self):
        """NWS current observations API should return temperature."""
        obs = fetch_nws_observations(39.17, -120.145)
        self.assertIsInstance(obs, dict)
        if obs:  # station may not always be available
            self.assertIn("temp_f", obs)
            self.assertIn("wind_mph", obs)
            self.assertIn("feels_like_f", obs)
            # Temperature sanity check (-40 to 120F for Tahoe)
            self.assertGreater(obs["temp_f"], -40)
            self.assertLess(obs["temp_f"], 120)

    def test_nws_forecast(self):
        """NWS forecast API should return periods and hourly data."""
        nws = fetch_nws_forecast(39.17, -120.145)
        self.assertIsInstance(nws, dict)
        self.assertIn("periods", nws)
        self.assertIn("hourly", nws)
        self.assertGreater(len(nws["periods"]), 0)
        self.assertGreater(len(nws["hourly"]), 0)

        # Check period structure
        p = nws["periods"][0]
        self.assertIn("temperature", p)
        self.assertIn("windSpeed", p)

    def test_open_meteo_multi_model(self):
        """Open-Meteo should return data for all three models."""
        om = fetch_open_meteo(39.17, -120.145)
        self.assertIsInstance(om, dict)
        self.assertNotIn("error", om)
        self.assertIn("hourly", om)
        self.assertIn("time", om["hourly"])

        # Verify all models present
        for model in MODELS:
            key = f"temperature_2m_{model}"
            self.assertIn(key, om["hourly"], f"Missing model data: {key}")
            self.assertGreater(len(om["hourly"][key]), 100,
                               f"Too few hours for {model}")

    def test_snotel_current(self):
        """SNOTEL should return data for most stations."""
        snotel = fetch_snotel_current()
        self.assertIsInstance(snotel, dict)
        self.assertGreater(len(snotel), 5, "Too few SNOTEL stations returned")

        # Stations should have at minimum name and elev_ft
        for name, data in snotel.items():
            if "error" not in data:
                self.assertIn("name", data)
                self.assertIn("elev_ft", data)

    def test_snotel_history(self):
        """SNOTEL 10-day history should return daily values."""
        hist = fetch_snotel_history("652", "NV", days=10)
        self.assertIsInstance(hist, dict)
        self.assertIn("SNWD", hist)
        self.assertGreater(len(hist["SNWD"]), 5, "Too few history days")

    def test_snotel_season(self):
        """SNOTEL season data should show peak values."""
        season = fetch_snotel_season("652", "NV")
        self.assertIsInstance(season, dict)
        if season:  # may not have data before Oct
            self.assertIn("SNWD", season)
            self.assertIn("peak", season["SNWD"])
            self.assertGreater(season["SNWD"]["peak"], 0)

    def test_avalanche(self):
        """Avalanche.org should return SAC danger rating."""
        avy = fetch_avalanche()
        self.assertIsInstance(avy, dict)
        if "error" not in avy:
            self.assertIn("danger_level", avy)
            self.assertIn("danger_label", avy)
            self.assertIn(avy["danger_level"], [0, 1, 2, 3, 4, 5])

    def test_forecast_discussion(self):
        """NWS AFD should return text content."""
        afd = fetch_forecast_discussion()
        self.assertIsInstance(afd, str)
        self.assertGreater(len(afd), 100, "AFD text too short")
        # Should contain typical AFD markers
        self.assertTrue(any(m in afd.upper() for m in [".DISCUSSION", ".SYNOPSIS",
                            "FORECAST", "NWS"]))


# ===================================================================
# 4. Data Processing Tests
# ===================================================================

class TestDataProcessing(TestCase):
    """Test data parsing, aggregation, and analysis."""

    def setUp(self):
        """Fetch real data once for processing tests."""
        self.om = fetch_open_meteo(39.17, -120.145)

    def test_parse_open_meteo_structure(self):
        """Parsed Open-Meteo data should have all models."""
        parsed = parse_open_meteo(self.om, 9800)
        self.assertNotIn("error", parsed)
        self.assertIn("models", parsed)
        for label in MODEL_LABELS.values():
            self.assertIn(label, parsed["models"])
            hours = parsed["models"][label]
            self.assertGreater(len(hours), 100)
            # Check hour structure
            h = hours[0]
            self.assertIn("temp_f", h)
            self.assertIn("feels_like_f", h)
            self.assertIn("snowfall_in", h)
            self.assertIn("slr", h)
            self.assertIn("wind_mph", h)
            self.assertIn("precip_type", h)

    def test_parse_elevation_adjustment(self):
        """Higher elevations should have lower temperatures."""
        low = parse_open_meteo(self.om, 6500)
        high = parse_open_meteo(self.om, 10000)
        if "error" not in low and "error" not in high:
            low_temp = low["models"]["GFS"][0]["temp_f"]
            high_temp = high["models"]["GFS"][0]["temp_f"]
            if low_temp is not None and high_temp is not None:
                self.assertGreater(low_temp, high_temp,
                                   "Higher elevation should be colder")

    def test_aggregate_daily(self):
        """Daily aggregation should produce day/night buckets."""
        parsed = parse_open_meteo(self.om, 9800)
        if "error" not in parsed:
            gfs = parsed["models"]["GFS"]
            buckets = aggregate_daily(gfs)
            self.assertGreater(len(buckets), 5)
            b = buckets[0]
            self.assertIn("date", b)
            self.assertIn("period", b)
            self.assertIn(b["period"], ("Day", "Night"))
            self.assertIn("snow_in", b)
            self.assertIn("temp_high_f", b)
            self.assertIn("feels_like_low_f", b)

    def test_multi_model_spread(self):
        """Model spread should compute agreement/confidence."""
        parsed = parse_open_meteo(self.om, 9800)
        if "error" not in parsed:
            spread = multi_model_spread(parsed)
            self.assertGreater(len(spread), 5)
            d = spread[0]
            self.assertIn("date", d)
            self.assertIn("models", d)
            self.assertIn("snow_spread", d)
            self.assertIn("confidence", d)
            self.assertIn(d["confidence"], ("High", "Medium", "Low"))

    def test_full_analysis_pipeline(self):
        """Full analysis should produce all expected sections."""
        obs = fetch_nws_observations(39.17, -120.145)
        nws = fetch_nws_forecast(39.17, -120.145)
        snotel = fetch_snotel_current()
        afd = fetch_forecast_discussion()
        avy = fetch_avalanche()

        analysis = analyze_all(obs, nws, self.om, snotel, afd, avy, {})

        self.assertIn("generated", analysis)
        self.assertIn("current_conditions", analysis)
        self.assertIn("resorts", analysis)
        self.assertIn("comparison", analysis)
        self.assertIn("snotel_current", analysis)
        self.assertIn("snotel_history", analysis)
        self.assertIn("season_stats", analysis)
        self.assertIn("avalanche", analysis)
        self.assertIn("summary", analysis)

        # Verify all resorts present
        for rn in RESORTS:
            self.assertIn(rn, analysis["resorts"])
            rd = analysis["resorts"][rn]
            self.assertIn("zones", rd)
            for zk in ("base", "mid", "peak"):
                self.assertIn(zk, rd["zones"])

        # Comparison
        comp = analysis["comparison"]
        self.assertIn("resorts", comp)
        self.assertIn("rankings", comp)

        # Summary should be non-empty
        self.assertGreater(len(analysis["summary"]), 50)

    def test_format_report(self):
        """Report formatting should produce valid output."""
        obs = fetch_nws_observations(39.17, -120.145)
        nws = fetch_nws_forecast(39.17, -120.145)
        snotel = fetch_snotel_current()
        afd = fetch_forecast_discussion()
        avy = fetch_avalanche()
        analysis = analyze_all(obs, nws, self.om, snotel, afd, avy, {})

        report = format_report(analysis)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 1000)

        # Should contain key sections
        self.assertIn("CURRENT CONDITIONS", report)
        self.assertIn("HEAVENLY", report)
        self.assertIn("NORTHSTAR", report)
        self.assertIn("KIRKWOOD", report)
        self.assertIn("MOUNTAIN COMPARISON", report)
        self.assertIn("SNOTEL", report)

        # Compact mode
        compact = format_report(analysis, compact=True)
        self.assertLess(len(compact), len(report),
                        "Compact report should be shorter")

    def test_json_output(self):
        """Analysis should be JSON-serializable."""
        obs = fetch_nws_observations(39.17, -120.145)
        nws = fetch_nws_forecast(39.17, -120.145)
        snotel = fetch_snotel_current()
        analysis = analyze_all(obs, nws, self.om, snotel, "", {}, {})

        # Should not raise
        json_str = json.dumps(analysis, default=str)
        self.assertGreater(len(json_str), 500)

        # Should round-trip
        parsed = json.loads(json_str)
        self.assertIn("resorts", parsed)


# ===================================================================
# 5. New Physics Unit Tests (Tier 4)
# ===================================================================

class TestPrecipPhaseProbability(TestCase):
    """Test probabilistic precipitation phase classification."""

    def test_very_cold_is_snow(self):
        """Well below freezing should be nearly certain snow."""
        result = precip_phase_probability(-5.0, 2500, 2000)
        self.assertGreater(result["p_snow"], 0.90)
        self.assertLess(result["p_rain"], 0.05)
        self.assertEqual(result["dominant_type"], "Snow")

    def test_warm_is_rain(self):
        """Well above freezing should be nearly certain rain."""
        result = precip_phase_probability(5.0, 1500, 2500)
        self.assertGreater(result["p_rain"], 0.90)
        self.assertLess(result["p_snow"], 0.05)
        self.assertEqual(result["dominant_type"], "Rain")

    def test_transition_zone_has_mix(self):
        """Near 0.5C wet-bulb should have significant mix probability."""
        result = precip_phase_probability(0.5, 2000, 2000)
        self.assertGreater(result["p_mix"], 0.01)
        # Should sum to 1.0
        total = result["p_snow"] + result["p_mix"] + result["p_rain"]
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_probabilities_sum_to_one(self):
        """Probabilities should always sum to 1.0."""
        for wb in [-10, -5, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 5, 10]:
            for elev in [1500, 2000, 2500, 3000]:
                result = precip_phase_probability(wb, elev, 2000)
                total = result["p_snow"] + result["p_mix"] + result["p_rain"]
                self.assertAlmostEqual(total, 1.0, places=2,
                    msg=f"Sum != 1.0 at wb={wb}, elev={elev}: {result}")

    def test_elevation_boost_snow(self):
        """Being well above freezing level should boost p_snow."""
        # Near transition: 1.0C wet-bulb
        low = precip_phase_probability(1.0, 1500, 2000)  # below freezing level
        high = precip_phase_probability(1.0, 2500, 2000)  # above freezing level
        self.assertGreater(high["p_snow"], low["p_snow"])

    def test_no_freezing_level(self):
        """Should work with None freezing level."""
        result = precip_phase_probability(-3.0, 2500, None)
        self.assertGreater(result["p_snow"], 0.90)

    def test_dominant_type_matches(self):
        """dominant_type should match the highest probability."""
        result = precip_phase_probability(-5.0, 2500, 2000)
        probs = {"Snow": result["p_snow"], "Mix": result["p_mix"], "Rain": result["p_rain"]}
        expected = max(probs, key=probs.get)
        self.assertEqual(result["dominant_type"], expected)


class TestTerrainAdjustedTemperature(TestCase):
    """Test aspect-based temperature correction."""

    def test_south_facing_peak_solar(self):
        """South-facing slope at peak solar should be warmer."""
        base = 0.0
        adj = terrain_adjusted_temperature(base, 2500, 180, 13, cloud_cover_pct=0)
        self.assertGreater(adj, base)
        self.assertAlmostEqual(adj, base + 2.0, places=1)

    def test_north_facing_daytime(self):
        """North-facing slope during day should be cooler."""
        base = 0.0
        adj = terrain_adjusted_temperature(base, 2500, 0, 12, cloud_cover_pct=0)
        self.assertLess(adj, base)

    def test_clouds_eliminate_aspect_effect(self):
        """Full cloud cover should eliminate aspect correction."""
        base = 0.0
        adj = terrain_adjusted_temperature(base, 2500, 180, 13, cloud_cover_pct=100)
        self.assertAlmostEqual(adj, base, places=5)

    def test_east_morning_warm(self):
        """East-facing slope in morning should be warmer."""
        base = 0.0
        adj = terrain_adjusted_temperature(base, 2500, 90, 9, cloud_cover_pct=0)
        self.assertGreater(adj, base)

    def test_west_afternoon_warm(self):
        """West-facing slope in afternoon should be warmer."""
        base = 0.0
        adj = terrain_adjusted_temperature(base, 2500, 270, 14, cloud_cover_pct=0)
        self.assertGreater(adj, base)

    def test_night_south_facing_cooler(self):
        """South-facing slope at night should be slightly cooler."""
        base = 0.0
        adj = terrain_adjusted_temperature(base, 2500, 180, 2, cloud_cover_pct=0)
        self.assertLess(adj, base)

    def test_partial_clouds_scale_linearly(self):
        """50% clouds should produce half the aspect correction."""
        base = 0.0
        clear = terrain_adjusted_temperature(base, 2500, 180, 13, cloud_cover_pct=0)
        half = terrain_adjusted_temperature(base, 2500, 180, 13, cloud_cover_pct=50)
        full_effect = clear - base
        half_effect = half - base
        self.assertAlmostEqual(half_effect, full_effect * 0.5, places=3)


class TestSnowSettling(TestCase):
    """Test Kojima (1967) snow settling model."""

    def test_no_settling_at_time_zero(self):
        """No settling should occur at t=0."""
        result = settled_snow_depth(12.0, 0, 20, 10)
        self.assertEqual(result, 12.0)

    def test_settling_reduces_depth(self):
        """Depth should decrease over time."""
        fresh = 12.0
        after_12h = settled_snow_depth(fresh, 12, 25, 5)
        self.assertLess(after_12h, fresh)
        self.assertGreater(after_12h, 0)

    def test_warm_settles_faster(self):
        """Warmer temperatures should produce more settling."""
        cold = settled_snow_depth(12.0, 12, 10, 5)
        warm = settled_snow_depth(12.0, 12, 35, 5)
        self.assertGreater(cold, warm)

    def test_wind_settles_faster(self):
        """Higher wind should produce more settling."""
        calm = settled_snow_depth(12.0, 12, 20, 0)
        windy = settled_snow_depth(12.0, 12, 20, 25)
        self.assertGreater(calm, windy)

    def test_minimum_60_percent(self):
        """Should never settle below 60% of original."""
        result = settled_snow_depth(10.0, 100, 40, 30)
        self.assertGreaterEqual(result, 6.0)

    def test_zero_snow(self):
        """Zero snowfall should return zero."""
        result = settled_snow_depth(0, 12, 25, 10)
        self.assertEqual(result, 0)

    def test_negative_input(self):
        """Negative snowfall should return the input unchanged."""
        result = settled_snow_depth(-1.0, 12, 25, 10)
        self.assertEqual(result, -1.0)


class TestLakeEffectEnhancement(TestCase):
    """Test Lake Tahoe lake-effect snowfall parameterization."""

    def test_no_effect_warm_air(self):
        """Warm air (small lake-air diff) should produce no enhancement."""
        factor = lake_effect_enhancement(345, 20, 0.0, lake_temp_c=4.5)
        self.assertAlmostEqual(factor, 1.0, places=1)

    def test_enhancement_cold_nw(self):
        """Cold NNW wind should produce enhancement on east shore."""
        factor = lake_effect_enhancement(345, 20, -15.0, lake_temp_c=4.5)
        self.assertGreater(factor, 1.05)
        self.assertLessEqual(factor, 1.3)

    def test_no_effect_east_wind(self):
        """East wind should produce minimal enhancement (wrong fetch)."""
        factor = lake_effect_enhancement(90, 20, -15.0, lake_temp_c=4.5)
        self.assertLess(factor, 1.05)

    def test_no_effect_calm(self):
        """Very calm winds should produce minimal enhancement."""
        factor = lake_effect_enhancement(345, 0, -15.0, lake_temp_c=4.5)
        self.assertLess(factor, 1.05)

    def test_always_gte_one(self):
        """Enhancement factor should never be less than 1.0."""
        for wd in range(0, 360, 30):
            for ws in [0, 5, 15, 25, 40]:
                for tc in [-20, -10, 0, 5]:
                    factor = lake_effect_enhancement(wd, ws, tc)
                    self.assertGreaterEqual(factor, 1.0,
                        f"Factor < 1.0 at wd={wd}, ws={ws}, tc={tc}")

    def test_optimal_conditions(self):
        """Optimal conditions (NNW wind, 20mph, -15C) should give max effect."""
        factor = lake_effect_enhancement(345, 20, -15.0, lake_temp_c=4.5)
        # This should be close to the theoretical maximum
        self.assertGreater(factor, 1.15)

    def test_direction_symmetry(self):
        """345 and 350 degrees should produce similar enhancement."""
        f1 = lake_effect_enhancement(345, 20, -15.0)
        f2 = lake_effect_enhancement(350, 20, -15.0)
        self.assertAlmostEqual(f1, f2, places=1)


class TestResortConfig(TestCase):
    """Test that resort configuration includes new fields."""

    def test_all_zones_have_aspect_deg(self):
        """All resort zones should have aspect_deg field."""
        for rn, resort in RESORTS.items():
            for zk in ("base", "mid", "peak"):
                self.assertIn("aspect_deg", resort[zk],
                    f"{rn}.{zk} missing aspect_deg")
                self.assertIsInstance(resort[zk]["aspect_deg"], (int, float))
                self.assertGreaterEqual(resort[zk]["aspect_deg"], 0)
                self.assertLess(resort[zk]["aspect_deg"], 360)

    def test_all_resorts_have_east_shore(self):
        """All resorts should have east_shore flag."""
        for rn, resort in RESORTS.items():
            self.assertIn("east_shore", resort,
                f"{rn} missing east_shore flag")

    def test_heavenly_is_east_shore(self):
        """Heavenly should be flagged as east shore."""
        self.assertTrue(RESORTS["Heavenly"]["east_shore"])

    def test_kirkwood_not_east_shore(self):
        """Kirkwood should not be flagged as east shore."""
        self.assertFalse(RESORTS["Kirkwood"]["east_shore"])


class TestWebcamStub(TestCase):
    """Test webcam conditions placeholder."""

    def test_returns_list(self):
        result = fetch_webcam_conditions()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_fields_are_none(self):
        """All condition fields should be None (placeholder)."""
        for cam in fetch_webcam_conditions():
            self.assertIsNone(cam["conditions"])
            self.assertIsNone(cam["visibility"])
            self.assertIsNone(cam["snowing"])
            self.assertIsNotNone(cam["station"])
            self.assertIsNotNone(cam["url"])


# ===================================================================
# 6. Alerts System Tests
# ===================================================================

class TestAlerts(TestCase):
    """Test the powder alert system."""

    def test_alerts_import(self):
        """Alert module should import without errors."""
        from alerts import load_config, load_state, check_cooldown, DEFAULT_CONFIG
        self.assertIn("thresholds", DEFAULT_CONFIG)

    def test_alerts_config_structure(self):
        from alerts import DEFAULT_CONFIG
        self.assertIn("enabled", DEFAULT_CONFIG)
        self.assertIn("thresholds", DEFAULT_CONFIG)
        self.assertIn("notifications", DEFAULT_CONFIG)
        self.assertIn("snow_24h_inches", DEFAULT_CONFIG["thresholds"])
        self.assertIn("snow_48h_inches", DEFAULT_CONFIG["thresholds"])

    def test_cooldown_logic(self):
        from alerts import check_cooldown
        now = datetime.now(timezone.utc)
        state = {"last_alerts": {
            "recent": (now - timedelta(hours=1)).isoformat(),
            "old": (now - timedelta(hours=12)).isoformat(),
        }}
        # Recent alert should be in cooldown
        self.assertTrue(check_cooldown(state, "recent", 6))
        # Old alert should not be
        self.assertFalse(check_cooldown(state, "old", 6))
        # Unknown alert should not be
        self.assertFalse(check_cooldown(state, "never_sent", 6))


# ===================================================================
# 7. Edge Cases
# ===================================================================

class TestEdgeCases(TestCase):
    """Test error handling and edge cases."""

    def test_parse_empty_open_meteo(self):
        """Should handle missing Open-Meteo data gracefully."""
        result = parse_open_meteo({}, 9800)
        self.assertIn("error", result)

    def test_parse_error_open_meteo(self):
        result = parse_open_meteo({"error": "test"}, 9800)
        self.assertIn("error", result)

    def test_aggregate_empty(self):
        self.assertEqual(aggregate_daily([]), [])

    def test_extreme_temperatures(self):
        """SLR should handle extreme temps without errors."""
        for t in [-50, -40, -30, 0, 10, 20, 30]:
            slr = compute_slr(t)
            self.assertGreater(slr, 0)

    def test_wind_chill_extremes(self):
        wc = wind_chill_f(-30, 60)
        self.assertLess(wc, -50)
        self.assertGreater(wc, -100)

    def test_wind_dir_edge(self):
        """Wind direction at boundary values."""
        self.assertEqual(wind_dir_str(360), "N")
        self.assertEqual(wind_dir_str(359), "N")
        self.assertEqual(wind_dir_str(0), "N")

    def test_summary_with_minimal_data(self):
        """Summary should not crash with minimal data."""
        minimal = {
            "current_conditions": {"observation": {}, "lake_level_temp_f": 30, "snow_level_ft": 7000},
            "comparison": {"resorts": {}},
            "multi_model_spread_peak": [],
            "snotel_current": {},
            "avalanche": {},
            "season_stats": {},
        }
        summary = generate_summary(minimal)
        self.assertIsInstance(summary, str)


if __name__ == "__main__":
    print("=" * 70)
    print("  TAHOE SNOW ANALYZER — COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()
    unittest_main(verbosity=2)
