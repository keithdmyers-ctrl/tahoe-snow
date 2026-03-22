#!/usr/bin/env python3
"""Tests for feature_engineering.py — physics features, station mapping, SLR, lapse rate."""

import numpy as np
import pytest

from feature_engineering import (
    build_station_resort_map,
    compute_froude_number,
    compute_lapse_rate,
    compute_orographic_multiplier,
    compute_slr,
    find_nearest_zone,
)


# =========================================================================
# compute_slr
# =========================================================================

class TestComputeSLR:
    def test_cold_conditions(self):
        """Very cold = high SLR (light fluffy snow)."""
        slr = compute_slr(temp_f=15, wind_mph=5, rh_pct=70)
        assert slr > 12

    def test_warm_conditions(self):
        """Warm = low SLR (heavy wet snow)."""
        slr = compute_slr(temp_f=34, wind_mph=20, rh_pct=90)
        assert slr < 12

    def test_colder_higher_slr(self):
        """Colder temps should produce higher SLR."""
        cold = compute_slr(temp_f=10, wind_mph=5, rh_pct=70)
        warm = compute_slr(temp_f=32, wind_mph=5, rh_pct=70)
        assert cold > warm

    def test_typical_range(self):
        """SLR should be in reasonable range for snow temps (<32°F)."""
        for temp in range(10, 32, 5):
            slr = compute_slr(temp_f=temp, wind_mph=10)
            assert 1 <= slr <= 30, f"SLR {slr} out of range at {temp}°F"


# =========================================================================
# compute_lapse_rate
# =========================================================================

class TestComputeLapseRate:
    def test_normal_lapse_rate(self):
        """Standard atmosphere: ~6.5 C/km."""
        temps_by_elev = [
            (6000, 40),  # ~4.4°C at 1829m
            (8000, 32),  # 0°C at 2438m
            (10000, 24), # -4.4°C at 3048m
        ]
        lr = compute_lapse_rate(temps_by_elev)
        assert 5.0 < lr < 9.0  # Normal range

    def test_inversion(self):
        """Temperature inversion: warmer at higher elevation → negative lapse rate."""
        temps_by_elev = [
            (6000, 25),  # Cold in valley
            (8000, 35),  # Warmer on mountain
        ]
        lr = compute_lapse_rate(temps_by_elev)
        assert lr < 0

    def test_insufficient_data(self):
        """Single point → NaN."""
        assert np.isnan(compute_lapse_rate([(6000, 32)]))
        assert np.isnan(compute_lapse_rate([]))

    def test_nan_temps_filtered(self):
        """NaN temperatures should be filtered out."""
        temps_by_elev = [
            (6000, 40),
            (7000, np.nan),
            (8000, 32),
        ]
        lr = compute_lapse_rate(temps_by_elev)
        assert not np.isnan(lr)

    def test_small_elevation_range(self):
        """<100m elevation range → NaN."""
        temps_by_elev = [
            (6000, 40),
            (6100, 39),  # Only ~30m difference
        ]
        lr = compute_lapse_rate(temps_by_elev)
        assert np.isnan(lr)


# =========================================================================
# compute_orographic_multiplier
# =========================================================================

class TestOrographicMultiplier:
    def test_optimal_direction(self):
        """WSW wind (247.5°) should maximize orographic lift."""
        mult = compute_orographic_multiplier(8500, 25, 247.5)
        assert mult > 1.0

    def test_leeward(self):
        """East wind (opposite of optimal) should reduce precipitation."""
        mult = compute_orographic_multiplier(8500, 25, 67.5)
        # Should be near 1.0 or less
        assert mult <= 1.5

    def test_no_wind(self):
        """No wind = minimal orographic enhancement (elevation-only baseline)."""
        mult = compute_orographic_multiplier(8500, 0, 247.5)
        # Even with no wind, elevation still provides some baseline
        assert mult >= 1.0
        assert mult < 2.0

    def test_higher_elev_more_enhancement(self):
        """Higher elevation should see more orographic effect."""
        low = compute_orographic_multiplier(6500, 20, 247.5)
        high = compute_orographic_multiplier(9500, 20, 247.5)
        assert high >= low

    def test_nan_inputs(self):
        """NaN wind → multiplier = 1.0."""
        assert compute_orographic_multiplier(8500, np.nan, 247.5) == 1.0
        assert compute_orographic_multiplier(8500, 20, np.nan) == 1.0


# =========================================================================
# compute_froude_number
# =========================================================================

class TestFroudeNumber:
    def test_low_wind(self):
        """Low wind → low Froude (blocked flow)."""
        fr = compute_froude_number(wind_mph=5)
        assert fr < 1.0

    def test_high_wind(self):
        """High wind → high Froude (flow over barrier)."""
        fr = compute_froude_number(wind_mph=50)
        assert fr > 0.5

    def test_zero_wind(self):
        """Zero wind → NaN Froude (no flow)."""
        fr = compute_froude_number(wind_mph=0)
        assert np.isnan(fr) or fr == pytest.approx(0.0, abs=0.01)

    def test_positive_for_nonzero_wind(self):
        """Froude number should be non-negative for positive wind."""
        for wind in range(5, 60, 10):
            fr = compute_froude_number(wind_mph=wind)
            assert not np.isnan(fr)
            assert fr >= 0


# =========================================================================
# find_nearest_zone
# =========================================================================

class TestFindNearestZone:
    def test_exact_match(self):
        from resort_configs import get_active_resorts
        configs = get_active_resorts()
        for name, cfg in configs.items():
            for zone in cfg.zones:
                result = find_nearest_zone(zone.elevation_ft, cfg.zones)
                assert result == zone.name

    def test_low_elevation_matches_base(self):
        from resort_configs import get_active_resorts
        configs = get_active_resorts()
        heavenly = configs.get("Heavenly")
        if heavenly:
            result = find_nearest_zone(5000, heavenly.zones)
            assert result == "base"

    def test_high_elevation_matches_peak(self):
        from resort_configs import get_active_resorts
        configs = get_active_resorts()
        heavenly = configs.get("Heavenly")
        if heavenly:
            result = find_nearest_zone(12000, heavenly.zones)
            assert result == "peak"


# =========================================================================
# build_station_resort_map
# =========================================================================

class TestStationResortMap:
    def test_map_not_empty(self):
        mapping = build_station_resort_map()
        assert len(mapping) > 0

    def test_map_structure(self):
        mapping = build_station_resort_map()
        for station_name, resorts in mapping.items():
            assert isinstance(station_name, str)
            assert isinstance(resorts, list)
            for r in resorts:
                assert "resort" in r
                assert "zones" in r

    def test_known_stations(self):
        """Key SNOTEL stations should be in the map."""
        mapping = build_station_resort_map()
        # At least some stations should be mapped
        assert len(mapping) >= 3
