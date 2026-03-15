#!/usr/bin/env python3
"""
Daily forecast verification cron job.

Fetches current observations and SNOTEL data, then logs verification
entries comparing yesterday's forecasts against today's actuals.

Designed to run via cron at 6 AM Pacific:

    0 6 * * * cd /home/keith/projects/tahoe-snow && python3 verify_cron.py >> /var/log/tahoe-verify.log 2>&1

Use --dry-run to preview what would be logged without writing to disk.
"""

import argparse
import json
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tahoe_snow import (
    fetch_nws_observations, fetch_nws_forecast, fetch_nws_gridpoints,
    fetch_open_meteo_multi,
    fetch_snotel_current, fetch_snotel_history,
    fetch_avalanche, fetch_forecast_discussion,
    analyze_all, RESORTS, SNOTEL_STATIONS,
)
from forecast_verification import (
    log_daily_verification, log_snow_verification,
    log_elevation_verification, get_verification_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Daily forecast verification — compare forecasts to observations"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be logged without writing files"
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    print(f"[{now.isoformat()}] Starting daily verification run")

    # --- Fetch current observations ---
    print("  Fetching NWS observations...")
    home_obs = fetch_nws_observations(39.17, -120.145)
    if not home_obs:
        print("  WARNING: No NWS observations available")
        home_obs = {}

    print("  Fetching NWS forecast...")
    home_fc = fetch_nws_forecast(39.17, -120.145)
    if not home_fc:
        print("  WARNING: No NWS forecast available")
        home_fc = {}

    # --- Fetch SNOTEL and build analysis ---
    print("  Fetching SNOTEL data...")
    snotel = fetch_snotel_current()
    snotel_ok = sum(1 for s in snotel.values() if "error" not in s)
    print(f"  Got {snotel_ok}/{len(snotel)} SNOTEL stations")

    print("  Fetching Open-Meteo multi-resort...")
    resort_points = {
        rn: {"lat": r["base"]["lat"], "lon": r["base"]["lon"]}
        for rn, r in RESORTS.items()
    }
    om = fetch_open_meteo_multi(resort_points)

    print("  Fetching supplemental data...")
    avy = fetch_avalanche()
    afd = fetch_forecast_discussion()
    nws_grids = fetch_nws_gridpoints(39.17, -120.145)

    print("  Running analyze_all...")
    analysis = analyze_all(
        home_obs, home_fc, om, snotel, afd, avy, {},
        nws_grids=nws_grids if "error" not in nws_grids else None,
    )

    if args.dry_run:
        print("\n  DRY RUN — would log the following:")
        print(f"  - Observation temp: {home_obs.get('temp_f', 'N/A')}F")
        print(f"  - SNOTEL stations: {snotel_ok}")
        snotel_hist = analysis.get("snotel_history", {})
        for name, data in snotel.items():
            if "error" in data:
                continue
            depth = data.get("snow_depth_in")
            hist = snotel_hist.get(name, {})
            snwd = hist.get("SNWD", [])
            prev = snwd[-2][1] if len(snwd) >= 2 else None
            change = f"{depth - prev:+.1f}" if (depth is not None and prev is not None) else "N/A"
            print(f"    {name}: {depth}\" (change: {change}\")")
        print("\n  Verification would be logged but --dry-run is active")
        return

    # --- Log verifications ---
    print("\n  Logging daily verification...")
    try:
        # Handle the known bug: log_daily_verification may reference
        # undefined 'hourly' variable. Wrap defensively.
        log_daily_verification(home_obs, home_fc, analysis)
        print("  Daily verification logged OK")
    except NameError as e:
        # Known bug: 'hourly' variable not defined in log_daily_verification
        print(f"  WARNING: Daily verification partial failure: {e}")
        print("  (This is a known bug with the 'hourly' variable)")
    except Exception as e:
        print(f"  ERROR logging daily verification: {e}")

    print("  Logging snow verification (SNOTEL)...")
    try:
        snow_result = log_snow_verification(analysis)
        print(f"  Snow verification: {snow_result.get('stations_logged', 0)} stations logged")
    except Exception as e:
        print(f"  ERROR logging snow verification: {e}")

    print("  Logging elevation verification...")
    try:
        elev_result = log_elevation_verification(analysis)
        bands = elev_result.get('bands_updated', [])
        print(f"  Elevation verification: {len(bands)} bands updated ({', '.join(bands)})")
    except Exception as e:
        print(f"  ERROR logging elevation verification: {e}")

    # --- Print summary ---
    print("\n  Current verification summary:")
    try:
        summary = get_verification_summary()
        print(f"    Days of data: {summary.get('days', 0)}")
        if summary.get('skill_score') is not None:
            print(f"    Skill score: {summary['skill_score']}")
        if summary.get('best_model'):
            print(f"    Best model: {summary['best_model']}")
        if summary.get('worst_model'):
            print(f"    Worst model: {summary['worst_model']}")
        weights = summary.get('model_weights', {})
        if weights:
            weight_str = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(weights.items(), key=lambda x: -x[1]))
            print(f"    Model weights: {weight_str}")
        snow = summary.get('snow_stats', {})
        if snow:
            print(f"    Snow MAE: {snow.get('mae', 'N/A')}\" ({snow.get('n', 0)} obs)")
    except Exception as e:
        print(f"    ERROR getting summary: {e}")

    elapsed = (datetime.now(timezone.utc) - now).total_seconds()
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Verification complete ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
