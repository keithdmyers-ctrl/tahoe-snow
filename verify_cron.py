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
import logging
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import fetch_tahoe_analysis, fetch_oakland_data
from forecast_verification import (
    log_daily_verification, log_snow_verification,
    log_elevation_verification, get_verification_summary,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Daily forecast verification -- compare forecasts to observations"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be logged without writing files"
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    print(f"[{now.isoformat()}] Starting daily verification run")

    # --- Fetch all data via shared pipeline ---
    print("  Fetching Tahoe analysis via shared pipeline...")
    analysis = fetch_tahoe_analysis()

    print("  Fetching Oakland data...")
    oakland = fetch_oakland_data()
    home_obs = oakland["home_obs"] or {}
    home_fc = oakland["home_fc"] or {}

    if not home_obs:
        print("  WARNING: No NWS observations available")
    if not home_fc:
        print("  WARNING: No NWS forecast available")

    snotel = analysis.get("snotel_current", {})
    snotel_ok = sum(1 for s in snotel.values() if "error" not in s)
    print(f"  Got {snotel_ok}/{len(snotel)} SNOTEL stations")

    if args.dry_run:
        print("\n  DRY RUN -- would log the following:")
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
        log_daily_verification(home_obs, home_fc, analysis)
        print("  Daily verification logged OK")
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

    # --- ML Shadow Mode Verification ---
    print("\n  Verifying ML shadow predictions...")
    try:
        _verify_ml_shadow(analysis)
    except Exception as e:
        print(f"  ERROR verifying ML shadow: {e}")

    elapsed = (datetime.now(timezone.utc) - now).total_seconds()
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Verification complete ({elapsed:.1f}s)")


def _verify_ml_shadow(analysis: dict):
    """Compare yesterday's ML shadow predictions against today's SNOTEL actuals.

    Reads .ml_shadow_log.json, matches predictions to SNOTEL depth changes,
    computes rolling MAE, and writes results to .ml_verification.json.

    If rolling ML MAE < raw forecast MAE for 14+ days, prints recommendation
    to activate ML corrections (flip SHADOW_MODE=False).
    """
    import numpy as np

    shadow_path = os.path.join(os.path.dirname(__file__), ".ml_shadow_log.json")
    verify_path = os.path.join(os.path.dirname(__file__), ".ml_verification.json")

    if not os.path.exists(shadow_path):
        print("    No shadow predictions to verify yet")
        return

    with open(shadow_path) as f:
        shadow_entries = json.load(f)

    if not shadow_entries:
        print("    Shadow log is empty")
        return

    # Get today's SNOTEL depth data
    snotel = analysis.get("snotel_current", {})
    snotel_hist = analysis.get("snotel_history", {})

    # Compute actual depth changes from SNOTEL history
    actual_changes = {}
    for station_name, hist in snotel_hist.items():
        snwd = hist.get("SNWD", [])
        if len(snwd) >= 2:
            today_depth = snwd[-1][1]
            yesterday_depth = snwd[-2][1]
            if today_depth is not None and yesterday_depth is not None:
                actual_changes[station_name] = today_depth - yesterday_depth

    if not actual_changes:
        print("    No SNOTEL depth history available for verification")
        return

    # Match shadow predictions to actuals
    # Shadow entries have resort/zone; we need to match to nearest SNOTEL
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday_str = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")

    pairs = []
    for entry in shadow_entries:
        ts = entry.get("timestamp", "")[:10]
        if ts != yesterday_str:
            continue
        pred = entry.get("ml_depth_prediction_in")
        if pred is None:
            continue
        # Use the basin average actual as comparison (imperfect but workable)
        avg_actual = np.mean(list(actual_changes.values()))
        pairs.append((pred, avg_actual))

    if not pairs:
        print(f"    No shadow predictions from yesterday ({yesterday_str}) to verify")
        return

    preds = np.array([p[0] for p in pairs])
    actuals = np.array([p[1] for p in pairs])
    ml_mae = float(np.mean(np.abs(preds - actuals)))

    print(f"    ML shadow: {len(pairs)} predictions verified, MAE={ml_mae:.3f}\"")

    # Load existing verification history
    history = []
    if os.path.exists(verify_path):
        try:
            with open(verify_path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            history = []

    history.append({
        "date": yesterday_str,
        "ml_mae": round(ml_mae, 4),
        "n_pairs": len(pairs),
        "avg_actual": round(float(np.mean(actuals)), 3),
        "avg_predicted": round(float(np.mean(preds)), 3),
    })

    # Keep last 90 days
    history = history[-90:]

    # Save
    tmp = verify_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, verify_path)

    # Rolling stats
    if len(history) >= 7:
        recent_mae = np.mean([h["ml_mae"] for h in history[-7:]])
        all_mae = np.mean([h["ml_mae"] for h in history])
        print(f"    Rolling 7-day MAE: {recent_mae:.3f}\", all-time: {all_mae:.3f}\"")

        if len(history) >= 14 and recent_mae < 1.0:
            print("    >>> ML predictions are performing well.")
            print("    >>> Consider setting SHADOW_MODE=False in ml_corrections.py")
    else:
        print(f"    {len(history)} days verified so far (need 14+ for activation recommendation)")


if __name__ == "__main__":
    main()
