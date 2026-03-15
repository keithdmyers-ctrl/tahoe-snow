"""
Shared data pipeline for Tahoe Snow.

Consolidates data fetching and analysis orchestration used by:
- webapp.py (web dashboard)
- eink_scenes.py (e-ink display)
- alerts.py (powder alerts)
- verify_cron.py (daily verification)

This is the single source of truth for fetch coordinates, data source
wiring, and the analyze_all() call.  Individual consumers add their own
caching, locking, and presentation layers on top.
"""

import logging
from tahoe_snow import (
    fetch_nws_observations, fetch_nws_forecast, fetch_nws_gridpoints,
    fetch_open_meteo_multi, fetch_open_meteo,
    fetch_nbm, fetch_pws_nearby, aggregate_pws,
    fetch_synoptic_stations, fetch_cssl_snow,
    fetch_nws_alerts, fetch_sounding, fetch_climate_normals,
    fetch_ensemble, fetch_caltrans_chains, fetch_all_lift_status,
    fetch_snotel_current, fetch_snotel_history, fetch_snotel_season,
    fetch_avalanche, fetch_forecast_discussion,
    fetch_rwis_stations, fetch_radar_nowcast,
    analyze_all, compute_ski_decision, generate_storm_narrative,
    log_storm_event, get_storm_history,
    RESORTS, SNOTEL_STATIONS,
)
from pressure_forecast import get_storm_total

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralized coordinates
# ---------------------------------------------------------------------------
TAHOE = {"lat": 39.17, "lon": -120.145}
OAKLAND = {"lat": 37.8024, "lon": -122.1828}
SOUNDING_STATION = "REV"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_fetch(func, *args, default=None, **kwargs):
    """Call a fetch function, returning *default* on any exception."""
    if default is None:
        default = {}
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning("Fetch %s failed: %s", func.__name__, e)
        return default


def _check_error(data, default=None):
    """Return *default* when *data* is a dict containing an ``error`` key."""
    if isinstance(data, dict) and "error" in data:
        return default
    return data


# ---------------------------------------------------------------------------
# Full Tahoe analysis pipeline
# ---------------------------------------------------------------------------

def fetch_tahoe_analysis():
    """
    Fetch all data sources and run the full analysis pipeline.

    Returns the complete analysis dict ready for display/alerting.

    This is the single source of truth -- webapp, eink, alerts, and
    verify_cron should all call this instead of duplicating fetch logic.
    """
    logger.info("Starting Tahoe analysis pipeline")

    # Core data
    obs = _safe_fetch(fetch_nws_observations, TAHOE["lat"], TAHOE["lon"])
    nws = _safe_fetch(fetch_nws_forecast, TAHOE["lat"], TAHOE["lon"])

    # Per-resort Open-Meteo (NOT single-point -- each resort gets its own grid)
    resort_points = {rn: {"lat": r["base"]["lat"], "lon": r["base"]["lon"]}
                     for rn, r in RESORTS.items()}
    om = _safe_fetch(fetch_open_meteo_multi, resort_points)

    snotel = _safe_fetch(fetch_snotel_current)
    avy = _safe_fetch(fetch_avalanche)
    afd = _safe_fetch(fetch_forecast_discussion, default="")

    # Enrichment sources (all best-effort)
    tahoe_nbm = _safe_fetch(fetch_nbm, TAHOE["lat"], TAHOE["lon"])
    nws_grids = _safe_fetch(fetch_nws_gridpoints, TAHOE["lat"], TAHOE["lon"])
    cssl = _safe_fetch(fetch_cssl_snow)
    tahoe_alerts = _safe_fetch(fetch_nws_alerts, TAHOE["lat"], TAHOE["lon"], default=[])
    sounding = _safe_fetch(fetch_sounding, SOUNDING_STATION)
    normals = _safe_fetch(fetch_climate_normals, TAHOE["lat"], TAHOE["lon"])
    ensemble = _safe_fetch(fetch_ensemble, TAHOE["lat"], TAHOE["lon"])
    synoptic = _safe_fetch(fetch_synoptic_stations, TAHOE["lat"], TAHOE["lon"],
                           radius_miles=30)
    storm = _safe_fetch(get_storm_total, snotel)
    chains = _safe_fetch(fetch_caltrans_chains)
    lifts = _safe_fetch(fetch_all_lift_status)
    rwis = _safe_fetch(fetch_rwis_stations, TAHOE["lat"], TAHOE["lon"], default=[])
    radar = _safe_fetch(fetch_radar_nowcast, TAHOE["lat"], TAHOE["lon"])

    # Run analysis pipeline
    analysis = analyze_all(obs, nws, om, snotel, afd, avy, {},
                           nws_grids=_check_error(nws_grids),
                           sounding=_check_error(sounding),
                           ensemble=ensemble if ensemble.get("models") else None,
                           synoptic=_check_error(synoptic),
                           rwis=rwis if rwis else None,
                           cssl=_check_error(cssl),
                           radar_nowcast=radar)

    # Attach supplementary data for display
    analysis["nbm"] = _check_error(tahoe_nbm)
    if not analysis.get("nws_grids"):
        analysis["nws_grids"] = _check_error(nws_grids)
    analysis["cssl"] = _check_error(cssl)
    analysis["alerts"] = tahoe_alerts if isinstance(tahoe_alerts, list) else []
    if not analysis.get("sounding"):
        analysis["sounding"] = _check_error(sounding)
    analysis["normals"] = _check_error(normals)
    if not analysis.get("ensemble"):
        analysis["ensemble"] = ensemble if ensemble.get("models") else None
    analysis["synoptic"] = _check_error(synoptic)
    analysis["storm"] = storm
    analysis["chains"] = chains
    analysis["lifts"] = lifts
    analysis["rwis"] = rwis if rwis else []
    analysis["radar"] = radar

    # Compute decision (needs chains, lifts, storm attached)
    analysis["decision"] = compute_ski_decision(analysis)

    # Storm history
    analysis["storm_history"] = get_storm_history()

    logger.info("Tahoe analysis pipeline complete")
    return analysis


# ---------------------------------------------------------------------------
# Oakland-specific data (e-ink Oakland scene, verification)
# ---------------------------------------------------------------------------

def fetch_oakland_data():
    """
    Fetch Oakland-specific data for the e-ink Oakland scene and
    daily verification.

    Returns dict with home_obs, home_fc, home_om, home_nbm, home_pws,
    home_alerts.
    """
    logger.info("Fetching Oakland data")

    home_obs = _safe_fetch(fetch_nws_observations, OAKLAND["lat"], OAKLAND["lon"])
    home_fc = _safe_fetch(fetch_nws_forecast, OAKLAND["lat"], OAKLAND["lon"])
    home_om = _safe_fetch(fetch_open_meteo, OAKLAND["lat"], OAKLAND["lon"])
    home_nbm = _safe_fetch(fetch_nbm, OAKLAND["lat"], OAKLAND["lon"])
    home_pws = _safe_fetch(fetch_pws_nearby, OAKLAND["lat"], OAKLAND["lon"], default=[])
    home_alerts = _safe_fetch(fetch_nws_alerts, OAKLAND["lat"], OAKLAND["lon"], default=[])
    home_normals = _safe_fetch(fetch_climate_normals, OAKLAND["lat"], OAKLAND["lon"])

    logger.info("Oakland data fetch complete")
    return {
        "home_obs": home_obs,
        "home_fc": home_fc,
        "home_om": home_om,
        "home_nbm": home_nbm,
        "home_pws": home_pws,
        "home_alerts": home_alerts,
        "home_normals": _check_error(home_normals),
    }
