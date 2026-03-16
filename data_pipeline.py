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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    fetch_air_quality,
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
    logger.info("Starting Tahoe analysis pipeline (parallel fetch)")

    lat, lon = TAHOE["lat"], TAHOE["lon"]
    resort_points = {rn: {"lat": r["base"]["lat"], "lon": r["base"]["lon"]}
                     for rn, r in RESORTS.items()}

    # Parallel fetch all data sources — typically 20+ HTTP calls in ~15s instead of ~120s
    results = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            pool.submit(_safe_fetch, fetch_nws_observations, lat, lon): "obs",
            pool.submit(_safe_fetch, fetch_nws_forecast, lat, lon): "nws",
            pool.submit(_safe_fetch, fetch_open_meteo_multi, resort_points): "om",
            pool.submit(_safe_fetch, fetch_snotel_current): "snotel",
            pool.submit(_safe_fetch, fetch_avalanche): "avy",
            pool.submit(_safe_fetch, fetch_forecast_discussion, default=""): "afd",
            pool.submit(_safe_fetch, fetch_nbm, lat, lon): "tahoe_nbm",
            pool.submit(_safe_fetch, fetch_nws_gridpoints, lat, lon): "nws_grids",
            pool.submit(_safe_fetch, fetch_cssl_snow): "cssl",
            pool.submit(_safe_fetch, fetch_nws_alerts, lat, lon, default=[]): "tahoe_alerts",
            pool.submit(_safe_fetch, fetch_sounding, SOUNDING_STATION): "sounding",
            pool.submit(_safe_fetch, fetch_climate_normals, lat, lon): "normals",
            pool.submit(_safe_fetch, fetch_ensemble, lat, lon): "ensemble",
            pool.submit(_safe_fetch, fetch_synoptic_stations, lat, lon, radius_miles=30): "synoptic",
            pool.submit(_safe_fetch, fetch_caltrans_chains): "chains",
            pool.submit(_safe_fetch, fetch_all_lift_status): "lifts",
            pool.submit(_safe_fetch, fetch_rwis_stations, lat, lon, default=[]): "rwis",
            pool.submit(_safe_fetch, fetch_radar_nowcast, lat, lon): "radar",
            pool.submit(_safe_fetch, fetch_air_quality, lat, lon): "tahoe_aqi",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning("Parallel fetch %s failed: %s", key, e)
                results[key] = {} if key not in ("rwis", "tahoe_alerts") else []

    obs = results.get("obs", {})
    nws = results.get("nws", {})
    om = results.get("om", {})
    snotel = results.get("snotel", {})
    avy = results.get("avy", {})
    afd = results.get("afd", "")
    tahoe_nbm = results.get("tahoe_nbm", {})
    nws_grids = results.get("nws_grids", {})
    cssl = results.get("cssl", {})
    tahoe_alerts = results.get("tahoe_alerts", [])
    sounding = results.get("sounding", {})
    normals = results.get("normals", {})
    ensemble = results.get("ensemble", {})
    synoptic = results.get("synoptic", {})
    chains = results.get("chains", {})
    lifts = results.get("lifts", {})
    rwis = results.get("rwis", [])
    radar = results.get("radar", {})
    tahoe_aqi = results.get("tahoe_aqi", {})

    # Storm total needs snotel data (sequential dependency)
    storm = _safe_fetch(get_storm_total, snotel)

    logger.info("All data fetched, running analysis")

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
    analysis["air_quality"] = _check_error(tahoe_aqi)

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
    logger.info("Fetching Oakland data (parallel)")

    lat, lon = OAKLAND["lat"], OAKLAND["lon"]
    r = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_safe_fetch, fetch_nws_observations, lat, lon): "obs",
            pool.submit(_safe_fetch, fetch_nws_forecast, lat, lon): "fc",
            pool.submit(_safe_fetch, fetch_open_meteo, lat, lon): "om",
            pool.submit(_safe_fetch, fetch_nbm, lat, lon): "nbm",
            pool.submit(_safe_fetch, fetch_pws_nearby, lat, lon, default=[]): "pws",
            pool.submit(_safe_fetch, fetch_nws_alerts, lat, lon, default=[]): "alerts",
            pool.submit(_safe_fetch, fetch_climate_normals, lat, lon): "normals",
            pool.submit(_safe_fetch, fetch_air_quality, lat, lon): "aqi",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                r[key] = future.result()
            except Exception:
                r[key] = [] if key in ("pws", "alerts") else {}

    logger.info("Oakland data fetch complete")
    return {
        "home_obs": r.get("obs", {}),
        "home_fc": r.get("fc", {}),
        "home_om": r.get("om", {}),
        "home_nbm": r.get("nbm", {}),
        "home_pws": r.get("pws", []),
        "home_alerts": r.get("alerts", []),
        "home_normals": _check_error(r.get("normals", {})),
        "home_aqi": _check_error(r.get("aqi", {})),
    }
