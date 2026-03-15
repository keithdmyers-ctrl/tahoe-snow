#!/usr/bin/env python3
"""
Multi-Range Resort Configuration System for Tahoe Snow

Externalizes resort configuration for easy expansion to additional resorts.
Currently supports Heavenly, Northstar, and Kirkwood (enabled), with stubbed
entries for Palisades Tahoe, Sugar Bowl, Sierra-at-Tahoe, Boreal, and Mt. Rose.

All coordinates, elevations, and SNOTEL associations are maintained here.
Activate additional resorts by setting enabled=True.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ZoneConfig:
    """Configuration for a single elevation zone at a resort."""
    name: str               # e.g., "base", "mid", "peak"
    label: str              # Human-readable label (e.g., "California Lodge")
    lat: float              # Latitude (WGS84)
    lon: float              # Longitude (WGS84)
    elevation_ft: int       # Elevation in feet above sea level
    aspect_deg: float = 0.0    # Dominant slope aspect in degrees (0=N, 90=E, 180=S, 270=W)
    slope_angle_deg: float = 0.0  # Average slope angle in degrees


@dataclass
class ResortConfig:
    """Full configuration for a ski resort."""
    name: str                  # Display name
    region: str                # Geographic grouping (e.g., "tahoe_south", "tahoe_north")
    zones: list                # List of ZoneConfig (base, mid, peak)
    liftie_slug: str = ""      # Liftie API slug for lift status
    snotel_stations: list = field(default_factory=list)  # SNOTEL station IDs
    nearest_sounding: str = "REV"  # Nearest upper-air sounding station (REV = Reno)
    chain_route: str = ""      # Primary highway route (e.g., "I-80", "US-50")
    webcam_urls: list = field(default_factory=list)
    timezone: str = "America/Los_Angeles"
    enabled: bool = True       # Set False to exclude from active analysis
    aspect: str = "N"          # Dominant aspect shorthand (N, NE, SW, etc.)

    def get_zone(self, zone_name: str) -> Optional[ZoneConfig]:
        """Get a zone by name (base/mid/peak)."""
        for z in self.zones:
            if z.name == zone_name:
                return z
        return None

    def base(self) -> Optional[ZoneConfig]:
        return self.get_zone("base")

    def mid(self) -> Optional[ZoneConfig]:
        return self.get_zone("mid")

    def peak(self) -> Optional[ZoneConfig]:
        return self.get_zone("peak")


# ---------------------------------------------------------------------------
# Resort Registry
# ---------------------------------------------------------------------------

RESORT_REGISTRY: dict[str, ResortConfig] = {

    # ===================================================================
    # ACTIVE RESORTS (enabled=True) — currently analyzed by tahoe_snow.py
    # ===================================================================

    "Heavenly": ResortConfig(
        name="Heavenly",
        region="tahoe_south",
        zones=[
            ZoneConfig("base", "California Lodge", 38.9353, -119.9406, 6540,
                       aspect_deg=45, slope_angle_deg=15),
            ZoneConfig("mid", "Sky Deck / Tamarack", 38.9310, -119.9250, 8500,
                       aspect_deg=45, slope_angle_deg=25),
            ZoneConfig("peak", "Monument Peak", 38.9280, -119.9070, 10067,
                       aspect_deg=45, slope_angle_deg=30),
        ],
        liftie_slug="heavenly",
        snotel_stations=["473", "518"],  # Fallen Leaf, Hagan's Meadow
        nearest_sounding="REV",
        chain_route="US-50",
        webcam_urls=[
            "https://www.skiheavenly.com/the-mountain/mountain-conditions/web-cams.aspx",
        ],
        aspect="NE",
    ),

    "Northstar": ResortConfig(
        name="Northstar",
        region="tahoe_north",
        zones=[
            ZoneConfig("base", "Village", 39.2744, -120.1210, 6330,
                       aspect_deg=225, slope_angle_deg=12),
            ZoneConfig("mid", "Vista Express", 39.2680, -120.1150, 7600,
                       aspect_deg=225, slope_angle_deg=22),
            ZoneConfig("peak", "Mt Pluto", 39.2630, -120.1100, 8610,
                       aspect_deg=225, slope_angle_deg=28),
        ],
        liftie_slug="northstar",
        snotel_stations=["539", "540", "809"],  # Indep. Lake, Indep. Camp, Tahoe City Cross
        nearest_sounding="REV",
        chain_route="I-80",
        webcam_urls=[
            "https://www.northstarcalifornia.com/the-mountain/mountain-conditions/web-cams.aspx",
        ],
        aspect="SW",
    ),

    "Kirkwood": ResortConfig(
        name="Kirkwood",
        region="tahoe_south",
        zones=[
            ZoneConfig("base", "Lodge", 38.6850, -120.0650, 7800,
                       aspect_deg=0, slope_angle_deg=18),
            ZoneConfig("mid", "Sunrise / Solitude", 38.6820, -120.0720, 8800,
                       aspect_deg=0, slope_angle_deg=28),
            ZoneConfig("peak", "Thimble Peak", 38.6790, -120.0780, 9800,
                       aspect_deg=0, slope_angle_deg=35),
        ],
        liftie_slug="kirkwood",
        snotel_stations=["428", "518"],  # CSS Lab, Hagan's Meadow
        nearest_sounding="REV",
        chain_route="SR-88",
        webcam_urls=[
            "https://www.kirkwood.com/the-mountain/mountain-conditions/web-cams.aspx",
        ],
        aspect="N",
    ),

    # ===================================================================
    # STUBBED RESORTS — Activate by setting enabled=True
    # ===================================================================

    "Palisades Tahoe": ResortConfig(  # Activate by setting enabled=True
        name="Palisades Tahoe",
        region="tahoe_north",
        zones=[
            ZoneConfig("base", "Village at Palisades", 39.1968, -120.2354, 6200,
                       aspect_deg=315, slope_angle_deg=15),
            ZoneConfig("mid", "Gold Coast", 39.1930, -120.2400, 7800,
                       aspect_deg=315, slope_angle_deg=25),
            ZoneConfig("peak", "Granite Chief", 39.1850, -120.2550, 9050,
                       aspect_deg=315, slope_angle_deg=35),
        ],
        liftie_slug="palisades-tahoe",
        snotel_stations=["784", "539"],  # Squaw Valley GC, Independence Lake
        nearest_sounding="REV",
        chain_route="I-80",
        aspect="NW",
        enabled=False,
    ),

    "Sugar Bowl": ResortConfig(  # Activate by setting enabled=True
        name="Sugar Bowl",
        region="tahoe_north",
        zones=[
            ZoneConfig("base", "Judah Lodge", 39.3050, -120.3340, 6883,
                       aspect_deg=180, slope_angle_deg=18),
            ZoneConfig("mid", "Christmas Tree", 39.3020, -120.3300, 7500,
                       aspect_deg=180, slope_angle_deg=25),
            ZoneConfig("peak", "Mt Lincoln", 39.2990, -120.3260, 8383,
                       aspect_deg=180, slope_angle_deg=32),
        ],
        liftie_slug="sugar-bowl",
        snotel_stations=["784", "809"],  # Squaw Valley GC, Tahoe City Cross
        nearest_sounding="REV",
        chain_route="I-80",
        aspect="S",
        enabled=False,
    ),

    "Sierra-at-Tahoe": ResortConfig(  # Activate by setting enabled=True
        name="Sierra-at-Tahoe",
        region="tahoe_south",
        zones=[
            ZoneConfig("base", "Sierra Lodge", 38.7990, -120.0800, 6640,
                       aspect_deg=0, slope_angle_deg=14),
            ZoneConfig("mid", "Grandview Express", 38.7960, -120.0850, 7600,
                       aspect_deg=0, slope_angle_deg=22),
            ZoneConfig("peak", "Huckleberry Peak", 38.7930, -120.0900, 8852,
                       aspect_deg=0, slope_angle_deg=30),
        ],
        liftie_slug="sierra-at-tahoe",
        snotel_stations=["428", "473"],  # CSS Lab, Fallen Leaf
        nearest_sounding="REV",
        chain_route="US-50",
        aspect="N",
        enabled=False,
    ),

    "Boreal": ResortConfig(  # Activate by setting enabled=True
        name="Boreal",
        region="tahoe_north",
        zones=[
            ZoneConfig("base", "Boreal Lodge", 39.3320, -120.3480, 7200,
                       aspect_deg=180, slope_angle_deg=12),
            ZoneConfig("mid", "Discovery", 39.3340, -120.3460, 7520,
                       aspect_deg=180, slope_angle_deg=18),
            ZoneConfig("peak", "Boreal Summit", 39.3360, -120.3440, 7850,
                       aspect_deg=180, slope_angle_deg=22),
        ],
        liftie_slug="boreal",
        snotel_stations=["784", "809"],  # Squaw Valley GC, Tahoe City Cross
        nearest_sounding="REV",
        chain_route="I-80",
        aspect="S",
        enabled=False,
    ),

    "Mt. Rose": ResortConfig(  # Activate by setting enabled=True
        name="Mt. Rose",
        region="tahoe_north",
        zones=[
            ZoneConfig("base", "Reno Side Lodge", 39.3150, -119.8850, 8260,
                       aspect_deg=315, slope_angle_deg=18),
            ZoneConfig("mid", "Lakeview", 39.3120, -119.8880, 8800,
                       aspect_deg=315, slope_angle_deg=25),
            ZoneConfig("peak", "Slide Mountain Summit", 39.3100, -119.8900, 9700,
                       aspect_deg=315, slope_angle_deg=32),
        ],
        liftie_slug="mt-rose",
        snotel_stations=["652"],  # Mt Rose Ski Area
        nearest_sounding="REV",
        chain_route="SR-431",
        aspect="NW",
        enabled=False,
    ),
}


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_active_resorts() -> dict[str, ResortConfig]:
    """Return only enabled resorts.

    Returns:
        Dict of {resort_name: ResortConfig} for all enabled resorts.
    """
    return {name: cfg for name, cfg in RESORT_REGISTRY.items() if cfg.enabled}


def get_resort(name: str) -> Optional[ResortConfig]:
    """Return a single resort config by name.

    Args:
        name: Resort name (e.g., "Heavenly").

    Returns:
        ResortConfig if found, None otherwise.
    """
    return RESORT_REGISTRY.get(name)


def get_all_snotel_stations() -> list[str]:
    """Return deduplicated list of SNOTEL station IDs across all active resorts.

    Returns:
        Sorted list of unique SNOTEL station ID strings.
    """
    station_ids = set()
    for cfg in get_active_resorts().values():
        for sid in cfg.snotel_stations:
            station_ids.add(sid)
    return sorted(station_ids)


def get_resorts_by_region(region: str) -> dict[str, ResortConfig]:
    """Return active resorts in a specific region.

    Args:
        region: Region name (e.g., "tahoe_south", "tahoe_north").

    Returns:
        Dict of {resort_name: ResortConfig} matching the region.
    """
    return {
        name: cfg for name, cfg in get_active_resorts().items()
        if cfg.region == region
    }


def to_legacy_format(resort_cfg: ResortConfig) -> dict:
    """Convert ResortConfig to the legacy RESORTS dict format used in tahoe_snow.py.

    This produces the exact structure that tahoe_snow.py currently expects:
    {"base": {"label": ..., "lat": ..., "lon": ..., "elev_ft": ...}, ...}

    Args:
        resort_cfg: ResortConfig to convert.

    Returns:
        Dict in the legacy format.
    """
    result = {"aspect": resort_cfg.aspect}
    snotel_names_by_id = {
        "473": "Fallen Leaf",
        "518": "Hagan's Meadow",
        "539": "Independence Lake",
        "540": "Independence Camp",
        "809": "Tahoe City Cross",
        "724": "Rubicon #2",
        "784": "Squaw Valley GC",
        "848": "Ward Creek #3",
        "652": "Mt Rose Ski Area",
        "428": "CSS Lab",
    }
    result["nearest_snotel"] = [
        snotel_names_by_id.get(sid, sid) for sid in resort_cfg.snotel_stations
    ]
    for zone in resort_cfg.zones:
        result[zone.name] = {
            "label": zone.label,
            "lat": zone.lat,
            "lon": zone.lon,
            "elev_ft": zone.elevation_ft,
        }
    return result


def get_active_resorts_legacy() -> dict:
    """Return active resorts in the legacy RESORTS dict format.

    This is a drop-in replacement for the RESORTS dict in tahoe_snow.py.

    Returns:
        Dict compatible with current tahoe_snow.py RESORTS format.
    """
    return {name: to_legacy_format(cfg) for name, cfg in get_active_resorts().items()}


# === Migration from tahoe_snow.py ===
# To use this config, replace RESORTS dict in tahoe_snow.py with:
#
#   from resort_configs import get_active_resorts, RESORT_REGISTRY
#   RESORTS = {name: {
#       "zones": {z.name: {"lat": z.lat, "lon": z.lon, "elev": z.elevation_ft} for z in cfg.zones}
#   } for name, cfg in get_active_resorts().items()}
#
# Or for a drop-in replacement that preserves the exact legacy format:
#
#   from resort_configs import get_active_resorts_legacy
#   RESORTS = get_active_resorts_legacy()
