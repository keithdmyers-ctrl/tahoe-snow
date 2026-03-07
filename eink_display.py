#!/usr/bin/env python3
"""
Tahoe Snow E-Ink Display — Inky Impression 7.3" (2025 Edition)

Layout (local-focused):
  TOP 60%:  Indoor/outdoor temp, today's high, current conditions, 5-day forecast
  BOT 40%:  Tahoe conditions, resort snow, snowpack

Resolution: 800x480 (landscape) — 7-color (black, white, red, green, blue, yellow, orange)

Setup:
  pip install inky[rpi] pillow smbus2
  # Cron: */30 * * * * cd /home/keith/projects/tahoe-snow && .venv/bin/python3 eink_display.py

Usage:
  python eink_display.py            # render to display
  python eink_display.py --preview  # save preview PNG instead
"""

import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from tahoe_snow import (
    fetch_nws_observations, fetch_nws_forecast, fetch_open_meteo,
    fetch_snotel_current, fetch_avalanche, fetch_forecast_discussion,
    analyze_all, RESORTS,
)
from sensors import read_all as read_sensors

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: PIL/Pillow required. Install with: pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOME = {"label": "Oakland", "lat": 37.7792, "lon": -122.1958}
TAHOE = {"label": "Tahoe", "lat": 39.17, "lon": -120.145}

DISPLAY_W = 800
DISPLAY_H = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 150, 0)
BLUE = (0, 0, 200)
YELLOW = (200, 200, 0)
ORANGE = (200, 100, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (120, 120, 120)

RESORT_COLORS = {"Heavenly": RED, "Northstar": GREEN, "Kirkwood": BLUE}


def load_fonts():
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    ]
    bold_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    ]
    font_file = next((p for p in font_paths if os.path.exists(p)), None)
    bold_file = next((p for p in bold_paths if os.path.exists(p)), None)

    s = {}
    if font_file:
        b = bold_file or font_file
        s["sm"] = ImageFont.truetype(font_file, 11)
        s["md"] = ImageFont.truetype(font_file, 14)
        s["lg"] = ImageFont.truetype(b, 18)
        s["xl"] = ImageFont.truetype(b, 24)
        s["xxl"] = ImageFont.truetype(b, 40)
        s["huge"] = ImageFont.truetype(b, 56)
        s["bold"] = ImageFont.truetype(b, 14)
        s["bold_sm"] = ImageFont.truetype(b, 12)
    else:
        default = ImageFont.load_default()
        for key in ("sm", "md", "lg", "xl", "xxl", "huge", "bold", "bold_sm"):
            s[key] = default
    return s


def _t(val):
    if val is None or val == "?":
        return "--"
    try:
        return str(int(round(float(val))))
    except (ValueError, TypeError):
        return "--"


def _danger_color(level):
    return {0: GRAY, 1: GREEN, 2: YELLOW, 3: ORANGE, 4: RED, 5: RED}.get(level, GRAY)


def _extract_daily_forecast(nws_data: dict, days: int = 5) -> list:
    periods = nws_data.get("periods", [])
    by_day = {}
    for p in periods:
        name = p.get("name", "")
        temp = p.get("temperature")
        short = p.get("shortForecast", "")
        is_day = p.get("isDaytime", True)
        start = p.get("startTime", "")
        date_key = start[:10] if len(start) >= 10 else name
        if date_key not in by_day:
            by_day[date_key] = {"day": name, "hi": None, "lo": None, "short": ""}
        if is_day:
            by_day[date_key]["hi"] = temp
            by_day[date_key]["short"] = short
            by_day[date_key]["day"] = name
        else:
            by_day[date_key]["lo"] = temp
    return list(by_day.values())[:days]


def _snow_str(val):
    if val >= 0.5:
        return f'{val:.0f}"'
    elif val > 0:
        return f'{val:.1f}"'
    return "--"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(analysis, home_obs, home_forecast, sensor_data, preview_path=None):
    img = Image.new("RGB", (DISPLAY_W, DISPLAY_H), WHITE)
    draw = ImageDraw.Draw(img)
    f = load_fonts()

    cur = analysis["current_conditions"]
    tahoe_obs = cur.get("observation", {})
    avy = analysis.get("avalanche", {})
    snotel = analysis.get("snotel_current", {})
    indoor = sensor_data.get("indoor", {})
    outdoor = sensor_data.get("outdoor", {})

    y = 0

    # ==================================================================
    # HEADER BAR
    # ==================================================================
    draw.rectangle([0, 0, DISPLAY_W, 24], fill=BLACK)
    draw.text((8, 2), HOME["label"].upper(), fill=WHITE, font=f["lg"])
    try:
        dt = datetime.fromisoformat(analysis["generated"])
        ts = dt.strftime("%a %b %d  %I:%M%p")
    except Exception:
        ts = ""
    draw.text((DISPLAY_W - 195, 4), ts, fill=GRAY, font=f["md"])
    y = 26

    # ==================================================================
    # SECTION 1: BIG TEMPS  (y=26..130) — ~22% of screen
    # ==================================================================

    # Indoor (left)
    draw.text((12, y), "INDOOR", fill=DARK_GRAY, font=f["bold_sm"])
    in_temp = _t(indoor.get("temp_f"))
    draw.text((12, y + 14), f"{in_temp}°", fill=BLACK, font=f["huge"])
    in_hum = indoor.get("humidity_pct")
    if in_hum is not None:
        draw.text((12, y + 72), f"{in_hum:.0f}% rh", fill=GRAY, font=f["sm"])

    # Outdoor (center-left)
    draw.text((170, y), "OUTDOOR", fill=DARK_GRAY, font=f["bold_sm"])
    out_temp = _t(outdoor.get("temp_f"))
    stale = outdoor.get("stale", False)
    draw.text((170, y + 14), f"{out_temp}°", fill=GRAY if stale else BLACK, font=f["huge"])
    out_hum = outdoor.get("humidity_pct")
    if out_hum is not None:
        draw.text((170, y + 72), f"{out_hum:.0f}% rh", fill=GRAY, font=f["sm"])

    # Today's high + conditions (center-right)
    daily = _extract_daily_forecast(home_forecast, 1)
    today_hi = _t(daily[0]["hi"]) if daily and daily[0].get("hi") is not None else "--"
    draw.text((340, y), "TODAY'S HIGH", fill=DARK_GRAY, font=f["bold_sm"])
    draw.text((340, y + 14), f"{today_hi}°", fill=BLACK, font=f["huge"])
    h_cond = home_obs.get("conditions", "")
    if h_cond:
        draw.text((340, y + 72), h_cond, fill=BLUE, font=f["md"])

    # Wind + humidity from NWS (below today's high)
    h_wd = home_obs.get("wind_dir", "")
    h_ws = home_obs.get("wind_mph", 0)
    if h_ws:
        draw.text((500, y + 72), f"Wind: {h_wd} {h_ws:.0f}mph", fill=GRAY, font=f["sm"])

    # NWS temp as small reference (in case sensors are offline)
    nws_temp = _t(home_obs.get("temp_f"))
    if nws_temp != "--":
        draw.text((500, y + 14), f"NWS: {nws_temp}°F", fill=GRAY, font=f["sm"])

    y += 90

    # ==================================================================
    # SECTION 2: 5-DAY FORECAST  (y=116..210) — ~20% of screen
    # ==================================================================
    draw.line([(0, y), (DISPLAY_W, y)], fill=BLACK, width=2)
    y += 4
    draw.text((8, y), "5-DAY FORECAST", fill=BLACK, font=f["lg"])
    y += 22

    forecast_days = _extract_daily_forecast(home_forecast, 5)
    fc_w = (DISPLAY_W - 16) // 5
    for i, day in enumerate(forecast_days):
        cx = 8 + i * fc_w
        name = day.get("day", "")[:3]
        hi = _t(day.get("hi"))
        lo = _t(day.get("lo"))
        short = day.get("short", "")

        # Day name
        draw.text((cx, y), name, fill=BLACK, font=f["bold"])

        # Hi temp (big)
        draw.text((cx + 30, y - 4), f"{hi}°", fill=BLACK, font=f["xl"])

        # Lo temp
        if lo != "--":
            draw.text((cx + 75, y + 2), f"/{lo}°", fill=GRAY, font=f["md"])

        # Condition (two lines if needed)
        words = short.split()
        line1 = ""
        line2 = ""
        for w in words:
            if len(line1 + " " + w) <= 18 and not line2:
                line1 = (line1 + " " + w).strip()
            else:
                line2 = (line2 + " " + w).strip()
        draw.text((cx, y + 22), line1[:20], fill=BLUE, font=f["sm"])
        if line2:
            draw.text((cx, y + 34), line2[:20], fill=BLUE, font=f["sm"])

    y += 50

    # ==================================================================
    # SECTION 3: TAHOE CONDITIONS  (y=260..380) — ~25% of screen
    # ==================================================================
    # Blue header bar for Tahoe section
    draw.rectangle([0, y, DISPLAY_W, y + 22], fill=BLUE)
    draw.text((8, y + 2), "LAKE TAHOE", fill=WHITE, font=f["lg"])

    # Tahoe current on header bar
    t_temp = _t(tahoe_obs.get("temp_f", cur.get("lake_level_temp_f")))
    t_cond = tahoe_obs.get("conditions", "")[:20]
    draw.text((160, y + 3), f"{t_temp}°F  {t_cond}", fill=WHITE, font=f["bold"])

    # Snow level / freeze / avy on header bar
    sl = cur.get("snow_level_ft")
    fl_val = cur.get("freezing_level_ft")
    avy_str = ""
    if avy and "danger_label" in avy:
        avy_str = f"Avy:{avy['danger_label']}"
    info = ""
    if sl is not None:
        info += f"Snow:{sl}ft  "
    if fl_val:
        info += f"Frz:{fl_val}ft  "
    info += avy_str
    draw.text((420, y + 3), info, fill=WHITE, font=f["bold_sm"])
    y += 24

    # Resort columns
    resort_names = list(RESORTS.keys())
    col_w = DISPLAY_W // 3
    col_x = [i * col_w for i in range(3)]

    for i, rn in enumerate(resort_names):
        x = col_x[i]
        rc = RESORT_COLORS.get(rn, BLACK)
        draw.rectangle([x, y, x + col_w - 2, y + 15], fill=rc)
        draw.text((x + 4, y), rn.upper(), fill=WHITE, font=f["bold_sm"])
    y += 17

    # Compact zone data: peak and base only (skip mid to save space)
    for zone in ("peak", "base"):
        zone_label = "Pk" if zone == "peak" else "Bs"
        y_start = y

        for i, rn in enumerate(resort_names):
            x = col_x[i] + 4
            rd = analysis.get("resorts", {}).get(rn, {})
            zd = rd.get("zones", {}).get(zone, {})
            snap = zd.get("current", {})
            elev = zd.get("elev_ft", "?")
            snow24 = zd.get("snow_24h", 0)

            ty = y_start
            if i == 0:
                draw.text((x, ty), zone_label, fill=BLACK, font=f["bold_sm"])
            draw.text((x + 18, ty), f"{elev}ft", fill=GRAY, font=f["sm"])

            # Temp + snow
            tf = _t(snap.get("temp_f"))
            sc = RED if snow24 > 6 else BLUE if snow24 > 0 else GRAY
            draw.text((x + 75, ty), f"{tf}°", fill=BLACK, font=f["bold_sm"])
            draw.text((x + 100, ty), _snow_str(snow24), fill=sc, font=f["bold_sm"])

            ty += 13
            # Wind
            wd = snap.get("wind_dir", "")
            ws = snap.get("wind_mph", 0)
            wg = snap.get("wind_gust_mph", 0)
            wc = RED if wg > 40 else ORANGE if wg > 25 else GRAY
            wt = f"{wd}{ws:.0f}" if ws else ""
            if wg > 0:
                wt += f"g{wg:.0f}"
            draw.text((x, ty), wt, fill=wc, font=f["sm"])

            # Quality
            if snow24 > 0:
                q = snap.get("snow_quality", "")[:10]
                if q:
                    draw.text((x + 60, ty), q, fill=BLUE, font=f["sm"])

        y = y_start + 28
        if zone == "peak":
            draw.line([(4, y - 1), (DISPLAY_W - 4, y - 1)], fill=GRAY, width=1)

    y += 2

    # 5-day peak snow (compact, single row per resort)
    draw.line([(0, y), (DISPLAY_W, y)], fill=BLACK, width=1)
    y += 2
    draw.text((4, y), "5-DAY SNOW", fill=BLACK, font=f["bold_sm"])
    y += 13

    for i, rn in enumerate(resort_names):
        rd = analysis.get("resorts", {}).get(rn, {})
        peak = rd.get("zones", {}).get("peak", {})
        buckets = peak.get("day_night_buckets", [])

        daily_snow = {}
        for b in buckets:
            d = b["date"]
            if d not in daily_snow:
                daily_snow[d] = 0
            daily_snow[d] += b["snow_in"]

        dates = sorted(daily_snow.keys())[:5]
        rc = RESORT_COLORS.get(rn, BLACK)
        ry = y + i * 16

        draw.text((4, ry), rn[:4], fill=rc, font=f["bold_sm"])

        day_w = (DISPLAY_W - 50) // 5
        for j, date in enumerate(dates):
            dx = 45 + j * day_w
            snow = daily_snow[date]

            if i == 0:
                try:
                    ddt = datetime.strptime(date, "%Y-%m-%d")
                    draw.text((dx, y - 13), ddt.strftime("%a"), fill=BLACK, font=f["sm"])
                except Exception:
                    pass

            sc = RED if snow > 6 else BLUE if snow > 0 else GRAY
            draw.text((dx, ry), _snow_str(snow), fill=sc, font=f["bold_sm"])

            # Inline bar
            if snow > 0:
                bar_w = min(int(snow * 2), day_w - 35)
                draw.rectangle([dx + 28, ry + 3, dx + 28 + bar_w, ry + 9], fill=sc)

    y += 50

    # ==================================================================
    # SECTION 4: SNOWPACK (bottom strip)
    # ==================================================================
    snotel_y = DISPLAY_H - 22
    draw.line([(0, snotel_y - 2), (DISPLAY_W, snotel_y - 2)], fill=BLACK, width=1)
    draw.text((4, snotel_y), "PACK", fill=BLACK, font=f["bold_sm"])

    stations = [(n, d) for n, d in snotel.items()
                if "error" not in d and d.get("snow_depth_in") is not None
                and d.get("snow_depth_in") > 0]
    stations.sort(key=lambda x: x[1]["snow_depth_in"], reverse=True)

    sx = 45
    for name, data in stations[:6]:
        depth = data["snow_depth_in"]
        draw.text((sx, snotel_y), f"{name[:9]}:{depth:.0f}\"", fill=DARK_GRAY, font=f["sm"])
        sx += 125

    # ── Output ──
    if preview_path:
        img.save(preview_path)
        print(f"Preview saved to {preview_path}")
    else:
        try:
            from inky.auto import auto
            display = auto()
            display.set_image(img)
            display.show()
            print("Display updated.")
        except ImportError:
            fallback = os.path.join(os.path.dirname(__file__), "eink_preview.png")
            img.save(fallback)
            print(f"Inky library not available. Preview saved to {fallback}")
        except Exception as e:
            fallback = os.path.join(os.path.dirname(__file__), "eink_preview.png")
            img.save(fallback)
            print(f"Display error: {e}. Preview saved to {fallback}")


def main():
    preview = "--preview" in sys.argv
    preview_path = os.path.join(os.path.dirname(__file__), "eink_preview.png") if preview else None

    print("Reading local sensors...")
    sensor_data = read_sensors()

    print("Fetching Oakland weather...")
    home_obs = fetch_nws_observations(HOME["lat"], HOME["lon"])
    home_forecast = fetch_nws_forecast(HOME["lat"], HOME["lon"])

    print("Fetching Tahoe data...")
    obs = fetch_nws_observations(TAHOE["lat"], TAHOE["lon"])
    nws = fetch_nws_forecast(TAHOE["lat"], TAHOE["lon"])
    om = fetch_open_meteo(TAHOE["lat"], TAHOE["lon"])
    snotel = fetch_snotel_current()
    avy = fetch_avalanche()
    afd = fetch_forecast_discussion()

    print("Analyzing...")
    analysis = analyze_all(obs, nws, om, snotel, afd, avy, {})

    print("Rendering...")
    render(analysis, home_obs, home_forecast, sensor_data,
           preview_path=preview_path if preview else None)

    if not preview:
        render(analysis, home_obs, home_forecast, sensor_data,
               preview_path=os.path.join(os.path.dirname(__file__), "eink_preview.png"))


if __name__ == "__main__":
    main()
