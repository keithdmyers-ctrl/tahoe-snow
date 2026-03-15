# Oakland Weather Display — Design Spec

800x480px fixed canvas, white background (#FFF), sans-serif font. Seven horizontal bands stacked vertically. No scrolling.

```
+------------------------------------------------------------------+
|  HEADER BAR                                               28px   |
+------------------------------------------------------------------+
|  TEMPERATURE ROW                                         ~90px   |
|  CONDITIONS ROW                                          ~20px   |
+------------------------------------------------------------------+
|  BAROMETER STRIP                                         ~22px   |
+------------------------------------------------------------------+
|  HOURLY FORECAST STRIP                                    68px   |
+------------------------------------------------------------------+
|  5-DAY FORECAST CARDS                            ~230px (grows)  |
+------------------------------------------------------------------+
|  BOTTOM BAR                                               18px   |
+------------------------------------------------------------------+
```

## 1. Header Bar (28px)
Full-width black (#111) bar. Left: "OAKLAND" white, bold, 16px. Right: "Fri Mar 06 10:20PM" gray #AAA, 12px.

## 2. Temperature Row (~90px) + Conditions (~20px)
Three center-aligned blocks side by side, 12px gaps:
- **Indoor**: label "INDOOR" (10px uppercase gray #888), value "68deg" (52px bold black), sub "47.3% rh" (12px gray)
- **Outdoor**: same layout, "54deg". Turns gray #AAA when stale (>15min).
- **Today's High**: same layout, "73deg", sub "54deg low"

Below temps, a single conditions row (13px, 16px gap between items):
- Conditions text: blue #00C bold ("Mostly Cloudy")
- Wind: gray #888 ("S 12mph gusts 18")
- NWS ref: gray #AAA 11px ("NWS: 56degF")

## 3. Barometer Strip (~22px)
Single-line bar, border-top 1px #EEE. Five elements left to right, 12px gaps:

1. **Pressure**: bold 13px, "1014 hPa"
2. **Trend arrow**: 14px, color-coded — green #090 up-triangle (rising), red #C00 down-triangle (falling), gray #888 bar (steady)
3. **Trend label**: gray #666, "Falling (-4.1/3h)"
4. **Rain summary**: flex-grow, right-aligned, #333, "Rain likely by 6pm (5h)"
5. **Rain badge**: bold 11px, rounded pill (3px radius), min-width 44px
   - <25%: green bg #E8F5E9, text #2E7D32 — shows "dry" when no rain
   - 25-54%: orange bg #FFF3E0, text #E65100
   - 55%+: red bg #FFEBEE, text #C62828

## 4. Hourly Strip (68px)
16 equal-width cells in a row. Border-top 1px #CCC, border-bottom 2px #111. Each cell is a centered vertical stack:
- Time: 9px gray #888 ("6pm")
- Icon: 16px emoji (sun/cloud/rain)
- Temp: 14px bold ("62deg")
- Precip: 9px blue #00C, only if >0% ("80%")

## 5. Five-Day Forecast (~230px, flex-grow)
Five equal-width cards, 6px gap. Each card: 1px border #CCC, 6px radius, content vertically centered, 6px internal gap:
- Day name: bold 16px ("Thu")
- Icon: 32px emoji
- Temps: bold 20px high + normal 15px gray low ("73deg 54deg")
- Condition: 10px blue #00C, centered, max 22px ("Sunny")

## 6. Bottom Bar (18px)
Full-width black #111 bar, 10px gray #AAA text. Left: "A: Oakland | B: Heavenly | C: Refresh | D: Detail". Right: "Active: Oakland".

## Colors
| Hex | Usage |
|-----|-------|
| #111 | Header/footer bg, text, heavy borders |
| #FFF | Background, header text |
| #00C | Conditions, precip %, forecast labels |
| #C00 | Falling arrow, high-rain badge |
| #090 | Rising arrow, low-rain badge |
| #888 | Labels, sub-text, wind, time |
| #AAA | Timestamps, NWS ref, stale values |
| #CCC | Card/strip borders |

## Type Scale
52px bold (big temps) > 32px (forecast icons) > 20px bold (forecast hi) > 16px bold (day names, location) > 14px bold (hourly temps) > 13px (conditions, pressure) > 12px (sub-text) > 10-11px (labels, badges) > 9px (time, precip)
