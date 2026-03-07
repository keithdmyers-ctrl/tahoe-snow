# How the Models Work

Overview of data sources, physics models, and fusion logic in Tahoe Snow.

---

## TL;DR

**Tahoe Snow** is a DIY weather station that beats phone apps by doing what they can't:

1. **Snow forecasts by elevation** — Your phone says "snow in Lake Tahoe." Cool, but the parking lot at 6,500ft is getting rain while the peak at 10,000ft is getting a foot of powder. This system forecasts each zone separately, adjusts temperature for altitude, and applies real snow physics — cold storms make fluffy 20:1 powder, warm storms make heavy 5:1 Sierra cement. Same liquid precipitation, completely different ski day.

2. **Four models, not one** — Apple uses one model. Google uses another. This runs GFS, ECMWF, ICON, and HRRR side by side. When they all agree on 8 inches, confidence is high and you should call in sick. When they range from 2 to 14 inches, it tells you that honestly instead of just picking a number.

3. **Local sensors in the loop** — A barometer on my porch detects dropping pressure 2-6 hours before model forecasts update. Five nearby personal weather stations vote on whether it's actually raining right now. A twice-daily weather balloon from Reno measures the real freezing level instead of guessing. SNOTEL stations across the basin track storm totals in real time. All of that gets fused together with weighted averaging — models get the most weight, but ground truth always wins.

4. **Two different problems, two different approaches** — Oakland and Tahoe need completely different forecasting. Oakland rain prediction fuses four sources (NWS bias-corrected blend, raw model consensus, hourly forecast, and the porch barometer) into a single weighted probability with a confidence rating — the question is just "will it rain and when?" Tahoe snow is a much harder problem: it runs full snow physics across three elevation zones per resort, tracks orographic enhancement from wind direction, computes snow-to-liquid ratios from temperature, and cross-references ten SNOTEL stations. Same system, but the mountain side does way more work because altitude changes everything.

All free public data, no API keys, no subscriptions. Runs on a Raspberry Pi with a 7-color e-ink display on the kitchen counter and a web dashboard on my phone.

---

## Architecture at a Glance

```
                     ┌──────────────────────────────┐
                     │        Data Sources           │
                     │  (free public APIs, sensors)  │
                     └──────────┬───────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Numerical   │     │ Observation │     │   Local     │
    │  Weather     │     │  Networks   │     │   Sensors   │
    │  Models      │     │             │     │             │
    │ GFS, ECMWF,  │     │ NWS, SNOTEL │     │  BME280     │
    │ ICON, HRRR,  │     │ PWS, CSSL,  │     │  (ESP32)    │
    │ NBM          │     │ Soundings   │     │             │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └────────────────┬──┴───────────────────┘
                            ▼
              ┌──────────────────────────┐
              │     Snow Physics &       │
              │     Elevation Adjustment │
              │  (lapse rate, SLR, oro)  │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │   Multi-Source Fusion     │
              │  (weighted average,      │
              │   bias correction,       │
              │   ground truth override) │
              └────────────┬─────────────┘
                           ▼
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐   ┌───────────┐   ┌──────────┐
      │ E-Ink   │   │  Web App  │   │  Alerts  │
      │ Display │   │  (Flask)  │   │  System  │
      └─────────┘   └───────────┘   └──────────┘
```

---

## 1. Numerical Weather Models

Four global/regional models are queried via Open-Meteo, plus the NWS blended model:

| Model | Resolution | Update Cycle | Horizon | Strength |
|-------|-----------|--------------|---------|----------|
| **GFS** (Global Forecast System) | 13 km | Every 6h | 16 days | Reliable baseline, good beyond day 3 |
| **ECMWF** (European Centre) | 25 km | Every 12h | 15 days | Best overall skill globally, strongest synoptic patterns |
| **ICON** (DWD German) | 13 km | Every 12h | 7 days | Independent third perspective, good for Europe/Pacific storms |
| **HRRR** (High-Res Rapid Refresh) | 3 km | Every hour | 18h | Best 0-12h accuracy, convective-scale features |
| **NBM** (National Blend of Models) | 2.5 km | Every hour | 7 days | NWS bias-corrected blend of 31 models — highest skill for US |

### How they're used

- **GFS is the primary model** for zone forecasts (hourly timeline, day/night buckets, snow totals). It has the longest horizon and most complete parameter set.
- **ECMWF and ICON** run in parallel as independent cross-checks. Their forecasts are compared to GFS in the multi-model spread analysis.
- **HRRR** provides the highest-resolution short-term data. It's included in the Open-Meteo multi-model query and contributes to model agreement scoring.
- **NBM** is fetched separately and used as the highest-weight input for precipitation probability fusion (see Section 5).

### Multi-model spread / confidence

For each forecast day, all four models' daily snow totals are compared:

```
snow_spread = max(model_snows) - min(model_snows)
temp_spread = max(model_highs) - min(model_highs)

Confidence = High   if spread < 2" snow AND < 8°F temp AND 3+ models
             Medium if spread < 5" snow AND < 15°F temp
             Low    otherwise (or only 1-2 models have data)
```

This tells the user "all models agree on 6-8 inches" (high confidence) vs. "models range from 2 to 14 inches" (low confidence).

---

## 2. Snow Physics

Raw model output gives precipitation in millimeters of liquid water. Converting that to inches of snow on the ground requires three physics steps:

### 2a. Snow-to-Liquid Ratio (SLR)

Based on Roebber (2003) — the same research used by NWS offices. Temperature determines crystal structure:

| Temperature | SLR | Snow Type |
|-------------|-----|-----------|
| Below -18°C | 20-25:1 | Blower powder (cold smoke) |
| -18 to -12°C | 15-20:1 | Light dry powder |
| -12 to -6°C | 12-15:1 | Classic powder |
| -6 to -1°C | 8-12:1 | Packable powder |
| -1 to 0°C | 5-8:1 | Sierra cement |
| Above 0°C | 1-5:1 | Wet / slushy |

The function `compute_slr(temp_c)` is continuous and monotonically decreasing — colder temperatures always produce fluffier snow.

### 2b. Orographic Enhancement

Mountains force air upward, squeezing out more precipitation on windward slopes. The `orographic_multiplier()` function models this with three factors:

- **Wind direction alignment**: WSW flow (247.5°) is ideal for the Sierra crest. Due-east wind gets minimum enhancement. Linear interpolation between.
- **Elevation gain**: Higher zones get more enhancement. Scales from 1.0x at lake level (6225 ft) to 1.5x at 10,000 ft.
- **Wind speed**: Stronger wind = more forced uplift. Adds up to 0.3x boost for high winds.

These multiply together: a 10,000 ft peak in a strong WSW storm can get ~2x the precipitation that the base would see.

### 2c. Precipitation Type

Simple temperature threshold with a mixed zone:

```
temp_c <= -2°C  →  Snow
-2 to +1°C      →  Mix (snow counted at 50% of SLR)
above +1°C      →  Rain (no snow accumulation)
```

### 2d. Snow Quality Labels

Derived from SLR for human-readable display:

```
SLR >= 18  →  "Blower pow (cold smoke)"
SLR >= 14  →  "Light dry powder"
SLR >= 11  →  "Classic powder"
SLR >= 8   →  "Packable powder"
SLR >= 5   →  "Sierra cement"
SLR < 5    →  "Wet / slushy"
```

---

## 3. Elevation Adjustment (Lapse Rate)

Weather models report temperature at their grid elevation (typically ~1900m for the Tahoe grid cell). Actual resort zones range from 6,200 to 10,067 ft. Temperature must be adjusted.

### Three lapse rate sources, in priority order:

1. **Reno sounding** (best): Twice-daily rawinsonde from Reno (REV) measures the actual atmosphere profile. The lapse rate is computed from levels between 1500-3500m — exactly the Tahoe resort elevation range.

2. **SNOTEL regression** (good): Linear regression of temperature vs. elevation across 10 SNOTEL stations at different elevations (6,200-8,790 ft). Detects real-time inversions that models miss.

3. **Standard lapse rates** (fallback): 5.5°C/km during precipitation (moist adiabatic), 6.5°C/km otherwise (environmental average).

The lapse rate is applied by `estimate_temp_c()`:

```python
target_temp = base_temp - lapse_rate * (target_elev - base_elev)
```

**Inversions**: Tahoe frequently has temperature inversions (cold air pooling in the basin). The SNOTEL regression and sounding can return negative lapse rates, meaning it's warmer at the peaks than at lake level. This directly affects snow level calculations.

---

## 4. Observation Networks

### NWS Stations
Current conditions from the nearest staffed weather station. Provides temperature, wind, humidity, conditions text, barometric pressure. Used as the baseline observation for Tahoe (Truckee airport area) and Oakland.

### SNOTEL (10 stations)
USDA automated stations across the Tahoe basin. Each reports:
- **Snow depth** (inches) — ultrasonic depth sensor
- **SWE** (snow water equivalent) — pressure pillow
- **Temperature** — at station elevation

Used for: snowpack display, lapse rate computation, storm total tracking, and ground truth for model verification.

### CSSL (Central Sierra Snow Lab)
UC Berkeley's research station at Donner Summit (6890 ft). Accessed via California Data Exchange Center (CDEC). More granular than SNOTEL for real-time storm tracking.

### Personal Weather Stations (PWS)
Up to 5 nearby Weather Underground stations. Aggregated using **median** for temperature (robust to outlier stations) and **mean** for wind/pressure. Provides:
- Cross-check against NWS airport station
- Ground truth for "is it raining right now?" — used as an override in precipitation fusion
- Fallback for outdoor temperature when local BME280 sensor goes stale

### Upper-Air Sounding (Reno REV)
Twice-daily radiosonde (weather balloon) from Reno, accessed via Iowa State Mesonet. Provides the actual measured atmosphere profile — real lapse rate, freezing level, snow level, and moisture at every altitude. When available, sounding-derived snow and freeze levels **override** the model-derived values.

---

## 5. Precipitation Probability Fusion

For Oakland rain prediction, multiple sources are fused with a **weighted average**:

```
Source          │ Weight │ What it provides
────────────────┼────────┼──────────────────────────────────────
NBM             │  0.35  │ Max POP over next 24h (bias-corrected consensus)
NWS hourly POP  │  0.30  │ Peak probability from radar-informed forecast
Open-Meteo      │  0.20  │ Model agreement fraction → probability score
Barometer       │  0.15  │ Zambretti-derived rain probability
```

### Weights rationale

- **NBM gets the highest weight** because it's NWS's own bias-corrected blend of 31 models — specifically tuned for US locations.
- **NWS POP** is radar-informed and human-reviewed, but represents a single model chain.
- **Open-Meteo** provides independent multi-model consensus. Agreement is converted to a probability: 0 models wet = 5%, all models wet = 80%.
- **Barometer** (Zambretti algorithm) is a local early warning signal but not a calibrated probability.

### Modifiers applied after weighted average

- **Dewpoint signal**: If dewpoint approaches air temperature (depression < 3°C), probability is boosted by 8%. This is a physical moisture signal.
- **PWS ground truth**: If nearby personal weather stations report active rain, the combined probability is floored at 90%. You can't argue with current observations.

### Timing

The best timing comes from NWS hourly periods (onset = first hour with POP ≥ 20%). If the barometer suggests rain sooner, the two timing estimates are averaged. Rain end is the first NWS hour below 20% after the onset window.

### Confidence

Based on source agreement:
- **High**: 3+ sources above the 15% rain threshold
- **Medium**: 2 sources agree
- **Low**: Only 1 source (or only barometer)

---

## 6. Barometric Pressure Forecasting (Zambretti)

The local BME280 sensor feeds a **Zambretti Forecaster** — the same algorithm used by commercial weather stations (Davis Vantage, La Crosse, Oregon Scientific) since 1920.

### How it works

1. **Record** sea-level-corrected pressure every ~5 minutes (rolling 24h window)
2. **Classify** 3-hour pressure change:
   - Rapidly falling: < -6.0 hPa/3h
   - Falling: < -1.6 hPa/3h
   - Steady: -0.5 to +0.5 hPa/3h
   - Rising: > +0.5 hPa/3h
   - Rapidly rising: > +6.0 hPa/3h
3. **Compute** Zambretti Z-number from pressure + trend + season
4. **Look up** forecast text and rain probability from Z-tables

### Seasonal adjustment

Winter months (Nov-Feb) shift toward worse weather predictions. Summer months (May-Aug) shift toward better. This accounts for the Sierra's wet-winter/dry-summer climate.

### Humidity boost

High humidity (>85%) combined with falling pressure increases rain probability by up to 15 percentage points.

---

## 7. Storm Total Tracking

Multi-day storms are tracked using a state machine driven by barometric pressure:

### Storm detection

```
Storm STARTS when:  pressure < 1013 hPa AND 6h trend < -1.0 hPa
Storm ENDS when:    pressure > 1013 hPa AND 6h trend > +1.0 hPa
```

### Accumulation

When a storm starts, current SNOTEL depths are recorded as the baseline. During the storm, accumulation = max(current_depth - baseline, 0) across all stations. Peak values are tracked separately because snow depth can decrease from settling/compression even while snow is still falling.

The storm total shown to the user is the maximum peak accumulation across all SNOTEL stations.

---

## 8. Forecast Verification

A feedback loop that logs predictions vs. actuals to compute bias corrections over time.

### What's logged daily

- **Forecasts** for tomorrow: NWS high/low temp, GFS peak high, rain probability
- **Actuals** for today: observed temperature, whether it rained

### Bias correction

After 14+ days of data:
1. Group all (predicted - actual) errors by source and metric
2. Compute **median** error (robust to outliers)
3. If median bias ≥ 0.5°F, store as a correction to apply to future forecasts

Example: if GFS consistently predicts 3°F too warm for Tahoe peaks, a -3°F correction accumulates and can be applied to future GFS output.

---

## 9. Per-Zone Forecast Pipeline

Each resort zone (e.g., Heavenly Peak at 10,067 ft) goes through this pipeline:

```
Open-Meteo grid point (resort base lat/lon, ~1900m model elevation)
  │
  ├─ For each model (GFS, ECMWF, ICON, HRRR):
  │    │
  │    ├─ Lapse-rate adjust temperature to zone elevation
  │    ├─ Determine precip type (Snow / Mix / Rain)
  │    ├─ Compute SLR from adjusted temperature
  │    ├─ Apply orographic multiplier (elevation + wind direction + wind speed)
  │    ├─ Calculate snowfall: liquid_precip × SLR × orographic
  │    ├─ Compute wind chill from adjusted temp + wind
  │    └─ Classify snow quality from SLR
  │
  ├─ GFS hours → primary 48h timeline + day/night buckets
  ├─ All models → daily spread analysis (confidence rating)
  │
  └─ Output per zone:
       - Current snapshot (temp, feels-like, wind, precip type, snow quality)
       - 48-hour hourly timeline
       - Day/night accumulation buckets
       - 24h snow total
       - 7-day snow forecast
       - Multi-model spread with confidence
```

---

## 10. Data Source Summary

| Source | API | Update Freq | Used For |
|--------|-----|------------|----------|
| Open-Meteo (GFS/ECMWF/ICON/HRRR) | `api.open-meteo.com/v1/forecast` | Varies by model | Zone forecasts, snow physics, model spread |
| Open-Meteo NBM | `api.open-meteo.com` with `ncep_nbm_conus` | Hourly | Rain probability fusion (highest weight) |
| NWS Observations | `api.weather.gov/stations/{id}/observations` | Hourly | Current conditions baseline |
| NWS Forecast | `api.weather.gov/points/{lat},{lon}` | Every 6h | 7-day periods, 48h hourly, POP |
| NWS Alerts | `api.weather.gov/alerts/active` | Real-time | Watch/warning/advisory banners |
| SNOTEL (NRCS) | `wcc.sc.egov.usda.gov/awdbRestApi` | Daily | Snowpack, lapse rate, storm totals |
| CSSL (CDEC) | `cdec.water.ca.gov/dynamicapp/req/JSONDataServlet` | Daily | Donner Summit snow depth + SWE |
| Reno Sounding | `mesonet.agron.iastate.edu/json/raob.py` | Every 12h (00Z/12Z) | Measured lapse rate, freeze/snow levels |
| Climate Normals | `climate-api.open-meteo.com/v1/climate` | Static (30-year) | Temperature anomaly display |
| Weather Underground PWS | `api.weather.com/v3/location/near` | ~5 min | Ground truth temp, "is it raining?" |
| BME280 (local) | ESP32 serial/HTTP | ~5 min | Zambretti pressure forecast, humidity |
| Caltrans Chains | `cwwp2.dot.ca.gov/data/d3/cc/ccStatusD03.json` | Every 5 min | I-80/US-50 chain control status |
| Liftie.info | `liftie.info/api/resort/{name}` | ~1 min | Lift open/closed counts |
| Avalanche.org | `api.avalanche.org/v2/public/products/map-layer` | Daily | Sierra avalanche danger rating |
| NWS Reno AFD | `api.weather.gov/products/types/AFD/locations/REV` | ~4x/day | Forecaster discussion text |

All sources are free, public, and require no API keys (the WU PWS key is a well-known public widget key).
