# ML Calibration Plan: Historical Model Training for Tahoe Snow Forecasting

**Author**: Claude Code (research & plan)
**Date**: 2026-03-16
**Status**: Research complete, implementation pending

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Data Source Feasibility](#data-source-feasibility)
3. [Data Collection Pipeline](#a-data-collection-pipeline)
4. [Feature Engineering](#b-feature-engineering)
5. [Model Architecture](#c-model-architecture)
6. [Validation Strategy](#d-validation-strategy)
7. [Integration with Existing Pipeline](#e-integration-with-existing-pipeline)
8. [Differentiation Analysis](#f-differentiation-analysis)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This plan proposes training ML correction models using ~10 years of historical weather model data (Open-Meteo Historical Weather/Forecast APIs) paired with SNOTEL ground-truth observations. The goal is to learn systematic biases in NWP model snow output specific to Tahoe's terrain — elevation-dependent temperature errors, SLR miscalculations, orographic precipitation under/overestimates — and apply learned corrections to real-time forecasts.

**Key finding**: This is feasible today with freely available data. The approach mirrors what OpenSnow's PEAKS model does (38 weather variables + 7 terrain features, 42% precipitation accuracy improvement) and what the NWS MOS system has done for decades (multiple linear regression on archived model output vs. observations). Our advantage is hyperlocal: 10 SNOTEL stations spanning 6,200-8,790 ft across the Tahoe basin, paired with per-elevation-zone forecasts for 5+ resorts.

---

## Data Source Feasibility

### 1. Open-Meteo Historical Weather API (ERA5 Reanalysis)

**Verdict: Fully available and suitable**

- **URL**: `https://api.open-meteo.com/v1/archive`
- **Date range**: 1940 to present (85+ years); data from 2017 onward uses newer models with 9 km resolution
- **Resolution**: 0.25 degrees (~25 km) for ERA5, 0.1 degrees (~9 km) for ERA5-Land
- **Available variables** (all confirmed available):
  - Temperature (2m), apparent temperature, dewpoint
  - Precipitation, rain, snowfall, snow depth
  - Wind speed (10m, 100m), wind direction (10m, 100m), wind gusts
  - Relative humidity, cloud cover (total/low/mid/high)
  - Pressure (sea level, surface)
  - Soil temperature/moisture (4 depths)
  - Solar radiation, evapotranspiration
- **License**: CC BY 4.0 (free for commercial use with attribution)
- **Rate limits**: 10,000 API calls/day for non-commercial; commercial requires paid plan
- **Limitation**: This is *reanalysis* data (what the atmosphere actually did, as reconstructed), NOT archived forecasts. It is ground truth, not "what the model predicted." Useful as supplementary ground truth alongside SNOTEL, but not for training forecast-vs-observation corrections.

### 2. Open-Meteo Historical Forecast API (Archived Model Runs)

**Verdict: Available but limited date range — critical for true ML post-processing**

- **URL**: `https://api.open-meteo.com/v1/forecast` with `past_days` parameter, or dedicated Historical Forecast API
- **Date range**: 2021-2022 onward depending on model (only ~4-5 years, NOT 10)
- **Models archived**: ECMWF IFS, GFS, HRRR, DWD ICON, Meteo-France ARPEGE, and 20+ others
- **Key advantage**: This contains *what the model predicted* at initialization time — the exact data needed for training forecast correction models
- **Variables**: Same as the live forecast API (temperature, precipitation, snowfall, wind, humidity, freezing level, CAPE, etc.)
- **Why this matters**: To train "forecast - observation = bias" models, you need the forecast (from this API) and the observation (from SNOTEL). The Historical Weather API provides reanalysis (quasi-truth), not forecasts.

**Implication**: We can get ~4 winter seasons (2021/22 through 2024/25) of archived forecasts, not 10. For 10 years, we'd use ERA5 reanalysis as a proxy for "model output," which is conceptually different but still useful for learning elevation/terrain biases.

### 3. Open-Meteo Previous Runs API

- **URL**: Dedicated endpoint for accessing specific past model initialization times
- **Date range**: Generally from January 2024 onward (limited)
- **Use case**: Analyzing forecast degradation with lead time (day 1 vs day 3 vs day 5 accuracy)
- **Not suitable** for long historical training but useful for recent lead-time-dependent bias analysis

### 4. SNOTEL Historical Data (NRCS AWDB REST API)

**Verdict: Fully available, 20+ years of daily data**

- **API endpoint**: `https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data`
- **Query parameters**:
  - `stationTriplets`: e.g., `809:CA:SNTL` (station_id:state:network)
  - `elements`: `SNWD` (snow depth), `WTEQ` (SWE), `TOBS` (observed temp), `TMAX`, `TMIN`, `TAVG`, `PREC` (accumulated precip)
  - `beginDate` / `endDate`: `YYYY-MM-DD` format
  - `duration`: `DAILY`
- **Date range**: Most Tahoe SNOTEL stations have data going back to the 1980s-1990s
- **Our stations** (10 stations, already defined in `tahoe_snow.py`):

  | Station | ID | State | Elev (ft) | Lat | Lon |
  |---------|-----|-------|-----------|------|------|
  | Tahoe City Cross | 809 | CA | 6,230 | 39.17 | -120.15 |
  | Fallen Leaf | 473 | CA | 6,250 | 38.89 | -120.06 |
  | Squaw Valley GC | 784 | CA | 6,200 | 39.20 | -120.25 |
  | Ward Creek #3 | 848 | NV | 6,600 | 39.13 | -120.23 |
  | CSS Lab | 428 | CA | 6,890 | 39.33 | -120.37 |
  | Independence Camp | 540 | CA | 6,900 | 39.43 | -120.30 |
  | Independence Lake | 539 | CA | 7,000 | 39.43 | -120.30 |
  | Rubicon #2 | 724 | CA | 6,700 | 39.00 | -120.15 |
  | Hagan's Meadow | 518 | CA | 8,200 | 39.07 | -119.88 |
  | Mt Rose Ski Area | 652 | NV | 8,790 | 39.32 | -119.88 |

- **Data quality**: Generally reliable but with occasional sensor failures, snow bridging artifacts (depth readings plateau when snow bridges form over the sensor), and maintenance gaps. Will need quality-control filters.

### 5. ERA5 Reanalysis (Direct Access)

**Verdict: Available but Open-Meteo wraps it more conveniently**

- **Source**: Copernicus Climate Change Service (ECMWF)
- **Resolution**: 0.25 degrees (~31 km), hourly, 137 vertical levels
- **Date range**: January 1940 to present (5-day latency)
- **Format**: NetCDF-4 / GRIB (large files, complex tooling)
- **Key variables**: Full atmospheric state including snow depth, snow density, 2D optimal interpolation
- **Access**: Free via Copernicus Climate Data Store (registration required), also on AWS Open Data (NSF NCAR curated)
- **Recommendation**: Use Open-Meteo's Historical Weather API instead — it wraps ERA5 with simpler JSON access. Only go direct to ERA5 if we need vertical profile data or variables Open-Meteo doesn't expose.

### 6. NWS Model Output Statistics (MOS)

**Verdict: Foundational concept, same approach we're implementing**

MOS is the NWS's decades-old system for the exact same thing we're doing:
- **Method**: Multiple linear regression relating archived NWP model forecast fields to surface observations
- **Training**: Uses multi-year archives of model output paired with station observations
- **Output**: Bias corrections per station, per lead time, per variable
- **Variables corrected**: Temperature, wind, precipitation probability, cloud cover, visibility
- **Key insight**: MOS equations are retrained whenever the underlying model changes (GFS version updates invalidate old corrections). We face the same issue — model retraining will be needed when Open-Meteo upstream models update.

Our approach extends MOS by using gradient-boosted trees (nonlinear) instead of linear regression, and by incorporating terrain/physics features that MOS doesn't use (SLR, orographic multiplier, Froude number).

### 7. OpenSnow PEAKS Model

**Verdict: Our primary competitor and conceptual template**

- **Architecture**: Proprietary ML model (exact architecture undisclosed, likely gradient-boosted trees or a lightweight neural network)
- **Inputs**: 38 dynamic weather variables + 7 static surface/terrain variables
- **Output**: Precipitation, temperature, and wind on a 3 km grid (vs input models at 10-25 km)
- **Training data**: Historical storm patterns (exact duration undisclosed, likely 5-10 years of archived forecasts)
- **Accuracy gains** (vs raw NWP):
  - Precipitation: +42% (MSE reduction)
  - Temperature: +82%
  - Wind: +72%
- **Key technique**: Learns "how mountains impact weather" — terrain-aware downscaling from coarse NWP grid to fine resolution
- **Development time**: 18 months, extensive testing before launch (December 2025)

---

## A. Data Collection Pipeline

### Architecture

```
historical_data_collector.py
├── fetch_open_meteo_historical()     # ERA5 reanalysis at each resort zone lat/lon
├── fetch_open_meteo_forecast_archive()  # Archived model forecasts (2021+)
├── fetch_snotel_historical()          # Daily SNWD, WTEQ, TOBS for all 10 stations
├── align_and_merge()                  # Temporal alignment, gap-filling
└── save_training_dataset()            # Parquet/CSV output
```

### Open-Meteo Historical Data Collection

Fetch ERA5 reanalysis at each resort zone lat/lon point (3 zones x 5 resorts = 15 points):

```python
# Example: 10 years of hourly data for Heavenly Peak
import requests

params = {
    "latitude": 38.9280,
    "longitude": -119.9070,
    "start_date": "2016-03-16",
    "end_date": "2026-03-16",
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "apparent_temperature", "precipitation", "rain", "snowfall",
        "snow_depth", "weather_code", "pressure_msl", "surface_pressure",
        "cloud_cover", "wind_speed_10m", "wind_direction_10m",
        "wind_gusts_10m", "wind_speed_100m", "wind_direction_100m",
        # Missing from reanalysis: freezing_level_height, cape, visibility
    ],
    "daily": [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "rain_sum", "snowfall_sum",
        "wind_speed_10m_max", "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
    ],
    "timezone": "America/Los_Angeles",
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
}
resp = requests.get("https://api.open-meteo.com/v1/archive", params=params)
```

### Open-Meteo Archived Forecast Collection (2021+)

For true forecast-vs-observation training, fetch archived model predictions:

```python
# Archived GFS/ECMWF forecasts for comparison against SNOTEL
params = {
    "latitude": 38.9280,
    "longitude": -119.9070,
    "start_date": "2021-10-01",
    "end_date": "2026-03-16",
    "hourly": [
        "temperature_2m", "precipitation", "snowfall",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "relative_humidity_2m", "dew_point_2m", "pressure_msl",
        "cloud_cover", "freezing_level_height", "cape",
        "snow_depth", "weather_code", "visibility",
    ],
    "models": ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"],
    "timezone": "America/Los_Angeles",
}
# Use the Historical Forecast API endpoint
resp = requests.get(
    "https://historical-forecast-api.open-meteo.com/v1/forecast",
    params=params
)
```

### SNOTEL Historical Data Collection

Fetch 10 years of daily data for all 10 stations:

```python
def fetch_snotel_10yr(station_id, state):
    """Fetch 10 years of daily SNOTEL data."""
    resp = requests.get(
        "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
        params={
            "stationTriplets": f"{station_id}:{state}:SNTL",
            "elements": "SNWD,WTEQ,TOBS,TMAX,TMIN,PREC",
            "beginDate": "2016-03-16",
            "endDate": "2026-03-16",
            "duration": "DAILY",
        },
        timeout=60,
    )
    return resp.json()
```

### Storage Requirements Estimate

| Dataset | Points | Variables | Duration | Hourly Rows | Est. Size |
|---------|--------|-----------|----------|-------------|-----------|
| Open-Meteo ERA5 (15 zones) | 15 lat/lon | 17 hourly vars | 10 years | 15 x 87,600 = 1.31M | ~250 MB CSV |
| Open-Meteo Archived Forecasts (15 zones x 3 models) | 45 series | 15 hourly vars | 4 years | 45 x 35,040 = 1.58M | ~350 MB CSV |
| SNOTEL daily (10 stations) | 10 stations | 6 elements | 10 years | 10 x 3,650 = 36,500 | ~5 MB CSV |
| **Total** | | | | | **~605 MB** |

In compressed Parquet format, this would be approximately **80-120 MB** total. Entirely manageable.

### Data Collection Rate Limiting

Open-Meteo allows 10,000 API calls/day for non-commercial use. Our collection strategy:

- **ERA5 historical**: 15 points x 1 call each (date range can span 10 years per call) = 15 calls
- **Archived forecasts**: 15 points x 3 models x possible date-range chunking = ~90 calls
- **SNOTEL**: 10 stations x 1 call each = 10 calls
- **Total**: ~115 API calls (well within daily limits, collectible in a single run)

---

## B. Feature Engineering

### Raw Model Outputs (Predictors)

These come from Open-Meteo archived forecasts, evaluated at each resort zone:

| Feature | Description | Source |
|---------|-------------|--------|
| `temp_2m_f` | 2m temperature (F) at model grid | Open-Meteo |
| `temp_apparent_f` | Apparent temperature (F) | Open-Meteo |
| `precip_in` | Precipitation (inches) | Open-Meteo |
| `snowfall_cm` | Model-predicted snowfall (cm) | Open-Meteo |
| `snow_depth_m` | Model snow depth (m) | Open-Meteo |
| `wind_speed_mph` | 10m wind speed (mph) | Open-Meteo |
| `wind_dir_deg` | 10m wind direction (degrees) | Open-Meteo |
| `wind_gust_mph` | 10m wind gusts (mph) | Open-Meteo |
| `wind_100m_mph` | 100m wind speed (ridge-level proxy) | Open-Meteo |
| `rh_pct` | Relative humidity (%) | Open-Meteo |
| `dewpoint_f` | 2m dewpoint (F) | Open-Meteo |
| `pressure_msl_hpa` | Mean sea level pressure (hPa) | Open-Meteo |
| `cloud_cover_pct` | Total cloud cover (%) | Open-Meteo |
| `freezing_level_ft` | Freezing level height (ft) | Open-Meteo |
| `cape_jkg` | CAPE (J/kg) | Open-Meteo |
| `weather_code` | WMO weather code | Open-Meteo |
| `visibility_mi` | Visibility (miles) | Open-Meteo |

### Derived Physics Features (Calculated)

These use the same functions already in `tahoe_snow.py`:

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `slr` | Snow-to-liquid ratio | `compute_slr(temp_c, wind_mph, rh_pct)` |
| `orographic_mult` | Orographic enhancement factor | `orographic_multiplier(elev_ft, wind_mph, wind_dir, cape, froude)` |
| `lapse_rate_c_km` | Temperature lapse rate | Derived from multi-elevation temp gradient |
| `froude_number` | Flow blocking indicator | `compute_froude_number(wind_mps, N, barrier_height)` |
| `wet_bulb_c` | Wet-bulb temperature | For rain/snow discrimination at the boundary |
| `precip_type` | Rain/Snow/Mix classification | `precip_type(temp_c, is_precip, rh_pct)` |
| `inversion_present` | Temperature inversion flag | `1 if lapse_rate < 3.0 else 0` |
| `wind_alignment` | Alignment with optimal orographic axis (WSW=247.5) | `cos(wind_dir - 247.5)` |
| `temp_elev_adjusted` | Temperature at target elevation | `estimate_temp_c()` with lapse rate |
| `precip_orographic_in` | Orographic-adjusted precip | `precip_in * orographic_multiplier` |

### Spatial/Terrain Features (Static)

| Feature | Description | Value Type |
|---------|-------------|------------|
| `elevation_ft` | Zone elevation | Static per zone |
| `aspect_deg` | Slope aspect | Static per zone |
| `slope_angle_deg` | Slope steepness | Static per zone |
| `dist_to_crest_km` | Distance from Sierra crest | Static per zone |
| `east_shore` | East shore of Tahoe flag | Boolean per resort |
| `barrier_relative_elev` | Elevation relative to Sierra crest | Static |

### Calendar/Temporal Features

| Feature | Description |
|---------|-------------|
| `month` | Month (1-12) |
| `day_of_season` | Days since October 1 (water year) |
| `hour_utc` | Hour of day (for diurnal cycle) |
| `is_weekend` | Weekend flag (not meteorologically relevant, but useful for verification data quality) |

### Lagged/Preceding Condition Features

| Feature | Description | Lag |
|---------|-------------|-----|
| `snowpack_depth_in` | Current SNOTEL snow depth | Day-1 |
| `swe_in` | Current SWE | Day-1 |
| `temp_trend_24h` | 24h temperature change | Computed |
| `precip_48h_prior_in` | Accumulation in prior 48h | Computed |
| `pressure_change_12h` | Pressure tendency (hPa/12h) | Computed |
| `depth_change_3d_in` | Snow depth change over prior 3 days | Computed |
| `swe_change_3d_in` | SWE change over prior 3 days | Computed |

### Target Variables

| Target | Description | Source |
|--------|-------------|--------|
| `snotel_depth_change_in` | Daily change in SNOTEL snow depth (primary) | SNOTEL SNWD delta |
| `snotel_swe_change_in` | Daily change in SWE (better for total precip) | SNOTEL WTEQ delta |
| `snotel_tmax_f` | Daily max temperature | SNOTEL TMAX |
| `snotel_tmin_f` | Daily min temperature | SNOTEL TMIN |

**Why depth change and SWE change, not absolute depth?**

- Absolute depth is dominated by compaction, melt, and wind redistribution — not just new snowfall
- Daily depth change captures new accumulation minus settling
- SWE change captures true water equivalent of new precipitation
- Effective SLR can be back-calculated: `observed_SLR = depth_change / swe_change` (when both are positive)

**Caveat**: SNOTEL depth change is noisy. A depth increase of 6" at a station does not mean it snowed 6" at a resort 5 miles away and 2,000 feet higher. The ML model must learn the elevation/spatial transfer function.

### Total Feature Count

- Raw model outputs: ~17
- Per-model (x3 models = GFS, ECMWF, ICON): ~51
- Derived physics: ~10
- Terrain/static: ~6
- Calendar: ~4
- Lagged conditions: ~7
- **Total: ~78 features** (comparable to PEAKS' 38 dynamic + 7 static = 45)

---

## C. Model Architecture

### Primary: XGBoost/LightGBM (Gradient-Boosted Trees)

Already scaffolded in `ml_pipeline.py`. Gradient-boosted trees are the standard for this task:

- **Handles missing data natively** (critical — SNOTEL sensors fail, API calls timeout)
- **Handles mixed feature types** (continuous, categorical, boolean)
- **Feature importance is interpretable** (SHAP values, gain-based importance)
- **Fast training** on ~10K-50K samples
- **No GPU required** (unlike deep learning)
- **Well-studied for NWP post-processing** (WRF/XGBoost reduced RMSE by 10.34% in Sciencedirect study; random forest explained nearly half of SLR variability vs. one quarter for operational models)

**Hyperparameters** (starting point, tune via Bayesian optimization):

```python
xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,       # L1 regularization
    reg_lambda=1.0,       # L2 regularization
    min_child_weight=5,   # Prevent overfitting to rare events
    gamma=0.1,            # Minimum loss reduction for split
    random_state=42,
)
```

### Model Granularity: Per-Elevation-Band, Per-Target

The existing `ml_pipeline.py` already structures models as `(elevation_band, lead_time)`. Expand to:

| Dimension | Options | Rationale |
|-----------|---------|-----------|
| Elevation band | base (6000-7000ft), mid (7000-8500ft), peak (8500-11000ft) | Snow physics differ dramatically by elevation |
| Target variable | depth_change, swe_change, temp_high, temp_low | Each has different bias patterns |
| Lead time | Day 1 (0-24h), Day 2 (24-48h), Day 3 (48-72h) | Forecast skill degrades with lead time |

This gives **3 bands x 4 targets x 3 lead times = 36 models**. Each model trains on data from ALL stations in that elevation band (pooling Fallen Leaf + Tahoe City Cross + Squaw Valley for "base" band, for example).

**Alternative: Per-station models** — Rejected. With only ~150 storm days per year x 10 years = 1,500 samples per station, individual station models would overfit. Pooling by elevation band gives 5-7 stations per band = 7,500-10,500 samples per band.

### Fallback: Bias Correction Lookup Table

Before the full ML model is ready, implement a simpler bias correction table:

```python
# Structure: bias_table[month][wind_direction_bin][temp_bin] = correction_factor
# Wind directions: N, NE, E, SE, S, SW, W, NW (8 bins)
# Temp bins: <20F, 20-28F, 28-32F, 32-36F, >36F (5 bins)
# Months: Oct-May (8 months)
# = 8 x 5 x 8 = 320 cells per elevation band
```

This can be populated from the first pass of historical data analysis. It won't capture nonlinear interactions but will catch the most common systematic biases (e.g., "GFS consistently overpredicts snowfall by 25% in SW flow above 8,000 ft in January").

### Deep Learning Option (Future)

If gradient-boosted trees show ceiling effects, consider a lightweight neural network:

- **U-Net or ConvLSTM**: For spatial downscaling (learning terrain effects across multiple grid points)
- **Training data requirement**: Would need 50K+ samples (5+ years of hourly data)
- **Complexity**: Significantly harder to deploy on Raspberry Pi; would need a server-side inference endpoint
- **Recommendation**: Start with XGBoost. Upgrade to neural network only if XGBoost plateaus below target accuracy.

---

## D. Validation Strategy

### Train/Validate/Test Split

**Time-based split** (never random — weather data is autocorrelated):

| Split | Years | Seasons | Purpose |
|-------|-------|---------|---------|
| **Train** | 2016-2023 | 7 complete winters | Learn bias patterns |
| **Validate** | 2023-2024 | 1 winter | Hyperparameter tuning, early stopping |
| **Test** | 2024-2025 | 1 winter (most recent) | Final skill assessment |

If using only archived forecasts (2021+), the split becomes:

| Split | Years | Seasons | Purpose |
|-------|-------|---------|---------|
| **Train** | 2021-2024 | 3 winters | Learn bias patterns |
| **Validate** | 2024-2025 | 0.5-1 winter | Hyperparameter tuning |
| **Test** | 2025-2026 (current) | Ongoing | Live verification |

### Cross-Validation: Leave-One-Season-Out (LOSO)

Standard k-fold cross-validation would leak information across correlated storm events. Instead:

```
Fold 1: Train on 2017-2024, validate on 2016/17 season
Fold 2: Train on 2016-2017 + 2018-2024, validate on 2017/18 season
...
Fold 8: Train on 2016-2023, validate on 2023/24 season
```

This tests whether the model generalizes to unseen *seasons* (different large-scale climate patterns, ENSO phases, etc.), not just unseen *days*.

### Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| MAE (Mean Absolute Error) | Average error magnitude in inches | < raw NWP MAE |
| RMSE (Root Mean Square Error) | Penalizes large errors | < raw NWP RMSE |
| Bias | Systematic over/under-prediction | Near zero |
| CRPS (Continuous Ranked Probability Score) | Probabilistic calibration | Lower = better |
| Hit rate (>6" storms) | Did we correctly predict significant events? | > 80% |
| False alarm rate (>6" predicted, <2" observed) | Crying wolf | < 20% |
| Conditional bias by wind direction | Does bias change with flow regime? | Uniform across bins |
| Conditional bias by elevation | Does bias change with altitude? | Uniform across bands |

### Overfitting Prevention

1. **Regularization**: L1/L2 regularization in XGBoost (already configured)
2. **Early stopping**: Monitor validation loss, stop when it stops improving
3. **Min samples per leaf**: `min_child_weight=5` prevents fitting to single-event patterns
4. **Feature selection**: Use recursive feature elimination to remove low-importance features
5. **Physical constraints**: Post-process predictions to enforce physical bounds (no negative snowfall, SLR between 1 and 30, temperature monotonically decreasing with elevation)
6. **Ensemble of models**: Train 5 models with different random seeds, average predictions (reduces variance)
7. **Climate regime awareness**: Include ENSO index (ONI) as a feature to account for year-to-year variability rather than memorizing specific winter patterns

### Specific Overfitting Risks

- **AR (atmospheric river) storms**: A few massive storms dominate total seasonal snowfall. The model could overweight features that correlate with these rare events. Mitigation: cap sample weights, use quantile regression for extreme events.
- **Season-specific snow levels**: 2024 had unusually low snow levels due to El Nino. The model might learn that "March = rain at base" when the actual pattern is ENSO-dependent. Mitigation: include ENSO state as a feature.
- **SNOTEL-specific quirks**: Each station has unique wind exposure, forest canopy, and sensor characteristics. Mitigation: include station-specific features (elevation, aspect) rather than station ID.

---

## E. Integration with Existing Pipeline

### Current Architecture

```
tahoe_snow.py::analyze_all()
├── fetch_open_meteo_multi()     → raw NWP forecasts per model
├── parse_open_meteo()           → elevation-adjusted, SLR/orographic applied
├── skill_weighted_blend()       → multi-model weighted average
├── fetch_snotel()               → current ground truth
└── forecast_verification.py     → tracks prediction vs observation (ongoing)
```

### Proposed Integration (ML corrections supplement, not replace, existing physics)

```
tahoe_snow.py::analyze_all()
├── fetch_open_meteo_multi()     → raw NWP forecasts per model
├── parse_open_meteo()           → elevation-adjusted, SLR/orographic applied
├── skill_weighted_blend()       → multi-model weighted average
├── ml_pipeline.apply_corrections()  ← NEW: ML post-processing
│   ├── Extract features from current forecast state
│   ├── Run XGBoost prediction for each zone/lead-time
│   ├── Apply correction factors to blended forecast
│   └── Tag hours with ml_corrected=True, ml_confidence
├── fetch_snotel()               → current ground truth
└── forecast_verification.py     → now also verifies ML-corrected output
```

### Key Design Decisions

**1. ML corrections are multiplicative, not replacement**

The ML model predicts a correction factor (e.g., 0.85 = reduce by 15%), not an absolute snow amount. This preserves the physical structure of the forecast (timing, spatial pattern) while correcting systematic biases.

```python
# In ml_pipeline.py::apply_corrections()
correction = model.predict(features)  # e.g., 0.85
hour["snowfall_in"] = round(hour["snowfall_in"] * max(0.1, correction), 1)
hour["ml_corrected"] = True
hour["ml_correction_factor"] = round(correction, 3)
```

**2. ML corrections are gated by confidence**

When the ML model's prediction uncertainty is high (ensemble of 5 models disagrees), fall back to uncorrected output:

```python
corrections = [m.predict(features) for m in model_ensemble]
correction_std = np.std(corrections)
if correction_std < 0.15:  # Models agree within 15%
    apply_correction(np.mean(corrections))
else:
    # Low confidence — don't apply ML correction
    hour["ml_corrected"] = False
    hour["ml_confidence"] = "low"
```

**3. Relationship to `skill_weighted_blend()`**

The skill-weighted blend adjusts model *weights* based on recent verified performance. ML corrections adjust the *output* of the blended forecast based on learned elevation/terrain/flow-regime biases. They are complementary:

- `skill_weighted_blend()`: "ECMWF has been better than GFS this week, weight it higher"
- `ml_pipeline.apply_corrections()`: "In SW flow at 9,000ft, the blended forecast overestimates snowfall by 20%"

Both should remain active. The skill-weighted blend handles short-term model skill drift; ML corrections handle systematic terrain biases.

**4. Display integration**

On the e-ink display and web dashboard, indicate when ML corrections are active:

- Small "ML" badge next to corrected snow totals
- Tooltip/detail showing correction factor and confidence
- Feature importances shown in dashboard status panel (already supported by `MLPipeline.get_status()`)

---

## F. Differentiation Analysis

### vs. OpenSnow PEAKS

| Dimension | OpenSnow PEAKS | Tahoe Snow ML |
|-----------|----------------|---------------|
| **Coverage** | Global mountains (3 km grid) | Tahoe basin only (hyperlocal) |
| **Training data** | Undisclosed (likely 5-10 years, many stations) | 10 SNOTEL stations, 10 years |
| **Resolution** | 3 km output grid | Per-resort-zone (3 zones per resort) |
| **Variables predicted** | Precip, temp, wind | Snow depth change, SWE change, temp |
| **Input features** | 38 dynamic + 7 static = 45 | ~78 features (more physics-derived) |
| **Terrain handling** | Statistical (learned from data) | Physics + statistical (SLR, orographic multiplier, Froude as features) |
| **Transparency** | Black box | Open source, feature importances published |
| **Pricing** | Subscription ($30/yr All-Access) | Free |
| **Update frequency** | Unknown | Weekly retraining with expanding window |

### Our Unique Advantages

1. **Physics-informed features**: We compute SLR, orographic multiplier, Froude number, lapse rate, and wet-bulb temperature as explicit features. PEAKS likely learns these implicitly, but encoding domain knowledge as features should improve sample efficiency (better accuracy with less training data).

2. **Hyperlocal calibration**: PEAKS covers all mountains globally at 3 km. We calibrate specifically to 10 SNOTEL stations in the Tahoe basin. For Tahoe users, our calibration will be tighter.

3. **Explainability**: XGBoost provides feature importance rankings and SHAP values. We can show users *why* the forecast was adjusted: "Model correction driven by: SW wind alignment (35%), freezing level below 7000ft (22%), high ensemble spread (18%)."

4. **Open source transparency**: Users can inspect the model, its training data, and its correction logic. Trust through transparency.

5. **Integrated verification loop**: Our `forecast_verification.py` already tracks predictions vs observations. ML corrections feed back into this loop — we can show users whether ML corrections are actually improving accuracy.

### Feature Importances as a Marketing Asset

Publishing feature importance rankings serves dual purposes:

1. **Trust building**: "Our model found that wind direction and freezing level are the two most important factors for Tahoe snowfall accuracy" — this validates our approach to meteorology-savvy users.

2. **Content marketing**: Blog posts like "What Really Drives Tahoe Snow Forecasting Accuracy" with SHAP beeswarm plots would generate engagement among the skiing community.

3. **Competitive moat documentation**: By publishing our methodology openly, we establish intellectual credibility that proprietary competitors (OpenSnow) cannot match because their methods are trade secrets.

### What We Cannot Match

- **PEAKS' global coverage**: We are Tahoe-only. Expanding to other ranges would require significant effort per range (identifying SNOTEL stations, tuning orographic parameters).
- **PEAKS' 18 months of dedicated development**: OpenSnow had a funded team working full-time. Our approach must be pragmatic and incremental.
- **PEAKS' spatial coherence**: PEAKS outputs a continuous 3 km grid; we output point forecasts per zone. Users comparing neighboring zones might see inconsistencies.

---

## Implementation Roadmap

### Phase 1: Data Collection (1-2 weeks)

1. Implement `historical_data_collector.py`:
   - Fetch 10 years of ERA5 reanalysis via Open-Meteo Historical Weather API for all 15 resort zone points
   - Fetch 4 years of archived GFS/ECMWF/ICON forecasts via Historical Forecast API
   - Fetch 10 years of daily SNOTEL data for all 10 stations
   - Save as Parquet files in `data/historical/`
2. Implement data quality control:
   - Filter SNOTEL snow-bridging artifacts (depth plateau while SWE increases)
   - Interpolate small gaps (1-2 days)
   - Flag and exclude large gaps (sensor failures)
3. Verify data completeness: at least 80% coverage per station per winter

### Phase 2: Feature Engineering & EDA (1-2 weeks)

1. Implement feature computation pipeline:
   - Compute all derived physics features (SLR, orographic multiplier, etc.)
   - Compute lagged features (prior 24h/48h/72h conditions)
   - Compute calendar features
   - Align model output timestamps with SNOTEL daily observations
2. Exploratory analysis:
   - Distribution of forecast errors by elevation, wind direction, temperature
   - Seasonal bias patterns (is the model worse in early vs. late season?)
   - Inter-model disagreement patterns (when do GFS and ECMWF diverge most?)
   - SNOTEL station correlation analysis (which stations co-vary?)

### Phase 3: Model Training (1-2 weeks)

1. Train XGBoost models per (elevation_band, target, lead_time)
2. Implement leave-one-season-out cross-validation
3. Tune hyperparameters via Bayesian optimization (optuna)
4. Implement bias correction lookup table as fallback
5. Evaluate against raw NWP and current skill-weighted-blend output
6. Generate feature importance rankings and SHAP analysis

### Phase 4: Integration & Testing (1 week)

1. Activate XGBoost code in `ml_pipeline.py` (currently commented out)
2. Wire `ml_pipeline.apply_corrections()` into `analyze_all()`
3. Add ML correction indicators to e-ink display and web dashboard
4. Run parallel evaluation: corrected vs uncorrected for 2-4 weeks
5. Verify no regressions in existing forecast quality

### Phase 5: Ongoing Operations

1. Weekly retraining with expanding window (cron job or manual trigger)
2. Monitor correction factor distributions (alert if corrections become extreme)
3. Retrain from scratch when upstream models change (GFS/ECMWF version updates)
4. Publish monthly accuracy reports comparing ML-corrected vs raw output
5. Publish feature importance analysis as content (blog/README)

---

## Academic References

Key papers supporting this approach:

1. **Hoopes et al. (2023)** — "Improving prediction of mountain snowfall in the southwestern United States using machine learning methods" (Meteorological Applications). Random forest model for SLR prediction; temperature in 850-700 hPa layer and humidity are top predictors. [Wiley](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/met.2153)

2. **Hieta et al. (2025)** — "Operational Machine Learning Post-Processing of Short-Range Temperature, Humidity, Wind Speed and Gust Forecasts" (Meteorological Applications). Operational ML post-processing reduces RMSE 24-29% vs raw NWP. [Wiley](https://rmets.onlinelibrary.wiley.com/doi/10.1002/met.70074)

3. **Integrated WRF/XGBoost snowfall study** — Atmospheric variables from WRF fed to XGBoost reduced RMSE by 10.34%; RF reduced RMSE by 9.72%, with correlation improvements of 11-12%. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0022169424015099)

4. **Scheuerer & Hamill (2023)** — "Robust weather-adaptive post-processing using model output statistics random forests" (NPG). MOS-RF: random forest extension of classical MOS. [Copernicus](https://npg.copernicus.org/articles/30/503/2023/)

5. **Grönquist et al. (2020)** — "Deep learning for post-processing ensemble weather forecasts" (Phil. Trans. Royal Society A). Neural networks for ensemble calibration. [Royal Society](https://royalsocietypublishing.org/doi/10.1098/rsta.2020.0092)

6. **Duan et al. (2024)** — "Using Temporal Deep Learning Models to Estimate Daily Snow Water Equivalent Over the Rocky Mountains" (Water Resources Research). LSTM models for SWE estimation. [AGU](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR035009)

7. **Rain-snow partitioning ML study (2025)** — "Machine learning shows a limit to rain-snow partitioning accuracy when using near-surface meteorology" (Nature Communications). Fundamental limits of predicting rain vs snow from surface obs. [Nature](https://www.nature.com/articles/s41467-025-58234-2)

---

## Web Sources

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
- [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
- [Open-Meteo Pricing](https://open-meteo.com/en/pricing)
- [SNOTEL Historic Data](https://wcc.sc.egov.usda.gov/nwcc/tabget)
- [NRCS AWDB REST API](https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui.html)
- [ERA5 (ECMWF)](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- [ERA5 on AWS Open Data](https://registry.opendata.aws/nsf-ncar-era5/)
- [OpenSnow PEAKS Model Announcement](https://opensnow.com/news/post/peaks-ai-model)
- [OpenSnow PEAKS 42% Accuracy (SnowBrains)](https://snowbrains.com/opensnows-new-peaks-a-i-model-makes-42-more-accurate-forecasts/)
- [PEAKS Support Guide](https://support.opensnow.com/feature-guides/peaks-model)
- [NWS Model Output Statistics (MOS)](https://vlab.noaa.gov/web/mdl/mos)
- [MOS Wikipedia](https://en.wikipedia.org/wiki/Model_output_statistics)
- [NWS MOS Guidance Paper](https://www.weather.gov/media/publications/front/06june-front.pdf)
