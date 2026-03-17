# ML Learning Guide: Weather Forecast Correction with XGBoost

**Built alongside the Tahoe Snow ML pipeline — every concept maps to real code.**

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Data Collection — Know Your Sources](#2-data-collection)
3. [Feature Engineering — The 80% Rule](#3-feature-engineering)
4. [Model Training — Gradient Boosted Trees](#4-model-training)
5. [Cross-Validation — Don't Cheat](#5-cross-validation)
6. [Evaluation — Beyond Accuracy](#6-evaluation)
7. [SHAP — Opening the Black Box](#7-shap)
8. [Deployment — Making It Real](#8-deployment)
9. [What to Study Next](#9-next-steps)

---

## 1. The Big Picture

### What We're Doing

Weather models (GFS, ECMWF, ICON) predict snow at ~10-25 km grid resolution. But the Sierra Nevada has peaks, valleys, and ridges that change weather within 1 km. Our ML model learns to correct these predictions using 10 years of historical data.

### The ML Post-Processing Pattern

```
Raw NWP Forecast ──→ ML Correction ──→ Better Forecast
     (input)           (our model)         (output)

"GFS says 8"        "In SW flow at     "Corrected:
 of snow"            9000ft, GFS         6.5" of snow"
                     overpredicts
                     by ~20%"
```

This is called **Model Output Statistics (MOS)** — the NWS has done this since the 1970s with linear regression. We use gradient-boosted trees instead, which can learn nonlinear patterns.

### The Code

| File | Purpose |
|------|---------|
| `historical_data_collector.py` | Fetches training data (Step 1) |
| `feature_engineering.py` | Creates features from raw data (Step 2) |
| `train_models.py` | Trains XGBoost models (Step 3) |
| `evaluate_models.py` | Evaluates and interprets models (Step 4) |
| `ml_corrections.py` | Applies corrections to live forecasts (Step 5) |

---

## 2. Data Collection

### Key Concept: Training Pairs

Supervised ML needs pairs: **(input, correct answer)**

| Input (Features) | Correct Answer (Target) |
|---|---|
| What the model predicted | What actually happened |
| ERA5/GFS temp forecast | SNOTEL observed temperature |
| ERA5/GFS snow forecast | SNOTEL measured depth change |

### Our Three Data Sources

1. **SNOTEL** (ground truth): 10 automated stations measuring snow depth, SWE, and temperature since 2016. This is our answer key.

2. **ERA5 Reanalysis**: ECMWF's reconstruction of what the atmosphere actually did. Like having a perfect weather model that can see the past.

3. **Archived Forecasts**: What GFS/ECMWF/ICON actually predicted at the time. This is the key to learning model-specific biases.

### Data Quality Rule

> **Garbage in, garbage out.** If your training data has errors, your model learns errors.

Our quality control (`_snotel_quality_control()`) catches:
- Snow bridging artifacts (snow bridges over the sensor)
- Impossible values (negative snow depth, 500+ inches)
- Sensor failures (huge day-to-day jumps)

See: `historical_data_collector.py:175-230`

### Try It

```bash
python historical_data_collector.py --snotel    # ~30 seconds
python historical_data_collector.py --era5      # ~5 minutes
python historical_data_collector.py --check     # See what you have
```

---

## 3. Feature Engineering

### Key Concept: Features Are Clues

A feature is a number that helps the model make predictions. Better features = better predictions, regardless of how fancy your model is.

### Three Feature Categories

**Raw Features** — directly from the data:
```python
temperature_2m_max   # Today's high temperature
wind_speed_10m_max   # Maximum wind speed
precipitation_sum    # Total precipitation
```

**Derived Physics Features** — encoding domain knowledge:
```python
slr                  # Snow-to-Liquid Ratio (how fluffy is the snow?)
orographic_mult      # How much does the mountain enhance precipitation?
froude_number        # Does air flow over or around the mountain?
wet_bulb_f           # Will precipitation fall as rain or snow?
lapse_rate_c_km      # How stable is the atmosphere?
```

**Temporal Features** — what happened recently:
```python
depth_change_3d      # Snow depth change over last 3 days
swe_change_7d        # SWE change over last 7 days
tmax_lag1d           # Yesterday's high temperature
month                # Time of year (early vs late season)
```

### Why Physics Features Matter

Without `slr` as a feature, the model has to learn from scratch that:
- At 25°F with light winds → fluffy snow (SLR ~15:1)
- At 33°F with strong winds → dense snow (SLR ~6:1)

With `slr` as a feature, we're telling it: "Here's the relationship. Use it."

This is called **physics-informed machine learning** and it dramatically reduces the amount of training data needed.

### The Money Feature: Orographic Multiplier

```python
# From feature_engineering.py
def compute_orographic_multiplier(elev_ft, wind_mph, wind_dir_deg, aspect_deg):
    alignment = cos(wind_dir - optimal_axis)  # WSW = 247.5° for Sierra
    if alignment > 0:  # Windward
        mult = 1.0 + 0.3 * alignment * (wind_mph / 30)
    else:  # Leeward (rain shadow)
        mult = 1.0 + 0.15 * alignment
```

This single feature encodes: "When wind blows from the southwest into the Sierra, high elevations get more precipitation than the model predicts."

### Try It

```bash
python feature_engineering.py   # Builds training_data.parquet + prints EDA
```

Look at the EDA output. Key things to notice:
- **Correlations**: Which features correlate with snow depth change?
- **Class imbalance**: Only 15.7% of days are snow days
- **Seasonal patterns**: January has the most snow; April is melting

---

## 4. Model Training

### Key Concept: Gradient Boosted Trees

XGBoost builds a sequence of decision trees, where each tree corrects the mistakes of all previous trees.

```
Tree 1: "If snowfall_sum > 3 cm and temp < 30°F → predict 4" of new snow"
         Error: off by 2" for many samples

Tree 2: "For samples where Tree 1 was too high AND wind > 20 mph → subtract 1.5""
         Remaining error: smaller

Tree 3: "For samples where Trees 1+2 were too low AND elevation > 8000ft → add 0.8""
         Remaining error: even smaller

... repeat 100-300 times
```

### Key Hyperparameters

```python
xgb.XGBRegressor(
    n_estimators=300,      # Number of trees (more = better, but slower)
    max_depth=5,           # How complex each tree is (deeper = more complex)
    learning_rate=0.05,    # How aggressively each tree corrects (smaller = safer)
    subsample=0.8,         # Fraction of data used per tree (randomness helps!)
    colsample_bytree=0.7,  # Fraction of features per tree
    reg_alpha=0.1,         # L1 regularization (makes model simpler)
    reg_lambda=1.0,        # L2 regularization (prevents extreme weights)
    min_child_weight=5,    # Minimum samples per leaf (prevents overfitting to rare events)
)
```

**Learning rate** is the most important one to understand: it controls the tradeoff between learning speed and generalization. Too high → overfits to training data. Too low → needs too many trees.

### Per-Elevation-Band Models

We train separate models for base (<7000ft), mid (7000-8500ft), and peak (>8500ft) because the physics differ:

- **Base**: Often near the rain/snow line. Temperature is the dominant predictor.
- **Mid**: Best snow accumulation zone. Wind direction (orographic enhancement) matters most.
- **Peak**: Wind scouring can remove snow. Wind speed is a key predictor.

### Try It

```bash
python train_models.py --quick     # ~25 seconds, 9 models
python train_models.py             # ~1 minute, more thorough hyperparameters
```

---

## 5. Cross-Validation

### Key Concept: Don't Cheat

If you train on Monday's data and test on Tuesday's, you're cheating — Monday and Tuesday have similar weather, so the model looks better than it actually is.

### Leave-One-Season-Out (LOSO)

```
Fold 1: Train on WY2017-2025, test on WY2016
Fold 2: Train on WY2016+WY2018-2025, test on WY2017
...
Fold 10: Train on WY2016-2024, test on WY2025
```

Each fold holds out an ENTIRE WINTER. This tests whether the model works on weather patterns it has never seen, which is what matters for real forecasting.

### Why This Matters

If random CV gives MAE = 0.5 but LOSO gives MAE = 0.9, your model is **overfitting** to year-specific patterns. Our results:

| Method | MAE |
|--------|-----|
| LOSO CV | 0.825 |
| Test set (WY2026) | 0.533 |
| Climatology baseline | 1.231 |

The LOSO score being worse than the test set is expected — it averages over all seasons, including unusual ones (like the mega-snow of WY2017 and WY2023).

See: `train_models.py:leave_one_season_out_cv()`

---

## 6. Evaluation

### Key Concept: Always Have a Baseline

A fancy ML model means nothing if a simple baseline does just as well.

Our baselines:
1. **Climatology**: Always predict the historical average (MAE = 1.23")
2. **Raw model**: Use the NWP forecast as-is (MAE = 0.99")
3. **ML-corrected**: Our XGBoost model (MAE = 0.53")

**Result: 57% improvement over climatology, 46% over raw model.**

### Conditional Analysis

Overall MAE hides important patterns:

| Condition | MAE | Interpretation |
|-----------|-----|----------------|
| No snow days | 0.51" | Very accurate |
| Snow days | 1.63" | Less accurate (harder problem) |
| SW wind | 0.71" | Good — this is the main storm direction |
| S wind | 0.90" | Worse — atmospheric rivers are harder |
| January | 0.86" | Active storm month, higher error |
| October | 0.15" | Easy — rarely snows |

**Lesson**: The model is most valuable on snow days (when it matters most) but also has the most room for improvement there.

### Try It

```bash
python evaluate_models.py --text-only   # Full evaluation report
```

---

## 7. SHAP

### Key Concept: Explain Every Prediction

SHAP (SHapley Additive exPlanations) decomposes every prediction into per-feature contributions:

```
Prediction: 8.0" of new snow

Breakdown:
  Base (average):          +0.11"
  snowfall_sum = 12 cm:    +5.20"  (model says lots of snow)
  depth_change_3d = +4":   +1.80"  (already been snowing)
  temperature = 22°F:      +0.50"  (cold enough for snow)
  wind_dir = 215° (SW):    +0.80"  (good orographic direction)
  rain_sum = 0.0:          +0.20"  (all snow, no rain)
  temperature_max = 28°F:  -0.30"  (could warm up)
  slr = 15:                -0.31"  (high SLR = less compaction error)
  ───────────────────────
  Total:                    8.00"
```

### What SHAP Tells Us About Tahoe Snow

From our analysis:

1. **`snowfall_sum` is king** (SHAP = 0.99): The NWP snowfall forecast is by far the most important input. This makes sense — the model is correcting the forecast, not replacing it.

2. **`depth_change_3d` matters** (SHAP = 0.67): Recent snow history is a strong predictor. If it's been snowing for 3 days, it's likely to continue.

3. **Temperature pushes both ways**: When temp pushes UP, avg value = 24°F (cold → more snow). When it pushes DOWN, avg = 38°F (warm → less/no snow).

4. **Wind direction = 153° pushes UP** (SE-S wind): These southerly flows bring moisture from the Pacific. **Wind direction = 252° pushes DOWN** (WSW wind): Counterintuitive? Not really — WSW flow produces orographic enhancement but also wind scouring at the highest elevations.

See: `evaluate_models.py:run_shap_analysis()`

---

## 8. Deployment

### Key Concept: Graceful Degradation

The ML corrections are integrated into the live pipeline (`tahoe_snow.py`) but designed to fail silently:

```python
try:
    from ml_corrections import apply_ml_corrections
    result = apply_ml_corrections(result)
except Exception as e:
    logger.debug("ML corrections not applied: %s", e)
    result["ml_status"] = {"active": False, "reason": str(e)}
```

If XGBoost isn't installed, models aren't found, or predictions fail, you get the original forecast with no ML corrections. No crashes.

### Correction Design

Corrections are **bounded** to prevent extreme adjustments:
```python
correction = max(0.3, min(2.0, correction))  # Never more than 3x or less than 0.3x
```

This means the ML model can increase or decrease the snow forecast by up to 2x, but can't do anything crazier than that.

### What Gets Added to the Forecast

Each resort zone gets:
- `ml_corrected`: Was ML applied? (bool)
- `ml_depth_prediction_in`: ML's predicted snow depth change (inches)
- `ml_correction_factor`: Multiplier applied to existing forecast
- `ml_new_snow_prob`: Probability of new snow (0-1)

---

## 9. What to Study Next

### Improving This Pipeline

1. **Add archived forecast data**: Once `--forecasts` collection completes, rerun feature engineering and training. Having actual model predictions (not just ERA5) will improve model-specific bias corrections.

2. **Hyperparameter tuning with Optuna**: Currently using manually-set hyperparameters. Optuna can search the hyperparameter space automatically.

3. **Time series features**: Add Fourier features for seasonality, or use lagged ensemble predictions as features.

4. **Ensemble of ML models**: Train 5 XGBoost models with different seeds, average predictions for lower variance.

### Broader ML Topics

| Topic | Why It Matters | Resource |
|-------|---------------|----------|
| Bias-variance tradeoff | Understanding overfitting | [Hastie et al. Ch 7](https://hastie.su.domains/Papers/ESLII.pdf) |
| Feature selection | Removing noise features | scikit-learn `SelectKBest` |
| Bayesian optimization | Better hyperparameter tuning | Optuna docs |
| Probabilistic forecasting | Uncertainty quantification | XGBoost quantile regression |
| Neural networks for weather | When tree models aren't enough | Pangu-Weather, GraphCast papers |
| MLOps | Automated retraining pipelines | MLflow, DVC |

### Key Papers

1. **Grinsztajn et al. 2022** — "Why do tree-based models still outperform deep learning on typical tabular data?" (NeurIPS). Confirms our choice of XGBoost over neural nets.

2. **Lundberg & Lee 2017** — "A Unified Approach to Interpreting Model Predictions" (NeurIPS). The original SHAP paper.

3. **Hoopes et al. 2023** — "Improving prediction of mountain snowfall using ML methods" (Meteorological Applications). Directly relevant to our Tahoe application.

---

## Quick Reference: Running the Pipeline

```bash
# Step 1: Collect data
python historical_data_collector.py --snotel     # 30 seconds
python historical_data_collector.py --era5       # 5 minutes
python historical_data_collector.py --forecasts  # 15-30 minutes

# Step 2: Build features
python feature_engineering.py                    # 20 seconds

# Step 3: Train models
python train_models.py --quick                   # 25 seconds
python train_models.py                           # 1 minute

# Step 4: Evaluate
python evaluate_models.py --text-only            # 10 seconds

# Step 5: Integration (automatic — built into tahoe_snow.py)
# ML corrections are applied every time analyze_all() runs
```
