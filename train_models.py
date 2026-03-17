#!/usr/bin/env python3
"""
XGBoost Model Training for Tahoe Snow Forecast Correction
==========================================================

Trains gradient-boosted tree models that learn to correct NWP forecast
biases using historical forecast-vs-observation data.

=== ML LEARNING NOTE: Supervised Learning Overview ===

This is SUPERVISED LEARNING: we have labeled data (features + known outcomes)
and we're teaching the model to predict outcomes from features.

The workflow is:
  1. SPLIT data into train/validate/test sets (by season, not randomly!)
  2. TRAIN a model on training data — it learns patterns
  3. VALIDATE on held-out data — tune hyperparameters
  4. TEST on final held-out data — unbiased accuracy estimate
  5. INTERPRET — understand WHY the model makes its predictions

We use XGBoost (eXtreme Gradient Boosting) because:
  - Handles missing data natively (SNOTEL sensors fail in storms)
  - Handles nonlinear relationships (weather is nonlinear!)
  - Fast to train (seconds, not hours)
  - Provides feature importance rankings
  - Industry standard for tabular data (won most Kaggle competitions 2015-2020)
  - No GPU needed (critical for Raspberry Pi deployment)

=== ML LEARNING NOTE: Why NOT Neural Networks? ===

Deep learning (LSTMs, transformers) is NOT the right tool here because:
  1. We have ~20K samples — deep learning needs 100K+ to outperform XGBoost
  2. Our features are tabular (columns of numbers), not images or text
  3. We need interpretability (SHAP values) — neural nets are black boxes
  4. We need fast inference on a Raspberry Pi — no GPU available
  5. XGBoost handles missing data; neural nets require imputation

Rule of thumb: For tabular data < 100K rows, tree-based methods beat
neural networks ~90% of the time. This is well-established in the ML
literature (Grinsztajn et al. 2022, "Why do tree-based models still
outperform deep learning on typical tabular data?").

Usage:
    python train_models.py                   # Train all models
    python train_models.py --quick           # Quick training (fewer hyperparameters)
    python train_models.py --evaluate-only   # Load and evaluate existing models
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import optuna

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historical")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models")


# =========================================================================
# Feature Selection
# =========================================================================

# Core features: ERA5 + physics + temporal + lagged + terrain (32 features).
# These are available for all 10 years of data (2016-2026) and perform best
# for base/mid elevation bands where we have abundant training samples.
CORE_FEATURES = [
    # ERA5 weather variables
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "snowfall_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",

    # Derived physics features
    "slr",
    "orographic_mult",
    "lapse_rate_c_km",
    "froude_number",
    "wet_bulb_f",
    "precip_orographic_in",

    # Temporal features
    "temp_trend_3d",
    "month",
    "day_of_year",
    "water_year_day",

    # Lagged SNOTEL features (what happened recently)
    "depth_lag1d",
    "swe_lag1d",
    "tmax_lag1d",
    "depth_change_3d",
    "swe_change_3d",
    "tmax_avg_3d",
    "depth_change_7d",
    "swe_change_7d",

    # Terrain features
    "station_elev_ft",
    "zone_elev_ft",
    "elev_diff_ft",
    "aspect_deg",
    "slope_angle_deg",

    # Improvement 1: Rain/snow discrimination
    "rain_fraction",
    "snow_fraction",
    "snowfall_adjusted_cm",

    # Improvement 2: Storm state features
    "consecutive_snow_days",
    "storm_total_in",
    "settling_pressure",

    # Improvement 3: Model spread (moved from peak-only to all bands)
    "model_snow_spread",
    "model_snow_mean",
    "model_snow_range",
    "model_temp_spread",
    "model_temp_mean",

    # Improvement 5: Station-specific bias
    "station_mean_depth_change",
    "station_std_depth_change",

    # Improvement 7: Climate indices
    "oni",
    "enso_phase",

    # Met 1: Freezing level
    "est_freezing_level_ft",
    "freezing_level_margin_ft",

    # Met 2: AR detection
    "moisture_transport_proxy",
    "ar_score",
    "is_ar_event",

    # PREC-based precipitation features
    "prec_change_in",
    "prec_3d_in",
    "prec_7d_in",
    "rain_event",

    # Snow physics features
    "observed_density",
    "compaction_rate",
    "diurnal_range_f",
    "freeze_thaw",
    "tmax_above_freezing",
    "tmin_below_freezing",
]

# Extended features: Core + archived NWP model forecasts (55 features).
# The forecast columns are NaN for 2016-2020 (pre-archive era); XGBoost
# handles NaN natively. These help most for the peak band where we have
# fewer samples and terrain effects are hardest for the base model to learn.
FORECAST_FEATURES = [
    # Archived NWP model forecasts (available for 2021+ data only)
    "ncep_gfs013_tmax_f",
    "ncep_gfs013_tmean_f",
    "ncep_gfs013_precip_in",
    "ncep_gfs013_snow_cm",
    "ncep_gfs013_wind_max_mph",
    "ncep_gfs013_wind_dir_deg",
    "ecmwf_tmax_f",
    "ecmwf_tmean_f",
    "ecmwf_precip_in",
    "ecmwf_snow_cm",
    "ecmwf_wind_max_mph",
    "ecmwf_wind_dir_deg",
    "icon_tmax_f",
    "icon_tmean_f",
    "icon_precip_in",
    "icon_snow_cm",
    "icon_wind_max_mph",
    "icon_wind_dir_deg",

]

# Hybrid strategy: use CORE_FEATURES for base/mid (more data, less complexity
# needed) and CORE_FEATURES + FORECAST_FEATURES for peak (data-starved,
# needs the extra signal from real model forecasts).
BAND_FEATURE_MAP = {
    "all": CORE_FEATURES,
    "base": CORE_FEATURES,
    "mid": CORE_FEATURES,
    "peak": CORE_FEATURES + FORECAST_FEATURES,
}

# Features that are derived from the new_snow target (depth_change_in > 0.5)
# and MUST be excluded from the new_snow classifier to avoid data leakage.
# These are fine for regression targets (depth_change_in, swe_change_in)
# because knowing "it snowed" (binary) is legitimate context for predicting
# "how much" (continuous), but for predicting "will it snow" (binary) they
# encode the answer directly.
# Features derived from same-day SNOTEL observations. These are NOT available
# at forecast time — SNOTEL reports once per day, so when making a forecast
# for today/tomorrow, we only have YESTERDAY's values.
#
# These must be excluded from ALL models (not just classification) to get
# honest forecast-time performance metrics. Including them inflates accuracy
# by giving the model access to the answer.
SAME_DAY_OBSERVATION_FEATURES = {
    # Directly derived from today's target (depth_change_in, swe_change_in)
    "consecutive_snow_days",  # counts days where depth_change > 0.5
    "storm_total_in",         # cumulative depth during storm
    "settling_pressure",      # depth * swe_change
    "observed_density",       # swe_change / depth_change
    "compaction_rate",        # derived from negative depth_change
    "rain_event",             # derived from depth_change + prec_change

    # Today's SNOTEL gauge readings (not available until end of day)
    "prec_change_in",         # today's PREC cumulative change
    "prec_3d_in",             # includes today's PREC reading
    "prec_7d_in",             # includes today's PREC reading

    # .diff() features that include today's SNOTEL observation
    # (today's depth minus N days ago = uses today's reading)
    "depth_change_3d",        # SNWD.diff(3) includes today
    "depth_change_7d",        # SNWD.diff(7) includes today
    "swe_change_3d",          # WTEQ.diff(3) includes today
    "swe_change_7d",          # WTEQ.diff(7) includes today

    # SAFE (not excluded): depth_lag1d, swe_lag1d, tmax_lag1d are .shift(1)
    # = yesterday's value, legitimately available at forecast time
}

# For classification, these are also leaky (they're the same features plus
# they directly encode the binary target)
LEAKY_FOR_CLASSIFICATION = SAME_DAY_OBSERVATION_FEATURES

# Default for backwards compat
FEATURE_COLUMNS = CORE_FEATURES + FORECAST_FEATURES

# Target variables we train models for
TARGET_CONFIGS = {
    "depth_change_in": {
        "description": "Daily snow depth change (inches)",
        "metric": "regression",
        "eval_metric": "mae",
    },
    "swe_change_in": {
        "description": "Daily SWE change (inches water equivalent)",
        "metric": "regression",
        "eval_metric": "mae",
    },
    "new_snow": {
        "description": "Did new snow fall? (binary: 0 or 1)",
        "metric": "classification",
        "eval_metric": "logloss",
    },
}


# =========================================================================
# Leave-One-Season-Out Cross Validation
# =========================================================================

def leave_one_season_out_cv(df: pd.DataFrame, features: list, target: str,
                            params: dict, is_classification: bool = False) -> dict:
    """
    Leave-One-Season-Out Cross-Validation (LOSO-CV).

    === ML LEARNING NOTE: Why Not Random Cross-Validation? ===

    In standard k-fold CV, you randomly split data into k groups. This
    works for independent data (e.g., images of cats vs dogs), but weather
    data is TEMPORALLY AUTOCORRELATED — today's weather is very similar to
    yesterday's. Random splitting would put Jan 14 in training and Jan 15
    in validation, which leaks information.

    LOSO-CV solves this by holding out ENTIRE SEASONS:
      Fold 1: Train on 2017-2025, validate on 2016/17
      Fold 2: Train on 2016 + 2018-2025, validate on 2017/18
      ...

    This tests whether the model generalizes to UNSEEN WEATHER PATTERNS,
    not just unseen days within familiar patterns. This is critical because:

    1. Each winter has different ENSO state (El Nino, La Nina, Neutral)
    2. Storm tracks shift year to year
    3. We need confidence that the model works for NEXT winter, which
       by definition has weather patterns we haven't seen yet

    If your LOSO-CV score is much worse than random CV, your model is
    OVERFITTING to year-specific patterns and won't generalize.
    """
    water_years = sorted(df["water_year"].unique())
    if len(water_years) < 3:
        logger.warning("Need at least 3 water years for LOSO-CV, got %d", len(water_years))
        return {"scores": [], "mean_score": np.nan}

    fold_scores = []
    fold_details = []

    for hold_out_year in water_years:
        train_mask = df["water_year"] != hold_out_year
        val_mask = df["water_year"] == hold_out_year

        X_train = df.loc[train_mask, features].values
        y_train = df.loc[train_mask, target].values
        X_val = df.loc[val_mask, features].values
        y_val = df.loc[val_mask, target].values

        # Skip folds with insufficient data
        if len(y_val) < 10 or len(y_train) < 50:
            continue

        # Skip if target is all NaN
        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)
        if valid_train.sum() < 50 or valid_val.sum() < 10:
            continue

        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        X_val = X_val[valid_val]
        y_val = y_val[valid_val]

        if is_classification:
            model = xgb.XGBClassifier(**params, random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import log_loss, accuracy_score, f1_score
            score = -log_loss(y_val, y_pred_proba)  # Negative because higher is better
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            fold_details.append({
                "water_year": int(hold_out_year),
                "n_val": len(y_val),
                "logloss": -score,
                "accuracy": round(acc, 3),
                "f1": round(f1, 3),
            })
        else:
            model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            bias = np.mean(y_pred - y_val)
            r2 = r2_score(y_val, y_pred)
            score = -mae  # Negative MAE (higher is better)
            fold_details.append({
                "water_year": int(hold_out_year),
                "n_val": len(y_val),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "bias": round(bias, 4),
                "r2": round(r2, 4),
            })

        fold_scores.append(score)

    mean_score = np.mean(fold_scores) if fold_scores else np.nan
    std_score = np.std(fold_scores) if fold_scores else np.nan

    return {
        "scores": fold_scores,
        "mean_score": round(mean_score, 4),
        "std_score": round(std_score, 4),
        "n_folds": len(fold_scores),
        "fold_details": fold_details,
    }


# =========================================================================
# Model Training
# =========================================================================

def _optuna_tune(df: pd.DataFrame, features: list, target: str,
                 is_classification: bool, n_trials: int = 40) -> dict:
    """
    Use Optuna to find optimal XGBoost hyperparameters via LOSO-CV.

    Runs n_trials random+TPE search over the hyperparameter space.
    Returns the best parameter dict found.
    """
    # Use 3 folds instead of full LOSO for speed during tuning
    water_years = sorted(df["water_year"].unique())
    # Pick 3 spread-out years to validate on
    val_years = [water_years[len(water_years)//4],
                 water_years[len(water_years)//2],
                 water_years[3*len(water_years)//4]]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        }

        scores = []
        for val_year in val_years:
            train_mask = df["water_year"] != val_year
            val_mask = df["water_year"] == val_year

            X_train = df.loc[train_mask, features].values
            y_train = df.loc[train_mask, target].values
            X_val = df.loc[val_mask, features].values
            y_val = df.loc[val_mask, target].values

            valid_train = ~np.isnan(y_train)
            valid_val = ~np.isnan(y_val)
            if valid_train.sum() < 50 or valid_val.sum() < 10:
                continue

            X_train, y_train = X_train[valid_train], y_train[valid_train]
            X_val, y_val = X_val[valid_val], y_val[valid_val]

            if is_classification:
                model = xgb.XGBClassifier(**params, random_state=42, verbosity=0)
                model.fit(X_train, y_train)
                from sklearn.metrics import log_loss
                y_proba = model.predict_proba(X_val)[:, 1]
                scores.append(-log_loss(y_val, y_proba))
            else:
                model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(-mean_absolute_error(y_val, y_pred))

        return np.mean(scores) if scores else -999

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info("    Optuna best (n=%d trials): MAE=%.4f, depth=%d, lr=%.4f, n_est=%d",
                n_trials, -study.best_value, best["max_depth"],
                best["learning_rate"], best["n_estimators"])
    return best


def train_model(df: pd.DataFrame, features: list, target: str,
                target_config: dict, elevation_band: str = "all",
                quick: bool = False) -> dict:
    """
    Train an XGBoost model for a specific target and elevation band.

    === ML LEARNING NOTE: The Training Loop ===

    Training a gradient-boosted tree model is an iterative process:

    1. Start with a simple prediction (the mean of the target)
    2. Compute RESIDUALS (how wrong we are for each sample)
    3. Train a shallow decision tree to predict the residuals
    4. Add this tree's predictions (scaled by learning rate) to our
       running prediction
    5. Repeat steps 2-4 for n_estimators rounds

    Each new tree corrects the mistakes of the previous trees. The
    learning_rate controls how aggressively each tree corrects — too
    high and we overfit, too low and we need too many trees.

    Key hyperparameters:
    - n_estimators: How many trees to add (100-500 typical)
    - max_depth: How deep each tree goes (3-6 typical; deeper = more complex)
    - learning_rate: Step size for corrections (0.01-0.1 typical)
    - subsample: Fraction of data used per tree (0.7-0.9; adds randomness)
    - colsample_bytree: Fraction of features per tree (0.5-0.8)
    - reg_alpha/lambda: L1/L2 regularization (prevents overfitting)
    """
    logger.info("  Training %s model for elevation band '%s'...", target, elevation_band)

    # Filter by elevation band
    if elevation_band != "all":
        band_df = df[df["elevation_band"] == elevation_band].copy()
    else:
        band_df = df.copy()

    if len(band_df) < 100:
        logger.warning("    Only %d rows for %s/%s — skipping", len(band_df), target, elevation_band)
        return None

    # Drop rows with NaN target
    valid_mask = band_df[target].notna()
    band_df = band_df[valid_mask]

    if len(band_df) < 100:
        logger.warning("    Only %d valid target rows for %s/%s — skipping",
                        len(band_df), target, elevation_band)
        return None

    # Get available features (some may not be present)
    available_features = [f for f in features if f in band_df.columns]
    is_classification = target_config["metric"] == "classification"

    # Exclude same-day observation features from ALL models.
    # These features use today's SNOTEL readings which don't exist at
    # forecast time. Including them inflates accuracy metrics.
    before = len(available_features)
    available_features = [f for f in available_features
                          if f not in SAME_DAY_OBSERVATION_FEATURES]
    if len(available_features) < before:
        logger.info("    Excluded %d same-day observation features (not available at forecast time)",
                    before - len(available_features))

    # === Hyperparameters (Improvement 4: Optuna tuning) ===
    if quick:
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
        }
    else:
        params = _optuna_tune(band_df, available_features, target, is_classification)

    # === Leave-One-Season-Out CV ===
    logger.info("    Running LOSO cross-validation (%d water years)...",
                band_df["water_year"].nunique())
    cv_results = leave_one_season_out_cv(
        band_df, available_features, target, params, is_classification
    )

    if is_classification:
        logger.info("    LOSO CV: logloss=%.4f +/- %.4f over %d folds",
                    -cv_results["mean_score"], cv_results["std_score"],
                    cv_results["n_folds"])
    else:
        logger.info("    LOSO CV: MAE=%.4f +/- %.4f over %d folds",
                    -cv_results["mean_score"], cv_results["std_score"],
                    cv_results["n_folds"])

    # === Train final model on all data ===
    # Hold out most recent water year as final test set
    water_years = sorted(band_df["water_year"].unique())
    test_year = water_years[-1]
    train_years = water_years[:-1]

    train_mask = band_df["water_year"].isin(train_years)
    test_mask = band_df["water_year"] == test_year

    X_train = band_df.loc[train_mask, available_features].values
    y_train = band_df.loc[train_mask, target].values
    X_test = band_df.loc[test_mask, available_features].values
    y_test = band_df.loc[test_mask, target].values

    if is_classification:
        final_model = xgb.XGBClassifier(**params, random_state=42, verbosity=0)
    else:
        final_model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)

    final_model.fit(X_train, y_train)

    # === Test Set Evaluation ===
    if is_classification:
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = final_model.predict(X_test)
        from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score
        test_metrics = {
            "logloss": round(log_loss(y_test, y_pred_proba), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        logger.info("    Test set (%d, WY%d): accuracy=%.3f, F1=%.3f, precision=%.3f, recall=%.3f",
                    len(y_test), test_year, test_metrics["accuracy"],
                    test_metrics["f1"], test_metrics["precision"], test_metrics["recall"])
    else:
        y_pred = final_model.predict(X_test)
        test_metrics = {
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "bias": round(float(np.mean(y_pred - y_test)), 4),
            "r2": round(r2_score(y_test, y_pred), 4),
        }
        logger.info("    Test set (%d, WY%d): MAE=%.3f, RMSE=%.3f, bias=%+.3f, R2=%.3f",
                    len(y_test), test_year, test_metrics["mae"],
                    test_metrics["rmse"], test_metrics["bias"], test_metrics["r2"])

    # === Feature Importance ===
    importances = final_model.feature_importances_
    feature_importance = dict(sorted(
        zip(available_features, [round(float(x), 4) for x in importances]),
        key=lambda x: x[1], reverse=True,
    ))

    # === Baseline Comparison ===
    baseline_metrics = compute_baseline(y_test, X_test, available_features, target, is_classification)

    # Log top features
    top_features = list(feature_importance.items())[:5]
    logger.info("    Top 5 features: %s",
                ", ".join(f"{k}={v:.3f}" for k, v in top_features))

    # === Improvement 6: Quantile Regression for Uncertainty ===
    quantile_models = {}
    if not is_classification:
        for q_label, q_val in [("p10", 0.1), ("p90", 0.9)]:
            q_params = {k: v for k, v in params.items()}
            q_params["objective"] = "reg:quantileerror"
            q_params["quantile_alpha"] = q_val
            q_model = xgb.XGBRegressor(**q_params, random_state=42, verbosity=0)
            q_model.fit(X_train, y_train)
            q_pred = q_model.predict(X_test)
            coverage = np.mean((y_test >= q_model.predict(X_test)) if q_val < 0.5
                               else (y_test <= q_model.predict(X_test)))
            quantile_models[q_label] = q_model
            logger.info("    Quantile %s: coverage=%.1f%% on test set", q_label, coverage * 100)

    return {
        "target": target,
        "elevation_band": elevation_band,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_year": int(test_year),
        "params": params,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "feature_importance": feature_importance,
        "model": final_model,
        "quantile_models": quantile_models,
        "features_used": available_features,
    }


def compute_baseline(y_test, X_test, features, target, is_classification):
    """
    Compute baseline metrics for comparison.

    === ML LEARNING NOTE: Always Have a Baseline ===

    The most important question in ML is: "Is my model better than
    doing nothing (or something simple)?"

    Common baselines:
    1. CLIMATOLOGY: Predict the historical average (always predict
       "0.11 inches of depth change" because that's the training mean)
    2. PERSISTENCE: Predict today's weather = yesterday's weather
    3. RAW MODEL: Use the NWP forecast as-is without correction

    If your fancy XGBoost model can't beat these simple baselines,
    it's not adding value and you should investigate why.

    A good ML model should improve over the baseline by at least 10-20%.
    If it's only 1-2% better, the added complexity may not be worth it.
    """
    if is_classification:
        # Baseline: predict most common class
        from sklearn.metrics import accuracy_score
        majority_class = int(np.round(np.mean(y_test)))
        baseline_pred = np.full_like(y_test, majority_class)
        return {
            "strategy": "majority_class",
            "accuracy": round(accuracy_score(y_test, baseline_pred), 4),
        }
    else:
        # Baseline 1: predict the mean
        mean_pred = np.full_like(y_test, np.mean(y_test))
        mean_mae = mean_absolute_error(y_test, mean_pred)

        # Baseline 2: use raw ERA5 snowfall as prediction (if available)
        snowfall_idx = None
        for i, f in enumerate(features):
            if f == "snowfall_sum":
                snowfall_idx = i
                break

        raw_model_mae = None
        if snowfall_idx is not None and target == "depth_change_in":
            # ERA5 snowfall is in cm, convert to inches and use as prediction
            raw_pred = X_test[:, snowfall_idx] * 0.3937  # cm to inches
            raw_model_mae = round(mean_absolute_error(y_test, raw_pred), 4)

        return {
            "strategy": "climatology (mean)",
            "mean_mae": round(mean_mae, 4),
            "raw_model_mae": raw_model_mae,
        }


# =========================================================================
# Save/Load Models
# =========================================================================

def save_model_results(results: list[dict]):
    """Save trained models and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    for result in results:
        if result is None:
            continue

        target = result["target"]
        band = result["elevation_band"]
        model = result.pop("model")
        quantile_models = result.pop("quantile_models", {})

        # Save XGBoost model
        model_path = os.path.join(MODEL_DIR, f"{target}_{band}.json")
        model.save_model(model_path)

        # Save quantile models (Improvement 6)
        for q_label, q_model in quantile_models.items():
            q_path = os.path.join(MODEL_DIR, f"{target}_{band}_{q_label}.json")
            q_model.save_model(q_path)

        # Save metadata
        meta_path = os.path.join(MODEL_DIR, f"{target}_{band}_meta.json")
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2, default=str)
        os.replace(tmp, meta_path)

        logger.info("  Saved model: %s (+ %d quantile models)", model_path, len(quantile_models))

    # Save combined results summary
    summary = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "models": [
            {k: v for k, v in r.items() if k not in ("model", "quantile_models")}
            for r in results if r is not None
        ],
    }
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    tmp = summary_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    os.replace(tmp, summary_path)


# =========================================================================
# Main Training Pipeline
# =========================================================================

def train_all(quick: bool = False):
    """
    Train models for all target variables and elevation bands.

    === ML LEARNING NOTE: Model Granularity ===

    We train SEPARATE models for:
    - Each elevation band (base/mid/peak) because snow physics differ
      dramatically by elevation. At 6,500 ft, it might rain while snowing
      at 10,000 ft. A single model would average these very different regimes.
    - Each target variable because depth change and SWE change have
      different error patterns.
    - "all" band (combined) for comparison — sometimes more data helps
      even if the physics differ.

    This gives us: 3 targets x 4 bands (base/mid/peak/all) = 12 models.
    Each trains in seconds on ~5K-20K rows.
    """
    # Load training data
    training_path = os.path.join(DATA_DIR, "training_data.parquet")
    if not os.path.exists(training_path):
        logger.error("Training data not found. Run: python feature_engineering.py")
        sys.exit(1)

    logger.info("Loading training data...")
    df = pd.read_parquet(training_path)
    logger.info("  %d rows, %d columns, water years %d-%d",
                len(df), len(df.columns),
                int(df["water_year"].min()), int(df["water_year"].max()))

    # Check which features are available per band
    # Hybrid strategy: base/mid use core features only (32), peak uses
    # core + forecast features (55). This avoids overfitting at lower
    # elevations where we have plenty of data, while giving the data-starved
    # peak band the extra signal from real NWP forecasts.
    band_available = {}
    for band, band_features in BAND_FEATURE_MAP.items():
        avail = [f for f in band_features if f in df.columns]
        miss = [f for f in band_features if f not in df.columns]
        band_available[band] = avail
        logger.info("  Band '%s': %d features available, %d missing", band, len(avail), len(miss))

    elevation_bands = ["all", "base", "mid", "peak"]
    all_results = []

    for target, config in TARGET_CONFIGS.items():
        logger.info("\n" + "=" * 60)
        logger.info("TARGET: %s (%s)", target, config["description"])
        logger.info("=" * 60)

        for band in elevation_bands:
            available = band_available[band]
            result = train_model(df, available, target, config, band, quick=quick)
            if result is not None:
                all_results.append(result)

    # Save all models
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODELS")
    logger.info("=" * 60)
    save_model_results(all_results)

    # Print summary
    print_training_summary(all_results)

    return all_results


def print_training_summary(results: list[dict]):
    """Print a formatted summary of training results."""
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}\n")

    for target in TARGET_CONFIGS:
        target_results = [r for r in results if r is not None and r["target"] == target]
        if not target_results:
            continue

        config = TARGET_CONFIGS[target]
        print(f"--- {target}: {config['description']} ---\n")

        if config["metric"] == "regression":
            print(f"  {'Band':<8} {'N_train':>7} {'N_test':>7} "
                  f"{'MAE':>8} {'RMSE':>8} {'Bias':>8} {'R2':>6} "
                  f"{'CV_MAE':>8} {'Base_MAE':>8} {'Improvement':>12}")
            for r in target_results:
                tm = r["test_metrics"]
                bm = r["baseline_metrics"]
                improvement = ""
                if bm.get("mean_mae") and tm["mae"] < bm["mean_mae"]:
                    pct = (1 - tm["mae"] / bm["mean_mae"]) * 100
                    improvement = f"{pct:+.1f}% vs clim"
                elif bm.get("raw_model_mae") and tm["mae"] < bm["raw_model_mae"]:
                    pct = (1 - tm["mae"] / bm["raw_model_mae"]) * 100
                    improvement = f"{pct:+.1f}% vs raw"

                print(f"  {r['elevation_band']:<8} {r['n_train']:>7} {r['n_test']:>7} "
                      f"{tm['mae']:>8.3f} {tm['rmse']:>8.3f} {tm['bias']:>+8.3f} "
                      f"{tm['r2']:>6.3f} "
                      f"{-r['cv_results']['mean_score']:>8.3f} "
                      f"{bm.get('mean_mae', 'N/A'):>8} "
                      f"{improvement:>12}")
        else:
            print(f"  {'Band':<8} {'N_train':>7} {'N_test':>7} "
                  f"{'Accuracy':>8} {'F1':>6} {'Precision':>9} {'Recall':>7} "
                  f"{'Base_Acc':>8}")
            for r in target_results:
                tm = r["test_metrics"]
                bm = r["baseline_metrics"]
                print(f"  {r['elevation_band']:<8} {r['n_train']:>7} {r['n_test']:>7} "
                      f"{tm['accuracy']:>8.3f} {tm['f1']:>6.3f} "
                      f"{tm['precision']:>9.3f} {tm['recall']:>7.3f} "
                      f"{bm.get('accuracy', 'N/A'):>8}")

        # Print top features for the "all" band model
        all_result = next((r for r in target_results if r["elevation_band"] == "all"), None)
        if all_result:
            print(f"\n  Top 10 features (all bands):")
            for feat, imp in list(all_result["feature_importance"].items())[:10]:
                bar = "#" * int(imp * 100)
                print(f"    {feat:35s} {imp:.4f} {bar}")

        print()


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for Tahoe snow forecast correction",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick training with fewer hyperparameters")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Load and evaluate existing models only")
    args = parser.parse_args()

    if args.evaluate_only:
        logger.info("Evaluate-only mode — loading existing models (no retraining)")
        from evaluate_models import generate_full_report
        generate_full_report(text_only=True)
        return

    t0 = time.time()
    results = train_all(quick=args.quick)
    elapsed = time.time() - t0

    logger.info("Training complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
