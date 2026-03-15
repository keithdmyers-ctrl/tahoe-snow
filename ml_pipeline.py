#!/usr/bin/env python3
"""
Machine Learning Post-Processing Pipeline for Tahoe Snow Forecasts

Trains location-specific correction models using accumulated verification data.
Designed to activate after one season (~50+ storm events) of verification data.

Architecture:
- Input features: NWP model outputs, sounding data, pressure trends, SNOTEL temps, model spread
- Target: observed snow/temp at SNOTEL stations
- Model: Gradient-boosted trees (XGBoost or LightGBM)
- Retraining: Weekly with expanding window
- Output: Bias corrections per variable, per elevation band, per lead time

Dependencies (install when activating):
    pip install xgboost scikit-learn pandas
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Verification data files produced by forecast_verification.py
VERIFICATION_LOG = os.path.join(_BASE_DIR, ".forecast_verification.json")
SNOW_VERIFICATION = os.path.join(_BASE_DIR, ".snow_verification.json")

# Elevation bands (ft) used for per-band correction models
DEFAULT_ELEVATION_BANDS = [
    (6000, 7000, "base"),
    (7000, 8500, "mid"),
    (8500, 11000, "peak"),
]

DEFAULT_FEATURE_LIST = [
    # NWP model outputs
    "gfs_temp_f", "ecmwf_temp_f", "icon_temp_f", "hrrr_temp_f",
    "gfs_snow_in", "ecmwf_snow_in", "icon_snow_in", "hrrr_snow_in",
    "gfs_wind_mph", "ecmwf_wind_mph", "icon_wind_mph",
    # Sounding features
    "freeze_level_ft", "snow_level_ft", "lapse_rate_c_km", "inversion_present",
    # Pressure features
    "pressure_trend_3h", "pressure_trend_12h", "pressure_hpa",
    # Ensemble features
    "ensemble_spread", "ensemble_iqr", "ensemble_skewness",
    # Calendar features
    "month", "day_of_year",
]

DEFAULT_TARGET_VARIABLES = ["snow_in", "temp_high_f", "temp_low_f"]


@dataclass
class MLPipelineConfig:
    """Configuration for the ML post-processing pipeline."""
    min_training_days: int = 90
    retrain_interval_days: int = 7
    feature_list: list = field(default_factory=lambda: list(DEFAULT_FEATURE_LIST))
    target_variables: list = field(default_factory=lambda: list(DEFAULT_TARGET_VARIABLES))
    elevation_bands: list = field(default_factory=lambda: list(DEFAULT_ELEVATION_BANDS))
    model_path: str = os.path.join(_BASE_DIR, ".ml_models")


class FeatureExtractor:
    """Extracts training features from analysis and verification data structures.

    Reads from the data structures produced by analyze_all() and
    forecast_verification.py to build flat feature dicts suitable for
    tabular ML models.
    """

    def extract(self, analysis_dict: dict, verification_dict: dict) -> dict:
        """Extract a flat feature dictionary from analysis + verification data.

        Args:
            analysis_dict: Output of analyze_all() — contains resorts, current_conditions,
                model_weights, ensemble, sounding, etc.
            verification_dict: A single day's verification record from
                .forecast_verification.json (has "source", "metric", "predicted", "actual").

        Returns:
            Flat dict of feature_name -> float suitable for model input.
        """
        features = {}

        # --- NWP model outputs ---
        # Extract per-model temp/snow from the peak zone of Heavenly (primary target)
        resorts = analysis_dict.get("resorts", {})
        heavenly = resorts.get("Heavenly", {})
        peak = heavenly.get("zones", {}).get("peak", {})

        model_spread = peak.get("model_spread", [])
        if model_spread:
            first_day = model_spread[0] if model_spread else {}
            models = first_day.get("models", {})
            for model_key in ("GFS", "ECMWF", "ICON", "HRRR"):
                model_data = models.get(model_key, {})
                features[f"{model_key.lower()}_temp_f"] = model_data.get("temp_high_f", np.nan)
                features[f"{model_key.lower()}_snow_in"] = model_data.get("snow_in", np.nan)

        # Wind from current snapshot
        snap = peak.get("current", {})
        for model_key in ("GFS", "ECMWF", "ICON"):
            # Wind is only available from the blended snapshot, not per-model
            features[f"{model_key.lower()}_wind_mph"] = snap.get("wind_mph", np.nan)

        # --- Sounding features ---
        sounding = analysis_dict.get("sounding") or {}
        features["freeze_level_ft"] = _safe_float(sounding.get("freezing_level_ft"))
        features["snow_level_ft"] = _safe_float(
            analysis_dict.get("current_conditions", {}).get("snow_level_ft")
        )
        features["lapse_rate_c_km"] = _safe_float(
            analysis_dict.get("current_conditions", {}).get("observed_lapse_rate_c_km")
        )
        # Inversion detection: lapse rate < 3.0 C/km indicates temperature inversion
        lr = features["lapse_rate_c_km"]
        features["inversion_present"] = 1.0 if (not np.isnan(lr) and lr < 3.0) else 0.0

        # --- Pressure features ---
        # These come from the current observation or sensor data
        obs = analysis_dict.get("current_conditions", {}).get("observation") or {}
        features["pressure_hpa"] = _safe_float(obs.get("pressure_hpa"))
        # Pressure trends require historical data; use NaN if unavailable
        features["pressure_trend_3h"] = _safe_float(obs.get("pressure_trend_3h"))
        features["pressure_trend_12h"] = _safe_float(obs.get("pressure_trend_12h"))

        # --- Ensemble features ---
        ensemble = analysis_dict.get("ensemble") or {}
        ensemble_models = ensemble.get("models", {})
        snow_values = []
        for model_label, daily_list in ensemble_models.items():
            if isinstance(daily_list, list) and daily_list:
                day0 = daily_list[0] if daily_list else {}
                snow_p50 = day0.get("snow_p50")
                if snow_p50 is not None:
                    snow_values.append(snow_p50)

        if len(snow_values) >= 2:
            arr = np.array(snow_values)
            features["ensemble_spread"] = float(np.std(arr))
            q75, q25 = np.percentile(arr, [75, 25])
            features["ensemble_iqr"] = float(q75 - q25)
            # Skewness: Fisher's definition
            mean = np.mean(arr)
            std = np.std(arr)
            if std > 0:
                features["ensemble_skewness"] = float(
                    np.mean(((arr - mean) / std) ** 3)
                )
            else:
                features["ensemble_skewness"] = 0.0
        else:
            features["ensemble_spread"] = np.nan
            features["ensemble_iqr"] = np.nan
            features["ensemble_skewness"] = np.nan

        # --- Calendar features ---
        generated = analysis_dict.get("generated", "")
        try:
            dt = datetime.fromisoformat(generated)
            features["month"] = float(dt.month)
            features["day_of_year"] = float(dt.timetuple().tm_yday)
        except (ValueError, TypeError):
            now = datetime.now(timezone.utc)
            features["month"] = float(now.month)
            features["day_of_year"] = float(now.timetuple().tm_yday)

        return features


class SnowCorrectionModel:
    """Per-elevation-band, per-lead-time correction model.

    When XGBoost is available, trains a gradient-boosted regressor.
    Otherwise falls back to a no-op (returns 1.0 = no correction).
    """

    def __init__(self, elevation_band: str, lead_time_hours: int):
        self.elevation_band = elevation_band
        self.lead_time_hours = lead_time_hours
        self._model = None
        self._feature_names: list = []
        self._feature_importances: dict = {}
        self._trained = False

    def train(self, features_df, targets_df):
        """Train correction model on historical features and targets.

        Args:
            features_df: numpy array or list-of-dicts, shape (n_samples, n_features)
            targets_df: numpy array, shape (n_samples,) — observed values
        """
        if isinstance(features_df, list):
            features_df = np.array(features_df)
        if isinstance(targets_df, list):
            targets_df = np.array(targets_df)

        if len(features_df) < 10:
            logger.warning(
                "Insufficient training data (%d samples) for %s/%dh",
                len(features_df), self.elevation_band, self.lead_time_hours,
            )
            return

        # TODO: uncomment when xgboost available
        # import xgboost as xgb
        # from sklearn.model_selection import cross_val_score
        #
        # # Replace NaN with column medians for XGBoost
        # nan_mask = np.isnan(features_df)
        # col_medians = np.nanmedian(features_df, axis=0)
        # for col_idx in range(features_df.shape[1]):
        #     features_df[nan_mask[:, col_idx], col_idx] = col_medians[col_idx]
        #
        # model = xgb.XGBRegressor(
        #     n_estimators=100,
        #     max_depth=4,
        #     learning_rate=0.1,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     reg_alpha=0.1,
        #     reg_lambda=1.0,
        #     random_state=42,
        # )
        #
        # # Cross-validate to check for overfitting
        # if len(features_df) >= 30:
        #     cv_scores = cross_val_score(model, features_df, targets_df,
        #                                  cv=min(5, len(features_df) // 5),
        #                                  scoring="neg_mean_absolute_error")
        #     logger.info("CV MAE for %s/%dh: %.2f +/- %.2f",
        #                 self.elevation_band, self.lead_time_hours,
        #                 -cv_scores.mean(), cv_scores.std())
        #
        # model.fit(features_df, targets_df)
        # self._model = model
        #
        # # Extract feature importances
        # importances = model.feature_importances_
        # if self._feature_names and len(self._feature_names) == len(importances):
        #     self._feature_importances = {
        #         name: round(float(imp), 4)
        #         for name, imp in sorted(
        #             zip(self._feature_names, importances),
        #             key=lambda x: x[1], reverse=True,
        #         )
        #     }
        #
        # self._trained = True
        # logger.info("Trained %s/%dh model on %d samples",
        #             self.elevation_band, self.lead_time_hours, len(features_df))

        # Numpy-only fallback: no correction
        self._trained = True
        logger.info(
            "Fallback mode (no xgboost): %s/%dh model marked trained on %d samples, "
            "returning identity correction",
            self.elevation_band, self.lead_time_hours, len(features_df),
        )

    def predict(self, features) -> float:
        """Return correction factor for given features.

        Args:
            features: flat dict or 1D array of feature values

        Returns:
            Correction factor (multiply forecast by this). 1.0 = no correction.
        """
        if not self._trained:
            return 1.0

        # TODO: uncomment when xgboost available
        # if self._model is not None:
        #     if isinstance(features, dict):
        #         arr = np.array([features.get(f, np.nan) for f in self._feature_names])
        #     else:
        #         arr = np.array(features)
        #     arr = arr.reshape(1, -1)
        #     # Replace NaN with 0 for prediction
        #     arr = np.nan_to_num(arr, nan=0.0)
        #     return float(self._model.predict(arr)[0])

        # Numpy-only fallback: no correction
        return 1.0

    def save(self, path: str):
        """Save trained model to disk using joblib.

        Args:
            path: Directory to save model files into.
        """
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(
            path, f"correction_{self.elevation_band}_{self.lead_time_hours}h.json"
        )
        state = {
            "elevation_band": self.elevation_band,
            "lead_time_hours": self.lead_time_hours,
            "trained": self._trained,
            "feature_names": self._feature_names,
            "feature_importances": self._feature_importances,
        }
        tmp = filename + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, filename)

        # TODO: uncomment when xgboost available
        # if self._model is not None:
        #     import joblib
        #     model_file = os.path.join(
        #         path, f"correction_{self.elevation_band}_{self.lead_time_hours}h.joblib"
        #     )
        #     joblib.dump(self._model, model_file)

    @classmethod
    def load(cls, path: str, elevation_band: str, lead_time_hours: int) -> "SnowCorrectionModel":
        """Load a saved model from disk.

        Args:
            path: Directory containing saved model files.
            elevation_band: Elevation band name (e.g., "base", "mid", "peak").
            lead_time_hours: Lead time in hours.

        Returns:
            Loaded SnowCorrectionModel instance.
        """
        instance = cls(elevation_band, lead_time_hours)
        filename = os.path.join(
            path, f"correction_{elevation_band}_{lead_time_hours}h.json"
        )
        if not os.path.exists(filename):
            return instance

        try:
            with open(filename) as f:
                state = json.load(f)
            instance._trained = state.get("trained", False)
            instance._feature_names = state.get("feature_names", [])
            instance._feature_importances = state.get("feature_importances", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load model state %s: %s", filename, e)

        # TODO: uncomment when xgboost available
        # model_file = os.path.join(
        #     path, f"correction_{elevation_band}_{lead_time_hours}h.joblib"
        # )
        # if os.path.exists(model_file):
        #     import joblib
        #     instance._model = joblib.load(model_file)

        return instance

    def get_feature_importance(self) -> dict:
        """Return sorted feature importance dict.

        Returns:
            Dict of {feature_name: importance_score}, sorted descending by importance.
        """
        return dict(
            sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)
        )


class MLPipeline:
    """Orchestrates training and application of ML correction models.

    Manages per-elevation-band, per-lead-time correction models that
    learn from accumulated forecast verification data.
    """

    def __init__(self, config: Optional[MLPipelineConfig] = None):
        self.config = config or MLPipelineConfig()
        self._models: dict = {}  # (elevation_band, lead_time) -> SnowCorrectionModel
        self._extractor = FeatureExtractor()

    def is_ready(self) -> bool:
        """Check if enough verification data exists to train models.

        Reads .forecast_verification.json and .snow_verification.json to
        count the number of days with verification data.
        """
        days = self._count_verification_days()
        return days >= self.config.min_training_days

    def _count_verification_days(self) -> int:
        """Count distinct days of verification data across all sources."""
        dates = set()

        # Read forecast verification log
        if os.path.exists(VERIFICATION_LOG):
            try:
                with open(VERIFICATION_LOG) as f:
                    data = json.load(f)
                for score in data.get("daily_scores", []):
                    date = score.get("date")
                    if date:
                        dates.add(date)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", VERIFICATION_LOG, e)

        # Read snow verification log
        if os.path.exists(SNOW_VERIFICATION):
            try:
                with open(SNOW_VERIFICATION) as f:
                    data = json.load(f)
                for entry in data if isinstance(data, list) else data.get("entries", []):
                    date = entry.get("date") or entry.get("obs_date")
                    if date:
                        dates.add(date)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read %s: %s", SNOW_VERIFICATION, e)

        return len(dates)

    def build_training_data(self) -> tuple:
        """Read verification logs and extract features for training.

        Returns:
            Tuple of (features_list, targets_dict) where features_list is
            a list of flat feature dicts and targets_dict maps target variable
            names to lists of observed values.
        """
        if not os.path.exists(VERIFICATION_LOG):
            return [], {}

        try:
            with open(VERIFICATION_LOG) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return [], {}

        scores = data.get("daily_scores", [])
        if not scores:
            return [], {}

        # Group scores by date for building features
        by_date = {}
        for s in scores:
            date = s.get("date", "")
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(s)

        features_list = []
        targets = {var: [] for var in self.config.target_variables}

        for date, day_scores in by_date.items():
            # Build a pseudo-analysis dict from the verification scores
            # This is a simplified version — in production, we'd store full
            # analysis snapshots alongside verification data
            pseudo_analysis = self._build_pseudo_analysis(day_scores, date)
            pseudo_verification = {"date": date, "scores": day_scores}

            features = self._extractor.extract(pseudo_analysis, pseudo_verification)
            features_list.append(features)

            # Extract target values from actuals
            for s in day_scores:
                metric = s.get("metric", "")
                actual = s.get("actual")
                if actual is not None and metric in targets:
                    targets[metric].append(actual)

        return features_list, targets

    def _build_pseudo_analysis(self, day_scores: list, date: str) -> dict:
        """Build a simplified analysis dict from verification scores.

        This reconstructs enough of the analyze_all() output structure
        for FeatureExtractor to work with.
        """
        # Extract per-model predictions
        model_preds = {}
        for s in day_scores:
            src = s.get("source", "").upper()
            metric = s.get("metric", "")
            predicted = s.get("predicted")
            if predicted is not None:
                if src not in model_preds:
                    model_preds[src] = {}
                model_preds[src][metric] = predicted

        # Build model_spread-like structure
        models = {}
        for src, preds in model_preds.items():
            models[src] = {
                "temp_high_f": preds.get("temp_high_f"),
                "snow_in": preds.get("snow_in", 0),
            }

        return {
            "generated": f"{date}T12:00:00+00:00",
            "resorts": {
                "Heavenly": {
                    "zones": {
                        "peak": {
                            "model_spread": [{"date": date, "models": models}],
                            "current": {},
                        }
                    }
                }
            },
            "current_conditions": {"observation": {}},
            "sounding": {},
            "ensemble": {},
        }

    def train_all_models(self):
        """Train correction models for each elevation band and lead time.

        Reads accumulated verification data, extracts features, and trains
        one model per (elevation_band, lead_time) combination.
        """
        features_list, targets = self.build_training_data()
        if not features_list:
            logger.warning("No training data available")
            return

        lead_times = [24, 48, 72]

        for low, high, band_name in self.config.elevation_bands:
            for lead_h in lead_times:
                model = SnowCorrectionModel(band_name, lead_h)
                model._feature_names = self.config.feature_list

                # Build feature matrix
                feature_matrix = []
                for f_dict in features_list:
                    row = [f_dict.get(feat, np.nan) for feat in self.config.feature_list]
                    feature_matrix.append(row)

                feature_array = np.array(feature_matrix, dtype=np.float64)

                # Use snow_in targets for snow correction models
                target_key = "snow_in"
                target_values = targets.get(target_key, [])

                if len(target_values) >= len(features_list):
                    target_array = np.array(target_values[:len(features_list)],
                                            dtype=np.float64)
                elif target_values:
                    # Pad with NaN if fewer targets than features
                    target_array = np.full(len(features_list), np.nan)
                    target_array[:len(target_values)] = target_values
                else:
                    # No targets for this variable; skip
                    continue

                # Remove samples with NaN targets
                valid_mask = ~np.isnan(target_array)
                if valid_mask.sum() < 10:
                    continue

                model.train(
                    feature_array[valid_mask],
                    target_array[valid_mask],
                )
                model.save(self.config.model_path)
                self._models[(band_name, lead_h)] = model

        logger.info("Trained %d models total", len(self._models))

    def apply_corrections(self, analysis: dict) -> dict:
        """Apply trained model corrections to a current forecast analysis.

        Args:
            analysis: Output of analyze_all()

        Returns:
            Modified analysis dict with ML corrections applied.
        """
        if not self._models:
            # Try loading saved models
            self._load_all_models()

        if not self._models:
            return analysis

        # Extract current features
        features = self._extractor.extract(analysis, {})

        # Apply corrections to each resort zone
        resorts = analysis.get("resorts", {})
        for resort_name, resort_data in resorts.items():
            zones = resort_data.get("zones", {})
            for zone_key, zone_data in zones.items():
                elev = zone_data.get("elev_ft", 0)
                band_name = self._get_elevation_band(elev)

                # Apply to timeline hours
                timeline = zone_data.get("timeline_48h", [])
                for i, hour in enumerate(timeline):
                    lead_h = self._quantize_lead_time(i)
                    model = self._models.get((band_name, lead_h))
                    if model and hour.get("snowfall_in", 0) > 0:
                        correction = model.predict(features)
                        hour["snowfall_in"] = round(
                            hour["snowfall_in"] * max(0.1, correction), 1
                        )
                        hour["ml_corrected"] = True

        return analysis

    def _get_elevation_band(self, elev_ft: int) -> str:
        """Map elevation to band name."""
        for low, high, name in self.config.elevation_bands:
            if low <= elev_ft < high:
                return name
        # Default to peak for elevations above all bands
        return self.config.elevation_bands[-1][2]

    def _quantize_lead_time(self, hour_index: int) -> int:
        """Map hour index to nearest lead time bucket."""
        if hour_index < 24:
            return 24
        elif hour_index < 48:
            return 48
        else:
            return 72

    def _load_all_models(self):
        """Load all saved models from disk."""
        if not os.path.exists(self.config.model_path):
            return

        lead_times = [24, 48, 72]
        for _, _, band_name in self.config.elevation_bands:
            for lead_h in lead_times:
                model = SnowCorrectionModel.load(
                    self.config.model_path, band_name, lead_h
                )
                if model._trained:
                    self._models[(band_name, lead_h)] = model

    def get_status(self) -> dict:
        """Return pipeline status for dashboard display.

        Returns:
            Dict with keys: ready, days_of_data, days_needed,
            models_trained, last_train_date, feature_importances.
        """
        days = self._count_verification_days()

        # Find most recent verification date
        last_date = None
        if os.path.exists(VERIFICATION_LOG):
            try:
                with open(VERIFICATION_LOG) as f:
                    data = json.load(f)
                dates = [s.get("date") for s in data.get("daily_scores", [])
                         if s.get("date")]
                if dates:
                    last_date = max(dates)
            except (json.JSONDecodeError, OSError):
                pass

        # Aggregate feature importances across models
        all_importances = {}
        for model in self._models.values():
            for feat, imp in model.get_feature_importance().items():
                if feat not in all_importances:
                    all_importances[feat] = []
                all_importances[feat].append(imp)

        avg_importances = {
            feat: round(np.mean(vals), 4)
            for feat, vals in all_importances.items()
        }

        return {
            "ready": days >= self.config.min_training_days,
            "days_of_data": days,
            "days_needed": self.config.min_training_days,
            "models_trained": len(self._models),
            "last_train_date": last_date,
            "feature_importances": dict(
                sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
            ),
        }


def _safe_float(value) -> float:
    """Convert a value to float, returning NaN for None/invalid."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# === Integration with tahoe_snow.py ===
# To activate ML corrections, add to analyze_all() after skill_weighted_blend():
#
#   from ml_pipeline import MLPipeline
#   ml = MLPipeline()
#   if ml.is_ready():
#       analysis = ml.apply_corrections(analysis)
#       analysis["ml_status"] = ml.get_status()
