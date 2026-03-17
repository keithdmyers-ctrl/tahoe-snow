# Tahoe Snow / Weather Dashboard

## Overview
Multi-source weather forecasting app for Oakland local weather and Tahoe ski resort conditions. Deployed as a web app on Hugging Face Spaces and designed for a Raspberry Pi + Inky Impression 7.3" e-ink display.

## Deployment
- **Hugging Face**: Push to `keithdmyers` HF account after changes to webapp.py or weather logic
- **E-ink display**: Raspberry Pi 3/5 with Inky Impression 7.3", ESP32 for outdoor temp sensor

## Architecture
- **data_pipeline.py**: Shared data fetching orchestration (single source of truth for all consumers)
- **tahoe_snow.py**: Core physics, analysis pipeline, snow models (~4450 lines)
- **resort_configs.py**: Externalized resort configuration — 6 active resorts (Heavenly, Northstar, Kirkwood, Palisades, Sugar Bowl, Mt. Rose)
- **webapp.py**: Flask web app — `/api/data`, `/api/activities`, `/api/decision`
- **ml_corrections.py**: ML forecast corrections (shadow mode), snow quality, wind holds, powder alerts, resort ranking, chain forecasts
- **outdoor_conditions.py**: 10-activity decision engine (ski, BC ski, hike, MTB, fish, climb, paddle, cycle, camp, photo) with solar exposure, trail conditions, USGS stream flow, route wind, fishing pressure
- **feature_engineering.py**: ML training data builder (physics features, SNOTEL targets, forecast-aligned)
- **train_models.py**: XGBoost + Optuna + LOSO-CV + quantile regression (12 models)
- **historical_data_collector.py**: SNOTEL + ERA5 + archived GFS/ECMWF/ICON forecast collection
- **evaluate_models.py**: SHAP analysis, conditional bias, model comparison
- **eink_scenes.py**: E-ink scene manager — imports from data_pipeline.py
- **alerts.py**: Powder alert system — ML-enhanced with wind hold warnings
- **verify_cron.py**: Daily verification cron + ML shadow mode verification
- **forecast_verification.py**: Bias tracking, model skill scoring, per-model weights
- **pressure_forecast.py**: Zambretti barometric rain prediction, storm tracking
- **sensor_server.py**: HTTP server for ESP32 sensor data (port 8081)

## Key Data Sources
- NOAA/NWS (observations, forecast, gridpoints, alerts, AFD)
- Open-Meteo (GFS, ECMWF, ICON, HRRR + ensemble 80-member)
- SNOTEL (10 stations), CSSL (Donner Summit hourly)
- Synoptic/MesoWest mesonet (requires SYNOPTIC_TOKEN env var)
- Local BME280 sensors via ESP32
- Caltrans chains, Liftie lift status, avalanche.org

## After Changes
- Test the web app locally before pushing
- Verify e-ink display layout renders correctly
- Push to both GitHub and Hugging Face
- Check that API data sources are still responding
