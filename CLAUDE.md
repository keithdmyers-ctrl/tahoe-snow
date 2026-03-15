# Tahoe Snow / Weather Dashboard

## Overview
Multi-source weather forecasting app for Oakland local weather and Tahoe ski resort conditions. Deployed as a web app on Hugging Face Spaces and designed for a Raspberry Pi + Inky Impression 7.3" e-ink display.

## Deployment
- **Hugging Face**: Push to `keithdmyers` HF account after changes to webapp.py or weather logic
- **E-ink display**: Raspberry Pi 3/5 with Inky Impression 7.3", ESP32 for outdoor temp sensor

## Architecture
- **data_pipeline.py**: Shared data fetching orchestration (single source of truth for all consumers)
- **tahoe_snow.py**: Core physics, analysis pipeline, snow models (~4000 lines)
- **resort_configs.py**: Externalized resort configuration (RESORTS dict imported here)
- **webapp.py**: Flask web app — imports from data_pipeline.py
- **eink_scenes.py**: E-ink scene manager — imports from data_pipeline.py
- **alerts.py**: Powder alert system — imports from data_pipeline.py
- **verify_cron.py**: Daily verification cron — imports from data_pipeline.py
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
