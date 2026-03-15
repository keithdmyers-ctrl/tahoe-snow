# Tahoe Snow / Weather Dashboard

## Overview
Multi-source weather forecasting app for Oakland local weather and Tahoe ski resort conditions. Deployed as a web app on Hugging Face Spaces and designed for a Raspberry Pi + Inky Impression 7.3" e-ink display.

## Deployment
- **Hugging Face**: Push to `keithdmyers` HF account after changes to webapp.py or weather logic
- **E-ink display**: Raspberry Pi 3/5 with Inky Impression 7.3", ESP32 for outdoor temp sensor

## Key Components
- Weather data aggregation from NOAA/NWS, Open-Meteo, SNOTEL, CDEC, Synoptic/MesoWest
- NWS raw gridpoint forecasts (forecaster-edited snow amounts, snow levels, QPF)
- Open-Meteo ensemble probabilistic forecasts (GFS 30-member, ECMWF 50-member)
- Synoptic/MesoWest mesonet stations (requires SYNOPTIC_TOKEN env var)
- Local barometric pressure prediction (BME280 sensor)
- Tahoe resort conditions: Heavenly, Northstar, Kirkwood
- E-ink display renderer with Oakland + Tahoe split layout

## After Changes
- Test the web app locally before pushing
- Verify e-ink display layout renders correctly
- Push to both GitHub and Hugging Face
- Check that API data sources are still responding
