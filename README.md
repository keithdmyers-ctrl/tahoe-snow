---
title: Tahoe Snow
emoji: ❄️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# Tahoe Snow Dashboard

Multi-source weather forecasting for Oakland local weather and Lake Tahoe ski resort conditions. Runs as a web dashboard and on a Raspberry Pi with a 7.3" color e-ink display.

## Features

- **14+ data sources**: NWS, Open-Meteo (GFS/ECMWF/ICON/HRRR), SNOTEL, NBM, avalanche center, Caltrans chain controls, Weather Underground PWS, and more
- **Snow physics**: Temperature-dependent snow-to-liquid ratios (Roebber 2003), orographic enhancement, snow quality labels
- **Multi-model comparison**: Side-by-side GFS, ECMWF, ICON forecasts with confidence ratings
- **Local weather**: Barometric pressure rain prediction, indoor/outdoor sensors via ESP32
- **Forecast verification**: Automatic bias tracking and correction
- **Ski conditions**: Heavenly, Northstar, Kirkwood — per-zone forecasts (base/mid/peak)

## Quick Start

### Web Dashboard

```bash
pip install -r requirements.txt
python webapp.py                  # http://localhost:5000
python webapp.py --port 8080      # custom port
```

### E-Ink Display (Raspberry Pi)

See [EINK_BUILD_GUIDE.md](EINK_BUILD_GUIDE.md) for full hardware setup.

```bash
pip install requests numpy pillow jinja2 inky[rpi] smbus2 flask
python eink_scenes.py --scene oakland --preview    # preview without display
python eink_scenes.py --listen                     # button-driven daemon
```

### CLI Report

```bash
python tahoe_snow.py              # full text report
python tahoe_snow.py --compact    # shorter version
python tahoe_snow.py --json       # JSON output
```

## Documentation

- [EINK_BUILD_GUIDE.md](EINK_BUILD_GUIDE.md) — Hardware parts list, wiring, Pi setup
- [MODELS.md](MODELS.md) — Data sources, algorithms, snow physics
- [OAKLAND_DISPLAY_SPEC.md](OAKLAND_DISPLAY_SPEC.md) — E-ink display layout spec

## Deployment

The web dashboard deploys to [Hugging Face Spaces](https://huggingface.co/spaces) via Docker:

```bash
git remote add hf https://huggingface.co/spaces/keithdmyers/tahoe-snow
git push origin main && git push hf main
```
