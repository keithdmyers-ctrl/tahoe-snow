# E-Ink Weather & Snow Display — Build Guide

A Raspberry Pi-powered e-ink dashboard showing local Oakland weather (indoor/outdoor temps, forecast) and Lake Tahoe snow conditions on a 7.3" color e-ink display.

## Parts List

| Part | Model | ~Price | Notes |
|------|-------|--------|-------|
| Single-board computer | Raspberry Pi 3 Model B (1GB) | $35 | Runs display, sensor server, data fetching |
| E-ink display | Pimoroni Inky Impression 7.3" (2025 Edition) | $85 | 800x480, 7-color Spectra 6, plugs onto Pi GPIO |
| MicroSD card | Any 32GB+ Class 10 | $8 | For Raspberry Pi OS |
| Pi power supply | Micro-USB 5V/2.5A | $10 | Official Pi 3 PSU or any good 2.5A micro-USB adapter |
| Indoor sensor board | ESP32 dev board (any) | $8 | Reads indoor BME280, POSTs to Pi over WiFi |
| Outdoor sensor board | ESP32 dev board (any) | $8 | Reads outdoor BME280, POSTs to Pi over WiFi |
| Indoor temp sensor | BME280 breakout (I2C) | $8 | Temp + humidity + pressure |
| Outdoor temp sensor | BME280 breakout (I2C) | $8 | Temp + humidity + pressure |
| Outdoor power | Wyze Outdoor Power Adapter (IP67) | $15 | Weatherproof USB, long cable run |
| Outdoor enclosure | IP65 junction box / weatherproof case | $5 | Houses outdoor ESP32 + BME280 |
| Wiring | Dupont jumper wires (F-F) | $3 | ESP32 to BME280 connections |
| **Total** | | **~$195** | |

### Where to buy

- **Inky Impression 7.3"**: [Pimoroni](https://shop.pimoroni.com/en-us/products/inky-impression-7-3), [PiShop.us](https://www.pishop.us/product/inky-impression-7-3-2025-edition/), [The Pi Hut](https://thepihut.com/products/inky-impression-7-3-2025-edition)
- **Pi 3**: [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-3-model-b/), Amazon, Micro Center — also check used/refurbished
- **ESP32 + BME280**: Amazon, AliExpress, Adafruit, SparkFun
- **Wyze Outdoor Power Adapter**: [wyze.com](https://www.wyze.com/products/wyze-outdoor-power-adapter)

## Architecture

```
ESP32 #1 (indoor)                  Raspberry Pi 3
  BME280 sensor                      Inky Impression 7.3" (SPI/GPIO)
  USB powered                        |
       |                             eink_scenes.py (cron, every 30min)
  WiFi POST ────────────────>          ├── reads sensor_data.json
                                       ├── fetches NWS Oakland forecast
ESP32 #2 (outdoor)                     ├── fetches Tahoe snow analysis
  BME280 sensor                        │     (Open-Meteo, NWS, SNOTEL,
  Wyze outdoor USB                     │      Avalanche, Forecast Discussion)
       |                               └── renders 800x480 image to display
  WiFi POST ────────────────>
                              sensor_server.py (systemd, port 8081)
                                └── saves to sensor_data.json
```

## Display Layout

```
┌──────────────────────────────────────────────────┐
│ OAKLAND                      Thu Mar 05  07:31PM │ header
├──────────────────────────────────────────────────┤
│                                                  │
│  INDOOR      OUTDOOR      TODAY'S HIGH           │
│   68°         54°           67°        NWS: 54°F │ big temps
│  47% rh      62% rh                   Sunny      │
│                                                  │
├──────────────────────────────────────────────────┤
│ 5-DAY FORECAST                                   │
│ Tod 67°/50°  Fri 73°/55°  Sat 77°/53°  ...      │ local forecast
│ Sunny        Sunny        Sunny                  │
├──────────────────────────────────────────────────┤
│ LAKE TAHOE    35°F          Snow:6755ft Avy:Low  │ tahoe header
├──────────────┬──────────────┬────────────────────┤
│ HEAVENLY     │ NORTHSTAR    │ KIRKWOOD           │
│ Pk 10067ft   │ Pk 8610ft    │ Pk 9800ft          │ resort data
│ Bs 6540ft    │ Bs 6330ft    │ Bs 7800ft          │ (peak + base)
├──────────────┴──────────────┴────────────────────┤
│ 5-DAY SNOW   Heav -- -- -- -- --                 │ snow forecast
│              Nort -- -- -- -- --                  │
│              Kirk -- -- -- -- --                  │
├──────────────────────────────────────────────────┤
│ PACK  Mt Rose S:69"  Squaw Val:48"  CSS Lab:41" │ snowpack strip
└──────────────────────────────────────────────────┘
```

Top ~60% is local conditions. Bottom ~40% is Tahoe.

## File Overview

```
tahoe-snow/
├── tahoe_snow.py          # Core analyzer — 7 data sources, 3 resorts, physics engine
├── eink_scenes.py        # E-ink renderer — fetches all data, renders to Inky Impression
├── sensors.py             # Reads indoor/outdoor sensor data from sensor_data.json
├── sensor_server.py       # HTTP server receiving ESP32 POSTs (port 8081)
├── sensor_data.json       # Latest sensor readings (written by sensor_server.py)
├── alerts.py              # Powder alert system (cron-based, optional)
├── alerts_config.json     # Alert thresholds and notification config
├── webapp.py              # Flask web API (optional, for browser dashboard)
├── esp32/
│   ├── main.py            # MicroPython — reads BME280, POSTs to Pi
│   ├── bme280.py          # MicroPython BME280 I2C driver
│   └── config.py          # WiFi credentials, Pi IP, sensor location
└── test_tahoe_snow.py     # Test suite
```

## Setup Instructions

### Step 1: Raspberry Pi

1. Flash **Raspberry Pi OS Lite (32-bit)** to the MicroSD card using Raspberry Pi Imager
   - In Imager, choose device **Raspberry Pi 3** → OS **Raspberry Pi OS (other)** → **Raspberry Pi OS Lite (32-bit)**
   - 32-bit is recommended for Pi 3's 1GB RAM
2. In the OS customisation settings, enable SSH, set WiFi credentials, set username to `keith`
3. Boot the Pi, SSH in: `ssh keith@raspberrypi.local`

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv git fonts-dejavu chromium-browser

# IMPORTANT: Pi 3 has only 1GB RAM — add a swap file so Chromium doesn't crash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=512/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Clone the project
git clone <your-repo-url> ~/projects/tahoe-snow
cd ~/projects/tahoe-snow

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install requests numpy pillow jinja2 inky[rpi] smbus2 flask gpiod
```

> **Install path note:** The shipped systemd service files assume installation at `/home/keith/projects/tahoe-snow`. If you cloned elsewhere, update the paths in `services/tahoe-eink.service` and `services/tahoe-sensors.service` before copying them.

#### Synoptic/MesoWest API token (optional)

Synoptic/MesoWest provides RWIS road weather stations and mesonet station data. It requires a free API token.

1. Register at https://synopticdata.com (free tier)
2. Set the environment variable:
   ```bash
   export SYNOPTIC_TOKEN=your_token_here
   ```
3. Add to `.bashrc` for persistence, or add to the systemd service file's `Environment=` line:
   ```
   Environment=SYNOPTIC_TOKEN=your_token_here
   ```

If not set, those data sources return empty results (the system still works, just fewer sources).

> **Pi 3 performance note:** The Pi 3 is slower than Pi 5 but fully capable of running
> this project. Display rendering takes ~45-60 seconds (vs ~30 on Pi 5). Data fetching
> may take a bit longer. The 30-minute cron interval gives plenty of headroom.

### Step 2: Attach the Inky Impression

1. Power off the Pi
2. Press the Inky Impression onto the 40-pin GPIO header — it plugs straight on, no soldering
3. Power on the Pi

```bash
# Verify the display is detected
python3 -c "from inky.auto import auto; d = auto(); print(f'{d.width}x{d.height}')"
# Should print: 800x480
```

### Step 3: Flash the ESP32s

You need Python + mpremote on your dev machine (not the Pi):

```bash
pip install mpremote esptool

# Flash MicroPython firmware (one-time per ESP32)
# Download from https://micropython.org/download/ESP32_GENERIC/
esptool.py --port /dev/ttyUSB0 erase_flash
esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 ESP32_GENERIC-*.bin
```

For each ESP32:

1. Edit `esp32/config.py`:
   - Set `WIFI_SSID` and `WIFI_PASS`
   - Set `PI_HOST` to your Pi's local IP (find it with `hostname -I` on the Pi)
   - Set `SENSOR_LOCATION` to `"indoor"` or `"outdoor"`

2. Flash the files:
```bash
mpremote cp esp32/config.py esp32/bme280.py esp32/main.py :
```

3. Reset the ESP32 — it will connect to WiFi and start POSTing readings

### Step 4: Wire BME280 to each ESP32

Same wiring for both ESP32s:

```
ESP32           BME280
─────           ──────
3V3  ────────── VIN
GND  ────────── GND
GPIO21 (SDA) ── SDA
GPIO22 (SCL) ── SCL
```

4 wires per sensor. Dupont jumper wires work fine.

### Step 5: Start the sensor server on the Pi

```bash
cd ~/projects/tahoe-snow

# Test it manually first
.venv/bin/python3 sensor_server.py
# You should see "Sensor receiver listening on :8081"
# Check that ESP32 readings arrive: curl http://localhost:8081/sensor
```

Set up as a systemd service for auto-start using the shipped service file:

```bash
sudo cp ~/projects/tahoe-snow/services/tahoe-sensors.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tahoe-sensors
```

See `services/README.md` for full service setup instructions.

### Step 6: Test the display

```bash
cd ~/projects/tahoe-snow

# Generate a preview image (saves eink_preview.png)
.venv/bin/python3 eink_scenes.py --preview

# Render to the actual display
.venv/bin/python3 eink_scenes.py
```

The display takes ~30-40 seconds to refresh (normal for 7-color e-ink).

### Step 7: Set up cron for automatic updates

```bash
crontab -e
```

Add:

```
# Update e-ink display every 30 minutes
*/30 * * * * cd /home/keith/projects/tahoe-snow && .venv/bin/python3 eink_scenes.py >> /tmp/eink.log 2>&1
```

### Step 8: Mount the outdoor sensor

1. Place the outdoor ESP32 + BME280 in the weatherproof enclosure
2. Run the Wyze outdoor USB cable from an indoor outlet through a window/wall to the enclosure
3. Mount in a **shaded spot** — north-facing wall, under eaves, out of direct sunlight
4. Seal the enclosure cable entry with silicone if needed

## Outdoor Sensor Placement Tips

- **Shade is critical** — direct sun will read 20-30°F high
- North-facing wall under eaves is ideal
- 4-6 feet off the ground, away from heat sources (dryer vents, AC units)
- The weatherproof enclosure should have small ventilation holes (not sealed airtight) so air circulates over the BME280
- Drill 2-3 small holes (3mm) in the bottom of the enclosure for airflow and moisture drainage

## Data Sources (all free, most require no API keys)

| Source | Data | Update Frequency |
|--------|------|-----------------|
| ESP32 sensors | Indoor/outdoor temp, humidity | Every 5 minutes |
| NWS API | Oakland forecast, current conditions | Every 30 minutes |
| Open-Meteo | 16-day multi-model Tahoe forecast (GFS, ECMWF, ICON) | Every 30 minutes |
| SNOTEL/NRCS | Snowpack depth, SWE at 10 stations | Every 30 minutes |
| Avalanche.org | Sierra Avalanche Center danger rating | Every 30 minutes |
| NWS Reno WFO | Forecaster discussion text | Every 30 minutes |

## Optional: Powder Alerts

The alert system sends notifications when snow thresholds are met:

```bash
# Edit thresholds
nano alerts_config.json

# Test
.venv/bin/python3 alerts.py

# Add to cron (every 30 min)
*/30 * * * * cd /home/keith/projects/tahoe-snow && .venv/bin/python3 alerts.py >> /tmp/alerts.log 2>&1
```

Supports desktop notifications (notify-send) and webhooks (Discord, Slack, ntfy.sh).

## Troubleshooting

**Display shows nothing after running eink_scenes.py:**
- Check SPI is enabled: `ls /dev/spidev*` should show devices
- Check the display is seated firmly on the GPIO header
- Try: `python3 -c "from inky.auto import auto; print(auto())"`

**Sensor data shows "--" on display:**
- Check sensor server is running: `systemctl status tahoe-sensors`
- Check ESP32 is POSTing: `curl http://localhost:8081/sensor`
- Check sensor_data.json has recent timestamps

**ESP32 won't connect to WiFi:**
- Verify SSID/password in config.py (2.4GHz only for most ESP32s)
- Check Pi IP hasn't changed (consider setting a static IP or using mDNS)

**Outdoor temp reads too high:**
- Sensor is in direct sunlight — move to shade
- Enclosure is sealed too tight — add ventilation holes

**Display shows "stale" (grayed out) sensor data:**
- ESP32 hasn't reported in >15 minutes
- Check WiFi connectivity, power supply, sensor wiring

**Chromium crashes or rendering hangs (Pi 3):**
- Check swap is enabled: `free -h` should show 512M swap
- If swap is missing, re-run the swap setup commands from Step 1
- Check available memory: `free -h` — rendering needs ~300MB free
- Reduce other running processes if needed
