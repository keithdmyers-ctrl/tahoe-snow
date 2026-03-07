# E-Ink Weather & Snow Display — Build Guide

A Raspberry Pi-powered e-ink dashboard showing local Oakland weather (indoor/outdoor temps, forecast) and Lake Tahoe snow conditions on a 7.3" color e-ink display.

## Parts List

| Part | Model | ~Price | Notes |
|------|-------|--------|-------|
| Single-board computer | Raspberry Pi 5 (2GB) | $60 | Runs display, sensor server, data fetching |
| E-ink display | Pimoroni Inky Impression 7.3" (2025 Edition) | $85 | 800x480, 7-color Spectra 6, plugs onto Pi GPIO |
| MicroSD card | Any 32GB+ Class 10 | $8 | For Raspberry Pi OS |
| Pi power supply | USB-C 27W (Pi 5 official) | $12 | Must be 5V/5A for Pi 5 |
| Indoor sensor board | ESP32 dev board (any) | $8 | Reads indoor BME280, POSTs to Pi over WiFi |
| Outdoor sensor board | ESP32 dev board (any) | $8 | Reads outdoor BME280, POSTs to Pi over WiFi |
| Indoor temp sensor | BME280 breakout (I2C) | $8 | Temp + humidity + pressure |
| Outdoor temp sensor | BME280 breakout (I2C) | $8 | Temp + humidity + pressure |
| Outdoor power | Wyze Outdoor Power Adapter (IP67) | $15 | Weatherproof USB, long cable run |
| Outdoor enclosure | IP65 junction box / weatherproof case | $5 | Houses outdoor ESP32 + BME280 |
| Wiring | Dupont jumper wires (F-F) | $3 | ESP32 to BME280 connections |
| **Total** | | **~$220** | |

### Where to buy

- **Inky Impression 7.3"**: [Pimoroni](https://shop.pimoroni.com/en-us/products/inky-impression-7-3), [PiShop.us](https://www.pishop.us/product/inky-impression-7-3-2025-edition/), [The Pi Hut](https://thepihut.com/products/inky-impression-7-3-2025-edition)
- **Pi 5**: [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-5/), Amazon, Micro Center
- **ESP32 + BME280**: Amazon, AliExpress, Adafruit, SparkFun
- **Wyze Outdoor Power Adapter**: [wyze.com](https://www.wyze.com/products/wyze-outdoor-power-adapter)

## Architecture

```
ESP32 #1 (indoor)                  Raspberry Pi 5
  BME280 sensor                      Inky Impression 7.3" (SPI/GPIO)
  USB powered                        |
       |                             eink_display.py (cron, every 30min)
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
├── eink_display.py        # E-ink renderer — fetches all data, renders to Inky Impression
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

1. Flash **Raspberry Pi OS Lite (64-bit)** to the MicroSD card using Raspberry Pi Imager
2. Enable SSH and set WiFi credentials during flashing
3. Boot the Pi, SSH in

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv git fonts-dejavu

# Clone the project
git clone <your-repo-url> ~/tahoe-snow
cd ~/tahoe-snow

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install requests numpy pillow inky[rpi] smbus2 flask
```

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
cd ~/tahoe-snow

# Test it manually first
.venv/bin/python3 sensor_server.py
# You should see "Sensor receiver listening on :8081"
# Check that ESP32 readings arrive: curl http://localhost:8081/sensor
```

Set up as a systemd service for auto-start:

```bash
sudo tee /etc/systemd/system/sensor-server.service << 'EOF'
[Unit]
Description=ESP32 Sensor Receiver
After=network.target

[Service]
Type=simple
User=keith
WorkingDirectory=/home/keith/tahoe-snow
ExecStart=/home/keith/tahoe-snow/.venv/bin/python3 sensor_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable sensor-server
sudo systemctl start sensor-server
```

### Step 6: Test the display

```bash
cd ~/tahoe-snow

# Generate a preview image (saves eink_preview.png)
.venv/bin/python3 eink_display.py --preview

# Render to the actual display
.venv/bin/python3 eink_display.py
```

The display takes ~30-40 seconds to refresh (normal for 7-color e-ink).

### Step 7: Set up cron for automatic updates

```bash
crontab -e
```

Add:

```
# Update e-ink display every 30 minutes
*/30 * * * * cd /home/keith/tahoe-snow && .venv/bin/python3 eink_display.py >> /tmp/eink.log 2>&1
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

## Data Sources (all free, no API keys)

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
*/30 * * * * cd /home/keith/tahoe-snow && .venv/bin/python3 alerts.py >> /tmp/alerts.log 2>&1
```

Supports desktop notifications (notify-send) and webhooks (Discord, Slack, ntfy.sh).

## Troubleshooting

**Display shows nothing after running eink_display.py:**
- Check SPI is enabled: `ls /dev/spidev*` should show devices
- Check the display is seated firmly on the GPIO header
- Try: `python3 -c "from inky.auto import auto; print(auto())"`

**Sensor data shows "--" on display:**
- Check sensor_server is running: `systemctl status sensor-server`
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
