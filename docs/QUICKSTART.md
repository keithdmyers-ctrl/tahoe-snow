# Tahoe Snow Weather Station — Quickstart Guide

A step-by-step guide for building the weather station from scratch. Written for someone who has never worked with a Raspberry Pi, ESP32, or electronics before.

## What You're Building

A color e-ink weather display that shows Oakland local weather + Tahoe ski resort conditions, updated every 30 minutes. Two wireless temperature sensors (indoor/outdoor) feed live readings to the display.

```
┌──────────────────────────────────────────────────┐
│ OAKLAND                      Thu Mar 05  07:31PM │
├──────────────────────────────────────────────────┤
│  INDOOR      OUTDOOR      TODAY'S HIGH           │
│   68°         54°           67°        NWS: 54°F │
│  47% rh      62% rh                   Sunny      │
├──────────────────────────────────────────────────┤
│ 5-DAY FORECAST                                   │
│ Tod 67°/50°  Fri 73°/55°  Sat 77°/53°  ...      │
├──────────────────────────────────────────────────┤
│ LAKE TAHOE    35°F          Snow:6755ft Avy:Low  │
├──────────────┬──────────────┬────────────────────┤
│ HEAVENLY     │ NORTHSTAR    │ KIRKWOOD           │
│ Pk 10067ft   │ Pk 8610ft    │ Pk 9800ft          │
├──────────────┴──────────────┴────────────────────┤
│ 5-DAY SNOW   Heav -- -- -- -- --                 │
│              Nort -- -- -- -- --                  │
│              Kirk -- -- -- -- --                  │
├──────────────────────────────────────────────────┤
│ PACK  Mt Rose S:69"  Squaw Val:48"  CSS Lab:41" │
└──────────────────────────────────────────────────┘
```

## What's in Your Boxes

| Box | What's Inside | What It Does |
|-----|---------------|--------------|
| **Raspberry Pi 3** | Small green circuit board (~credit card size) | The "brain" — runs all the software, drives the display |
| **Pi Power Supply** | Micro-USB cable + wall plug (5V/2.5A) | Powers the Pi (it has no battery) |
| **MicroSD Card** | Tiny card (~fingernail size), may include adapter | The Pi's "hard drive" — holds the operating system |
| **Inky Impression 7.3"** | E-ink display with a black connector on the back | The screen — like a Kindle, holds its image with no power |
| **2x ESP32 dev boards** | Small blue/black boards with a USB port and antenna | Wireless sensor brains — read temp and send it to the Pi |
| **2x BME280 breakout boards** | Tiny purple/blue boards (~1cm x 1.5cm) with 4 pins | Temperature + humidity + pressure sensors |
| **Dupont jumper wires (F-F)** | Colorful ribbon of wires with plastic connectors on both ends | Connect the sensors to the ESP32 boards (no soldering) |
| **Wyze Outdoor Power Adapter** | Weatherproof USB cable with a flat window-feed section | Powers the outdoor ESP32 — runs cable through a window seal |
| **IP65 junction box** | Small plastic weatherproof box with a lid | Outdoor enclosure for the ESP32 + sensor |

## How It All Connects

```
                        YOUR HOUSE
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────────┐         ┌─────────────────────┐       │
│  │  ESP32 #1   │  WiFi   │   Raspberry Pi 3    │       │
│  │  + BME280   │ ──────> │                     │       │
│  │  (indoor)   │         │  ┌─────────────────┐│       │
│  │  USB power  │         │  │ Inky Impression ││       │
│  └─────────────┘         │  │   7.3" e-ink    ││       │
│                          │  │   800 x 480     ││       │
│                          │  └─────────────────┘│       │
│                          └─────────────────────┘       │
│                                    ^                    │
│  OUTSIDE                           │ WiFi               │
│  ┌─────────────┐                   │                    │
│  │  ESP32 #2   │ ─────────────────┘                    │
│  │  + BME280   │                                        │
│  │  (outdoor)  │                                        │
│  │  Wyze USB   │                                        │
│  └─────────────┘                                        │
└─────────────────────────────────────────────────────────┘
```

The ESP32 sensors read temperature/humidity/pressure every 5 minutes and wirelessly send the data to the Pi. The Pi also pulls weather data from 10+ free online sources (NWS, Open-Meteo, SNOTEL, etc.) and renders everything to the e-ink display.

---

## Step 1: Flash the SD Card (on your laptop/desktop)

You need to put an operating system onto the SD card. This is called "flashing."

### What you need for this step
- Your laptop or desktop computer
- The MicroSD card (+ adapter if your laptop has a full-size SD slot)
- Internet connection

### Instructions

1. **Download and install** [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your laptop
2. **Insert** the MicroSD card into your laptop (use the adapter if needed)
3. **Open** Raspberry Pi Imager
4. Click **"Choose Device"** → select **Raspberry Pi 3**
5. Click **"Choose OS"** → **Raspberry Pi OS (other)** → **Raspberry Pi OS Lite (32-bit)**
   - "Lite" means no desktop — we don't need one since the Pi just runs scripts
   - 32-bit is recommended for the Pi 3's 1GB of RAM
6. Click **"Choose Storage"** → select your MicroSD card
7. Click **"Next"** → when asked **"Would you like to apply OS customisation settings?"** → click **"Edit Settings"**

### Fill in the settings (important!)

**General tab:**

| Setting | What to enter |
|---------|---------------|
| Set hostname | `weatherpi` |
| Set username and password | Username: `keith`, Password: pick something you'll remember |
| Configure wireless LAN | Your home WiFi network name + password |
| Wireless LAN country | `US` |
| Set locale settings | Timezone: `America/Los_Angeles` (or yours) |

**Services tab:**
- Check **"Enable SSH"**
- Select **"Use password authentication"**

> **What is SSH?** It lets you control the Pi from your laptop's terminal over WiFi — you type commands on your laptop, they run on the Pi. You won't need a monitor or keyboard plugged into the Pi.

8. Click **Save**, then **Yes** to apply, then **Yes** to write
9. Wait for it to finish (~5-10 minutes) — it writes the OS and then verifies it
10. **Eject** the SD card from your laptop

---

## Step 2: Boot the Pi

1. **Insert** the flashed MicroSD card into the Pi's card slot (on the underside of the board — it clicks in)
2. **Plug in** the micro-USB power supply — the Pi boots automatically (no power button)
3. **Wait ~90 seconds** for it to boot and connect to your WiFi

### Connect to the Pi from your laptop

Open a terminal on your laptop:
- **Mac**: open the Terminal app (in Applications → Utilities)
- **Windows**: open PowerShell or Command Prompt
- **Linux**: open your terminal emulator

Type:
```bash
ssh keith@weatherpi.local
```

- If it asks "Are you sure you want to continue connecting?" → type `yes` and press Enter
- Enter the password you set in Step 1

> **If `weatherpi.local` doesn't work:** Your router might not support mDNS. Try finding the Pi's IP address in your router's admin page (usually 192.168.1.x), then use `ssh keith@192.168.1.XXX` instead.

You should now see a command prompt like `keith@weatherpi:~ $` — you're controlling the Pi remotely!

**Everything from here on happens in this SSH session.**

---

## Step 3: Install Software on the Pi

Copy-paste these commands one block at a time into your SSH session. Each block does one thing.

### Update the operating system
```bash
sudo apt update && sudo apt upgrade -y
```
This takes 3-5 minutes. `sudo` means "run as administrator." It will ask for your password the first time.

### Install required system packages
```bash
sudo apt install -y python3-pip python3-venv git fonts-dejavu chromium-browser
```

### Add swap space (critical for Pi 3)
The Pi 3 only has 1GB of RAM. Chromium (used to render the display) needs more, so we add "swap" — disk space that acts as extra memory.

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=512/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

Verify it worked:
```bash
free -h
```
You should see a "Swap" row showing 512M.

### Download the project
```bash
git clone https://github.com/keithdmyers-ctrl/tahoe-snow.git ~/tahoe-snow
cd ~/tahoe-snow
```

### Set up Python and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests numpy pillow jinja2 inky[rpi] smbus2 flask gpiod
```

> **What is a virtual environment (venv)?** It's an isolated Python install so this project's packages don't interfere with anything else on the Pi. You'll always need to run `source .venv/bin/activate` before running project commands.

### Enable SPI (display communication)
```bash
sudo raspi-config nonint do_spi 0
```

> **What is SPI?** A communication protocol — it's how the Pi talks to the e-ink display through the GPIO pins. This command turns it on.

---

## Step 4: Attach the E-Ink Display

**Power off the Pi first!** Never connect/disconnect the display while powered on.

```bash
sudo shutdown now
```

Wait 10 seconds, then unplug the power cable.

### Physical connection

1. **Look at the back of the Inky Impression** — there's a black 40-pin female header (two rows of rectangular holes)
2. **Look at the Pi** — there's a matching 40-pin male header (two rows of gold pins)
3. **Align the connectors** — the display sits on top of the Pi like a hat. The display's connector lines up with the Pi's GPIO header. There's only one correct orientation — the display will cover the Pi board
4. **Press firmly and evenly** until the connector is fully seated — you should feel it click into place. Don't force it at an angle

### Power back on and test

1. Plug the power cable back in — the Pi boots with the display attached
2. SSH back in: `ssh keith@weatherpi.local`
3. Test the display is detected:

```bash
cd ~/tahoe-snow
source .venv/bin/activate
python3 -c "from inky.auto import auto; d = auto(); print(f'{d.width}x{d.height}')"
```

Should print: `800x480`

> **If it errors:** Check that SPI is enabled (`ls /dev/spidev*` should show devices). Check the display is firmly seated on the GPIO header. Try replugging it (power off first!).

---

## Step 5: Wire the BME280 Sensors to the ESP32s

Each ESP32 gets one BME280 sensor wired to it. The wiring is identical for both units.

### What the pins are

Your BME280 board has 4 pins labeled on the board. Your ESP32 board also has pin labels printed on it. You need to connect 4 pairs:

```
ESP32 Pin        BME280 Pin      Suggested Wire Color
─────────        ──────────      ────────────────────
3V3              VIN             Red    (power)
GND              GND             Black  (ground)
GPIO 21          SDA             Blue   (data)
GPIO 22          SCL             Yellow (clock)
```

### How to connect

1. Take 4 female-to-female Dupont jumper wires (they have plastic connectors on both ends)
2. Push one end of a wire onto an ESP32 pin, the other end onto the matching BME280 pin
3. They friction-fit — just push until snug, no tools needed
4. Match the 4 pairs in the table above

**Do this twice** — one ESP32+BME280 pair for indoors, one for outdoors.

> **Which side of the ESP32 has GPIO 21/22?** Look at the labels printed on the board. They're usually on the right side. If your board labels them differently (like D21/D22), those are the same pins.

> **What are SDA and SCL?** They're the two wires of the I2C communication protocol — SDA carries data, SCL carries a timing signal (clock). The ESP32 reads the sensor through these wires.

---

## Step 6: Flash MicroPython onto the ESP32s (on your laptop)

The ESP32s come with no software. You need to install MicroPython (a lightweight Python) and then copy the sensor code onto them.

**Do this from your laptop, NOT the Pi.** The ESP32 connects to your laptop via USB.

### Install tools on your laptop

```bash
pip install mpremote esptool
```

### Download MicroPython firmware

Go to https://micropython.org/download/ESP32_GENERIC/ and download the latest `.bin` file (e.g., `ESP32_GENERIC-20240602-v1.23.0.bin`). Save it somewhere you can find it.

### Flash the first ESP32

1. **Plug the ESP32 into your laptop via USB** (micro-USB or USB-C depending on your board)

2. **Find the serial port:**
   - **Linux**: usually `/dev/ttyUSB0` — run `ls /dev/ttyUSB*` to check
   - **Mac**: `/dev/cu.usbserial-*` or `/dev/cu.SLAB_USBtoUART` — run `ls /dev/cu.*` to check
   - **Windows**: `COM3` or similar — check Device Manager → Ports

3. **Erase and flash** (replace `/dev/ttyUSB0` with your actual port):
   ```bash
   esptool.py --port /dev/ttyUSB0 erase_flash
   esptool.py --port /dev/ttyUSB0 write_flash -z 0x1000 ESP32_GENERIC-*.bin
   ```

   > **If it hangs on "Connecting...":** Hold the **BOOT** button on the ESP32 board (small button near the USB port) while the command is connecting. Release once you see progress.

4. **Edit the config for this ESP32.** In the project folder on your laptop, edit `esp32/config.py`:
   ```python
   WIFI_SSID = "YourWiFiName"        # Your 2.4GHz WiFi network name
   WIFI_PASS = "YourWiFiPassword"    # WiFi password
   PI_HOST = "192.168.1.XXX"         # Your Pi's IP (run 'hostname -I' on the Pi to find it)
   PI_PORT = 8081
   SENSOR_LOCATION = "indoor"        # Set to "indoor" for the first ESP32
   REPORT_INTERVAL = 300             # 5 minutes between readings
   BME280_ADDR = 0x76                # Try 0x77 if 0x76 doesn't work
   ```

   > **How to find the Pi's IP:** In your SSH session to the Pi, run `hostname -I`. It will print something like `192.168.1.42` — use that number.

   > **2.4GHz only!** Most ESP32 boards cannot connect to 5GHz WiFi networks. If your router broadcasts both, make sure you're using the 2.4GHz network name (sometimes it has "-2G" in the name).

5. **Copy the code onto the ESP32:**
   ```bash
   mpremote connect /dev/ttyUSB0 cp esp32/config.py esp32/bme280.py esp32/main.py :
   ```

6. **Unplug and replug the ESP32** — it reboots, connects to WiFi, and starts reading the sensor

### Flash the second ESP32

Repeat steps 1-6 for the second ESP32, but change one thing in `esp32/config.py`:
```python
SENSOR_LOCATION = "outdoor"    # This one goes outside
```

### How to tell it's working

- **1 blink** every 5 minutes = successful reading sent to Pi
- **3 rapid blinks** = something failed (WiFi, sensor, or Pi not reachable)
- **5 rapid blinks** on boot = BME280 sensor not detected (check wiring)

---

## Step 7: Start the Sensor Server on the Pi

The sensor server is a small program on the Pi that listens for data from the ESP32s.

SSH into the Pi and test it:

```bash
cd ~/tahoe-snow
source .venv/bin/activate

# Start the server manually to test
python3 sensor_server.py
```

You should see: `Sensor receiver listening on :8081`

Wait up to 5 minutes — you should see readings arrive from your ESP32s. Press `Ctrl+C` to stop.

### Set it up to run automatically on boot

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

Verify it's running:
```bash
sudo systemctl status sensor-server
```

You should see "active (running)" in green.

> **What is systemd?** It's the Linux service manager. By creating a "service file," we tell the Pi to automatically start the sensor server on boot, and restart it if it ever crashes.

---

## Step 8: Test the Display

This is the moment of truth.

```bash
cd ~/tahoe-snow
source .venv/bin/activate

# Generate a preview image first (saves a PNG, doesn't need the display)
python3 eink_scenes.py --scene oakland --preview

# Now render to the actual e-ink display
python3 eink_scenes.py --scene oakland
```

The display will slowly draw the weather data. **This takes 45-60 seconds** — that's normal for 7-color e-ink (it's like watching an inkjet printer fill in the image). Don't unplug or interrupt it.

### Try all three scenes

```bash
python3 eink_scenes.py --scene oakland     # Oakland local weather
python3 eink_scenes.py --scene heavenly    # Tahoe ski resort conditions
python3 eink_scenes.py --scene detail      # Mountain zone deep-dive
```

---

## Step 9: Set Up Automatic Updates

You want the display to refresh on its own every 30 minutes.

### Add a cron job

```bash
crontab -e
```

If it asks which editor to use, choose **1** (nano — the simplest one).

Add this line at the very bottom of the file:
```
*/30 * * * * cd /home/keith/tahoe-snow && .venv/bin/python3 eink_scenes.py --refresh >> /tmp/eink.log 2>&1
```

Save and exit: press `Ctrl+O`, then `Enter`, then `Ctrl+X`.

> **What is cron?** It's a built-in Linux scheduler. The line above says "every 30 minutes, run the display refresh script." The `>> /tmp/eink.log` part saves any output to a log file so you can debug later.

### Enable button switching

The Inky Impression has 4 physical buttons (A, B, C, D) on its edge. Set up a service to listen for button presses:

```bash
sudo tee /etc/systemd/system/eink-buttons.service << 'EOF'
[Unit]
Description=E-Ink Button Listener
After=network.target

[Service]
Type=simple
User=keith
WorkingDirectory=/home/keith/tahoe-snow
ExecStart=/home/keith/tahoe-snow/.venv/bin/python3 eink_scenes.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable eink-buttons
sudo systemctl start eink-buttons
```

### Button map

| Button | What it does |
|--------|-------------|
| **A** | Show Oakland local weather |
| **B** | Show Heavenly/Tahoe ski conditions |
| **C** | Force refresh the current scene |
| **D** | Show mountain zone detail view |

---

## Step 10: Mount the Outdoor Sensor

### Prepare the enclosure

1. **Drill 2-3 small holes (3mm / 1/8")** in the **bottom** of the junction box — this lets air flow over the sensor for accurate readings and lets moisture drain out
2. Place the outdoor ESP32 + BME280 inside the box
3. Run the Wyze outdoor USB cable through the cable entry hole
4. Close the lid

### Mount it outside

1. Run the Wyze cable from an indoor outlet, through a window or wall gap, to the enclosure
   - The Wyze cable has a flat section designed to fit through a window seal
2. Mount the box on a **north-facing wall, under the eaves**
3. Mount **4-6 feet off the ground**

### Placement rules (important for accurate readings)

| Do | Don't |
|----|-------|
| Mount in full shade | Mount in direct sunlight (reads 20-30°F too high!) |
| North-facing wall, under roof overhang | South or west-facing wall |
| Away from walls that radiate heat | Near dryer vents, AC exhaust, or chimney |
| Bottom-drilled ventilation holes | Seal the box airtight (traps heat) |
| 4-6 feet above ground | On the ground or roof level |

---

## You're Done!

Your weather station is now:
- Reading indoor/outdoor temperature, humidity, and pressure every 5 minutes
- Pulling forecasts from 10+ weather sources (NWS, Open-Meteo, SNOTEL, avalanche centers)
- Rendering a beautiful 7-color e-ink display every 30 minutes
- Switchable between Oakland weather and Tahoe ski conditions via buttons

### Optional extras

**Powder alerts** — get notified when big snow is forecast:
```bash
cd ~/tahoe-snow && source .venv/bin/activate
nano alerts_config.json    # edit thresholds
python3 alerts.py          # test it
```
Add to cron for automatic alerts:
```
*/30 * * * * cd /home/keith/tahoe-snow && .venv/bin/python3 alerts.py >> /tmp/alerts.log 2>&1
```

**Web dashboard** — view the same data in a browser:
```bash
python3 webapp.py
```
Then open `http://weatherpi.local:5000` in your browser.

---

## Troubleshooting

### Pi won't connect / can't SSH

| Symptom | Fix |
|---------|-----|
| `ssh: Connection refused` | Pi is still booting — wait 90 seconds and try again |
| `ssh: No route to host` | Pi didn't connect to WiFi — re-flash the SD card, double-check WiFi name/password (it's case-sensitive) |
| `weatherpi.local` doesn't resolve | Your router may not support mDNS — find the Pi's IP in your router admin page and use `ssh keith@192.168.1.XXX` |

### Display problems

| Symptom | Fix |
|---------|-----|
| Display test errors about SPI | Run `sudo raspi-config nonint do_spi 0` then `sudo reboot` |
| Display shows nothing | Check it's firmly seated on the GPIO header. Power off, reseat, power on |
| Display shows "--" for temps | Sensor data hasn't arrived — check `sudo systemctl status sensor-server` |
| Display shows "stale" sensor data | ESP32 hasn't reported in 15+ min — check its power and WiFi |
| Rendering hangs or crashes | Pi 3 memory issue — verify swap is enabled with `free -h` (should show 512M swap). Re-run the swap setup from Step 3 if missing |

### ESP32 problems

| Symptom | Fix |
|---------|-----|
| `esptool.py` hangs on "Connecting..." | Hold the **BOOT** button on the ESP32 while it connects |
| 5 rapid blinks on boot | BME280 not detected — check wiring (SDA↔SDA, SCL↔SCL, not swapped). Try `BME280_ADDR = 0x77` in config |
| 3 blinks every 5 minutes | Can't reach the Pi — verify WiFi is 2.4GHz, check `PI_HOST` IP is correct |
| No blinks at all | Not running — check USB power, try unplugging and replugging |
| Outdoor temp way too high | Sensor is in direct sunlight — must be in shade |

### Checking logs

```bash
# E-ink display log
cat /tmp/eink.log

# Sensor server status
sudo systemctl status sensor-server
sudo journalctl -u sensor-server --since "1 hour ago"

# Button listener status
sudo systemctl status eink-buttons
sudo journalctl -u eink-buttons --since "1 hour ago"

# Current sensor data
cat ~/tahoe-snow/sensor_data.json
```

---

## Glossary

| Term | What It Means |
|------|---------------|
| **SSH** | Remote terminal — type commands on your laptop that run on the Pi |
| **GPIO** | General Purpose Input/Output — the 40 gold pins on the Pi |
| **SPI** | Serial Peripheral Interface — how the Pi talks to the e-ink display |
| **I2C** | Inter-Integrated Circuit — how the ESP32 talks to the BME280 sensor (uses SDA + SCL wires) |
| **MicroPython** | A lightweight version of Python that runs on tiny microcontrollers like the ESP32 |
| **systemd** | Linux service manager — keeps programs running and starts them on boot |
| **cron** | Linux task scheduler — runs commands on a timer (every 30 min in our case) |
| **Flashing** | Writing software onto a device's storage (SD card or ESP32 chip) |
| **BME280** | A sensor chip that measures temperature, humidity, and barometric pressure |
| **E-ink** | Electronic ink display technology (like a Kindle) — holds its image with no power, slow to refresh |
| **Swap** | Disk space used as overflow memory when RAM runs out |
| **venv** | Python virtual environment — an isolated set of packages for one project |
| **mDNS** | Multicast DNS — what lets you use `weatherpi.local` instead of an IP address |

---

## Pi 3 Performance Notes

The Raspberry Pi 3 (1GB) is fully capable of running this project. Here's what to expect compared to newer Pi models:

| Task | Pi 3 | Pi 5 |
|------|------|------|
| Display rendering | ~45-60 seconds | ~30 seconds |
| Data fetching (all sources) | ~30-45 seconds | ~15-20 seconds |
| Boot time | ~45 seconds | ~20 seconds |
| pip install (first time) | ~10-15 minutes | ~3-5 minutes |
| `apt upgrade` | ~10-20 minutes | ~5 minutes |

The 30-minute cron interval gives plenty of headroom. The Pi 3's WiFi (802.11n) is slower than Pi 5's (802.11ac) but more than sufficient for the small API requests this project makes.

**Memory management:** With 512MB of swap configured, the Pi 3 has effectively 1.5GB of usable memory. Chromium headless rendering is the most memory-intensive operation (~300MB). The sensor server and button listener use negligible memory (~15MB each).
