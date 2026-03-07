# ESP32 Sensor Configuration
# Edit these values before flashing to your ESP32

WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASS = "YOUR_WIFI_PASSWORD"

# IP address of the Raspberry Pi running sensor_server.py
PI_HOST = "192.168.1.100"
PI_PORT = 8081

# Set to "indoor" or "outdoor" — identifies this sensor to the Pi
SENSOR_LOCATION = "outdoor"

# How often to send readings (seconds)
REPORT_INTERVAL = 300  # 5 minutes

# BME280 I2C address (0x76 or 0x77 depending on board)
BME280_ADDR = 0x76
