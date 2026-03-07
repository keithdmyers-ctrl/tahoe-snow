"""
ESP32 Weather Sensor — MicroPython

Reads BME280 every REPORT_INTERVAL seconds and POSTs JSON to the Pi.
Set SENSOR_LOCATION in config.py to "indoor" or "outdoor".

Wiring (ESP32 -> BME280):
  3V3 -> VIN
  GND -> GND
  GPIO21 (SDA) -> SDA
  GPIO22 (SCL) -> SCL

Flash with: mpremote cp config.py bme280.py main.py :
"""

import machine
import network
import time
import json
import urequests

from config import (WIFI_SSID, WIFI_PASS, PI_HOST, PI_PORT,
                    REPORT_INTERVAL, BME280_ADDR, SENSOR_LOCATION)
from bme280 import BME280

led = machine.Pin(2, machine.Pin.OUT)


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        return wlan
    print(f"Connecting to {WIFI_SSID}...")
    wlan.connect(WIFI_SSID, WIFI_PASS)
    for _ in range(30):
        if wlan.isconnected():
            print(f"Connected: {wlan.ifconfig()[0]}")
            return wlan
        time.sleep(1)
    print("WiFi connection failed")
    return None


def blink(times=1, on_ms=100, off_ms=100):
    for _ in range(times):
        led.on()
        time.sleep_ms(on_ms)
        led.off()
        time.sleep_ms(off_ms)


def main():
    wlan = connect_wifi()
    if not wlan:
        machine.deepsleep(60000)

    i2c = machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21), freq=100000)
    try:
        sensor = BME280(i2c, addr=BME280_ADDR)
    except Exception as e:
        print(f"BME280 init failed: {e}")
        blink(5, 50, 50)
        machine.deepsleep(60000)

    url = f"http://{PI_HOST}:{PI_PORT}/sensor"

    while True:
        try:
            temp_c, humidity, pressure = sensor.read()
            temp_f = round(temp_c * 9 / 5 + 32, 1)

            payload = {
                "location": SENSOR_LOCATION,
                "temp_f": temp_f,
                "temp_c": temp_c,
                "humidity_pct": humidity,
                "pressure_hpa": pressure,
            }
            print(f"[{SENSOR_LOCATION}] {temp_f}F {humidity}% {pressure}hPa")

            resp = urequests.post(url, json=payload,
                                 headers={"Content-Type": "application/json"})
            if resp.status_code == 200:
                blink(1)
            else:
                print(f"HTTP {resp.status_code}")
                blink(3, 50, 50)
            resp.close()

        except Exception as e:
            print(f"Error: {e}")
            blink(3, 50, 50)
            if not wlan.isconnected():
                wlan = connect_wifi()

        time.sleep(REPORT_INTERVAL)


main()
