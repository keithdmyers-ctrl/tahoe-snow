"""
Minimal BME280 MicroPython driver.

Usage:
    from machine import I2C, Pin
    from bme280 import BME280
    i2c = I2C(0, scl=Pin(22), sda=Pin(21))
    sensor = BME280(i2c, addr=0x76)
    temp_c, humidity, pressure_hpa = sensor.read()
"""

import struct
import time


class BME280:
    def __init__(self, i2c, addr=0x76):
        self.i2c = i2c
        self.addr = addr
        self._load_calibration()
        # Set config: standby 1000ms, filter coeff 4
        self.i2c.writeto_mem(self.addr, 0xF5, bytes([0x90]))
        # Set ctrl_hum: oversampling x1
        self.i2c.writeto_mem(self.addr, 0xF2, bytes([0x01]))
        # Set ctrl_meas: temp x2, press x2, normal mode
        self.i2c.writeto_mem(self.addr, 0xF4, bytes([0x4B]))
        time.sleep_ms(100)

    def _read_reg(self, reg, length):
        return self.i2c.readfrom_mem(self.addr, reg, length)

    def _load_calibration(self):
        # Temperature and pressure calibration (0x88-0xA1)
        cal1 = self._read_reg(0x88, 26)
        self.dig_T1 = struct.unpack_from("<H", cal1, 0)[0]
        self.dig_T2 = struct.unpack_from("<h", cal1, 2)[0]
        self.dig_T3 = struct.unpack_from("<h", cal1, 4)[0]
        self.dig_P1 = struct.unpack_from("<H", cal1, 6)[0]
        self.dig_P2 = struct.unpack_from("<h", cal1, 8)[0]
        self.dig_P3 = struct.unpack_from("<h", cal1, 10)[0]
        self.dig_P4 = struct.unpack_from("<h", cal1, 12)[0]
        self.dig_P5 = struct.unpack_from("<h", cal1, 14)[0]
        self.dig_P6 = struct.unpack_from("<h", cal1, 16)[0]
        self.dig_P7 = struct.unpack_from("<h", cal1, 18)[0]
        self.dig_P8 = struct.unpack_from("<h", cal1, 20)[0]
        self.dig_P9 = struct.unpack_from("<h", cal1, 22)[0]
        # Humidity calibration
        self.dig_H1 = cal1[25]
        cal2 = self._read_reg(0xE1, 7)
        self.dig_H2 = struct.unpack_from("<h", cal2, 0)[0]
        self.dig_H3 = cal2[2]
        self.dig_H4 = (cal2[3] << 4) | (cal2[4] & 0x0F)
        self.dig_H5 = (cal2[5] << 4) | ((cal2[4] >> 4) & 0x0F)
        self.dig_H6 = struct.unpack_from("<b", cal2, 6)[0]

    def read(self):
        """Read sensor. Returns (temp_c, humidity_pct, pressure_hpa)."""
        data = self._read_reg(0xF7, 8)
        raw_p = ((data[0] << 16) | (data[1] << 8) | data[2]) >> 4
        raw_t = ((data[3] << 16) | (data[4] << 8) | data[5]) >> 4
        raw_h = (data[6] << 8) | data[7]

        # Temperature
        v1 = (raw_t / 16384.0 - self.dig_T1 / 1024.0) * self.dig_T2
        v2 = ((raw_t / 131072.0 - self.dig_T1 / 8192.0) ** 2) * self.dig_T3
        t_fine = v1 + v2
        temp_c = t_fine / 5120.0

        # Pressure
        v1 = t_fine / 2.0 - 64000.0
        v2 = v1 * v1 * self.dig_P6 / 32768.0
        v2 = v2 + v1 * self.dig_P5 * 2.0
        v2 = v2 / 4.0 + self.dig_P4 * 65536.0
        v1 = (self.dig_P3 * v1 * v1 / 524288.0 + self.dig_P2 * v1) / 524288.0
        v1 = (1.0 + v1 / 32768.0) * self.dig_P1
        if v1 == 0:
            pressure = 0
        else:
            pressure = 1048576.0 - raw_p
            pressure = ((pressure - v2 / 4096.0) * 6250.0) / v1
            v1 = self.dig_P9 * pressure * pressure / 2147483648.0
            v2 = pressure * self.dig_P8 / 32768.0
            pressure = pressure + (v1 + v2 + self.dig_P7) / 16.0
        pressure_hpa = pressure / 100.0

        # Humidity
        h = t_fine - 76800.0
        if h == 0:
            humidity = 0
        else:
            h = (raw_h - (self.dig_H4 * 64.0 + self.dig_H5 / 16384.0 * h)) * \
                (self.dig_H2 / 65536.0 * (1.0 + self.dig_H6 / 67108864.0 * h *
                (1.0 + self.dig_H3 / 67108864.0 * h)))
            humidity = h * (1.0 - self.dig_H1 * h / 524288.0)
            humidity = max(0.0, min(100.0, humidity))

        return round(temp_c, 2), round(humidity, 1), round(pressure_hpa, 1)
