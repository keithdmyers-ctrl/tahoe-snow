#!/usr/bin/env python3
"""
Sensor Receiver — lightweight HTTP server for ESP32 sensor data.

Receives POST /sensor from indoor and outdoor ESP32s.
Each POST includes {"location": "indoor"|"outdoor", "temp_f": ..., ...}
Saves latest readings to sensor_data.json.

Usage:
  python sensor_server.py              # listen on :8081
  python sensor_server.py --port 8081  # custom port
"""

import json
import sys
import os
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pressure_forecast import record_pressure

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_data.json")


def load_data():
    try:
        with open(DATA_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class SensorHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/sensor":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            reading = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Bad JSON")
            return

        location = reading.get("location", "unknown")
        if location not in ("indoor", "outdoor"):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"location must be 'indoor' or 'outdoor'")
            return

        data = load_data()
        data[location] = {
            "temp_f": reading.get("temp_f"),
            "temp_c": reading.get("temp_c"),
            "humidity_pct": reading.get("humidity_pct"),
            "pressure_hpa": reading.get("pressure_hpa"),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        save_data(data)

        # Record outdoor pressure for trend tracking / rain prediction
        if location == "outdoor" and reading.get("pressure_hpa"):
            record_pressure(reading["pressure_hpa"], reading.get("humidity_pct"))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "location": location}).encode())

    def do_GET(self):
        if self.path == "/sensor":
            data = load_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    port = 8081
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    server = HTTPServer(("0.0.0.0", port), SensorHandler)
    print(f"Sensor receiver listening on :{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
