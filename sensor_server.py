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
import logging
import sys
import os
import time
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pressure_forecast import record_pressure

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_data.json")

# Rate limiting: max 1 POST per 3 seconds per IP
RATE_LIMIT_SECONDS = 3
_rate_limit_lock = threading.Lock()
_last_post_by_ip: dict[str, float] = {}
_RATE_LIMIT_CLEANUP_INTERVAL = 300  # clean up stale entries every 5 min
_last_cleanup_time = 0.0


def _check_rate_limit(ip: str) -> bool:
    """Return True if the request should be allowed, False if rate-limited."""
    global _last_cleanup_time
    now = time.monotonic()
    with _rate_limit_lock:
        # Periodic cleanup of stale entries
        if now - _last_cleanup_time > _RATE_LIMIT_CLEANUP_INTERVAL:
            cutoff = now - RATE_LIMIT_SECONDS * 10
            stale = [k for k, v in _last_post_by_ip.items() if v < cutoff]
            for k in stale:
                del _last_post_by_ip[k]
            _last_cleanup_time = now

        last_time = _last_post_by_ip.get(ip, 0.0)
        if now - last_time < RATE_LIMIT_SECONDS:
            return False
        _last_post_by_ip[ip] = now
        return True


def load_data():
    try:
        with open(DATA_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_data(data):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, DATA_FILE)


class SensorHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/sensor":
            self.send_response(404)
            self.end_headers()
            return

        # Rate limiting per IP
        client_ip = self.client_address[0]
        if not _check_rate_limit(client_ip):
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b"Too many requests - max 1 POST per 3 seconds")
            logger.debug("Rate limited POST from %s", client_ip)
            return

        length = int(self.headers.get("Content-Length", 0))
        if length > 4096:
            self.send_response(413)
            self.end_headers()
            self.wfile.write(b"Request too large")
            return
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
