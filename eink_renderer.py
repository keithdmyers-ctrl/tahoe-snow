#!/usr/bin/env python3
"""
HTML → E-Ink renderer via headless Chromium.

Renders a Jinja2 HTML template to an 800x480 PIL Image by:
  1. Rendering template with data → temp HTML file
  2. Screenshotting with headless Chromium
  3. Loading screenshot as PIL Image

Requires: chromium-browser (apt install chromium-browser)
"""

import os
import subprocess
import tempfile

from jinja2 import Environment, FileSystemLoader
from PIL import Image

DISPLAY_W = 800
DISPLAY_H = 480
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)


def _chromium_screenshot(html_path: str, png_path: str) -> bool:
    """Try Chromium-based browsers for screenshot. Returns True on success."""
    for browser in ("chromium-browser", "chromium", "google-chrome"):
        try:
            subprocess.run([
                browser,
                "--headless",
                f"--screenshot={png_path}",
                f"--window-size={DISPLAY_W},{DISPLAY_H}",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-software-rasterizer",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--hide-scrollbars",
                f"file://{html_path}",
            ], capture_output=True, timeout=30)
            if os.path.exists(png_path):
                return True
        except FileNotFoundError:
            continue
    return False


def _selenium_screenshot(html_path: str, png_path: str) -> bool:
    """Fallback: use Selenium with Firefox for screenshot."""
    try:
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        opts = Options()
        opts.add_argument("--headless")
        driver = webdriver.Firefox(options=opts)
        try:
            # Set content viewport to display dimensions
            # Firefox headless needs larger window to compensate for UI chrome
            driver.set_window_size(DISPLAY_W, DISPLAY_H + 150)
            # Use JS to set precise viewport via inner dimensions
            driver.execute_script(
                f"window.resizeTo({DISPLAY_W}, {DISPLAY_H + 150});"
            )
            driver.get(f"file://{html_path}")
            driver.save_screenshot(png_path)
            return os.path.exists(png_path)
        finally:
            driver.quit()
    except Exception:
        return False


def render_template(template_name: str, context: dict) -> Image.Image:
    """Render an HTML template to a PIL Image."""
    template = _env.get_template(template_name)
    html = template.render(**context)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        f.write(html)
        html_path = f.name

    png_path = os.path.splitext(html_path)[0] + ".png"

    try:
        if not _chromium_screenshot(html_path, png_path):
            if not _selenium_screenshot(html_path, png_path):
                raise RuntimeError(
                    "Screenshot failed. Install chromium-browser or firefox+selenium."
                )

        img = Image.open(png_path).convert("RGB")
        # Crop to exact display size (browser may add padding)
        img = img.crop((0, 0, DISPLAY_W, DISPLAY_H))
        return img

    finally:
        for p in (html_path, png_path):
            if os.path.exists(p):
                os.unlink(p)


def send_to_display(img: Image.Image):
    """Send PIL Image to Inky Impression."""
    try:
        from inky.auto import auto
        display = auto()
        display.set_image(img)
        display.show()
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"Display error: {e}")
        return False
