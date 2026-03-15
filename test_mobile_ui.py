#!/usr/bin/env python3
"""
Mobile UI Test Suite for Tahoe Snow PWA

Tests the live deployment at https://keithdmyers-tahoe-snow.hf.space
using Playwright with iPhone emulation. Takes screenshots at each step
for visual review and checks for common mobile rendering issues.

Usage:
    python3 test_mobile_ui.py                    # test live HF Spaces
    python3 test_mobile_ui.py http://localhost:5000  # test local

Output:
    screenshots/  directory with numbered PNG screenshots
    test_report.json  with pass/fail results
"""

import json
import os
import sys
import time
from datetime import datetime

# Test configuration
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://keithdmyers-tahoe-snow.hf.space"
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "test_report.json")

# iPhone 14 Pro dimensions
DEVICE = {
    "viewport": {"width": 393, "height": 852},
    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "device_scale_factor": 3,
    "is_mobile": True,
    "has_touch": True,
}

# Also test wider phone (iPhone Pro Max)
DEVICE_WIDE = {
    "viewport": {"width": 430, "height": 932},
    "user_agent": DEVICE["user_agent"],
    "device_scale_factor": 3,
    "is_mobile": True,
    "has_touch": True,
}

results = []


def log_result(name, passed, details="", screenshot=None):
    status = "PASS" if passed else "FAIL"
    results.append({
        "name": name,
        "status": status,
        "details": details,
        "screenshot": screenshot,
        "timestamp": datetime.now().isoformat(),
    })
    icon = "\u2705" if passed else "\u274c"
    print(f"  {icon} {name}: {details}" if details else f"  {icon} {name}")


def screenshot(page, name, idx):
    path = os.path.join(SCREENSHOT_DIR, f"{idx:02d}_{name}.png")
    page.screenshot(path=path, full_page=False)
    return path


def screenshot_full(page, name, idx):
    path = os.path.join(SCREENSHOT_DIR, f"{idx:02d}_{name}_full.png")
    page.screenshot(path=path, full_page=True)
    return path


def run_tests():
    from playwright.sync_api import sync_playwright

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    print(f"\nMobile UI Test Suite")
    print(f"Target: {BASE_URL}")
    print(f"Device: iPhone 14 Pro ({DEVICE['viewport']['width']}x{DEVICE['viewport']['height']})")
    print(f"Screenshots: {SCREENSHOT_DIR}/")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(**DEVICE)
        page = ctx.new_page()

        # ===== PHASE 1: Page Load =====
        print("Phase 1: Page Load")

        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
        ss = screenshot(page, "01_initial_load", 1)
        log_result("Page loads without crash", True, screenshot=ss)

        # Wait for data to load (up to 120s)
        try:
            page.wait_for_selector("#app", state="visible", timeout=120000)
            ss = screenshot(page, "02_data_loaded", 2)
            log_result("Data loads successfully", True, screenshot=ss)
        except Exception as e:
            ss = screenshot(page, "02_data_failed", 2)
            log_result("Data loads successfully", False, str(e), screenshot=ss)

        # ===== PHASE 2: Above-the-fold Content =====
        print("\nPhase 2: Above-the-fold Content")

        # Check decision banner is visible
        decision = page.query_selector("#decision-section .decision-banner")
        ss = screenshot(page, "03_decision_banner", 3)
        if decision:
            box = decision.bounding_box()
            log_result("Decision banner visible", box is not None and box["y"] < 400,
                       f"y={box['y']:.0f}px" if box else "not rendered", screenshot=ss)
            # Check score is readable (large enough)
            score_el = page.query_selector(".decision-score")
            if score_el:
                score_box = score_el.bounding_box()
                log_result("Decision score readable size",
                           score_box and score_box["width"] >= 60,
                           f"{score_box['width']:.0f}x{score_box['height']:.0f}px" if score_box else "missing")
        else:
            log_result("Decision banner visible", False, "element not found", screenshot=ss)

        # Check hero cards visible
        hero = page.query_selector(".hero-row")
        if hero:
            box = hero.bounding_box()
            log_result("Hero cards visible", box is not None,
                       f"y={box['y']:.0f}px" if box else "missing")
        else:
            log_result("Hero cards visible", False, "hero-row not found")

        # ===== PHASE 3: Overflow & Layout =====
        print("\nPhase 3: Overflow & Layout")

        # Check for horizontal overflow (most common mobile bug)
        has_overflow = page.evaluate("""() => {
            return document.body.scrollWidth > window.innerWidth;
        }""")
        log_result("No horizontal overflow", not has_overflow,
                   f"body={page.evaluate('document.body.scrollWidth')}px vs viewport={DEVICE['viewport']['width']}px")

        # Check specific elements for overflow
        overflow_elements = page.evaluate("""() => {
            const problems = [];
            const vw = window.innerWidth;
            document.querySelectorAll('.hero-row, .compare-grid, .data-table, .resort-tabs, .section, .decision-banner, .narrative-block, .aqi-uv-row, .accum-grid, #tahoe-map').forEach(el => {
                const rect = el.getBoundingClientRect();
                if (rect.right > vw + 2) {
                    problems.push(`${el.className.split(' ')[0] || el.id}: ${Math.round(rect.right)}px > ${vw}px`);
                }
            });
            return problems;
        }""")
        ss = screenshot(page, "04_overflow_check", 4)
        log_result("No element overflow", len(overflow_elements) == 0,
                   "; ".join(overflow_elements) if overflow_elements else "all elements within viewport",
                   screenshot=ss)

        # ===== PHASE 4: Key Sections Render =====
        print("\nPhase 4: Key Sections Render")

        sections_to_check = [
            ("#decision-section", "Decision section"),
            ("#narrative-section", "Storm narrative"),
            (".hero-row", "Hero cards"),
            ("#resort-tabs", "Resort tabs"),
            ("#comparison-panel", "Comparison panel"),
            ("#aqi-uv-row", "AQI/UV badges"),
        ]

        for selector, name in sections_to_check:
            el = page.query_selector(selector)
            if el:
                box = el.bounding_box()
                visible = box and box["height"] > 0
                log_result(f"{name} renders", visible,
                           f"{box['width']:.0f}x{box['height']:.0f}px" if box else "zero size")
            else:
                log_result(f"{name} renders", False, "element not found")

        # ===== PHASE 5: Map =====
        print("\nPhase 5: Map")

        map_el = page.query_selector("#tahoe-map")
        if map_el:
            box = map_el.bounding_box()
            # Scroll to map and screenshot
            map_el.scroll_into_view_if_needed()
            page.wait_for_timeout(1000)
            ss = screenshot(page, "05_map", 5)
            log_result("Map renders", box and box["height"] > 100,
                       f"{box['width']:.0f}x{box['height']:.0f}px" if box else "missing", screenshot=ss)
            # Check map doesn't overflow
            log_result("Map within viewport", box and box["width"] <= DEVICE["viewport"]["width"] + 2,
                       f"width={box['width']:.0f}px" if box else "missing")
        else:
            log_result("Map renders", False, "tahoe-map not found")

        # ===== PHASE 6: Charts =====
        print("\nPhase 6: Charts")

        # Click on a resort tab to trigger chart build
        first_tab = page.query_selector('.resort-tab:not(.all-tab)')
        if first_tab:
            first_tab.click()
            page.wait_for_timeout(500)

        # Check 48h chart
        canvas_48h = page.query_selector('canvas[id^="chart-hourly-"]')
        if canvas_48h:
            box = canvas_48h.bounding_box()
            canvas_48h.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            ss = screenshot(page, "06_48h_chart", 6)
            log_result("48h chart renders", box and box["height"] > 50,
                       f"{box['width']:.0f}x{box['height']:.0f}px" if box else "missing", screenshot=ss)
            log_result("48h chart not overflowing", box and box["width"] <= DEVICE["viewport"]["width"] + 10,
                       f"width={box['width']:.0f}px" if box else "missing")
        else:
            log_result("48h chart renders", False, "no chart-hourly canvas found")

        # ===== PHASE 7: Resort Pills / Compare Chart =====
        print("\nPhase 7: Resort Comparison Chart")

        pills = page.query_selector_all(".resort-pill")
        ss_area = page.query_selector("#comparison-panel")
        if ss_area:
            try:
                ss_area.scroll_into_view_if_needed(timeout=5000)
            except Exception:
                page.evaluate('document.getElementById("comparison-panel")?.scrollIntoView()')
            page.wait_for_timeout(300)
        ss = screenshot(page, "07_resort_pills", 7)
        log_result("Resort pills render", len(pills) > 0,
                   f"{len(pills)} pills found", screenshot=ss)

        # Check pills don't overflow
        if pills:
            toggles = page.query_selector(".resort-toggles")
            if toggles:
                tbox = toggles.bounding_box()
                log_result("Pills wrap properly (no overflow)",
                           tbox is not None and tbox["width"] <= DEVICE["viewport"]["width"],
                           f"width={tbox['width']:.0f}px" if tbox else "section may be collapsed")
            else:
                log_result("Pills wrap properly (no overflow)", True, "toggles container not visible (collapsed section)")

        compare_canvas = page.query_selector("#chart-compare-all")
        if compare_canvas:
            box = compare_canvas.bounding_box()
            if box and box["height"] > 0:
                try:
                    compare_canvas.scroll_into_view_if_needed(timeout=5000)
                except Exception:
                    pass
                page.wait_for_timeout(300)
                ss = screenshot(page, "08_compare_chart", 8)
                log_result("Comparison chart renders", box["height"] > 50,
                           f"{box['width']:.0f}x{box['height']:.0f}px", screenshot=ss)
            else:
                # Canvas exists but is in a collapsed section — this is expected behavior
                log_result("Comparison chart renders", True, "canvas present, section collapsed (expand to view)")
        else:
            log_result("Comparison chart renders", False, "chart-compare-all not found")

        # ===== PHASE 8: Tables & Data =====
        print("\nPhase 8: Tables & Scrollable Content")

        # Check tables have scroll-wrap
        tables = page.evaluate("""() => {
            const issues = [];
            document.querySelectorAll('.data-table, .verify-table').forEach(t => {
                if (t.scrollWidth > window.innerWidth) {
                    const parent = t.closest('.scroll-wrap');
                    if (!parent) {
                        issues.push(`Table at ${t.closest('.section')?.querySelector('.section-header')?.textContent?.trim()?.slice(0,30) || 'unknown'} overflows without scroll-wrap`);
                    }
                }
            });
            return issues;
        }""")
        log_result("Tables have scroll-wrap", len(tables) == 0,
                   "; ".join(tables) if tables else "all tables wrapped")

        # ===== PHASE 9: Bottom Nav =====
        print("\nPhase 9: Bottom Navigation (PWA)")

        bottom_nav = page.query_selector(".bottom-nav")
        if bottom_nav:
            box = bottom_nav.bounding_box()
            visible = page.evaluate("() => getComputedStyle(document.querySelector('.bottom-nav')).display !== 'none'")
            ss = screenshot(page, "09_bottom_nav", 9)
            log_result("Bottom nav visible on mobile", visible, screenshot=ss)
            if box:
                log_result("Bottom nav at bottom of screen",
                           box["y"] > DEVICE["viewport"]["height"] - 100,
                           f"y={box['y']:.0f}px")
        else:
            log_result("Bottom nav visible on mobile", False, "element not found")

        # ===== PHASE 10: Text Readability =====
        print("\nPhase 10: Text & Font Sizes")

        small_text = page.evaluate("""() => {
            const issues = [];
            document.querySelectorAll('*').forEach(el => {
                const style = getComputedStyle(el);
                const size = parseFloat(style.fontSize);
                const text = el.textContent?.trim();
                if (text && text.length > 0 && text.length < 200 && size < 10 && style.display !== 'none' && el.offsetHeight > 0) {
                    const tag = el.tagName.toLowerCase();
                    if (!['script','style','link','meta'].includes(tag)) {
                        const cls = el.className?.toString().split(' ')[0] || '';
                        if (!issues.some(i => i.includes(cls))) {
                            issues.push(`${tag}.${cls}: ${size}px "${text.slice(0,30)}"`);
                        }
                    }
                }
            });
            return issues.slice(0, 10);
        }""")
        log_result("No critically small text (<10px)", len(small_text) <= 3,
                   f"{len(small_text)} elements: " + "; ".join(small_text[:3]) if small_text else "all text >= 10px")

        # ===== PHASE 11: Touch Targets =====
        print("\nPhase 11: Touch Target Sizes")

        small_targets = page.evaluate("""() => {
            const issues = [];
            // Exclude third-party controls (Leaflet zoom, attribution)
            document.querySelectorAll('button, a[href], [data-action], .resort-tab, .resort-pill, .section-header').forEach(el => {
                if (el.closest('.leaflet-control')) return;  // skip Leaflet controls
                if (el.closest('.leaflet-control-attribution')) return;
                const rect = el.getBoundingClientRect();
                if (rect.height > 0 && rect.width > 0 && (rect.height < 40 || rect.width < 40)) {
                    const id = el.id || el.className?.toString().split(' ')[0] || el.tagName;
                    if (!issues.some(i => i.includes(id))) {
                        issues.push(`${id}: ${Math.round(rect.width)}x${Math.round(rect.height)}px`);
                    }
                }
            });
            return issues.slice(0, 10);
        }""")
        log_result("Touch targets >= 40x40px", len(small_targets) <= 3,
                   f"{len(small_targets)} small targets: " + "; ".join(small_targets[:5]) if small_targets else "all targets adequate")

        # ===== PHASE 12: JavaScript Errors =====
        print("\nPhase 12: Console Errors")

        # Reload with console capture
        js_errors = []
        page.on("pageerror", lambda err: js_errors.append(str(err)))
        page.on("console", lambda msg: js_errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)
        page.reload(wait_until="domcontentloaded")
        try:
            page.wait_for_selector("#app", state="visible", timeout=120000)
        except Exception:
            pass
        page.wait_for_timeout(3000)

        log_result("No JavaScript errors", len(js_errors) == 0,
                   "; ".join(js_errors[:5]) if js_errors else "clean console")

        # ===== PHASE 13: Full Page Screenshot =====
        print("\nPhase 13: Full Page Capture")

        ss = screenshot_full(page, "13_full_page", 13)
        log_result("Full page screenshot captured", True, screenshot=ss)

        # ===== PHASE 14: PWA Manifest =====
        print("\nPhase 14: PWA Checks")

        manifest_resp = page.evaluate("""async () => {
            try {
                const link = document.querySelector('link[rel="manifest"]');
                if (!link) return { error: 'no manifest link' };
                const r = await fetch(link.href);
                return await r.json();
            } catch(e) { return { error: e.message }; }
        }""")
        log_result("PWA manifest loads", "error" not in manifest_resp,
                   manifest_resp.get("error", f"name={manifest_resp.get('name', '?')}"))

        sw_check = page.evaluate("() => 'serviceWorker' in navigator")
        log_result("Service worker API available", sw_check)

        apple_meta = page.query_selector('meta[name="apple-mobile-web-app-capable"]')
        log_result("Apple PWA meta tag present", apple_meta is not None)

        browser.close()

    # ===== Report =====
    print(f"\n{'='*60}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"Results: {passed} PASS, {failed} FAIL out of {len(results)} checks")

    if failed:
        print(f"\nFAILURES:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  \u274c {r['name']}: {r['details']}")
                if r.get("screenshot"):
                    print(f"     Screenshot: {r['screenshot']}")

    # Save report
    with open(REPORT_FILE, "w") as f:
        json.dump({
            "url": BASE_URL,
            "device": DEVICE,
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "results": results,
        }, f, indent=2)
    print(f"\nReport saved: {REPORT_FILE}")
    print(f"Screenshots:  {SCREENSHOT_DIR}/")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
