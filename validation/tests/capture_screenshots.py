#!/usr/bin/env python3
"""Capture screenshots of the running app for the evidence folder (requires app on localhost:8599 and Chrome)."""
import re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "evidence" / "screenshots"; OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8599"

def pick(page, index, value):
    box = page.locator('[data-testid="stSelectbox"]').nth(index)
    box.click(); page.keyboard.type(value); page.keyboard.press("Enter"); page.wait_for_timeout(2500)

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1500, "height": 1300})
    pg.goto(BASE + "/overview"); pg.wait_for_selector("text=Total ED arrivals", timeout=60000); pg.wait_for_timeout(2500)
    pg.screenshot(path=str(OUT / "S01_overview_kpis_default_filter.png"))
    pg.goto(BASE + "/scenarios"); pg.wait_for_selector("text=Scenario Comparison", timeout=60000); pg.wait_for_timeout(3000)
    pg.screenshot(path=str(OUT / "S02_scenario_comparison.png"))
    pg.goto(BASE + "/predictive-models"); pg.wait_for_selector("text=Recommended actions", timeout=60000); pg.wait_for_timeout(4000)
    pg.screenshot(path=str(OUT / "S03_recommended_actions.png"))
    # invalid what-if: demand model, date 2024-01-13, 7-day forecast 1495.6 (ratio 1.50x)
    pick(pg, 2, "2024-01-13")
    ni = pg.locator('input[aria-label^="Seven-day arrival forecast"]').first
    ni.click(); pg.keyboard.press("Control+A"); pg.keyboard.type("1495.6"); pg.keyboard.press("Tab")
    pg.get_by_role("button", name=re.compile("Run model-based scenario")).click(); pg.wait_for_timeout(3000)
    pg.get_by_text("outside the plausible band").first.scroll_into_view_if_needed()
    pg.screenshot(path=str(OUT / "S04_invalid_scenario_rejected.png"))
    # valid scenario with generated summary: default date, 7-day forecast 1485.3 (IVKIT, 2025-12-24 is default)
    pg.reload(); pg.wait_for_selector("text=Recommended actions", timeout=60000); pg.wait_for_timeout(4000)
    ni = pg.locator('input[aria-label^="Seven-day arrival forecast"]').first
    ni.click(); pg.keyboard.press("Control+A"); pg.keyboard.type("1485.3"); pg.keyboard.press("Tab")
    pg.get_by_role("button", name=re.compile("Run model-based scenario")).click(); pg.wait_for_timeout(3500)
    pg.get_by_text("Automatically generated scenario summary").first.scroll_into_view_if_needed()
    pg.screenshot(path=str(OUT / "S05_valid_scenario_with_summary.png"))
    b.close()
print("done")
