"""Real Chromium interaction test; optional test dependency playwright."""

import threading
import pytest
from proofreading.demo import create
from proofreading.volume import Dataset
from proofreading.session import Session
from proofreading.server import make_server


def test_browser_click_merge_undo(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    expect = playwright.expect
    dataset = Dataset.load(create(tmp_path / "data"))
    session = Session(dataset, tmp_path / "episode")
    session.view.center_nm = [2920, 3920, 5480]
    session.view.pixel_size_nm = 9.0
    server = make_server(session, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with playwright.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            assert page.locator("#xy").evaluate("(el)=>el.naturalWidth") == 512
            # Actual displayed pixel click into atom 101, rather than an API shortcut.
            v = session.views["view-000001"][0]
            xyz = [2200, 3960, 5480]
            import numpy as np

            delta = np.array(xyz) - v.center_nm
            x = float(delta @ np.array(v.right) / v.pixel_size_nm + (v.width - 1) / 2)
            y = float(delta @ np.array(v.down) / v.pixel_size_nm + (v.height - 1) / 2)
            box = page.locator("#xy").bounding_box()
            page.mouse.click(
                box["x"] + (x + 0.5) * box["width"] / 512,
                box["y"] + (y + 0.5) * box["height"] / 512,
            )
            expect(page.locator("#segments")).to_contain_text("11")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            # Select the second segment through the universal tool UI.
            page.select_option("#tool", "select")
            page.fill("#args", '{"segments":["12"]}')
            page.click("#execute")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            # Automatic orthogonal redraws must not hide the user's last action.
            expect(page.locator("#undoAction")).to_have_attribute("title", "Undo select at action " + str(session.state()["undo_action_target"]["step"]))
            page.click("#undoAction")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            expect(page.locator("#segments")).to_have_text("11")
            page.fill("#args", '{"segments":["12"]}')
            page.click("#execute")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            page.click("#merge")
            expect(page.locator("#commit")).to_be_enabled()
            assert len(set(session.partition.values())) == 3
            page.click("#commit")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            assert session.partition[101] == session.partition[102]
            page.click("#undo")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            assert session.partition[101] != session.partition[102]
            page.click("#surface")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            old_surface = page.locator("#mesh").get_attribute("src")
            page.select_option("#tool", "select")
            page.fill("#args", '{"segments":["11"],"mode":"replace"}')
            page.click("#execute")
            expect(page.locator("#status")).to_have_text("Ready", timeout=30000)
            expect(page.locator("#segments")).to_have_text("11")
            assert page.locator("#mesh").get_attribute("src") != old_surface
            expect(page.locator("#previewLabel")).to_have_text("Current selection · 3D")
            expect(page.locator("#commit")).to_be_disabled()
            page.screenshot(path=str(tmp_path / "workbench.png"), full_page=True)
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
