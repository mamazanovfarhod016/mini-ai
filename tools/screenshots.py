"""Скриншоты интерфейса: headless Chrome 390×844 @2x (как телефон).
Запуск: /tmp/jv/bin/python tools/screenshots.py"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = (ROOT / "app/www/index.html").as_uri()
OUT = ROOT / "docs/screenshots"
OUT.mkdir(parents=True, exist_ok=True)
DIALOG = ["Привет!", "как дила?", "Кто тебя создал?", "Salom, qalaysan?", "расскажи шутку", "Rahmat!"]
CHROME = "/opt/hermes-data/bin/google-chrome"


def shot(page, name):
    page.wait_for_timeout(450)
    page.screenshot(path=str(OUT / name))
    print("saved", OUT / name)


def run(scheme, suffix):
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--headless=new"])
        ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                            color_scheme=scheme, is_mobile=True, has_touch=True, locale="ru-RU")
        page = ctx.new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(URL)
        page.evaluate("localStorage.clear()")
        page.reload()
        page.evaluate("window.__FAST__ = true")
        if suffix == "":
            shot(page, "1_empty_chat.png")
        for q in DIALOG:
            page.evaluate("q => MiniApp.ask(q)", q)
            page.wait_for_timeout(80)
        shot(page, f"2_dialog{suffix}.png")
        page.evaluate("MiniApp.openSheet()")
        shot(page, f"3_how_it_works{suffix}.png")
        if suffix == "":
            page.evaluate("MiniApp.closeSheet(); MiniApp.newChat();")
            page.evaluate("window.__FAST__ = true")
            page.evaluate("q => MiniApp.ask(q)", "превет как дела")
            page.wait_for_timeout(80)
            page.evaluate("MiniApp.openDrawer()")
            shot(page, "4_drawer.png")
        else:
            page.evaluate("MiniApp.closeSheet(); MiniApp.newChat();")
            shot(page, "5_empty_dark.png")
        if errs:
            print("JS ERRORS:", errs)
            sys.exit(1)
        b.close()


run("light", "")
run("dark", "_dark")
