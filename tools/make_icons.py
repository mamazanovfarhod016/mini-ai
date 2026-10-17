"""Иконка приложения (своя 8-лучевая звёздочка) → PNG всех размеров через headless Chrome."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
STAR = ("M12 1.5c.6 0 1 .4 1.1 1l.9 6.1 4.9-3.8c.5-.4 1.1-.3 1.5.1.4.4.4 1 0 1.5l-3.8 4.9 6.1.9c.6.1 1 .5 1 1.1s-.4 1-1 1.1"
        "l-6.1.9 3.8 4.9c.4.5.3 1.1-.1 1.5-.4.4-1 .4-1.5 0L14 16.4l-.9 6.1c-.1.6-.5 1-1.1 1s-1-.4-1.1-1l-.9-6.1-4.9 3.8"
        "c-.5.4-1.1.3-1.5-.1-.4-.4-.4-1 0-1.5L7.4 14l-6.1-.9c-.6-.1-1-.5-1-1.1s.4-1 1-1.1l6.1-.9-3.8-4.9c-.4-.5-.3-1.1.1-1.5"
        ".4-.4 1-.4 1.5 0L10 7.4l.9-6c.1-.6.5-.9 1.1-.9Z")
BG, FG = "#D97757", "#FFF8F0"


def svg(kind):
    if kind == "full":   # обычная иконка: скруглённый квадрат + звезда
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108 108"><rect width="108" height="108" rx="24" fill="{BG}"/>'
                f'<g transform="translate(26 26) scale(2.333)"><path d="{STAR}" fill="{FG}"/></g></svg>')
    if kind == "fg":     # передний план adaptive icon (безопасная зона 66/108)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108 108">'
                f'<g transform="translate(33 33) scale(1.75)"><path d="{STAR}" fill="{FG}"/></g></svg>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108 108"><rect width="108" height="108" fill="{BG}"/></svg>'


SIZES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/hermes-data/bin/google-chrome", headless=True, args=["--headless=new"])
    page = b.new_page()
    def render(kind, px, out):
        page.set_viewport_size({"width": px, "height": px})
        page.set_content(f'<html><body style="margin:0;background:transparent">'
                         f'<div style="width:{px}px;height:{px}px">{svg(kind).replace("<svg ", f"<svg width={px} height={px} ")}</div></body></html>')
        out.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out), omit_background=True)
    for d, s in SIZES.items():
        render("full", s, ROOT / f"android/res/mipmap-{d}/ic_launcher.png")
        render("fg", int(s * 2.25), ROOT / f"android/res/mipmap-{d}/ic_launcher_fg.png")
        render("bg", int(s * 2.25), ROOT / f"android/res/mipmap-{d}/ic_launcher_bg.png")
    render("full", 512, ROOT / "docs/icon-512.png")
    b.close()
print("ok")
