from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("request", lambda r: print(f"REQ: {r.url}"))
        page.on("websocket", lambda ws: print(f"WS: {ws.url}"))
        page.goto("https://www.flashscore.com/match/jNRjAUSo/")
        time.sleep(10)
        browser.close()

run()
