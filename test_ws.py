from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        def log_ws(ws):
            print(f"WebSocket Opened: {ws.url}")

        page.on("websocket", log_ws)
        page.goto("https://www.flashscore.com/match/jNRjAUSo/")
        time.sleep(5)
        browser.close()

run()
