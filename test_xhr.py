from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        def log_req(req):
            if "flashscore" in req.url and ("/feed/" in req.url or "/sync/" in req.url or"/x/" in req.url or "api" in req.url):
                print(f"XHR/Fetch: {req.url}")

        page.on("request", log_req)
        page.goto("https://www.flashscore.com/match/jNRjAUSo/")
        time.sleep(10)
        browser.close()

run()
