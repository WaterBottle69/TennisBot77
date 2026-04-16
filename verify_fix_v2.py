import asyncio
import datetime
from config import Config
from live_score_scraper import LiveScoreScraper

async def test_date(date_str):
    c = Config()
    s = LiveScoreScraper(c)
    
    player_a = "Shimabukuro"
    player_b = "Simakin"
    
    print(f"Checking schedule for {date_str}...")
    url = s.daily_url_template.format(date=date_str)
    matches = await s._fetch_url(url)
    match = s.find_match(player_a, player_b, matches)
    
    if match:
        print(f"SUCCESS: Found match in {date_str} schedule!")
        return True
    return False

async def run():
    today = datetime.datetime.now().strftime("%Y%m%d")
    tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y%m%d")
    
    if not await test_date(today):
        await test_date(tomorrow)

if __name__ == "__main__":
    asyncio.run(run())
