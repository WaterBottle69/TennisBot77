import asyncio
import datetime
import json
from config import Config
from live_score_scraper import LiveScoreScraper

async def run():
    c = Config()
    s = LiveScoreScraper(c)
    today = datetime.datetime.now().strftime("%Y%m%d")
    url = s.daily_url_template.format(date=today)
    matches = await s._fetch_url(url)
    
    print(f"Total matches found for {today}: {len(matches)}")
    # Print first 5 matches to see what kind of data we get
    for m in matches[:5]:
        print(f" - {m['player_a']} vs {m['player_b']} ({m['status']})")
    
    # Keyword search
    search_term = "Shimabukuro"
    found = [m for m in matches if search_term.lower() in m['player_a'].lower() or search_term.lower() in m['player_b'].lower()]
    if found:
        print(f"Found keyword {search_term} in {len(found)} matches!")
    else:
        print(f"Keyword {search_term} NOT FOUND in feed.")

if __name__ == "__main__":
    asyncio.run(run())
