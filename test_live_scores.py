import asyncio
from config import Config
from live_score_scraper import LiveScoreScraper

async def run():
    c = Config()
    s = LiveScoreScraper(c)
    matches = await s.fetch_live_scores()
    print(f"Found {len(matches)} live matches on LiveScore.com")
    for m in matches:
        print(f"{m['player_a']} vs {m['player_b']} | Sets: {m['sets']} Games: {m['games']} Pts: {m['raw_points']} | P1 Serving: {m['p1_serving']}")

if __name__ == "__main__":
    asyncio.run(run())
