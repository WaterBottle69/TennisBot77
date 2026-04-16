import asyncio
from config import Config
from live_score_scraper import LiveScoreScraper

async def test_upcoming():
    c = Config()
    s = LiveScoreScraper(c)
    
    player_a = "Sho Shimabukuro"
    player_b = "Kirill Simakin"
    
    print(f"Checking schedule for {player_a} vs {player_b}...")
    daily_matches = await s.fetch_all_today()
    match = s.find_match(player_a, player_b, daily_matches)
    
    if match:
        print(f"SUCCESS: Found match in daily schedule!")
        print(f"Match details: {match['player_a']} vs {match['player_b']} | Status: {match['status']}")
    else:
        print("FAILED: Match not found in daily schedule.")

if __name__ == "__main__":
    asyncio.run(test_upcoming())
