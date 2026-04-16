import asyncio
import logging
from config import Config
from live_score_scraper import LiveScoreScraper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def test_live_scraper():
    config = Config()
    scraper = LiveScoreScraper(config)
    
    log.info("Fetching live scores from LiveScore.com CDN...")
    matches = await scraper.fetch_live_scores()
    
    if not matches:
        log.warning("No live matches found. Check if any matches are currently in progress.")
        return

    log.info(f"Found {len(matches)} live matches.")
    for m in matches:
        p1 = m["player_a"]
        p2 = m["player_b"]
        sets = m["sets"]
        games = m["games"]
        pts = m["points"]
        srv = "P1" if m["p1_serving"] else "P2"
        
        # Convert points back to display format (15, 30, 40, Ad)
        def display_pt(p):
            if p == 1: return "15"
            if p == 2: return "30"
            if p == 3: return "40"
            if p == 4: return "Ad"
            return "0"

        log.info(f"[{m['tournament']}] {p1} {sets[0]}({games[0]}-{display_pt(pts[0])}) vs {p2} {sets[1]}({games[1]}-{display_pt(pts[1])}) | Srv: {srv}")

    # Test fuzzy matching with a known match if available
    if len(matches) > 0:
        target = matches[0]
        p1_name = target["player_a"]
        p2_name = target["player_b"]
        
        # Simulate Kalshi-style name (sometimes truncated)
        search_a = p1_name.split()[-1]
        search_b = p2_name.split()[-1]
        
        found = scraper.find_match(search_a, search_b, matches)
        if found:
            log.info(f"SUCCESS: Fuzzy matched '{search_a}' vs '{search_b}' to '{p1_name}' vs '{p2_name}'")
        else:
            log.error(f"FAILURE: Could not fuzzy match '{search_a}' vs '{search_b}'")

if __name__ == "__main__":
    asyncio.run(test_live_scraper())
