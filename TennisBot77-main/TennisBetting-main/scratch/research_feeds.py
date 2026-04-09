
import asyncio
import aiohttp
import ssl
import json
import logging
from config import Config
from live_score_scraper import LiveScoreScraper
from kalshi_client import KalshiClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def test_live_feeds():
    config = Config()
    config.KALSHI_USE_PROD = False # Use demo for testing
    
    # 1. Test LiveScore API
    log.info("--- Testing LiveScore API ---")
    scraper = LiveScoreScraper(config)
    live_matches = await scraper.fetch_live_scores()
    log.info(f"Found {len(live_matches)} live matches on LiveScore.")
    for m in live_matches[:5]:
        log.info(f"  {m['player_a']} vs {m['player_b']} ({m['status']}) - {m['points']}")

    # 2. Test Kalshi API
    log.info("--- Testing Kalshi ATP Markets ---")
    kalshi = KalshiClient(config)
    try:
        atp_matches = await kalshi.get_atp_markets()
        log.info(f"Found {len(atp_matches)} ATP markets on Kalshi.")
        for m in atp_matches[:5]:
            log.info(f"  Ticker: {m['ticker']} | {m['player_a']} vs {m['player_b']}")
            
            # Fuzzy match check
            match = scraper.find_match(m['player_a'], m['player_b'], live_matches)
            if match:
                log.info(f"    MATCHED with LiveScore! Status: {match['status']}")
            else:
                log.info(f"    NO MATCH on LiveScore.")
    except Exception as e:
        log.error(f"Kalshi test failed: {e}")
    finally:
        await kalshi.close()

if __name__ == "__main__":
    asyncio.run(test_live_feeds())
