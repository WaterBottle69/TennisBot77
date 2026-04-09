import asyncio
import logging
from kalshi_client import _is_tennis_event, _score_tennis_event
from tennis_scraper import TennisStatsScraper

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_filtering():
    print("--- Testing Kalshi Filtering ---")
    
    # Test 1: Ambiguous H2H that is actually Soccer (Jordan vs Argentina KXWCGAME)
    soccer_event = {
        "title": "Jordan vs Argentina",
        "series_ticker": "KXWCGAME",
        "event_ticker": "KXWCGAME-26JUN27JORARG",
        "category": "Sports",
        "sub_title": ""
    }
    is_tennis = _is_tennis_event(soccer_event)
    score = _score_tennis_event(soccer_event)
    print(f"Soccer (Jordan/Arg): is_tennis={is_tennis}, score={score} (Expected: False, -1 or low)")

    # Test 2: Explicit ATP Match (ATP Houston)
    atp_event = {
        "title": "Ben Shelton vs Frances Tiafoe",
        "series_ticker": "ATP-HOUSTON",
        "event_ticker": "ATP-HOU-2024MS",
        "category": "Sports",
        "sub_title": "ATP Houston Final"
    }
    is_tennis = _is_tennis_event(atp_event)
    score = _score_tennis_event(atp_event)
    print(f"ATP (Shelton/Tiafoe): is_tennis={is_tennis}, score={score} (Expected: True, high score)")

    # Test 3: WTA Match
    wta_event = {
        "title": "Iga Swiatek vs Aryna Sabalenka",
        "series_ticker": "WTA-FINALS",
        "event_ticker": "WTA-FIN-2024",
        "category": "Sports"
    }
    is_tennis = _is_tennis_event(wta_event)
    score = _score_tennis_event(wta_event)
    print(f"WTA (Swiatek/Sabalenka): is_tennis={is_tennis}, score={score} (Expected: True, high score)")

    # Test 4: Generic Sports Match (Soccer hint in title)
    generic_soccer = {
        "title": "Brazil vs Chile (World Cup Qualifier)",
        "series_ticker": "QUALIFIER",
        "category": "Sports"
    }
    is_tennis = _is_tennis_event(generic_soccer)
    score = _score_tennis_event(generic_soccer)
    print(f"Generic Soccer: is_tennis={is_tennis}, score={score} (Expected: False)")

    print("\n--- Testing Scraper Name Resolution ---")
    scraper = TennisStatsScraper()
    
    # Mocking rankings for test
    mock_rankings = {
        "jordan-thompson": {"slug": "jordan-thompson", "rank": 30},
        "argentina-something": {"slug": "argentina-something", "rank": 999},
        "valentin-vanta": {"slug": "valentin-vanta", "rank": 400}
    }
    
    # Try resolving "Argentina"
    slug = scraper._resolve_slug("Argentina", mock_rankings)
    print(f"Resolve 'Argentina': {slug} (Expected: None, blacklisted)")
    
    # Try resolving "Jordan"
    slug = scraper._resolve_slug("Jordan", mock_rankings)
    print(f"Resolve 'Jordan': {slug} (Expected: None, blacklisted)")

    # Try resolving "Thompson"
    slug = scraper._resolve_slug("Thompson", mock_rankings)
    print(f"Resolve 'Thompson' (partial match): {slug} (Expected: jordan-thompson)")

if __name__ == "__main__":
    asyncio.run(test_filtering())
