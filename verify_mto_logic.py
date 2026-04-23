import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
import markov_engine
from bet_manager import BetManager
from live_score_scraper import LiveScoreScraper
from config import Config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("verify")

async def verify_mto_scraper():
    log.info("--- 1. Testing LiveScoreScraper MTO NLP Override ---")
    scraper = LiveScoreScraper(Config())
    # Simulate a JSON payload with a Medical Timeout event
    mock_data = {
        "Stages": [{
            "Nm": "Test Tournament",
            "Events": [{
                "T1": [{"Nm": "Player A"}],
                "T2": [{"Nm": "Player B"}],
                "Eps": "S1",
                "Tr1": "0", "Tr2": "0",
                "Tr1S1": "3", "Tr2S1": "3",
                "Tr1G": "40", "Tr2G": "40",
                "Stat": "Medical Timeout: Player A is receiving treatment",
                # The scraper converts the raw event to string, so if 'medical timeout' is in any field it will be caught.
            }]
        }]
    }
    
    matches = scraper._parse_api_response(mock_data)
    assert len(matches) == 1
    match = matches[0]
    
    if match.get("injury_flag") is True:
        log.info("✅ SUCCESS: LiveScoreScraper successfully set injury_flag=True when MTO was present.")
    else:
        log.error("❌ FAILED: LiveScoreScraper did not set injury_flag.")

async def verify_bet_manager_mto():
    log.info("--- 2. Testing BetManager Order Cancellation and Buy Blocking ---")
    mock_kalshi = MagicMock()
    mock_kalshi.cancel_order = AsyncMock()
    mock_kalshi.buy = AsyncMock()
    mock_kalshi.get_balance = AsyncMock(return_value=100.0)
    mock_kalshi.private_key = "test"
    
    bm = BetManager(kalshi=mock_kalshi, config=Config())
    
    # Setup some pending limits
    bm._pending_limits["TEST_yes"] = {
        "ticker": "TEST_MARKET",
        "order_id": "order_123",
        "price_cents": 50,
        "count": 10
    }
    bm._pending_limits["OTHER_yes"] = {
        "ticker": "OTHER_MARKET",
        "order_id": "order_456",
        "price_cents": 50,
        "count": 10
    }

    mock_market = {
        "id": "TEST_MARKET",
        "active": True,
        "yes_price": 0.50,
        "no_price": 0.50,
        "player_a": "Player A",
        "question": "Player A vs Player B"
    }
    mock_win_prob = {"team_a": 0.60, "team_b": 0.40}
    mock_event = {"injury_flag": True}
    
    await bm.evaluate_and_act(
        market=mock_market,
        win_prob=mock_win_prob,
        game_state=None,
        event=mock_event,
        is_live_match=True
    )
    
    # Assertions
    if "TEST_yes" not in bm._pending_limits:
        log.info("✅ SUCCESS: BetManager canceled pending limits for the affected ticker.")
    else:
        log.error("❌ FAILED: BetManager did not remove the pending limit from its state.")
        
    if "OTHER_yes" in bm._pending_limits:
        log.info("✅ SUCCESS: BetManager did not cancel limits for other unaffected tickers.")
    else:
        log.error("❌ FAILED: BetManager incorrectly canceled limits for other tickers.")
        
    mock_kalshi.cancel_order.assert_called_with("order_123")
    log.info("✅ SUCCESS: kalshi.cancel_order() was explicitly called with the correct order_id.")
    
    if not mock_kalshi.buy.called:
        log.info("✅ SUCCESS: BetManager blocked new buys despite a positive edge (model_prob 0.60 vs price 0.50).")
    else:
        log.error("❌ FAILED: BetManager placed a buy order even though injury_flag was True.")

def verify_markov_fatigue():
    log.info("--- 3. Testing Markov Engine Fatigue Spike ---")
    
    # leverage index >= 0.9 is reached at 3-3 (Deuce) or higher
    # Let's test a baseline serve prob of 0.65
    base_p = 0.65
    
    # Low leverage (0-0)
    p_low = markov_engine.conditional_serve_prob(base_p, leverage=0.2)
    
    # High leverage, but not extreme (1-3)
    p_med = markov_engine.conditional_serve_prob(base_p, leverage=0.85)
    
    # Extreme leverage (Deuce 3-3)
    p_extreme = markov_engine.conditional_serve_prob(base_p, leverage=0.9)
    
    log.info(f"Base Serve P: {base_p}")
    log.info(f"Low Leverage P: {p_low} (Expected {base_p})")
    log.info(f"Med Leverage P: {p_med} (Expected {base_p - 0.02})")
    log.info(f"Extreme Leverage P: {p_extreme} (Expected {base_p - 0.02 - 0.035})")
    
    if abs(p_low - base_p) < 0.001:
        log.info("✅ SUCCESS: Low leverage probability matches base.")
    else:
        log.error("❌ FAILED: Low leverage probability mismatch.")
        
    if abs(p_med - (base_p - markov_engine.BETA_PRESSURE)) < 0.001:
        log.info("✅ SUCCESS: Medium leverage correctly applies BETA_PRESSURE (-0.02).")
    else:
        log.error("❌ FAILED: Medium leverage probability mismatch.")
        
    if abs(p_extreme - (base_p - markov_engine.BETA_PRESSURE - 0.035)) < 0.001:
        log.info("✅ SUCCESS: Extreme leverage correctly applies Fatigue Spike (-0.035).")
    else:
        log.error("❌ FAILED: Extreme leverage probability mismatch.")

async def main():
    await verify_mto_scraper()
    await verify_bet_manager_mto()
    verify_markov_fatigue()
    log.info("--- Verification Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
