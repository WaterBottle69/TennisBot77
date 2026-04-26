import asyncio
import logging
import time
from unittest.mock import MagicMock, AsyncMock
from market_monitor import MarketMonitor, MarketSignal, FlowDirection

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("verify_ws")

async def test_ws_update():
    log.info("--- Testing MarketMonitor WebSocket Logic ---")
    mock_kalshi = MagicMock()
    mock_kalshi.stream_orderbook = AsyncMock()

    monitor = MarketMonitor(kalshi_client=mock_kalshi, ticker="TEST_TICKER")
    monitor.set_model_prob(0.60) # We think YES is 60%

    # Simulate Kalshi V2 orderbook snapshot message
    # Bids are what we can sell for, Asks are what we can buy for.
    # E.g. someone is bidding 50c for YES. Someone is asking 52c for YES.
    # yes_bid = 50, yes_ask = 52.
    # yes_price should be (50+52)/2 = 51c = 0.51
    # spread = 2c = 0.02
    msg_1 = {
        "type": "orderbook_snapshot",
        "msg": {
            "bids": [[50, 100], [49, 200]],
            "asks": [[52, 50], [53, 300]]
        }
    }

    log.info("Sending first update...")
    await monitor._on_ws_update(msg_1)
    
    sig = monitor.current_signal
    assert abs(sig.yes_price - 0.51) < 0.001, f"Expected 0.51, got {sig.yes_price}"
    assert abs(sig.spread - 0.02) < 0.001, f"Expected 0.02, got {sig.spread}"
    
    # Send another message to test velocity/volatility calculations
    time.sleep(1) # simulate time passing
    msg_2 = {
        "type": "orderbook_delta",
        "msg": {
            "bids": [[51, 100], [50, 200]],
            "asks": [[53, 50], [54, 300]]
        }
    }
    log.info("Sending second update...")
    await monitor._on_ws_update(msg_2)
    sig = monitor.current_signal
    
    # New mid price is (51+53)/2 = 52c = 0.52
    assert abs(sig.yes_price - 0.52) < 0.001, f"Expected 0.52, got {sig.yes_price}"
    assert sig.price_velocity > 0, "Price went up, velocity should be positive"
    
    log.info("Testing Z-score mean reversion signal method...")
    mean_rev = monitor.mean_reversion_signal()
    log.info(f"Mean Reversion Data: {mean_rev}")
    
    # Test fallback logic if asks are empty
    msg_3 = {
        "type": "orderbook_delta",
        "msg": {
            "bids": [[55, 100]],
            "asks": []
        }
    }
    await monitor._on_ws_update(msg_3)
    sig = monitor.current_signal
    assert abs(sig.yes_price - 0.55) < 0.001, f"Expected 0.55 fallback, got {sig.yes_price}"
    
    log.info("✅ All WS logic verifications passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_ws_update())
