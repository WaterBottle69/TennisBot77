
import asyncio
import logging
from kalshi_client import KalshiClient
from config import Config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def test_order_payload():
    cfg = Config()
    # Ensure keys are present for signing, even if we don't actually send to a real ticker
    client = KalshiClient(cfg)
    
    ticker = "TEST-TICKER"
    price = 50
    count = 1
    side = "yes"
    
    log.info("Testing BUY payload formation...")
    try:
        # We don't want to actually place an order on a real account if not intended,
        # but the user's error was a validation error from Kalshi's server.
        # Here we just verify the client side logic doesn't crash and prints the payload.
        
        # Override _request to just print the json
        async def mock_request(method, path, **kwargs):
            log.info(f"MOCK REQUEST: {method} {path}")
            log.info(f"PAYLOAD: {kwargs.get('json')}")
            return {"status": "mock_success"}
            
        client._request = mock_request
        
        await client.buy(ticker, price, count, side)
        await client.sell(ticker, price, count, side)
        
        log.info("Payload test complete. verify 'side' is 'buy'/'sell' and 'yes_no' is 'yes'/'no'.")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_order_payload())
