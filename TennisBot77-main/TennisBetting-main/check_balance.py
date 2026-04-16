import asyncio
import logging
from kalshi_client import KalshiClient
from config import Config

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

async def check_balance():
    config = Config()
    client = KalshiClient(config)
    
    try:
        print(f"Checking balance on Kalshi API: {config.KALSHI_API_URL}")
        # Standard Kalshi v2 balance endpoint
        resp = await client._request("GET", "/portfolio/balance")
        print(f"Balance Info: {resp}")
    except Exception as e:
        print(f"Failed to check balance: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_balance())
