
import asyncio
import json
import os
from config import Config, normalize_kalshi_pem
from kalshi_client import KalshiClient

async def test_auth():
    print("--- Kalshi Diagnostic Script ---")
    config = Config()
    
    # Force use of Production for testing if keys are provided
    if config.KALSHI_API_KEY_ID and config.KALSHI_PRIVATE_KEY_PEM:
        print(f"Keys found: {config.KALSHI_API_KEY_ID}")
        print(f"PEM Length: {len(config.KALSHI_PRIVATE_KEY_PEM)}")
        print(f"PEM Start: {config.KALSHI_PRIVATE_KEY_PEM[:60]}...")
        
        # Test Production first
        print("\nTesting PRODUCTION Exchange...")
        config.KALSHI_USE_PROD = True
        config.KALSHI_API_URL = config.KALSHI_PROD_BASE
        
        kalshi_prod = KalshiClient(config)
        try:
            # Try to fetch account info (requires auth)
            print("Authenticating with Production API...")
            # We don't have a direct 'get_account' method exposed in KalshiClient, 
            # but we can try 'get_atp_markets' which uses public events but 
            # initializes the session if possible.
            markets = await kalshi_prod.get_atp_markets()
            print(f"Success! Found {len(markets)} tennis markets on PRODUCTION.")
            for m in markets[:3]:
                print(f" - {m['title']} ({m['ticker']})")
        except Exception as e:
            print(f"Production Error: {e}")
        finally:
            await kalshi_prod.close()
            
        print("\nTesting DEMO Exchange...")
        config.KALSHI_USE_PROD = False
        config.KALSHI_API_URL = config.KALSHI_DEMO_BASE
        kalshi_demo = KalshiClient(config)
        try:
            markets = await kalshi_demo.get_atp_markets()
            print(f"Found {len(markets)} tennis markets on DEMO.")
        except Exception as e:
            print(f"Demo Error: {e}")
        finally:
            await kalshi_demo.close()
    else:
        print("No keys found in kalshi_keys.json or environment.")

if __name__ == "__main__":
    asyncio.run(test_auth())
