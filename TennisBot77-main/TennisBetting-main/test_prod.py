import asyncio
import aiohttp
import ssl
import certifi

async def test_prod():
    url = "https://trading-api.kalshi.com/trade-api/v2/events?status=open&limit=200"
    print(f"Checking Production API: {url}")
    
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession() as session:
        async with session.get(url, ssl=ssl_ctx) as resp:
            if resp.status == 200:
                data = await resp.json()
                events = data.get("events") or []
                print(f"SUCCESS: Found {len(events)} open events in Production.")
                
                # Check for Tennis/ATP/WTA
                for e in events:
                    title = e.get("title", "")
                    st = e.get("series_ticker", "")
                    blob = f"{title} {st}".upper()
                    if any(x in blob for x in ["TENNIS", "ATP", "WTA"]):
                        print(f"FOUND TENNIS: '{title}' | Series='{st}'")
            else:
                print(f"FAILED: Status {resp.status}")
                print(await resp.text())

if __name__ == "__main__":
    asyncio.run(test_prod())
