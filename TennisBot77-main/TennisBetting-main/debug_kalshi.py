import asyncio
import logging
from kalshi_client import KalshiClient
from config import Config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

async def debug_kalshi():
    config = Config()
    client = KalshiClient(config)
    
    print(f"Connecting to Kalshi API: {config.KALSHI_API_URL}")
    
    try:
        cursor = ""
        found_any_sports = 0
        all_tennis_candidates = []
        
        for page in range(10): # Scan 10 pages
            q = f"status=open&limit=200"
            if cursor:
                q += f"&cursor={cursor}"
            
            resp = await client._request("GET", f"/events?{q}")
            events = resp.get("events") or []
            cursor = resp.get("cursor") or ""
            
            for e in events:
                cat = e.get("category", "")
                if cat == "Sports":
                    found_any_sports += 1
                
                title = e.get("title", "")
                st = e.get("series_ticker", "")
                et = e.get("event_ticker", "")
                blob = f"{title} {st} {et}".upper()
                
                # Broad search
                if any(x in blob for x in ["TENNIS", "ATP", "WTA", "US OPEN", "FRENCH OPEN", "WIMBLEDON", "AUSTRALIAN OPEN"]):
                    all_tennis_candidates.append(e)
            
            if not cursor:
                break
        
        print(f"Scanned {page+1} pages. Found {found_any_sports} total Sports events.")
        print(f"Found {len(all_tennis_candidates)} potential Tennis events.")
        
        from kalshi_client import _is_tennis_event, _players_from_event
        
        for e in all_tennis_candidates:
            is_tennis = _is_tennis_event(e)
            players = _players_from_event(e)
            print(f"FILTER TEST: Title='{e.get('title')}' | Series='{e.get('series_ticker')}' -> is_tennis={is_tennis}, players={players}")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(debug_kalshi())
