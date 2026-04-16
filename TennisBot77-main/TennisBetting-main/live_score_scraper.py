import aiohttp
import logging
import asyncio
import ssl
from typing import Dict, List, Optional, Tuple
from config import Config
from difflib import get_close_matches

log = logging.getLogger(__name__)

# Bypassing SSL for local dev / certifi issues on macOS
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

class LiveScoreScraper:
    """
    Scraper for LiveScore.com's public CDN API.
    Provides real-time tennis scores down to the point level.
    """

    def __init__(self, config: Config):
        self.url = config.LIVESCORE_API_URL
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Origin": "https://www.livescore.com",
            "Referer": "https://www.livescore.com/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def fetch_live_scores(self) -> List[Dict]:
        """Fetch all currently live tennis matches from LiveScore."""
        try:
            # We use a lower timeout and more retries if needed
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            async with aiohttp.ClientSession(headers=self.headers, connector=connector) as session:
                async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        log.error(f"LiveScore API returned status {resp.status} for {self.url}")
                        return []
                    data = await resp.json()
                    return self._parse_api_response(data)
        except Exception as e:
            log.error(f"Failed to fetch live scores from LiveScore: {e}")
            return []

    def _parse_api_response(self, data: Dict) -> List[Dict]:
        """Extract relevant match details from the JSON structure."""
        active_matches = []
        stages = data.get("Stages", [])
        for stage in stages:
            tournament = stage.get("Nm", "Unknown Tournament")
            events = stage.get("Events", [])
            for event in events:
                # Players
                p1_data = event.get("T1", [{}])[0]
                p2_data = event.get("T2", [{}])[0]
                p1_name = p1_data.get("Nm", "Unknown")
                p2_name = p2_data.get("Nm", "Unknown")

                # Match status (Eps: "NS" = Not Started, "S1", "S2", "S3" = Live, "FT" = Finished)
                status = event.get("Eps", "").strip()
                # NS = Not Started, FT = Finished, CANC = Cancelled
                if status in ["NS", "FT", "CANC", "AP", "POST"]:
                    continue

                # Current serving player (Esrv is common in MEV updates)
                server = str(event.get("Esrv", event.get("Srv", "0"))) # "1" or "2"

                # Sets won
                sets_a = int(event.get("Tr1", 0) or 0)
                sets_b = int(event.get("Tr2", 0) or 0)

                # Games in current set
                cur_set_idx = (sets_a + sets_b) + 1
                games_a = int(event.get(f"Tr1S{cur_set_idx}", 0) or 0)
                games_b = int(event.get(f"Tr2S{cur_set_idx}", 0) or 0)

                # Current game points (15, 30, 40, Ad)
                points_a_raw = str(event.get("Tr1G", "0") or "0")
                points_b_raw = str(event.get("Tr2G", "0") or "0")

                def map_point(p):
                    p = p.strip().lower()
                    if p in ["0", ""]: return 0
                    if p == "15": return 1
                    if p == "30": return 2
                    if p == "40": return 3
                    if p in ["a", "ad", "40a", "adv"]: return 4
                    return 0

                active_matches.append({
                    "tournament": tournament,
                    "player_a": p1_name,
                    "player_b": p2_name,
                    "sets": (sets_a, sets_b),
                    "games": (games_a, games_b),
                    "points": (map_point(points_a_raw), map_point(points_b_raw)),
                    "p1_serving": server == "1",
                    "raw_points": (points_a_raw, points_b_raw),
                    "status": status
                })
        
        return active_matches

    def find_match(self, player_a: str, player_b: str, live_data: List[Dict]) -> Optional[Dict]:
        """
        Fuzzy match a Kalshi pair with the live scores.
        """
        # Create slugs for comparison
        def slugify(name):
            return "".join(e for e in name.lower() if e.isalnum())

        k_a = slugify(player_a)
        k_b = slugify(player_b)

        for match in live_data:
            m_a = slugify(match["player_a"])
            m_b = slugify(match["player_b"])

            # Check for direct or reverse match (sometimes Kalshi swaps names)
            if (k_a in m_a or m_a in k_a) and (k_b in m_b or m_b in k_b):
                return match
            if (k_a in m_b or m_b in k_a) and (k_b in m_a or m_a in k_b):
                # Reverse match
                match_rev = match.copy()
                match_rev["player_a"], match_rev["player_b"] = match["player_b"], match["player_a"]
                match_rev["sets"] = (match["sets"][1], match["sets"][0])
                match_rev["games"] = (match["games"][1], match["games"][0])
                match_rev["points"] = (match["points"][1], match["points"][0])
                match_rev["p1_serving"] = not match["p1_serving"]
                return match_rev
        
        return None

async def poll_live_score_real(player_a: str, player_b: str, config: Config, interval: float = 10.0):
    """
    Async generator that polls the real LiveScore API for a specific match.
    Now persists indefinitely even if the match is missing, allowing the bot 
    to trade on market drift / pre-game edge when the feed is down.
    """
    scraper = LiveScoreScraper(config)
    consecutive_misses = 0

    while True:
        log.debug(f"Polling LiveScore for {player_a} vs {player_b}...")
        try:
            matches = await scraper.fetch_live_scores()
            match = scraper.find_match(player_a, player_b, matches)

            if match:
                consecutive_misses = 0
                yield {
                    "points": match["points"],
                    "games": match["games"],
                    "sets": match["sets"],
                    "p1_serving": match["p1_serving"],
                    "is_live": True
                }
            else:
                consecutive_misses += 1
                if consecutive_misses % 6 == 0: # Log every minute at 10s interval
                    log.warning(
                        f"Match {player_a} vs {player_b} not found in LiveScore feed "
                        f"(misses: {consecutive_misses}). Bot is active in STALE MODE."
                    )
                yield {"is_live": False, "misses": consecutive_misses}
        except Exception as e:
            log.error(f"Error in poll_live_score_real: {e}")
            yield {"is_live": False, "error": str(e)}

        await asyncio.sleep(interval)
