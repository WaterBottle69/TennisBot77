import aiohttp
import logging
import asyncio
import ssl
import datetime
from typing import Dict, List, Optional
from config import Config
from flashscore_pipeline.discovery import FlashscoreDiscovery

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
        self.daily_url_template = config.LIVESCORE_DAILY_URL

    async def fetch_live_scores(self) -> List[Dict]:
        """Fetch all currently live tennis matches from LiveScore."""
        return await self._fetch_url(self.url)

    async def fetch_all_today(self) -> List[Dict]:
        """Fetch the entire daily schedule to find matches that haven't started yet.

        Passes include_scheduled=True so NS (Not Started) matches are returned —
        these are exactly the upcoming games the caller wants to detect.
        """
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        daily_url = self.daily_url_template.format(date=today_str)
        return await self._fetch_url(daily_url, include_scheduled=True)

    async def _fetch_url(self, url: str, include_scheduled: bool = False) -> List[Dict]:
        try:
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            async with aiohttp.ClientSession(headers=self.headers, connector=connector) as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        log.error(f"LiveScore API returned status {resp.status} for {url}")
                        return []
                    data = await resp.json()
                    return self._parse_api_response(data, include_scheduled=include_scheduled)
        except Exception as e:
            log.error(f"Failed to fetch from LiveScore ({url}): {e}")
            return []

    def _parse_api_response(self, data: Dict, include_scheduled: bool = False) -> List[Dict]:
        """Extract relevant match details from the JSON structure.

        Args:
            include_scheduled: When True, NS (Not Started) matches are kept so
                               the daily-schedule fallback can detect upcoming games.
        """
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
                # FT = Finished, CANC = Cancelled, AP/POST = postponed.
                # NS (Not Started) is only skipped for the live feed; the daily
                # schedule explicitly needs it to surface upcoming matches.
                skip = {"FT", "CANC", "AP", "POST"}
                if not include_scheduled:
                    skip.add("NS")
                if status in skip:
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

    def find_match(self, player_a: str, player_b: str, matches: List[Dict]) -> Optional[Dict]:
        """
        Token-based fuzzy matching. Handles name variations like:
        - "Sho Shimabukuro" vs "S. Shimabukuro"
        - "Ui-Sung Park" vs "Park, U"
        - "Carlos Alcaraz" vs "Alcaraz Garfia, C"
        """
        import re
        def get_tokens(name):
            # Remove punctuation and split into lowercase alphanumeric tokens
            clean = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
            return set(t for t in clean.split() if len(t) > 1 or t.isdigit())

        k_a_tokens = get_tokens(player_a)
        k_b_tokens = get_tokens(player_b)

        for m in matches:
            m_a_tokens = get_tokens(m["player_a"])
            m_b_tokens = get_tokens(m["player_b"])

            # Check for name intersection. If the last names match and initials match, it's likely the same.
            # We look for at least one overlapping token per player.
            a_match = bool(k_a_tokens.intersection(m_a_tokens))
            b_match = bool(k_b_tokens.intersection(m_b_tokens))
            
            # Check direct match
            if a_match and b_match:
                return m
            
            # Check reverse match (Kalshi/LiveScore swapping positions)
            a_rev_match = bool(k_a_tokens.intersection(m_b_tokens))
            b_rev_match = bool(k_b_tokens.intersection(m_a_tokens))
            
            if a_rev_match and b_rev_match:
                match_rev = m.copy()
                match_rev["player_a"], match_rev["player_b"] = m["player_b"], m["player_a"]
                match_rev["sets"] = (m["sets"][1], m["sets"][0])
                match_rev["games"] = (m["games"][1], m["games"][0])
                match_rev["points"] = (m["points"][1], m["points"][0])
                match_rev["p1_serving"] = not m["p1_serving"]
                return match_rev
        return None

async def poll_live_score_real(player_a: str, player_b: str, config: Config, interval: float = 10.0):
    """
    Indefinitely polls LiveScore (with two fallback layers) to find a match.

    Priority order:
      1. LiveScore live feed — primary, lowest latency.
      2. LiveScore daily schedule — catches NS (Not Started) upcoming matches.
      3. Flashscore discovery — second-source confirmation when LiveScore can't
         find the match at all (common for Challengers / minor ATP events).
    """
    scraper = LiveScoreScraper(config)
    flashscore = FlashscoreDiscovery()
    consecutive_misses = 0

    while True:
        try:
            # ── 1. Live feed ─────────────────────────────────────────────────
            live_matches = await scraper.fetch_live_scores()
            match = scraper.find_match(player_a, player_b, live_matches)

            if match:
                consecutive_misses = 0
                yield {
                    "points":     match["points"],
                    "games":      match["games"],
                    "sets":       match["sets"],
                    "p1_serving": match["p1_serving"],
                    "is_live":    True,
                    "status":     match.get("status", "LIVE"),
                }
                await asyncio.sleep(interval)
                continue

            # ── 2. Daily schedule (every 3 misses to save bandwidth) ─────────
            is_upcoming = False
            if consecutive_misses % 3 == 0:
                daily_matches = await scraper.fetch_all_today()
                upcoming_match = scraper.find_match(player_a, player_b, daily_matches)
                if upcoming_match:
                    is_upcoming = True
                    log.info(f"[SCRAPER] {player_a} vs {player_b} found in daily schedule (NS).")

            # ── 3. Flashscore fallback (every 6 misses) ──────────────────────
            if not is_upcoming and consecutive_misses % 6 == 0:
                fs_id = await flashscore.find_match_id(player_a, player_b)
                if fs_id:
                    is_upcoming = True
                    log.info(f"[FLASHSCORE] Fallback confirmed match exists: ID={fs_id}")

            consecutive_misses += 1
            yield {
                "is_live":      False,
                "is_scheduled": is_upcoming,
                "misses":       consecutive_misses,
            }

        except Exception as e:
            log.error(f"Error in poll_live_score_real: {e}")
            yield {"is_live": False, "error": str(e)}

        await asyncio.sleep(interval)
