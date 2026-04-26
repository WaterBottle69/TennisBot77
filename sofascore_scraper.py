"""
sofascore_scraper.py — SofaScore live tennis feed.

Covers ALL ATP/WTA/Challenger/ITF singles events globally.
Used as tertiary live-score source in poll_live_score_real
(after LiveScore CDN and ESPN).
"""

import aiohttp
import asyncio
import ssl
import time
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
    "x-locale": "en_INT",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
}

_URL = "https://api.sofascore.com/api/v1/sport/tennis/events/live"
_CACHE_TTL = 20.0


def _map_point(s: str) -> int:
    s = str(s).strip().upper()
    if s in ("0", ""):   return 0
    if s == "15":        return 1
    if s == "30":        return 2
    if s in ("40", "3"): return 3
    if s in ("A", "AD", "ADV"): return 4
    try:
        return min(4, int(s))
    except ValueError:
        return 0


class SofaScoreScraper:
    """
    Fetches in-progress singles tennis matches from the SofaScore API.
    Returns dicts in the same format as LiveScoreScraper / ESPNTennisScorer.
    """

    def __init__(self):
        self._cache_ts: float = 0.0
        self._cache_data: List[Dict] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            self._session = aiohttp.ClientSession(headers=_HEADERS, connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_live(self) -> List[Dict]:
        """Return all in-progress singles ATP/WTA/Challenger matches."""
        now = time.time()
        if now - self._cache_ts < _CACHE_TTL:
            return self._cache_data

        async with self._lock:
            # Re-check after acquiring lock — another worker may have just fetched
            if time.time() - self._cache_ts < _CACHE_TTL:
                return self._cache_data
            try:
                session = self._get_session()
                async with session.get(_URL, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        log.warning("[SOFASCORE] HTTP %s", r.status)
                        return self._cache_data
                    data = await r.json()
            except Exception as exc:
                log.warning("[SOFASCORE] Fetch failed: %s", exc)
                self._session = None
                return self._cache_data

            matches = self._parse(data)
            self._cache_ts = time.time()
            self._cache_data = matches
            if matches:
                log.info("[SOFASCORE] %d live singles match(es)", len(matches))
            return matches

    def _parse(self, data: dict) -> List[Dict]:
        result = []
        for ev in data.get("events", []):
            if ev.get("status", {}).get("type") != "inprogress":
                continue

            # Exclude doubles: pairs use "/" in team names; tournament slug may say "doubles"
            home_name = ev.get("homeTeam", {}).get("name", "")
            away_name = ev.get("awayTeam", {}).get("name", "")
            if "/" in home_name or "/" in away_name:
                continue
            tournament_slug = ev.get("tournament", {}).get("slug", "").lower()
            if "double" in tournament_slug:
                continue

            p1_name = ev.get("homeTeam", {}).get("name", "")
            p2_name = ev.get("awayTeam", {}).get("name", "")
            if not p1_name or not p2_name:
                continue

            hs = ev.get("homeScore", {})
            aws = ev.get("awayScore", {})

            # Sets won (SofaScore uses 'current' = completed sets)
            p1_sets = int(hs.get("current", 0) or 0)
            p2_sets = int(aws.get("current", 0) or 0)

            # Current set index (1-based)
            cur_set = p1_sets + p2_sets + 1

            # Games in current set
            p1_games = int(hs.get(f"period{cur_set}", 0) or 0)
            p2_games = int(aws.get(f"period{cur_set}", 0) or 0)

            # Current game points
            p1_pts_raw = str(hs.get("point", "0") or "0")
            p2_pts_raw = str(aws.get("point", "0") or "0")

            # Serving player: count all completed games to derive current server
            first_serve = int(ev.get("firstToServe", 1) or 1)
            total_prev_games = sum(
                int(hs.get(f"period{i}", 0) or 0) + int(aws.get(f"period{i}", 0) or 0)
                for i in range(1, cur_set)
            )
            total_games_in_match = total_prev_games + p1_games + p2_games
            # Home serves games 1, 3, 5... (total_games % 2 == 0 means we're starting a new game)
            p1_serving = (first_serve == 1) == (total_games_in_match % 2 == 0)

            status_code = ev.get("status", {}).get("code", 0)
            # Code 100 = tiebreak
            status_str = "TB" if status_code == 100 else f"S{cur_set}"

            result.append({
                "tournament": ev.get("tournament", {}).get("name", "SofaScore"),
                "player_a":   p1_name,
                "player_b":   p2_name,
                "sets":       (p1_sets, p2_sets),
                "games":      (p1_games, p2_games),
                "points":     (_map_point(p1_pts_raw), _map_point(p2_pts_raw)),
                "p1_serving": p1_serving,
                "raw_points": (p1_pts_raw, p2_pts_raw),
                "status":     status_str,
                "_source":    "sofascore",
            })

        return result


# Module-level singleton
sofascore_scraper = SofaScoreScraper()
