"""
espn_scraper.py — ESPN ATP/WTA live score feed.

Covers all ATP 250/500/1000 and Slam events that LiveScore CDN misses
(LiveScore CDN primarily covers Challengers and minor events).

Used as a secondary live-score source in poll_live_score_real.
"""

import aiohttp
import asyncio
import ssl
import time
import logging
from typing import Dict, List

log = logging.getLogger(__name__)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}

_TOUR_URLS = [
    "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard",
    "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard",
]

_CACHE_TTL = 20.0


def _map_point(s: str) -> int:
    s = str(s).strip().upper()
    if s in ("0", ""):   return 0
    if s == "15":        return 1
    if s == "30":        return 2
    if s in ("40", "3"): return 3
    if s in ("A", "AD", "ADV", "4"): return 4
    try:
        return min(4, int(s))
    except ValueError:
        return 0


class ESPNTennisScorer:
    """
    Fetches in-progress ATP/WTA matches from the ESPN scoreboard API
    and returns them in the same dict format as LiveScoreScraper.
    """

    def __init__(self):
        self._cache: Dict[str, tuple] = {}   # url → (timestamp, parsed_list)
        self._session: aiohttp.ClientSession | None = None

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
        """Return all in-progress ATP+WTA matches (two tour URLs fetched in parallel)."""
        results = await asyncio.gather(
            self._fetch_url(_TOUR_URLS[0]),
            self._fetch_url(_TOUR_URLS[1]),
            return_exceptions=True,
        )
        merged = []
        for r in results:
            if isinstance(r, list):
                merged.extend(r)
        return merged

    async def _fetch_url(self, url: str) -> List[Dict]:
        now = time.time()
        cached_ts, cached_data = self._cache.get(url, (0.0, []))
        if now - cached_ts < _CACHE_TTL:
            return cached_data

        try:
            session = self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    log.warning("[ESPN] HTTP %s for %s", r.status, url)
                    return cached_data
                data = await r.json()
        except Exception as exc:
            log.warning("[ESPN] Fetch failed (%s): %s", url, exc)
            self._session = None
            return cached_data

        matches = self._parse(data)
        self._cache[url] = (now, matches)
        if matches:
            log.info("[ESPN] %d live match(es) found", len(matches))
        return matches

    def _parse(self, data: dict) -> List[Dict]:
        result = []
        for event in data.get("events", []):
            tournament_name = event.get("name", "ATP/WTA")
            # Some ESPN responses nest competitions under groupings; others put them directly
            # on the event. Collect from both so we never miss a match.
            competitions = []
            for grouping in event.get("groupings", []):
                competitions.extend(grouping.get("competitions", []))
            if not competitions:
                competitions = event.get("competitions", [])
            for comp in competitions:
                    state = (
                        comp.get("status", {})
                        .get("type", {})
                        .get("state", "")
                    )
                    if state != "in":
                        continue

                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue

                    # order=1 → home/p1, order=2 → away/p2
                    competitors = sorted(
                        competitors, key=lambda c: c.get("order", 99)
                    )
                    p1, p2 = competitors[0], competitors[1]

                    p1_name = p1.get("athlete", {}).get("displayName", "")
                    p2_name = p2.get("athlete", {}).get("displayName", "")
                    if not p1_name or not p2_name:
                        continue

                    # Per-set linescores ─────────────────────────────────────
                    p1_ls = p1.get("linescores", [])
                    p2_ls = p2.get("linescores", [])

                    p1_sets_won = sum(1 for ls in p1_ls if ls.get("winner"))
                    p2_sets_won = sum(1 for ls in p2_ls if ls.get("winner"))

                    # Current-set games (last linescore entry)
                    p1_games = int(p1_ls[-1].get("value", 0)) if p1_ls else 0
                    p2_games = int(p2_ls[-1].get("value", 0)) if p2_ls else 0

                    # Current-game point score from situation ────────────────
                    situation = comp.get("situation") or {}
                    home_score_raw = str(
                        situation.get("homeScore",
                        situation.get("playerOneGameScore", "0"))
                    )
                    away_score_raw = str(
                        situation.get("awayScore",
                        situation.get("playerTwoGameScore", "0"))
                    )

                    # Serving player
                    serve_raw = str(
                        situation.get("server",
                        situation.get("servingPlayer", ""))
                    ).strip()
                    # ESPN uses "1"/"2" or athlete id; default to p1
                    p1_serving = (serve_raw in ("1", p1.get("id", "__NO__")))

                    set_num = comp.get("status", {}).get("period", 1)

                    result.append({
                        "tournament":  tournament_name,
                        "player_a":    p1_name,
                        "player_b":    p2_name,
                        "sets":        (p1_sets_won, p2_sets_won),
                        "games":       (p1_games, p2_games),
                        "points":      (_map_point(home_score_raw),
                                        _map_point(away_score_raw)),
                        "p1_serving":  p1_serving,
                        "raw_points":  (home_score_raw, away_score_raw),
                        "status":      f"S{set_num}",
                        "_source":     "espn",
                    })

        return result


# Module-level singleton
espn_scorer = ESPNTennisScorer()
