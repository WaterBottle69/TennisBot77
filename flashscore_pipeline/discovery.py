"""
Flashscore Discovery Module
============================
Reverse-engineered from Flashscore's internal XHR feed.

The main feed URL is:
  https://www.flashscore.com/x/feed/f_2_0_1_en_1

Response format is a proprietary `¬`/`~` delimited protocol:
  AA÷{matchId}¬AD÷{timestamp}¬AE÷{player_a}¬AF÷{player_b}¬...

This module parses that feed to discover Match IDs for any player pair,
enabling automatic failover from LiveScore when Challenger matches are missing.
"""

import re
import ssl
import logging
import asyncio
import aiohttp
from typing import Optional

log = logging.getLogger(__name__)

# SSL bypass for macOS cert issues
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# Flashscore XHR feed — returns ALL tennis matches (scheduled + live + finished)
_FEED_URL = "https://www.flashscore.com/x/feed/f_2_0_1_en_1"

# Required headers from browser reverse engineering
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.flashscore.com/",
    "x-fsign": "SW9D1eY",  # Static client-side signature identified in browser session
}


class FlashscoreDiscovery:
    """
    Fetches the Flashscore live/scheduled tennis feed and finds Match IDs
    by fuzzy-matching player surnames against the Flashscore feed data.
    """

    def __init__(self, cache_ttl: int = 120):
        self._cache: dict = {}        # (surname_a, surname_b) → match_id
        self._feed_cache: list = []   # Parsed list of all matches
        self._feed_ttl = cache_ttl
        self._feed_fetched_at = 0.0

    async def _fetch_feed(self) -> list:
        """Fetch and parse the Flashscore tennis index feed."""
        import time
        now = time.time()
        if self._feed_cache and (now - self._feed_fetched_at) < self._feed_ttl:
            return self._feed_cache

        try:
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            async with aiohttp.ClientSession(headers=_HEADERS, connector=connector) as session:
                async with session.get(_FEED_URL, timeout=10) as resp:
                    if resp.status != 200:
                        log.error(f"Flashscore feed HTTP {resp.status}")
                        return []
                    raw = await resp.text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.error(f"Flashscore feed fetch error: {e}")
            return []

        matches = self._parse_feed(raw)
        log.info(f"Flashscore Discovery: Parsed {len(matches)} matches from feed.")
        self._feed_cache = matches
        self._feed_fetched_at = now
        return matches

    def _parse_feed(self, raw: str) -> list:
        """
        Parse the Flashscore `¬`/`~` delimited feed into structured dicts.
        Example record:
          AA÷Y951wEjB¬AD÷1713054600¬AE÷Shimabukuro S.¬AF÷Simakin I.¬...~
        """
        matches = []
        # Split on ~ to get individual match records
        records = raw.split("~")
        for record in records:
            if "AA÷" not in record or "AE÷" not in record:
                continue
            fields = {}
            # Parse key÷value pairs separated by ¬
            for part in record.split("¬"):
                if "÷" in part:
                    code, _, value = part.partition("÷")
                    fields[code.strip()] = value.strip()

            match_id = fields.get("AA", "")
            player_a = fields.get("AE", "")
            player_b = fields.get("AF", "")

            if match_id and player_a and player_b:
                matches.append({
                    "match_id": match_id,
                    "player_a": player_a,
                    "player_b": player_b,
                    "timestamp": fields.get("AD", ""),
                    "status": fields.get("AB", ""),
                })
        return matches

    async def find_match_id(self, player_a: str, player_b: str) -> Optional[str]:
        """
        Fuzzy search the Flashscore feed for a matching pair.
        Returns the Match ID string (e.g. 'Y951wEjB') or None.
        """
        # Cache check
        pair = tuple(sorted([player_a.lower(), player_b.lower()]))
        if pair in self._cache:
            cached = self._cache[pair]
            log.info(f"Discovery cache hit: {player_a} vs {player_b} → {cached}")
            return cached

        matches = await self._fetch_feed()

        def get_tokens(name):
            clean = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
            return set(t for t in clean.split() if len(t) > 1)

        k_a = get_tokens(player_a)
        k_b = get_tokens(player_b)

        for m in matches:
            m_a = get_tokens(m["player_a"])
            m_b = get_tokens(m["player_b"])

            # Direct match
            if k_a.intersection(m_a) and k_b.intersection(m_b):
                log.info(f"Discovery: {player_a} vs {player_b} → {m['match_id']} "
                         f"({m['player_a']} vs {m['player_b']})")
                self._cache[pair] = m["match_id"]
                return m["match_id"]

            # Reverse match (Kalshi sometimes swaps player order)
            if k_a.intersection(m_b) and k_b.intersection(m_a):
                log.info(f"Discovery (reversed): {player_a} vs {player_b} → {m['match_id']}")
                self._cache[pair] = m["match_id"]
                return m["match_id"]

        log.warning(f"Discovery: No Flashscore match found for {player_a} vs {player_b}")
        return None


# ── Quick test ───────────────────────────────────────────────────────────────
async def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = FlashscoreDiscovery()

    pairs = [
        ("Shimabukuro", "Simakin"),
        ("Park", "Zhukayev"),
        ("Zhou", "Kotov"),
    ]

    for a, b in pairs:
        mid = await d.find_match_id(a, b)
        print(f"{a} vs {b} → Match ID: {mid}")

if __name__ == "__main__":
    asyncio.run(main())
