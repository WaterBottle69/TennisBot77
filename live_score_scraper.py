import aiohttp
import logging
import asyncio
import ssl
import time
import datetime
from typing import Dict, List, Optional, Tuple
from config import Config
from flashscore_pipeline.discovery import FlashscoreDiscovery

log = logging.getLogger(__name__)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_LIVE_CACHE_TTL  = 20.0   # seconds a cached feed response stays fresh
_STALE_THRESHOLD = 90.0   # seconds before trading is blocked as stale
# After this many consecutive misses with no live data ever seen, give up and
# release the worker slot so it isn't wasted on unsupported matches (e.g. Korean ITF).
_GIVE_UP_MISSES  = 25


class LiveScoreScraper:
    """
    Scraper for LiveScore.com's public CDN API.
    Provides real-time tennis scores down to the point level.

    Cache layer: live feed responses are cached for LIVE_CACHE_TTL seconds so
    that rapid polling in poll_live_score_real does not hammer the CDN on every
    tick.  The daily schedule feed is cached separately with a 3× longer TTL.
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

        # Per-URL response cache: url → (timestamp, parsed_list)
        self._cache: Dict[str, Tuple[float, List[Dict]]] = {}
        # Per-URL fetch lock — prevents cache stampede when many workers share this instance
        self._locks: Dict[str, asyncio.Lock] = {}
        # Timestamp of the last response that contained at least one live match
        self._last_live_hit: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    def data_age_seconds(self) -> float:
        """Seconds since we last got a fresh live match from LiveScore."""
        if self._last_live_hit == 0.0:
            return float("inf")
        return time.time() - self._last_live_hit

    async def fetch_live_scores(self) -> List[Dict]:
        """Fetch all currently live tennis matches from LiveScore (cached)."""
        return await self._fetch_url(self.url, cache_ttl=_LIVE_CACHE_TTL)

    async def fetch_all_today(self) -> List[Dict]:
        """Fetch the entire daily schedule to find matches that haven't started yet."""
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        daily_url = self.daily_url_template.format(date=today_str)
        return await self._fetch_url(daily_url, include_scheduled=True,
                                     cache_ttl=_LIVE_CACHE_TTL * 3)

    def invalidate_cache(self):
        """Force-expire the cache so the next fetch hits the network."""
        self._cache.clear()

    async def close(self):
        """Close the persistent HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            self._session = aiohttp.ClientSession(headers=self.headers, connector=connector)
        return self._session

    async def _fetch_url(self, url: str, include_scheduled: bool = False,
                         cache_ttl: float = _LIVE_CACHE_TTL) -> List[Dict]:
        cached_ts, cached_data = self._cache.get(url, (0.0, []))
        if time.time() - cached_ts < cache_ttl:
            return cached_data

        if url not in self._locks:
            self._locks[url] = asyncio.Lock()
        async with self._locks[url]:
            cached_ts, cached_data = self._cache.get(url, (0.0, []))
            if time.time() - cached_ts < cache_ttl:
                return cached_data
            try:
                session = self._get_session()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        log.warning("[LIVESCORE] HTTP %s for %s", resp.status, url)
                        return cached_data
                    data = await resp.json()
                    result = self._parse_api_response(data, include_scheduled=include_scheduled)
                    now = time.time()
                    self._cache[url] = (now, result)
                    if result and not include_scheduled:
                        self._last_live_hit = now
                    return result
            except Exception as e:
                log.warning("[LIVESCORE] Fetch failed (%s): %s", url, e)
                self._session = None
                return cached_data

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

                # MTO NLP Override: Scan raw event for medical timeout
                event_raw_str = str(event).lower()
                injury_flag = "medical timeout" in event_raw_str

                active_matches.append({
                    "tournament": tournament,
                    "player_a": p1_name,
                    "player_b": p2_name,
                    "sets": (sets_a, sets_b),
                    "games": (games_a, games_b),
                    "points": (map_point(points_a_raw), map_point(points_b_raw)),
                    "p1_serving": server == "1",
                    "raw_points": (points_a_raw, points_b_raw),
                    "status": status,
                    "injury_flag": injury_flag
                })
        
        return active_matches

    def find_match(self, player_a: str, player_b: str, matches: List[Dict]) -> Optional[Dict]:
        """
        Token-based fuzzy matching. Handles name variations and prevents sibling mismatches.
        """
        import re
        def get_tokens(name):
            clean = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
            return set(clean.split())

        k_a_tokens = get_tokens(player_a)
        k_b_tokens = get_tokens(player_b)

        def is_match(k_toks, m_toks):
            overlap = k_toks.intersection(m_toks)
            # Require at least one token >= 3 chars (usually the last name)
            # or if the name is very short (e.g. "Wu"), just require any overlap
            has_strong_overlap = any(len(t) >= 2 for t in overlap)
            if not has_strong_overlap:
                return False
                
            # If initials conflict strictly, reject to prevent sibling mismatches
            k_inits = {t[0] for t in k_toks if len(t) == 1}
            m_inits = {t[0] for t in m_toks if len(t) == 1}
            if k_inits and m_inits and not k_inits.intersection(m_inits):
                # E.g. k="M Ymer" (m), m="E Ymer" (e) -> fail
                return False
                
            return True

        for m in matches:
            m_a_tokens = get_tokens(m["player_a"])
            m_b_tokens = get_tokens(m["player_b"])

            if is_match(k_a_tokens, m_a_tokens) and is_match(k_b_tokens, m_b_tokens):
                return m
            
            if is_match(k_a_tokens, m_b_tokens) and is_match(k_b_tokens, m_a_tokens):
                match_rev = m.copy()
                match_rev["player_a"], match_rev["player_b"] = m["player_b"], m["player_a"]
                match_rev["sets"] = (m["sets"][1], m["sets"][0])
                match_rev["games"] = (m["games"][1], m["games"][0])
                match_rev["points"] = (m["points"][1], m["points"][0])
                match_rev["p1_serving"] = not m["p1_serving"]
                match_rev["injury_flag"] = m.get("injury_flag", False)
                return match_rev
        return None

# Module-level singleton — all workers share one session and one cache,
# so 20+ concurrent workers make at most one HTTP request per TTL window.
_shared_scraper: Optional[LiveScoreScraper] = None

def _get_shared_scraper(config: Config) -> LiveScoreScraper:
    global _shared_scraper
    if _shared_scraper is None:
        _shared_scraper = LiveScoreScraper(config)
    return _shared_scraper


async def poll_live_score_real(player_a: str, player_b: str, config: Config, interval: float = 10.0):
    """
    Indefinitely yields live-score updates for a match, using sources in
    priority order:

      1. SportRadar WebSocket push  — sub-second latency, official ATP/WTA data
                                      (requires SPORTRADAR_API_KEY in kalshi_keys.json)
      2. LiveScore CDN live feed    — primary polling fallback (Challengers/minor ATP)
      3. ESPN scoreboard            — ATP 250/500/1000/Slams
      4. SofaScore                  — broadest global coverage (ATP/WTA/Challenger/ITF)
      5. LiveScore daily schedule   — catches NS (Not Started) upcoming matches
      6. Flashscore discovery       — secondary confirmation that a match exists

    Stale = no source returned a live score. Bot will NOT trade when stale.
    """
    from espn_scraper import espn_scorer
    from sofascore_scraper import sofascore_scraper
    from sportradar_scraper import SportRadarLiveStream

    scraper    = _get_shared_scraper(config)
    flashscore = FlashscoreDiscovery()
    consecutive_misses = 0
    ever_seen_live = False
    _is_scheduled = False

    # ── 1. SportRadar WebSocket (priority 1) ─────────────────────────────────
    sr_key     = getattr(config, "SPORTRADAR_API_KEY", "")
    sr_enabled = getattr(config, "SPORTRADAR_ENABLED", False)

    if sr_enabled and sr_key:
        log.info("[SPORTRADAR-WS] Starting WebSocket feed for %s vs %s", player_a, player_b)
        sr_stream = SportRadarLiveStream(player_a, player_b, sr_key)
        try:
            async for update in sr_stream.listen():
                scraper._last_live_hit = time.time()
                yield update
                # After each WS event, briefly yield control so the event loop
                # can handle Kalshi / bet-manager tasks without blocking.
                await asyncio.sleep(0)
        except Exception as exc:
            log.warning("[SPORTRADAR-WS] Stream terminated unexpectedly: %s — "
                        "falling through to polling sources.", exc)
        finally:
            sr_stream.stop()
        # If we reach here the WS ended (match over or permanent error).
        # Fall through to polling so the bot doesn't go dark mid-session.
        log.info("[SPORTRADAR-WS] WebSocket ended — switching to polling fallback.")

    # ── 2–6. Polling fallback ────────────────────────────────────────────────
    while True:
        try:
            # Force cache invalidation after 3 misses to prevent serving stale CDN data.
            if consecutive_misses > 0 and consecutive_misses % 3 == 0:
                scraper.invalidate_cache()

            # ── 2. LiveScore CDN live feed ────────────────────────────────────
            live_matches = await scraper.fetch_live_scores()
            match = scraper.find_match(player_a, player_b, live_matches)
            if live_matches and not match:
                log.debug(
                    "[LIVESCORE] %d matches live, none matched '%s' vs '%s'. "
                    "Available: %s",
                    len(live_matches), player_a, player_b,
                    [(m["player_a"], m["player_b"]) for m in live_matches],
                )

            # ── 3. ESPN fallback (ATP 250/500/1000/Slams) ────────────────────
            if not match:
                espn_matches = await espn_scorer.fetch_live()
                match = scraper.find_match(player_a, player_b, espn_matches)
                if match:
                    log.info(
                        "[ESPN] Live data found for %s vs %s: sets=%s games=%s",
                        player_a, player_b, match["sets"], match["games"],
                    )
                elif espn_matches:
                    log.debug(
                        "[ESPN] %d matches live, none matched '%s' vs '%s'.",
                        len(espn_matches), player_a, player_b,
                    )

            # ── 4. SofaScore fallback (broadest global coverage) ─────────────
            if not match:
                sofa_matches = await sofascore_scraper.fetch_live()
                match = scraper.find_match(player_a, player_b, sofa_matches)
                if match:
                    log.info(
                        "[SOFASCORE] Live data found for %s vs %s: sets=%s games=%s",
                        player_a, player_b, match["sets"], match["games"],
                    )
                elif sofa_matches:
                    log.debug(
                        "[SOFASCORE] %d matches live, none matched '%s' vs '%s'. "
                        "Available: %s",
                        len(sofa_matches), player_a, player_b,
                        [(m["player_a"], m["player_b"]) for m in sofa_matches],
                    )

            if match:
                consecutive_misses = 0
                ever_seen_live = True
                _is_scheduled = False
                scraper._last_live_hit = time.time()
                yield {
                    "points":        match["points"],
                    "games":         match["games"],
                    "sets":          match["sets"],
                    "p1_serving":    match["p1_serving"],
                    "tournament":    match.get("tournament", ""),
                    "is_live":       True,
                    "status":        match.get("status", "LIVE"),
                    "source":        match.get("_source", "livescore"),
                    "data_age_secs": 0.0,
                    "injury_flag":   match.get("injury_flag", False),
                }
                await asyncio.sleep(interval)
                continue

            # ── 5. Daily schedule (every 3 misses) ───────────────────────────
            if consecutive_misses % 3 == 0:
                daily_matches = await scraper.fetch_all_today()
                upcoming_match = scraper.find_match(player_a, player_b, daily_matches)
                if upcoming_match and not _is_scheduled:
                    log.info("[SCRAPER] %s vs %s found in daily schedule (NS).",
                             player_a, player_b)
                _is_scheduled = bool(upcoming_match)

            # ── 6. Flashscore confirmation (every 6 misses) ──────────────────
            if not _is_scheduled and consecutive_misses % 6 == 0:
                fs_id = await flashscore.find_match_id(player_a, player_b)
                if fs_id:
                    _is_scheduled = True
                    log.info("[FLASHSCORE] Match confirmed: ID=%s", fs_id)

            consecutive_misses += 1
            data_age = scraper.data_age_seconds()

            # Give up on matches that have never appeared on any scraper after
            # _GIVE_UP_MISSES polls — these are likely ITF/minor events with no
            # coverage. Releasing the slot frees workers for supported matches.
            if not ever_seen_live and consecutive_misses >= _GIVE_UP_MISSES:
                log.warning(
                    "[LIVESCORE] UNSUPPORTED: %s vs %s never found on any scraper "
                    "after %d polls — releasing worker slot.",
                    player_a, player_b, consecutive_misses,
                )
                return

            if consecutive_misses >= 5 and data_age > _STALE_THRESHOLD:
                log.warning(
                    "[LIVESCORE] STALE: no live data for %.0fs (%d misses) for "
                    "%s vs %s — trading BLOCKED until live score resumes.",
                    data_age, consecutive_misses, player_a, player_b,
                )

            yield {
                "is_live":        False,
                "is_scheduled":   _is_scheduled,
                "misses":         consecutive_misses,
                "data_age_secs":  data_age,
            }

        except Exception as e:
            log.error("Error in poll_live_score_real: %s", e)
            yield {"is_live": False, "error": str(e)}

        await asyncio.sleep(interval)
