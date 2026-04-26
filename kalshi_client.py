"""
kalshi_client.py — Kalshi Demo API wrapper using async HTTP

Handles:
  - Discovering open tennis / head-to-head style markets from live events
  - Authenticated order placement (buy / sell) via RSA-PSS
  - Position tracking
"""

import aiohttp
import logging
import base64
import time
import ssl
import re
import certifi
import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from urllib.parse import urlparse, quote
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from config import Config, normalize_kalshi_pem
import websockets
import json as _json

log = logging.getLogger(__name__)

# Comprehensive exclusion list for non-tennis contexts and mixed sports.
_EXCLUDE_SERIES_SUBSTR = (
    "GOLFTENNIS", "GOLF", "BEZEL", "ROLEX", "KXWC", "KXWCGAME", "FIFA", "WORLD CUP", 
    "SOCCER", "NCAA", "LACROSSE", "LACR", "NHL", "NBA", "NFL", "MLB", "UFC", "MMA",
    "IPL", "CRICKET", "DARTS", "SNOOKER", "NASCAR", "F1", "RACING", "HORSE",
    "CS2", "ESPORTS", "VALORANT", "DOTA", "LOL", "COUNTERSTRIKE"
)

# Mandatory tennis-positive indicators. 
_TENNIS_HINTS = re.compile(
    r"(atp|wta|itf|tennis|grand slam|wimbledon|roland garros|french open|australian open|us open|masters 1000|challenger|clay court|hard court|grass court)",
    re.I,
)

# Comprehensive list of non-tennis sports and indicators to fast-fail.
_NON_TENNIS_HINTS = re.compile(
    r"\b(golf|boxing|world cup|soccer|football|nba|nfl|mlb|nhl|formula 1|ufc|mma|fifa|cricket|rugby|basketball|darts|snooker|lacrosse|baseball|hockey|volleyball|esports|valorant|league of legends|cs2|cs:go|counter-strike|dota|overwatch|call of duty)\b",
    re.I,
)


def _parse_yes_no_cents(m: Dict[str, Any]) -> Tuple[int, int]:
    """Parse YES/NO ask prices from a Kalshi market dict.

    The Kalshi API returns prices in two different formats depending on endpoint:
    - Events endpoint (/events):   yes_ask_dollars / no_ask_dollars  (float, dollars)
    - Markets endpoint (/markets): yes_ask / no_ask                  (int, cents)

    We try both. A sanity check ensures YES+NO is within a realistic spread
    (~85–115 cents). If it fails (e.g. yes_ask_dollars missing → 50 default
    while no_ask_dollars=0.99 → 99, total=149), we fall back to last_price.
    """
    def _try(dollar_key: str, cents_key: str, bid_key: str = None):
        # 1. Dollar-decimal format (events endpoint)
        raw = m.get(dollar_key)
        if raw is not None:
            try:
                v = int(round(float(str(raw)) * 100))
                if 1 <= v <= 99:
                    return v
            except (TypeError, ValueError):
                pass
        # 2. Cents-integer format (markets endpoint)
        raw = m.get(cents_key)
        if raw is not None:
            try:
                v = int(raw)
                if 1 <= v <= 99:
                    return v
            except (TypeError, ValueError):
                pass
        # 3. Bid price as last resort
        if bid_key:
            raw = m.get(bid_key)
            if raw is not None:
                try:
                    v = int(raw)
                    if 1 <= v <= 99:
                        return v
                except (TypeError, ValueError):
                    pass
        return None

    y = _try("yes_ask_dollars", "yes_ask", "yes_bid")
    n = _try("no_ask_dollars", "no_ask", "no_bid")

    # If exactly one side was found, infer the complement (YES + NO ≈ 100 cents).
    # This handles e.g. no_ask_dollars=0.99 present but yes_ask_dollars absent:
    # → y = None, n = 99  → infer y = 1  → total = 100, sanity passes.
    if y is not None and n is None:
        n = max(1, min(99, 100 - y))
    elif n is not None and y is None:
        y = max(1, min(99, 100 - n))

    # Sanity check: in any live binary market YES + NO should be ~100 cents.
    # If the total is wildly off, both values are unreliable — fall back to last_price.
    if y is not None and n is not None and not (85 <= y + n <= 115):
        log.debug(
            "Price sanity failed (yes=%s + no=%s = %s) — re-deriving from last_price.",
            y, n, y + n,
        )
        y, n = None, None

    # Derive from last_price (always in cents on /markets endpoint) as final fallback
    if y is None or n is None:
        lp = m.get("last_price")
        if lp is not None:
            try:
                lp_cents = max(1, min(99, int(lp)))
                y = lp_cents
                n = 100 - lp_cents
            except (TypeError, ValueError):
                pass

    return (y or 50), (n or 50)


def _parse_iso_ts_to_epoch(ts: str) -> float:
    if not ts:
        return float("inf")
    try:
        # Kalshi emits e.g. 2026-09-28T14:00:00Z
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return float("inf")


def _safe_float(val) -> Optional[float]:
    """Convert ATP stat value to float, returning None on failure."""
    if val is None:
        return None
    try:
        s = str(val).replace("%", "").strip()
        return float(s)
    except (TypeError, ValueError):
        return None


def _split_vs_title(title: str) -> Optional[Tuple[str, str]]:
    """Extract two sides from 'A vs B' / 'A vs. B' (optionally after 'Tournament:')."""
    if not title:
        return None
    t = title.strip()
    # Strip leading "Will ...?" style — not head-to-head
    if re.match(r"^will\s+", t, re.I):
        return None
    m = re.search(r"[:]\s*(.+)$", t)
    rest = m.group(1).strip() if m else t
    # Handle 'vs', 'v', '/', and '&' as player separators
    parts = re.split(r"\s+(?:vs\.?|v|/|&)\s+", rest, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    
    # Trim trailing qualifiers that Kalshi adds.
    # Includes set/game markers ("Set 2", "Set 2 Winner") that bleed into player_b.
    _QUAL = (
        r"(?i)\s+(?:"
        r"set\s*\d*\s*winner?"   # "Set 2 Winner", "Set2 Win", "Set 2"
        r"|game\s*\d+"           # "Game 3"
        r"|winner?"              # standalone "Winner"
        r"|atp|wta|itf|challenger|tennis|tournament"
        r"|quaterfinal|quarterfinal|semifinal|final"
        r"|odds|prediction"
        r"|[·|—]"
        r"|\- "
        r"|\("
        r")"
    )
    b = re.split(_QUAL, b)[0].strip()
    a = re.split(_QUAL, a)[0].strip()

    if len(a) < 2 or len(b) < 2:
        return None
    return a, b


def _score_tennis_event(event: Dict[str, Any]) -> int:
    """Higher = better candidate for a tennis head-to-head matchup."""
    st = (event.get("series_ticker") or "").upper()
    et = (event.get("event_ticker") or "").upper()
    title = (event.get("title") or "").upper()
    
    for bad in _EXCLUDE_SERIES_SUBSTR:
        if bad in st or bad in et or bad in title:
            return -1

    sub = event.get("sub_title") or ""
    cat = event.get("category") or ""
    blob = f"{title} {sub} {st} {et} {cat}"

    score = 0
    # Explicit tennis markers are high priority. 
    if "TENNIS" in blob.upper():
        score += 200
        
    if re.search(r"(ATP|WTA|ITF|CHALLENGER)", blob, re.I):
        score += 100
    
    if _split_vs_title(title):
        score += 80
    
    if _TENNIS_HINTS.search(blob):
        score += 70
        
    if cat.lower() == "sports":
        score += 10
        
    if _NON_TENNIS_HINTS.search(blob) and "TENNIS" not in blob.upper():
        score -= 200  # Heavy penalty for non-tennis hints
        
    return score


def _is_tennis_event(event: Dict[str, Any]) -> bool:
    st = (event.get("series_ticker") or "").upper()
    et = (event.get("event_ticker") or "").upper()
    title = (event.get("title") or "").upper()
    sub = event.get("sub_title") or ""
    cat = event.get("category") or ""
    blob = f"{title} {sub} {st} {et} {cat}"

    # Strict exclusion check
    for bad in _EXCLUDE_SERIES_SUBSTR:
        if bad in st or bad in et or bad in title:
            return False
            
    if "TENNIS" in blob.upper():
        return True
            
    if _NON_TENNIS_HINTS.search(blob):
        return False
        
    # MUST have an explicit tennis hint to pass.
    has_explicit = bool(re.search(r"(ATP|WTA|ITF|CHALLENGER)", blob, re.I))
    if has_explicit:
        return True
        
    return bool(_TENNIS_HINTS.search(blob))


def _players_from_event(event: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Strict H2H extraction from title or sub_title."""
    title = event.get("title") or ""
    sub_title = event.get("sub_title") or ""
    
    # Prioritize sub_title because Kalshi often puts FULL names there 
    # (e.g. "Francisco Cerundolo vs Pedro Martinez") 
    # while title contains short names ("Cerundolo vs Martinez")
    pair = _split_vs_title(sub_title)
    if not pair:
        pair = _split_vs_title(title)
        
    if not pair:
        return None
    a, b = pair
    # Must be two distinct player names (not props / group labels).
    if a.lower() == b.lower():
        return None

    # Blacklist for teams, politics, and generic labels
    bad_tokens = (
        "any ", "other than", "combined", "field", "team", "club", "fc ", " united", 
        "republic", "state", "city", "town", "rovers", "wanderers", "athletic", 
        "union", "democrat", "republican", "biden", "trump", "harris", "election",
        "maps", "total", "bet", "points", "games", "spread", "over/under"
    )
    lc = f"{a.lower()} {b.lower()}"
    if any(tok in lc for tok in bad_tokens):
        return None
        
    # Tennis players don't usually have "at" in their name parts if split incorrectly
    if " at " in lc:
        return None

    if len(a) > 48 or len(b) > 48:
        return None
    return a, b


class KalshiClient:
    def __init__(self, config: Config):
        self.cfg = config
        self.base_url = config.KALSHI_API_URL
        # Discovery often works better on the dedicated public events endpoint
        self.discovery_url = config.KALSHI_PUBLIC_EVENT_BASE if config.KALSHI_USE_PROD else config.KALSHI_DEMO_BASE
        
        self._positions: dict = {}
        self._http: Optional[aiohttp.ClientSession] = None
        # Balance cache: (fetched_at, value_usd) — re-fetch only when stale
        self._balance_cache: Tuple[float, float] = (0.0, 0.0)
        self._BALANCE_CACHE_TTL = 15.0

        self.api_key_id = config.KALSHI_API_KEY_ID
        self.private_key_pem = (
            normalize_kalshi_pem(config.KALSHI_PRIVATE_KEY_PEM)
            if config.KALSHI_PRIVATE_KEY_PEM
            else ""
        )
        self.private_key = None

        if self.api_key_id and self.private_key_pem:
            try:
                self.private_key = serialization.load_pem_private_key(
                    self.private_key_pem.encode("utf-8"),
                    password=None,
                )
                log.info(
                    "KalshiClient AUTHENTICATED on %s — REAL trades will execute.",
                    "PROD" if config.KALSHI_USE_PROD else "DEMO",
                )
            except Exception as e:
                log.error(
                    "Failed to load Kalshi private key: %s  "
                    "→ falling back to DRY-RUN (no real trades will execute). "
                    "Fix: paste a valid RSA-PSS PEM key via /api/set_kalshi_keys.", e
                )
                self.private_key = None
        else:
            missing = []
            if not self.api_key_id:
                missing.append("KALSHI_API_KEY_ID")
            if not self.private_key_pem:
                missing.append("KALSHI_PRIVATE_KEY_PEM")
            log.warning(
                "Kalshi credentials missing (%s). DRY-RUN mode on %s — "
                "no real orders will be placed. Paste keys via the dashboard.",
                ", ".join(missing), "PROD" if config.KALSHI_USE_PROD else "DEMO",
            )

    async def close(self):
        if self._http and not self._http.closed:
            await self._http.close()
            self._http = None

    def _sign_request(self, timestamp: str, method: str, path: str) -> str:
        if not self.private_key:
            return ""
        # Kalshi V2 signature requires the full path including query parameters
        message = f"{timestamp}{method}{path}"
        message_bytes = message.encode("utf-8")
        signature = self.private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    async def _request(self, method: str, path: str, use_discovery: bool = False, **kwargs) -> dict:
        base = self.discovery_url if use_discovery else self.base_url
        url = f"{base}{path}"
        parsed = urlparse(base)
        # Signature uses the full absolute path from the root
        sign_path = f"{parsed.path}{path}"

        headers = kwargs.pop("headers", {})

        # Only add auth headers if we have a private key AND we're not using the public discovery endpoint
        if self.private_key and not use_discovery:
            timestamp = str(int(time.time() * 1000))
            headers["KALSHI-ACCESS-KEY"] = self.api_key_id
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp
            headers["KALSHI-ACCESS-SIGNATURE"] = self._sign_request(
                timestamp, method, sign_path
            )

        kwargs["headers"] = headers

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        _timeout = aiohttp.ClientTimeout(total=20, connect=10)
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with self._http.request(method, url, ssl=ssl_ctx, timeout=_timeout, **kwargs) as resp:
                        if resp.status == 401 or resp.status == 403:
                            log.error(f"Kalshi AUTH ERROR {resp.status} on {method} {path}. Check your keys and KALSHI_USE_PROD setting.")
                        
                        data = await resp.json()

                        # 502/503/504 are transient — retry
                        if resp.status in [502, 503, 504] and attempt < max_retries - 1:
                            log.warning(f"Kalshi API {resp.status} on {method} {path}. Retrying ({attempt+1}/{max_retries})...")
                            await asyncio.sleep(2 ** attempt)
                            continue

                        # 4xx errors are permanent — never retry
                        if resp.status >= 400:
                            log.error(f"Kalshi API Error {resp.status} on {method} {path}: {data}")
                            raise Exception(f"Kalshi API Error: {data}")
                        return data
                except Exception as e:
                    # Don't retry permanent API errors (4xx); only retry connection/network issues
                    if "Kalshi API Error" in str(e):
                        raise
                    if isinstance(e, aiohttp.ClientConnectorError):
                        log.error(f"Failed to connect to Kalshi at {url}: {e}")
                        if attempt == max_retries - 1:
                            raise Exception(f"Connection error to {base}. Make sure the URL is correct and you have internet access.")

                    if attempt == max_retries - 1:
                        raise
                    log.warning(f"Kalshi request failed ({e}). Retrying ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(2 ** attempt)
        except Exception as e:
            raise
    # Known Kalshi series tickers that contain live H2H tennis matchups.
    # Querying these directly bypasses the need to paginate thousands of events.
    _TENNIS_SERIES = (
        "KXATPMATCH",
        "KXWTAMATCH",
        "KXITFMATCH",
        "KXITFWMATCH",
        "KXATPCHALLENGERMATCH",
        "KXWTACHALLENGERMATCH",
        "KXATPSETWINNER",
    )

    async def _fetch_series_events(self, series_ticker: str) -> List[Dict]:
        """Fetch all open events for a single known tennis series ticker."""
        events: List[Dict] = []
        cursor = ""
        for _ in range(10):
            q = f"status=open&limit=200&with_nested_markets=true&series_ticker={quote(series_ticker, safe='')}"
            if cursor:
                q += f"&cursor={quote(cursor, safe='')}"
            try:
                resp = await self._request("GET", f"/events?{q}", use_discovery=True)
            except Exception as exc:
                log.warning("[DISCOVER] series %s fetch failed: %s", series_ticker, exc)
                break
            events.extend(resp.get("events") or [])
            cursor = resp.get("cursor") or ""
            if not cursor:
                break
        return events

    async def get_atp_markets(self) -> List[Dict]:
        """
        Discover open H2H tennis markets.

        Fast path: query known tennis series tickers directly (bypasses pagination).
        Fallback: general open-events pagination for the configured number of pages.
        """
        candidates: List[Tuple[int, Dict, Dict]] = []
        env_label = "PRODUCTION" if self.cfg.KALSHI_USE_PROD else "DEMO"
        log.info("Scanning for tennis markets on %s...", env_label)

        # ── Fast path: query known H2H series directly ─────────────────────────
        direct_events: List[Dict] = []
        for series in self._TENNIS_SERIES:
            batch = await self._fetch_series_events(series)
            if batch:
                log.info("[DISCOVER] %s: %d event(s) found", series, len(batch))
            direct_events.extend(batch)

        # ── Fallback: paginated general scan ───────────────────────────────────
        max_pages = getattr(self.cfg, "KALSHI_EVENTS_MAX_PAGES", 100)
        paged_events: List[Dict] = []
        cursor = ""
        for page in range(max_pages):
            q = "status=open&limit=200&with_nested_markets=true"
            if cursor:
                q += f"&cursor={quote(cursor, safe='')}"
            try:
                resp = await self._request("GET", f"/events?{q}", use_discovery=True)
            except Exception as exc:
                log.warning("Kalshi events page %d failed: %s", page + 1, exc)
                break
            paged_events.extend(resp.get("events") or [])
            cursor = resp.get("cursor") or ""
            if not cursor:
                break

        # Merge, deduplicate by event_ticker
        seen_tickers: set = set()
        all_events: List[Dict] = []
        for ev in direct_events + paged_events:
            et = ev.get("event_ticker") or ""
            if et and et not in seen_tickers:
                seen_tickers.add(et)
                all_events.append(ev)

        total_scanned = len(all_events)
        log.info("[DISCOVER] %d unique events to evaluate (direct=%d + paged=%d)",
                 total_scanned, len(direct_events), len(paged_events))

        # ── Score and filter ───────────────────────────────────────────────────
        now_ts = time.time()
        for event in all_events:
            title = event.get("title", "")
            sc = _score_tennis_event(event)
            is_tennis = _is_tennis_event(event)

            if is_tennis and sc > 0:
                log.info("TENNIS DETECTED: %s (Ticker: %s)", title, event.get("event_ticker"))

            if not is_tennis or sc <= 0:
                continue

            players = _players_from_event(event)
            if not players:
                log.debug("SKIPPING event (could not extract players): %s", title)
                continue
            player_a, player_b = players

            markets = event.get("markets") or []
            if not markets:
                et = event.get("event_ticker")
                try:
                    m_resp = await self._request(
                        "GET", f"/markets?event_ticker={quote(et, safe='')}"
                    )
                    markets = m_resp.get("markets") or []
                except Exception as ex:
                    log.warning("No nested markets and fetch failed for %s: %s", et, ex)
                    continue

            for m in markets:
                if m.get("status") != "active":
                    continue
                t = m.get("ticker") or ""
                if not t:
                    continue
                close_ts = _parse_iso_ts_to_epoch(m.get("close_time", ""))
                if close_ts < now_ts:
                    log.debug("Skipping expired market %s", t)
                    continue
                yc, nc = _parse_yes_no_cents(m)
                candidates.append((
                    sc,
                    event,
                    {
                        "ticker":         t,
                        "player_a":       player_a,
                        "player_b":       player_b,
                        "title":          event.get("title") or "",
                        "event_ticker":   event.get("event_ticker") or "",
                        "series_ticker":  event.get("series_ticker") or "",
                        "close_time":     m.get("close_time", ""),
                        "yes_price_cents": yc,
                        "no_price_cents":  nc,
                        "market_data":    m,
                    },
                ))
                break  # one market per event is enough

        def _sort_key(item: Tuple[int, Dict, Dict]):
            sc, event, _ = item
            close_ts = _parse_iso_ts_to_epoch(item[2].get("close_time", ""))
            # Primary: closest-closing tennis market; secondary: relevance score.
            return (close_ts, -sc)

        candidates.sort(key=_sort_key)
        out = [c[2] for c in candidates]
        seen = set()
        unique: List[Dict] = []
        for row in out:
            tk = row["ticker"]
            if tk in seen:
                continue
            seen.add(tk)
            unique.append(row)
            if len(unique) >= 48:
                break

        if not unique:
            log.warning(
                "Kalshi: Scanned %d events. No qualifying H2H tennis markets found. "
                "Markets may not be live yet — retrying in %ds.",
                total_scanned, getattr(self.cfg, "DISCOVERY_INTERVAL", 60),
            )
        else:
            log.info(
                "Kalshi: Scanned %d events. Found %d tennis matchup(s). "
                "Primary: %s vs %s (%s)",
                total_scanned,
                len(unique),
                unique[0]["player_a"],
                unique[0]["player_b"],
                unique[0]["ticker"],
            )
        return unique

    async def get_market(self, ticker: str) -> dict:
        """Fetch current market state by ticker."""
        resp = await self._request("GET", f"/markets/{quote(ticker, safe='')}")
        m = resp.get("market", {})
        log.debug(
            "RAW market fields for %s: yes_ask_dollars=%s yes_ask=%s yes_bid=%s "
            "no_ask_dollars=%s no_ask=%s no_bid=%s last_price=%s status=%s",
            ticker,
            m.get("yes_ask_dollars"), m.get("yes_ask"), m.get("yes_bid"),
            m.get("no_ask_dollars"), m.get("no_ask"), m.get("no_bid"),
            m.get("last_price"), m.get("status"),
        )
        yc, nc = _parse_yes_no_cents(m)
        return {
            "id": ticker,
            "question": m.get("title", ""),
            "yes_price": yc / 100.0,
            "no_price": nc / 100.0,
            "active": m.get("status") == "active",
            "_raw": m,
        }

    async def buy(
        self,
        ticker: str,
        price_cents: int,
        count: int,
        side: str,
    ) -> dict:
        log.info(f"BUY {count} contracts of {side.upper()} @ {price_cents}c on {ticker}")

        if not self.private_key:
            self._update_position(ticker, "BUY", side, count, price_cents)
            return {
                "status": "DRY_RUN",
                "ticker": ticker,
                "side": side,
                "count": count,
                "price_cents": price_cents,
            }

        # Slippage cap: price_cents is the maximum we'll pay per contract.
        # Kalshi treats {side}_price on a market buy as the worst-case fill price.
        cap_cents = max(1, min(99, int(price_cents)))
        payload = {
            "ticker": ticker,
            "action": "buy",
            "type": "market",
            "side": side.lower(),
            "client_order_id": str(uuid.uuid4()),
            "count": count,
            f"{side.lower()}_price": cap_cents,
        }

        log.info(
            "REAL_TRADE BUY executing: ticker=%s side=%s count=%d price=%dc env=%s",
            ticker, side, count, cap_cents, "PROD" if self.cfg.KALSHI_USE_PROD else "DEMO",
        )
        resp = await self._request("POST", "/portfolio/orders", json=payload)
        log.info("REAL_TRADE BUY response: %s", resp)
        self._update_position(ticker, "BUY", side, count, price_cents)
        return resp

    async def sell(
        self,
        ticker: str,
        price_cents: int,
        count: int,
        side: str,
    ) -> dict:
        log.info(f"SELL {count} contracts of {side.upper()} @ {price_cents}c on {ticker}")

        if not self.private_key:
            self._update_position(ticker, "SELL", side, count, price_cents)
            return {
                "status": "DRY_RUN",
                "ticker": ticker,
                "side": side,
                "count": count,
                "price_cents": price_cents,
            }

        # Slippage floor: price_cents is the minimum we'll accept per contract.
        # Kalshi treats {side}_price on a market sell as the worst-case fill price.
        floor_cents = max(1, min(99, int(price_cents)))
        payload = {
            "ticker": ticker,
            "action": "sell",
            "type": "market",
            "side": side.lower(),
            "client_order_id": str(uuid.uuid4()),
            "count": count,
            f"{side.lower()}_price": floor_cents,
        }

        log.info(
            "REAL_TRADE SELL executing: ticker=%s side=%s count=%d price=%dc env=%s",
            ticker, side, count, floor_cents, "PROD" if self.cfg.KALSHI_USE_PROD else "DEMO",
        )
        resp = await self._request("POST", "/portfolio/orders", json=payload)
        log.info("REAL_TRADE SELL response: %s", resp)
        self._update_position(ticker, "SELL", side, count, price_cents)
        return resp

    async def place_limit_order(
        self,
        ticker: str,
        price_cents: int,
        count: int,
        side: str,
    ) -> dict:
        """
        Place a resting limit order at an exact price.

        Unlike buy() (market order), this does NOT fill immediately — it sits
        in the Kalshi order book until the market price reaches price_cents, or
        until cancelled via cancel_order().

        Use for predictive entries: compute the anticipated post-point price,
        place the order BEFORE the point is played, get filled at the pre-move
        price when the score changes.
        """
        log.info(f"LIMIT {count}x {side.upper()} @ {price_cents}¢ on {ticker}")

        if not self.private_key:
            self._update_position(ticker, "BUY", side, count, price_cents)
            return {
                "status": "DRY_RUN",
                "ticker": ticker,
                "side": side,
                "count": count,
                "price_cents": price_cents,
                "order_type": "limit",
            }

        cap_cents = max(1, min(99, int(price_cents)))
        payload = {
            "ticker": ticker,
            "action": "buy",
            "type": "limit",
            "side": side.lower(),
            "client_order_id": str(uuid.uuid4()),
            "count": count,
            f"{side.lower()}_price": cap_cents,
        }
        resp = await self._request("POST", "/portfolio/orders", json=payload)
        return resp

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel a resting limit order by its order ID."""
        if not self.private_key:
            log.info(f"DRY_RUN cancel order {order_id}")
            return {"status": "DRY_RUN_CANCELLED"}
        try:
            resp = await self._request("DELETE", f"/portfolio/orders/{quote(order_id, safe='')}")
            log.info(f"Cancelled order {order_id}")
            return resp
        except Exception as e:
            log.warning(f"Cancel order {order_id} failed: {e}")
            return {"status": "cancel_failed", "error": str(e)}

    async def get_balance(self, force: bool = False) -> float:
        """Fetch available (free) balance in USD, cached for _BALANCE_CACHE_TTL seconds."""
        if not self.private_key:
            return 1000.0  # Default for dry-run
        cached_ts, cached_val = self._balance_cache
        if not force and (time.time() - cached_ts) < self._BALANCE_CACHE_TTL:
            return cached_val
        try:
            resp = await self._request("GET", "/portfolio/balance")
            cents = resp.get("available_balance", resp.get("balance", 0))
            value = float(cents) / 100.0
            self._balance_cache = (time.time(), value)
            return value
        except Exception as e:
            log.error(f"Failed to fetch balance: {e}")
            return cached_val  # return stale rather than $0 on transient error

    def get_position(self, pos_key: str) -> dict:
        return self._positions.get(pos_key, {"count": 0, "avg_price": 0.0, "side": None})

    def _update_position(self, ticker: str, action: str, side: str, count: int, price: int):
        pos_key = f"{ticker}_{side}"
        pos = self._positions.get(pos_key, {"count": 0, "avg_price": 0.0, "side": side})

        if action == "BUY":
            total_cost = pos["count"] * pos["avg_price"] + count * price
            pos["count"] += count
            pos["avg_price"] = total_cost / pos["count"] if pos["count"] else price
        elif action == "SELL":
            pos["count"] = max(0, pos["count"] - count)
            if pos["count"] == 0:
                pos["avg_price"] = 0.0

        self._positions[pos_key] = pos

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: WebSocket Streams
    # ──────────────────────────────────────────────────────────────────────────

    async def stream_orderbook(self, ticker: str, callback):
        """
        Subscribe to Kalshi WebSocket orderbook channel for a single ticker.
        Calls callback(snapshot_dict) on every update.

        Replaces polling GET /markets/{ticker}/orderbook — eliminates HTTP overhead
        and gives sub-millisecond latency vs the 800ms poll cycle.
        """
        if not self.private_key:
            log.info(f"[WS] orderbook stream skipped for {ticker} — DRY-RUN mode (no credentials)")
            return

        ws_base = (
            "wss://api.elections.kalshi.com/trade-api/ws/v2"
            if self.cfg.KALSHI_USE_PROD
            else "wss://demo-api.kalshi.co/trade-api/ws/v2"
        )
        sid = 1
        subscribe_msg = _json.dumps({
            "id": sid,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [ticker],
            }
        })

        backoff = 1.0
        while True:
            try:
                timestamp = str(int(time.time() * 1000))
                path = "/trade-api/ws/v2"
                
                # Attach authentication headers if private key exists
                headers = {}
                if self.private_key:
                    signature = self._sign_request(timestamp, "GET", path)
                    headers = {
                        "KALSHI-ACCESS-KEY": self.api_key_id,
                        "KALSHI-ACCESS-TIMESTAMP": timestamp,
                        "KALSHI-ACCESS-SIGNATURE": signature,
                    }
                    
                _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                async with websockets.connect(ws_base, additional_headers=headers,
                                              ssl=_ssl_ctx,
                                              ping_interval=20, ping_timeout=10) as ws:
                    log.info(f"[WS] Connected to orderbook stream for {ticker}")
                    backoff = 1.0
                    await ws.send(subscribe_msg)
                    async for raw in ws:
                        try:
                            msg = _json.loads(raw)
                            msg_type = msg.get("type", "")
                            if msg_type in ("orderbook_snapshot", "orderbook_delta"):
                                await callback(msg)
                        except Exception as e:
                            log.debug(f"[WS] orderbook parse error: {e}")
            except asyncio.CancelledError:
                log.info(f"[WS] orderbook stream cancelled for {ticker}")
                return
            except Exception as e:
                log.warning(f"[WS] orderbook stream error: {e} — reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def stream_user_fills(self, callback):
        """
        Subscribe to authenticated user_fills WebSocket channel.
        Calls callback(fill_dict) on every fill/partial-fill event.

        Replaces polling GET /portfolio/orders — fills arrive in real-time
        so position state is always current without additional HTTP calls.
        """
        if not self.private_key:
            log.warning("[WS] user_fills stream requires authenticated credentials — skipping.")
            return

        ws_base = (
            "wss://api.elections.kalshi.com/trade-api/ws/v2"
            if self.cfg.KALSHI_USE_PROD
            else "wss://demo-api.kalshi.co/trade-api/ws/v2"
        )

        backoff = 1.0
        while True:
            try:
                timestamp = str(int(time.time() * 1000))
                path = "/trade-api/ws/v2"
                signature = self._sign_request(timestamp, "GET", path)
                headers = {
                    "KALSHI-ACCESS-KEY": self.api_key_id,
                    "KALSHI-ACCESS-TIMESTAMP": timestamp,
                    "KALSHI-ACCESS-SIGNATURE": signature,
                }
                _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                async with websockets.connect(ws_base, additional_headers=headers,
                                              ssl=_ssl_ctx,
                                              ping_interval=20, ping_timeout=10) as ws:
                    log.info("[WS] Connected to user_fills stream")
                    backoff = 1.0
                    subscribe_msg = _json.dumps({
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {"channels": ["user_fills", "user_orders"]},
                    })
                    await ws.send(subscribe_msg)
                    async for raw in ws:
                        try:
                            msg = _json.loads(raw)
                            msg_type = msg.get("type", "")
                            if msg_type in ("user_fill", "user_order"):
                                await callback(msg)
                        except Exception as e:
                            log.debug(f"[WS] user_fills parse error: {e}")
            except asyncio.CancelledError:
                log.info("[WS] user_fills stream cancelled")
                return
            except Exception as e:
                log.warning(f"[WS] user_fills stream error: {e} — reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def fetch_atp_live_stats(self, _session=None) -> dict:
        """
        Scrape the ATP Infosys live stats endpoint using cloudscraper to bypass
        Cloudflare. Returns a dict of {match_id: {serve_pct, break_pct, ...}}.

        ATP uses AJAX/RequireJS — standard requests/BeautifulSoup fail.
        cloudscraper handles the JS challenge transparently.
        """
        import cloudscraper
        url = "https://www.atptour.com/-/ajax/Scores/GetInitialScores"
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            # Run blocking call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: scraper.get(url, timeout=10)
            )
            if resp.status_code != 200:
                log.warning(f"[ATP] Scrape returned {resp.status_code}")
                return {}
            data = resp.json()
            return self._parse_atp_live_json(data)
        except Exception as e:
            log.warning(f"[ATP] Live stats fetch failed: {e}")
            return {}

    def _parse_atp_live_json(self, data: dict) -> dict:
        """
        Parse deeply nested ATP Infosys JSON response.
        Structure: data["liveScores"]["Tournaments"][n]["Matches"][m]

        Returns dict keyed by player pair tuple: {(p1_name, p2_name): stats_dict}
        """
        result = {}
        try:
            tournaments = (data.get("liveScores") or data).get("Tournaments") or []
            for tourney in tournaments:
                for match in (tourney.get("Matches") or []):
                    status = (match.get("MatchStatus") or "").lower()
                    if status not in ("in progress", "live", "playing"):
                        continue
                    teams = match.get("Teams") or match.get("Players") or []
                    if len(teams) < 2:
                        continue
                    p1 = (teams[0].get("PlayerName") or teams[0].get("Name") or "").strip()
                    p2 = (teams[1].get("PlayerName") or teams[1].get("Name") or "").strip()
                    stats_raw = match.get("Statistics") or match.get("Stats") or {}
                    # Normalize common field names across ATP API versions
                    stats = {
                        "first_serve_pct_p1":     _safe_float(stats_raw.get("FirstServePercentage1") or stats_raw.get("1stServePct1")),
                        "first_serve_pct_p2":     _safe_float(stats_raw.get("FirstServePercentage2") or stats_raw.get("1stServePct2")),
                        "pts_won_on_1st_serve_p1": _safe_float(stats_raw.get("PointsWonOn1stServe1")),
                        "pts_won_on_1st_serve_p2": _safe_float(stats_raw.get("PointsWonOn1stServe2")),
                        "pts_won_on_2nd_serve_p1": _safe_float(stats_raw.get("PointsWonOn2ndServe1")),
                        "pts_won_on_2nd_serve_p2": _safe_float(stats_raw.get("PointsWonOn2ndServe2")),
                        "break_pts_converted_p1":  _safe_float(stats_raw.get("BreakPointsConverted1")),
                        "break_pts_converted_p2":  _safe_float(stats_raw.get("BreakPointsConverted2")),
                        "aces_p1":                 _safe_float(stats_raw.get("Aces1")),
                        "aces_p2":                 _safe_float(stats_raw.get("Aces2")),
                        "double_faults_p1":        _safe_float(stats_raw.get("DoubleFaults1")),
                        "double_faults_p2":        _safe_float(stats_raw.get("DoubleFaults2")),
                        "serve_speed_avg_p1":      _safe_float(stats_raw.get("AverageServeSpeed1")),
                        "serve_speed_avg_p2":      _safe_float(stats_raw.get("AverageServeSpeed2")),
                    }
                    result[(p1, p2)] = stats
        except Exception as e:
            log.warning(f"[ATP] JSON parse failed: {e}")
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4: Kalshi Order Groups (OCO / take-profit management)
    # ──────────────────────────────────────────────────────────────────────────

    async def create_order_group(
        self,
        order_ids: list,
        contracts_limit: int,
        group_type: str = "OCO",
    ) -> dict:
        """
        POST /portfolio/order_groups/create

        Links multiple resting limit orders under one group_id so Kalshi's
        matching engine auto-cancels the rest the instant contracts_limit fills.
        Eliminates the need for local state management to prevent double-sells.

        Args:
            order_ids: list of order IDs to link
            contracts_limit: cancel remaining when this many contracts fill
            group_type: "OCO" (one-cancels-other) or "LIMIT"

        Returns dict with "order_group_id" key.
        """
        if not self.private_key:
            log.info(f"[ORDER_GROUP] DRY_RUN create_order_group {order_ids}")
            return {"order_group_id": f"dry_run_{int(time.time())}"}

        payload = {
            "order_ids": order_ids,
            "contracts_limit": contracts_limit,
            "type": group_type,
        }
        try:
            resp = await self._request("POST", "/portfolio/order_groups/create", json=payload)
            group_id = (resp.get("order_group") or resp).get("order_group_id", "")
            log.info(f"[ORDER_GROUP] Created group {group_id} for orders {order_ids}")
            return resp
        except Exception as e:
            log.warning(f"[ORDER_GROUP] create failed: {e}")
            return {}

    async def trigger_order_group(self, order_group_id: str) -> dict:
        """
        PUT /portfolio/order_groups/{order_group_id}/trigger

        Instantly cancels ALL resting orders in the group with a single API call.
        Used for emergency liquidation when the model reverses or adverse news hits —
        far faster than issuing individual DELETE requests per order.
        """
        if not self.private_key:
            log.info(f"[ORDER_GROUP] DRY_RUN trigger_order_group {order_group_id}")
            return {"status": "DRY_RUN_TRIGGERED"}

        try:
            resp = await self._request(
                "PUT",
                f"/portfolio/order_groups/{quote(order_group_id, safe='')}/trigger",
            )
            log.info(f"[ORDER_GROUP] Triggered (cancelled all) group {order_group_id}")
            return resp
        except Exception as e:
            log.warning(f"[ORDER_GROUP] trigger failed for {order_group_id}: {e}")
            return {"status": "trigger_failed", "error": str(e)}

    async def place_limit_order_in_group(
        self,
        ticker: str,
        price_cents: int,
        count: int,
        side: str,
        order_group_id: str = None,
    ) -> dict:
        """
        Place a resting limit order, optionally attached to an order group.
        post_only=True ensures maker-only execution (0% fee on sporting markets).
        """
        log.info(f"LIMIT {count}x {side.upper()} @ {price_cents}¢ on {ticker}"
                 + (f" [group={order_group_id}]" if order_group_id else ""))

        if not self.private_key:
            return {
                "status": "DRY_RUN",
                "ticker": ticker, "side": side,
                "count": count, "price_cents": price_cents,
                "order_type": "limit",
            }

        cap_cents = max(1, min(99, int(price_cents)))
        payload = {
            "ticker": ticker,
            "action": "buy",
            "type": "limit",
            "side": side.lower(),
            "client_order_id": str(uuid.uuid4()),
            "count": count,
            f"{side.lower()}_price": cap_cents,
            "post_only": True,   # maker-only → 0% fee
        }
        if order_group_id:
            payload["order_group_id"] = order_group_id

        resp = await self._request("POST", "/portfolio/orders", json=payload)
        return resp
