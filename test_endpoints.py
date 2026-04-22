"""
test_endpoints.py — Live endpoint connectivity tests for all data sources.

Run with:
    python test_endpoints.py
    python test_endpoints.py --sportradar-ws   # also test the WS feed (waits 15s)
    python test_endpoints.py --verbose         # show full response payloads

Tests every source the bot relies on:
  1.  LiveScore CDN live feed
  2.  LiveScore CDN daily schedule
  3.  ESPN ATP scoreboard
  4.  ESPN WTA scoreboard
  5.  SofaScore live feed
  6.  tennisstats.com ATP rankings
  7.  tennisstats.com WTA rankings
  8.  tennisstats.com player profile (Carlos Alcaraz)
  9.  ATP Stats Centre AJAX (GetInitialScores)
  10. SportRadar REST — live schedule   (requires API key)
  11. SportRadar REST — daily schedule  (requires API key)
  12. SportRadar WebSocket push feed    (requires API key + --sportradar-ws flag)
  13. Flashscore discovery feed         (expected 401 — graceful-degradation check)
"""

import argparse
import asyncio
import json
import os
import ssl
import sys
import time
from datetime import date
from typing import Optional

import aiohttp

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_VERBOSE = False

PASS = "  \033[32m✓ PASS\033[0m"
FAIL = "  \033[31m✗ FAIL\033[0m"
WARN = "  \033[33m⚠ WARN\033[0m"
SKIP = "  \033[90m– SKIP\033[0m"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_key(key_name: str) -> str:
    if os.path.exists("kalshi_keys.json"):
        try:
            with open("kalshi_keys.json") as f:
                return json.load(f).get(key_name, "")
        except Exception:
            pass
    return os.getenv(key_name.upper(), "")


def _vprint(label: str, data):
    if _VERBOSE:
        txt = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
        lines = txt.splitlines()[:30]
        for ln in lines:
            print(f"      {ln}")
        if len(txt.splitlines()) > 30:
            print("      … (truncated)")


async def _get_json(session: aiohttp.ClientSession, url: str,
                    headers: dict = None, timeout: float = 12.0) -> tuple:
    """Return (status_code, parsed_json_or_None, elapsed_ms)."""
    t0 = time.time()
    try:
        async with session.get(
            url,
            headers=headers or {},
            timeout=aiohttp.ClientTimeout(total=timeout),
            ssl=_SSL_CTX,
        ) as r:
            elapsed = (time.time() - t0) * 1000
            try:
                body = await r.json(content_type=None)
            except Exception:
                body = None
            return r.status, body, elapsed
    except asyncio.TimeoutError:
        return 0, None, (time.time() - t0) * 1000
    except Exception as exc:
        return -1, str(exc), (time.time() - t0) * 1000


def _result_line(name: str, status: int, elapsed: float,
                 note: str = "", ok: bool = True) -> str:
    icon = PASS if ok else FAIL
    return f"{icon}  [{status}]  {name}  ({elapsed:.0f}ms){('  — ' + note) if note else ''}"


# ── individual test functions ─────────────────────────────────────────────────

async def test_livescore_live(session: aiohttp.ClientSession):
    print("\n[1] LiveScore CDN — live tennis feed")
    url = "https://prod-cdn-mev-api.livescore.com/v1/api/app/live/tennis/-5?countryCode=US&locale=en"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Origin": "https://www.livescore.com",
        "Referer": "https://www.livescore.com/",
    }
    status, body, ms = await _get_json(session, url, headers)
    ok = status == 200 and isinstance(body, dict)
    stages = (body or {}).get("Stages", []) if ok else []
    n_matches = sum(len(s.get("Events", [])) for s in stages)
    print(_result_line("livescore.com live feed", status, ms,
                       f"{n_matches} event(s) in {len(stages)} stage(s)", ok))
    _vprint("livescore live", body)
    return ok


async def test_livescore_daily(session: aiohttp.ClientSession):
    print("\n[2] LiveScore CDN — daily schedule")
    today = date.today().strftime("%Y%m%d")
    url = f"https://prod-cdn-mev-api.livescore.com/v1/api/app/date/tennis/{today}/0?countryCode=US&locale=en"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Origin": "https://www.livescore.com",
        "Referer": "https://www.livescore.com/",
    }
    status, body, ms = await _get_json(session, url, headers)
    ok = status == 200 and isinstance(body, dict)
    stages = (body or {}).get("Stages", []) if ok else []
    n_matches = sum(len(s.get("Events", [])) for s in stages)
    print(_result_line("livescore.com daily schedule", status, ms,
                       f"{n_matches} event(s) today", ok))
    _vprint("livescore daily", body)
    return ok


async def test_espn_atp(session: aiohttp.ClientSession):
    print("\n[3] ESPN — ATP scoreboard")
    url = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
    status, body, ms = await _get_json(session, url)
    ok = status == 200 and isinstance(body, dict)
    n_events = len((body or {}).get("events", [])) if ok else 0
    print(_result_line("ESPN ATP scoreboard", status, ms, f"{n_events} event(s)", ok))
    _vprint("espn atp", body)
    return ok


async def test_espn_wta(session: aiohttp.ClientSession):
    print("\n[4] ESPN — WTA scoreboard")
    url = "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard"
    status, body, ms = await _get_json(session, url)
    ok = status == 200 and isinstance(body, dict)
    n_events = len((body or {}).get("events", [])) if ok else 0
    print(_result_line("ESPN WTA scoreboard", status, ms, f"{n_events} event(s)", ok))
    _vprint("espn wta", body)
    return ok


async def test_sofascore(session: aiohttp.ClientSession):
    print("\n[5] SofaScore — live tennis events")
    url = "https://api.sofascore.com/api/v1/sport/tennis/events/live"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://www.sofascore.com/",
        "x-locale": "en_INT",
    }
    status, body, ms = await _get_json(session, url, headers)
    ok = status == 200 and isinstance(body, dict)
    n_events = len((body or {}).get("events", [])) if ok else 0
    print(_result_line("sofascore.com live", status, ms, f"{n_events} live event(s)", ok))
    _vprint("sofascore", body)
    return ok


async def test_tennisstats_rankings(session: aiohttp.ClientSession, tour: str = "atp"):
    label_num = 6 if tour == "atp" else 7
    print(f"\n[{label_num}] tennisstats.com — {tour.upper()} rankings")
    url = f"https://tennisstats.com/rankings/{tour}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    t0 = time.time()
    try:
        async with session.get(url, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=15),
                                ssl=_SSL_CTX) as r:
            ms = (time.time() - t0) * 1000
            html = await r.text() if r.status == 200 else ""
            ok = r.status == 200 and "/players/" in html
            n_players = html.count("/players/") if ok else 0
            print(_result_line(f"tennisstats.com {tour.upper()} rankings", r.status, ms,
                               f"~{n_players} player links", ok))
            return ok
    except Exception as exc:
        ms = (time.time() - t0) * 1000
        print(f"{FAIL}  [ERR]  tennisstats.com {tour.upper()} rankings  ({ms:.0f}ms)  — {exc}")
        return False


async def test_tennisstats_profile(session: aiohttp.ClientSession):
    print("\n[8] tennisstats.com — player profile (carlos-alcaraz)")
    url = "https://tennisstats.com/players/carlos-alcaraz"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    t0 = time.time()
    try:
        async with session.get(url, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=15),
                                ssl=_SSL_CTX) as r:
            ms = (time.time() - t0) * 1000
            html = await r.text() if r.status == 200 else ""
            ok = r.status == 200 and "Alcaraz" in html
            rank_present = "ATP Rank" in html or "ranking" in html.lower()
            print(_result_line("tennisstats.com profile", r.status, ms,
                               "rank found" if rank_present else "rank NOT found in HTML", ok))
            return ok
    except Exception as exc:
        ms = (time.time() - t0) * 1000
        print(f"{FAIL}  [ERR]  tennisstats.com profile  ({ms:.0f}ms)  — {exc}")
        return False


async def test_atp_ajax(session: aiohttp.ClientSession):
    print("\n[9] ATP Tour — GetInitialScores AJAX")
    url = "https://www.atptour.com/-/ajax/Scores/GetInitialScores"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.atptour.com/en/scores/",
    }
    status, body, ms = await _get_json(session, url, headers)
    if status == 200 and isinstance(body, dict):
        tournaments = ((body.get("liveScores") or body).get("Tournaments") or [])
        n = sum(len(t.get("Matches", [])) for t in tournaments)
        print(_result_line("ATP GetInitialScores", status, ms, f"{n} live match(es)", True))
        _vprint("atp ajax", body)
        return True
    elif status in (403, 429):
        print(f"{WARN}  [{status}]  ATP GetInitialScores  ({ms:.0f}ms)  "
              "— Cloudflare blocked (expected without cloudscraper)")
        return True   # not a bot failure — cloudscraper handles this in production
    else:
        print(_result_line("ATP GetInitialScores", status, ms, "unexpected response", False))
        return False


async def test_flashscore(session: aiohttp.ClientSession):
    print("\n[13] Flashscore — discovery feed (expect 401 — graceful-degradation check)")
    url = "https://www.flashscore.com/x/feed/f_1_0_3_en_1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-fsign": "SW9D1eY",
    }
    status, _, ms = await _get_json(session, url, headers)
    if status in (401, 403):
        print(f"{PASS}  [{status}]  Flashscore discovery  ({ms:.0f}ms)  "
              "— blocked as expected; 30-min backoff will suppress retries")
        return True
    elif status == 200:
        print(f"{PASS}  [200]  Flashscore discovery  ({ms:.0f}ms)  — unexpectedly accessible")
        return True
    else:
        print(_result_line("Flashscore discovery", status, ms,
                           "unexpected status", False))
        return False


async def test_sportradar_rest(session: aiohttp.ClientSession, api_key: str):
    print("\n[10] SportRadar REST — live schedule")
    if not api_key:
        print(f"{SKIP}  SportRadar REST — no API key (add 'sportradar_api_key' to kalshi_keys.json)")
        return None

    url = f"https://api.sportradar.com/tennis/trial/v3/en/schedules/live/schedule.json?api_key={api_key}"
    status, body, ms = await _get_json(session, url, timeout=15.0)
    if status == 200:
        n = len((body or {}).get("sport_events") or [])
        print(_result_line("SportRadar live schedule", status, ms, f"{n} live event(s)", True))
        _vprint("sportradar live schedule", body)
        return True
    elif status == 403:
        print(f"{FAIL}  [403]  SportRadar live schedule  ({ms:.0f}ms)  "
              "— invalid API key or quota exhausted")
        return False
    elif status == 401:
        print(f"{FAIL}  [401]  SportRadar live schedule  ({ms:.0f}ms)  "
              "— API key missing or malformed")
        return False
    else:
        print(_result_line("SportRadar live schedule", status, ms, "", status == 200))
        return status == 200


async def test_sportradar_daily(session: aiohttp.ClientSession, api_key: str):
    print("\n[11] SportRadar REST — daily schedule")
    if not api_key:
        print(f"{SKIP}  SportRadar daily schedule — no API key")
        return None

    today = date.today().strftime("%Y-%m-%d")
    url = (f"https://api.sportradar.com/tennis/trial/v3/en/schedules/{today}"
           f"/schedule.json?api_key={api_key}")
    status, body, ms = await _get_json(session, url, timeout=15.0)
    ok = status == 200
    if ok:
        n = len((body or {}).get("sport_events") or [])
        print(_result_line("SportRadar daily schedule", status, ms, f"{n} event(s) today", True))
        _vprint("sportradar daily", body)
    elif status == 403:
        print(f"{FAIL}  [403]  SportRadar daily schedule  ({ms:.0f}ms)  — quota or key issue")
    else:
        print(_result_line("SportRadar daily schedule", status, ms, "", ok))
    return ok


async def test_sportradar_ws(api_key: str, wait_secs: int = 15):
    print(f"\n[12] SportRadar WebSocket — push feed (listening {wait_secs}s)")
    if not api_key:
        print(f"{SKIP}  SportRadar WebSocket — no API key")
        return None

    url = f"wss://api.sportradar.com/tennis/trial/v3/en/stream/events/subscribe?api_key={api_key}"
    connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
    msgs_received = 0
    error: Optional[str] = None
    t0 = time.time()

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.ws_connect(
                url,
                heartbeat=20.0,
                timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
            ) as ws:
                print(f"      Connected. Listening for {wait_secs}s …")
                deadline = time.time() + wait_secs
                async for msg in ws:
                    if time.time() > deadline:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        msgs_received += 1
                        if _VERBOSE:
                            try:
                                parsed = json.loads(msg.data)
                                _vprint(f"WS msg #{msgs_received}", parsed)
                            except Exception:
                                print(f"      raw: {msg.data[:200]}")
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        break
    except aiohttp.WSServerHandshakeError as exc:
        error = f"Handshake error {exc.status}"
        if exc.status == 401:
            error = "401 Unauthorized — check API key"
        elif exc.status == 403:
            error = "403 Forbidden — trial quota may be exhausted"
    except Exception as exc:
        error = str(exc)

    ms = (time.time() - t0) * 1000
    if error:
        print(f"{FAIL}  [WS]  SportRadar WebSocket  ({ms:.0f}ms)  — {error}")
        return False
    else:
        print(_result_line("SportRadar WebSocket", "WS", ms,
                           f"{msgs_received} message(s) in {wait_secs}s", True))
        return True


# ── main ──────────────────────────────────────────────────────────────────────

async def main(test_ws: bool = False, ws_wait: int = 15):
    global _VERBOSE

    sr_key = _load_key("sportradar_api_key")

    connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = {}

        results["livescore_live"]    = await test_livescore_live(session)
        results["livescore_daily"]   = await test_livescore_daily(session)
        results["espn_atp"]          = await test_espn_atp(session)
        results["espn_wta"]          = await test_espn_wta(session)
        results["sofascore"]         = await test_sofascore(session)
        results["tennisstats_atp"]   = await test_tennisstats_rankings(session, "atp")
        results["tennisstats_wta"]   = await test_tennisstats_rankings(session, "wta")
        results["tennisstats_prof"]  = await test_tennisstats_profile(session)
        results["atp_ajax"]          = await test_atp_ajax(session)
        results["sr_live"]           = await test_sportradar_rest(session, sr_key)
        results["sr_daily"]          = await test_sportradar_daily(session, sr_key)
        results["flashscore"]        = await test_flashscore(session)

    if test_ws:
        results["sr_ws"] = await test_sportradar_ws(sr_key, wait_secs=ws_wait)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ENDPOINT SUMMARY")
    print("=" * 62)

    passed = skipped = failed = 0
    for name, r in results.items():
        if r is None:
            skipped += 1
        elif r:
            passed += 1
        else:
            failed += 1

    critical = ["livescore_live", "espn_atp", "sofascore", "tennisstats_atp"]
    critical_ok = all(results.get(k) for k in critical)

    print(f"  Passed : {passed}")
    print(f"  Skipped: {skipped}  (API key not configured)")
    print(f"  Failed : {failed}")
    print()
    if critical_ok:
        print("  \033[32mCore live-score sources (LiveScore/ESPN/SofaScore) are UP.\033[0m")
        print("  Bot can trade even without SportRadar key.")
    else:
        failing = [k for k in critical if not results.get(k)]
        print(f"  \033[31mCRITICAL sources failing: {failing}\033[0m")
        print("  Fix these before running the bot live.")

    if results.get("sr_live") is False:
        print("\n  \033[33mSportRadar key issue detected.\033[0m")
        print("  Add your key to kalshi_keys.json:")
        print('    "sportradar_api_key": "YOUR_KEY_FROM_DEVELOPER.SPORTRADAR.COM"')
    elif results.get("sr_live") is None:
        print("\n  \033[90mSportRadar not configured — WebSocket feed disabled.\033[0m")
        print("  Free trial: https://developer.sportradar.com  (no credit card)")
        print('  Add key to kalshi_keys.json: "sportradar_api_key": "YOUR_KEY"')

    print("=" * 62 + "\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TennisBot77 endpoint connectivity test")
    parser.add_argument("--sportradar-ws", action="store_true",
                        help="Also open the SportRadar WebSocket and listen for messages")
    parser.add_argument("--ws-wait", type=int, default=15,
                        help="Seconds to listen on the WebSocket (default: 15)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print truncated response payloads")
    args = parser.parse_args()
    _VERBOSE = args.verbose

    exit_code = asyncio.run(main(test_ws=args.sportradar_ws, ws_wait=args.ws_wait))
    sys.exit(exit_code)
