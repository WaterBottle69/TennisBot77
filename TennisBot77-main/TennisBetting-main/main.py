"""
Polymarket Sports Betting Bot - Main Orchestrator
Integrates: Kalshi, ATP stats (tennisstats scraper), Markov chain engine, parallel match tracking limit.
"""

import os
import asyncio
import logging
import time
import csv
import json
import random
import aiohttp
from config import Config
from elo_engine import EloEngine
from markov_engine import LiveMatchState
from kalshi_client import KalshiClient
from historical_analyzer import HistoricalAnalyzer
from bet_manager import BetManager
from live_score_scraper import poll_live_score_real

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KALSHI_MATCH_STATE = os.path.join(BASE_DIR, "kalshi_match_state.json")
LIVE_STATE_PATH    = os.path.join(BASE_DIR, "live_state.json")
TRADING_MODE_PATH  = os.path.join(BASE_DIR, "trading_mode.json")
BOT_LOG_PATH       = os.path.join(BASE_DIR, "bot.log")
MAX_CONCURRENT_MATCHES = 3

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(module)s: %(message)s")

def _setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured (e.g. when imported by server.py)
    root.setLevel(logging.INFO)

    # Always write to console
    ch = logging.StreamHandler()
    ch.setFormatter(_fmt)
    root.addHandler(ch)

    # Always write to bot.log so the dashboard can tail it regardless of
    # whether main.py is launched directly or via server.py's subprocess.
    fh = logging.FileHandler(BOT_LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(_fmt)
    root.addHandler(fh)

_setup_logging()
log = logging.getLogger(__name__)

# ── High-Frequency mode parameters ────────────────────────────────────────────
HF_MIN_EDGE       = 0.003   # 0.3% edge threshold (much lower than normal 1%)
HF_KELLY_FRACTION = 0.05    # very small Kelly fraction — 5%
HF_MAX_BET_USDC   = 15.0    # small position size per trade
HF_POLL_INTERVAL  = 0.8     # seconds between score polls (vs 2.0 normal)
HF_EXTREME_ODDS_MIN = 0.01  # allow trading on almost any price in HF
HF_EXTREME_ODDS_MAX = 0.99
HF_MAX_MODEL_DIVERGENCE = 0.99 # essentially disable divergence guard in HF
HF_PROFIT_THRESHOLD = 50.0  # auto-switch back to normal after $50 profit

def get_trading_mode() -> str:
    if os.path.exists(TRADING_MODE_PATH):
        try:
            with open(TRADING_MODE_PATH, "r") as f:
                return json.load(f).get("mode", "normal")
        except Exception:
            pass
    return "normal"


def _write_kalshi_match_state(matches: list) -> None:
    """Dashboard reads this to sync player names with the active Kalshi market."""
    state: dict = {"ok": bool(matches), "matches": []}
    if matches:
        for m in matches[:12]:
            state["matches"].append(
                {
                    "player_a": m["player_a"],
                    "player_b": m["player_b"],
                    "ticker": m["ticker"],
                    "title": m.get("title", ""),
                    "event_ticker": m.get("event_ticker", ""),
                    "series_ticker": m.get("series_ticker", ""),
                }
            )
        m0 = matches[0]
        state["player_a"] = m0["player_a"]
        state["player_b"] = m0["player_b"]
        state["ticker"] = m0["ticker"]
        state["title"] = m0.get("title", "")
        state["event_ticker"] = m0.get("event_ticker", "")
    else:
        state["message"] = (
            "No open Kalshi matchup found after scanning events. "
            "Enter two player names manually and run the simulation."
        )
    with open(KALSHI_MATCH_STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


async def poll_live_score(player_a: str, player_b: str, interval: float = 2.0):
    """
    Simulates real live score scraping by polling an API.
    In production, this would hit the official ATP or an API provider.
    Yields (p1_pts, p2_pts) updates over time.
    """
    # Mocking real live-score updates
    p1_pts, p2_pts = 0, 0
    p1_serving = True
    while True:
        await asyncio.sleep(interval)  # Polling interval
        # Simulate a point won
        if random.random() < 0.55:
            p1_pts += 15 if p1_pts < 30 else 10
        else:
            p2_pts += 15 if p2_pts < 30 else 10
            
        yield {"points": (p1_pts, p2_pts), "p1_serving": p1_serving}
        
        # Super simplified game resolution for the mock
        if p1_pts >= 40 or p2_pts >= 40:
            p1_pts, p2_pts = 0, 0
            p1_serving = not p1_serving

async def process_match(match: dict, kalshi: KalshiClient, history: HistoricalAnalyzer, bets: BetManager, config: Config):
    import copy
    ticker = match["ticker"]
    player_a = match["player_a"]
    player_b = match["player_b"]

    # BUG FIX: Each worker gets its own Config copy so concurrent workers
    # don't mutate the shared Config object and race each other.
    cfg = copy.copy(config)
    mode = get_trading_mode()
    if mode == "hf":
        cfg.MIN_EDGE             = HF_MIN_EDGE
        cfg.KELLY_FRACTION       = HF_KELLY_FRACTION
        cfg.MAX_BET_USDC         = HF_MAX_BET_USDC
        cfg.EXTREME_ODDS_MIN     = HF_EXTREME_ODDS_MIN
        cfg.EXTREME_ODDS_MAX     = HF_EXTREME_ODDS_MAX
        cfg.MAX_MODEL_DIVERGENCE = HF_MAX_MODEL_DIVERGENCE
        poll_interval            = HF_POLL_INTERVAL
        log.info(f"[HF MODE] Edge={HF_MIN_EDGE*100:.1f}%  Odds=[{HF_EXTREME_ODDS_MIN},{HF_EXTREME_ODDS_MAX}]  Div={HF_MAX_MODEL_DIVERGENCE}")
    else:
        poll_interval = 2.0

    # Give the BetManager this match's own config so HF settings are isolated
    bets.cfg = cfg

    hf_cumulative_profit = 0.0
    log.info(f"--- Processing Match Live (Markov DP) [{mode.upper()}]: {player_a} vs {player_b} (Ticker: {ticker}) ---")

    h2h = await history.get_h2h_matchup(player_a, player_b)
    base_p_a = h2h.get("team_a_win_rate", 0.5)

    # Convert base probabilities to server/returner parameters for Markov engine
    # Use configured scaling factors so model reflects `Config.MARKOV_*_SCALE`.
    cfg = config
    p_serve = 0.65 + (base_p_a - 0.5) * cfg.MARKOV_SERVE_SCALE
    p_return = 0.35 + (base_p_a - 0.5) * cfg.MARKOV_RETURN_SCALE
    
    lms = LiveMatchState(p_serve, p_return)

    metrics_path = os.path.join(BASE_DIR, "latency_metrics.csv")
    try:
        with open(metrics_path, mode="a", newline="") as f:
            writer = csv.writer(f)

            async for score_update in poll_live_score_real(player_a, player_b, config, interval=poll_interval):
                pipeline_start = time.time()

                if score_update.get("is_live"):
                    lms.update(score_update)
                    win_prob_a = lms.win_probability()
                    status_str = f"L: {score_update['points'][0]}-{score_update['points'][1]}"
                else:
                    # Feed stale or missing — use last known probability (could be pre-game)
                    win_prob_a = lms.win_probability()
                    status_str = "STALE"

                win_prob = {"team_a": win_prob_a, "team_b": 1.0 - win_prob_a}

                if score_update.get("is_live") or score_update.get("misses", 1) % 10 == 0:
                    log.info(f"[{mode.upper()}][{status_str}] {player_a}:{win_prob['team_a']:.1%}  {player_b}:{win_prob['team_b']:.1%}")

                with open(LIVE_STATE_PATH, "w") as lf:
                    json.dump({
                        "player_a": player_a, "player_b": player_b,
                        "elo_a": 1500, "elo_b": 1500,
                        "win_prob_a": win_prob["team_a"],
                        "win_prob_b": win_prob["team_b"],
                        "trading_mode": mode,
                        "feed_status": "live" if score_update.get("is_live") else "stale"
                    }, lf)

                market = await kalshi.get_market(ticker)
                eval_start = time.time()
                eval_latency = (eval_start - pipeline_start) * 1000

                try:
                    result = await bets.evaluate_and_act(
                        market=market,
                        win_prob=win_prob,
                        game_state=None,
                        event={"type": "point_won", "clock": status_str},
                        pipeline_start=pipeline_start,
                    )
                    # Track HF profit and auto-exit when threshold is hit
                    if mode == "hf" and isinstance(result, dict):
                        hf_cumulative_profit += float(result.get("profit", 0.0))
                        if hf_cumulative_profit >= HF_PROFIT_THRESHOLD:
                            log.info(
                                f"[HF MODE] Profit threshold ${HF_PROFIT_THRESHOLD} reached "
                                f"(${hf_cumulative_profit:.2f}). Switching to NORMAL mode."
                            )
                            with open(TRADING_MODE_PATH, "w") as mf:
                                json.dump({"mode": "normal"}, mf)
                            mode = "normal"
                            config.MIN_EDGE       = 0.01
                            config.KELLY_FRACTION = 0.25
                except Exception as e:
                    log.error(f"Trading evaluation failed: {e}")

                writer.writerow(["point_won", status_str, win_prob["team_a"], win_prob["team_b"], eval_latency, hf_cumulative_profit])

                # Termination condition: market closed or random conclusion
                if random.random() < 0.05:
                    log.info(f"Match {ticker} concluded or market closed.")
                    break

    except asyncio.CancelledError:
        log.info(f"Session cancelled for {ticker}.")
    finally:
        log.info(f"Session ended for {ticker}. Final positions:")
        bets.print_summary()


async def worker(queue: asyncio.Queue, kalshi: KalshiClient, history: HistoricalAnalyzer, bets: BetManager, config: Config):
    """Worker task to process matches up to the concurrency limit."""
    while True:
        match = await queue.get()
        try:
            await process_match(match, kalshi, history, bets, config)
        except Exception as e:
            log.error(f"Error processing match {match['ticker']}: {e}")
        finally:
            queue.task_done()

async def run_game_session():
    """
    Full lifecycle:
    1. Fetch dynamic ATP markets from Kalshi
    2. Load them into a queue (matches to watch)
    3. Process using a capped pool of parallel worker tasks
    """
    config = Config()
    kalshi = KalshiClient(config)
    history = HistoricalAnalyzer(config)
    bets = BetManager(kalshi, config)

    try:
        log.info("Fetching open matchup markets from Kalshi...")
        atp_matches = await kalshi.get_atp_markets()
        _write_kalshi_match_state(atp_matches)
        
        if not atp_matches:
            log.warning("No open tennis Kalshi markets found — bot idle.")
            return

        # Queue for upcoming matches to process
        queue = asyncio.Queue()
        for match in atp_matches:
            queue.put_nowait(match)

        # Create workers to cap parallel tasks
        workers = []
        for _ in range(MAX_CONCURRENT_MATCHES):
            task = asyncio.create_task(worker(queue, kalshi, history, bets, config))
            workers.append(task)
            
        log.info(f"Started parallel processing. Maximum active live matches: {MAX_CONCURRENT_MATCHES}")

        # Wait until all queued matches have been processed
        await queue.join()

        # Cancel our workers
        for w in workers:
            w.cancel()

    finally:
        await kalshi.close()


if __name__ == "__main__":
    asyncio.run(run_game_session())
