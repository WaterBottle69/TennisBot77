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
from ml_engine import ml_engine
from market_monitor import MarketMonitor
from adaptive_controller import AdaptiveController

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KALSHI_MATCH_STATE  = os.path.join(BASE_DIR, "kalshi_match_state.json")
LIVE_STATE_PATH     = os.path.join(BASE_DIR, "live_state.json")
TRADING_MODE_PATH   = os.path.join(BASE_DIR, "trading_mode.json")
ADAPTIVE_STATE_PATH = os.path.join(BASE_DIR, "adaptive_state.json")
BOT_LOG_PATH        = os.path.join(BASE_DIR, "bot.log")
MAX_CONCURRENT_MATCHES = 3

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(module)s: %(message)s")

def _setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured (e.g. when imported by server.py)
    root.setLevel(logging.INFO)

    # File handler only — avoids double-write when stdout is also redirected to bot.log
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

async def process_match(
    match: dict,
    kalshi: KalshiClient,
    history: HistoricalAnalyzer,
    bets: BetManager,
    config: Config,
    adaptive: AdaptiveController = None,
):
    ticker = match["ticker"]
    player_a = match["player_a"]
    player_b = match["player_b"]

    mode = get_trading_mode()
    if mode == "hf":
        config.MIN_EDGE             = HF_MIN_EDGE
        config.KELLY_FRACTION       = HF_KELLY_FRACTION
        config.MAX_BET_USDC         = HF_MAX_BET_USDC
        config.EXTREME_ODDS_MIN     = HF_EXTREME_ODDS_MIN
        config.EXTREME_ODDS_MAX     = HF_EXTREME_ODDS_MAX
        config.MAX_MODEL_DIVERGENCE = HF_MAX_MODEL_DIVERGENCE
        poll_interval               = HF_POLL_INTERVAL
        log.info(f"[HF MODE] Edge={HF_MIN_EDGE*100:.1f}%  Odds=[{HF_EXTREME_ODDS_MIN},{HF_EXTREME_ODDS_MAX}]  Div={HF_MAX_MODEL_DIVERGENCE}")
    else:
        poll_interval = 2.0

    hf_cumulative_profit = 0.0
    log.info(f"--- Processing Match Live (Markov DP) [{mode.upper()}]: {player_a} vs {player_b} (Ticker: {ticker}) ---")

    h2h = await history.get_h2h_matchup(player_a, player_b)
    p1_stats = h2h.get("meta", {}).get("player_a", {})
    p2_stats = h2h.get("meta", {}).get("player_b", {})
    p1_stats["surface"] = "Hard"
    p1_stats["best_of"] = 3
    p2_stats["surface"] = "Hard"
    p2_stats["best_of"] = 3

    # Fetch recent match sequences for the Neural LSTM Engine
    seq_a = await history._scraper.fetch_recent_matches(p1_stats.get("slug", ""), n=10)
    seq_b = await history._scraper.fetch_recent_matches(p2_stats.get("slug", ""), n=10)
    
    # Run the live player stats through the Hybrid Neural-XGBoost Engine
    ml_res = ml_engine.predict_win_prob(p1_stats, p2_stats, seq1=seq_a, seq2=seq_b)
    base_p_a = ml_res["hybrid_prob"]
    nn_prob = ml_res["nn_prob"]
    xgb_prob = ml_res["xgb_prob"]
    
    log.info(f"[ML Prediction] Hybrid Probability {player_a}: {base_p_a:.1%} (Neural={nn_prob:.1%}, XGB={xgb_prob:.1%})")

    # Convert base probabilities to server/returner parameters for Markov engine
    # Use configured scaling factors so model reflects `Config.MARKOV_*_SCALE`.
    cfg = config
    p_serve = 0.65 + (base_p_a - 0.5) * cfg.MARKOV_SERVE_SCALE
    p_return = 0.35 + (base_p_a - 0.5) * cfg.MARKOV_RETURN_SCALE
    
    lms = LiveMatchState(p_serve, p_return)
    # Live hold-rate tracking for dynamic Markov parameter updates
    _serve_pts_won   = 0
    _serve_pts_total = 0
    _atp_tick        = 0    # counts live ticks; triggers ATP stat refresh every 30
    _live_hold_rate  = 0.5  # empirical server hold rate from ATP stats (default 50%)

    metrics_path = os.path.join(BASE_DIR, "latency_metrics.csv")
    try:
        with open(metrics_path, mode="a", newline="") as f:
            writer = csv.writer(f)

            async for score_update in poll_live_score_real(player_a, player_b, config, interval=poll_interval):
                pipeline_start = time.time()

                if score_update.get("is_live"):
                    lms.update(score_update)

                    # Track serve-point outcomes for live parameter updates.
                    # score_update["p1_serving"] tells us who was serving this point.
                    # Whoever won the point is inferred from the current vs. previous
                    # game score change direction (approximated per-tick as +1 serve win).
                    if score_update.get("p1_serving") is not None:
                        _serve_pts_total += 1
                        # A point in the current game where p1 is serving:
                        # if p1 won the point, it's a serve-side win.
                        pts = score_update.get("points", (0, 0))
                        if score_update["p1_serving"] and pts[0] > pts[1]:
                            _serve_pts_won += 1
                        elif not score_update["p1_serving"] and pts[1] > pts[0]:
                            _serve_pts_won += 1

                        if _serve_pts_total >= 10:
                            live_hold = _serve_pts_won / _serve_pts_total
                            new_p_serve  = 0.7 * lms.p_serve  + 0.3 * live_hold
                            new_p_return = 0.7 * lms.p_return + 0.3 * (1.0 - live_hold)
                            lms.update_params(new_p_serve, new_p_return)
                            log.info(
                                "Updating p_serve/p_return based on live hold rate: %.3f",
                                live_hold,
                            )

                    # ── Periodic ATP live stat refresh (every 30 live ticks) ──────
                    _atp_tick += 1
                    if _atp_tick % 30 == 0:
                        try:
                            atp = await history.get_live_atp_stats(player_a, player_b)
                            if atp:
                                raw_pct   = atp.get("first_serve_pct_a", 0.0)
                                pts_1st   = atp.get("pts_won_1st_serve_a", 0.0)
                                if raw_pct  > 1.0: raw_pct  /= 100.0
                                if pts_1st  > 1.0: pts_1st  /= 100.0
                                if raw_pct > 0 and pts_1st > 0:
                                    live_eff = raw_pct * pts_1st   # combined serve effectiveness
                                    if 0.1 < live_eff < 0.9:
                                        new_ps = 0.6 * lms.p_serve  + 0.4 * live_eff
                                        new_pr = 0.6 * lms.p_return + 0.4 * (1.0 - live_eff)
                                        lms.update_params(new_ps, new_pr)
                                        log.info(
                                            "[ATP] Markov refined: 1st_in=%.1f%% pts_won=%.1f%% p_serve→%.4f",
                                            raw_pct * 100, pts_1st * 100, new_ps,
                                        )
                                bp = atp.get("break_pts_converted_a", 0.0)
                                if bp > 1.0: bp /= 100.0
                                if 0 < bp < 1:
                                    _live_hold_rate = 1.0 - bp
                        except Exception as _ae:
                            log.debug("[ATP] Stat refresh error: %s", _ae)

                    win_prob_a = lms.win_probability()
                    status_str = f"L: {score_update['points'][0]}-{score_update['points'][1]}"
                    feed_status = "live"
                elif score_update.get("is_scheduled"):
                    # Match found in daily schedule but not yet live
                    win_prob_a = lms.win_probability()
                    status_str = "UPCOMING"
                    feed_status = "upcoming"
                else:
                    # Feed stale or missing — use last known probability (could be pre-game)
                    win_prob_a = lms.win_probability()
                    status_str = "STALE"
                    feed_status = "stale"

                win_prob = {"team_a": win_prob_a, "team_b": 1.0 - win_prob_a}

                if score_update.get("is_live") or score_update.get("misses", 1) % 10 == 0:
                    log_type = logging.INFO if score_update.get("is_live") or score_update.get("is_scheduled") else logging.WARNING
                    log.log(log_type, f"[{mode.upper()}][{status_str}] {player_a}:{win_prob['team_a']:.1%}  {player_b}:{win_prob['team_b']:.1%}")

                # ── Market price: use MarketMonitor cache when fresh, else re-fetch ──
                # MarketMonitor polls every 5 s. If its last update is within that
                # window we skip a full HTTP round-trip, saving ~100-300 ms of latency.
                _monitor = _market_monitors.get(ticker)
                # Feed latest Markov win probability so MarketMonitor can compute Z-score
                if _monitor:
                    _monitor.set_model_prob(win_prob_a)
                flow_sig = _monitor.current_signal if _monitor else None
                monitor_age = time.time() - (flow_sig.timestamp if flow_sig else 0)

                if flow_sig and monitor_age < 6.0:
                    # Build a market dict from the cached signal — no network call.
                    yes_p = flow_sig.yes_price
                    market = {
                        "id":        ticker,
                        "yes_price": yes_p,
                        "no_price":  max(0.01, min(0.99, 1.0 - yes_p)),
                        "active":    True,
                        "_raw":      {},
                        "player_a":  player_a,
                        "player_b":  player_b,
                    }
                    log.debug(f"[CACHE] Using MarketMonitor price {yes_p:.3f} (age {monitor_age:.1f}s)")
                else:
                    # Cache stale or monitor not running — fetch from Kalshi.
                    market = await kalshi.get_market(ticker)
                    market["player_a"] = player_a
                    market["player_b"] = player_b

                flow_data = {
                    "direction":  flow_sig.direction.value if flow_sig else "NEUTRAL",
                    "velocity":   round(flow_sig.price_velocity * 100, 4) if flow_sig else 0.0,
                    "vol_regime": flow_sig.vol_regime if flow_sig else "LOW",
                    "yes_price":  round(flow_sig.yes_price, 4) if flow_sig else 0.5,
                    "z_score":    round(flow_sig.z_score, 4) if flow_sig else 0.0,
                }

                with open(LIVE_STATE_PATH, "w") as lf:
                    json.dump({
                        "player_a": player_a, "player_b": player_b,
                        "elo_a": 1500, "elo_b": 1500,
                        "win_prob_a": win_prob["team_a"],
                        "win_prob_b": win_prob["team_b"],
                        "nn_prob": nn_prob,
                        "xgb_prob": xgb_prob,
                        "trading_mode": mode,
                        "feed_status": feed_status,
                        "flow": flow_data,
                        "last_update": time.time()
                    }, lf)

                eval_start = time.time()
                eval_latency = (eval_start - pipeline_start) * 1000

                # ── Fetch balance once; pass to both predictive limiter and evaluate ──
                available_balance = await kalshi.get_balance()
                kelly_mult = adaptive.kelly_multiplier if adaptive else 1.0

                # ── Predictive limit orders (placed BEFORE the next point) ────────────
                # Cancel any limits that are now stale (point already played / price moved).
                # Then place a fresh resting limit at the anticipated post-point price.
                if score_update.get("is_live"):
                    await bets.cancel_stale_limits(market["yes_price"])
                    await bets.place_predictive_limit_order(
                        market=market,
                        lms=lms,
                        available_balance=available_balance,
                        kelly_mult=kelly_mult,
                        adaptive=adaptive,
                    )

                try:
                    result = await bets.evaluate_and_act(
                        market=market,
                        win_prob=win_prob,
                        game_state=None,
                        event={"type": "point_won", "clock": status_str},
                        pipeline_start=pipeline_start,
                        market_monitor=_market_monitors.get(ticker),
                        adaptive=adaptive,
                        available_balance=available_balance,
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
                            config.MIN_EDGE       = 0.015
                            config.KELLY_FRACTION = 0.25

                    # Persist adaptive controller state for the dashboard.
                    if adaptive:
                        try:
                            with open(ADAPTIVE_STATE_PATH, "w") as af:
                                json.dump(adaptive.status_dict(), af)
                        except Exception:
                            pass
                except Exception as e:
                    log.error(f"Trading evaluation failed: {e}")

                # ── Break-point state-aware fractional divestment ────────────────
                # Detects high-leverage game states (break points / deuce) and
                # triggers half take-profit BEFORE the critical point is played.
                if score_update.get("is_live") and lms:
                    try:
                        await bets.check_breakpoint_exits(
                            market=market,
                            lms=lms,
                            live_hold_rate=_live_hold_rate,
                        )
                    except Exception as _bpe:
                        log.debug("[BREAK-PT] check_breakpoint_exits error: %s", _bpe)

                writer.writerow(["point_won", status_str, win_prob["team_a"], win_prob["team_b"], eval_latency, hf_cumulative_profit])

                # Termination condition: market closed/settled
                # Re-fetch market status to check if settled/closed
                if not market.get("active", True):
                    log.info(f"Market {ticker} is no longer active — session complete.")
                    break

    except asyncio.CancelledError:
        log.info(f"Session cancelled for {ticker}.")
    finally:
        log.info(f"Session ended for {ticker}. Final positions:")
        bets.print_summary()


async def worker(
    queue: asyncio.Queue,
    kalshi: KalshiClient,
    history: HistoricalAnalyzer,
    bets: BetManager,
    config: Config,
    adaptive: AdaptiveController,
):
    """Worker task to process matches up to the concurrency limit."""
    while True:
        match = await queue.get()
        ticker = match["ticker"]
        # Start a MarketMonitor for this match
        monitor = MarketMonitor(kalshi, ticker)
        _market_monitors[ticker] = monitor
        monitor_task = asyncio.create_task(monitor.run())

        # ── Phase 1: WebSocket streams (real-time data, replaces 5s polling) ──
        async def _orderbook_cb(msg, _mon=monitor):
            """Push WebSocket orderbook delta directly into MonitorMonitor."""
            try:
                data = msg.get("msg") or {}
                price_raw = data.get("yes") or data.get("yes_price") or data.get("price")
                if price_raw is not None:
                    yes_p = max(0.01, min(0.99, int(price_raw) / 100.0))
                    _mon._price_history.append((time.time(), yes_p))
            except Exception:
                pass

        ws_ob_task    = asyncio.create_task(kalshi.stream_orderbook(ticker, _orderbook_cb))
        ws_fills_task = asyncio.create_task(kalshi.stream_user_fills(lambda _msg: None))

        try:
            await process_match(match, kalshi, history, bets, config, adaptive=adaptive)
        except Exception as e:
            log.error(f"Error processing match {ticker}: {e}")
        finally:
            monitor.stop()
            monitor_task.cancel()
            ws_ob_task.cancel()
            ws_fills_task.cancel()
            _market_monitors.pop(ticker, None)
            queue.task_done()

async def run_game_session():
    """
    Full lifecycle:
    1. Fetch dynamic ATP markets from Kalshi
    2. Load them into a queue (matches to watch)
    3. Process using a capped pool of parallel worker tasks
    """
    global _market_monitors
    _market_monitors = {}   # ticker -> MarketMonitor

    config = Config()
    kalshi = KalshiClient(config)
    history = HistoricalAnalyzer(config)
    bets = BetManager(kalshi, config)

    # Shared adaptive controller — survives across all matches this session
    adaptive = AdaptiveController(base_min_edge=max(config.MIN_EDGE, 0.02))
    log.info(f"[ADAPTIVE] Controller started. Base MIN_EDGE={adaptive.base_min_edge:.4f}")

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
            task = asyncio.create_task(
                worker(queue, kalshi, history, bets, config, adaptive)
            )
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
