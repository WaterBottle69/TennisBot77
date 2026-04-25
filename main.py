"""
Polymarket Sports Betting Bot - Main Orchestrator
Integrates: Kalshi, ATP stats (tennisstats scraper), Markov chain engine, parallel match tracking limit.
"""

import os
import asyncio
import logging
import logging.handlers
import time
import csv
import json
import datetime
from config import Config
from markov_engine import NonIIDLiveMatchState
from bayesian_updater import BayesianServeProbUpdater, GaussianSkillDrift
from kalshi_client import KalshiClient
from historical_analyzer import HistoricalAnalyzer
from bet_manager import BetManager
from live_score_scraper import poll_live_score_real, LiveScoreScraper
from ml_engine import ml_engine
from player_serve_cache import get_serve_stats, load_cache as _load_serve_cache
from market_monitor import MarketMonitor
from adaptive_controller import AdaptiveController
from location_engine import LocationEngine
from trade_tracker import TradeTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KALSHI_MATCH_STATE  = os.path.join(BASE_DIR, "kalshi_match_state.json")
LIVE_STATE_PATH     = os.path.join(BASE_DIR, "live_state.json")
TRADING_MODE_PATH   = os.path.join(BASE_DIR, "trading_mode.json")
ADAPTIVE_STATE_PATH = os.path.join(BASE_DIR, "adaptive_state.json")
BOT_LOG_PATH        = os.path.join(BASE_DIR, "bot.log")
MAX_CONCURRENT_MATCHES = 3
_market_monitors: dict = {}   # ticker → MarketMonitor; populated per session

# How many consecutive "upcoming" (not-yet-live) polls before a worker releases
# its slot so a live match can take it.  At 10 s/poll this is ~10 min.
_MAX_UPCOMING_TICKS = 60
# How often (seconds) the background task re-scans Kalshi for new markets.
_DISCOVERY_INTERVAL = 60

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(module)s: %(message)s")

def _setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured (e.g. when imported by server.py)
    root.setLevel(logging.INFO)

    # Rotating file handler: 50 MB per file, keep 10 backups (~500 MB max)
    fh = logging.handlers.RotatingFileHandler(
        BOT_LOG_PATH, mode="a", maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
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


def _write_live_state(path, player_a, player_b, p1_stats, p2_stats,
                      win_prob, nn_prob, xgb_prob, mode, feed_status, flow_data,
                      surface=None, score=None, markov_prob=None):
    try:
        payload = json.dumps({
            "player_a":    player_a,
            "player_b":    player_b,
            "elo_a":       p1_stats.get("elo", 1500),
            "elo_b":       p2_stats.get("elo", 1500),
            "win_prob_a":  win_prob["team_a"],
            "win_prob_b":  win_prob["team_b"],
            "nn_prob":     nn_prob,
            "xgb_prob":    xgb_prob,
            "markov_prob": markov_prob,
            "trading_mode": mode,
            "feed_status": feed_status,
            "flow":        flow_data,
            "surface":     surface or p1_stats.get("surface", "Hard"),
            "score":       score,
            "yes_price":   flow_data.get("yes_price") if flow_data else None,
            "last_update": time.time(),
        })
        # Atomic write: write to temp then rename to avoid partial reads by server
        tmp = path + ".tmp"
        with open(tmp, "w") as lf:
            lf.write(payload)
        os.replace(tmp, path)
    except Exception as _e:
        pass


async def process_match(
    match: dict,
    kalshi: KalshiClient,
    history: HistoricalAnalyzer,
    bets: BetManager,
    config: Config,
    adaptive: AdaptiveController = None,
    location_engine: LocationEngine = None,
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
    _upcoming_ticks = 0   # counts consecutive non-live polls
    log.info(f"--- Processing Match Live (Markov DP) [{mode.upper()}]: {player_a} vs {player_b} (Ticker: {ticker}) ---")

    h2h = await history.get_h2h_matchup(player_a, player_b)
    p1_stats = h2h.get("meta", {}).get("player_a", {})
    p2_stats = h2h.get("meta", {}).get("player_b", {})

    # ── Stats quality gate ───────────────────────────────────────────────────
    # Refuse to trade if either player's profile could not be loaded.
    # A missing ELO or ranking means the model is working blind and its
    # probability estimates are unreliable — better to skip than to lose.
    def _stats_ok(stats: dict, label: str) -> bool:
        slug = stats.get("slug", "")
        elo  = stats.get("elo",  0)
        rank = stats.get("ranking", 0)
        # Reject if: no slug, slug looks like a Kalshi title fragment ("set-2"),
        # or both elo and ranking are at their defaults (0 / None).
        if not slug:
            log.warning("[STATS-GATE] %s: no slug — aborting match.", label)
            return False
        bad_slug_tokens = {"set", "winner", "game", "match", "final", "semifinal"}
        if any(tok in slug.split("-") for tok in bad_slug_tokens):
            log.warning("[STATS-GATE] %s: slug '%s' looks like a title fragment — aborting.", label, slug)
            return False
        if not elo and not rank:
            log.warning("[STATS-GATE] %s: no ELO and no ranking — aborting match.", label)
            return False
        return True

    if not _stats_ok(p1_stats, player_a) or not _stats_ok(p2_stats, player_b):
        log.error(
            "[STATS-GATE] Insufficient player data for %s vs %s — "
            "trade BLOCKED. Fix: check player name parsing in kalshi_client.",
            player_a, player_b,
        )
        return

    # BUG FIX: Infer surface from tournament name BEFORE ML inference.
    # Previously hardcoded to 'Hard', causing wrong XGBoost surface features
    # for all Clay and Grass tournaments.
    def _infer_surface(tournament_name: str) -> str:
        """Infer court surface from tournament name. Defaults to Hard."""
        t = (tournament_name or '').lower()
        _clay_keywords = [
            'roland garros', 'french open', 'clay', 'monte carlo', 'barcelona',
            'madrid', 'rome', 'hamburg', 'bastad', 'gstaad', 'umag', 'kitzbuhel',
            'buenos aires', 'estoril', 'marrakech', 'houston', 'sao paulo',
        ]
        _grass_keywords = [
            'wimbledon', 'grass', "queen's", 'halle', 'eastbourne',
            'hertogenbosch', "'s-hertogenbosch", 'nottingham', 'newport',
            'birmingham', 'mallorca',
        ]
        if any(k in t for k in _clay_keywords):
            return 'Clay'
        if any(k in t for k in _grass_keywords):
            return 'Grass'
        return 'Hard'

    # Use the tournament from initial pregame matchup or a placeholder;
    # it will be refined when the first live score tick arrives.
    _initial_tournament = p1_stats.get('tournament', '') or p2_stats.get('tournament', '')
    _resolved_surface   = _infer_surface(_initial_tournament)
    p1_stats["surface"]     = _resolved_surface
    p1_stats["best_of"]     = 3
    p2_stats["surface"]     = _resolved_surface
    p2_stats["best_of"]     = 3
    # Expose ranking_pts explicitly so ml_engine reads the correct key
    # (the 'elo' field from tennisstats.com actually contains ATP ranking points).
    p1_stats["ranking_pts"] = p1_stats.get("elo", 1000)
    p2_stats["ranking_pts"] = p2_stats.get("elo", 1000)
    log.info(
        "[SURFACE] %s vs %s: surface=%s (from tournament='%s')",
        player_a, player_b, _resolved_surface, _initial_tournament,
    )

    # Fetch recent match sequences for the Neural LSTM Engine
    seq_a = await history._scraper.fetch_recent_matches(p1_stats.get("slug", ""), n=10)
    seq_b = await history._scraper.fetch_recent_matches(p2_stats.get("slug", ""), n=10)
    
    # Run the live player stats through the Hybrid Neural-XGBoost Engine
    ml_res = ml_engine.predict_win_prob(p1_stats, p2_stats, seq1=seq_a, seq2=seq_b)
    base_p_a = ml_res["hybrid_prob"]
    nn_prob = ml_res["nn_prob"]
    xgb_prob = ml_res["xgb_prob"]

    # Ablation test: log all model variants so we can compare them per match
    ml_engine.run_ablation(p1_stats, p2_stats, seq1=seq_a, seq2=seq_b)

    log.info(f"[ML Prediction] Hybrid Probability {player_a}: {base_p_a:.1%} (Neural={nn_prob:.1%}, XGB={xgb_prob:.1%})")

    # Cache player ages for age×temperature adjustment (applied once venue+weather resolve)
    _p1_age = float(p1_stats.get("age", 25.0))
    _p2_age = float(p2_stats.get("age", 25.0))
    # Cache physical/structural attributes for one-shot edge adjustments
    _p1_ht  = float(p1_stats.get("height_cm", 185.0))
    _p2_ht  = float(p2_stats.get("height_cm", 185.0))
    _p1_hand = str(p1_stats.get("hand", "R")).upper()
    _p2_hand = str(p2_stats.get("hand", "R")).upper()
    _p1_pts  = float(p1_stats.get("elo", 0) or 0)   # tennisstats "elo" field = ATP ranking pts
    _p2_pts  = float(p2_stats.get("elo", 0) or 0)

    # ── Serve-quality edge (pre-match, applied once) ──────────────────────────
    # WFO-validated: 2nd-serve won % diff (ROI +2.24%, p=0.000) and
    # BP save rate diff (ROI +2.86%, p=0.000) over 3.4M ATP matches 2010-2024.
    _sq_logit_adj = 0.0
    if config.SERVE_QUALITY_ENABLED:
        try:
            import math as _math
            _sq1 = get_serve_stats(player_a)
            _sq2 = get_serve_stats(player_b)
            _s2_diff  = _sq1["second_serve_won_pct"] - _sq2["second_serve_won_pct"]
            _bps_diff = _sq1["bp_save_rate"]         - _sq2["bp_save_rate"]
            _sq_logit_adj = (
                config.SERVE2_COEF  * _s2_diff
                + config.BP_SAVE_COEF * _bps_diff
            )
            _sq_max = _math.log(
                (0.5 + config.SERVE_QUALITY_MAX_ADJ) / (0.5 - config.SERVE_QUALITY_MAX_ADJ)
            )
            _sq_logit_adj = max(-_sq_max, min(_sq_max, _sq_logit_adj))
            _logit_base = _math.log(base_p_a / (1.0 - base_p_a))
            base_p_a = 1.0 / (1.0 + _math.exp(-(_logit_base + _sq_logit_adj)))
            log.info(
                "[SERVE-Q] %s vs %s | 2nd_won_diff=%+.3f bp_save_diff=%+.3f "
                "logit_adj=%+.4f | base_p %.3f→%.3f",
                player_a, player_b, _s2_diff, _bps_diff,
                _sq_logit_adj, ml_res["hybrid_prob"], base_p_a,
            )
        except Exception as _sqe:
            log.warning("[SERVE-Q] adjustment failed: %s", _sqe)

    # Convert base probabilities to server/returner parameters for Markov engine
    # Use configured scaling factors so model reflects `Config.MARKOV_*_SCALE`.
    cfg = config
    p_serve = 0.65 + (base_p_a - 0.5) * cfg.MARKOV_SERVE_SCALE
    p_return = 0.35 + (base_p_a - 0.5) * cfg.MARKOV_RETURN_SCALE
    
    lms = NonIIDLiveMatchState(p_serve, p_return)

    # ── Location engine: fetch venue once per match (async, cached 24 h) ──────
    # tournament name comes from the live score scraper via score_update["tournament"].
    # We don't have it yet at this point so we resolve it lazily on the first live tick.
    _venue           = None   # VenueInfo — populated on first live score update
    _venue_resolved  = False  # True once lookup attempted (prevents repeated geocoding)
    _sun_data        = None   # SunData  — updated every live tick
    _age_temp_applied = False  # age×temperature adjustment applied once per match
    _phys_applied     = False  # altitude/handedness/rank-pts adjustment applied once per match
    # ── Serve-convergence gate (Path 1) ─────────────────────────────────────
    _serve_prior_initial   = p_serve   # baseline before any live Bayesian updates
    _serve_divergence      = 0.0       # new_p_serve - prior; positive = outperforming
    _pts_vs_rank_edge      = 0.0       # |pts_rank logit component| for convergence gate
    # ── Signal pipeline state (for TradeTracker) ────────────────────────────
    _lh_net_cached         = 0.0
    _logit_age_temp        = 0.0
    _base_p_after_age_temp = base_p_a
    _logit_phys_total      = 0.0
    _base_p_after_phys     = base_p_a
    _alt_x_ht_comp         = 0.0
    _alt_x_age_comp        = 0.0
    _lh_hard_comp          = 0.0
    _lh_clay_comp          = 0.0
    _pts_rank_comp         = 0.0
    _pts_vs_rank_raw       = 0.0
    _weather_data          = None

    # Bayesian live-state fusion: seed prior from scraped serve stats so we
    # converge in ~3 games instead of ~15.  Surface is initially "hard" (the
    # most common surface); it is re-seeded once the venue is resolved below.
    from prior_seeder import compute_serve_prior, reseed_for_surface
    _bayes_prior_mean, _bayes_prior_conc = compute_serve_prior(p1_stats, surface="hard")
    _bayes_serve = BayesianServeProbUpdater(
        prior_mean=_bayes_prior_mean, concentration=_bayes_prior_conc
    )
    _bayes_surface_reseeded = False   # set True once venue surface is known
    _skill_drift = GaussianSkillDrift(drift_std=0.5, concentration_decay=0.98)
    # Live hold-rate tracking for dynamic Markov parameter updates
    _serve_pts_won   = 0
    _serve_pts_total = 0
    _serve_games_completed = 0  # BUG FIX: game counter for logging (replaces per-game reset)
    _prev_pts        = (0, 0)   # previous tick's game-point tuple for delta detection
    _atp_tick        = 0    # counts live ticks; triggers ATP stat refresh every 30
    _live_hold_rate  = 0.5  # empirical server hold rate from ATP stats (default 50%)

    def _open_metrics_file():
        day = datetime.date.today().strftime("%Y%m%d")
        path = os.path.join(BASE_DIR, f"latency_metrics_{day}.csv")
        return open(path, mode="a", newline=""), day

    _metrics_f, _metrics_day = _open_metrics_file()
    try:
        writer = csv.writer(_metrics_f)

        async for score_update in poll_live_score_real(player_a, player_b, config, interval=poll_interval):
            # Rotate metrics file when the calendar day rolls over
            _today = datetime.date.today().strftime("%Y%m%d")
            if _today != _metrics_day:
                _metrics_f.close()
                _metrics_f, _metrics_day = _open_metrics_file()
                writer = csv.writer(_metrics_f)
                pipeline_start = time.time()

            # ── Upcoming-match guard ─────────────────────────────────────────
            # While the match hasn't started yet, skip all trading logic so
            # the worker slot stays available for truly live matches.
            # After _MAX_UPCOMING_TICKS polls we release the slot; the
            # background discovery loop will re-queue it when it goes live.
            if not score_update.get("is_live"):
                _upcoming_ticks += 1
                if score_update.get("is_scheduled"):
                    log.info(
                        "[UPCOMING] %s vs %s — waiting for match to start "
                        "(%d/%d polls)", player_a, player_b,
                        _upcoming_ticks, _MAX_UPCOMING_TICKS,
                    )
                if _upcoming_ticks >= _MAX_UPCOMING_TICKS:
                    log.warning(
                        "[%s] %s vs %s still not live after %d polls — "
                        "releasing worker slot. Will be re-queued when live.",
                        ticker, player_a, player_b, _upcoming_ticks,
                    )
                    return
                continue  # skip balance fetch / trading / CSV write
            else:
                _upcoming_ticks = 0  # reset counter once match goes live

                if score_update.get("is_live"):
                    lms.update(score_update)

                    # ── Location engine: resolve venue once, then compute sun each tick ──
                    if location_engine is not None:
                        if not _venue_resolved:
                            _venue_resolved = True
                            _tournament = score_update.get("tournament", "")
                            log.info("[LOCATION] First live tick — tournament=%r source=%s",
                                     _tournament, score_update.get("source", "?"))
                            if _tournament:
                                try:
                                    _venue = await location_engine.get_venue_info(_tournament)
                                except Exception as _le:
                                    log.warning("[LOCATION] Venue fetch error: %s", _le)
                        if _venue is not None:
                            try:
                                _sun_data = location_engine.get_sun_data(_venue, score_update)
                                log.debug("[SUN] az=%.0f° elev=%.1f° glare=%s | %s",
                                          _sun_data.azimuth_deg, _sun_data.elevation_deg,
                                          _sun_data.glare_active, _sun_data.description)
                            except Exception as _se:
                                log.warning("[LOCATION] Sun calc error: %s", _se)

                            # ── Age × temperature adjustment (one-shot, on first tick) ──
                            # WFO-validated edge: permutation p=0.000, bootstrap CI [+0.17,+0.37].
                            # Applied once per match after venue + live temperature are known.
                            if not _age_temp_applied and cfg.AGE_TEMP_ENABLED:
                                _age_temp_applied = True  # set before await to prevent re-entry
                                try:
                                    import math as _math
                                    _weather = await location_engine.get_weather(
                                        _venue.lat, _venue.lon, _venue.city
                                    )
                                    _temp_c    = _weather.temperature_c
                                    _age_diff  = _p1_age - _p2_age
                                    _logit_adj = (
                                        cfg.AGE_TEMP_AGE_COEF  * _age_diff
                                        + cfg.AGE_TEMP_TEMP_COEF * _temp_c * _age_diff
                                    )
                                    # Clamp so no single factor moves the needle more than ±MAX_ADJ
                                    _max_logit = _math.log(
                                        (0.5 + cfg.AGE_TEMP_MAX_ADJ)
                                        / (0.5 - cfg.AGE_TEMP_MAX_ADJ)
                                    )
                                    _logit_adj = max(-_max_logit, min(_max_logit, _logit_adj))
                                    _logit_base = _math.log(base_p_a / (1.0 - base_p_a))
                                    base_p_a    = 1.0 / (1.0 + _math.exp(-(_logit_base + _logit_adj)))
                                    _weather_data         = _weather
                                    _logit_age_temp       = _logit_adj
                                    _base_p_after_age_temp = base_p_a
                                    # Re-derive Markov serve/return params from adjusted baseline
                                    _new_ps = 0.65 + (base_p_a - 0.5) * cfg.MARKOV_SERVE_SCALE
                                    _new_pr = 0.35 + (base_p_a - 0.5) * cfg.MARKOV_RETURN_SCALE
                                    lms.update_params(_new_ps, _new_pr)
                                    log.info(
                                        "[AGE×TEMP] %.0f°C | p1_age=%.1f p2_age=%.1f "
                                        "age_diff=%+.1f | logit_adj=%+.4f | "
                                        "base_p %.3f→%.3f | p_serve %.3f→%.3f",
                                        _temp_c, _p1_age, _p2_age, _age_diff,
                                        _logit_adj,
                                        ml_res["hybrid_prob"], base_p_a,
                                        p_serve, _new_ps,
                                    )
                                except Exception as _ate:
                                    log.warning("[AGE×TEMP] adjustment failed: %s", _ate)

                            # ── Physical & structural edges (one-shot, same tick) ──
                            # Altitude×Height(Hard), Altitude×Age, Handedness, Rank-pts divergence.
                            # WFO-validated: all p<0.05 permutation, OOS ROI +10–22%.
                            if not _phys_applied and cfg.PHYS_ENABLED:
                                _phys_applied = True
                                try:
                                    import math as _math
                                    _alt_m    = _venue.altitude_m
                                    _ht_diff  = _p1_ht - _p2_ht
                                    _age_diff = _p1_age - _p2_age
                                    _surf     = (_venue.court_surface or "").lower()
                                    _is_hard  = 1.0 if "hard" in _surf else 0.0
                                    _is_clay  = 1.0 if "clay" in _surf else 0.0
                                    _lh_net   = 0.0
                                    if _p1_hand == "L" and _p2_hand == "R":
                                        _lh_net = 1.0
                                    elif _p1_hand == "R" and _p2_hand == "L":
                                        _lh_net = -1.0
                                    _pts_term = 0.0
                                    # Require both > 100 to exclude generic-fallback elo=0
                                    # and cap log-ratio at ±2 to prevent outlier blow-up
                                    if _p1_pts > 100 and _p2_pts > 100:
                                        _log_rank = _math.log(
                                            max(float(p2_stats.get("ranking", 50)), 1)
                                            / max(float(p1_stats.get("ranking", 50)), 1)
                                        )
                                        _pts_term = max(-2.0, min(2.0,
                                            _math.log(_p1_pts / _p2_pts) - _log_rank
                                        ))
                                    _logit_phys = (
                                        cfg.ALT_HT_HARD_COEF * _alt_m * _ht_diff * _is_hard
                                        + cfg.ALT_AGE_COEF   * _alt_m * _age_diff
                                        + cfg.LH_HARD_COEF   * _lh_net * _is_hard
                                        + cfg.LH_CLAY_COEF   * _lh_net * _is_clay
                                        + cfg.PTS_RANK_COEF  * _pts_term
                                    )
                                    _max_lp = _math.log(
                                        (0.5 + cfg.PHYS_MAX_ADJ) / (0.5 - cfg.PHYS_MAX_ADJ)
                                    )
                                    _logit_phys = max(-_max_lp, min(_max_lp, _logit_phys))
                                    _logit_cur  = _math.log(base_p_a / (1.0 - base_p_a))
                                    base_p_a    = 1.0 / (1.0 + _math.exp(-(_logit_cur + _logit_phys)))
                                    _alt_x_ht_comp  = cfg.ALT_HT_HARD_COEF * _alt_m * _ht_diff * _is_hard
                                    _alt_x_age_comp = cfg.ALT_AGE_COEF * _alt_m * _age_diff
                                    _lh_hard_comp   = cfg.LH_HARD_COEF * _lh_net * _is_hard
                                    _lh_clay_comp   = cfg.LH_CLAY_COEF * _lh_net * _is_clay
                                    _pts_rank_comp  = cfg.PTS_RANK_COEF * _pts_term
                                    _pts_vs_rank_raw  = _pts_term
                                    _pts_vs_rank_edge = abs(_pts_rank_comp)
                                    _logit_phys_total = _logit_phys
                                    _base_p_after_phys = base_p_a
                                    _lh_net_cached    = _lh_net
                                    _new_ps = 0.65 + (base_p_a - 0.5) * cfg.MARKOV_SERVE_SCALE
                                    _new_pr = 0.35 + (base_p_a - 0.5) * cfg.MARKOV_RETURN_SCALE
                                    lms.update_params(_new_ps, _new_pr)
                                    log.info(
                                        "[PHYS] alt=%.0fm ht_diff=%+.0fcm lh=%+.0f surf=%s "
                                        "pts_term=%+.3f | logit_adj=%+.4f | base_p %.3f→%.3f",
                                        _alt_m, _ht_diff, _lh_net, _surf,
                                        _pts_term, _logit_phys,
                                        ml_res["hybrid_prob"], base_p_a,
                                    )
                                except Exception as _pe:
                                    log.warning("[PHYS] adjustment failed: %s", _pe)

                            # ── Bayesian prior reseed with actual surface ─────────
                            # Once _surf is known, replace the initial "hard" prior
                            # with a surface-calibrated one (clay/grass/indoor).
                            if not _bayes_surface_reseeded and _venue is not None:
                                _bayes_surface_reseeded = True
                                _resolved_surf = (_venue.court_surface or "hard")
                                try:
                                    reseed_for_surface(
                                        _bayes_serve, p1_stats,
                                        surface=_resolved_surf, is_server=True,
                                    )
                                    log.info(
                                        "[PRIOR-RESEED] %s Bayesian prior updated for "
                                        "surface=%s → posterior=%.4f",
                                        player_a, _resolved_surf,
                                        _bayes_serve.get_posterior_mean(),
                                    )
                                except Exception as _pre:
                                    log.warning("[PRIOR-RESEED] failed: %s", _pre)

                    # Detect who won the last point by comparing current vs previous
                    # game-point tuple. Only count when exactly one side advanced by 1
                    # (ignores game resets to 0-0 and deuce oscillations cleanly).
                    pts = score_update.get("points", (0, 0))
                    p1_serving = score_update.get("p1_serving")

                    # BUG FIX: Accumulate serve stats across the ENTIRE match, not per game.
                    # Previously reset every time pts==(0,0), which meant the Bayesian
                    # updater (threshold=10 points) almost never fired (a game has 4-7 pts).
                    # Now we only increment a game counter for context logging.
                    if pts == (0, 0) and _prev_pts != (0, 0):
                        _serve_games_completed += 1
                        log.debug(
                            "[BAYESIAN] Game %d complete — cumulative serve pts: %d/%d",
                            _serve_games_completed, _serve_pts_won, _serve_pts_total,
                        )
                        # DO NOT reset _serve_pts_won/_serve_pts_total here —
                        # accumulate across the whole match for a meaningful posterior.

                    if p1_serving is not None and pts != _prev_pts:
                        p1_delta = pts[0] - _prev_pts[0]
                        p2_delta = pts[1] - _prev_pts[1]
                        # A single point was played: exactly one side moved up by 1
                        if (p1_delta == 1 and p2_delta == 0) or (p1_delta == 0 and p2_delta == 1):
                            p1_won_point = p1_delta == 1
                            _serve_pts_total += 1
                            if (p1_serving and p1_won_point) or (not p1_serving and not p1_won_point):
                                _serve_pts_won += 1

                            if _serve_pts_total >= 10:
                                live_hold = _serve_pts_won / _serve_pts_total
                                _bayes_serve.update(
                                    points_won=_serve_pts_won,
                                    points_total=_serve_pts_total,
                                )
                                _skill_drift.apply(_bayes_serve)
                                new_p_serve  = _bayes_serve.get_posterior_mean()
                                _serve_divergence = new_p_serve - _serve_prior_initial
                                new_p_return = max(0.01, min(0.99, 1.0 - new_p_serve + (lms.p_return - (1.0 - lms.p_serve))))
                                lms.update_params(new_p_serve, new_p_return)
                                lms.record_point(0 if p1_won_point else 1)
                                log.info(
                                    "Bayesian p_serve update: hold=%.3f → posterior=%.4f (unc=%.4f)",
                                    live_hold, new_p_serve, _bayes_serve.get_uncertainty(),
                                )

                    _prev_pts = pts

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
                                # 2nd-serve won % live refinement (WFO coef=3.0, p=0.000)
                                pts_2nd_a = atp.get("pts_won_2nd_serve_a", 0.0)
                                pts_2nd_b = atp.get("pts_won_2nd_serve_b", 0.0)
                                if pts_2nd_a > 1.0: pts_2nd_a /= 100.0
                                if pts_2nd_b > 1.0: pts_2nd_b /= 100.0
                                if 0.1 < pts_2nd_a < 0.9 and 0.1 < pts_2nd_b < 0.9:
                                    import math as _math
                                    _live_s2_diff = pts_2nd_a - pts_2nd_b
                                    _live_logit_adj = config.SERVE2_COEF * _live_s2_diff
                                    _sq_cap = _math.log(
                                        (0.5 + config.SERVE_QUALITY_MAX_ADJ)
                                        / (0.5 - config.SERVE_QUALITY_MAX_ADJ)
                                    )
                                    _live_logit_adj = max(-_sq_cap, min(_sq_cap, _live_logit_adj))
                                    _cur_p = lms.win_probability()
                                    _cur_logit = _math.log(_cur_p / (1.0 - _cur_p))
                                    _refined_p = 1.0 / (1.0 + _math.exp(-(_cur_logit + _live_logit_adj * 0.3)))
                                    _refined_ps = 0.65 + (_refined_p - 0.5) * cfg.MARKOV_SERVE_SCALE
                                    _refined_pr = 0.35 + (_refined_p - 0.5) * cfg.MARKOV_RETURN_SCALE
                                    lms.update_params(
                                        0.8 * lms.p_serve  + 0.2 * _refined_ps,
                                        0.8 * lms.p_return + 0.2 * _refined_pr,
                                    )
                                    log.info(
                                        "[SERVE-Q LIVE] 2nd_won: A=%.1f%% B=%.1f%% diff=%+.3f logit=%+.4f",
                                        pts_2nd_a * 100, pts_2nd_b * 100,
                                        _live_s2_diff, _live_logit_adj,
                                    )
                        except Exception as _ae:
                            log.debug("[ATP] Stat refresh error: %s", _ae)

                    win_prob_a = lms.win_probability()

                    # [SUN GLARE DISABLED — net negative edge]
                    # WFO backtest (9,860 ATP matches 2018-2024) shows bet_roi = -23.5%
                    # across 20/20 folds with p=0.90 (no statistical significance).
                    # The penalty was adjusting win_prob by up to ±4%, hurting expected
                    # value on every bet it fired on.  Kept for dashboard display only.
                    _hard_court = _venue is not None and _venue.court_surface == "hard"
                    if _sun_data is not None and _sun_data.glare_active and _hard_court:
                        _p1_srv = score_update.get("p1_serving", True)
                        _penalty = _sun_data.p1_sun_penalty if _p1_srv else _sun_data.p2_sun_penalty
                        if _penalty > 0.001:
                            log.debug(
                                "[SUN] ☀ Glare detected (DISABLED): %s penalty=%.1f%% | %s"
                                " — win_prob unchanged (feature removed, see WFO results).",
                                player_a if _p1_srv else player_b,
                                _penalty * 100, _sun_data.description,
                            )

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
                    # Balance is cached — fetch both in parallel even when market is cached
                    available_balance = await kalshi.get_balance()
                else:
                    # Cache stale or monitor not running — fetch market + balance in parallel.
                    _market_raw, available_balance = await asyncio.gather(
                        kalshi.get_market(ticker),
                        kalshi.get_balance(),
                    )
                    market = _market_raw
                    market["player_a"] = player_a
                    market["player_b"] = player_b

                flow_data = {
                    "direction":  flow_sig.direction.value if flow_sig else "NEUTRAL",
                    "velocity":   round(flow_sig.price_velocity * 100, 4) if flow_sig else 0.0,
                    "vol_regime": flow_sig.vol_regime if flow_sig else "LOW",
                    "yes_price":  round(flow_sig.yes_price, 4) if flow_sig else 0.5,
                    "z_score":    round(flow_sig.z_score, 4) if flow_sig else 0.0,
                }

                _live_score_str = score_update.get("score") or status_str
                _markov_prob = round(win_prob_a, 4)
                asyncio.get_running_loop().run_in_executor(None, lambda: _write_live_state(
                    LIVE_STATE_PATH, player_a, player_b, p1_stats, p2_stats,
                    win_prob, nn_prob, xgb_prob, mode, feed_status, flow_data,
                    surface=_resolved_surface,
                    score=_live_score_str,
                    markov_prob=_markov_prob,
                ))

                eval_start = time.time()
                eval_latency = (eval_start - pipeline_start) * 1000
                kelly_mult = adaptive.kelly_multiplier if adaptive else 1.0

                # Determine live/stale status FIRST — must be done before any order logic.
                market_is_active = market.get("active", True)
                is_live_match = market_is_active and feed_status == "live"

                if feed_status == "stale" and market_is_active:
                    log.warning(
                        "[SCORE-FEED] STALE — no live score for %s vs %s. "
                        "Trading BLOCKED until live score resumes.",
                        player_a, player_b,
                    )

                # ── Predictive limit orders (placed BEFORE the next point) ────────────
                # Cancel any limits that are now stale (point already played / price moved).
                # Then place a fresh resting limit at the anticipated post-point price.
                # Only place new predictive limits when we have a confirmed live feed.
                if score_update.get("is_live") and is_live_match:
                    await bets.cancel_stale_limits(market["yes_price"])
                    await bets.place_predictive_limit_order(
                        market=market,
                        lms=lms,
                        available_balance=available_balance,
                        kelly_mult=kelly_mult,
                        adaptive=adaptive,
                    )

                try:
                    # Attach full signal context for convergence gate + TradeTracker
                    win_prob.update({
                        "pts_vs_rank_edge":         _pts_vs_rank_edge,
                        "serve_divergence":         _serve_divergence,
                        "model_prob_ml_base":       ml_res.get("hybrid_prob", float("nan")),
                        "model_prob_nn":            nn_prob,
                        "model_prob_xgb":           xgb_prob,
                        "logit_adj_age_temp":       _logit_age_temp,
                        "model_prob_after_age_temp":_base_p_after_age_temp,
                        "logit_adj_phys":           _logit_phys_total,
                        "model_prob_after_phys":    _base_p_after_phys,
                        "model_prob_final":         base_p_a,
                        "markov_p_serve_initial":   _serve_prior_initial,
                        "markov_p_serve_now":       lms.p_serve,
                        "markov_p_return_now":      lms.p_return,
                        "bayesian_posterior":       _bayes_serve.get_posterior_mean(),
                        "bayes_uncertainty":        _bayes_serve.get_uncertainty(),
                        "pts_vs_rank_raw":          _pts_vs_rank_raw,
                        "alt_x_ht_comp":            _alt_x_ht_comp,
                        "alt_x_age_comp":           _alt_x_age_comp,
                        "lh_hard_comp":             _lh_hard_comp,
                        "lh_clay_comp":             _lh_clay_comp,
                        "pts_rank_comp":            _pts_rank_comp,
                        "lh_net":                   _lh_net_cached,
                        "age_diff_cached":          _p1_age - _p2_age,
                        "height_diff_cached":       _p1_ht - _p2_ht,
                        "atp_tick":                 _atp_tick,
                        "score_update":             score_update,
                        "player_stats_a":           p1_stats,
                        "player_stats_b":           p2_stats,
                        "weather_data":             _weather_data,
                        "venue_data":               _venue,
                        "sun_data":                 _sun_data,
                    })
                    result = await bets.evaluate_and_act(
                        market=market,
                        win_prob=win_prob,
                        game_state=None,
                        event={"type": "point_won", "clock": status_str},
                        pipeline_start=pipeline_start,
                        market_monitor=_market_monitors.get(ticker),
                        adaptive=adaptive,
                        available_balance=available_balance,
                        is_live_match=is_live_match,
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
        _metrics_f.close()
        log.info(f"Session ended for {ticker}. Final positions:")
        bets.print_summary()


async def worker(
    queue: asyncio.PriorityQueue,
    kalshi: KalshiClient,
    history: HistoricalAnalyzer,
    bets: BetManager,
    config: Config,
    adaptive: AdaptiveController,
    active_tickers: set,
    location_engine: LocationEngine = None,
):
    """Worker task to process matches up to the concurrency limit."""
    while True:
        # PriorityQueue items are (priority, seq, match_dict)
        # priority 0 = currently live on LiveScore, 1 = upcoming
        _priority, _seq, match = await queue.get()
        ticker = match["ticker"]
        live_hint = "LIVE" if _priority == 0 else "UPCOMING"
        log.info("[WORKER] Picked up %s %s vs %s (priority=%d)",
                 live_hint, match["player_a"], match["player_b"], _priority)

        monitor = MarketMonitor(kalshi, ticker)
        _market_monitors[ticker] = monitor
        monitor_task = asyncio.create_task(monitor.run())

        async def _orderbook_cb(msg, _mon=monitor):
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
            await asyncio.wait_for(
                process_match(match, kalshi, history, bets, config,
                              adaptive=adaptive, location_engine=location_engine),
                timeout=8 * 3600,  # 8-hour hard cap — releases worker slot if match stalls
            )
        except asyncio.TimeoutError:
            log.warning("[WORKER] Match %s exceeded 8-hour timeout — releasing worker slot.", ticker)
        except Exception as e:
            log.error(f"Error processing match {ticker}: {e}")
        finally:
            monitor.stop()
            monitor_task.cancel()
            ws_ob_task.cancel()
            ws_fills_task.cancel()
            _market_monitors.pop(ticker, None)
            active_tickers.discard(ticker)
            queue.task_done()


async def _discover_markets(
    kalshi: KalshiClient,
    config: Config,
    queue: asyncio.PriorityQueue,
    active_tickers: set,
    seq_counter: list,
):
    """
    Fetch Kalshi ATP markets, check LiveScore for which are currently live,
    and enqueue any not already being tracked.
    Live matches get priority 0; upcoming get priority 1.
    """
    scraper = LiveScoreScraper(config)
    try:
        atp_matches = await kalshi.get_atp_markets()
    except Exception as e:
        log.error("[DISCOVER] Failed to fetch Kalshi markets: %s", e)
        return

    if not atp_matches:
        log.info("[DISCOVER] No open tennis markets found.")
        return

    _write_kalshi_match_state(atp_matches)

    try:
        live_now = await scraper.fetch_live_scores()
    except Exception:
        live_now = []

    added = 0
    for match in atp_matches:
        ticker = match["ticker"]
        if ticker in active_tickers:
            continue

        found = scraper.find_match(match["player_a"], match["player_b"], live_now)
        priority = 0 if found else 1
        match["_is_live_now"] = bool(found)

        active_tickers.add(ticker)
        seq_counter[0] += 1
        await queue.put((priority, seq_counter[0], match))
        added += 1
        log.info(
            "[DISCOVER] Queued %s vs %s as %s (ticker=%s)",
            match["player_a"], match["player_b"],
            "LIVE" if found else "UPCOMING", ticker,
        )

    if added:
        log.info("[DISCOVER] Added %d new market(s) to queue.", added)


async def run_game_session():
    """
    Full lifecycle:
    1. Discover open ATP markets from Kalshi, check LiveScore for live status
    2. Priority-queue them: live matches (priority=0) before upcoming (priority=1)
    3. Process using a capped pool of parallel worker tasks
    4. Background task re-scans every 60 s so new matches join the queue live-first
    """
    global _market_monitors
    _market_monitors = {}

    config          = Config()
    kalshi          = KalshiClient(config)
    history         = HistoricalAnalyzer(config)
    tracker         = TradeTracker(base_dir=BASE_DIR)
    bets            = BetManager(kalshi, config, tracker=tracker)
    adaptive        = AdaptiveController(base_min_edge=max(config.MIN_EDGE, 0.02))
    location_engine = LocationEngine()
    log.info("[ADAPTIVE] Controller started. Base MIN_EDGE=%.4f", adaptive.base_min_edge)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_serve_cache)
    log.info("[SERVE-Q] Player serve-quality cache loaded.")

    # Write an initial "scanning" status so the TUI shows the bot is alive.
    _write_live_state(
        LIVE_STATE_PATH, "SCANNING", "MARKETS",
        {}, {}, {"team_a": 0.5, "team_b": 0.5},
        None, None, get_trading_mode(), "scanning", None,
    )

    # BUG FIX: Reconcile open positions from Kalshi portfolio at startup.
    # Without this, a server restart loses all position state and the bot may
    # re-enter trades it already holds or fail to exit positions it thinks are closed.
    await bets.reconcile_positions_from_kalshi()

    # Shared state for deduplication across discovery cycles
    active_tickers: set = set()
    seq_counter: list   = [0]   # mutable int so the nested coroutine can increment it

    queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

    async def discovery_loop():
        while True:
            try:
                await _discover_markets(kalshi, config, queue, active_tickers, seq_counter)
            except Exception as e:
                log.error("[DISCOVER] Unexpected error: %s", e)
            await asyncio.sleep(_DISCOVERY_INTERVAL)

    disc_task = None
    workers   = []
    try:
        # Initial discovery (don't wait for the 60-s loop)
        await _discover_markets(kalshi, config, queue, active_tickers, seq_counter)

        if queue.empty():
            log.warning(
                "[DISCOVER] No open tennis Kalshi markets found — waiting for markets to open. "
                "Background scanner will check every %ds.", _DISCOVERY_INTERVAL
            )

        # Background re-discovery: keeps running even when queue starts empty.
        disc_task = asyncio.create_task(discovery_loop())

        workers = [
            asyncio.create_task(
                worker(queue, kalshi, history, bets, config, adaptive, active_tickers,
                       location_engine=location_engine)
            )
            for _ in range(MAX_CONCURRENT_MATCHES)
        ]
        log.info(
            "[SESSION] %d worker(s) started — waiting for markets. Max concurrent: %d",
            MAX_CONCURRENT_MATCHES, MAX_CONCURRENT_MATCHES,
        )

        # Run until cancelled (SIGTERM/SIGINT) rather than until queue drains.
        # The discovery loop continuously re-queues new markets as they appear.
        await asyncio.Event().wait()

    except asyncio.CancelledError:
        log.info("Session cancelled via system signal. Shutting down gracefully...")
    finally:
        if disc_task:
            disc_task.cancel()
        for w in workers:
            w.cancel()
        log.info("Closing all external connections and saving state...")
        await kalshi.close()
        await history.close()


if __name__ == "__main__":
    import signal
    
    # Python 3.10+ requires explicit loop creation if one isn't running
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    main_task = loop.create_task(run_game_session())
    
    def handle_shutdown():
        log.info("Received shutdown signal. Initiating graceful shutdown...")
        main_task.cancel()
        
    try:
        loop.add_signal_handler(signal.SIGTERM, handle_shutdown)
        loop.add_signal_handler(signal.SIGINT, handle_shutdown)
    except NotImplementedError:
        # Windows compatibility
        pass
        
    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    finally:
        log.info("Bot successfully terminated with persistence intact.")
