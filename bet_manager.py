"""
bet_manager.py — Bet sizing, edge detection, and position management.

Kelly criterion sizing + edge threshold gating.
Manages buy/sell decisions based on Elo-derived win probability vs market price.
"""

import asyncio
import logging
import re
import time
from kalshi_client import KalshiClient
from config import Config
from market_monitor import MarketMonitor, MarketSignal, FlowDirection
from adaptive_controller import AdaptiveController
from trade_tracker import TradeTracker
from ml_engine import ml_engine

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
import plotly.graph_objects as go

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Robust Kelly with Uncertainty Shrinkage  (research-paper derived)
# ──────────────────────────────────────────────────────────────────────────────

def robust_kelly(
    p_model: float,
    p0: float,
    p1: float,
    odds: float,
    bankroll: float,
    lambda_shrink: float = 0.5,
) -> float:
    """
    Kelly bet sizing with Venn-Abers uncertainty shrinkage.

    Standard Kelly assumes `p_model` is exact; Venn-Abers gives us a calibrated
    interval [p0, p1].  The width |p1 - p0| is epistemic uncertainty; we
    shrink the Kelly fraction in proportion:

        f_robust = f_standard * (1 - lambda_shrink * |p1 - p0|)

    Args:
        p_model:      point estimate of win probability (Venn-Abers midpoint).
        p0, p1:       lower / upper Venn-Abers calibrated probabilities.
        odds:         decimal odds (e.g. 2.10).  Must be > 1 for a bet to exist.
        bankroll:     current bankroll in dollars.
        lambda_shrink:shrinkage strength in [0, 1] (0 = no shrinkage).

    Returns:
        Bet size in dollars (non-negative, capped at bankroll).
    """
    if odds <= 1.0 or bankroll <= 0:
        return 0.0
    p_model = float(max(0.0, min(1.0, p_model)))
    b = float(odds) - 1.0
    q = 1.0 - p_model
    f_std = (p_model * b - q) / b
    if f_std <= 0:
        return 0.0

    uncertainty = abs(float(p1) - float(p0))
    shrink_factor = max(0.0, 1.0 - float(lambda_shrink) * uncertainty)
    f_robust = f_std * shrink_factor

    dollars = float(f_robust) * float(bankroll)
    return max(0.0, min(dollars, float(bankroll)))


def adaptive_edge_threshold(market_price: float) -> float:
    """
    Market-price-dependent minimum edge required to place a bet.

    Rationale: thin-price markets (longshots / heavy favourites) have
    fatter-tailed estimation error, so require a larger cushion.

    Returns the minimum edge as a probability difference.
    """
    p = float(market_price)
    if p < 0.10:
        return 0.15
    if p < 0.20:
        return 0.08
    if p < 0.80:
        return 0.02
    return 0.01


def build_alpha_surface(bet_log_df):
    if bet_log_df.empty or len(bet_log_df) < 5:
        return None

    model_range  = np.linspace(0.3, 0.8, 50)
    market_range = np.linspace(0.3, 0.8, 50)
    X, Y = np.meshgrid(model_range, market_range)

    points = bet_log_df[['model_p', 'market_p']].values
    values = bet_log_df['roi'].values

    # Add tiny jitter so duplicate / collinear points never produce a
    # singular polynomial matrix (was crashing with thin_plate_spline).
    points = points + np.random.default_rng(seed=42).uniform(-1e-6, 1e-6, points.shape)

    try:
        # gaussian kernel requires no polynomial augmentation → rank-safe.
        rbf = RBFInterpolator(
            points, values,
            kernel='gaussian',
            epsilon=2.0,
            smoothing=0.5,
        )
        Z = rbf(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    except Exception as exc:
        log.warning("RBFInterpolator failed (%s) — skipping surface rebuild.", exc)
        return None
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='RdYlGn',   # red = losing zones, green = edge zones
        colorbar=dict(title='ROI'),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white")
        )
    )])
    
    # Overlay actual bet points as scatter
    fig.add_trace(go.Scatter3d(
        x=bet_log_df['model_p'],
        y=bet_log_df['market_p'],
        z=bet_log_df['roi'],
        mode='markers',
        marker=dict(
            size=4,
            color=bet_log_df['roi'],
            colorscale='RdYlGn',
            opacity=0.8
        )
    ))
    
    fig.update_layout(
        title='Model vs Market Edge Surface',
        scene=dict(
            xaxis_title='Your Model P',
            yaxis_title='Kalshi Market P',
            zaxis_title='ROI %',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )
    )
    
    return fig, rbf

class BetManager:
    def __init__(self, kalshi: KalshiClient, config: Config, tracker: TradeTracker = None):
        self.kalshi  = kalshi
        self.cfg     = config
        self.tracker = tracker
        self._total_bet_usdc = 0.0
        self._trade_log = []
        self._open_positions: dict = {}   # pos_key → entry data
        self._sell_cooldown: dict = {}    # pos_key → timestamp of last sell
        self._pending_limits: dict = {}   # pos_key → {order_id, price_cents, count, side, ticker}
        self.surface_model = None
        self._buy_lock = asyncio.Lock()

    async def reconcile_positions_from_kalshi(self) -> None:
        """
        Seed _open_positions from the real Kalshi portfolio at startup.

        Prevents duplicate entries and missed exits after server restarts.
        In DRY_RUN mode (no private key) this is a no-op — positions are
        simulated in memory only.
        """
        if not self.kalshi.private_key:
            log.info(
                "[RECONCILE] DRY_RUN mode — skipping Kalshi position sync "
                "(no real positions to reconcile)."
            )
            return
        try:
            resp = await self.kalshi._request("GET", "/portfolio/positions")
            positions = resp.get("market_positions") or resp.get("positions") or []
            loaded = 0
            for pos in positions:
                ticker = pos.get("market_ticker") or pos.get("ticker")
                # Kalshi returns net signed position: positive = YES contracts held
                yes_count = int(pos.get("position", 0))
                no_count  = int(pos.get("position_no", 0))
                for side, count in (("yes", yes_count), ("no", no_count)):
                    if count <= 0:
                        continue
                    pos_key = f"{ticker}_{side}"
                    if pos_key not in self._open_positions:
                        # Reconstruct a minimal position record so exit logic works
                        self._open_positions[pos_key] = {
                            "ticker":        ticker,
                            "side":          side,
                            "count":         count,
                            "cost_usdc":     float(pos.get("total_cost", 0)) / 100.0,
                            "entry_price":   0.0,   # unknown; exit logic will use market price
                            "reconciled":    True,   # flag: loaded from API, not from this session
                            "player_a":      "",
                            "player_b":      "",
                        }
                        log.info(
                            "[RECONCILE] Loaded position: %s %s x%d (from Kalshi portfolio)",
                            ticker, side.upper(), count,
                        )
                        loaded += 1
            log.info(
                "[RECONCILE] Done — %d pre-existing position(s) loaded into tracking.",
                loaded,
            )
        except Exception as e:
            log.warning(
                "[RECONCILE] Could not load Kalshi positions: %s — "
                "starting with empty position state. "
                "Existing positions (if any) will NOT be double-counted on first entry check.",
                e,
            )


    # ──────────────────────────────────────────────────────────────────────────
    # Main decision loop
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fee_per_contract(price: float, fee_rate: float) -> float:
        """Kalshi per-contract entry fee as a fraction of $1.
        Approximates ceil(fee_rate * P * (1-P) * 100) / 100 ≈ fee_rate * P * (1-P).
        """
        if price <= 0 or price >= 1:
            return 0.0
        return fee_rate * price * (1.0 - price)

    async def evaluate_and_act(
        self,
        market: dict,
        win_prob: dict,
        game_state,        # reserved for future in-game state
        event: dict,       # reserved for event metadata
        pipeline_start: float = None,
        market_monitor: MarketMonitor = None,
        adaptive: AdaptiveController = None,
        available_balance: float = None,
        is_live_match: bool = True,
    ):
        """
        Called after every live event.
        1. Compute edge on YES and NO sides
        2. If edge > adaptive MIN_EDGE and flow confirms → buy
        3. If we hold a position that's now overpriced → sell
        """
        if not market.get("active", True):
            log.info("Market inactive — skipping.")
            return
            
        p_a_quality = win_prob.get("player_stats_a", {}).get("data_quality", "high")
        p_b_quality = win_prob.get("player_stats_b", {}).get("data_quality", "high")
        if p_a_quality == "low" or p_b_quality == "low":
            log.warning(f"[DATA GATE] Low data quality detected for player(s) (A:{p_a_quality}, B:{p_b_quality}). Blocking trade.")
            return

        injury_flag = event.get("injury_flag", False)
        if injury_flag:
            log.warning(f"🚨 MEDICAL TIMEOUT / INJURY FLAG DETECTED for {market.get('id')}. Halting Limit Orders.")
            # Cancel all pending limits for this ticker
            for pos_key, pending in list(self._pending_limits.items()):
                if pending["ticker"] == market["id"]:
                    order_id = pending.get("order_id")
                    if order_id:
                        try:
                            await self.kalshi.cancel_order(order_id)
                        except Exception as e:
                            log.error(f"Failed to cancel MTO order {order_id}: {e}")
                    del self._pending_limits[pos_key]
            # Do not return here completely, allow the 'sell' logic to run so we dump positions, 
            # but we will block the 'buy' logic below.

        # Evict stale cooldown entries (older than 2 h) to prevent unbounded growth
        _now_mono = time.monotonic()
        if self._sell_cooldown:
            self._sell_cooldown = {
                k: v for k, v in self._sell_cooldown.items()
                if _now_mono - v < 7200
            }

        # Market prices
        yes_price = market["yes_price"]   # cost of 1 YES share
        no_price  = market["no_price"]    # cost of 1 NO share

        # Our model probabilities
        p_a = win_prob["team_a"]   # probability team A (home/YES) wins
        p_b = win_prob["team_b"]

        # YES/NO Side Resolution (Bug Fix: was single last-name token match, which silently
        # flipped bets whenever Kalshi and LiveScore list players in a different order).
        #
        # New logic: multi-token match on all non-trivial parts of player_a's name.
        # - Match left side  of '<name> vs <name>' → player_a IS YES
        # - Match right side of '<name> vs <name>' → player_a IS NO
        # - Both sides match, or neither matches → AMBIGUOUS: skip the bet entirely.
        #   It is far better to miss a bet than to bet on the wrong player.
        player_a_name = market.get("player_a", "")
        market_title  = market.get("question", "") or (market.get("_raw") or {}).get("title", "")
        yes_is_player_a: bool | None = None  # None = ambiguous → will abstain
        if player_a_name and market_title:
            pa_lower = player_a_name.lower()
            title_lower = market_title.lower()
            # Extract all meaningful name tokens (>2 chars, alpha only)
            pa_tokens = [
                t for t in re.sub(r"[^a-z ]", "", pa_lower).split()
                if len(t) > 2
            ]
            vs_pos = title_lower.find(" vs")
            if vs_pos > 0 and pa_tokens:
                left_side  = title_lower[:vs_pos]
                right_side = title_lower[vs_pos + 4:]  # skip " vs "
                left_match  = any(tok in left_side  for tok in pa_tokens)
                right_match = any(tok in right_side for tok in pa_tokens)

                if left_match and not right_match:
                    yes_is_player_a = True
                    log.info(
                        "[YES/NO] Resolved: %s = YES (left side of '%s')",
                        player_a_name, market_title,
                    )
                elif right_match and not left_match:
                    yes_is_player_a = False
                    log.info(
                        "[YES/NO] Resolved: %s = NO (right side of '%s')",
                        player_a_name, market_title,
                    )
                else:
                    log.warning(
                        "[YES/NO] AMBIGUOUS mapping for '%s' in title '%s' "
                        "(left_match=%s right_match=%s) — SKIPPING BET to avoid side flip.",
                        player_a_name, market_title, left_match, right_match,
                    )
                    return  # Safe abstain: never bet on an ambiguous side
            else:
                # No 'vs' separator or no tokens — can't determine side safely
                log.warning(
                    "[YES/NO] Cannot parse 'vs' in title '%s' for player '%s' — SKIPPING BET.",
                    market_title, player_a_name,
                )
                return
        else:
            log.warning(
                "[YES/NO] Missing player_a or market title — SKIPPING BET."
            )
            return

        if not yes_is_player_a:
            p_a, p_b = p_b, p_a   # swap so p_a always = probability that YES wins

        # Edge = model probability - market implied probability - per-contract fee.
        # Kalshi charges ~0.07 * P * (1-P) on every winning contract at entry,
        # so the raw model edge has to clear that bar before we enter.
        fee_rate = getattr(self.cfg, "KALSHI_FEE_RATE", 0.07)
        fee_yes  = self._fee_per_contract(yes_price, fee_rate)
        fee_no   = self._fee_per_contract(no_price,  fee_rate)
        edge_yes = p_a - yes_price - fee_yes
        edge_no  = p_b - no_price  - fee_no

        log.info(f"  Market: YES={yes_price:.3f} NO={no_price:.3f} | "
                 f"Model: A={p_a:.3f} B={p_b:.3f} | "
                 f"Edge(net fee): YES={edge_yes:+.3f} NO={edge_no:+.3f} "
                 f"[fee YES={fee_yes:.3f} NO={fee_no:.3f}]")
        
        # Periodically rebuild alpha surface from trades
        finished_trades = [t for t in self._trade_log if "roi" in t]
        if len(finished_trades) >= 5 and len(finished_trades) % 5 == 0:
            df = pd.DataFrame(finished_trades)
            result = build_alpha_surface(df)
            if result is not None:
                fig, rbf = result
                self.surface_model = rbf
                try:
                    import os as _os
                    surface_path = _os.path.join(
                        _os.path.dirname(_os.path.abspath(__file__)),
                        "static", "alpha_surface.html"
                    )
                    fig.write_html(surface_path)
                except Exception as e:
                    log.warning("Could not write alpha_surface.html: %s", e)

        # --- Exit logic (check before entry) ---
        await self._check_exits(market, win_prob, market_monitor=market_monitor)

        # --- Entry logic ---
        if available_balance is None:
            available_balance = await self.kalshi.get_balance()
            
        # Allow pregame limit orders (Market Maker mode)
        if not is_live_match:
            log.info("  [PREGAME] Market is pre-game. Spooling pre-match limit orders based on baseline ML.")
        else:
            # Live Match: Activate Predictive Sniper limit orders
            if game_state and hasattr(self, 'place_predictive_limit_order'):
                if hasattr(self, 'cancel_stale_limits'):
                    await self.cancel_stale_limits(yes_price)
                await self.place_predictive_limit_order(
                    market, game_state, available_balance=available_balance, 
                    kelly_mult=(adaptive.kelly_multiplier if adaptive else 1.0), 
                    adaptive=adaptive
                )

        # Adaptive MIN_EDGE: compute per-side thresholds so a longshot YES price
        # (< 10c → 15% floor) does not incorrectly block the NO side, which as the
        # heavy-favourite side only warrants a 1-4% floor.
        controller_edge = adaptive.current_min_edge if adaptive else max(self.cfg.MIN_EDGE, 0.02)
        effective_min_edge_yes = max(controller_edge, adaptive_edge_threshold(yes_price))
        effective_min_edge_no  = max(controller_edge, adaptive_edge_threshold(no_price))

        # Market flow signal — get directional signal for each side
        flow_yes = market_monitor.signal_for_side(betting_on_yes=True)  if market_monitor else MarketSignal()
        flow_no  = market_monitor.signal_for_side(betting_on_yes=False) if market_monitor else MarketSignal()

        if market_monitor:
            log.info(
                f"  [FLOW] YES={flow_yes.direction.value} (vel={flow_yes.price_velocity*100:+.3f}¢/s) | "
                f"NO={flow_no.direction.value}"
            )

        def should_bet_meta(model_p, market_price, edge, flow: MarketSignal, min_edge: float):
            if flow.direction == FlowDirection.FADE:
                return False, f"[FLOW] FADE signal — market moving against our edge"
            if edge < min_edge:
                return False, f"Edge {edge:+.3f} below adaptive threshold {min_edge:.3f}"
            if market_price <= 0 or market_price >= 1:
                return False, f"Invalid market price {market_price}"
            b = (1.0 / market_price) - 1.0
            kelly_f = (model_p * b - (1.0 - model_p)) / b
            if kelly_f <= 0:
                return False, f"Kelly fraction negative (no edge)"
            return True, ""

        should_yes, reason_yes = should_bet_meta(p_a, yes_price, edge_yes, flow_yes, effective_min_edge_yes)
        should_no, reason_no   = should_bet_meta(p_b, no_price,  edge_no,  flow_no,  effective_min_edge_no)

        if injury_flag:
            should_yes = False
            should_no = False
            reason_yes = "[MTO] Injury flag active"
            reason_no = "[MTO] Injury flag active"

        # ── Mean-reversion Z-score override (Phase 3 — information.md) ──────────
        # If normal edge didn't fire but MarketMonitor detects a statistically
        # significant underpricing (Z > 2.0, positive net edge after fees),
        # override should_yes to capture the dip.
        if not should_yes and not should_no and market_monitor:
            mr = market_monitor.mean_reversion_signal(fee_rate=fee_rate)
            if mr["should_buy_dip"]:
                should_yes = True
                edge_yes   = mr["net_edge"]
                log.info(
                    "  [MEAN-REV] Buy-the-dip override: Z=%+.2f net_edge=%+.4f "
                    "(P_model=%.3f P_market=%.3f)",
                    mr["z_score"], mr["net_edge"],
                    mr["p_model"], mr["p_market"],
                )

        # Kelly multiplier from adaptive controller (0.0 = protection mode = block).
        # Fetch balance exactly once per tick — caller may also pass it in.
        kelly_mult = adaptive.kelly_multiplier if adaptive else 1.0

        # ── Path 1: Serve-convergence Kelly gate ──────────────────────────────
        # When pts_vs_rank edge > 4% AND live serve rate is outperforming the
        # pre-match prior by > 2% → model-market gap is compressing in real time
        # → boost Kelly 1.5×. When contradicted (pts edge present but serve
        # diverging negatively) → reduce to 0.25× to avoid fading a deterioration.
        _pts_edge = win_prob.get("pts_vs_rank_edge", 0.0)
        _srv_div  = win_prob.get("serve_divergence",  0.0)
        kelly_mult_conv = 1.0
        if _pts_edge > 0.04 and _srv_div > 0.02:
            kelly_mult_conv = 1.5
            log.info("[CONVERGENCE] BOOST ×1.5: pts_edge=%.4f srv_div=%+.4f", _pts_edge, _srv_div)
        elif _pts_edge > 0.04 and _srv_div < -0.02:
            kelly_mult_conv = 0.25
            log.info("[CONVERGENCE] REDUCE ×0.25: pts_edge=%.4f srv_div=%+.4f (model-market diverge)", _pts_edge, _srv_div)
        kelly_mult = min(kelly_mult * kelly_mult_conv, 2.0)

        if adaptive:
            adaptive.update_bankroll(available_balance)

        # Snapshot trade log length before entry logic so we can measure new PnL.
        pre_sell_count = sum(
            1 for t in self._trade_log if t["action"] in ("SELL", "PARTIAL_SELL")
        )

        if should_yes:
            await self._attempt_buy(
                market=market,
                token_key="yes",
                edge=edge_yes,
                model_prob=p_a,
                market_price=yes_price,
                pipeline_start=pipeline_start,
                kelly_mult=kelly_mult,
                flow_signal=flow_yes,
                total_balance=available_balance,
                signal_context=win_prob,
                kelly_mult_conv=kelly_mult_conv,
            )
        elif should_no:
            await self._attempt_buy(
                market=market,
                token_key="no",
                edge=edge_no,
                model_prob=p_b,
                market_price=no_price,
                pipeline_start=pipeline_start,
                kelly_mult=kelly_mult,
                flow_signal=flow_no,
                total_balance=available_balance,
                signal_context=win_prob,
                kelly_mult_conv=kelly_mult_conv,
            )
        else:
            log.info(f"  Skipping: YES:({reason_yes}) | NO:({reason_no})")

        # Sum realized PnL from any sells that happened this evaluation cycle.
        new_sells = [
            t for t in self._trade_log
            if t["action"] in ("SELL", "PARTIAL_SELL")
        ][pre_sell_count:]
        realized_pnl = sum(t.get("pnl", 0.0) for t in new_sells)

        return {
            "profit":         realized_pnl,
            "open_positions": len(self._open_positions),
            "total_bet":      self._total_bet_usdc,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Buy logic
    # ──────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────
    # Predictive limit orders
    # ──────────────────────────────────────────────────────────────────────────

    async def place_predictive_limit_order(
        self,
        market: dict,
        lms,                          # LiveMatchState instance
        available_balance: float,
        kelly_mult: float = 1.0,
        adaptive=None,
    ):
        """
        Place a resting limit order BEFORE the next point is played.

        Strategy:
          1. Ask the Markov engine what the YES price will be in each point branch.
          2. If the expected post-point probability gives us edge at the PRE-move
             price, place a limit order at that price now.
          3. When the score changes and Kalshi re-prices, our order fills at the
             cheaper pre-move price instead of chasing the new ask.
          4. If the score goes the wrong way, cancel the stale limit.

        This is compatible with all existing Kelly sizing and exit logic —
        a lower entry price only increases realised edge.
        """
        ticker    = market["id"]
        yes_price = market["yes_price"]
        no_price  = market["no_price"]

        # Get one-point lookahead from Markov engine
        try:
            lookahead = lms.predict_post_point_state()
        except Exception as e:
            log.debug(f"[PREDICTIVE] lookahead failed: {e}")
            return

        prob_if_win  = lookahead["prob_if_win"]
        prob_if_lose = lookahead["prob_if_lose"]
        p_win_next   = lookahead["p_win_next"]

        fee_rate = getattr(self.cfg, "KALSHI_FEE_RATE", 0.07)
        controller_edge_limit = adaptive.current_min_edge if adaptive else max(self.cfg.MIN_EDGE, 0.02)

        # Determine which side has predictive edge in either branch.
        candidates = []

        # YES side: we expect YES price to rise (A is about to win a point)
        anticipated_yes = p_win_next * prob_if_win + (1 - p_win_next) * prob_if_lose
        edge_yes_limit = anticipated_yes - yes_price - self._fee_per_contract(yes_price, fee_rate)
        min_edge_yes_limit = max(controller_edge_limit, adaptive_edge_threshold(yes_price))
        if edge_yes_limit >= min_edge_yes_limit:
            candidates.append(("yes", yes_price, edge_yes_limit))

        # NO side: we expect NO price to rise (A is about to lose a point)
        anticipated_no = p_win_next * (1 - prob_if_win) + (1 - p_win_next) * (1 - prob_if_lose)
        edge_no_limit = anticipated_no - no_price - self._fee_per_contract(no_price, fee_rate)
        min_edge_no_limit = max(controller_edge_limit, adaptive_edge_threshold(no_price))
        if edge_no_limit >= min_edge_no_limit:
            candidates.append(("no", no_price, edge_no_limit))

        if not candidates:
            return

        side, limit_price, edge = max(candidates, key=lambda x: x[2])
        pos_key = f"{ticker}_{side}"

        # Skip if already holding or in cooldown
        if pos_key in self._open_positions or pos_key in self._pending_limits:
            return
        last_sell = self._sell_cooldown.get(pos_key, 0.0)
        if time.monotonic() - last_sell < 90.0:
            return

        price_cents = max(1, min(99, int(limit_price * 100)))
        count = int(self._kelly_size(
            anticipated_yes if side == "yes" else anticipated_no,
            limit_price,
            available_balance=available_balance,
            kelly_mult=kelly_mult,
        ) / limit_price) if limit_price > 0 else 0

        if count <= 0:
            return

        log.info(
            f"[PREDICTIVE] {side.upper()} limit {count}x @ {price_cents}¢ "
            f"(expected_prob={anticipated_yes if side == 'yes' else anticipated_no:.3f} "
            f"edge={edge:+.3f}) on {ticker}"
        )

        try:
            resp = await self.kalshi.place_limit_order(
                ticker=ticker, price_cents=price_cents, count=count, side=side
            )
        except Exception as e:
            log.warning(f"[PREDICTIVE] limit order failed: {e}")
            return

        order_id = None
        if isinstance(resp, dict):
            order_id = (resp.get("order") or {}).get("order_id") or resp.get("order_id")

        self._pending_limits[pos_key] = {
            "order_id":    order_id,
            "ticker":      ticker,
            "side":        side,
            "price_cents": price_cents,
            "count":       count,
            "placed_at":   time.time(),
        }
        log.info(f"[PREDICTIVE] limit resting — order_id={order_id}")

    async def cancel_stale_limits(self, current_yes_price: float, max_age_secs: float = 8.0):
        """
        Cancel any pending limit orders that are now stale:
          - older than max_age_secs (point was played, order didn't fill)
          - or the market moved far enough away that the edge is gone

        Call this at the top of each tick, before placing new limits.
        """
        now = time.time()
        for pos_key, pending in list(self._pending_limits.items()):
            age = now - pending["placed_at"]
            price_moved = abs(current_yes_price - pending["price_cents"] / 100.0)

            if age >= max_age_secs or price_moved >= 0.05:
                order_id = pending.get("order_id")
                if order_id:
                    await self.kalshi.cancel_order(order_id)
                del self._pending_limits[pos_key]
                log.info(
                    f"[PREDICTIVE] cancelled stale limit {pos_key} "
                    f"(age={age:.1f}s moved={price_moved*100:.1f}¢)"
                )

    async def _attempt_buy(
        self,
        market: dict,
        token_key: str,         # "yes" | "no"
        edge: float,
        model_prob: float,
        market_price: float,
        pipeline_start: float = None,
        kelly_mult: float = 1.0,
        flow_signal: MarketSignal = None,
        total_balance: float = None,
        signal_context: dict = None,
        kelly_mult_conv: float = 1.0,
    ):
        # Don't double-dip on same side in same market (fast check before lock)
        pos_key = f"{market['id']}_{token_key}"
        if pos_key in self._open_positions:
            log.debug(f"Already holding {token_key.upper()} position — skipping.")
            return

        # Cooldown: don't re-enter within 90s of a sell on this side
        import time as _time
        last_sell = self._sell_cooldown.get(pos_key, 0.0)
        cooldown_secs = 90.0
        elapsed = _time.monotonic() - last_sell
        if elapsed < cooldown_secs:
            log.debug(f"Cooldown active for {pos_key} ({cooldown_secs - elapsed:.0f}s remaining) — skipping.")
            return

        # Exposure check (fast check before lock)
        if self._total_bet_usdc >= self.cfg.MAX_GAME_EXPOSURE:
            log.warning(f"Max game exposure (${self._total_bet_usdc:.2f}) reached — no new bets.")
            return

        # Serialize balance-check + order-placement so concurrent workers don't
        # each see the full balance and all submit orders that exceed collateral.
        async with self._buy_lock:
            # Re-check inside lock in case another worker just placed an order
            if pos_key in self._open_positions:
                log.debug(f"Already holding {token_key.upper()} position (post-lock) — skipping.")
                return
            if self._total_bet_usdc >= self.cfg.MAX_GAME_EXPOSURE:
                log.warning(f"Max game exposure reached (post-lock) — no new bets.")
                return

            await self._attempt_buy_locked(
                market=market,
                token_key=token_key,
                pos_key=pos_key,
                edge=edge,
                model_prob=model_prob,
                market_price=market_price,
                pipeline_start=pipeline_start,
                kelly_mult=kelly_mult,
                flow_signal=flow_signal,
                total_balance=total_balance,
                signal_context=signal_context,
                kelly_mult_conv=kelly_mult_conv,
            )

    async def _attempt_buy_locked(
        self,
        market: dict,
        token_key: str,
        pos_key: str,
        edge: float,
        model_prob: float,
        market_price: float,
        pipeline_start: float = None,
        kelly_mult: float = 1.0,
        flow_signal: MarketSignal = None,
        total_balance: float = None,
        signal_context: dict = None,
        kelly_mult_conv: float = 1.0,
    ):
        """Called only while _buy_lock is held — balance check and order are atomic."""
        # Kelly sizing — use pre-fetched balance if available to avoid redundant API call.
        if total_balance is None:
            total_balance = await self.kalshi.get_balance()

        # Subtract collateral locked in positions we've already opened
        locked_collateral = sum(p.get("cost_usdc", 0.0) for p in self._open_positions.values())
        available_balance = max(0.0, total_balance - locked_collateral)

        log.info(
            f"  Balance: ${total_balance:.2f} total  "
            f"${locked_collateral:.2f} locked  "
            f"${available_balance:.2f} free"
        )

        if available_balance < self.cfg.MIN_BET_USDC:
            log.warning(f"Insufficient free balance (${available_balance:.2f}) — skipping.")
            return

        size_usdc = self._kelly_size(
            model_prob, market_price,
            available_balance=available_balance,
            kelly_mult=kelly_mult,
        )
        if size_usdc < self.cfg.MIN_BET_USDC:
            log.info(f"Kelly size ${size_usdc:.2f} below minimum — skipping.")
            return

        # Slippage guard: Kalshi limit orders use integer cents.
        price_cents = int(market_price * 100)
        slippage_limit_cents = min(99, int(price_cents * (1 + self.cfg.MAX_SLIPPAGE)))

        # Calculate contracts from Kelly allocation
        count = int(size_usdc / market_price)

        # Hard cap: never spend more than 85% of free balance (leaves cushion for fees/slippage)
        max_contracts_possible = int((available_balance * 0.85) / (slippage_limit_cents / 100.0))
        if count > max_contracts_possible:
            log.warning(
                f"Downsizing order from {count} to {max_contracts_possible} "
                f"(free balance ${available_balance:.2f})."
            )
            count = max_contracts_possible

        if count <= 0:
            log.info("Contract count is 0 after collateral cap — skipping.")
            return

        ticker = market["id"]

        flow_label = f" [{flow_signal.direction.value}]" if flow_signal else ""
        log.info(f"→ PLACING BUY: {token_key.upper()} {count} contracts @ {price_cents}c "
                 f"(edge={edge:+.3f}, kelly_alloc=${size_usdc:.2f}{flow_label})")

        if pipeline_start:
            log.info(f"[LATENCY] Initiating BUY order {(time.time() - pipeline_start)*1000:.2f}ms after video ingest.")

        try:
            resp = await self.kalshi.buy(
                ticker=ticker,
                price_cents=slippage_limit_cents,
                count=count,
                side=token_key
            )
        except Exception as e:
            err = str(e)
            if "503" in err or "service_unavailable" in err:
                log.warning(f"Kalshi Demo Engine offline (503). Simulating BUY order for {count} contracts locally.")
                resp = {"status": "simulated_success"}
            elif "insufficient_balance" in err:
                log.warning(f"Kalshi rejected order: insufficient balance. Skipping {ticker}.")
                return
            elif "market_not_found" in err:
                log.warning(f"Kalshi rejected order: market not found (likely settled). Skipping {ticker}.")
                return
            else:
                raise

        # Check how many contracts actually filled. Kalshi wraps fill data inside
        # resp["order"] on the v2 API — fall back to top-level keys for older wrappers.
        filled = (
            (resp.get("order") or {}).get("taker_fill_count")
            or resp.get("taker_fill_count")
            or resp.get("maker_fill_count")
            or count
        )
        if isinstance(filled, int) and filled <= 0 and resp.get("status") not in ("simulated_success",):
            log.warning(f"Order accepted but 0 contracts filled (resting order) — not tracking as open position.")
            return

        if pipeline_start:
            log.info(f"[LATENCY E2E] Kalshi order execution confirmed! Total Latency: {(time.time() - pipeline_start)*1000:.2f}ms")

        # Use actual filled count if available
        filled_count = filled if isinstance(filled, int) and filled > 0 else count
        actual_cost = filled_count * market_price
        self._total_bet_usdc += actual_cost

        self._open_positions[pos_key] = {
            "ticker":          ticker,
            "token_key":       token_key,
            "entry_price":     market_price,
            "peak_price":      market_price,     # trailing high-water mark
            "min_hold_until":  time.time() + 60, # don't exit for first 60s
            "tranches_sold":   0,                # how many 25% tranches sold
            "count":           filled_count,
            "cost_usdc":       actual_cost,
            "model_prob":      model_prob,
            "edge":            edge,
            "order_group_id":  None,             # set when OCO group is created
        }
        log.info(f"[FILLED] {token_key.upper()} {filled_count} contracts @ {market_price:.2f} on {ticker}")
        self._trade_log.append({"action": "BUY", **self._open_positions[pos_key], "resp": resp})

        if self.tracker is not None:
            sc = signal_context or {}
            _fee_rate = getattr(self.cfg, "KALSHI_FEE_RATE", 0.07)
            _b = (1.0 / market_price) - 1.0 if market_price > 0 else 0.0
            _kelly_f_raw = max(0.0, (model_prob * _b - (1.0 - model_prob)) / _b) if _b > 0 else 0.0
            _su  = sc.get("score_update") or {}
            _pts = _su.get("points", (0, 0))
            _psa = sc.get("player_stats_a") or {}
            _psb = sc.get("player_stats_b") or {}
            _ven = sc.get("venue_data")
            _sun = sc.get("sun_data")
            _wx  = sc.get("weather_data")
            _fs  = flow_signal
            try:
                _trade_id = self.tracker.record_entry(
                    ticker=ticker,
                    player_a=market.get("player_a", ""),
                    player_b=market.get("player_b", ""),
                    betting_on=token_key,
                    yes_price_at_bet=market.get("yes_price", market_price),
                    no_price_at_bet=market.get("no_price", 1.0 - market_price),
                    model_prob_for_side=model_prob,
                    market_price_for_side=market_price,
                    edge_gross=edge + self._fee_per_contract(market_price, _fee_rate),
                    fee_estimate=self._fee_per_contract(market_price, _fee_rate),
                    edge_net=edge,
                    kelly_fraction_raw=_kelly_f_raw,
                    kelly_mult_convergence=kelly_mult_conv,
                    kelly_mult_adaptive=kelly_mult,
                    kelly_mult_combined=kelly_mult,
                    stake_usdc=actual_cost,
                    contracts_filled=filled_count,
                    entry_price_actual=market_price,
                    order_id=str(((resp.get("order") or {}).get("order_id")) or ""),
                    latency_ms=(time.time() - pipeline_start) * 1000 if pipeline_start else 0.0,
                    model_prob_ml_base=float(sc.get("model_prob_ml_base", float("nan"))),
                    model_prob_nn=float(sc.get("model_prob_nn", float("nan"))),
                    model_prob_xgb=float(sc.get("model_prob_xgb", float("nan"))),
                    logit_adj_age_temp=float(sc.get("logit_adj_age_temp", 0.0)),
                    model_prob_after_age_temp=float(sc.get("model_prob_after_age_temp", float("nan"))),
                    logit_adj_phys=float(sc.get("logit_adj_phys", 0.0)),
                    model_prob_after_phys=float(sc.get("model_prob_after_phys", float("nan"))),
                    model_prob_final_at_bet=float(sc.get("model_prob_final", float("nan"))),
                    pts_vs_rank_edge=float(sc.get("pts_vs_rank_edge", 0.0)),
                    serve_divergence=float(sc.get("serve_divergence", 0.0)),
                    markov_p_serve_initial=float(sc.get("markov_p_serve_initial", float("nan"))),
                    markov_p_serve_at_bet=float(sc.get("markov_p_serve_now", float("nan"))),
                    markov_p_return_at_bet=float(sc.get("markov_p_return_now", float("nan"))),
                    bayesian_posterior_at_bet=float(sc.get("bayesian_posterior", float("nan"))),
                    bayes_uncertainty=float(sc.get("bayes_uncertainty", float("nan"))),
                    pts_vs_rank_raw=float(sc.get("pts_vs_rank_raw", 0.0)),
                    alt_x_ht_component=float(sc.get("alt_x_ht_comp", 0.0)),
                    alt_x_age_component=float(sc.get("alt_x_age_comp", 0.0)),
                    lh_hard_component=float(sc.get("lh_hard_comp", 0.0)),
                    lh_clay_component=float(sc.get("lh_clay_comp", 0.0)),
                    pts_rank_component=float(sc.get("pts_rank_comp", 0.0)),
                    lh_net=float(sc.get("lh_net", 0.0)),
                    tournament=str(_su.get("tournament", "")),
                    venue_city=str(getattr(_ven, "city", "")),
                    venue_country=str(getattr(_ven, "country", "")),
                    court_surface=str(getattr(_ven, "court_surface", "")),
                    altitude_m=float(getattr(_ven, "altitude_m", 0.0)),
                    temp_celsius=float(getattr(_wx, "temperature_c", float("nan"))) if _wx else float("nan"),
                    apparent_temp_c=float(getattr(_wx, "apparent_temperature_c", float("nan"))) if _wx else float("nan"),
                    humidity_pct=float(getattr(_wx, "relative_humidity_pct", float("nan"))) if _wx else float("nan"),
                    wind_speed_kmh=float(getattr(_wx, "wind_speed_kmh", float("nan"))) if _wx else float("nan"),
                    wind_direction_deg=float(getattr(_wx, "wind_direction_deg", float("nan"))) if _wx else float("nan"),
                    precipitation_mm=float(getattr(_wx, "precipitation_mm", float("nan"))) if _wx else float("nan"),
                    sun_azimuth_deg=float(getattr(_sun, "azimuth_deg", float("nan"))) if _sun else float("nan"),
                    sun_elevation_deg=float(getattr(_sun, "elevation_deg", float("nan"))) if _sun else float("nan"),
                    sun_glare_active=bool(getattr(_sun, "glare_active", False)) if _sun else False,
                    sun_description=str(getattr(_sun, "description", "")) if _sun else "",
                    p1_ranking=int(_psa.get("ranking", 0) or 0),
                    p1_rank_points=float(_psa.get("elo", 0.0) or 0.0),
                    p1_age=float(_psa.get("age", float("nan"))),
                    p1_height_cm=float(_psa.get("height_cm", float("nan"))),
                    p1_hand=str(_psa.get("hand", "")),
                    p1_win_rate=float(_psa.get("win_rate", float("nan"))),
                    p1_season_wins=_psa.get("season_wins"),
                    p1_season_losses=_psa.get("season_losses"),
                    p1_aces_per_match=_psa.get("aces_per_match"),
                    p1_bp_conversion_pct=_psa.get("bp_conversion_pct"),
                    p1_first_serve_pct=_psa.get("first_serve_pct"),
                    p2_ranking=int(_psb.get("ranking", 0) or 0),
                    p2_rank_points=float(_psb.get("elo", 0.0) or 0.0),
                    p2_age=float(_psb.get("age", float("nan"))),
                    p2_height_cm=float(_psb.get("height_cm", float("nan"))),
                    p2_hand=str(_psb.get("hand", "")),
                    p2_win_rate=float(_psb.get("win_rate", float("nan"))),
                    p2_season_wins=_psb.get("season_wins"),
                    p2_season_losses=_psb.get("season_losses"),
                    p2_aces_per_match=_psb.get("aces_per_match"),
                    p2_bp_conversion_pct=_psb.get("bp_conversion_pct"),
                    p2_first_serve_pct=_psb.get("first_serve_pct"),
                    rank_diff=float(int(_psa.get("ranking", 0) or 0) - int(_psb.get("ranking", 0) or 0)),
                    age_diff=float(sc.get("age_diff_cached", 0.0)),
                    height_diff_cm=float(sc.get("height_diff_cached", 0.0)),
                    atp_live_ticks_elapsed=int(sc.get("atp_tick", 0)),
                    score_points_a=int(_pts[0]) if isinstance(_pts, (list, tuple)) and len(_pts) > 0 else 0,
                    score_points_b=int(_pts[1]) if isinstance(_pts, (list, tuple)) and len(_pts) > 1 else 0,
                    p1_serving_at_bet=_su.get("p1_serving"),
                    total_live_ticks=int(sc.get("atp_tick", 0)),
                    flow_direction=(_fs.direction.value if _fs and _fs.direction else "NEUTRAL"),
                    flow_velocity_cents_per_s=float(getattr(_fs, "price_velocity", 0.0) * 100),
                    flow_z_score=float(getattr(_fs, "z_score", 0.0)),
                    flow_vol_regime=str(getattr(_fs, "vol_regime", "")),
                )
                self._open_positions[pos_key]["trade_id"] = _trade_id
            except Exception as _te:
                log.warning("[TRACKER] record_entry failed: %s", _te)

    # ──────────────────────────────────────────────────────────────────────────
    # Exit logic
    # ──────────────────────────────────────────────────────────────────────────

    async def _check_exits(
        self,
        market: dict,
        win_prob: dict,
        market_monitor: "MarketMonitor" = None,
    ):
        """
        Smart exit system with:
          1. Minimum hold period  — no exits in first N seconds
          2. Trailing stop        — sell if price drops X% off peak
          3. Volatility fast exit — in HIGH vol, exit any +10% gain immediately
          4. Tiered scale-out     — partial sells at 2x, 3x, 4x entry price
          5. Model reversal guard — exit if Markov says we're now wrong
          6. Near-resolution exit — lock in value at 92¢+
        """
        import time as _t
        now = _t.time()
        market_id = market.get("id", "")

        # Get volatility regime from MarketMonitor (or default LOW)
        vol_regime   = market_monitor.current_signal.vol_regime if market_monitor else "LOW"
        vol_label    = f"[{vol_regime} VOL]"

        # Regime-specific thresholds
        if vol_regime == "HIGH":
            trail_pct        = 0.10    # 10% trailing stop below peak
            min_hold_secs    = 15.0    # very short hold — take profit fast
            fast_exit_gain   = 0.10    # exit ALL at 10% gain in high vol
        elif vol_regime == "MEDIUM":
            trail_pct        = 0.15
            min_hold_secs    = 30.0
            fast_exit_gain   = None    # no fast exit in medium vol
        else:  # LOW
            trail_pct        = 0.25
            min_hold_secs    = 60.0
            fast_exit_gain   = None

        for pos_key, pos in list(self._open_positions.items()):
            if pos.get("ticker") != market_id and not pos_key.startswith(market_id):
                continue

            token_key = pos["token_key"]
            entry_price = pos.get("entry_price", 0.50)

            current_price = (
                market.get("yes_price", entry_price)
                if token_key == "yes"
                else market.get("no_price", entry_price)
            )
            model_prob    = win_prob.get("team_a", 0.5) if token_key == "yes" else win_prob.get("team_b", 0.5)
            current_edge  = model_prob - current_price
            gain          = current_price - entry_price

            # ── 0. Update peak price ────────────────────────────────────────
            if current_price > pos.get("peak_price", entry_price):
                pos["peak_price"] = current_price
            peak_price = pos["peak_price"]

            # ── 1. Minimum hold period ──────────────────────────────────────
            if now < pos.get("min_hold_until", 0):
                remaining = pos["min_hold_until"] - now
                log.debug(f"  Min hold: {remaining:.0f}s remaining on {pos_key}")
                continue

            # ── 2. Near-resolution exit (always sells — highest priority) ───
            if current_price >= 0.92:
                log.info(f"EXIT Near-resolution {vol_label}: {token_key.upper()} @ {current_price:.3f}")
                await self._sell_position(pos_key, pos, current_price, exit_reason="near_resolution")
                continue

            # ── 3. Volatility fast exit ─────────────────────────────────────
            if fast_exit_gain is not None and gain >= fast_exit_gain:
                log.info(
                    f"EXIT VOLATILITY FAST {vol_label}: {token_key.upper()} "
                    f"gain=+{gain*100:.1f}¢ (entry={entry_price:.2f} now={current_price:.2f})"
                )
                await self._sell_position(pos_key, pos, current_price, exit_reason="volatility_fast_exit")
                continue

            # ── 4. Trailing stop (only activates if we're in profit) ────────
            if gain > 0:  # only trail if we've ever been in profit
                trail_stop = peak_price * (1.0 - trail_pct)
                if current_price <= trail_stop:
                    log.info(
                        f"EXIT TRAILING STOP {vol_label}: {token_key.upper()} "
                        f"price={current_price:.3f} ≤ stop={trail_stop:.3f} (peak={peak_price:.3f})"
                    )
                    await self._sell_position(pos_key, pos, current_price, exit_reason="trailing_stop")
                    continue

            # ── 5. Tiered scale-out (partial sells in LOW / MEDIUM vol) ─────
            if vol_regime != "HIGH" and entry_price > 0:
                gain_ratio = current_price / entry_price
                tranches = pos.get("tranches_sold", 0)
                tranche_targets = [
                    (2.0, "2× (100% gain)"),
                    (3.0, "3× (200% gain)"),
                    (4.0, "4× (300% gain)"),
                ]
                for target_ratio, label in tranche_targets:
                    tier = tranche_targets.index((target_ratio, label))
                    if gain_ratio >= target_ratio and tranches == tier:
                        total_count = pos.get("count", 1)
                        sell_count  = max(1, total_count // 4)  # sell 25%
                        log.info(
                            f"SCALE-OUT {vol_label} tranche {tier+1}/3: "
                            f"{token_key.upper()} {sell_count} of {total_count} @ {current_price:.3f} "
                            f"({label})"
                        )
                        await self._sell_partial(pos_key, pos, current_price, sell_count)
                        pos["tranches_sold"] = tranches + 1
                        break

            # ── 6. Model reversal guard ─────────────────────────────────────
            if current_edge < -self.cfg.MODEL_REVERSAL_EXIT_EDGE:
                log.info(
                    f"EXIT MODEL REVERSAL (threshold={self.cfg.MODEL_REVERSAL_EXIT_EDGE:.3f}) "
                    f"{vol_label}: {token_key.upper()} "
                    f"edge={current_edge:+.3f} (model now disagrees)"
                )
                await self._sell_position(pos_key, pos, current_price, exit_reason="model_reversal")
                continue


    async def _sell_position(self, pos_key: str, pos: dict, current_price: float, exit_reason: str = "sell"):
        ticker     = pos["ticker"]
        token_key  = pos["token_key"]

        # Remove from open positions immediately — prevents infinite retry if Kalshi rejects
        self._open_positions.pop(pos_key, None)
        import time as _time
        self._sell_cooldown[pos_key] = _time.monotonic()

        kalshi_pos = self.kalshi.get_position(pos_key)
        count      = kalshi_pos.get("count", 0)

        if count <= 0:
            log.debug(f"No contracts to sell for {pos_key} — already flat.")
            # Still release exposure tracking even if no contracts held
            cost = pos.get("cost_usdc", 0.0)
            self._total_bet_usdc = max(0.0, self._total_bet_usdc - cost)
            return

        price_cents = int(current_price * 100)
        # Sell at 1% slippage buffer (floor)
        sell_limit_cents = max(1, int(price_cents * 0.99))

        try:
            resp = await self.kalshi.sell(ticker=ticker, price_cents=sell_limit_cents, count=count, side=token_key)
        except Exception as e:
            err = str(e)
            if "503" in err or "service_unavailable" in err:
                log.warning(f"Kalshi Demo Engine offline (503). Simulating SELL for {count} contracts locally.")
                resp = {"status": "simulated_success"}
            elif "insufficient_balance" in err:
                # Position may not exist on exchange (from a previous bot run) — just release locally
                log.warning(f"SELL rejected (insufficient balance / no contracts held on exchange). Releasing local position for {pos_key}.")
                resp = {"status": "local_release"}
            else:
                log.warning(f"SELL failed for {pos_key}: {e}. Releasing local position.")
                resp = {"status": "local_release"}

        pnl  = (current_price - pos["entry_price"]) * count
        cost = pos.get("cost_usdc", 0.0)
        roi  = pnl / cost if cost > 0 else 0
        log.info(f"SOLD {token_key.upper()} {count} contracts | PnL: ${pnl:+.2f} (ROI: {roi:.2%})")
        self._trade_log.append({
            "action": "SELL", "pnl": pnl, "roi": roi,
            "model_p": pos["model_prob"], "market_p": pos["entry_price"],
            **pos, "resp": resp
        })
        ml_engine.record_trade_outcome(self._trade_log[-1])
        if self.tracker is not None:
            _tid = pos.get("trade_id")
            if _tid:
                try:
                    self.tracker.record_exit(
                        trade_id=_tid,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        contracts_exited=count,
                        pnl_usdc=pnl,
                    )
                except Exception as _te:
                    log.warning("[TRACKER] record_exit failed: %s", _te)
        # Release exposure tracking
        self._total_bet_usdc = max(0.0, self._total_bet_usdc - cost)

    async def _sell_partial(self, pos_key: str, pos: dict, current_price: float, sell_count: int):
        """Sell a partial number of contracts (for tiered scale-out). Keeps position open."""
        ticker    = pos["ticker"]
        token_key = pos["token_key"]
        total     = pos.get("count", 0)

        if sell_count <= 0 or total <= 0:
            return
        sell_count = min(sell_count, total)

        price_cents      = int(current_price * 100)
        sell_limit_cents = max(1, int(price_cents * 0.99))

        try:
            resp = await self.kalshi.sell(
                ticker=ticker, price_cents=sell_limit_cents,
                count=sell_count, side=token_key
            )
        except Exception as e:
            err = str(e)
            if "503" in err or "service_unavailable" in err:
                resp = {"status": "simulated_success"}
            else:
                log.warning(f"Partial SELL failed for {pos_key}: {e}")
                return

        pnl = (current_price - pos["entry_price"]) * sell_count
        log.info(f"PARTIAL SELL {token_key.upper()} {sell_count}/{total} @ {current_price:.3f} | PnL: ${pnl:+.2f}")
        if self.tracker is not None:
            _tid = pos.get("trade_id")
            if _tid:
                try:
                    self.tracker.record_exit(
                        trade_id=_tid,
                        exit_price=current_price,
                        exit_reason="partial_sell",
                        contracts_exited=sell_count,
                        pnl_usdc=pnl,
                    )
                except Exception as _te:
                    log.warning("[TRACKER] record_exit (partial) failed: %s", _te)

        # Update position — reduce count, don't remove
        remaining = total - sell_count
        pos["count"] = remaining
        partial_cost = pos.get("cost_usdc", 0.0) * (sell_count / total)
        pos["cost_usdc"] = pos.get("cost_usdc", 0.0) - partial_cost
        self._total_bet_usdc = max(0.0, self._total_bet_usdc - partial_cost)
        self._trade_log.append({"action": "PARTIAL_SELL", "pnl": pnl, **pos})

        if remaining <= 0:
            # All contracts sold via tranching — clean up position
            self._open_positions.pop(pos_key, None)
            import time as _time
            self._sell_cooldown[pos_key] = _time.monotonic()

    # ──────────────────────────────────────────────────────────────────────────
    # Kelly criterion
    # ──────────────────────────────────────────────────────────────────────────

    def _kelly_size(
        self,
        p: float,
        market_price: float,
        available_balance: float = 0.0,
        kelly_mult: float = 1.0,
    ) -> float:
        """
        Confidence-Tiered Fractional Kelly Sizing
        ==========================================
        kelly_mult: applied on top of tier fraction.
          - From AdaptiveController: 1.10 (hot), 1.0 (normal), 0.80 (cold), 0.0 (protection)
          - From MarketMonitor:      1.25 (CONFIRM), 1.0 (NEUTRAL) — FADE is blocked upstream
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0
        if kelly_mult <= 0:
            return 0.0

        b = (1.0 / market_price) - 1.0   # net decimal odds
        q = 1.0 - p
        f = (p * b - q) / b               # full Kelly fraction
        if f <= 0:
            return 0.0

        # --- Confidence tier ---
        if p >= 0.70:
            tier_fraction = 0.40
        elif p >= 0.60:
            tier_fraction = 0.25
        elif p >= 0.55:
            tier_fraction = 0.12
        else:
            tier_fraction = 0.05

        effective_fraction = tier_fraction * self.cfg.KELLY_FRACTION * kelly_mult

        bankroll  = available_balance if available_balance > 0 else (self.cfg.MAX_BET_USDC * 4)
        full_size = f * bankroll
        sized     = full_size * effective_fraction

        capped = min(sized, self.cfg.MAX_BET_USDC)
        if available_balance > 0:
            capped = min(capped, available_balance * 0.95)

        result = max(round(capped, 2), 0.0)
        log.info(
            f"  Kelly sizing: p={p:.3f} tier={tier_fraction:.2f} "
            f"f*={f:.4f} mult={kelly_mult:.2f} bankroll=${bankroll:.2f} → ${result:.2f}"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────────

    def print_summary(self):
        print("\n" + "="*60)
        print("TRADE SUMMARY")
        print("="*60)
        total_pnl = sum(t.get("pnl", 0) for t in self._trade_log if t["action"] == "SELL")
        buys  = [t for t in self._trade_log if t["action"] == "BUY"]
        sells = [t for t in self._trade_log if t["action"] == "SELL"]
        print(f"  Total buys  : {len(buys)}")
        print(f"  Total sells : {len(sells)}")
        print(f"  Total bet   : ${self._total_bet_usdc:.2f}")
        print(f"  Realized P&L: ${total_pnl:+.2f}")
        print("="*60 + "\n")

    def get_summary(self) -> dict:
        total_pnl = sum(t.get("pnl", 0) for t in self._trade_log if t["action"] == "SELL")
        buys  = [t for t in self._trade_log if t["action"] == "BUY"]
        sells = [t for t in self._trade_log if t["action"] == "SELL"]
        return {
            "total_buys":  len(buys),
            "total_sells": len(sells),
            "total_bet":   self._total_bet_usdc,
            "realized_pnl": total_pnl,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4: Break-Point State Detection + Order Group OCO Management
    # ──────────────────────────────────────────────────────────────────────────

    def _is_break_point(self, lms) -> bool:
        """
        Detect if the current game is at a break-point (high-leverage) state.

        A break point exists when:
          - The receiver (non-server) leads 40-0, 40-15, 40-30, or Deuce→Ad-out
          - In score terms (points tuple): receiver at advantage in deuce (3-3+)

        This is the highest-leverage point in tennis — the serving player's
        match win probability collapses if they lose the break point.
        Per information.md: trigger fractional divestment BEFORE this point is played.
        """
        if lms is None:
            return False
        try:
            pa, pb = lms.current_game_points
            p1_serving = lms.p1_serving
            # Server is index 0 if p1_serving else index 1
            server_pts   = pa if p1_serving else pb
            receiver_pts = pb if p1_serving else pa
            # Break point: receiver has advantage (40 = 3 in our encoding)
            # Cases: 0-3, 1-3, 2-3, or 3-3 with receiver winning ad (handled by ad logic)
            if receiver_pts >= 3 and receiver_pts > server_pts:
                return True
            # Deuce (3-3) is high-leverage but not technically break point yet
            if server_pts >= 3 and receiver_pts >= 3:
                return True  # treat deuce as high-leverage too
            return False
        except Exception:
            return False

    async def half_take_profit(
        self,
        pos_key: str,
        market: dict,
        _lms=None,
        confidence: float = 0.5,
    ):
        """
        State-aware fractional divestment triggered at break-point scenarios.

        Per information.md:
          - High confidence (hold_rate > 50%): sell 25%, keep 75%
          - Low confidence / coin-flip: sell 50% (half take-profit), lock in baseline

        Uses resting limit orders (not market orders) so we provide liquidity
        and pay 0% fees rather than crossing the spread.

        Args:
            pos_key:    e.g. "KXATPMATCH-..._yes"
            market:     current market dict with yes_price / no_price
            lms:        LiveMatchState (for break-point detection)
            confidence: 0–1, estimated probability server holds (from live ATP stats)
        """
        pos = self._open_positions.get(pos_key)
        if not pos:
            return

        token_key     = pos["token_key"]
        ticker        = pos["ticker"]
        total_count   = pos.get("count", 0)
        current_price = (
            market.get("yes_price") if token_key == "yes" else market.get("no_price")
        ) or pos["entry_price"]

        if total_count <= 0:
            return

        # Determine sell fraction: high confidence → 25% out; low → 50% out
        if confidence >= 0.55:
            sell_fraction = 0.25
            reason = "HIGH confidence (25% scale-out)"
        else:
            sell_fraction = 0.50
            reason = "LOW confidence / coin-flip (50% half take-profit)"

        sell_count = max(1, int(total_count * sell_fraction))
        price_cents = max(1, min(99, int(current_price * 100)))

        log.info(
            f"[HALF-TP] {pos_key} — {reason} | "
            f"selling {sell_count}/{total_count} @ {price_cents}¢"
        )

        # Place resting limit ask (maker order → 0% fee)
        try:
            resp = await self.kalshi.place_limit_order(
                ticker=ticker,
                price_cents=price_cents,
                count=sell_count,
                side=token_key,
            )
        except Exception as e:
            log.warning(f"[HALF-TP] limit sell failed: {e}")
            return

        # Extract order_id to potentially group it
        order_id = None
        if isinstance(resp, dict):
            order_id = (resp.get("order") or {}).get("order_id") or resp.get("order_id")

        # If we placed two orders (main take-profit + contingent), create OCO group
        existing_group = pos.get("order_group_id")
        if order_id and not existing_group:
            try:
                group_resp = await self.kalshi.create_order_group(
                    order_ids=[order_id],
                    contracts_limit=sell_count,
                )
                group_id = (group_resp.get("order_group") or group_resp).get("order_group_id")
                if group_id:
                    pos["order_group_id"] = group_id
                    log.info(f"[HALF-TP] OCO group created: {group_id}")
            except Exception as e:
                log.debug(f"[HALF-TP] order group creation failed: {e}")

        # Update local position accounting
        remaining = max(0, total_count - sell_count)
        pnl = (current_price - pos["entry_price"]) * sell_count
        partial_cost = pos.get("cost_usdc", 0.0) * (sell_count / total_count) if total_count else 0
        pos["count"]     = remaining
        pos["cost_usdc"] = max(0.0, pos.get("cost_usdc", 0.0) - partial_cost)
        self._total_bet_usdc = max(0.0, self._total_bet_usdc - partial_cost)

        self._trade_log.append({
            "action": "HALF_TAKE_PROFIT",
            "pnl": pnl,
            "sell_fraction": sell_fraction,
            "reason": reason,
            **pos,
        })
        log.info(f"[HALF-TP] Done. Remaining: {remaining} contracts. PnL on sold: ${pnl:+.2f}")

        if remaining <= 0:
            self._open_positions.pop(pos_key, None)
            import time as _time
            self._sell_cooldown[pos_key] = _time.monotonic()

    async def emergency_liquidate_group(self, pos_key: str, market: dict):
        """
        Instantly cancel ALL resting take-profit orders in the position's order group.

        Uses PUT /portfolio/order_groups/{id}/trigger — single API call cancels
        everything, far faster than individual DELETEs when adverse news hits.

        Falls back to individual order cancellation if no group exists.
        """
        pos = self._open_positions.get(pos_key)
        if not pos:
            return

        order_group_id = pos.get("order_group_id")
        if order_group_id:
            log.info(f"[EMERGENCY] Triggering (cancelling) order group {order_group_id} for {pos_key}")
            try:
                await self.kalshi.trigger_order_group(order_group_id)
                pos["order_group_id"] = None
                log.info(f"[EMERGENCY] Order group {order_group_id} cancelled successfully.")
            except Exception as e:
                log.warning(f"[EMERGENCY] trigger_order_group failed: {e}")
        else:
            # No group — fall back to selling the whole position at market
            token_key     = pos["token_key"]
            current_price = (
                market.get("yes_price") if token_key == "yes" else market.get("no_price")
            ) or pos["entry_price"]
            log.info(f"[EMERGENCY] No order group for {pos_key} — selling full position")
            await self._sell_position(pos_key, pos, current_price)

    async def check_breakpoint_exits(self, market: dict, lms=None, live_hold_rate: float = 0.5):
        """
        Called once per tick (in main.py process_match loop) when score is live.

        Detects break-point scenarios and triggers fractional divestment BEFORE
        the critical point is played, converting a directional bet into a
        volatility-harvesting operation.

        Per information.md: "A smart take-profit algorithm must mathematically
        acknowledge this severe inflection point and react preemptively."

        Args:
            market:          current market dict
            lms:             LiveMatchState instance
            live_hold_rate:  empirical serve hold rate from live ATP stats (0–1)
        """
        if not self._open_positions:
            return

        is_bp = self._is_break_point(lms)
        if not is_bp:
            return

        for pos_key, pos in list(self._open_positions.items()):
            market_id = market.get("id", "")
            if not pos_key.startswith(market_id):
                continue

            # Skip if min_hold period hasn't elapsed
            if time.time() < pos.get("min_hold_until", 0):
                continue

            # Skip if we've already done a half take-profit this hold
            if pos.get("half_tp_done"):
                continue

            # Only act if we're in profit (ensure we have something to protect)
            token_key = pos["token_key"]
            current_price = (
                market.get("yes_price") if token_key == "yes" else market.get("no_price")
            ) or pos["entry_price"]
            gain = current_price - pos["entry_price"]
            if gain <= 0:
                continue

            log.info(
                f"[BREAK-POINT] High-leverage state detected for {pos_key}. "
                f"Gain={gain*100:+.1f}¢ hold_rate={live_hold_rate:.2f}"
            )

            await self.half_take_profit(
                pos_key=pos_key,
                market=market,
                _lms=lms,
                confidence=live_hold_rate,
            )
            pos["half_tp_done"] = True   # prevent re-triggering this game

