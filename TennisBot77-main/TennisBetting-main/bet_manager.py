"""
bet_manager.py — Bet sizing, edge detection, and position management.

Kelly criterion sizing + edge threshold gating.
Manages buy/sell decisions based on Elo-derived win probability vs market price.
"""

import asyncio
import logging
import time
from kalshi_client import KalshiClient
from config import Config

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
import plotly.graph_objects as go

log = logging.getLogger(__name__)


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
    def __init__(self, kalshi: KalshiClient, config: Config):
        self.kalshi  = kalshi
        self.cfg     = config
        self._total_bet_usdc = 0.0
        self._trade_log = []
        self._open_positions: dict = {}  # token_id → entry data
        self.surface_model = None  # Store RBFInterpolator
        self._buy_lock = asyncio.Lock()  # Prevent race conditions on concurrent balance checks


    # ──────────────────────────────────────────────────────────────────────────
    # Main decision loop
    # ──────────────────────────────────────────────────────────────────────────

    async def evaluate_and_act(
        self,
        market: dict,
        win_prob: dict,
        game_state,        # reserved for future in-game state
        event: dict,       # reserved for event metadata
        pipeline_start: float = None,
    ) -> dict:
        """
        Called after every live event.
        1. Compute edge on YES and NO sides
        2. Pick the HIGHEST edge side above MIN_EDGE threshold (not just YES-first)
        3. If we hold a position that's now overpriced → sell
        Returns a dict with 'profit' key so the HF auto-switch works correctly.
        """
        if not market.get("active", True):
            log.info("Market inactive — skipping.")
            return {"profit": 0.0}

        # Market prices
        yes_price = market["yes_price"]   # cost of 1 YES share
        no_price  = market["no_price"]    # cost of 1 NO share

        # Our model probabilities
        p_a = win_prob["team_a"]   # probability team A (home/YES) wins
        p_b = win_prob["team_b"]

        # Edge = model probability - market implied probability
        edge_yes = p_a - yes_price
        edge_no  = p_b - no_price

        log.info(f"  Market: YES={yes_price:.3f} NO={no_price:.3f} | "
                 f"Model: A={p_a:.3f} B={p_b:.3f} | "
                 f"Edge: YES={edge_yes:+.3f} NO={edge_no:+.3f}")
        
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
        # Capture realized profit from any exits this cycle
        sells_before = len([t for t in self._trade_log if t["action"] == "SELL"])
        await self._check_exits(market, win_prob)
        sells_after  = len([t for t in self._trade_log if t["action"] == "SELL"])
        cycle_profit = sum(
            t.get("pnl", 0.0)
            for t in self._trade_log[-(sells_after - sells_before):]
            if t["action"] == "SELL"
        ) if sells_after > sells_before else 0.0

        # --- Entry logic ---
        min_roi_threshold = 0.04

        def should_bet_meta(model_p, market_price, edge_thresh):
            if edge_thresh < self.cfg.MIN_EDGE:
                return False, f"Edge too low ({edge_thresh:+.3f} < {self.cfg.MIN_EDGE})"

            # Extreme-odds guard: skip any side priced outside the safe band.
            if market_price < self.cfg.EXTREME_ODDS_MIN or market_price > self.cfg.EXTREME_ODDS_MAX:
                return False, f"Odds outside safe band [{self.cfg.EXTREME_ODDS_MIN:.2f}, {self.cfg.EXTREME_ODDS_MAX:.2f}]"

            # Divergence guard
            divergence = abs(model_p - market_price)
            if divergence > self.cfg.MAX_MODEL_DIVERGENCE:
                return False, f"Miscalibrated divergence {divergence:.2f} > {self.cfg.MAX_MODEL_DIVERGENCE}"

            if self.surface_model is not None:
                predicted_roi = self.surface_model([[model_p, market_price]])[0]
                if predicted_roi < min_roi_threshold:
                    return False, f"Surface ROI {predicted_roi:.2f} < {min_roi_threshold}"
            return True, ""

        should_yes, reason_yes = should_bet_meta(p_a, yes_price, edge_yes)
        should_no,  reason_no  = should_bet_meta(p_b, no_price,  edge_no)

        # BUG FIX: Pick the SIDE with the HIGHER edge, not just YES-first.
        # The old `elif should_no` meant if YES had any edge at all (even tiny),
        # NO was never considered — even if it had far better edge.
        if should_yes and should_no:
            # Both sides qualify — pick the higher edge
            if edge_yes >= edge_no:
                should_no = False
                log.debug(f"Both sides qualify — choosing YES (edge={edge_yes:+.3f} > NO edge={edge_no:+.3f})")
            else:
                should_yes = False
                log.debug(f"Both sides qualify — choosing NO (edge={edge_no:+.3f} > YES edge={edge_yes:+.3f})")

        if should_yes:
            await self._attempt_buy(
                market=market,
                token_key="yes",
                edge=edge_yes,
                model_prob=p_a,
                market_price=yes_price,
                pipeline_start=pipeline_start,
            )
        elif should_no:
            await self._attempt_buy(
                market=market,
                token_key="no",
                edge=edge_no,
                model_prob=p_b,
                market_price=no_price,
                pipeline_start=pipeline_start,
            )
        else:
            log.info(f"  Skipping: YES:({reason_yes}) | NO:({reason_no})")

        return {"profit": cycle_profit}

    # ──────────────────────────────────────────────────────────────────────────
    # Buy logic
    # ──────────────────────────────────────────────────────────────────────────

    async def _attempt_buy(
        self,
        market: dict,
        token_key: str,         # "yes" | "no"
        edge: float,
        model_prob: float,
        market_price: float,
        pipeline_start: float = None,
    ):
        # Don't double-dip on same side in same market (fast check before lock)
        pos_key = f"{market['id']}_{token_key}"
        if pos_key in self._open_positions:
            log.debug(f"Already holding {token_key.upper()} position — skipping.")
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
    ):
        """Called only while _buy_lock is held — balance check and order are atomic."""
        # Kelly sizing with actual balance check
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

        size_usdc = self._kelly_size(model_prob, market_price, available_balance=available_balance)
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

        log.info(f"→ PLACING BUY: {token_key.upper()} {count} contracts @ {price_cents}c "
                 f"(slippage_ceil={slippage_limit_cents}c, edge={edge:+.3f}, kelly_alloc=${size_usdc:.2f})")

        if pipeline_start:
            log.info(f"[LATENCY] Initiating BUY order {(time.time() - pipeline_start)*1000:.2f}ms after video ingest.")

        try:
            resp = await self.kalshi.buy(
                ticker=ticker,
                price_cents=price_cents,          # BUG FIX: use actual market price, not slippage ceiling
                count=count,
                side=token_key
            )
            log.info(f"✅ BUY confirmed: {token_key.upper()} {count}x @ {price_cents}c | resp={resp.get('status', resp)}")
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

        if pipeline_start:
            log.info(f"[LATENCY E2E] Kalshi order execution confirmed! Total Latency: {(time.time() - pipeline_start)*1000:.2f}ms")

        # Each contract costs market_price USD
        actual_cost = count * market_price
        self._total_bet_usdc += actual_cost
        
        self._open_positions[pos_key] = {
            "ticker":      ticker,
            "token_key":   token_key,
            "entry_price": market_price,
            "count":       count,
            "cost_usdc":   actual_cost,
            "model_prob":  model_prob,
            "edge":        edge,
        }
        self._trade_log.append({"action": "BUY", **self._open_positions[pos_key], "resp": resp})

    # ──────────────────────────────────────────────────────────────────────────
    # Exit logic
    # ──────────────────────────────────────────────────────────────────────────

    async def _check_exits(self, market: dict, win_prob: dict):
        """
        Exit a position when:
          - Our model now agrees with the market (edge gone)
          - Edge has REVERSED (model now says opposite) — take profit early
          - Market has drifted to within 5% of resolution (0.95+)
        """
        for pos_key, pos in list(self._open_positions.items()):
            token_key   = pos["token_key"]
            current_market_price = market.get(f"{token_key}_price",
                                              market["yes_price"])
            model_prob  = win_prob["team_a"] if token_key == "yes" else win_prob["team_b"]
            current_edge = model_prob - current_market_price

            # Take profit if price moved 10%+ in our favor
            price_gain = current_market_price - pos["entry_price"]
            if price_gain >= 0.10:
                log.info(f"TAKE PROFIT: {token_key.upper()} gained {price_gain:.3f}")
                await self._sell_position(pos_key, pos, current_market_price)
                continue

            # Cut position if edge reversed significantly
            if current_edge < -self.cfg.MIN_EDGE:
                log.info(f"EDGE REVERSED on {token_key.upper()}: {current_edge:+.3f} — exiting.")
                await self._sell_position(pos_key, pos, current_market_price)
                continue

            # Near-resolution exit (lock in value)
            if current_market_price >= 0.92:
                log.info(f"Near-resolution exit: {token_key.upper()} @ {current_market_price:.3f}")
                await self._sell_position(pos_key, pos, current_market_price)

    async def _sell_position(self, pos_key: str, pos: dict, current_price: float):
        ticker     = pos["ticker"]
        token_key  = pos["token_key"]
        
        # BUG FIX: Use the count we recorded when we placed the buy, not the
        # KalshiClient's local tracker (which returns 0 in DRY_RUN mode and can
        # miss partial fills). The position dict is the single source of truth.
        count = pos.get("count", 0)
        if count <= 0:
            log.warning(f"Sell skipped for {pos_key}: recorded count={count}")
            self._open_positions.pop(pos_key, None)
            return

        price_cents = int(current_price * 100)
        # Sell at 1% slippage buffer (floor)
        sell_limit_cents = max(1, int(price_cents * 0.99))
        
        try:
            resp = await self.kalshi.sell(ticker=ticker, price_cents=sell_limit_cents, count=count, side=token_key)
        except Exception as e:
            if "503" in str(e) or "service_unavailable" in str(e):
                log.warning(f"Kalshi Demo Engine offline (503). Simulating SELL order for {count} contracts locally.")
                resp = {"status": "simulated_success"}
            else:
                raise
        
        pnl  = (current_price - pos["entry_price"]) * count
        cost = pos["cost_usdc"]
        roi = pnl / cost if cost > 0 else 0
        log.info(f"SOLD {token_key.upper()} {count} contracts | PnL: ${pnl:+.2f} (ROI: {roi:.2%})")
        self._trade_log.append({
            "action": "SELL", "pnl": pnl, "roi": roi, 
            "model_p": pos["model_prob"], "market_p": pos["entry_price"],
            **pos, "resp": resp
        })
        self._open_positions.pop(pos_key, None)

    # ──────────────────────────────────────────────────────────────────────────
    # Kelly criterion
    # ──────────────────────────────────────────────────────────────────────────

    def _kelly_size(self, p: float, market_price: float, available_balance: float = 0.0) -> float:
        """
        Full Kelly: f* = (p * b - q) / b
        where b = (1 / market_price) - 1  (decimal odds)
              q = 1 - p
        Scaled by KELLY_FRACTION for safety.
        Capped at MAX_BET_USDC and available balance.
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        b   = (1.0 / market_price) - 1.0   # decimal odds
        q   = 1.0 - p
        f   = (p * b - q) / b               # full Kelly fraction of bankroll

        # Use actual balance as bankroll base if provided, otherwise proxy
        bankroll = available_balance if available_balance > 0 else (self.cfg.MAX_BET_USDC * 4)
        
        full_size = f * bankroll
        sized     = full_size * self.cfg.KELLY_FRACTION
        
        # Absolute caps
        capped    = min(sized, self.cfg.MAX_BET_USDC)
        if available_balance > 0:
            capped = min(capped, available_balance * 0.95) # 5% safety margin
            
        return max(round(capped, 2), 0.0)

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

