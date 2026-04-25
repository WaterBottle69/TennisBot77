"""
market_monitor.py — Kalshi Order Flow Tracker
==============================================
Runs as a background async task per active market.
Polls the Kalshi order book every ~5 seconds and emits a MarketSignal:

  CONFIRM  → price moving in same direction as our edge   → full Kelly
  NEUTRAL  → price stable                                 → standard Kelly
  FADE     → price moving against our edge               → skip (market knows more)

Usage:
    monitor = MarketMonitor(kalshi_client, ticker="KXATPMATCH-...")
    asyncio.create_task(monitor.run())
    ...
    signal = monitor.current_signal
"""

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class FlowDirection(Enum):
    CONFIRM = "CONFIRM"   # price moving with our edge
    NEUTRAL = "NEUTRAL"   # price stable, no strong signal
    FADE    = "FADE"      # price moving against our edge


@dataclass
class MarketSignal:
    direction:          FlowDirection = FlowDirection.NEUTRAL
    price_velocity:     float = 0.0   # ¢/s YES price change (positive = rising)
    price_volatility:   float = 0.0   # std dev of price changes in window (proxy for choppiness)
    vol_regime:         str   = "LOW" # "LOW" | "MEDIUM" | "HIGH"
    volume_imbalance:   float = 0.5   # 0..1, >0.5 = more YES pressure
    spread:             float = 0.10  # ask - bid (proxy for liquidity)
    yes_price:          float = 0.50  # latest yes price (0..1)
    timestamp:          float = field(default_factory=time.time)
    z_score:            float = 0.0   # Z-score of (P_model - P_market) over rolling window
    delta_mean:         float = 0.0   # rolling mean of probability delta

    @property
    def kelly_multiplier(self) -> float:
        """Return a stake multiplier based on signal quality."""
        if self.direction == FlowDirection.CONFIRM:
            return 1.25   # boost 25% when flow confirms
        if self.direction == FlowDirection.FADE:
            return 0.0    # block entirely when flow fades
        return 1.0        # standard


class MarketMonitor:
    """
    Continuously polls Kalshi for a single market's price + volume,
    tracks velocity over a 30s window, and emits a MarketSignal.
    """

    POLL_INTERVAL  = 5.0    # seconds between polls
    VELOCITY_WINDOW = 30.0  # seconds of history for trend calc
    VELOCITY_THRESHOLD = 0.003  # 0.3¢/s sustained move to register as trend

    def __init__(self, kalshi_client, ticker: str):
        self.kalshi  = kalshi_client
        self.ticker  = ticker

        # Rolling deque: (timestamp, yes_price)
        self._price_history: deque = deque(maxlen=20)
        self._running = False
        self.current_signal = MarketSignal()

        # Z-score tracking: rolling window of (P_model - P_market) deltas
        self._delta_history: deque = deque(maxlen=30)
        self._model_prob: float = 0.5   # updated by caller via set_model_prob()

    async def run(self):
        """Background loop. Connects to WebSocket stream and blocks."""
        self._running = True
        log.info(f"[FLOW] MarketMonitor started for {self.ticker} via WebSocket")
        try:
            # This will block and continuously call _on_ws_update as messages arrive
            await self.kalshi.stream_orderbook(self.ticker, self._on_ws_update)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
        log.info(f"[FLOW] MarketMonitor stopped for {self.ticker}")

    def stop(self):
        self._running = False

    def set_model_prob(self, p_model: float):
        """Update the Markov model probability for Z-score computation.
        Call this every time win_probability() is recalculated in main.py.
        """
        self._model_prob = max(0.01, min(0.99, p_model))

    async def _on_ws_update(self, msg: dict):
        """Callback for Kalshi WebSocket orderbook messages."""
        msg_type = msg.get("type", "")
        # msg structure can vary between V1/V2, but Kalshi V2 puts payload in msg["msg"]
        payload = msg.get("msg", msg)
        
        bids = payload.get("bids", [])
        asks = payload.get("asks", [])
        
        # Kalshi usually sends arrays of [price_cents, quantity]
        # Bids are what people are willing to pay for YES (we can sell to them).
        # Asks are what people are selling YES for (we can buy from them).
        
        # To get the top of book YES price, we look at the lowest ask (if we want to buy)
        # or highest bid (if we want to sell). We'll take the mid-market or lowest ask.
        yes_ask_cents = min((p[0] for p in asks), default=None)
        yes_bid_cents = max((p[0] for p in bids), default=None)

        if yes_ask_cents is None and yes_bid_cents is None:
            return  # Empty orderbook update

        # Default fallback logic if one side of the book is empty
        if yes_ask_cents is not None and yes_bid_cents is not None:
            yes_price = (yes_ask_cents + yes_bid_cents) / 2.0 / 100.0
            spread = abs(yes_ask_cents - yes_bid_cents) / 100.0
        elif yes_ask_cents is not None:
            yes_price = yes_ask_cents / 100.0
            spread = 0.02
        else:
            yes_price = yes_bid_cents / 100.0
            spread = 0.02

        # Volume imbalance (if we want to approximate from book depth)
        # Sum quantities on both sides
        vol_yes_bid = sum(p[1] for p in bids)
        vol_yes_ask = sum(p[1] for p in asks)
        total_vol = vol_yes_bid + vol_yes_ask
        imbalance = (vol_yes_bid / total_vol) if total_vol > 0 else 0.5

        now = time.time()
        self._price_history.append((now, yes_price))

        # Compute velocity over VELOCITY_WINDOW
        velocity = self._compute_velocity(now)
        volatility = self._compute_volatility(now)
        vol_regime = self._classify_vol(volatility)

        # Classify direction
        direction = self._classify(velocity)

        z_score, delta_mean = self._compute_zscore(yes_price)

        self.current_signal = MarketSignal(
            direction=direction,
            price_velocity=velocity,
            price_volatility=volatility,
            vol_regime=vol_regime,
            volume_imbalance=imbalance,
            spread=spread,
            yes_price=yes_price,
            timestamp=now,
            z_score=z_score,
            delta_mean=delta_mean,
        )

        log.debug(
            f"[FLOW WS] {self.ticker} | YES={yes_price:.3f} "
            f"vel={velocity*100:+.3f}¢/s vol={volatility*100:.3f}¢ "
            f"regime={vol_regime} Z={z_score:+.2f} → {direction.value}"
        )



    def _compute_velocity(self, now: float) -> float:
        """Linear regression slope of yes_price over the trailing window."""
        cutoff = now - self.VELOCITY_WINDOW
        points = [(t, p) for t, p in self._price_history if t >= cutoff]
        if len(points) < 2:
            return 0.0

        times  = [p[0] - points[0][0] for p in points]  # relative seconds
        prices = [p[1] for p in points]
        n = len(times)
        sum_t  = sum(times)
        sum_p  = sum(prices)
        sum_tp = sum(t * p for t, p in zip(times, prices))
        sum_tt = sum(t * t for t in times)
        denom  = n * sum_tt - sum_t ** 2
        if abs(denom) < 1e-9:
            return 0.0
        slope = (n * sum_tp - sum_t * sum_p) / denom
        return slope  # units: price/second

    def _compute_volatility(self, now: float) -> float:
        """Std dev of point-to-point price changes over trailing window."""
        cutoff = now - self.VELOCITY_WINDOW
        prices = [p for t, p in self._price_history if t >= cutoff]
        if len(prices) < 3:
            return 0.0
        diffs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        mean = sum(diffs) / len(diffs)
        variance = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        return variance ** 0.5

    def _compute_zscore(self, yes_price: float) -> tuple:
        """
        Compute rolling Z-score of the probability delta (P_model - P_market).

        Z = (D_current - mu_D) / sigma_D

        where D_current = P_model - P_market (instantaneous delta),
              mu_D      = simple moving average of the delta over the window,
              sigma_D   = standard deviation of the delta.

        A Z-score > +2.0 means the market is severely UNDERPRICING our model
        probability → strong 'buy the dip' signal.
        A Z-score < -2.0 means market OVERPRICES us → avoid or fade.
        """
        d_current = self._model_prob - yes_price
        self._delta_history.append(d_current)

        n = len(self._delta_history)
        if n < 3:
            return 0.0, d_current   # not enough data yet

        deltas = list(self._delta_history)
        mu = sum(deltas) / n
        variance = sum((d - mu) ** 2 for d in deltas) / n
        sigma = math.sqrt(variance) if variance > 0 else 1e-9

        z = (d_current - mu) / sigma
        return round(z, 4), round(mu, 4)

    def _classify_vol(self, volatility: float) -> str:
        """Classify price choppiness into a regime."""
        if volatility >= 0.008:    # price bouncing > 0.8¢ std dev
            return "HIGH"
        elif volatility >= 0.004:  # price bouncing > 0.4¢ std dev
            return "MEDIUM"
        return "LOW"

    def _classify(self, velocity: float) -> FlowDirection:
        if abs(velocity) < self.VELOCITY_THRESHOLD:
            return FlowDirection.NEUTRAL
        # Positive velocity = YES price rising
        return FlowDirection.CONFIRM if velocity > 0 else FlowDirection.FADE

    def signal_for_side(self, betting_on_yes: bool) -> MarketSignal:
        """
        Flip the signal if we're betting NO — CONFIRM means price falling.
        NOTE: betting_on_yes=True → we want YES price to be undervalued
              betting_on_yes=False → we want NO price to be undervalued
        """
        sig = self.current_signal
        if betting_on_yes:
            return sig
        # For NO bets: rising YES price is bad (FADE), falling is good (CONFIRM)
        flipped = FlowDirection.NEUTRAL
        if sig.direction == FlowDirection.CONFIRM:
            flipped = FlowDirection.FADE
        elif sig.direction == FlowDirection.FADE:
            flipped = FlowDirection.CONFIRM
        else:
            flipped = FlowDirection.NEUTRAL
        return MarketSignal(
            direction=flipped,
            price_velocity=-sig.price_velocity,
            volume_imbalance=1.0 - sig.volume_imbalance,
            spread=sig.spread,
            yes_price=sig.yes_price,
            timestamp=sig.timestamp,
        )

    def mean_reversion_signal(self, fee_rate: float = 0.07) -> dict:
        """
        Evaluate whether the current Z-score constitutes a tradeable mean-reversion entry.

        Per information.md: entry is valid when:
          1. |Z-score| > 2.0  (statistically significant deviation)
          2. net edge = (P_model - P_market) - fee > 0  (positive after Kalshi fee)

        Returns:
            {
              "should_buy_dip": bool,
              "z_score": float,
              "net_edge": float,
              "direction": "underpriced" | "overpriced" | "neutral",
            }
        """
        sig = self.current_signal
        p_market = sig.yes_price
        p_model  = self._model_prob

        # Kalshi taker fee: ceil(0.07 * C * P * (1-P)) / C ≈ 0.07 * P * (1-P)
        taker_fee = fee_rate * p_market * (1.0 - p_market)
        raw_edge  = p_model - p_market
        net_edge  = raw_edge - taker_fee

        z = sig.z_score
        strong_signal = abs(z) >= 2.0

        # Underpriced = market price is LOW vs model → buy dip
        should_buy = strong_signal and z > 2.0 and net_edge > 0.0

        direction = "neutral"
        if z > 2.0:
            direction = "underpriced"
        elif z < -2.0:
            direction = "overpriced"

        if should_buy:
            log.info(
                f"[MEAN-REV] BUY-THE-DIP signal: Z={z:+.2f} "
                f"net_edge={net_edge:+.4f} P_model={p_model:.3f} P_market={p_market:.3f}"
            )

        return {
            "should_buy_dip": should_buy,
            "z_score":        z,
            "net_edge":       net_edge,
            "direction":      direction,
            "p_model":        p_model,
            "p_market":       p_market,
        }
