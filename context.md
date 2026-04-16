# Identified Profitability Bugs

The following bugs were identified during an end-to-end code review. These issues directly impact real-money profitability and system robustness.

## 🔴 CRITICAL — High Financial Risk

### 1. Slippage Caps Ignored
- **Location**: `kalshi_client.py:570` and `605`
- **Issue**: The `price_cents` argument passed to `buy()` and `sell()` is being ignored or hardcoded in some versions. Orders default to 99¢ (buy) or 1¢ (sell), providing zero slippage protection.
- **Impact**: Significant losses on thin order books or during price spikes.

### 2. YES Side Mapping Assumption
- **Location**: `bet_manager.py:128-132`
- **Issue**: The system assumes YES price always equals `P(player_a wins)`. Kalshi YES tokens can be mapped to either player depending on the ticker/title.
- **Impact**: Trading the inverse of the intended edge at full Kelly size if the mapping is reversed.

### 3. Tiered Scale-out Double-Selling
- **Location**: `bet_manager.py:508-529`
- **Issue**: The loop logic for partial sells (`tranches <= tier`) allows the bot to sell 25% of the position on every tick if the price oscillates around a target tier (e.g., 2× entry).
- **Impact**: Prematurely liquidating winning positions.

## 🟠 MEDIUM — Edge Erosion

### 4. Fill Count Response Path
- **Location**: `bet_manager.py:386`
- **Issue**: `resp.get("taker_fill_count")` is incorrect because Kalshi nests the order details under `resp["order"]`.
- **Impact**: Failing to detect partially filled or resting orders, causing the bot to track positions it doesn't actually own.

### 5. Model Reversal Whipsaw
- **Location**: `bet_manager.py:532`
- **Issue**: The reversal guard fires on `current_edge < 0` (or `-MIN_EDGE` where `MIN_EDGE` is 0). It needs a symmetric buffer.
- **Impact**: Entering at +2% edge and exiting at -0.1% edge due to noise, losing the spread and fees on every whipsaw.

### 6. Fees Not Modeled
- **Location**: `bet_manager.py` / `config.py`
- **Issue**: Kalshi fees (~7% of winning payout) are not subtracted from the edge calculation or reflected in PnL.
- **Impact**: Entering trades that are actually negative expected value (EV) after fees.

### 7. Repeated Balance Fetching (Latency)
- **Location**: `main.py` / `bet_manager.py`
- **Issue**: `get_balance()` is called twice per tick (once for adaptive control, once for order sizing), adding 100-300ms of unnecessary latency.
- **Impact**: Slower execution in a high-frequency environment.

### 8. Frozen Serve/Return Probabilities
- **Location**: `main.py:174-177`
- **Issue**: Point probabilities (`p_serve`, `p_return`) are set once pre-match and never updated based on live performance (actual hold rates).
- **Impact**: Repricing inaccuracy as the match unfolds over several hours.

---

## Recommendation
Fix items #1, #2, #3, and #6 immediately before production deployment. Fix #7 and #8 for optimized performance.
