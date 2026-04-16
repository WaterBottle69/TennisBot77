# Implementation Plan: Profitability Fixes

This plan addresses 8 bugs identified in the end-to-end review that impact real-money profitability and system robustness.

## User Review Required

> [!IMPORTANT]
> The **YES-Side Mapping** fix relies on fuzzy string matching of player names against the Kalshi market title. If Kalshi suddenly changes their title format to something totally unrelated to the players, this could still fail (though it's safer than the current hardcoded assumption).

> [!CAUTION]
> **Live Parameter Updates** will cause the Markov win probability to shift more dynamically. This is more accurate but might lead to more frequent trade exits/entries during the "discovery" phase of a match.

## Proposed Changes

### [Component 1] Kalshi Client
#### [MODIFY] kalshi_client.py
- Ensure `buy` and `sell` methods strictly use the `price_cents` argument for slippage protection in the `{side}_price` payload field.

---

### [Component 2] Config
#### [MODIFY] config.py
- Re-verify `KALSHI_FEE_RATE` and `MODEL_REVERSAL_EXIT_EDGE` are correctly tuned.

---

### [Component 3] Markov Engine
#### [MODIFY] markov_engine.py
- Add `update_params(p_serve, p_return)` method to `LiveMatchState` to allow live probability adjustments.

---

### [Component 4] Bet Manager
#### [MODIFY] bet_manager.py
- **YES Mapping**: Match `player_a` name against market title to determine if YES = player_a.
- **Scale-out**: Fix double-selling bug by changing the tranche loop check.
- **Fill Count**: Update Kalshi response parsing to `resp.get("order", {}).get("taker_fill_count")`.
- **Reversal Guard**: Use `MODEL_REVERSAL_EXIT_EDGE` for exit logic.
- **Fees**: Ensure `KALSHI_FEE_RATE` is subtracted from raw edge before entry.
- **Latency**: Use `available_balance` passed from `evaluate_and_act` to skip redundant API calls.

---

### [Component 5] Main Orchestrator
#### [MODIFY] main.py
- **Latency**: Fetch `kalshi.get_balance()` once per poll cycle and pass to `evaluate_and_act`.
- **Live Params**: Track observed hold rates (points won on serve) and call `lms.update_params()` to refine probabilities in real-time.

## Verification Plan

### Automated Tests
- Run `main.py` in DEMO mode and monitor `bot.log` for:
    - "YES mapping verified: PLAYER_X IS YES"
    - "Updating p_serve/p_return based on live hold rate"
    - "EXIT MODEL REVERSAL (threshold=0.015)"
- Verify order payloads in logs show correct slippage prices (not 1 or 99).

### Manual Verification
- Check the dashboard to ensure win probabilities are updating smoothly with the new parameters.
- Push to GitHub once verified locally.
