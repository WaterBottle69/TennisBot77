# GRAPH_REPORT — TennisBetting Knowledge Graph

## God Nodes (highest centrality)

- **BetManager** (score=0.000) — 
- **LiveMatchState** (score=0.000) — 
- **HybridMLEngine** (score=0.000) — 
- **Config** (score=0.000) — 
- **KalshiClient** (score=0.000) — 
- **MarketMonitor** (score=0.000) — 
- **AdaptiveController** (score=0.000) — 
- **TestTennisEventFilter** (score=0.000) — 
- **EloEngine** (score=0.000) — 
- **TennisStatsScraper** (score=0.000) — 

## Surprising Connections

- **?** ↔ **?**: inferred connection - not explicitly stated in source; connects across different repos/directories; bridges separate communities; peripheral node `In-Play Markov Backtest ======================== Simulates what the live bot wou` unexpectedly reaches hub `LiveMatchState`
- **?** ↔ **?**: inferred connection - not explicitly stated in source; connects across different repos/directories; bridges separate communities; peripheral node `Parse '6-3 7-5' or '6-3 3-6 7-6(4)' into [(6,3),(7,5)] etc.     Returns list of` unexpectedly reaches hub `LiveMatchState`
- **?** ↔ **?**: inferred connection - not explicitly stated in source; connects across different repos/directories; bridges separate communities; peripheral node `For each match:        - Replay game-by-game score through LiveMatchState` unexpectedly reaches hub `LiveMatchState`
- **?** ↔ **?**: inferred connection - not explicitly stated in source; connects across different repos/directories; bridges separate communities; peripheral node `Fetch current market state by ticker.` unexpectedly reaches hub `Config`
- **?** ↔ **?**: inferred connection - not explicitly stated in source; connects across different repos/directories; bridges separate communities; peripheral node `Fetch available (free) balance in USD.          Kalshi's /portfolio/balance retu` unexpectedly reaches hub `Config`
- **?** ↔ **?**: inferred connection - not explicitly stated in source; connects across different repos/directories; bridges separate communities

## Suggested Questions

- {'type': 'bridge_node', 'question': 'Why does `BetManager` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 9`, `Community 10`, `Community 13`?', 'why': 'High betweenness centrality (0.249) - this node is a cross-community bridge.'}
- {'type': 'bridge_node', 'question': 'Why does `Config` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 5`, `Community 10`?', 'why': 'High betweenness centrality (0.195) - this node is a cross-community bridge.'}
- {'type': 'bridge_node', 'question': 'Why does `LiveMatchState` connect `Community 0` to `Community 2`, `Community 3`, `Community 4`, `Community 9`, `Community 11`, `Community 13`?', 'why': 'High betweenness centrality (0.157) - this node is a cross-community bridge.'}
- {'type': 'verify_inferred', 'question': 'Are the 80 inferred relationships involving `BetManager` (e.g. with `KalshiClient` and `Config`) actually correct?', 'why': '`BetManager` has 80 INFERRED edges - model-reasoned connections that need verification.'}
- {'type': 'verify_inferred', 'question': 'Are the 86 inferred relationships involving `LiveMatchState` (e.g. with `PredictRequest` and `KalshiKeysRequest`) actually correct?', 'why': '`LiveMatchState` has 86 INFERRED edges - model-reasoned connections that need verification.'}
- {'type': 'verify_inferred', 'question': 'Are the 69 inferred relationships involving `HybridMLEngine` (e.g. with `TestGameWinProb` and `TestMatchWinProb`) actually correct?', 'why': '`HybridMLEngine` has 69 INFERRED edges - model-reasoned connections that need verification.'}

## Usage

- `graphify query "<question>"` — query graph (~50x fewer tokens than raw files)
- Re-run `/graphify .` after major code changes to refresh
