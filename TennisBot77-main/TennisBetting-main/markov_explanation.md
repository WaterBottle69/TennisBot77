# Markovian Probability Engine: Tennis Betting Bot

## 1. Introduction
The tennis betting bot uses a hierarchical Markov Chain model to calculate the real-time probability of a player winning a match. Unlike Monte Carlo simulations, this analytical approach provides exact probabilities for every state transitions.

---

## 2. Level 1: Point Winning Probability (Base Features)
The base inputs for the model are the win rates for each player depending on who is serving:
- **p_serve**: Probability Player A wins a point on their own serve.
- **p_return**: Probability Player A wins a point when the opponent is serving.

These features are derived from H2H (Head-to-Head) stats and ATP rankings during the pre-match analysis.

---

## 3. Level 2: Game Win Probability (P_G)
The probability of winning a game from a specific point score (i, j) where i, j are 0, 15, 30, or 40:

### Recursive Equation:
$$P(i, j) = p \cdot P(i+1, j) + (1-p) \cdot P(i, j+1)$$

### Deuce Boundary Condition:
If the score reaches 40-40 (Deuce), the infinite cycle of Advantage/Deuce converges analytically to:
$$P(3, 3) = \frac{p^2}{p^2 + (1-p)^2}$$

---

## 4. Level 3: Set Win Probability (P_S)
This level models the probability of winning a set based on games won (G1, G2). The server alternates every game.

### Transition Equation:
$$P(G_1, G_2) = P_G \cdot P(G_1+1, G_2) + (1-P_G) \cdot P(G_1, G_2+1)$$

### Tiebreak Logic (P_TB):
In a tiebreak (first to 7 points), servers alternate every 2 points. The probability of winning from 6-6 in a tiebreak is:
$$P_{TB}(6, 6) = \frac{p_{serve} \cdot p_{return}}{p_{serve} \cdot p_{return} + (1-p_{serve}) \cdot (1-p_{return})}$$

---

## 5. Level 4: Match Win Probability (P_M)
The top-level result for a Best of 3 sets match:
$$P(S_1, S_2) = P_S \cdot P(S_1+1, S_2) + (1-P_S) \cdot P(S_1, S_2+1)$$

- **S1, S2**: Current sets won by each player.
- **P_S**: Probability of winning the current set.

---

## 6. Real-Time Integration
- **Caching**: The engine uses `@lru_cache` to ensure millisecond-latency updates during high-frequency score changes.
- **Dynamic Updates**: Every point won or lost triggers a recalculation of the entire hierarchy, providing an instantaneous "Fair Value" for betting on Kalshi.
