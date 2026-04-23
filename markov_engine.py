from functools import lru_cache
import logging
from typing import Optional, Tuple
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Non-i.i.d. Markov extension constants (research-paper derived)
# ──────────────────────────────────────────────────────────────────────────────
# Momentum bonus applied to the previous point's winner.
ALPHA_MOMENTUM: float = 0.015
# Pressure penalty applied to the serving player at high-leverage states.
BETA_PRESSURE: float = 0.02
# Tiebreak dampening — shrinks serve advantage toward 0.5 in tiebreaks.
ALPHA_TIEBREAK: float = 0.06
# Leverage threshold above which a point is considered "high pressure".
LEVERAGE_TAU: float = 0.6


def leverage_index(pa: int, pb: int) -> float:
    """
    Compute a leverage index for a given (points_A, points_B) game state.

    High-leverage situations per the research paper:
      - 30-40 (2-3) or 15-40 (1-3): break-point(s) on return
      - 40-30 (3-2) or 40-15 (3-1): set/game-point on serve
      - Deuce (3-3) and any advantage state

    Returns a float in [0, 1] — values >= LEVERAGE_TAU are "high leverage".
    """
    # Deuce / advantage
    if pa >= 3 and pb >= 3:
        return 0.9
    # Break / game point scenarios
    if (pa == 2 and pb == 3) or (pa == 1 and pb == 3):
        return 0.85
    if (pa == 3 and pb == 2) or (pa == 3 and pb == 1):
        return 0.75
    if pa == 3 or pb == 3:
        return 0.65
    # Mid-game
    if pa + pb >= 3:
        return 0.4
    return 0.2


def conditional_serve_prob(
    p_serve: float,
    winner_t_minus_1: Optional[int] = None,
    leverage: float = 0.0,
    server_index: int = 0,
    is_tiebreak: bool = False,
) -> float:
    """
    Compute the non-i.i.d. conditional probability that the server wins the
    next point, given the previous winner and the leverage state.

    Rules (applied additively, then clamped):
      - If winner_t_minus_1 == server_index → +ALPHA_MOMENTUM
      - If winner_t_minus_1 is the opponent → -ALPHA_MOMENTUM (mirror effect)
      - If leverage > LEVERAGE_TAU         → -BETA_PRESSURE
      - If is_tiebreak                     → dampened toward 0.5 by ALPHA_TIEBREAK

    Args:
        p_serve: baseline serve-win probability for this server.
        winner_t_minus_1: 0 if server won last point, 1 if opponent did,
                          or None for the first point / unknown.
        leverage: output of leverage_index().
        server_index: 0 or 1 — which player is serving this point.
        is_tiebreak: whether the current game is a tiebreak.

    Returns:
        Adjusted probability clamped to [0.01, 0.99].
    """
    p = float(p_serve)

    if winner_t_minus_1 is not None:
        if winner_t_minus_1 == server_index:
            p += ALPHA_MOMENTUM
        else:
            p -= ALPHA_MOMENTUM

    if leverage > LEVERAGE_TAU:
        p -= BETA_PRESSURE
        
        # FATIGUE SPIKE OVERRIDE: Extreme anaerobic lactic accumulation during extended Deuce
        if leverage >= 0.9:
            p -= 0.035

    if is_tiebreak:
        # Pull toward 0.5 — tiebreaks attenuate the serve advantage.
        p = p + ALPHA_TIEBREAK * (0.5 - p)

    return max(0.01, min(0.99, p))

# --- Level 1: Game Win Probability ---

@lru_cache(maxsize=None)
def _game_win_prob(p, i, j):
    if i >= 4 and i - j >= 2: return 1.0
    if j >= 4 and j - i >= 2: return 0.0
    if i >= 3 and j >= 3:
        return p * p / (p * p + (1-p) * (1-p))
    return p * _game_win_prob(p, i+1, j) + (1-p) * _game_win_prob(p, i, j+1)

def game_win_prob(p, from_state=(0, 0)):
    return _game_win_prob(p, from_state[0], from_state[1])

# --- Level 2: Set Win Probability ---

@lru_cache(maxsize=None)
def _tiebreak_win_prob(p_serve, p_return, i, j, p1_serving):
    if i >= 7 and i - j >= 2: return 1.0
    if j >= 7 and j - i >= 2: return 0.0

    if i >= 6 and j >= 6 and i == j:
        pw = p_serve * p_return
        pl = (1-p_serve) * (1-p_return)
        return pw / (pw + pl) if (pw + pl) > 0 else 0.5

    k = i + j
    initial = 0 if p1_serving else 1
    if k == 0:
        server = initial
    else:
        b = (k - 1) // 2
        server = initial ^ (0 if (b % 2 == 1) else 1)

    prob = p_serve if server == 0 else p_return
    return prob * _tiebreak_win_prob(p_serve, p_return, i+1, j, p1_serving) + \
           (1-prob) * _tiebreak_win_prob(p_serve, p_return, i, j+1, p1_serving)

def tiebreak_win_prob(p_serve, p_return, from_state=(0, 0), p1_serving=True):
    return _tiebreak_win_prob(p_serve, p_return, from_state[0], from_state[1], p1_serving)


@lru_cache(maxsize=None)
def _set_win_prob(p_serve, p_return, i, j, server_turn, target=6):
    if i == 6 and j == 6:
        return _tiebreak_win_prob(p_serve, p_return, 0, 0, server_turn)
    if i >= target and i - j >= 2: return 1.0
    if j >= target and j - i >= 2: return 0.0
    g = _game_win_prob(p_serve, 0, 0) if server_turn else _game_win_prob(p_return, 0, 0)
    return g * _set_win_prob(p_serve, p_return, i+1, j, not server_turn, target) + \
           (1-g) * _set_win_prob(p_serve, p_return, i, j+1, not server_turn, target)

def set_win_prob(p_serve, p_return, from_state=(0, 0), p1_serving_next=True, target=6):
    return _set_win_prob(p_serve, p_return, from_state[0], from_state[1], p1_serving_next, target)

# --- Level 3: Match Win Probability ---

@lru_cache(maxsize=None)
def _match_win_prob(p_serve, p_return, i, j, p1_serving, target):
    if i == target: return 1.0
    if j == target: return 0.0
    s = _set_win_prob(p_serve, p_return, 0, 0, p1_serving)
    return s * _match_win_prob(p_serve, p_return, i+1, j, not p1_serving, target) + \
           (1-s) * _match_win_prob(p_serve, p_return, i, j+1, not p1_serving, target)

def match_win_prob(p_serve, p_return, from_state=(0, 0), p1_serving_set=True, best_of=3):
    target = best_of // 2 + 1
    return _match_win_prob(p_serve, p_return, from_state[0], from_state[1], p1_serving_set, target)


class LiveMatchState:
    def __init__(self, p_serve, p_return):
        # For our definitions:
        # p_serve: P(Player A wins a point on Player A's serve)
        # p_return: P(Player A wins a point on Player B's serve)
        # We need these two baselines!
        self.p_serve = p_serve    
        self.p_return = p_return  
        
        self.match_sets = (0, 0)
        self.current_set_games = (0, 0)
        self.current_game_points = (0, 0)
        self.p1_serving = True
    
    def win_probability(self):
        # Current point -> finishes the game
        # g1 is prob P1 wins the rest of the game assuming it's P1's serve
        p_game_finish = game_win_prob(self.p_serve if self.p1_serving else self.p_return, self.current_game_points)
        p_game_loss = 1.0 - p_game_finish

        # Instead of generic s = set_win_prob for a *fresh* set, we need set_win_prob from CURRENT STATE.
        # But wait, set_win_prob is implemented as recursive DP. We can modify it to handle partial sets
        # by passing the partial game outcome! 
        # Because the game is in progress, winning the set is:
        # P(win game) * P(win set from games i+1,j) + P(lose game) * P(win set from games i,j+1)
        
        def set_p_remaining(i, j, p1_serving):
            if i == 6 and j == 6:
                # Assuming tiebreak is fresh. If tiebreak is halfway, we need tiebreak remaining prob.
                return tiebreak_win_prob(self.p_serve, self.p_return, from_state=(0, 0), p1_serving=p1_serving)
            if i >= 6 and i - j >= 2: return 1.0
            if j >= 6 and j - i >= 2: return 0.0
            
            s_rest = set_win_prob(self.p_serve, self.p_return, (i, j), p1_serving)
            return s_rest

        if self.p1_serving:
            set_prob = p_game_finish * set_p_remaining(self.current_set_games[0]+1, self.current_set_games[1], not self.p1_serving) + \
                       p_game_loss * set_p_remaining(self.current_set_games[0], self.current_set_games[1]+1, not self.p1_serving)
        else:
            # P1 returns game. p_game_finish is P1 winning a game on P2's serve.
            set_prob = p_game_finish * set_p_remaining(self.current_set_games[0]+1, self.current_set_games[1], not self.p1_serving) + \
                       p_game_loss * set_p_remaining(self.current_set_games[0], self.current_set_games[1]+1, not self.p1_serving)

        # Likewise for match logic
        match_p_finish = set_prob * match_win_prob(self.p_serve, self.p_return, (self.match_sets[0]+1, self.match_sets[1]), not self.p1_serving) + \
                         (1.0 - set_prob) * match_win_prob(self.p_serve, self.p_return, (self.match_sets[0], self.match_sets[1]+1), not self.p1_serving)

        return match_p_finish

    def predict_post_point_state(self) -> dict:
        """
        One-point lookahead: compute the anticipated match win probability
        AFTER the next point is played, in both outcome branches.

        Used by BetManager to place predictive limit orders at the pre-move
        price before Kalshi re-prices following a score update.

        Returns:
            p_win_next   - P(Player A wins the next point)
            prob_if_win  - match win prob if A wins the point
            prob_if_lose - match win prob if A loses the point
            expected_prob - weighted average (≈ current win_probability())
        """
        p_win_next = self.p_serve if self.p1_serving else self.p_return
        pa, pb = self.current_game_points

        # Temporarily advance game state into each branch and call win_probability().
        # win_probability() is purely a function of self.* state, so snapshot/restore is safe.
        # game_win_prob handles all degenerate point states (4-2, 5-3, ad states) correctly
        # via its lru_cache recursion — no separate game-end detection needed here.
        orig_pts = self.current_game_points

        self.current_game_points = (pa + 1, pb)
        prob_if_win = self.win_probability()

        self.current_game_points = (pa, pb + 1)
        prob_if_lose = self.win_probability()

        self.current_game_points = orig_pts   # always restore

        return {
            "p_win_next":    round(p_win_next, 4),
            "prob_if_win":   round(prob_if_win, 4),
            "prob_if_lose":  round(prob_if_lose, 4),
            "expected_prob": round(p_win_next * prob_if_win + (1.0 - p_win_next) * prob_if_lose, 4),
        }

    def update(self, action_dict):
        """
        action_dict can specify if player_a won or player_b won the point, 
        and updates the states accordingly. Or directly override scores.
        """
        # Ex: action_dict = {"sets": (1,0), "games": (3,2), "points": (15, 30)}
        if "sets" in action_dict:
            self.match_sets = action_dict["sets"]
        if "games" in action_dict:
            self.current_set_games = action_dict["games"]
        if "points" in action_dict:
            pts = action_dict["points"]
            # Convert 0, 15, 30, 40 to 0, 1, 2, 3
            def map_pts(p):
                if p == 0: return 0
                if p == 15: return 1
                if p == 30: return 2
                if p == 40: return 3
                return p
            if isinstance(pts, tuple):
                self.current_game_points = (map_pts(pts[0]), map_pts(pts[1]))

        if "p1_serving" in action_dict:
            self.p1_serving = action_dict["p1_serving"]

    def update_params(self, p_serve: float, p_return: float):
        """Dynamically update serve/return probabilities from live ATP statistics.
        Clears all LRU caches since the probability basis has changed.
        """
        self.p_serve = max(0.01, min(0.99, p_serve))
        self.p_return = max(0.01, min(0.99, p_return))
        _game_win_prob.cache_clear()
        _tiebreak_win_prob.cache_clear()
        _set_win_prob.cache_clear()
        _match_win_prob.cache_clear()
        log.info(
            "Markov params updated: p_serve=%.4f  p_return=%.4f",
            self.p_serve, self.p_return,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Non-i.i.d. 4D state-space LiveMatchState
# ──────────────────────────────────────────────────────────────────────────────

class NonIIDLiveMatchState(LiveMatchState):
    """
    Extends LiveMatchState with a 4D state vector:
        (points_A, points_B, winner_t_minus_1, leverage_index)

    Instead of a constant p_serve / p_return per point, effective
    probabilities are recomputed each call via `conditional_serve_prob()`
    to incorporate momentum (alpha=0.015), pressure (beta=0.02) and
    tiebreak dampening (alpha_tiebreak=0.06).

    Backward compatibility:
      - All original LiveMatchState attributes and methods are inherited.
      - `win_probability()` is overridden; existing callers using
        `LiveMatchState` remain unchanged.
      - To opt-in, callers explicitly instantiate `NonIIDLiveMatchState`.
    """

    def __init__(
        self,
        p_serve: float,
        p_return: float,
        *,
        alpha_momentum: float = ALPHA_MOMENTUM,
        beta_pressure: float = BETA_PRESSURE,
        alpha_tiebreak: float = ALPHA_TIEBREAK,
        leverage_tau: float = LEVERAGE_TAU,
    ) -> None:
        super().__init__(p_serve, p_return)
        self.alpha_momentum = float(alpha_momentum)
        self.beta_pressure = float(beta_pressure)
        self.alpha_tiebreak = float(alpha_tiebreak)
        self.leverage_tau = float(leverage_tau)
        # 4D state components
        self.winner_t_minus_1: Optional[int] = None   # 0=A won, 1=B won
        self.leverage: float = 0.0

    # .....................................................................
    def _is_tiebreak(self) -> bool:
        g0, g1 = self.current_set_games
        return g0 == 6 and g1 == 6

    def current_leverage(self) -> float:
        pa, pb = self.current_game_points
        lev = leverage_index(pa, pb)
        self.leverage = lev
        return lev

    def effective_probs(self) -> Tuple[float, float]:
        """
        Return (p_serve_eff, p_return_eff) under the current state, applying
        momentum / pressure / tiebreak conditional adjustments.
        """
        lev = self.current_leverage()
        is_tb = self._is_tiebreak()

        # Server index is 0 when P1 (A) serves.
        server_idx = 0 if self.p1_serving else 1

        p_serve_eff = conditional_serve_prob(
            self.p_serve,
            winner_t_minus_1=self.winner_t_minus_1,
            leverage=lev,
            server_index=server_idx,
            is_tiebreak=is_tb,
        )
        # For p_return we model *B* serving: server is the opponent here.
        # Apply the same conditional mapping but with server_idx flipped.
        p_return_eff = conditional_serve_prob(
            self.p_return,
            winner_t_minus_1=self.winner_t_minus_1,
            leverage=lev,
            server_index=1 - server_idx,
            is_tiebreak=is_tb,
        )
        return p_serve_eff, p_return_eff

    # .....................................................................
    def win_probability(self) -> float:
        """Override: compute using conditionally-adjusted p_serve/p_return."""
        # Temporarily swap base probs for the duration of this call and clear
        # caches so the DP uses the new effective values.
        p_serve_eff, p_return_eff = self.effective_probs()
        orig_s, orig_r = self.p_serve, self.p_return
        self.p_serve, self.p_return = p_serve_eff, p_return_eff
        _game_win_prob.cache_clear()
        _tiebreak_win_prob.cache_clear()
        _set_win_prob.cache_clear()
        _match_win_prob.cache_clear()
        try:
            result = super().win_probability()
        finally:
            # Restore base probabilities so subsequent calls can recompute.
            self.p_serve, self.p_return = orig_s, orig_r
            _game_win_prob.cache_clear()
            _tiebreak_win_prob.cache_clear()
            _set_win_prob.cache_clear()
            _match_win_prob.cache_clear()
        return result

    # .....................................................................
    def record_point(self, winner: int) -> None:
        """
        Record which player won the last point so subsequent probability
        computations can apply momentum bonus.

        Args:
            winner: 0 if Player A (YES) won the point, 1 if Player B won.
        """
        if winner not in (0, 1):
            raise ValueError("winner must be 0 or 1")
        self.winner_t_minus_1 = int(winner)

    def get_state_vector(self) -> tuple:
        """Return the 4D (points_A, points_B, winner_t_minus_1, leverage) tuple."""
        pa, pb = self.current_game_points
        return (pa, pb, self.winner_t_minus_1, self.current_leverage())
