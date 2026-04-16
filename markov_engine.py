from functools import lru_cache
import logging
log = logging.getLogger(__name__)

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
