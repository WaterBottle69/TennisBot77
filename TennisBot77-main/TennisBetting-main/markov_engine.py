# ---------------------------------------------------------------------------
# Markov Tennis Engine
# ---------------------------------------------------------------------------
# IMPORTANT: All DP/cache functions are defined at module level so lru_cache
# tables persist across repeated calls. Defining them inside a function body
# creates a new (empty) cache on every call, defeating the purpose of caching.
# ---------------------------------------------------------------------------

# Analytical deuce formula:
#   P(server wins from deuce) = p² / (p² + q²)
#   This is the sum of the infinite series: p²·∑(pq)^k = p²/(1-pq-qp) = p²/(p²+q²)
def _deuce_prob(p: float) -> float:
    p2 = p * p
    q2 = (1 - p) * (1 - p)
    denom = p2 + q2
    return p2 / denom if denom > 0 else 0.5


# ---------------------------------------------------------------------------
# Level 1: Game Win Probability  (recursive DP, point-by-point)
# ---------------------------------------------------------------------------
# Cache key: (p_quantized, i, j). We quantize p to 4 decimal places so the
# cache doesn't grow unboundedly with floating-point noise.
_game_cache: dict = {}

def _game_P(p_key: int, p: float, i: int, j: int) -> float:
    key = (p_key, i, j)
    if key in _game_cache:
        return _game_cache[key]

    if i >= 4 and i - j >= 2:
        v = 1.0
    elif j >= 4 and j - i >= 2:
        v = 0.0
    elif i >= 3 and j >= 3 and i == j:          # deuce (includes 3-3, 4-4, ...)
        v = _deuce_prob(p)
    else:
        v = p * _game_P(p_key, p, i + 1, j) + (1 - p) * _game_P(p_key, p, i, j + 1)

    _game_cache[key] = v
    return v


def game_win_prob(p: float, from_state: tuple = (0, 0)) -> float:
    """
    Probability the server wins the game given:
      p           – P(server wins a point)
      from_state  – (server_pts, returner_pts); pts are 0-3 (0/15/30/40),
                    4 means the side has already won.
    """
    p_key = round(p, 4)  # quantize for cache keying
    return _game_P(int(p_key * 10000), p, from_state[0], from_state[1])

# ---------------------------------------------------------------------------
# Level 2a: Tiebreak Win Probability
# ---------------------------------------------------------------------------
# Official tiebreak serve pattern (USTA/ITF rule):
#   Point 0          → initial server (p1_serving=True → Player 1)
#   Points 1–2       → opponent
#   Points 3–4       → initial server
#   Points 5–6       → opponent
#   ... alternating every 2 points after the first.
#
# Formula for point k (0-indexed):
#   k == 0  → initial server
#   k >= 1  → block index b = (k-1)//2
#             even b → opponent of initial
#             odd  b → initial
#
# BUG FIX: The old code had the XOR reversed:
#   old: server = initial ^ (0 if b%2==1 else 1)   ← wrong (opponent on odd)
#   new: server = initial ^ (1 if b%2==0 else 0)   ← correct (opponent on even)
#
# Tiebreak super-deuce (6-6+):
#   One point each (one on P1 serve, one on P2 serve) forms a "mini-game".
#   P(win mini-game) = p_serve * p_return
#   P(lose mini-game) = (1-p_serve) * (1-p_return)
#   Overall P(win from deuce) = P(win mini) / (P(win mini) + P(lose mini))

_tiebreak_cache: dict = {}

def _tb_P(key: tuple, p_serve: float, p_return: float, p1_serving: bool, i: int, j: int) -> float:
    state = (key, i, j)
    if state in _tiebreak_cache:
        return _tiebreak_cache[state]

    if i >= 7 and i - j >= 2:
        v = 1.0
    elif j >= 7 and j - i >= 2:
        v = 0.0
    elif i >= 6 and j >= 6 and i == j:
        # Tiebreak super-deuce: need to win a 2-point sequence
        pw = p_serve * p_return          # win both (P1 serve then P2 serve)
        pl = (1 - p_serve) * (1 - p_return)  # lose both
        v = pw / (pw + pl) if (pw + pl) > 0 else 0.5
    else:
        k = i + j
        initial = 0 if p1_serving else 1   # 0 = P1, 1 = P2
        if k == 0:
            server = initial
        else:
            b = (k - 1) // 2
            # FIXED: even block → opponent serves; odd block → initial server
            server = initial ^ (1 if b % 2 == 0 else 0)

        prob = p_serve if server == 0 else p_return
        v = prob * _tb_P(key, p_serve, p_return, p1_serving, i + 1, j) + \
            (1 - prob) * _tb_P(key, p_serve, p_return, p1_serving, i, j + 1)

    _tiebreak_cache[state] = v
    return v


def tiebreak_win_prob(p_serve: float, p_return: float,
                      from_state: tuple = (0, 0), p1_serving: bool = True) -> float:
    """
    Probability Player 1 wins the tiebreak.
      p_serve   – P(P1 wins a point on P1's serve)
      p_return  – P(P1 wins a point on P2's serve)
      from_state – (p1_points, p2_points) already played in this tiebreak
      p1_serving – True if P1 served the very first point of the tiebreak
    """
    # Cache key encodes both probabilities (quantized) and who served first.
    cache_key = (round(p_serve, 4), round(p_return, 4), p1_serving)
    return _tb_P(cache_key, p_serve, p_return, p1_serving, from_state[0], from_state[1])


# ---------------------------------------------------------------------------
# Level 2b: Set Win Probability
# ---------------------------------------------------------------------------
_set_cache: dict = {}

def _set_P(key: tuple, p_serve: float, p_return: float, target: int,
           i: int, j: int, server_turn: bool) -> float:
    state = (key, i, j, server_turn)
    if state in _set_cache:
        return _set_cache[state]

    if i == 6 and j == 6:
        v = tiebreak_win_prob(p_serve, p_return, from_state=(0, 0), p1_serving=server_turn)
    elif i >= target and i - j >= 2:
        v = 1.0
    elif j >= target and j - i >= 2:
        v = 0.0
    else:
        g = game_win_prob(p_serve) if server_turn else game_win_prob(p_return)
        v = (g * _set_P(key, p_serve, p_return, target, i + 1, j, not server_turn) +
             (1 - g) * _set_P(key, p_serve, p_return, target, i, j + 1, not server_turn))

    _set_cache[state] = v
    return v


def set_win_prob(p_serve: float, p_return: float,
                from_state: tuple = (0, 0),
                p1_serving_next: bool = True,
                target: int = 6) -> float:
    """
    Probability Player 1 wins the set.
      p_serve        – P(P1 wins a point on P1's serve)
      p_return       – P(P1 wins a point when returning, i.e. on P2's serve)
      from_state     – (p1_games, p2_games) already won in this set
      p1_serving_next – True if P1 serves the next game
      target         – games required to win the set (normally 6)
    """
    cache_key = (round(p_serve, 4), round(p_return, 4), target)
    return _set_P(cache_key, p_serve, p_return, target,
                  from_state[0], from_state[1], p1_serving_next)

# ---------------------------------------------------------------------------
# Level 3: Match Win Probability
# ---------------------------------------------------------------------------
_match_cache: dict = {}

def _match_P(key: tuple, p_serve: float, p_return: float, target: int,
             i: int, j: int, p1_serving: bool) -> float:
    state = (key, i, j, p1_serving)
    if state in _match_cache:
        return _match_cache[state]

    if i == target:
        v = 1.0
    elif j == target:
        v = 0.0
    else:
        s = set_win_prob(p_serve, p_return, (0, 0), p1_serving)
        v = (s * _match_P(key, p_serve, p_return, target, i + 1, j, not p1_serving) +
             (1 - s) * _match_P(key, p_serve, p_return, target, i, j + 1, not p1_serving))

    _match_cache[state] = v
    return v


def match_win_prob(p_serve: float, p_return: float,
                  from_state: tuple = (0, 0),
                  p1_serving_set: bool = True,
                  best_of: int = 3) -> float:
    """
    Probability Player 1 wins the match.
      p_serve       – P(P1 wins a point on P1's serve)
      p_return      – P(P1 wins a point when returning)
      from_state    – (sets_won_p1, sets_won_p2)
      p1_serving_set – True if P1 serves first in the next set
      best_of       – 3 or 5
    """
    target = best_of // 2 + 1   # 2 for BO3, 3 for BO5
    cache_key = (round(p_serve, 4), round(p_return, 4), target)
    return _match_P(cache_key, p_serve, p_return, target,
                    from_state[0], from_state[1], p1_serving_set)


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
    
    def win_probability(self) -> float:
        """
        Compute P(Player 1 wins the match) from the current live state.

        We decompose the remaining match into three nested levels:
          1. Current in-progress game  → game_win_prob(partial points)
          2. Current in-progress set   → set_win_prob(partial games), wired
             through the outcome of the current game.
          3. Remaining sets in match   → match_win_prob(sets already won)

        Both branches of the game outcome (win/lose) are handled correctly
        regardless of who is serving — `p_game_finish` is always from P1's
        perspective, so the same formula applies in both serving directions.
        """
        # 1. Probability P1 wins/loses the current in-progress game
        p_pt = self.p_serve if self.p1_serving else self.p_return
        p_game_win  = game_win_prob(p_pt, self.current_game_points)
        p_game_loss = 1.0 - p_game_win

        # After the current game, the server flips.
        next_server = not self.p1_serving
        gi, gj = self.current_set_games

        # 2. Probability P1 wins the current set, conditioned on game outcome
        #    set_win_prob already handles the 6-6 tiebreak and all edge cases.
        p_set_if_win  = set_win_prob(self.p_serve, self.p_return,
                                      (gi + 1, gj), next_server)
        p_set_if_loss = set_win_prob(self.p_serve, self.p_return,
                                      (gi, gj + 1), next_server)
        set_prob = p_game_win * p_set_if_win + p_game_loss * p_set_if_loss

        # After the current set, serving changes again for the next set first game.
        # Convention: the player who DID NOT serve first in this set serves first
        # in the next set (standard ATP rules).
        next_set_server = self.p1_serving   # flips relative to current set's first server

        si, sj = self.match_sets

        # 3. Probability P1 wins the match, conditioned on set outcome
        p_match_if_set_win  = match_win_prob(self.p_serve, self.p_return,
                                              (si + 1, sj), next_set_server)
        p_match_if_set_loss = match_win_prob(self.p_serve, self.p_return,
                                              (si, sj + 1), next_set_server)

        return set_prob * p_match_if_set_win + (1.0 - set_prob) * p_match_if_set_loss

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
