"""
risk_manager.py — Multivariate Kelly optimizer for correlated bet portfolios.

Solves the log-optimal allocation problem across a simultaneous book of
tennis markets.  Correlation groups (e.g. same tournament, same surface)
induce a covariance penalty that reduces the combined allocation compared
to independent Kelly sizing.

Constraints:
    - total allocation <= 25% of bankroll
    - individual bets   <= 5% of bankroll
    - each bet >= 0

Falls back to scaled independent Kelly if `scipy.optimize` isn't available.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

try:
    from scipy.optimize import minimize
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    minimize = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

log = logging.getLogger(__name__)


@dataclass
class BetSpec:
    p: float
    odds: float           # decimal odds payout — e.g. 2.1 means net win = 1.1
    corr_group: str = ""


def _normalise_bets(bets: Sequence) -> List[BetSpec]:
    out: List[BetSpec] = []
    for b in bets:
        if isinstance(b, BetSpec):
            out.append(b)
        else:
            out.append(
                BetSpec(
                    p=float(b["p"]),
                    odds=float(b["odds"]),
                    corr_group=str(b.get("corr_group", "")),
                )
            )
    return out


def _build_correlation_matrix(
    specs: Sequence[BetSpec],
    within_group_rho: float = 0.6,
    cross_group_rho: float = 0.05,
) -> np.ndarray:
    """
    Construct an approximate correlation matrix keyed on `corr_group`.
    Bets in the same group are assumed to be correlated (same tournament,
    same weather, etc.); cross-group pairs get a small baseline correlation.
    """
    n = len(specs)
    R = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if specs[i].corr_group and specs[i].corr_group == specs[j].corr_group:
                R[i, j] = within_group_rho
            else:
                R[i, j] = cross_group_rho
            R[j, i] = R[i, j]
    return R


class MultivariateKellyOptimizer:
    """
    SLSQP-based log-optimal allocator with total- and per-bet caps.

    Utility approximation (so we have a closed-form gradient target):

        U(f) = sum_i f_i * (p_i * (odds_i - 1) - (1 - p_i))
             - 0.5 * f.T @ Sigma @ f

    where Sigma is built from a correlation matrix scaled by bet variance
    (p*(1-p)*(odds-1)^2).  Maximizing U is equivalent to maximizing
    expected log utility to second order.
    """

    TOTAL_CAP: float = 0.25   # 25% of bankroll
    BET_CAP:   float = 0.05   # 5% of bankroll per bet

    def __init__(self, bankroll: float) -> None:
        if bankroll <= 0:
            raise ValueError("bankroll must be positive.")
        self.bankroll = float(bankroll)

    # .....................................................................
    def optimize(self, bets: Sequence[Dict]) -> List[float]:
        """
        Args:
            bets: list of dicts with keys {p, odds, corr_group}.

        Returns:
            List of bet sizes in dollars, same order as input.
        """
        if not bets:
            return []

        specs = _normalise_bets(bets)
        n = len(specs)

        # Expected edge per unit wager on each bet (b = odds - 1).
        b = np.array([max(1e-6, spec.odds - 1.0) for spec in specs], dtype=float)
        p = np.array([spec.p for spec in specs], dtype=float)
        edge = p * b - (1.0 - p)

        # Variance per bet (approx).
        var = p * (1.0 - p) * (b ** 2) + 1e-9
        sigma = np.sqrt(var)

        R = _build_correlation_matrix(specs)
        Sigma = np.outer(sigma, sigma) * R

        # Analytic fallback: independent Kelly
        independent_f = np.where(b > 0, edge / (b ** 2 + 1e-9), 0.0)
        independent_f = np.clip(independent_f, 0.0, self.BET_CAP)

        if not _SCIPY_AVAILABLE:
            log.warning("scipy.optimize not available — using independent-Kelly fallback.")
            f_opt = self._cap_total(independent_f)
            return [float(fi * self.bankroll) for fi in f_opt]

        # Objective: maximize edge.f - 0.5 * f.T Sigma f
        def neg_utility(f: np.ndarray) -> float:
            return -(float(np.dot(edge, f)) - 0.5 * float(f @ Sigma @ f))

        def neg_utility_grad(f: np.ndarray) -> np.ndarray:
            return -(edge - Sigma @ f)

        # Constraints: 0 <= f_i <= BET_CAP,   sum(f) <= TOTAL_CAP
        bounds = [(0.0, self.BET_CAP) for _ in range(n)]
        constraints = [
            {"type": "ineq", "fun": lambda f: self.TOTAL_CAP - float(np.sum(f))},
        ]

        x0 = np.minimum(independent_f, self.BET_CAP / max(1.0, n))
        try:
            res = minimize(
                neg_utility,
                x0=x0,
                jac=neg_utility_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            if not res.success:
                log.debug("SLSQP failed (%s) — falling back to independent Kelly.", res.message)
                f_opt = self._cap_total(independent_f)
            else:
                f_opt = np.clip(res.x, 0.0, self.BET_CAP)
                f_opt = self._cap_total(f_opt)
        except Exception as exc:  # pragma: no cover
            log.warning("Kelly optimizer raised: %s — using independent Kelly.", exc)
            f_opt = self._cap_total(independent_f)

        return [float(fi * self.bankroll) for fi in f_opt]

    # .....................................................................
    def _cap_total(self, f: np.ndarray) -> np.ndarray:
        """Scale fractions down proportionally so sum(f) <= TOTAL_CAP."""
        total = float(np.sum(f))
        if total > self.TOTAL_CAP and total > 0:
            f = f * (self.TOTAL_CAP / total)
        return np.clip(f, 0.0, self.BET_CAP)


__all__ = ["MultivariateKellyOptimizer", "BetSpec"]
