"""Long only, fully invested mean variance optimization (needs scipy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class OptimizationError(Exception):
    """Raised when the optimizer cannot find a solution."""


def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0):
    """Return (expected return, volatility, Sharpe) for weights `w`."""
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = float("nan") if vol == 0 else (ret - risk_free) / vol
    return ret, vol, sharpe


def _solve(objective, n: int, extra_constraints=()) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *extra_constraints]
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise OptimizationError(result.message)
    w = np.clip(result.x, 0.0, 1.0)
    return w / w.sum()


def _check(mu: pd.Series, cov: pd.DataFrame) -> None:
    if len(mu) < 2:
        raise ValueError("Optimization needs at least two assets.")
    if list(mu.index) != list(cov.index) or list(cov.index) != list(cov.columns):
        raise ValueError("mu and cov must share the same asset order.")


def min_variance(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    _check(mu, cov)
    c = cov.to_numpy()
    w = _solve(lambda w: w @ c @ w, len(mu))
    return pd.Series(w, index=mu.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
    _check(mu, cov)
    m, c = mu.to_numpy(), cov.to_numpy()

    def neg_sharpe(w):
        _, vol, sharpe = portfolio_stats(w, m, c, risk_free)
        return 1e6 if np.isnan(sharpe) else -sharpe

    w = _solve(neg_sharpe, len(mu))
    return pd.Series(w, index=mu.index)


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 40) -> pd.DataFrame:
    """Minimum volatility portfolios across a range of target returns.

    Columns: Return, Volatility, then one column per asset weight.
    Points the solver cannot reach are skipped.
    """
    _check(mu, cov)
    m, c = mu.to_numpy(), cov.to_numpy()
    low = portfolio_stats(min_variance(mu, cov).to_numpy(), m, c)[0]
    high = float(m.max())

    rows = []
    for target in np.linspace(low, high, n_points):
        try:
            w = _solve(
                lambda w: w @ c @ w,
                len(mu),
                extra_constraints=[{"type": "eq", "fun": lambda w, t=target: w @ m - t}],
            )
        except OptimizationError:
            continue
        ret, vol, _ = portfolio_stats(w, m, c)
        rows.append({"Return": ret, "Volatility": vol, **dict(zip(mu.index, w))})

    if not rows:
        raise OptimizationError("Could not build the efficient frontier.")
    return pd.DataFrame(rows)
