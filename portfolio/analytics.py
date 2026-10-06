"""Return, risk and portfolio math. Pure pandas/numpy, no I/O."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Label shown in the UI -> internal rebalance code.
REBALANCE_OPTIONS = {
    "Daily": "daily",
    "Monthly": "monthly",
    "Quarterly": "quarterly",
    "Annually": "annual",
    "Never (buy and hold)": "none",
}


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """R_t = (P_t - P_{t-1}) / P_{t-1}."""
    return prices.pct_change().dropna(how="any")


def annualized_return(returns):
    """Geometric annualized return (CAGR) from daily returns."""
    n = len(returns)
    if n == 0:
        raise ValueError("Need at least one return observation.")
    growth = (1.0 + returns).prod()
    return growth ** (TRADING_DAYS / n) - 1.0


def annualized_volatility(returns):
    """Sample standard deviation (n-1) of daily returns, annualized."""
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(annual_return, annual_vol, risk_free: float = 0.0):
    """(Return - Rf) / Volatility. Zero volatility gives NaN."""
    excess = annual_return - risk_free
    if isinstance(annual_vol, pd.Series):
        return excess / annual_vol.where(annual_vol != 0)
    return float("nan") if annual_vol == 0 else excess / annual_vol


def asset_metrics(returns: pd.DataFrame, risk_free: float = 0.0) -> pd.DataFrame:
    """One row per asset with CAGR, volatility and Sharpe ratio."""
    ret = annualized_return(returns)
    vol = annualized_volatility(returns)
    return pd.DataFrame(
        {
            "Annualized Return (CAGR)": ret,
            "Annualized Volatility": vol,
            "Sharpe Ratio": sharpe_ratio(ret, vol, risk_free),
        }
    )


def covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """Sample covariance (n-1) of daily returns, optionally annualized."""
    cov = returns.cov()
    return cov * TRADING_DAYS if annualize else cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """rho_xy = cov_xy / (sigma_x * sigma_y)."""
    return returns.corr()


def equal_weights_pct(n: int) -> list[float]:
    """Equal percentage weights that sum to exactly 100 (remainder goes to the last asset)."""
    if n <= 0:
        return []
    base = round(100.0 / n, 2)
    weights = [base] * n
    weights[-1] = round(100.0 - base * (n - 1), 2)
    return weights


def _period_keys(index: pd.DatetimeIndex, rebalance: str) -> np.ndarray | None:
    if rebalance == "monthly":
        return (index.year * 12 + index.month).to_numpy()
    if rebalance == "quarterly":
        return (index.year * 4 + index.quarter).to_numpy()
    if rebalance == "annual":
        return index.year.to_numpy()
    return None


def simulate_portfolio(
    returns: pd.DataFrame, weights: np.ndarray, rebalance: str = "daily"
) -> pd.Series:
    """Daily portfolio returns under a rebalancing policy.

    Between rebalances each holding drifts with its own return, so the weights
    change. A rebalance resets the weights to `weights` at the start of the
    first trading day of the new period ("daily" resets every day, "none"
    never resets, which is buy and hold).
    """
    if rebalance not in REBALANCE_OPTIONS.values():
        raise ValueError(f"Unknown rebalance option: {rebalance}")
    target = np.asarray(weights, dtype=float)
    if target.shape != (returns.shape[1],):
        raise ValueError("Weights must have one entry per asset.")

    values = returns.to_numpy(dtype=float)
    keys = _period_keys(returns.index, rebalance)
    current = target.copy()
    out = np.empty(len(values))

    for i, day in enumerate(values):
        if rebalance == "daily":
            current = target.copy()
        elif keys is not None and i > 0 and keys[i] != keys[i - 1]:
            current = target.copy()
        port = float(current @ day)
        out[i] = port
        growth = 1.0 + port
        current = current * (1.0 + day) / growth if growth > 0 else target.copy()

    return pd.Series(out, index=returns.index, name="Portfolio")


def cumulative_growth(returns: pd.Series) -> pd.Series:
    """Cumulative return since the start, as a decimal (0.25 = +25%)."""
    return (1.0 + returns).cumprod() - 1.0


def drawdown(returns: pd.Series) -> pd.Series:
    """Percentage below the running peak. The starting value counts as a peak."""
    wealth = (1.0 + returns).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Slope of portfolio returns against benchmark returns."""
    var = benchmark.var(ddof=1)
    return float("nan") if var == 0 else float(returns.cov(benchmark) / var)


def series_summary(returns: pd.Series, risk_free: float = 0.0) -> dict[str, float]:
    """Headline statistics for a single daily return series."""
    ret = float(annualized_return(returns))
    vol = float(annualized_volatility(returns))
    return {
        "Total Return": float((1.0 + returns).prod() - 1.0),
        "Annualized Return (CAGR)": ret,
        "Annualized Volatility": vol,
        "Sharpe Ratio": float(sharpe_ratio(ret, vol, risk_free)),
        "Max Drawdown": max_drawdown(returns),
    }
