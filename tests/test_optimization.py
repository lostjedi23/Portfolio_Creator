import numpy as np
import pandas as pd
import pytest

from portfolio import optimization


@pytest.fixture
def inputs():
    mu = pd.Series([0.05, 0.10, 0.15], index=["A", "B", "C"])
    cov = pd.DataFrame(
        [[0.01, 0.002, 0.0], [0.002, 0.04, 0.005], [0.0, 0.005, 0.09]],
        index=mu.index,
        columns=mu.index,
    )
    return mu, cov


def test_min_variance_is_valid_and_lowest(inputs):
    mu, cov = inputs
    w = optimization.min_variance(mu, cov)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    equal = np.full(3, 1 / 3)
    assert optimization.portfolio_stats(w.to_numpy(), mu.to_numpy(), cov.to_numpy())[1] <= (
        optimization.portfolio_stats(equal, mu.to_numpy(), cov.to_numpy())[1] + 1e-9
    )


def test_max_sharpe_beats_equal_weight(inputs):
    mu, cov = inputs
    w = optimization.max_sharpe(mu, cov, 0.02)
    assert w.sum() == pytest.approx(1.0)
    best = optimization.portfolio_stats(w.to_numpy(), mu.to_numpy(), cov.to_numpy(), 0.02)[2]
    equal = optimization.portfolio_stats(np.full(3, 1 / 3), mu.to_numpy(), cov.to_numpy(), 0.02)[2]
    assert best >= equal - 1e-9


def test_frontier_is_monotonic_and_long_only(inputs):
    mu, cov = inputs
    f = optimization.efficient_frontier(mu, cov, n_points=15)
    assert f["Volatility"].is_monotonic_increasing or (f["Volatility"].diff().dropna() > -1e-6).all()
    weights = f[list(mu.index)]
    assert (weights >= -1e-9).all().all()
    assert weights.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(f)))


def test_needs_two_assets():
    mu = pd.Series([0.1], index=["A"])
    cov = pd.DataFrame([[0.01]], index=["A"], columns=["A"])
    with pytest.raises(ValueError):
        optimization.min_variance(mu, cov)
