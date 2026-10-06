import numpy as np
import pandas as pd
import pytest

from portfolio import analytics


def make_returns(n=300, seed=0, cols=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (n, len(cols))), index=idx, columns=list(cols))


def test_daily_returns_formula():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    out = analytics.daily_returns(prices)
    assert out["A"].tolist() == pytest.approx([0.10, -0.10])


def test_cagr_matches_total_growth():
    r = pd.Series([0.01] * 252)
    assert analytics.annualized_return(r) == pytest.approx(1.01**252 - 1)


def test_cagr_half_year_scaling():
    r = pd.Series([0.001] * 126)
    assert analytics.annualized_return(r) == pytest.approx((1.001**126) ** 2 - 1)


def test_volatility_uses_sample_std():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    assert analytics.annualized_volatility(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(analytics.sharpe_ratio(0.1, 0.0))
    s = analytics.sharpe_ratio(pd.Series([0.1, 0.2]), pd.Series([0.0, 0.1]), 0.05)
    assert np.isnan(s.iloc[0]) and s.iloc[1] == pytest.approx(1.5)


def test_correlation_equals_cov_over_std_product():
    r = make_returns()
    cov = analytics.covariance_matrix(r, annualize=False)
    std = np.sqrt(np.diag(cov))
    expected = cov.to_numpy() / np.outer(std, std)
    assert analytics.correlation_matrix(r).to_numpy() == pytest.approx(expected)


def test_annualized_cov_scaled_by_252():
    r = make_returns()
    assert analytics.covariance_matrix(r).to_numpy() == pytest.approx(r.cov().to_numpy() * 252)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 13])
def test_equal_weights_sum_to_100(n):
    w = analytics.equal_weights_pct(n)
    assert len(w) == n
    assert sum(w) == pytest.approx(100.0)


def test_daily_rebalance_equals_weighted_sum():
    r = make_returns()
    w = np.array([0.5, 0.3, 0.2])
    out = analytics.simulate_portfolio(r, w, "daily")
    assert out.to_numpy() == pytest.approx((r.to_numpy() @ w))


def test_buy_and_hold_matches_weighted_final_values():
    r = make_returns()
    w = np.array([0.5, 0.3, 0.2])
    out = analytics.simulate_portfolio(r, w, "none")
    total = (1 + out).prod()
    expected = float(w @ (1 + r).prod().to_numpy())
    assert total == pytest.approx(expected)


def test_single_asset_all_policies_equal():
    r = make_returns(cols=("A",))
    for policy in analytics.REBALANCE_OPTIONS.values():
        out = analytics.simulate_portfolio(r, np.array([1.0]), policy)
        assert out.to_numpy() == pytest.approx(r["A"].to_numpy())


def test_rebalance_policies_differ():
    r = make_returns(n=600)
    w = np.array([0.6, 0.3, 0.1])
    daily = (1 + analytics.simulate_portfolio(r, w, "daily")).prod()
    hold = (1 + analytics.simulate_portfolio(r, w, "none")).prod()
    assert daily != pytest.approx(hold)


def test_monthly_rebalance_resets_on_month_change():
    idx = pd.to_datetime(["2023-01-30", "2023-01-31", "2023-02-01", "2023-02-02"])
    r = pd.DataFrame({"A": [0.10, 0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.10, 0.0]}, index=idx)
    out = analytics.simulate_portfolio(r, np.array([0.5, 0.5]), "monthly")
    # Day 2 (Jan 31): weights drifted to 0.55/0.45 after A gained 10%, but A is flat so no effect.
    # Feb 1: reset to 50/50, so B's 10% gain contributes exactly 5%.
    assert out.iloc[2] == pytest.approx(0.05)


def test_unknown_rebalance_raises():
    with pytest.raises(ValueError):
        analytics.simulate_portfolio(make_returns(), np.array([0.5, 0.3, 0.2]), "weekly")


def test_drawdown_and_max_drawdown():
    r = pd.Series([0.10, -0.20, 0.05], index=pd.bdate_range("2023-01-02", periods=3))
    dd = analytics.drawdown(r)
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == pytest.approx(-0.20)
    assert analytics.max_drawdown(r) == pytest.approx(-0.20)


def test_drawdown_counts_initial_value_as_peak():
    r = pd.Series([-0.10, 0.02])
    assert analytics.max_drawdown(r) == pytest.approx(-0.10)


def test_beta_of_scaled_series():
    r = make_returns(cols=("A",))["A"]
    assert analytics.beta(2 * r, r) == pytest.approx(2.0)


def test_series_summary_keys():
    s = analytics.series_summary(make_returns(cols=("A",))["A"], 0.04)
    assert set(s) == {
        "Total Return",
        "Annualized Return (CAGR)",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
    }
