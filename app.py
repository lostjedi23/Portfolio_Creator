"""Streamlit entry point: streamlit run app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portfolio import analytics, charts, data, export, optimization

MIN_OBSERVATIONS = 30

st.set_page_config(page_title="Portfolio Analytics Engine", layout="wide")
st.title("Portfolio Analytics & Correlation Engine")
st.caption("Asset allocation, returns analysis, correlation mapping and optimization.")


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple[str, ...], years: int) -> pd.DataFrame:
    return data.fetch_prices(list(tickers), years)


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Portfolio Configuration")

ticker_input = st.sidebar.text_input(
    "Stock/ETF tickers (comma separated)", value="SPY, QQQ, VT, BND, GLD"
)
tickers = data.parse_tickers(ticker_input)

lookback_years = st.sidebar.slider("Historical lookback (years)", 1, 10, 3)
risk_free_pct = st.sidebar.number_input(
    "Risk free rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.25
)
risk_free = risk_free_pct / 100.0

rebalance_label = st.sidebar.selectbox(
    "Rebalancing", list(analytics.REBALANCE_OPTIONS), index=0
)
rebalance = analytics.REBALANCE_OPTIONS[rebalance_label]

benchmark = data.parse_tickers(st.sidebar.text_input("Benchmark ticker", value="SPY"))[:1]

st.sidebar.subheader("Asset weights (%)")
weights_pct: dict[str, float] = {}
defaults = analytics.equal_weights_pct(len(tickers))
for ticker, default in zip(tickers, defaults):
    weights_pct[ticker] = st.sidebar.number_input(
        f"{ticker}",
        min_value=0.0,
        max_value=100.0,
        value=default,
        step=1.0,
        format="%.2f",
        key=f"weight_{len(tickers)}_{ticker}",
    )
total_weight = sum(weights_pct.values())
st.sidebar.markdown(f"**Total weight:** {total_weight:.2f}%")
weights_ok = bool(tickers) and abs(total_weight - 100.0) <= 0.01
if tickers and not weights_ok:
    st.sidebar.error("Total weights must equal 100%.")

if not weights_ok:
    st.info("Enter tickers and make the weights add up to 100% to see results.")
    st.stop()

# ---------------------------------------------------------------- data
request = tuple(dict.fromkeys([*tickers, *benchmark]))
try:
    with st.spinner("Fetching market data from Yahoo Finance..."):
        prices = load_prices(request, lookback_years)
except data.DataError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # network or parsing failures from yfinance
    st.error(f"Could not download market data: {exc}")
    st.stop()

missing = [t for t in request if t not in prices.columns]
if any(t in missing for t in tickers):
    st.error(f"No data found for: {', '.join(t for t in tickers if t in missing)}")
    st.stop()

bench_ticker = benchmark[0] if benchmark and benchmark[0] not in missing else None
if benchmark and bench_ticker is None:
    st.warning(f"No data found for benchmark {benchmark[0]}. Skipping the comparison.")

prices = data.align_prices(prices)
returns_all = analytics.daily_returns(prices)
if len(returns_all) < MIN_OBSERVATIONS:
    st.error(
        f"Only {len(returns_all)} overlapping trading days found. "
        f"Need at least {MIN_OBSERVATIONS}. Try fewer tickers or a longer lookback."
    )
    st.stop()

returns = returns_all[tickers]
st.caption(
    f"{len(returns)} trading days, {returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d} "
    "(dates where every asset has a price)."
)

weight_vector = np.array([weights_pct[t] / 100.0 for t in tickers])
portfolio_returns = analytics.simulate_portfolio(returns, weight_vector, rebalance)

# ---------------------------------------------------------------- calculations
metrics = analytics.asset_metrics(returns, risk_free)
corr = analytics.correlation_matrix(returns)
cov = analytics.covariance_matrix(returns)
cum_portfolio = analytics.cumulative_growth(portfolio_returns)
portfolio_stats = analytics.series_summary(portfolio_returns, risk_free)

summary = pd.DataFrame({"Portfolio": portfolio_stats})
growth_series = {"Portfolio": cum_portfolio}
drawdown_series = {"Portfolio": analytics.drawdown(portfolio_returns)}
bench_returns = None
if bench_ticker:
    bench_returns = returns_all[bench_ticker]
    summary[bench_ticker] = analytics.series_summary(bench_returns, risk_free)
    growth_series[bench_ticker] = analytics.cumulative_growth(bench_returns)
    drawdown_series[bench_ticker] = analytics.drawdown(bench_returns)

pct_rows = ["Total Return", "Annualized Return (CAGR)", "Annualized Volatility", "Max Drawdown"]

# ---------------------------------------------------------------- layout
tab_overview, tab_corr, tab_frontier, tab_export = st.tabs(
    ["Overview", "Correlation", "Efficient Frontier", "Export"]
)

with tab_overview:
    st.subheader("Portfolio statistics")
    styled = summary.style.format("{:.2%}", subset=pd.IndexSlice[pct_rows, :]).format(
        "{:.2f}", subset=pd.IndexSlice[["Sharpe Ratio"], :]
    )
    st.dataframe(styled, width="stretch")
    if bench_returns is not None:
        st.caption(
            f"Beta vs {bench_ticker}: {analytics.beta(portfolio_returns, bench_returns):.2f}. "
            f"Rebalancing: {rebalance_label.lower()}. Risk free rate: {risk_free_pct:.2f}%."
        )

    st.subheader("Cumulative performance")
    st.plotly_chart(charts.growth_chart(growth_series), width="stretch")

    st.subheader("Drawdown")
    st.plotly_chart(charts.drawdown_chart(drawdown_series), width="stretch")

    st.subheader("Asset performance")
    asset_styled = metrics.style.format(
        {
            "Annualized Return (CAGR)": "{:.2%}",
            "Annualized Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
        }
    )
    st.dataframe(asset_styled, width="stretch")

with tab_corr:
    st.subheader("Correlation matrix")
    st.plotly_chart(charts.correlation_heatmap(corr), width="stretch")
    with st.expander("Annualized covariance matrix"):
        st.dataframe(cov.style.format("{:.4f}"), width="stretch")

with tab_frontier:
    st.subheader("Efficient frontier (long only, fully invested)")
    if len(tickers) < 2:
        st.info("Add at least two tickers to build a frontier.")
    else:
        mu = metrics["Annualized Return (CAGR)"]
        try:
            frontier = optimization.efficient_frontier(mu, cov)
            w_min = optimization.min_variance(mu, cov)
            w_sharpe = optimization.max_sharpe(mu, cov, risk_free)
        except optimization.OptimizationError as exc:
            st.error(f"Optimizer failed: {exc}")
        else:
            m, c = mu.to_numpy(), cov.to_numpy()
            current = optimization.portfolio_stats(weight_vector, m, c, risk_free)
            min_stats = optimization.portfolio_stats(w_min.to_numpy(), m, c, risk_free)
            sharpe_stats = optimization.portfolio_stats(w_sharpe.to_numpy(), m, c, risk_free)
            fig = charts.frontier_chart(
                frontier,
                pd.DataFrame(
                    {"Return": mu, "Volatility": metrics["Annualized Volatility"]}
                ),
                {
                    "Your portfolio": (current[1], current[0]),
                    "Min variance": (min_stats[1], min_stats[0]),
                    "Max Sharpe": (sharpe_stats[1], sharpe_stats[0]),
                },
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Frontier points use each asset's historical CAGR and covariance over the "
                "selected window and assume constant weights. Past data does not predict returns."
            )
            table = pd.DataFrame(
                {
                    "Your portfolio": [*weight_vector, *current],
                    "Min variance": [*w_min, *min_stats],
                    "Max Sharpe": [*w_sharpe, *sharpe_stats],
                },
                index=[*tickers, "Return", "Volatility", "Sharpe Ratio"],
            )
            weight_rows, stat_rows = tickers, ["Return", "Volatility"]
            st.dataframe(
                table.style.format("{:.2%}", subset=pd.IndexSlice[weight_rows + stat_rows, :]).format(
                    "{:.2f}", subset=pd.IndexSlice[["Sharpe Ratio"], :]
                ),
                width="stretch",
            )

with tab_export:
    st.subheader("Download data")
    downloads = {
        "Asset metrics": ("asset_metrics.csv", metrics),
        "Portfolio statistics": ("portfolio_statistics.csv", summary),
        "Correlation matrix": ("correlation_matrix.csv", corr),
        "Covariance matrix (annualized)": ("covariance_matrix.csv", cov),
        "Daily returns": ("daily_returns.csv", returns.assign(Portfolio=portfolio_returns)),
        "Cumulative portfolio return": (
            "cumulative_return.csv",
            cum_portfolio.rename("Cumulative Return"),
        ),
    }
    for label, (filename, frame) in downloads.items():
        st.download_button(
            f"Download {label}",
            data=export.to_csv_bytes(frame),
            file_name=filename,
            mime="text/csv",
            key=f"dl_{filename}",
        )
