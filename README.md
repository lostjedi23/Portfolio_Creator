# Portfolio_Creator

A Streamlit app for portfolio analytics. Enter tickers and weights, then see returns, volatility, correlations, drawdowns, an efficient frontier, and CSV exports.

## Layout

```
app.py                   Streamlit UI (sidebar, tabs, downloads)
portfolio/data.py        Yahoo Finance download and cleaning
portfolio/analytics.py   Returns, volatility, Sharpe, covariance, correlation, rebalancing, drawdown
portfolio/optimization.py  Min variance, max Sharpe, efficient frontier (long only, fully invested)
portfolio/charts.py      Plotly figures
portfolio/export.py      CSV helpers
tests/                   pytest unit tests
```

## Run it

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `pytest`.

## Math notes

- Daily return: `R_t = (P_t - P_{t-1}) / P_{t-1}`, using split and dividend adjusted closes.
- Annualized return is geometric (CAGR): `(prod(1 + R_t)) ^ (252 / n) - 1`.
- Annualized volatility: sample standard deviation (n - 1) times `sqrt(252)`.
- Sharpe ratio: `(CAGR - Rf) / volatility`.
- Covariance is annualized by 252. Correlation is covariance divided by the product of standard deviations.
- Portfolio returns follow your rebalancing choice. Daily rebalancing holds weights fixed every day. Monthly, quarterly and annual rebalancing reset the weights at the start of each new period. Buy and hold lets weights drift.
- The efficient frontier uses each asset's historical CAGR and covariance, long only, weights summing to 100%.
- Only dates where every asset has a price are used, so a young ETF shortens the window for all assets.
