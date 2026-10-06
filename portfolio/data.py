"""Market data loading and cleaning. No Streamlit imports, so it stays testable."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


class DataError(Exception):
    """Raised when market data is missing or unusable."""


def parse_tickers(text: str) -> list[str]:
    """Split a comma separated string into unique, uppercase tickers (order kept)."""
    seen: dict[str, None] = {}
    for part in text.split(","):
        ticker = part.strip().upper()
        if ticker:
            seen.setdefault(ticker, None)
    return list(seen)


def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the price columns out of a yfinance download.

    yfinance returns split and dividend adjusted prices under "Close" when
    auto_adjust=True. With several tickers the columns are a MultiIndex.
    """
    if raw is None or raw.empty:
        raise DataError("Yahoo Finance returned no data. Check the tickers and try again.")
    if "Close" not in raw.columns.get_level_values(0):
        raise DataError("Yahoo Finance response had no Close prices.")

    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.dropna(axis=1, how="all")
    ordered = [t for t in tickers if t in close.columns]
    return close[ordered]


def align_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep only the dates where every asset has a price."""
    return prices.dropna(how="any")


def fetch_prices(tickers: list[str], years: int, end: datetime | None = None) -> pd.DataFrame:
    """Download adjusted daily closes for `tickers` over the last `years` years."""
    if not tickers:
        raise DataError("Enter at least one ticker.")
    end = end or datetime.today()
    start = end - timedelta(days=years * 365)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    return extract_close(raw, tickers)
