import numpy as np
import pandas as pd
import pytest

from portfolio import data


def test_parse_tickers_dedupes_and_uppercases():
    assert data.parse_tickers(" spy, qqq ,SPY,, vt ") == ["SPY", "QQQ", "VT"]


def test_extract_close_multiindex_keeps_requested_order():
    idx = pd.bdate_range("2023-01-02", periods=3)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAA", "BBB"]])
    raw = pd.DataFrame(np.ones((3, 4)), index=idx, columns=cols)
    out = data.extract_close(raw, ["BBB", "AAA"])
    assert list(out.columns) == ["BBB", "AAA"]


def test_extract_close_drops_all_nan_columns():
    idx = pd.bdate_range("2023-01-02", periods=3)
    cols = pd.MultiIndex.from_product([["Close"], ["AAA", "BAD"]])
    raw = pd.DataFrame({("Close", "AAA"): [1.0, 2.0, 3.0], ("Close", "BAD"): [np.nan] * 3}, index=idx)
    raw.columns = cols
    assert list(data.extract_close(raw, ["AAA", "BAD"]).columns) == ["AAA"]


def test_extract_close_empty_raises():
    with pytest.raises(data.DataError):
        data.extract_close(pd.DataFrame(), ["AAA"])


def test_align_prices_drops_partial_rows():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [np.nan, 2.0, 3.0]})
    assert len(data.align_prices(df)) == 2
