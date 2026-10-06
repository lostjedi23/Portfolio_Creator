"""CSV export helpers."""
from __future__ import annotations

import pandas as pd


def to_csv_bytes(data: pd.DataFrame | pd.Series) -> bytes:
    return data.to_csv().encode("utf-8")
