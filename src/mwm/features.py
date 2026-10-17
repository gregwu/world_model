from __future__ import annotations
import numpy as np
import pandas as pd

BASE_COLUMNS = ["open", "high", "low", "close", "volume"]


def add_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal features only. Input must be sorted by symbol,timestamp."""
    x = df.copy()
    g = x.groupby("symbol", group_keys=False)
    prev_close = g["close"].shift(1)
    x["log_return"] = np.log(x["close"] / prev_close)
    x["range_pct"] = (x["high"] - x["low"]) / prev_close
    x["body_pct"] = (x["close"] - x["open"]) / prev_close
    span = (x["high"] - x["low"]).replace(0, np.nan)
    x["close_location"] = (x["close"] - x["low"]) / span
    x["log_volume"] = np.log1p(x["volume"].clip(lower=0))

    for w in (5, 20, 50):
        ma = g["close"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        x[f"close_to_sma_{w}"] = x["close"] / ma - 1.0
        vol_ma = g["volume"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        x[f"relvol_{w}"] = x["volume"] / vol_ma.replace(0, np.nan)

    tr1 = x["high"] - x["low"]
    tr2 = (x["high"] - prev_close).abs()
    tr3 = (x["low"] - prev_close).abs()
    x["true_range_pct"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) / prev_close
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def feature_columns() -> list[str]:
    return [
        "log_return", "range_pct", "body_pct", "close_location", "log_volume",
        "close_to_sma_5", "close_to_sma_20", "close_to_sma_50",
        "relvol_5", "relvol_20", "relvol_50", "true_range_pct",
    ]
