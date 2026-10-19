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
    span = x["high"] - x["low"]
    # Flat bars (high == low; ~1% of 5m bars) are mid-range. Leaving them NaN dropped every 455-bar window that
    # contained one, which removed about a third of all 5m samples.
    x["close_location"] = ((x["close"] - x["low"]) / span.replace(0, np.nan)).fillna(0.5).where(span.notna())
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


# --- Intraday technical indicators on 5m and 1h bars --------------------------------------------------------

NY = "America/New_York"
SESSION_OPEN_MIN = 9 * 60 + 30
HIGHER_TIMEFRAMES = (("h1", 60),)  # minutes, aligned to the 09:30 session open


def _pctb(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """Bollinger %B: 0 at the lower band, 1 at the upper band."""
    mid = close.rolling(n, min_periods=n).mean(); sd = close.rolling(n, min_periods=n).std(ddof=0)
    return (close - (mid - k * sd)) / (2 * k * sd)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI, scaled 0-1."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return up / (up + dn)


def _macd_hist_pct(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (MACD - signal) as a fraction of price, so it is comparable across symbols."""
    macd = (close.ewm(span=fast, adjust=False, min_periods=fast).mean()
            - close.ewm(span=slow, adjust=False, min_periods=slow).mean())
    return (macd - macd.ewm(span=signal, adjust=False, min_periods=signal).mean()) / close


def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add %B, RSI, MACD histogram and session-VWAP distance on 5m bars, and %B, RSI, MACD histogram on 1h bars,
    for ONE symbol's regular-session 5m bars sorted by time.

    1h bars are built from the 5m bars, aligned to the 09:30 open (a session has seven 1h bars, the last 30 min).
    Each 5m bar sees the indicators of the last COMPLETED 1h bar, never the one still forming, so every feature is
    causal.
    """
    x = df.copy()
    c, h, l, v = x["close"], x["high"], x["low"], x["volume"].clip(lower=0)
    et = x["timestamp"].dt.tz_convert(NY)
    session = et.dt.date
    x["m5_pctb"] = _pctb(c); x["m5_rsi"] = _rsi(c); x["m5_macd_hist"] = _macd_hist_pct(c)
    typical = (h + l + c) / 3
    vwap = (typical * v).groupby(session).cumsum() / v.groupby(session).cumsum().replace(0, np.nan)
    x["m5_vwap_dist"] = c / vwap - 1.0

    minutes = et.dt.hour * 60 + et.dt.minute - SESSION_OPEN_MIN
    for name, width in HIGHER_TIMEFRAMES:
        bucket = pd.Series(pd.factorize(session.astype(str) + "_" + (minutes // width).astype(str))[0], index=x.index)
        hc = c.groupby(bucket).last()  # bucket ids increase with time because the bars are sorted
        ind = pd.DataFrame({"pctb": _pctb(hc), "rsi": _rsi(hc), "macd_hist": _macd_hist_pct(hc)}).shift(1)  # completed only
        for col in ind.columns:
            x[f"{name}_{col}"] = ind[col].reindex(bucket.to_numpy()).to_numpy()
    return x.replace([np.inf, -np.inf], np.nan)


def intraday_indicator_columns() -> list[str]:
    cols = ["m5_pctb", "m5_rsi", "m5_macd_hist", "m5_vwap_dist"]
    for name, _ in HIGHER_TIMEFRAMES:
        cols += [f"{name}_pctb", f"{name}_rsi", f"{name}_macd_hist"]
    return cols
