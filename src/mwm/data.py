from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .features import add_bar_features, feature_columns


@dataclass
class SamplePack:
    daily: np.ndarray
    intraday: np.ndarray
    symbol_id: np.ndarray
    regime_next_mask: np.ndarray
    future_returns: np.ndarray
    timestamps_ns: np.ndarray
    symbols: list[str]


def _load_bars(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"symbol", timestamp_col, "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df.rename(columns={timestamp_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def _merge_daily_indicators(daily: pd.DataFrame, path: str, columns: list[str]) -> pd.DataFrame:
    """Join per-day indicators (symbol,date,...) onto daily bars by symbol and New York calendar date."""
    ind = pd.read_parquet(path)
    missing = set(columns) - set(ind.columns)
    if missing:
        raise ValueError(f"Missing indicator columns in {path}: {sorted(missing)}")
    ind = ind[["symbol", "date", *columns]].copy()
    ind["date"] = pd.to_datetime(ind["date"]).dt.date
    daily = daily.copy()
    daily["date"] = daily["timestamp"].dt.tz_convert("America/New_York").dt.date
    out = daily.merge(ind, on=["symbol", "date"], how="left").drop(columns="date")
    out[columns] = out[columns].replace([np.inf, -np.inf], np.nan)
    return out


def build_samples(
    daily_path: str,
    intraday_path: str,
    daily_lookback: int = 40,
    intraday_lookback: int = 455,
    horizons_5m: tuple[int, ...] = (1, 3, 12, 78),
    min_price: float = 1.0,
    daily_indicators_path: str | None = None,
    daily_indicator_columns: list[str] | None = None,
) -> SamplePack:
    """
    Build aligned samples ending on each 5-minute bar.
    Daily context uses completed daily bars strictly before the sample date.
    Intraday context uses bars up to and including t.
    Targets are future log returns from close_t to close_{t+h}.
    Optional daily indicators are appended to the daily features only; a day's indicators
    are computed from that day's close, so the strictly-before cutoff keeps them causal.
    """
    daily = add_bar_features(_load_bars(daily_path))
    intra = add_bar_features(_load_bars(intraday_path))
    feats = feature_columns()
    daily_feats = list(feats)
    if daily_indicators_path and daily_indicator_columns:
        daily = _merge_daily_indicators(daily, daily_indicators_path, list(daily_indicator_columns))
        daily_feats += list(daily_indicator_columns)
    symbols = sorted(set(daily.symbol.unique()) & set(intra.symbol.unique()))
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    daily_windows, intra_windows, ids, rets, ts = [], [], [], [], []
    max_h = max(horizons_5m)

    dmap = {s: g.reset_index(drop=True) for s, g in daily.groupby("symbol") if s in symbol_to_id}
    imap = {s: g.reset_index(drop=True) for s, g in intra.groupby("symbol") if s in symbol_to_id}

    for symbol in symbols:
        dg = dmap[symbol]
        ig = imap[symbol]
        if len(ig) < intraday_lookback + max_h:
            continue
        daily_dates = dg["timestamp"].dt.tz_convert("America/New_York").dt.date.to_numpy()
        intra_dates = ig["timestamp"].dt.tz_convert("America/New_York").dt.date.to_numpy()
        dvals = dg[daily_feats].to_numpy(np.float32)
        ivals = ig[feats].to_numpy(np.float32)
        closes = ig["close"].to_numpy(np.float64)

        for i in range(intraday_lookback - 1, len(ig) - max_h):
            if closes[i] < min_price:
                continue
            cutoff_date = intra_dates[i]
            d_end = np.searchsorted(daily_dates, cutoff_date, side="left")
            d_start = d_end - daily_lookback
            if d_start < 0:
                continue
            dw = dvals[d_start:d_end]
            iw = ivals[i - intraday_lookback + 1 : i + 1]
            if not (np.isfinite(dw).all() and np.isfinite(iw).all()):
                continue
            y = [np.log(closes[i + h] / closes[i]) for h in horizons_5m]
            if not np.isfinite(y).all():
                continue
            daily_windows.append(dw)
            intra_windows.append(iw)
            ids.append(symbol_to_id[symbol])
            rets.append(y)
            ts.append(ig["timestamp"].iloc[i].value)

    if not daily_windows:
        raise RuntimeError("No valid samples were built. Check history depth and feature warmup.")

    arr_ts = np.asarray(ts, dtype=np.int64)
    return SamplePack(
        daily=np.asarray(daily_windows, dtype=np.float32),
        intraday=np.asarray(intra_windows, dtype=np.float32),
        symbol_id=np.asarray(ids, dtype=np.int64),
        regime_next_mask=np.ones(len(ids), dtype=np.float32),
        future_returns=np.asarray(rets, dtype=np.float32),
        timestamps_ns=arr_ts,
        symbols=symbols,
    )


def save_pack(pack: SamplePack, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        daily=pack.daily,
        intraday=pack.intraday,
        symbol_id=pack.symbol_id,
        regime_next_mask=pack.regime_next_mask,
        future_returns=pack.future_returns,
        timestamps_ns=pack.timestamps_ns,
        symbols=np.asarray(pack.symbols, dtype=object),
    )


class WorldModelDataset(Dataset):
    def __init__(self, npz_path: str, start: str, end: str, stats: dict | None = None):
        z = np.load(npz_path, allow_pickle=True)
        ts = pd.to_datetime(z["timestamps_ns"], utc=True)
        start, end = str(start), str(end)  # YAML parses unquoted dates as datetime.date
        mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end + " 23:59:59", tz="UTC"))
        idx = np.flatnonzero(mask)
        self.daily = z["daily"][idx]
        self.intraday = z["intraday"][idx]
        self.symbol_id = z["symbol_id"][idx]
        self.returns = z["future_returns"][idx]
        self.timestamps_ns = z["timestamps_ns"][idx]
        self.symbols = z["symbols"].tolist()
        self.stats = stats
        if stats is not None:
            self.daily = (self.daily - stats["daily_mean"]) / stats["daily_std"]
            self.intraday = (self.intraday - stats["intra_mean"]) / stats["intra_std"]

        # Pair each sample with the immediately following 5-minute sample of the same symbol.
        # This keeps chronology intact even when DataLoader shuffles dataset items.
        self.next_idx = np.full(len(self.symbol_id), -1, dtype=np.int64)
        if len(self.symbol_id) > 1:
            same = self.symbol_id[:-1] == self.symbol_id[1:]
            dt = self.timestamps_ns[1:] - self.timestamps_ns[:-1]
            five_min_ns = 5 * 60 * 1_000_000_000
            valid = same & (dt == five_min_ns)
            self.next_idx[:-1][valid] = np.arange(1, len(self.symbol_id))[valid]
        self.pair_rows = np.flatnonzero(self.next_idx >= 0)

    def __len__(self): return len(self.pair_rows)

    def __getitem__(self, row):
        i = int(self.pair_rows[row]); j = int(self.next_idx[i])
        return {
            "daily": torch.from_numpy(self.daily[i]).float(),
            "intraday": torch.from_numpy(self.intraday[i]).float(),
            "symbol_id": torch.tensor(self.symbol_id[i], dtype=torch.long),
            "future_returns": torch.from_numpy(self.returns[i]).float(),
            "timestamp_ns": torch.tensor(self.timestamps_ns[i], dtype=torch.long),
            "next_daily": torch.from_numpy(self.daily[j]).float(),
            "next_intraday": torch.from_numpy(self.intraday[j]).float(),
        }


def fit_stats(ds: WorldModelDataset) -> dict:
    dmean = ds.daily.mean(axis=(0, 1), keepdims=True)
    dstd = ds.daily.std(axis=(0, 1), keepdims=True) + 1e-6
    imean = ds.intraday.mean(axis=(0, 1), keepdims=True)
    istd = ds.intraday.std(axis=(0, 1), keepdims=True) + 1e-6
    return {"daily_mean": dmean, "daily_std": dstd, "intra_mean": imean, "intra_std": istd}
