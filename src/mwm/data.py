from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from .features import add_bar_features, feature_columns

NORM_CLIP = 10.0  # normalized features are clipped to +-NORM_CLIP standard deviations


@dataclass
class SamplePack:
    daily: np.ndarray
    intraday: np.ndarray | None  # None for daily-only packs
    symbol_id: np.ndarray
    regime_next_mask: np.ndarray
    future_returns: np.ndarray
    timestamps_ns: np.ndarray
    symbols: list[str]
    bar_index: np.ndarray  # row of the sample's last bar within its symbol's series; consecutive => adjacent bars


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


def _utc_ns(ts: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(ts).tz_convert("UTC").as_unit("ns").asi8


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


def _load_daily(daily_path, daily_indicators_path, daily_indicator_columns):
    daily = add_bar_features(_load_bars(daily_path))
    daily_feats = list(feature_columns())
    if daily_indicators_path and daily_indicator_columns:
        daily = _merge_daily_indicators(daily, daily_indicators_path, list(daily_indicator_columns))
        daily_feats += list(daily_indicator_columns)
    return daily, daily_feats


def _pack(daily_windows, intra_windows, ids, rets, ts, bars, symbols) -> SamplePack:
    if not daily_windows:
        raise RuntimeError("No valid samples were built. Check history depth and feature warmup.")
    return SamplePack(
        daily=np.asarray(daily_windows, dtype=np.float32),
        intraday=np.asarray(intra_windows, dtype=np.float32) if intra_windows is not None else None,
        symbol_id=np.asarray(ids, dtype=np.int64),
        regime_next_mask=np.ones(len(ids), dtype=np.float32),
        future_returns=np.asarray(rets, dtype=np.float32),
        timestamps_ns=np.asarray(ts, dtype=np.int64),
        symbols=symbols,
        bar_index=np.asarray(bars, dtype=np.int64),
    )


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
    daily, daily_feats = _load_daily(daily_path, daily_indicators_path, daily_indicator_columns)
    intra = add_bar_features(_load_bars(intraday_path))
    feats = feature_columns()
    symbols = sorted(set(daily.symbol.unique()) & set(intra.symbol.unique()))
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    daily_windows, intra_windows, ids, rets, ts, bars = [], [], [], [], [], []
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
        stamps = _utc_ns(ig["timestamp"])

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
            ts.append(stamps[i])
            bars.append(i)

    return _pack(daily_windows, intra_windows, ids, rets, ts, bars, symbols)


def build_daily_samples(
    daily_path: str,
    daily_lookback: int = 40,
    horizons_days: tuple[int, ...] = (1, 5, 20),
    min_price: float = 1.0,
    daily_indicators_path: str | None = None,
    daily_indicator_columns: list[str] | None = None,
) -> SamplePack:
    """
    Daily-only samples ending on each daily bar t (after its close).
    Context uses bars up to and including t; targets are log returns close_t -> close_{t+h} in trading days.
    """
    daily, daily_feats = _load_daily(daily_path, daily_indicators_path, daily_indicator_columns)
    symbols = sorted(daily.symbol.unique())
    daily_windows, ids, rets, ts, bars = [], [], [], [], []
    max_h = max(horizons_days)

    for sid, (symbol, dg) in enumerate(daily.groupby("symbol", sort=True)):
        if len(dg) < daily_lookback + max_h:
            continue
        dvals = dg[daily_feats].to_numpy(np.float32)
        closes = dg["close"].to_numpy(np.float64)
        stamps = _utc_ns(dg["timestamp"])
        finite_row = np.isfinite(dvals).all(axis=1)
        # bad[i] = number of non-finite rows in the window ending at i
        bad = np.convolve(~finite_row, np.ones(daily_lookback, dtype=int))[: len(dvals)]

        for i in range(daily_lookback - 1, len(dg) - max_h):
            if closes[i] < min_price or bad[i]:
                continue
            y = [np.log(closes[i + h] / closes[i]) for h in horizons_days]
            if not np.isfinite(y).all():
                continue
            daily_windows.append(dvals[i - daily_lookback + 1 : i + 1])
            ids.append(sid)
            rets.append(y)
            ts.append(stamps[i])
            bars.append(i)

    return _pack(daily_windows, None, ids, rets, ts, bars, symbols)


def save_pack(pack: SamplePack, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        daily=pack.daily,
        symbol_id=pack.symbol_id,
        regime_next_mask=pack.regime_next_mask,
        future_returns=pack.future_returns,
        timestamps_ns=pack.timestamps_ns,
        bar_index=pack.bar_index,
        symbols=np.asarray(pack.symbols, dtype=object),
    )
    if pack.intraday is not None:
        arrays["intraday"] = pack.intraday
    np.savez_compressed(path, **arrays)


def _normalize(x, mean, std):
    return np.clip((x - mean) / std, -NORM_CLIP, NORM_CLIP).astype(np.float32)


class WorldModelDataset(Dataset):
    def __init__(self, npz_path: str, start: str, end: str, stats: dict | None = None):
        start, end = str(start), str(end)  # YAML parses unquoted dates as datetime.date
        z = np.load(npz_path, allow_pickle=True)
        ts = pd.to_datetime(z["timestamps_ns"], utc=True)
        mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts <= pd.Timestamp(end + " 23:59:59", tz="UTC"))
        idx = np.flatnonzero(mask)
        self.daily = z["daily"][idx]
        self.intraday = z["intraday"][idx] if "intraday" in z.files else None
        self.symbol_id = z["symbol_id"][idx]
        self.returns = z["future_returns"][idx]
        self.timestamps_ns = z["timestamps_ns"][idx]
        bar_index = z["bar_index"][idx] if "bar_index" in z.files else None
        self.symbols = z["symbols"].tolist()
        self.stats = stats
        if stats is not None:
            self.daily = _normalize(self.daily, stats["daily_mean"], stats["daily_std"])
            if self.intraday is not None:
                self.intraday = _normalize(self.intraday, stats["intra_mean"], stats["intra_std"])

        # Pair each sample with the immediately following sample (next bar) of the same symbol.
        # This keeps chronology intact even when DataLoader shuffles dataset items.
        self.next_idx = np.full(len(self.symbol_id), -1, dtype=np.int64)
        if len(self.symbol_id) > 1:
            same = self.symbol_id[:-1] == self.symbol_id[1:]
            if bar_index is not None:
                adjacent = bar_index[1:] - bar_index[:-1] == 1
            else:  # packs built before bar_index existed: 5-minute samples only
                adjacent = self.timestamps_ns[1:] - self.timestamps_ns[:-1] == 5 * 60 * 1_000_000_000
            valid = same & adjacent
            self.next_idx[:-1][valid] = np.arange(1, len(self.symbol_id))[valid]
        self.pair_rows = np.flatnonzero(self.next_idx >= 0)

    def __len__(self): return len(self.pair_rows)

    def __getitems__(self, rows):
        """Batched fetch: one fancy-index per array instead of one small copy per sample."""
        i = self.pair_rows[np.asarray(rows)]; j = self.next_idx[i]
        b = {
            "daily": torch.from_numpy(self.daily[i]),
            "symbol_id": torch.from_numpy(self.symbol_id[i]),
            "future_returns": torch.from_numpy(self.returns[i]),
            "timestamp_ns": torch.from_numpy(self.timestamps_ns[i]),
            "next_daily": torch.from_numpy(self.daily[j]),
        }
        if self.intraday is not None:
            b["intraday"] = torch.from_numpy(self.intraday[i])
            b["next_intraday"] = torch.from_numpy(self.intraday[j])
        return b

    def __getitem__(self, row):
        return {k: v[0] for k, v in self.__getitems__([row]).items()}


def _batch_passthrough(batch):
    return batch  # __getitems__ already returns a collated batch


def make_loader(ds: WorldModelDataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=_batch_passthrough, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def fit_stats(ds: WorldModelDataset) -> dict:
    dmean = ds.daily.mean(axis=(0, 1), keepdims=True)
    dstd = ds.daily.std(axis=(0, 1), keepdims=True) + 1e-6
    stats = {"daily_mean": dmean, "daily_std": dstd}
    if ds.intraday is not None:
        stats["intra_mean"] = ds.intraday.mean(axis=(0, 1), keepdims=True)
        stats["intra_std"] = ds.intraday.std(axis=(0, 1), keepdims=True) + 1e-6
    return stats
