"""Lazy sample packs for large intraday universes.

The in-memory pack (data.save_pack) materializes every 455x12 intraday window: ~33 KB per sample, so hundreds of
tickers x ten years of 5m bars would need hundreds of GB. A lazy pack is a directory that stores per-bar features
once and, per sample, only the row its windows end on:

  intra_feats.npy  [R, F]   per-bar intraday features, all symbols concatenated (memory-mapped at load)
  daily_feats.npy  [D, Fd]  per-day daily features, all symbols concatenated (memory-mapped at load)
  intra_ts.npy / daily_ts.npy        row timestamps (UTC ns), used to fit normalization on the train period
  anchor.npy       [N]      global intra row of each sample's last bar; window = rows anchor-L+1 .. anchor
  daily_end.npy    [N]      global daily row one past the sample's last completed day; window = daily_end-Ld .. -1
  symbol_id / future_returns / timestamps_ns / bar_index .npy        per-sample arrays, as in the in-memory pack
  meta.json        symbols, lookbacks, horizons, feature names

Windows are gathered per batch with one fancy-index into the memory-mapped arrays.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .data import NORM_CLIP, _load_bars, _load_daily, _utc_ns, pair_next
from .features import add_bar_features, add_intraday_indicators, feature_columns, intraday_indicator_columns

NY = "America/New_York"


def is_lazy_pack(path) -> bool:
    return (Path(path) / "meta.json").exists()


def _iter_intraday(path):
    """Yield (symbol, bars sorted by time). A directory is read one Parquet file (= one symbol) at a time."""
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.glob("*.parquet")):
            g = pd.read_parquet(f)
            if g.empty: continue
            g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
            yield g["symbol"].iloc[0], g.sort_values("timestamp").reset_index(drop=True)
    else:
        for s, g in _load_bars(path).groupby("symbol"):
            yield s, g.reset_index(drop=True)


def _windows_finite(finite_rows: np.ndarray, ends: np.ndarray, length: int) -> np.ndarray:
    """True where rows end-length+1 .. end are all finite (callers guarantee end-length+1 >= 0)."""
    cum = np.concatenate([[0], np.cumsum(~finite_rows)])
    return cum[ends + 1] - cum[ends + 1 - length] == 0


def build_lazy_pack(
    out_dir: str,
    daily_path: str,
    intraday_path: str,
    daily_lookback: int = 40,
    intraday_lookback: int = 455,
    horizons_5m: tuple[int, ...] = (1, 3, 12, 78),
    min_price: float = 1.0,
    daily_indicators_path: str | None = None,
    daily_indicator_columns: list[str] | None = None,
    intraday_indicators: bool = False,
) -> dict:
    """Same sample definition as data.build_samples (daily context strictly before the sample date), stored lazily.
    intraday_indicators appends 5m/1h %B, RSI, MACD histogram and 5m session-VWAP distance (features.py)."""
    daily, daily_feats = _load_daily(daily_path, daily_indicators_path, daily_indicator_columns)
    dgroups = {s: g.reset_index(drop=True) for s, g in daily.groupby("symbol")}
    feats = feature_columns() + (intraday_indicator_columns() if intraday_indicators else [])
    H = np.asarray(horizons_5m); max_h = int(H.max())
    L, Ld = intraday_lookback, daily_lookback
    symbols, skipped = [], []
    chunks = {k: [] for k in ("intra_feats", "intra_ts", "daily_feats", "daily_ts", "anchor", "daily_end",
                              "symbol_id", "future_returns", "timestamps_ns", "bar_index")}
    ioff = doff = 0
    for sym, ig in _iter_intraday(intraday_path):
        dg = dgroups.get(sym)
        if dg is None or len(ig) < L + max_h:
            skipped.append(sym); continue
        ig = add_bar_features(ig)
        if intraday_indicators:
            ig = add_intraday_indicators(ig)
        ivals = ig[feats].to_numpy(np.float32); closes = ig["close"].to_numpy(np.float64); stamps = _utc_ns(ig["timestamp"])
        dvals = dg[daily_feats].to_numpy(np.float32)
        idates = ig["timestamp"].dt.tz_convert(NY).dt.date.to_numpy()
        ddates = dg["timestamp"].dt.tz_convert(NY).dt.date.to_numpy()

        i = np.arange(L - 1, len(ig) - max_h)
        d_end = np.searchsorted(ddates, idates[i], side="left")  # days strictly before the sample date
        ok = (closes[i] >= min_price) & (d_end - Ld >= 0)
        ok &= _windows_finite(np.isfinite(ivals).all(1), i, L)
        dfin = np.isfinite(dvals).all(1)
        ok[ok] &= _windows_finite(dfin, d_end[ok] - 1, Ld)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(closes[i[:, None] + H[None, :]] / closes[i][:, None]).astype(np.float32)
        ok &= np.isfinite(y).all(1)
        if not ok.any():
            skipped.append(sym); continue

        sid = len(symbols); symbols.append(sym); i = i[ok]
        chunks["intra_feats"].append(ivals); chunks["intra_ts"].append(stamps)
        chunks["daily_feats"].append(dvals); chunks["daily_ts"].append(_utc_ns(dg["timestamp"]))
        chunks["anchor"].append(i + ioff); chunks["daily_end"].append(d_end[ok] + doff)
        chunks["symbol_id"].append(np.full(len(i), sid, dtype=np.int64)); chunks["future_returns"].append(y[ok])
        chunks["timestamps_ns"].append(stamps[i]); chunks["bar_index"].append(i)
        ioff += len(ivals); doff += len(dvals)
        print(f"  {sym}: {len(i):,} samples", flush=True)

    if not symbols:
        raise RuntimeError("No valid samples were built. Check history depth and feature warmup.")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dtypes = {"intra_feats": np.float32, "daily_feats": np.float32, "future_returns": np.float32}
    sizes = {}
    for k, parts in chunks.items():
        arr = np.concatenate(parts).astype(dtypes.get(k, np.int64), copy=False)
        np.save(out / f"{k}.npy", arr); sizes[k] = list(arr.shape); chunks[k] = None
    meta = {"format": "lazy", "symbols": symbols, "skipped_symbols": skipped, "intraday_lookback": L,
            "daily_lookback": Ld, "horizons": [int(h) for h in H], "intraday_features": feats,
            "daily_features": daily_feats, "shapes": sizes}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    return meta


class LazyWorldModelDataset(Dataset):
    """Drop-in for data.WorldModelDataset over a lazy pack. max_samples keeps a fixed random subset of pairs."""

    def __init__(self, pack_dir: str, start: str, end: str, stats: dict | None = None,
                 max_samples: int | None = None, seed: int = 0):
        d = Path(pack_dir); meta = json.load(open(d / "meta.json"))
        self.intra_feats = np.load(d / "intra_feats.npy", mmap_mode="r")
        self.daily_feats = np.load(d / "daily_feats.npy", mmap_mode="r")
        self._dir = d
        ts = np.load(d / "timestamps_ns.npy")
        lo = pd.Timestamp(str(start), tz="UTC").value; hi = pd.Timestamp(str(end) + " 23:59:59", tz="UTC").value
        idx = np.flatnonzero((ts >= lo) & (ts <= hi))
        self.timestamps_ns = ts[idx]; self._range = (lo, hi)
        self.anchor = np.load(d / "anchor.npy", mmap_mode="r")[idx]
        self.daily_end = np.load(d / "daily_end.npy", mmap_mode="r")[idx]
        self.symbol_id = np.load(d / "symbol_id.npy", mmap_mode="r")[idx]
        self.returns = np.load(d / "future_returns.npy", mmap_mode="r")[idx]
        bar_index = np.load(d / "bar_index.npy", mmap_mode="r")[idx]
        self.symbols = meta["symbols"]
        self._oi = np.arange(-meta["intraday_lookback"] + 1, 1); self._od = np.arange(-meta["daily_lookback"], 0)
        self.daily_dim = self.daily_feats.shape[1]; self.intra_dim = self.intra_feats.shape[1]
        self.intraday = None  # windows are never materialized; see intraday_windows()
        self.stats = stats
        self.next_idx, self.pair_rows = pair_next(self.symbol_id, bar_index, self.timestamps_ns)
        if max_samples and len(self.pair_rows) > max_samples:
            rng = np.random.default_rng(seed)
            self.pair_rows = np.sort(rng.choice(self.pair_rows, max_samples, replace=False))

    def _norm(self, x, key):
        if self.stats is None: return np.asarray(x, dtype=np.float32)
        return np.clip((x - self.stats[f"{key}_mean"]) / self.stats[f"{key}_std"], -NORM_CLIP, NORM_CLIP).astype(np.float32)

    def daily_windows(self, samples): return self._norm(self.daily_feats[self.daily_end[samples][:, None] + self._od], "daily")
    def intraday_windows(self, samples): return self._norm(self.intra_feats[self.anchor[samples][:, None] + self._oi], "intra")

    def __len__(self): return len(self.pair_rows)

    def __getitems__(self, rows):
        i = self.pair_rows[np.asarray(rows)]; j = self.next_idx[i]
        return {
            "daily": torch.from_numpy(self.daily_windows(i)),
            "intraday": torch.from_numpy(self.intraday_windows(i)),
            "symbol_id": torch.from_numpy(np.ascontiguousarray(self.symbol_id[i])),
            "future_returns": torch.from_numpy(np.ascontiguousarray(self.returns[i])),
            "timestamp_ns": torch.from_numpy(self.timestamps_ns[i]),
            "next_daily": torch.from_numpy(self.daily_windows(j)),
            "next_intraday": torch.from_numpy(self.intraday_windows(j)),
        }

    def __getitem__(self, row):
        return {k: v[0] for k, v in self.__getitems__([row]).items()}

    def fit_stats(self, max_rows: int = 5_000_000, seed: int = 0) -> dict:
        """Per-feature mean/std over rows inside this dataset's date range (a random subset of at most max_rows)."""
        rng = np.random.default_rng(seed); lo, hi = self._range; stats = {}
        for key, name in (("daily", "daily"), ("intra", "intra")):
            ts = np.load(self._dir / f"{name}_ts.npy", mmap_mode="r")
            rows = np.flatnonzero((ts >= lo) & (ts <= hi))
            if len(rows) > max_rows: rows = np.sort(rng.choice(rows, max_rows, replace=False))
            x = np.asarray((self.daily_feats if key == "daily" else self.intra_feats)[rows], dtype=np.float64)
            x = x[np.isfinite(x).all(1)]
            stats[f"{key}_mean"] = x.mean(0)[None, None].astype(np.float32)
            stats[f"{key}_std"] = (x.std(0)[None, None] + 1e-6).astype(np.float32)
        return stats
