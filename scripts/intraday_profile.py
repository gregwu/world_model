"""Intraday volatility and volume profile by 5-minute slot, from a lazy pack's per-bar features.

For each 5-minute slot of the regular session: median |log return|, median bar range ((high-low)/previous close),
median volume, and range/volume relative to midday (12:00-14:00 ET). The 09:30 bar's return is measured from the
previous session's close, so it includes the overnight gap; its range does not.

Usage: PYTHONPATH=src python3 scripts/intraday_profile.py --pack data/alpaca5m/pack [--sample 12000000] [--csv out.csv]
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/alpaca5m/pack")
    ap.add_argument("--sample", type=int, default=12_000_000, help="random bars to use (0 = all)")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    meta = json.load(open(f"{args.pack}/meta.json")); cols = meta["intraday_features"]
    X = np.load(f"{args.pack}/intra_feats.npy", mmap_mode="r"); ts = np.load(f"{args.pack}/intra_ts.npy", mmap_mode="r")
    idx = np.arange(len(ts))
    if args.sample and args.sample < len(ts):
        idx = np.sort(np.random.default_rng(0).choice(len(ts), args.sample, replace=False))
    feats = {c: np.asarray(X[idx, cols.index(c)], dtype=np.float64) for c in ("log_return", "range_pct", "log_volume")}
    et = pd.to_datetime(np.asarray(ts[idx]), utc=True).tz_convert("America/New_York")
    df = pd.DataFrame({"slot": (et.hour * 60 + et.minute).to_numpy(), "abs_return": np.abs(feats["log_return"]),
                       "range": feats["range_pct"], "volume": np.expm1(feats["log_volume"])}).dropna()
    g = df.groupby("slot").agg(median_abs_return=("abs_return", "median"), median_range=("range", "median"),
                               median_volume=("volume", "median"), bars=("range", "size"))
    midday = (g.index >= 12 * 60) & (g.index < 14 * 60)
    g["range_vs_midday"] = g.median_range / g.loc[midday, "median_range"].mean()
    g["volume_vs_midday"] = g.median_volume / g.loc[midday, "median_volume"].mean()
    g.index = [f"{s // 60:02d}:{s % 60:02d}" for s in g.index]
    if args.csv: g.to_csv(args.csv)
    print(f"{len(df):,} bars")
    print(g[["median_abs_return", "median_range", "range_vs_midday", "volume_vs_midday"]].round(5).to_string())


if __name__ == "__main__":
    main()
