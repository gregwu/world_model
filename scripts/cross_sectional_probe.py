"""Cross-sectional probe for the market world model.

The existing evaluation asks whether each stock's absolute future return sign is
predictable.  That can hide useful relative-value information because the common
market move dominates every stock.  This probe evaluates the same frozen model
cross-sectionally at each timestamp:

  * Pearson and Spearman information coefficient (predicted vs realised return)
  * relative-direction accuracy after removing each timestamp's cross-sectional mean
  * top-minus-bottom prediction-decile realised spread
  * coverage: number of symbols present per timestamp

No model is refit and no future information is used to form predictions.  Cross-sectional
demeaning/ranking is evaluation-only and uses realised returns only in the metric.

Example:
  PYTHONPATH=src python scripts/cross_sectional_probe.py \
      --checkpoint artifacts/alpaca5m/best.pt --split test --max-days 40
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import torch

from mwm.data import make_loader
from mwm.evaluate import load_checkpoint
from mwm.train import autocast_for, to_device


def _rank(x: np.ndarray) -> np.ndarray:
    # Average ranks for ties; pandas is reliable and this is only per timestamp.
    return pd.Series(x).rank(method="average").to_numpy(np.float64)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _select_recent_days(ds, max_days: int | None) -> None:
    """Restrict pair_rows to complete recent UTC calendar days, preserving cross-sections."""
    if not max_days:
        return
    rows = ds.pair_rows
    days = ds.timestamps_ns[rows].astype("datetime64[D]")
    uniq = np.unique(days)
    keep_days = uniq[-max_days:]
    ds.pair_rows = rows[np.isin(days, keep_days)]


def _per_horizon(ts: np.ndarray, pred: np.ndarray, actual: np.ndarray, min_assets: int) -> dict:
    order = np.argsort(ts, kind="stable")
    ts, pred, actual = ts[order], pred[order], actual[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(ts)) + 1, len(ts)]
    pearson, spearman, rel_acc, spreads, widths = [], [], [], [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo < min_assets:
            continue
        p, y = pred[lo:hi], actual[lo:hi]
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        if len(y) < min_assets:
            continue
        widths.append(len(y))
        pearson.append(_corr(p, y))
        spearman.append(_corr(_rank(p), _rank(y)))
        pr, yr = p - p.mean(), y - y.mean()
        moved = yr != 0
        if moved.any():
            rel_acc.append(float(np.mean(np.sign(pr[moved]) == np.sign(yr[moved]))))
        # Equal-count tails; require at least 5 names in each side.
        k = max(5, len(y) // 10)
        idx = np.argsort(p)
        spreads.append(float(y[idx[-k:]].mean() - y[idx[:k]].mean()))

    def mean_se(x):
        a = np.asarray(x, dtype=np.float64); a = a[np.isfinite(a)]
        if not len(a): return [None, None]
        return [float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None]

    p_mu, p_se = mean_se(pearson); s_mu, s_se = mean_se(spearman)
    a_mu, a_se = mean_se(rel_acc); d_mu, d_se = mean_se(spreads)
    return {
        "timestamps": len(widths),
        "mean_assets": float(np.mean(widths)) if widths else 0.0,
        "pearson_ic": p_mu, "pearson_ic_se": p_se,
        "spearman_ic": s_mu, "spearman_ic_se": s_se,
        "relative_direction_accuracy": a_mu, "relative_direction_accuracy_se": a_se,
        "top_minus_bottom_log_return": d_mu, "top_minus_bottom_se": d_se,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--max-days", type=int, default=40,
                    help="evaluate complete recent days; 0 means the whole split")
    ap.add_argument("--min-assets", type=int, default=100)
    args = ap.parse_args()

    ck, cfg, ds, dev, model = load_checkpoint(args.checkpoint, args.split, None)
    _select_recent_days(ds, args.max_days or None)
    loader = make_loader(ds, cfg["train"]["batch_size"], False)
    mus, ys, tss = [], [], []
    with torch.no_grad():
        for b in loader:
            b = to_device(b, dev)
            with autocast_for(dev, cfg["train"].get("amp")):
                out = model(b["daily"], b.get("intraday"), b["symbol_id"])
            mus.append(out["return_mu"].float().cpu().numpy())
            ys.append(b["future_returns"].float().cpu().numpy())
            tss.append(b["timestamp_ns"].cpu().numpy())

    mu, y, ts = np.concatenate(mus), np.concatenate(ys), np.concatenate(tss)
    horizons = cfg["data"].get("horizons_5m", list(range(y.shape[1])))
    result = {
        "split": args.split,
        "samples": int(len(y)),
        "max_days": args.max_days,
        "min_assets": args.min_assets,
        "interpretation": "Positive IC/spread means the frozen model ranks future cross-sectional returns better than chance.",
        "horizons": [],
    }
    for h in range(y.shape[1]):
        r = _per_horizon(ts, mu[:, h], y[:, h], args.min_assets)
        r["horizon"] = int(horizons[h]) if h < len(horizons) else h
        result["horizons"].append(r)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
