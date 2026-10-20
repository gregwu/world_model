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

Standard errors are computed across trading days (per-day means of the per-timestamp metrics): consecutive 5m
timestamps share most of their future-return window, so treating timestamps as independent overstates precision.

--signal relative ranks by the v0.2 model's market-neutral relative head; return ranks by the mean-return head.
The default is relative for pure-5m checkpoints and return otherwise.

Example:
  PYTHONPATH=src python scripts/cross_sectional_probe.py \
      --checkpoint artifacts/alpaca5m_pure/best.pt --split val --max-days 40
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import torch

from mwm.data import make_loader
from mwm.evaluate import load_checkpoint
from mwm.train import autocast_for, run_model, to_device


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
    # timestamps are int64 nanoseconds: view them as datetime64[ns] before flooring to days (casting the raw
    # integers to datetime64[D] reads them as day counts, turning every timestamp into its own "day")
    days = ds.timestamps_ns[rows].astype("datetime64[ns]").astype("datetime64[D]")
    uniq = np.unique(days)
    keep_days = uniq[-max_days:]
    ds.pair_rows = rows[np.isin(days, keep_days)]


def _mean_se_by_day(values, days):
    """Mean over timestamps, and standard error across per-day means."""
    v = np.asarray(values, dtype=np.float64); d = np.asarray(days); ok = np.isfinite(v)
    v, d = v[ok], d[ok]
    if not len(v): return [None, None, 0]
    per_day = pd.Series(v).groupby(d).mean().to_numpy()
    se = float(per_day.std(ddof=1) / np.sqrt(len(per_day))) if len(per_day) > 1 else None
    return [float(v.mean()), se, int(len(per_day))]


def _per_horizon(ts: np.ndarray, pred: np.ndarray, actual: np.ndarray, min_assets: int) -> dict:
    order = np.argsort(ts, kind="stable")
    ts, pred, actual = ts[order], pred[order], actual[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(ts)) + 1, len(ts)]
    rows = {"pearson": [], "spearman": [], "rel_acc": [], "spread": []}; days = {k: [] for k in rows}; widths = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo < min_assets:
            continue
        p, y = pred[lo:hi], actual[lo:hi]
        ok = np.isfinite(p) & np.isfinite(y)
        p, y = p[ok], y[ok]
        if len(y) < min_assets:
            continue
        day = np.int64(ts[lo]).astype("datetime64[ns]").astype("datetime64[D]")
        widths.append(len(y))
        rows["pearson"].append(_corr(p, y)); days["pearson"].append(day)
        rows["spearman"].append(_corr(_rank(p), _rank(y))); days["spearman"].append(day)
        pr, yr = p - p.mean(), y - y.mean()
        moved = yr != 0
        if moved.any():
            rows["rel_acc"].append(float(np.mean(np.sign(pr[moved]) == np.sign(yr[moved])))); days["rel_acc"].append(day)
        # Equal-count tails; require at least 5 names in each side.
        k = max(5, len(y) // 10)
        idx = np.argsort(p)
        rows["spread"].append(float(y[idx[-k:]].mean() - y[idx[:k]].mean())); days["spread"].append(day)

    p_mu, p_se, n_days = _mean_se_by_day(rows["pearson"], days["pearson"])
    s_mu, s_se, _ = _mean_se_by_day(rows["spearman"], days["spearman"])
    a_mu, a_se, _ = _mean_se_by_day(rows["rel_acc"], days["rel_acc"])
    d_mu, d_se, _ = _mean_se_by_day(rows["spread"], days["spread"])
    return {
        "timestamps": len(widths), "days": n_days,
        "mean_assets": float(np.mean(widths)) if widths else 0.0,
        "pearson_ic": p_mu, "pearson_ic_se": p_se,
        "spearman_ic": s_mu, "spearman_ic_se": s_se,
        "relative_direction_accuracy": a_mu, "relative_direction_accuracy_se": a_se,
        "top_minus_bottom_log_return": d_mu, "top_minus_bottom_se": d_se,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--max-days", type=int, default=40,
                    help="evaluate complete recent days; 0 means the whole split")
    ap.add_argument("--min-assets", type=int, default=100)
    ap.add_argument("--signal", choices=["auto", "relative", "return"], default="auto")
    ap.add_argument("--final-test", action="store_true", help="allow the test split when the config sets split.test_frozen")
    args = ap.parse_args()

    ck, cfg, ds, dev, model = load_checkpoint(args.checkpoint, args.split, None, args.final_test)
    signal = args.signal if args.signal != "auto" else ("relative" if getattr(model, "pure_intraday", False) else "return")
    _select_recent_days(ds, args.max_days or None)
    loader = make_loader(ds, cfg["train"]["batch_size"], False)
    mus, ys, tss = [], [], []
    with torch.no_grad():
        for b in loader:
            b = to_device(b, dev)
            with autocast_for(dev, cfg["train"].get("amp")):
                out = run_model(model, b)
            mus.append(out["relative_mu" if signal == "relative" else "return_mu"].float().cpu().numpy())
            ys.append(b["future_returns"].float().cpu().numpy())
            tss.append(b["timestamp_ns"].cpu().numpy())

    mu, y, ts = np.concatenate(mus), np.concatenate(ys), np.concatenate(tss)
    horizons = cfg["data"].get("horizons_5m")
    if horizons is None:  # lazy packs record their horizons in meta.json
        from pathlib import Path
        meta = Path(cfg["data"]["out_path"]) / "meta.json"
        horizons = json.load(open(meta))["horizons"] if meta.exists() else list(range(y.shape[1]))
    result = {
        "split": args.split, "signal": signal,
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
