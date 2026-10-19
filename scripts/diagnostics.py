"""Is there signal to find, and do the learned regimes capture it?

A. Gradient-boosting baseline: HistGradientBoostingClassifier on summary features of the same context
   windows, predicting the sign of each horizon's return. If it can't beat the majority class either,
   direction is not learnable from these inputs and the world model's ~50% is expected.
B. Regime analysis (with --checkpoint): eta^2 = share of variance explained by regime membership, for
   |future return| (volatility) and signed future return (direction). Compared against 64 quantile bins
   of trailing realized volatility, a deliberately simple benchmark the regimes should match or beat.

Usage: PYTHONPATH=src python3 scripts/diagnostics.py --config configs/daily.yaml [--checkpoint artifacts/daily/best.pt]
"""
from __future__ import annotations
import argparse, json
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from mwm.data import open_dataset
from mwm.features import feature_columns

LOG_RETURN = feature_columns().index("log_return")


def _chunked(fn, rows, chunk=20_000):
    return np.concatenate([fn(rows[k:k + chunk]) for k in range(0, len(rows), chunk)])


def summary_features(ds, rows: np.ndarray) -> np.ndarray:
    def one(r):
        d = ds.daily_windows(r)
        parts = [d[:, -1], d[:, -5:].mean(1), d.mean(1), d[:, :, LOG_RETURN].std(1, keepdims=True)]
        if ds.intra_dim:
            x = ds.intraday_windows(r)
            parts += [x[:, -1], x[:, -12:].mean(1), x[:, -78:].mean(1), x[:, -78:, LOG_RETURN].std(1, keepdims=True)]
        return np.concatenate(parts, axis=1)
    return _chunked(one, rows)


def trailing_vol(ds, rows: np.ndarray) -> np.ndarray:
    if ds.intra_dim:
        return _chunked(lambda r: ds.intraday_windows(r)[:, -78:, LOG_RETURN].std(1), rows)
    return _chunked(lambda r: ds.daily_windows(r)[:, -20:, LOG_RETURN].std(1), rows)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Between-group variance / total variance."""
    total = values.var()
    if total == 0: return 0.0
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=values) / counts
    return float((counts * (means - values.mean()) ** 2).sum() / len(values) / total)


def quantile_bins(x: np.ndarray, edges_from: np.ndarray, k: int = 64) -> np.ndarray:
    edges = np.quantile(edges_from, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, x)


def split_ranges(cfg):
    s = cfg["split"]
    return {"train": (s["train_start"], s["train_end"]), "val": (s["val_start"], s["val_end"]), "test": (s["test_start"], s["test_end"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--max-eval", type=int, default=None, help="fixed random subset of pairs per evaluated split")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); p = cfg["data"]["out_path"]; ranges = split_ranges(cfg)
    rng = np.random.default_rng(cfg["seed"])
    raw = {k: open_dataset(p, *ranges[k], max_samples=args.max_eval if k != "train" else None, seed=cfg["seed"])
           for k in ["train", *args.splits.split(",")]}
    report = {"config": args.config, "baseline": {}, "regimes": {}}

    tr = raw["train"]; tr_rows = tr.pair_rows
    if len(tr_rows) > args.max_train: tr_rows = np.sort(rng.choice(tr_rows, args.max_train, replace=False))
    if not args.skip_baseline:
        Xtr = summary_features(tr, tr_rows); Ytr = tr.returns[tr_rows]
    for split in ([] if args.skip_baseline else args.splits.split(",")):
        ds = raw[split]; rows = ds.pair_rows; X = summary_features(ds, rows); Y = ds.returns[rows]
        out = []
        for h in range(Y.shape[1]):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                 early_stopping=True, validation_fraction=0.1, random_state=cfg["seed"])
            clf.fit(Xtr, Ytr[:, h] > 0)
            up = Y[:, h] > 0; majority_up = (Ytr[:, h] > 0).mean() >= 0.5
            out.append({"horizon_index": h, "gbm_directional_accuracy": float((clf.predict(X) == up).mean()),
                        "majority_class_accuracy": float((up == majority_up).mean()), "share_up": float(up.mean())})
        report["baseline"][split] = out
        print(json.dumps({"baseline": split, "results": out}), flush=True)

    if args.checkpoint:
        from mwm.evaluate import load_checkpoint
        from mwm.transitions import infer_regimes
        train_vol = trailing_vol(tr, tr_rows)
        for split in args.splits.split(","):
            ck, mcfg, ds, dev, model = load_checkpoint(args.checkpoint, split, args.max_eval)
            regimes = infer_regimes(model, ds, dev, mcfg["train"]["batch_size"], mcfg["train"].get("amp"))  # aligned with ds.pair_rows
            rows = ds.pair_rows; Y = ds.returns[rows]
            vol_bins = quantile_bins(trailing_vol(raw[split], rows), train_vol)
            res = []
            for h in range(Y.shape[1]):
                a = np.abs(Y[:, h])
                res.append({"horizon_index": h,
                            "eta2_abs_return_by_regime": eta_squared(a, regimes),
                            "eta2_abs_return_by_vol_bins": eta_squared(a, vol_bins),
                            "eta2_abs_return_by_regime_within_vol_bins": eta_squared(a - _group_mean(a, vol_bins), regimes),
                            "eta2_signed_return_by_regime": eta_squared(Y[:, h], regimes)})
            report["regimes"][split] = res
            print(json.dumps({"regimes": split, "n": int(len(rows)), "results": res}), flush=True)
    return report


def _group_mean(values, groups):
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=values) / counts)[inv]


if __name__ == "__main__":
    main()
