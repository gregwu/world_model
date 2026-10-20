from __future__ import annotations
import argparse, yaml
from mwm.data import build_samples, build_daily_samples, save_pack
from mwm.lazy import build_lazy_pack

ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/base.yaml"); args=ap.parse_args()
cfg=yaml.safe_load(open(args.config)); d=cfg["data"]
if d.get("format")=="lazy":  # out_path is a directory; see mwm/lazy.py
    meta=build_lazy_pack(d["out_path"],d["daily_path"],d["intraday_path"],d["daily_lookback"],d["intraday_lookback"],
                         tuple(d["horizons_5m"]),d["min_price"],d.get("daily_indicators_path"),d.get("daily_indicator_columns"),
                         d.get("intraday_indicators",False))
    print(f"saved lazy pack to {d['out_path']}: {meta['shapes']['anchor'][0]:,} samples across {len(meta['symbols'])} symbols, "
          f"{meta['shapes']['intra_feats'][0]:,} intraday rows; skipped {len(meta['skipped_symbols'])}: {meta['skipped_symbols'][:20]}")
    raise SystemExit
if d.get("mode","intraday")=="daily":
    pack=build_daily_samples(d["daily_path"],d["daily_lookback"],tuple(d["horizons_days"]),d["min_price"],
                             d.get("daily_indicators_path"),d.get("daily_indicator_columns"))
else:
    pack=build_samples(d["daily_path"],d["intraday_path"],d["daily_lookback"],d["intraday_lookback"],tuple(d["horizons_5m"]),d["min_price"],
                       d.get("daily_indicators_path"),d.get("daily_indicator_columns"))
save_pack(pack,d["out_path"])
intra=pack.intraday.shape[-1] if pack.intraday is not None else 0
print(f"saved {len(pack.symbol_id):,} samples to {d['out_path']} across {len(pack.symbols)} symbols "
      f"(daily features={pack.daily.shape[-1]}, intraday features={intra})")
