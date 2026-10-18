from __future__ import annotations
import argparse, yaml
from mwm.data import build_samples, save_pack

ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/base.yaml"); args=ap.parse_args()
cfg=yaml.safe_load(open(args.config)); d=cfg["data"]
pack=build_samples(d["daily_path"],d["intraday_path"],d["daily_lookback"],d["intraday_lookback"],tuple(d["horizons_5m"]),d["min_price"],
                   d.get("daily_indicators_path"),d.get("daily_indicator_columns"))
save_pack(pack,d["out_path"])
print(f"saved {len(pack.symbol_id):,} samples to {d['out_path']} across {len(pack.symbols)} symbols "
      f"(daily features={pack.daily.shape[-1]}, intraday features={pack.intraday.shape[-1]})")
