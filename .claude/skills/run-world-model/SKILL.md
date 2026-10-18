---
name: run-world-model
description: Run the market world model end to end on this box - export bars from the local Postgres (ohlcv_daily, indicators_daily, yahoo5m), build samples, train on the GPU, evaluate, build the regime transition graph.
---

# Run the market world model

Verified on the GB10 box (CUDA, Python 3.12 system interpreter, torch/pandas/pyarrow already installed).
No venv or `pip install -e .` needed: use `PYTHONPATH=src`. pytest is not installed.

## 1. Export data from Postgres

```bash
python3 scripts/export_from_db.py            # 20 liquid tickers by default; --tickers SPY,AAPL,... to change
```

Reads `DATABASE_URL` from `.env` (don't `source .env`: it has lines that aren't valid shell). Writes
`data/daily.parquet`, `data/indicators_daily.parquet`, `data/intraday_5m.parquet`.

Data facts that drive the setup:
- `yahoo5m` tickers carry a `.US` suffix (`AAPL.US`); `ohlcv_daily` / `indicators_daily` use bare tickers. The export strips `.US`.
- `yahoo5m` only covers **2026-01-12 onward**, so `configs/base.yaml`'s 2015-2024 train split is empty. Use `configs/db_2026.yaml` (train Jan-Jun, val Jul-mid Aug, test mid Aug-Oct 2026).
- ~99% of extended-hours 5m bars have volume 0, which makes `relvol_*` NaN and drops nearly every 455-bar window. The export keeps 09:30-16:00 ET only.

## 2. Build, train, evaluate

```bash
export PYTHONPATH=src
python3 scripts/build_dataset.py --config configs/db_2026.yaml          # ~40s, ~240k samples for 20 tickers
python3 -m mwm.train --config configs/db_2026.yaml > artifacts/train.log 2>&1   # run in background; minutes per epoch
python3 -m mwm.evaluate --checkpoint artifacts/best.pt --split val
python3 -m mwm.evaluate --checkpoint artifacts/best.pt --split test
python3 -m mwm.transitions --checkpoint artifacts/best.pt --split train --out artifacts/transitions.npz
```

Health checks: train log lines are JSON per epoch - `val.usage_entropy` near 0 means the regime codebook collapsed;
evaluate's `regimes_used` should be well above 1 and transitions' `nonzero_edges` well above 1.

Memory: each sample is ~30 KB (455x12 intraday + 40x72 daily float32). The box has ~120 GB shared with other services;
keep the pack under ~300k samples or raise it carefully.
