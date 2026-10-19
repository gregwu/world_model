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

Daily-only model (intended 2015-2024 / 2025 / 2026 split, ~35-80 s per epoch):

```bash
python3 scripts/export_from_db.py --daily-only --top-liquid 200 --daily-start 2014-01-01 --out-dir data/daily200
python3 scripts/build_dataset.py --config configs/daily.yaml
python3 -u -m mwm.train --config configs/daily.yaml > artifacts/daily/train.log 2>&1   # stage 1 + stage-2 dynamics fine-tune
python3 -m mwm.train --config configs/daily.yaml --dynamics-only                      # stage 2 alone on an existing checkpoint
python3 scripts/diagnostics.py --config configs/daily.yaml --checkpoint artifacts/daily/best.pt   # GBM baseline + regime eta^2
```

Large 5m model on Alpaca history (2016-2026, 534 tickers; download ~4 h, pack ~13 GB, training ~10 min/epoch):

```bash
PYTHONPATH=scripts python3 -u scripts/download_alpaca_5m.py --tickers-file data/alpaca5m/universe.txt --end <2+ days ago> --workers 20 --out-dir data/alpaca5m/bars
python3 scripts/export_from_db.py --daily-only --tickers-file data/alpaca5m/universe.txt --daily-start 2014-06-01 --out-dir data/alpaca5m
python3 scripts/build_dataset.py --config configs/alpaca_5m.yaml
python3 -u -m mwm.train --config configs/alpaca_5m.yaml > artifacts/alpaca5m/train.log 2>&1
python3 -m mwm.evaluate --checkpoint artifacts/alpaca5m/best.pt --split test --max-samples 500000
python3 scripts/diagnostics.py --config configs/alpaca_5m.yaml --checkpoint artifacts/alpaca5m/best.pt --max-eval 300000 --max-train 300000
```

The downloader needs `PYTHONPATH=scripts` and the repo root as cwd (it reads `.env`). Alpaca allows 200 req/min shared by
the live and paper keys; `--end` must not reach the last 15 minutes. Judge direction with `directional_accuracy_moves`, not
`directional_accuracy` (unchanged prices count as misses). When waiting on a process with `pgrep -f`, the waiting
shell's own command line can match the pattern; check `ps` for the python process instead.

Always use `python3 -u` when redirecting training output, or the epoch lines only appear at exit. When killing a
training run, kill by PID: `pkill -f "mwm.train ..."` also matches the shell running it. Killed runs can leave
orphaned data-loader workers (ppid 1) behind; check `ps -eo pid,ppid,cmd | grep mwm.train`.

Health checks: train log lines are JSON per epoch - `val.usage_entropy` near 0 means the regime codebook collapsed;
evaluate's `regimes_used` should be well above 1 and transitions' `nonzero_edges` well above 1. The dynamics head
is only working if `next_regime_acc` >= `persistence_acc`.

Memory: each sample is ~30 KB (455x12 intraday + 40x72 daily float32). The box has ~120 GB shared with other services;
keep the pack under ~300k samples or raise it carefully.
