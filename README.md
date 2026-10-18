# Market World Model v0.1

A runnable first implementation of a **multi-timeframe market world model** for US equities.

The model learns:

1. `daily context + 5m context + stock identity -> latent state z_t`
2. `z_t -> discrete regime id` using a vector-quantized codebook (default 64 regimes)
3. `z_t -> distribution of future returns` at configurable 5m horizons
4. `z_t -> P(z_{t+1})` with a learned dynamics head trained on true adjacent 5-minute sample pairs
5. an empirical **regime transition graph** for inspection and comparison

The design deliberately keeps ancient/cosmic encoders **out of the baseline**. Ganzhi / I-Ching / Qimen features can later be added as independent encoders and tested by ablation against this learned latent state.

## Data schema

Provide two Parquet files:

`data/daily.parquet`

`data/intraday_5m.parquet`

Both must contain:

```text
symbol,timestamp,open,high,low,close,volume
```

`timestamp` can be timezone-aware or UTC-compatible; the loader normalizes it to UTC. Daily context only uses completed daily bars strictly before the 5-minute sample date, preventing same-day leakage.

## Default experiment split

- Train: 2015-01-01 through 2024-12-31
- Validation: 2025-01-01 through 2025-12-31
- Frozen test: 2026-01-01 through 2026-12-31

Do not tune on the 2026 test set.

## Install

```bash
cd market_world_model
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### From the local Postgres

```bash
python scripts/export_from_db.py                 # 20 liquid tickers by default
python scripts/export_from_db.py --tickers SPY,QQQ,AAPL --daily-start 2024-01-01
```

Reads `DATABASE_URL` from `.env` and writes:

| Source table | Output | Notes |
|---|---|---|
| `ohlcv_daily` | `data/daily.parquet` | bare tickers (`AAPL`) |
| `indicators_daily` | `data/indicators_daily.parquet` | per-day indicators, joined onto daily bars by symbol + New York date |
| `yahoo5m` | `data/intraday_5m.parquet` | `.US` suffix stripped; **09:30-16:00 ET only** |

Two data facts drive the setup:

- `yahoo5m` starts 2026-01-12, so the 2015-2024 train split in `base.yaml` is empty. Use `configs/db_2026.yaml`, which keeps all three splits inside 2026 (train Jan 12-Jun 30, val Jul 1-Aug 15, test Aug 16-Oct 16). This means there is no untouched multi-year test set yet.
- About 99% of extended-hours 5m bars have zero volume, which makes the relative-volume features undefined and drops nearly every 455-bar window. The export keeps regular-session bars only.

### Daily indicators

When `daily_indicators_path` and `daily_indicator_columns` are set in the config, those columns are appended to the daily features (the intraday features are unchanged). The daily context uses only days strictly before the sample date, so indicators computed from a day's close stay causal.

`configs/db_2026.yaml` lists 60 columns. It deliberately leaves out columns that would leak future information or are unusable:

- `realized_mfe_1d`, `realized_mae_1d`, `mfe_1d`, `mfe_1d_pct_rank*`: future outcomes
- `p_up`, `p_down`, `p_peak`, `p_bottom`, `pred_mfe_cls`, `pred_mae_cls`: other models' predictions, possibly fit on future data
- `net`: undocumented
- `m_width`: always null

Run `pip install -e .` or set `PYTHONPATH=src` before the commands below.

## Build samples

```bash
python scripts/build_dataset.py --config configs/base.yaml
```

Default contexts:

- 40 completed daily bars
- 455 5-minute bars (designed to accommodate roughly five prior regular sessions plus current premarket context if your input contains it)
- future 5m horizons: 1, 3, 12, 78 bars

The builder uses causal features only: returns, range/body, close location, volume, SMA distances, relative volume, and true range.

## Train

```bash
python -m mwm.train --config configs/base.yaml
```

On Apple Silicon the trainer automatically chooses MPS when available; CUDA is preferred when present.

Each epoch prints one JSON line. Besides the losses it reports:

- `entropy`: soft code-usage entropy (the term the loss maximizes)
- `usage_entropy`: hard-assignment usage entropy, normalized to 0-1. A value near 0 means the regime codebook has collapsed onto one or two codes.
- `code_restarts`: how many dead codes were re-seeded during the epoch

### Regime codebook

The first version collapsed: every sample mapped to a single regime, and the transition graph had one edge. Two causes:

1. The usage-entropy term was computed from hard `argmin` ids, so it had no gradient and `entropy_weight` had no effect.
2. Nothing stopped codes from going unused. The dynamics loss even rewards collapse, because predicting the next regime is trivial when only one regime exists.

The vector quantizer now:

- L2-normalizes latents and codes (cosine VQ);
- exposes soft assignments `softmax(-dist / vq_temperature)` so the entropy term has a gradient;
- tracks per-code usage with an EMA, and after each optimizer step re-seeds any code whose usage falls below `vq_dead_threshold` from a random latent in the batch.

`db_2026.yaml` raises `entropy_weight` to 0.1 now that the term has an effect.

## Evaluate frozen 2026 test

```bash
python -m mwm.evaluate --checkpoint artifacts/best.pt --split test
```

Reported metrics include per-horizon:

- MAE of future log return mean
- directional accuracy
- 90% predictive interval coverage
- number of latent regimes actually used

## Build latent transition graph

```bash
python -m mwm.transitions --checkpoint artifacts/best.pt --split train --out artifacts/transitions.npz
```

This produces:

- `counts[K,K]`
- `probs[K,K]`

for learned regime transitions `z_t -> z_{t+1}` within each symbol.

## Dynamics training

Each dataset item contains both the current context window and the immediately following 5-minute context window for the same symbol. Therefore random mini-batch shuffling is safe: chronological adjacency is preserved inside each training example. The model learns:

```text
P(z_{t+1} | z_t, stock_t)
```

The next extension is multi-step rollout:

```text
z_t -> z_{t+1} -> z_{t+2} -> ...
```

with scheduled sampling / latent imagination and a separate market-level context state.

## Recommended ablation ladder

Keep the 2026 test frozen and compare on validation first:

```text
A  Market features only (this baseline)
B  A + calendar/time embeddings
C  A + astronomy encoder
D  A + Ganzhi encoder
E  A + I-Ching encoder
F  A + Qimen encoder
G  learned latent alignment between A and D/E/F
```

Promote a new encoder only if it improves validation NLL / calibration / trading utility and then confirms once on frozen 2026.

## Important modeling note

The 64-code latent space is **not assumed to equal the 64 hexagrams**. The model is allowed to discover its own regimes. Only after training should we compare the learned transition graph with an I-Ching graph. That keeps the experiment falsifiable and avoids hard-coding the desired answer.
