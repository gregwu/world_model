# Market World Model v0.1

A runnable first implementation of a **multi-timeframe market world model** for US equities.

The model learns:

1. `daily context [+ 5m context] + stock identity -> latent state z_t` (the 5m encoder is optional; see [Daily-only mode](#daily-only-mode))
2. `z_t -> discrete regime id` using a vector-quantized codebook (default 64 regimes)
3. `z_t -> distribution of future returns` at configurable horizons (5m bars, or trading days in daily mode)
4. `z_t -> P(regime_{t+1})` with a gated dynamics head trained on true adjacent sample pairs (see [Dynamics head](#dynamics-head))
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
python scripts/export_from_db.py --daily-only --top-liquid 200 --daily-start 2014-01-01 --out-dir data/daily200
```

Reads `DATABASE_URL` from `.env` and writes:

| Source table | Output | Notes |
|---|---|---|
| `ohlcv_daily` | `daily.parquet` | bare tickers (`AAPL`) |
| `indicators_daily` | `indicators_daily.parquet` | per-day indicators, joined onto daily bars by symbol + New York date |
| `yahoo5m` | `intraday_5m.parquet` | `.US` suffix stripped; **09:30-16:00 ET only**; skipped with `--daily-only` |

`--top-liquid N` picks the N tickers with the highest median dollar volume over `--liquid-start`/`--liquid-end` (default: calendar 2014) instead of `--tickers`.

Data facts that drive the setup:

- `yahoo5m` starts 2026-01-12, so the 2015-2024 train split in `base.yaml` is empty for 5m models. `configs/db_2026.yaml` keeps all three splits inside 2026 (train Jan 12-Jun 30, val Jul 1-Aug 15, test Aug 16-Oct 16). There is no untouched multi-year 5m test set yet.
- About 99% of extended-hours 5m bars have zero volume, which makes the relative-volume features undefined and drops nearly every 455-bar window. The export keeps regular-session bars only.
- `ohlcv_daily` and `indicators_daily` go back decades (about 4,500 tickers from 2014), so daily models can use the intended 2015-2024 / 2025 / 2026 split.
- **Survivorship bias:** all 200 of the most liquid tickers of 2014 still have bars in 2026, so the database appears to keep only currently listed tickers. Picking the universe by 2014 liquidity does not remove the bias.
- **`indicators_daily` has no rows for any ticker from 2023-08-10 to 2023-11-01.** Daily windows that touch the gap are dropped, which removes roughly August-December 2023 from training.

### Daily indicators

When `daily_indicators_path` and `daily_indicator_columns` are set in the config, those columns are appended to the daily features (the intraday features are unchanged). In 5m mode the daily context uses only days strictly before the sample date; in daily mode it ends at the sample day's close. Either way, indicators computed from a day's close stay causal.

`configs/db_2026.yaml` lists 60 columns. It deliberately leaves out columns that would leak future information or are unusable:

- `realized_mfe_1d`, `realized_mae_1d`, `mfe_1d`, `mfe_1d_pct_rank*`: future outcomes
- `p_up`, `p_down`, `p_peak`, `p_bottom`, `pred_mfe_cls`, `pred_mae_cls`: other models' predictions, possibly fit on future data
- `net`: undocumented
- `m_width`: always null

`configs/daily.yaml` uses 48 of those 60. It also drops the 12 columns with more than 2% nulls in 2024 or 2025: `macd_momentum_zscore`, `sma_144_slope_pct`, `obv_momentum_zscore`, `d_width`, `w_pctb`, `w_width`, `m_pctb`, `w_pctb_chg1`, `w_pctb_chg5`, `m_pctb_chg1`, `m_pctb_chg5`, `since_dn`.

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

Flat bars (high == low) get `close_location = 0.5`. They were previously NaN, and any NaN drops the whole window. About 1% of 5m bars are flat, so that silently removed roughly a third of all 455-bar windows: the Alpaca pack went from 67M to 100M samples with the fix. Packs built before 2026-10-19 (daily, `db_2026`) were built with the old rule.

### Daily-only mode

With `data.mode: daily` (as in `configs/daily.yaml`), samples end on each daily close and the model has no intraday encoder:

- context: the last `daily_lookback` daily bars, including day t
- targets: log returns from close_t to close_{t+h}, with `horizons_days` (default 1, 5, 20 trading days)
- next-step pairs: day t and day t+1 of the same symbol

```bash
python scripts/export_from_db.py --daily-only --top-liquid 200 --daily-start 2014-01-01 --out-dir data/daily200
python scripts/build_dataset.py --config configs/daily.yaml    # ~592k samples, ~25 s
python -m mwm.train --config configs/daily.yaml                # ~35-80 s per epoch on the GB10
```

### Large 5m universe from Alpaca: `configs/alpaca_5m.yaml`

`yahoo5m` covers only 2026, so the large 5m model uses Alpaca's historical SIP bars instead: 2016-01-04 to 2026-10-16, split- and dividend-adjusted, regular session only.

```bash
# universe: current S&P 500 (from Wikipedia) + the 100 most liquid tickers of 2014 = 534 symbols in data/alpaca5m/universe.txt
python scripts/download_alpaca_5m.py --tickers-file data/alpaca5m/universe.txt --end 2026-10-17 --workers 20 --out-dir data/alpaca5m/bars
python scripts/export_from_db.py --daily-only --tickers-file data/alpaca5m/universe.txt --daily-start 2014-06-01 --out-dir data/alpaca5m
python scripts/build_dataset.py --config configs/alpaca_5m.yaml      # lazy pack, ~13 GB, a few minutes
python -m mwm.train --config configs/alpaca_5m.yaml                  # ~10 min per epoch on the GB10
```

Download notes:

- **Rate limit:** Alpaca's free plan allows 200 requests per minute. The live and paper keys share one quota (a second key at 180/min each produced 429s), so they don't add throughput.
- **Page size:** a page holds about 2,600 five-minute bars whatever `limit` says (the limit appears to count 1-minute source bars). A ticker therefore takes about 130 requests, and the full 534-ticker download took about 4 hours at 180 requests per minute.
- **Recent data:** the free plan rejects SIP queries that reach into the last 15 minutes, so `--end` must be in the past; it defaults to two days ago.
- **Resuming:** files are written atomically, one Parquet per ticker. Reruns skip finished tickers, so an interrupted run resumes.
- **Webull is not an alternative:** its `/market-data/bars` endpoint returns at most the newest 1,200 bars, with no start or end time to page back. That is about 15 trading days of regular-hours 5m bars (see rustish `trading-platform/crates/market/src/webull.rs`).

**Lazy packs** ([src/mwm/lazy.py](src/mwm/lazy.py)). Materialized windows cost about 33 KB per sample, so 100M samples would need hundreds of GB. With `data.format: lazy`, `out_path` is a directory that stores per-bar features once (memory-mapped `.npy` files) plus, for each sample, only the row its windows end on. Windows are gathered per batch. `train.samples_per_epoch` draws a fresh random subset of training pairs each epoch, and `train.val_max_samples` fixes a random validation subset. `evaluate`, `transitions` and `scripts/diagnostics.py` accept `--max-samples` / `--max-eval`.

**Intraday indicators** (`data.intraday_indicators: true`, [features.py](src/mwm/features.py)) add 7 intraday features to the 12 bar features:

| Timeframe | Features |
|---|---|
| 5m | `m5_pctb` (Bollinger %B, 20 bars, 2σ), `m5_rsi` (Wilder RSI 14, scaled 0-1), `m5_macd_hist` (MACD 12/26/9 histogram as a fraction of price), `m5_vwap_dist` (distance from the session-anchored VWAP) |
| 1h | `h1_pctb`, `h1_rsi`, `h1_macd_hist` |

1h bars are built from the 5m bars and aligned to the 09:30 open (seven per session, the last 30 minutes long). Each 5m bar sees only the last *completed* 1h bar. A truncation test (features computed on a prefix of the series equal the same rows computed on the full series) confirms every feature is causal.

**bf16:** `train.amp: bf16` uses bf16 autocast on CUDA. The vector quantizer, the gated dynamics head and the return head stay in fp32. Predicted mean returns are around 1e-4, and in bf16 up to 17% of them rounded to exactly 0, which made sign-based accuracy look like a strong anti-signal (41%) that wasn't there.

Each sample pack stores `bar_index` (the row of the sample's last bar within its symbol), and the dataset pairs consecutive bars through it. That replaces the old "exactly 5 minutes later" rule and works for any bar size. Packs built before this change fall back to the 5-minute rule.

## Train

```bash
python -m mwm.train --config configs/base.yaml
```

On Apple Silicon the trainer automatically chooses MPS when available; CUDA is preferred when present.

Training has two stages:

1. Encoder, codebook, return head and dynamics head train jointly; the best epoch by validation return NLL is checkpointed.
2. If `dynamics_finetune_epochs` > 0, that checkpoint is reloaded with everything frozen except the dynamics head, which is fitted alone. The best epoch by validation dynamics loss overwrites the checkpoint. See [Dynamics head](#dynamics-head) for why.

`python -m mwm.train --config <cfg> --dynamics-only` runs stage 2 alone on an existing checkpoint.

Each epoch prints one JSON line (stage-2 epochs are labelled `dyn1`, `dyn2`, ...). Besides the losses it reports:

- `entropy`: soft code-usage entropy (the term the loss maximizes)
- `usage_entropy`: hard-assignment usage entropy, normalized to 0-1. A value near 0 means the regime codebook has collapsed onto one or two codes. Validation batches are not shuffled, so per-batch validation values run lower than training ones.
- `code_restarts`: how many dead codes were re-seeded during the epoch
- `next_regime_acc`: how often the dynamics head's top prediction is the actual next regime
- `persistence_acc`: the same for the baseline "the regime stays the same". The dynamics head is only useful if it beats this.
- `seconds`: wall time for the epoch

Data loading fetches each batch with one NumPy fancy-index per array (`__getitems__`) instead of one small copy per sample, and metrics stay on the GPU until the end of the epoch. A daily epoch takes about 36 s with the GPU to itself. Run with `python3 -u` when redirecting output to a file, or the JSON lines only appear when the process exits.

### Regime codebook

The first version collapsed: every sample mapped to a single regime, and the transition graph had one edge. Two causes:

1. The usage-entropy term was computed from hard `argmin` ids, so it had no gradient and `entropy_weight` had no effect.
2. Nothing stopped codes from going unused. The dynamics loss even rewards collapse, because predicting the next regime is trivial when only one regime exists.

The vector quantizer now:

- L2-normalizes latents and codes (cosine VQ);
- exposes soft assignments `softmax(-dist / vq_temperature)` so the entropy term has a gradient;
- tracks per-code usage with an EMA, and after each optimizer step re-seeds any code whose usage falls below `vq_dead_threshold` from a random latent in the batch.

Restarts run for the first `vq_restart_epochs` epochs. The shipped configs set it to 30, which means every epoch: when restarts were stopped after epoch 3, the daily codebook collapsed back to 4 regimes. Both configs set `entropy_weight` to 0.1.

Normalized features are clipped to ±10 standard deviations so occasional indicator spikes cannot dominate a batch.

### Dynamics head

The dynamics head predicts which regime the next bar will be in. Making it at least match the trivial "stay in the same regime" baseline took several fixes:

| Problem | Fix |
|---|---|
| The target (the next bar's regime) was encoded with dropout active, so it was noisy. | Targets are encoded in eval mode. |
| The dynamics loss flowed into the encoder, moving the current regime (and so the target) while the head chased it. | The head's inputs are detached; the dynamics loss trains only the head. |
| An id-space softmax head cannot represent "mostly stay, sometimes move to A or B" without being knocked out each time ids drift; a single predicted latent with one temperature cannot either. | **Gated head** (`dynamics_head: gated`): p(next) = g · p(stay) + (1 − g) · softmax(move logits), with a per-sample gate g initialized to about 0.95, so the head starts at persistence. |
| p(stay) taken from `softmax(-dist / vq_temperature)` spread "stay" over several near-identical codes, so the gate learned to stop trusting it. | p(stay) is an exact one-hot on the current regime, smoothed by 1%. |
| During joint training the regime ids keep changing meaning, so the dynamics loss can't settle (on daily data it rose from 2.35 to 2.91 between epochs 1 and 2). | Stage-2 fine-tuning on the frozen best checkpoint makes it a stationary supervised problem. |

`dynamics_head` also accepts `logits` (the original id-space head, the default for old configs) and `latent` (next latent = z + residual, scored against the codebook).

## Diagnostics

```bash
python scripts/diagnostics.py --config configs/daily.yaml                                   # baseline only
python scripts/diagnostics.py --config configs/daily.yaml --checkpoint artifacts/daily/best.pt --skip-baseline
```

- **Gradient-boosting baseline:** a `HistGradientBoostingClassifier` on summary features of the same context windows (last bar, recent and full-window means, trailing volatility), predicting the sign of each horizon's return. It is compared with always predicting the training-set majority direction.
- **Regime analysis** (with `--checkpoint`): eta², the share of variance explained by regime membership, for |future return| (volatility) and for signed future return (direction). It is compared with 64 quantile bins of trailing realized volatility. A third figure measures what regimes explain *within* those volatility bins, that is, beyond trailing volatility. For intraday packs it also reports time of day (78 five-minute slots) and time of day × 8 trailing-volatility bins, plus what regimes explain within that table. Use the time of day × volatility table as the benchmark for 5m models: trailing volatility alone ignores the open/midday/close volatility pattern.

With 64 groups and no real effect, eta² is about 63 / n, which is 0.1-0.2% for these splits.

## Results

### 5m model on Alpaca history: `configs/alpaca_5m.yaml`, 529 tickers

Run on 2026-10-19. 100.1M samples across 529 tickers (S&P 500 + the 100 most liquid of 2014; `BRK.B` and `BF.B` have no daily data, and 3 recent listings have too little history). 12 bar features + 7 intraday indicators, and 60 daily features. Train 2016-2024 (81.8M pairs, 1M random pairs per epoch), validation 2025 (10.1M; fixed 200k subset during training), test 2026 (8.2M; 500k-pair subsets for evaluation). Training took about 3.5 hours: 20 epochs at ~10 minutes each, plus a 3-epoch dynamics fine-tune.

Validation NLL was still improving at epoch 20 (−3.29 at epoch 1, −3.85 at epoch 20), so a longer run would probably help.

Share of future |return| variance explained (300k-pair subsets; validation 2025 / test 2026):

| Horizon (5m bars) | 1 | 3 | 12 | 78 |
|---|---|---|---|---|
| Regimes (64) | 20.1 / 21.8% | 22.6 / 25.3% | 23.3 / 25.9% | 12.5 / 11.3% |
| 64 bins of trailing volatility | 7.3 / 4.8% | 7.8 / 5.7% | 9.7 / 6.9% | 12.0 / 10.5% |
| Time of day (78 five-minute slots) | 16.1 / 23.6% | 14.6 / 21.7% | 12.0 / 17.5% | 0.0 / 0.0% |
| **Time of day × 8 trailing-volatility bins** | **24.4 / 31.1%** | **23.5 / 29.8%** | 21.7 / 25.5% | 10.9 / 9.1% |
| Regimes, within time of day × volatility | 2.0 / 1.1% | 2.3 / 1.6% | 2.1 / 1.8% | 1.6 / 2.0% |

**Correction (2026-10-20):** an earlier version of this section compared the regimes only with trailing volatility and reported that they explain "about 4× as much short-horizon volatility". That overstated them. Intraday volatility follows a strong time-of-day pattern (high at the open, low at midday, rising into the close), which the regimes mostly rediscovered and the trailing-volatility benchmark ignores.

Against the fair benchmark, a simple lookup table of time of day × trailing volatility:

- **At 1 bar the table beats the regimes** (31.1% vs 21.8% on test).
- **At 3-12 bars they roughly tie.**
- **At 78 bars (one trading day) the regimes are slightly ahead.** Time of day cancels out over a full day there.
- **Beyond the table the regimes add 1-2%** at every horizon.

So as a volatility forecaster the model adds little over the lookup table. Giving it time of day as an explicit input (step B of the ablation ladder) would force the regimes to learn something beyond the clock.

| | Validation 2025 | Test 2026 |
|---|---|---|
| Regimes used | 64 of 64 | 64 of 64 |
| Next-regime accuracy vs persistence | 72.2% vs 71.2% | 72.5% vs 71.7% |
| Directional accuracy on moves (1 / 3 / 12 / 78 bars) | 50.0 / 49.6 / 50.8 / 50.1% | 50.1 / 49.9 / 50.0 / 50.0% |
| 90% interval coverage | 97.1 / 93.7 / 89.3 / 91.1% | 96.4 / 92.8 / 89.2 / 90.2% |
| Return MAE | 0.0014 / 0.0025 / 0.0051 / 0.0145 | 0.0015 / 0.0027 / 0.0057 / 0.0166 |

- **Dynamics:** this is the first model whose dynamics head beats persistence on 5m data, by about 1 point on both splits. The transition graph has 1,820 edges.
- **Direction:** none. The predicted mean return is uncorrelated with the realized return at every horizon (|corr| ≤ 0.005). Regimes explain at most 0.1% of signed-return variance, including at 78 bars, where the 20-ticker `db_2026` model had shown 2.4%; that now looks like noise from its small universe. Last-bar momentum scores 49-50%, with a faint short-term reversal (corr about −0.006).
- **Calibration:** intervals are too wide at 1 bar (96-97% coverage for a 90% band) and about right from 12 bars on.

### Intraday volatility profile

This is the time-of-day pattern behind the correction above. Medians over 12M randomly sampled regular-session 5m bars from the Alpaca pack (529 tickers, 2016-2026). "Range" is (high − low) / previous close.

| Time (ET) | Median 5m bar range | Range vs midday (12:00-14:00) | Volume vs midday |
|---|---|---|---|
| 09:30 (first bar) | 0.68% | **5.3×** | 4.1× |
| 09:35 | 0.45% | 3.5× | 1.9× |
| 10:00 | 0.30% | 2.4× | 1.9× |
| 10:30 | 0.22% | 1.8× | 1.5× |
| 11:30 | 0.17% | 1.3× | 1.3× |
| 12:30-13:30 | 0.13% | **1.0×** (the low) | 1.0× |
| 14:00 | 0.14% | 1.1× | 1.2× |
| 15:00 | 0.14% | 1.1× | 1.4× |
| 15:45 | 0.17% | 1.3× | 3.5× |
| 15:50 | 0.20% | 1.6× | 5.5× |
| 15:55 (last bar) | 0.23% | 1.8× | **12.5×** |

- **The open is by far the most volatile period.** The first bar moves about 5× more than a midday bar; including the overnight gap, its median |return| is 0.50%. Volatility halves within 30 minutes and keeps decaying until about 12:30.
- **Midday (12:30-13:30) is the quiet trough**, about a fifth of the open's activity.
- **There is a small bump at 14:00**, probably from FOMC days (announcements at 14:00 ET, about 8 times a year).
- **The close picks up**, but only to 1.6-1.8× midday on price moves. Volume rises far more, to 12.5× midday in the last bar, as orders pile into the closing auction.

So price volatility is a reverse J (high open, low midday, modest rise into the close), while volume is a strong U. This shape is why time of day alone explains 24% of next-bar |return| variance on the 2026 test set, more than the model's 64 regimes do. Regenerate it with `PYTHONPATH=src python scripts/intraday_profile.py --pack data/alpaca5m/pack` (add `--csv` for all 78 slots).

### Daily model: `configs/daily.yaml`, 200 tickers

Run on 2026-10-19. 592,234 samples across the 200 most liquid tickers of 2014, with 60 daily features (12 bar features plus 48 indicators). Train 2015-2024 (473k pairs), validation 2025 (49.8k), test 2026 (35.6k, evaluated once per model).

| | v1 | Final | Better |
|---|---|---|---|
| Setup | id-space head, restarts stopped after epoch 3, no stage 2 | gated head, restarts every epoch, stage 2 | |
| Validation NLL of returns (lower is better) | **−1.612** | −1.568 | v1, slightly |
| Return MAE, test (1d / 5d / 20d) | 0.0161 / 0.0367 / 0.0773 | 0.0164 / 0.0370 / 0.0773 | tie |
| 90% interval coverage, test | 89.5 / 89.0 / 87.9% | 89.7 / 89.0 / 88.4% | tie |
| Directional accuracy, test | 50.1 / 49.2 / 51.9% | 49.6 / 49.5 / 52.1% | tie (both coin flips) |
| Regimes used, test | 4 of 64 | 63 of 64 | **final** |
| Transition graph edges | 19 | 2,536 | **final** |
| Next-regime accuracy vs persistence, test | 86.9% vs 94.8% | 63.5% vs 63.3% | **final** |
| \|return\| variance explained by regimes, test | 2.4 / 3.3 / 4.4% | 4.6 / 5.5 / 7.0% | **final** |
| Same, within trailing-volatility bins, test | 0.3 / 0.5 / 1.1% | 0.5 / 0.8 / 1.8% | **final** |

The final model is better **as a world model**. v1 collapsed to 4 regimes, and its dynamics head did worse than always predicting "stay". The final model uses 63 regimes, its dynamics head matches persistence (+1.1 points on validation, +0.2 on test), and its regimes explain about twice as much future volatility.

**As a return forecaster the two are effectively tied.** v1's lower validation NLL comes from a single run each, and the gap (0.04) is about the size of the epoch-to-epoch swings in both runs.

For comparison, 64 bins of trailing volatility explain 4.9 / 5.7 / 6.1% of test |return| variance. The final regimes roughly match that benchmark: they beat it on validation at every horizon and on test at 20 days. In the final model, 63% of daily steps keep the same regime.

### No directional edge, even for a simple baseline

Directional accuracy of the gradient-boosting baseline against the majority direction:

| | Horizon 1 | Horizon 2 | Horizon 3 | Horizon 4 |
|---|---|---|---|---|
| Daily val (1d / 5d / 20d): GBM | 51.8% | 53.1% | 54.5% | |
| Daily val: always "up" | 52.4% | 55.3% | 58.6% | |
| Daily test: GBM | 49.7% | 51.2% | 52.2% | |
| Daily test: always "up" | 50.9% | 51.7% | 51.7% | |
| 5m val (1 / 3 / 12 / 78 bars): GBM | 50.9% | 50.0% | 50.5% | 52.7% |
| 5m val: majority class | 50.0% | 50.2% | 51.3% | 53.8% |
| 5m test: GBM | 50.0% | 50.5% | 50.9% | 51.7% |
| 5m test: majority class | 50.6% | 49.8% | 49.6% | 49.1% |
| Alpaca 5m val, with %B / RSI / MACD / VWAP features: GBM | 52.2% | 51.1% | 50.8% | 51.6% |
| Alpaca 5m val: majority class | 51.8% | 50.5% | 50.7% | 52.0% |
| Alpaca 5m test: GBM | 52.0% | 51.2% | 50.3% | 50.5% |
| Alpaca 5m test: majority class | 51.5% | 50.8% | 49.8% | 51.0% |

The 5m rows count unchanged prices as "down", which is why the majority class can exceed 50%. On Alpaca, the GBM gets at most 0.6 points over the majority class at 1-12 bars, and is 0.4-0.5 below it at 78 bars. That's too small to trust, given how correlated the samples are (overlapping windows, tickers moving together).

The GBM does worse than "always up" on daily validation at every horizon and is within noise on 5m. A simple model on the same inputs cannot predict direction either, so the world model's ~50% reflects the inputs, not a training failure.

### 5m model: `configs/db_2026.yaml`, 20 tickers

Run on 2026-10-19. 237,611 samples (72 daily features, 12 intraday features), 2026 only: train 131k pairs, validation 39k, test 67k. Horizons are 1 / 3 / 12 / 78 bars.

| | v1 | Final | Better |
|---|---|---|---|
| Setup | id-space head, restarts stopped after epoch 3, no stage 2 | gated head, restarts every epoch, stage 2 | |
| Validation NLL of returns (lower is better) | −3.503 | **−3.535** | final, slightly |
| Return MAE, test | 0.0022 / 0.0028 / 0.0056 / 0.0151 | 0.0020 / 0.0029 / 0.0052 / 0.0160 | tie |
| 90% interval coverage, test | 94.6 / 93.3 / 88.7 / 86.9% | 95.9 / 93.8 / 94.9 / 88.9% | tie |
| Directional accuracy, test | 49.5 / 49.7 / 49.5 / 50.3% | 49.9 / 49.9 / 50.5 / 48.5% | tie (both coin flips) |
| Regimes used, test | 63 of 64 | 63 of 64 | tie |
| Next-regime accuracy vs persistence, test | 15.9% vs 81.7% | 82.9% vs 83.0% | **final** |
| \|return\| variance explained by regimes, test | 5.3 / 9.0 / 10.0 / 5.1% | 8.2 / 11.3 / 12.8 / 11.1% | **final** |
| Same, within trailing-volatility bins, test | 4.9 / 8.7 / 9.4 / 2.5% | 6.3 / 9.5 / 11.0 / 4.7% | **final** |

For comparison, 64 bins of trailing volatility explain 3.7 / 3.9 / 4.9 / 8.9% of test |return| variance. Time of day × 8 trailing-volatility bins explains **39.2 / 35.8 / 30.2 / 7.0%**, and the regimes add only 0.7 / 0.8 / 1.2 / 5.2% within it.

**Correction (2026-10-20):** an earlier version said these regimes "beat the trailing-volatility benchmark at every horizon" and called that the clearest evidence that the regimes learn something useful. Measured against time of day × volatility, they are far behind at 1-12 bars. Their one notable figure is at 78 bars, 5.2% beyond the table, but it comes from 20 tickers and a single test period, and the much larger Alpaca model adds only 2.0% there.

The dynamics head now matches persistence, up from 15.9% for v1. Stage 2 lowered its validation loss from 1.17 to 0.77 but never changed its top prediction away from "stay". At 5-minute resolution, which regime comes next is no more predictable than "the same one" (84% of steps stay put).

Signed returns are a different story. Regimes explain almost nothing at 1-12 bars (≤0.6% on test). At 78 bars they explain 2.4% on test and 3.4% on validation, well above the ~0.1% noise level, but treat that with caution. Overlapping one-day returns and the 20 tickers moving together can make regimes stand in for particular trading days. The same caveat applies to the daily model's 20-day figure (1.8% on test). A date-clustered test is needed before reading it as directional information.

#### Earlier 5m run (before the dynamics fixes)

Best epoch 7 (validation NLL −3.48), 62 of 64 regimes used, 1,426 transition edges. Test directional accuracy 49.4 / 49.6 / 50.4 / 49.6% and 90% interval coverage 89.3 / 95.9 / 94.3 / 82.2% at 1 / 3 / 12 / 78 bars. Training took 3h18m with single-threaded data loading. The 6-ticker run before that, without indicators and with the original quantizer, used 1 regime with a 1-edge transition graph.

## Evaluate frozen 2026 test

```bash
python -m mwm.evaluate --checkpoint artifacts/best.pt --split test
```

Reported metrics include per-horizon:

- MAE of future log return mean
- directional accuracy
- 90% predictive interval coverage
- number of latent regimes actually used
- `next_regime_accuracy` against `persistence_accuracy` (the "stay the same" baseline)
- `directional_accuracy_moves`: directional accuracy over samples whose price actually moved, plus `zero_return_share`, `up_share_moves` and `predicted_up_share`. Plain `directional_accuracy` counts every unchanged price as a miss, which on 5m bars pulls it below 50% by itself.

`--max-samples N` evaluates a fixed random subset of pairs, which large lazy packs need.

## Build latent transition graph

```bash
python -m mwm.transitions --checkpoint artifacts/best.pt --split train --out artifacts/transitions.npz
```

This produces:

- `counts[K,K]`
- `probs[K,K]`

for learned regime transitions `z_t -> z_{t+1}` within each symbol. It also prints the self-transition share.

## Dynamics training

Each dataset item contains both the current context window and the immediately following window (next bar) for the same symbol. Therefore random mini-batch shuffling is safe: chronological adjacency is preserved inside each training example. The model learns:

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

The Alpaca 5m model is the strongest testbed: it has the intended 2016-2024 / 2025 / 2026 split and 529 tickers. Its regimes, however, mostly capture time of day and trailing volatility (see the correction above), so step B (calendar/time embeddings) comes first, and every later encoder should be judged against the time of day × volatility table. It takes hours to train. The daily model trains in minutes and is the quicker place to try an encoder before confirming on 5m. Since neither the world model nor the GBM baseline predicts direction, judge new encoders on regime quality and volatility explained, not only on directional accuracy.

## Important modeling note

The 64-code latent space is **not assumed to equal the 64 hexagrams**. The model is allowed to discover its own regimes. Only after training should we compare the learned transition graph with an I-Ching graph. That keeps the experiment falsifiable and avoids hard-coding the desired answer.
