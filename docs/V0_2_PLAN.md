# Market World Model v0.2 — market-wide state before more model scale

## Why

v0.1 learns persistent volatility/time-of-day regimes, while absolute directional return prediction is approximately chance. Before adding capacity, v0.2 tests whether the frozen representation contains **cross-sectional alpha**: can it rank stocks at the same timestamp even when it cannot predict the market's common direction?

This follows the world-model principle that the latent state should support multiple downstream tasks, not only one absolute-return target.

## Phase A (this PR): frozen cross-sectional probe

`scripts/cross_sectional_probe.py` groups predictions by timestamp and reports, for every horizon:

- Pearson IC across stocks
- Spearman/rank IC across stocks
- relative-direction accuracy after cross-sectional demeaning
- realised top-minus-bottom decile spread ranked by predicted return
- standard errors across timestamps

The probe deliberately does **not** retrain the network. This separates representation quality from a new head's capacity.

Run:

```bash
PYTHONPATH=src python scripts/cross_sectional_probe.py \
  --checkpoint artifacts/alpaca5m/best.pt \
  --split test --max-days 40 --min-assets 100
```

Use complete days rather than a random sample: random row subsampling destroys the cross-section at each timestamp.

## Decision gate

1. **If rank IC / spread is positive and stable in both 2025 and frozen 2026:** keep the encoder and add a cross-sectional return head trained on market-neutral targets.
2. **If cross-sectional metrics are also null:** the single-stock observation is insufficient. Add explicit market-wide context before changing the return head.

## Phase B: explicit global state

Add a separate global encoder `G_t` built only from information available at time `t`:

- SPY / QQQ / IWM / sector ETF sequences
- cross-sectional breadth (% above VWAP, % positive 5m return)
- cross-sectional return dispersion
- volume breadth / up-down volume
- market and sector realized volatility

Then factor the latent state into:

`Z_{i,t} = f(Local_{i,t}, Global_t, Sector_{i,t}, Stock_i)`

and predict both:

- common market component
- stock-specific residual / relative return

so `r_{i,t+h} = beta_{i,t} * r_market,t+h + alpha_{i,t+h}`.

The regime state should be learned primarily from `Global_t`; the stock latent should model conditional response to that regime. This avoids asking 529 independent stock windows to rediscover the same market state.

## Phase C: dynamics and counterfactuals

After Phase B establishes useful representation:

- predict multi-step latent rollouts, not only next 5m regime
- condition transitions on market shocks/actions (rates, index move, volatility shock)
- score latent rollouts by downstream probe accuracy and calibration
- retain 2026 as frozen OOS; tune architecture only on 2025

## Ancient time encoders

Do **not** add Ganzhi/I-Ching/Qimen features until the market-data baseline above is established. Later they can be introduced as isolated time encoders in an ablation arm and compared against ordinary calendar/astronomical encodings. This prevents symbolic priors from being mistaken for learned market information.
