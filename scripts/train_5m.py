"""Train the pure 5-minute v0.2 IntradayWorldModel (src/mwm/intraday_model.py).

Stage 1 trains the encoders, codebook, return head and market-neutral relative head and keeps the epoch with the
best validation NLL. Stage 2 (train.dynamics_finetune_epochs > 0) freezes everything except the dynamics head and
fits it on the frozen representation, keeping the epoch with the best validation dynamics loss.

With train.cross_sectional_timestamps = K, each batch holds K timestamps x batch_size/K symbols, so the relative
loss sees real cross-sections (random batches from ~100M pairs almost never put two symbols at one timestamp).

Usage: PYTHONPATH=src python scripts/train_5m.py --config configs/alpaca_5m_pure.yaml
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from mwm.data import fit_stats, make_cross_sectional_loader, make_loader, open_dataset
from mwm.intraday_model import IntradayWorldModel
from mwm.losses import code_usage_entropy, gaussian_nll, soft_code_usage_entropy
from mwm.train import autocast_for, device_from, encode_next_regime, to_device

DYNAMICS_PARAMS = ("next_regime", "stay_gate", "next_latent", "dynamics_log_scale")
METRICS = ("loss", "nll", "dyn", "relative", "entropy", "usage_entropy", "next_regime_acc", "persistence_acc",
           "relative_corr")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model(ds, cfg):
    return IntradayWorldModel(ds.intra_dim, len(ds.symbols), ds.returns.shape[-1], cfg["model"])


def cross_sectional_relative(y, timestamps, scale):
    """Future returns minus their mean across the symbols at the same timestamp, divided by a per-horizon scale.
    Returns (target, valid); rows whose timestamp has fewer than two symbols in the batch are invalid."""
    _, inv, counts = torch.unique(timestamps, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(counts), y.size(1), device=y.device, dtype=y.dtype).index_add_(0, inv, y)
    target = (y - (sums / counts[:, None])[inv]) / scale
    return target, counts[inv] >= 2


def run_epoch(model, loader, opt, dev, cfg, train, scale, allow_restarts=False, loss_cfg=None, module_train_mode=None):
    model.train(train if module_train_mode is None else module_train_mode)
    amp = cfg["train"].get("amp"); lc = loss_cfg or cfg["loss"]; k = cfg["model"].get("num_regimes", 64)
    totals = torch.zeros(len(METRICS), device=dev); restarts = torch.zeros((), device=dev); n = 0
    for b in loader:
        b = to_device(b, dev); y = b["future_returns"]
        nxt = encode_next_regime(model, b, amp)
        with torch.set_grad_enabled(train), autocast_for(dev, amp):
            out = model(b["intraday"], b["symbol_id"])
            nll = gaussian_nll(y, out["return_mu"].float(), out["return_log_sigma"].float())
            dyn = F.cross_entropy(out["next_regime_logits"].float(), nxt)
            rel_target, valid = cross_sectional_relative(y, b["timestamp_ns"], scale)
            rel_mu = out["relative_mu"].float()
            relative = F.smooth_l1_loss(rel_mu[valid], rel_target[valid]) if valid.any() else rel_mu.sum() * 0
            entropy = soft_code_usage_entropy(out["regime_probs"])
            loss = (lc["return_weight"] * nll + lc["dynamics_weight"] * dyn + lc["vq_weight"] * out["vq_loss"]
                    - lc["entropy_weight"] * entropy + lc.get("relative_weight", 0.0) * relative)
        if train:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["train"].get("grad_clip", 1.0))
            opt.step()
            if allow_restarts:
                restarts += model.vq.restart_dead_codes(out["z_e"].detach())
        with torch.no_grad():
            if valid.any():
                a, t = rel_mu[valid].flatten(), rel_target[valid].flatten()
                corr = ((a - a.mean()) * (t - t.mean())).mean() / (a.std() * t.std() + 1e-12)
            else:
                corr = torch.zeros((), device=dev)
            vals = torch.stack([loss, nll, dyn, relative, entropy, code_usage_entropy(out["regime"], k),
                                (out["next_regime_logits"].argmax(-1) == nxt).float().mean(),
                                (out["regime"] == nxt).float().mean(), corr]).detach()
        bs = y.size(0); n += bs; totals += vals * bs
    res = dict(zip(METRICS, (totals / max(n, 1)).tolist()))
    if train: res["code_restarts"] = int(restarts.item())
    return res


def make_loaders(train, val, cfg):
    tc = cfg["train"]; k = tc.get("cross_sectional_timestamps")
    if k:
        tr = make_cross_sectional_loader(train, tc["batch_size"], k, tc.get("samples_per_epoch") or len(train),
                                         tc["num_workers"], cfg["seed"])
        va = make_cross_sectional_loader(val, tc["batch_size"], k, tc.get("val_max_samples") or len(val),
                                         tc["num_workers"], cfg["seed"], fixed=True)
    else:
        tr = make_loader(train, tc["batch_size"], True, tc["num_workers"], tc.get("samples_per_epoch"))
        va = make_loader(val, tc["batch_size"], False, tc["num_workers"])
    return tr, va


def finetune_dynamics(model, ck, tr, va, dev, cfg, scale):
    """Stage 2: fit only the dynamics head on the frozen best checkpoint (a stationary target for the head)."""
    data = torch.load(ck, map_location="cpu", weights_only=False); model.load_state_dict(data["model"])
    params = [p for n, p in model.named_parameters() if n.split(".")[0] in DYNAMICS_PARAMS]
    for p in model.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    only_dyn = {"return_weight": 0.0, "dynamics_weight": 1.0, "vq_weight": 0.0, "entropy_weight": 0.0, "relative_weight": 0.0}
    best = float("inf")
    for epoch in range(1, cfg["train"]["dynamics_finetune_epochs"] + 1):
        t = time.time()
        tm = run_epoch(model, tr, opt, dev, cfg, True, scale, loss_cfg=only_dyn, module_train_mode=False)
        vm = run_epoch(model, va, None, dev, cfg, False, scale, loss_cfg=only_dyn)
        print(json.dumps({"epoch": f"dyn{epoch}", "seconds": round(time.time() - t, 1), "train": tm, "val": vm}), flush=True)
        if vm["dyn"] < best:
            best = vm["dyn"]; data["model"] = model.state_dict(); data["dynamics_finetune_epoch"] = epoch
            torch.save(data, ck)
    print(f"best_val_dynamics_ce={best:.6f} checkpoint={ck}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/alpaca_5m_pure.yaml"); args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); seed_all(cfg["seed"]); dev = device_from(cfg); s = cfg["split"]; tc = cfg["train"]
    p = cfg["data"]["out_path"]
    stats = fit_stats(open_dataset(p, s["train_start"], s["train_end"]))
    train = open_dataset(p, s["train_start"], s["train_end"], stats)
    # Cross-sectional validation batches need whole cross-sections, so subsample by batches, not by rows.
    val = open_dataset(p, s["val_start"], s["val_end"], stats,
                       max_samples=None if tc.get("cross_sectional_timestamps") else tc.get("val_max_samples"), seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    sample = train.returns[np.sort(rng.choice(len(train.returns), min(1_000_000, len(train.returns)), replace=False))]
    scale = torch.tensor(np.asarray(sample).std(0) + 1e-8, dtype=torch.float32, device=dev)  # per-horizon return scale
    tr, va = make_loaders(train, val, cfg)
    model = build_model(train, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    ck = Path(tc["checkpoint"]); ck.parent.mkdir(parents=True, exist_ok=True); best = float("inf"); bad = 0
    print(json.dumps({"train_pairs": len(train), "val_pairs": len(val), "batches_per_epoch": len(tr), "val_batches": len(va),
                      "device": str(dev), "pure_5m": True, "dynamics_head": cfg["model"].get("dynamics_head", "latent"),
                      "relative_scale": scale.tolist()}), flush=True)
    restart_epochs = tc.get("vq_restart_epochs", tc["epochs"])
    for epoch in range(1, tc["epochs"] + 1):
        t = time.time()
        tm = run_epoch(model, tr, opt, dev, cfg, True, scale, allow_restarts=epoch <= restart_epochs)
        vm = run_epoch(model, va, None, dev, cfg, False, scale)
        print(json.dumps({"epoch": epoch, "seconds": round(time.time() - t, 1), "train": tm, "val": vm}), flush=True)
        if vm["nll"] < best:
            best = vm["nll"]; bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "stats": stats, "symbols": train.symbols, "epoch": epoch,
                        "model_type": "pure_5m", "relative_scale": scale.tolist()}, ck)
        else:
            bad += 1
            if bad >= tc["patience"]: break
    print(f"best_val_nll={best:.6f} checkpoint={ck}", flush=True)
    if tc.get("dynamics_finetune_epochs", 0):
        finetune_dynamics(model, ck, tr, va, dev, cfg, scale)


if __name__ == "__main__":
    main()
