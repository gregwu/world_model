from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import yaml
import contextlib
from .data import fit_stats, make_loader, open_dataset
from .model import MarketWorldModel
from .losses import gaussian_nll, code_usage_entropy, soft_code_usage_entropy


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_from(cfg):
    wanted = cfg["train"]["device"]
    if wanted != "auto": return torch.device(wanted)
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def build_model(ds, num_symbols: int, model_cfg: dict) -> MarketWorldModel:
    return MarketWorldModel(ds.daily_dim, ds.intra_dim, num_symbols, ds.returns.shape[-1], model_cfg)


def autocast_for(device, amp: str | None):
    """bf16 autocast on CUDA when amp == "bf16"; the vector quantizer opts out and stays fp32."""
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def run_model(model, b, nxt=False):
    """Forward a batch through either model family: v0.1 MarketWorldModel(daily, intraday, symbol) or the pure-5m
    IntradayWorldModel(intraday, symbol). nxt=True uses the batch's next-bar windows."""
    p = "next_" if nxt else ""
    if getattr(model, "pure_intraday", False):
        return model(b[p + "intraday"], b["symbol_id"])
    return model(b.get(p + "daily"), b.get(p + "intraday"), b["symbol_id"])


def encode_regime(model, b, nxt=False):
    p = "next_" if nxt else ""
    if getattr(model, "pure_intraday", False):
        return model.encode(b[p + "intraday"], b["symbol_id"])[1]
    return model.encode(b.get(p + "daily"), b.get(p + "intraday"), b["symbol_id"])[1]


@torch.no_grad()
def encode_next_regime(model, b, amp=None):
    """Regime of the actual next observation, encoded in eval mode so dropout doesn't make the target noisy."""
    was_training = model.training
    model.eval()
    with autocast_for(b["symbol_id"].device, amp):
        regime = encode_regime(model, b, nxt=True)
    model.train(was_training)
    return regime


METRICS = ("loss", "nll", "vq", "entropy", "usage_entropy", "dynamics_ce", "next_regime_acc", "persistence_acc")


def run_epoch(model, loader, opt, device, loss_cfg, num_regimes, train=True, allow_restarts=False, grad_clip=1.0,
              module_train_mode=None, amp=None):
    model.train(train if module_train_mode is None else module_train_mode)
    totals = torch.zeros(len(METRICS), device=device); restarts = torch.zeros((), device=device); n = 0
    for b in loader:
        b = to_device(b, device); y = b["future_returns"]
        # The next-state target is encoded from the actual next observation, but detached.
        # This is self-supervised dynamics learning, not a label from the future-return head.
        next_regime = encode_next_regime(model, b, amp)
        with torch.set_grad_enabled(train), autocast_for(device, amp):
            out = run_model(model, b)
            nll = gaussian_nll(y, out["return_mu"].float(), out["return_log_sigma"].float())
            dyn = torch.nn.functional.cross_entropy(out["next_regime_logits"].float(), next_regime)
            entropy = soft_code_usage_entropy(out["regime_probs"])
            loss = (loss_cfg["return_weight"]*nll + loss_cfg["dynamics_weight"]*dyn
                    + loss_cfg["vq_weight"]*out["vq_loss"] - loss_cfg["entropy_weight"]*entropy)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip); opt.step()
                if allow_restarts:
                    restarts += model.vq.restart_dead_codes(out["z_e"].detach())
        with torch.no_grad():
            vals = torch.stack([
                loss, nll, out["vq_loss"], entropy, code_usage_entropy(out["regime"], num_regimes), dyn,
                (out["next_regime_logits"].argmax(-1) == next_regime).float().mean(),
                (out["regime"] == next_regime).float().mean(),  # baseline: "regime stays the same"
            ]).detach()
        bs = y.size(0); n += bs; totals += vals * bs
    res = dict(zip(METRICS, (totals / max(n, 1)).tolist()))
    if train: res["code_restarts"] = int(restarts.item())
    return res


DYNAMICS_PARAMS = ("next_regime", "stay_gate", "next_latent", "dynamics_log_scale")
DYNAMICS_ONLY_LOSS = {"return_weight": 0.0, "dynamics_weight": 1.0, "vq_weight": 0.0, "entropy_weight": 0.0}


def finetune_dynamics(model, ck, tr, va, dev, cfg):
    """Stage 2: freeze encoder, codebook and return head at the best checkpoint, then fit only the dynamics head.
    During joint training the regime ids keep changing meaning, so the head chases a moving target; on a frozen
    representation P(next regime | current state) is a stationary supervised problem."""
    tc = cfg["train"]; data = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model"])
    params = [p for n, p in model.named_parameters() if n.split(".")[0] in DYNAMICS_PARAMS]
    for p in model.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    best = float("inf")
    for epoch in range(1, tc["dynamics_finetune_epochs"] + 1):
        t0 = time.time()
        # module_train_mode=False: frozen backbone runs without dropout, matching how targets and eval are encoded
        tm = run_epoch(model, tr, opt, dev, DYNAMICS_ONLY_LOSS, cfg["model"]["num_regimes"], True, False,
                       tc.get("grad_clip", 1.0), module_train_mode=False, amp=tc.get("amp"))
        vm = run_epoch(model, va, None, dev, DYNAMICS_ONLY_LOSS, cfg["model"]["num_regimes"], False, amp=tc.get("amp"))
        print(json.dumps({"epoch": f"dyn{epoch}", "seconds": round(time.time() - t0, 1), "train": tm, "val": vm}), flush=True)
        if vm["dynamics_ce"] < best:
            best = vm["dynamics_ce"]; data["model"] = model.state_dict(); data["dynamics_finetune_epoch"] = epoch
            torch.save(data, ck)
    print(f"best_val_dynamics_ce={best:.6f} checkpoint={ck}", flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/base.yaml")
    ap.add_argument("--dynamics-only",action="store_true",help="skip stage 1; fine-tune the dynamics head of the existing checkpoint")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_all(cfg["seed"]); dev=device_from(cfg); tc=cfg["train"]
    p=cfg["data"]["out_path"]; s=cfg["split"]; amp=tc.get("amp")
    ck=Path(tc["checkpoint"]); ck.parent.mkdir(parents=True,exist_ok=True)
    if args.dynamics_only:
        stats=torch.load(ck,map_location="cpu",weights_only=False)["stats"]  # must match the checkpoint's normalization
    else:
        stats=fit_stats(open_dataset(p,s["train_start"],s["train_end"]))
    train=open_dataset(p,s["train_start"],s["train_end"],stats)
    # Large packs: validate on a fixed random subset; train on a fresh random subset each epoch.
    val=open_dataset(p,s["val_start"],s["val_end"],stats,max_samples=tc.get("val_max_samples"),seed=cfg["seed"])
    per_epoch=tc.get("samples_per_epoch")
    print(json.dumps({"train_pairs":len(train),"val_pairs":len(val),"samples_per_epoch":per_epoch or len(train),
                      "device":str(dev),"amp":amp}), flush=True)
    model=build_model(train,len(train.symbols),cfg["model"]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=tc["lr"],weight_decay=tc["weight_decay"])
    tr=make_loader(train,tc["batch_size"],True,tc["num_workers"],per_epoch)
    va=make_loader(val,tc["batch_size"],False,tc["num_workers"])
    restart_epochs=tc.get("vq_restart_epochs",3)
    best=float("inf"); bad=0
    if args.dynamics_only:
        finetune_dynamics(model,ck,tr,va,dev,{**cfg,"train":{**tc,"dynamics_finetune_epochs":tc.get("dynamics_finetune_epochs",5)}})
        return
    for epoch in range(1,tc["epochs"]+1):
        t0=time.time()
        # Dead-code restarts for the first vq_restart_epochs epochs (the shipped configs: every epoch).
        tm=run_epoch(model,tr,opt,dev,cfg["loss"],cfg["model"]["num_regimes"],True,epoch<=restart_epochs,tc.get("grad_clip",1.0),amp=amp)
        vm=run_epoch(model,va,None,dev,cfg["loss"],cfg["model"]["num_regimes"],False,amp=amp)
        print(json.dumps({"epoch":epoch,"seconds":round(time.time()-t0,1),"train":tm,"val":vm}), flush=True)
        if vm["nll"]<best:
            best=vm["nll"]; bad=0
            torch.save({"model":model.state_dict(),"cfg":cfg,"stats":stats,"symbols":train.symbols,"epoch":epoch},ck)
        else:
            bad+=1
            if bad>=tc["patience"]: break
    print(f"best_val_nll={best:.6f} checkpoint={ck}", flush=True)
    if tc.get("dynamics_finetune_epochs", 0):
        finetune_dynamics(model, ck, tr, va, dev, cfg)

if __name__=="__main__": main()
