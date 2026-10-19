from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import yaml
from .data import WorldModelDataset, fit_stats, make_loader
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


def build_model(ds: WorldModelDataset, num_symbols: int, model_cfg: dict) -> MarketWorldModel:
    intra_dim = ds.intraday.shape[-1] if ds.intraday is not None else 0
    return MarketWorldModel(ds.daily.shape[-1], intra_dim, num_symbols, ds.returns.shape[-1], model_cfg)


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def encode_next_regime(model, b):
    """Regime of the actual next observation, encoded in eval mode so dropout doesn't make the target noisy."""
    was_training = model.training
    model.eval()
    regime = model.encode(b["next_daily"], b.get("next_intraday"), b["symbol_id"])[1]
    model.train(was_training)
    return regime


METRICS = ("loss", "nll", "vq", "entropy", "usage_entropy", "dynamics_ce", "next_regime_acc", "persistence_acc")


def run_epoch(model, loader, opt, device, loss_cfg, num_regimes, train=True, allow_restarts=False, grad_clip=1.0,
              module_train_mode=None):
    model.train(train if module_train_mode is None else module_train_mode)
    totals = torch.zeros(len(METRICS), device=device); restarts = torch.zeros((), device=device); n = 0
    for b in loader:
        b = to_device(b, device); y = b["future_returns"]
        # The next-state target is encoded from the actual next observation, but detached.
        # This is self-supervised dynamics learning, not a label from the future-return head.
        next_regime = encode_next_regime(model, b)
        with torch.set_grad_enabled(train):
            out = model(b["daily"], b.get("intraday"), b["symbol_id"])
            nll = gaussian_nll(y, out["return_mu"], out["return_log_sigma"])
            dyn = torch.nn.functional.cross_entropy(out["next_regime_logits"], next_regime)
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
                       tc.get("grad_clip", 1.0), module_train_mode=False)
        vm = run_epoch(model, va, None, dev, DYNAMICS_ONLY_LOSS, cfg["model"]["num_regimes"], False)
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
    p=cfg["data"]["out_path"]; s=cfg["split"]
    raw_train=WorldModelDataset(p,s["train_start"],s["train_end"]); stats=fit_stats(raw_train); del raw_train
    train=WorldModelDataset(p,s["train_start"],s["train_end"],stats); val=WorldModelDataset(p,s["val_start"],s["val_end"],stats)
    print(json.dumps({"train_pairs":len(train),"val_pairs":len(val),"device":str(dev)}), flush=True)
    model=build_model(train,len(train.symbols),cfg["model"]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=tc["lr"],weight_decay=tc["weight_decay"])
    tr=make_loader(train,tc["batch_size"],True,tc["num_workers"])
    va=make_loader(val,tc["batch_size"],False,tc["num_workers"])
    restart_epochs=tc.get("vq_restart_epochs",3)
    best=float("inf"); bad=0; ck=Path(tc["checkpoint"]); ck.parent.mkdir(parents=True,exist_ok=True)
    if args.dynamics_only:
        stats=torch.load(ck,map_location="cpu",weights_only=False)["stats"]  # must match the checkpoint's normalization
        train=WorldModelDataset(p,s["train_start"],s["train_end"],stats); val=WorldModelDataset(p,s["val_start"],s["val_end"],stats)
        tr=make_loader(train,tc["batch_size"],True,tc["num_workers"]); va=make_loader(val,tc["batch_size"],False,tc["num_workers"])
        finetune_dynamics(model,ck,tr,va,dev,{**cfg,"train":{**tc,"dynamics_finetune_epochs":tc.get("dynamics_finetune_epochs",5)}})
        return
    for epoch in range(1,tc["epochs"]+1):
        t0=time.time()
        # Dead-code restarts only during warm-up; afterwards regime ids keep a stable meaning for the dynamics head.
        tm=run_epoch(model,tr,opt,dev,cfg["loss"],cfg["model"]["num_regimes"],True,epoch<=restart_epochs,tc.get("grad_clip",1.0))
        vm=run_epoch(model,va,None,dev,cfg["loss"],cfg["model"]["num_regimes"],False)
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
