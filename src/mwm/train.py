from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from .data import WorldModelDataset, fit_stats
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


def run_epoch(model, loader, opt, device, loss_cfg, num_regimes, train=True):
    model.train(train)
    totals = {"loss":0., "nll":0., "vq":0., "entropy":0., "usage_entropy":0., "dynamics_ce":0.}; n=0; restarts=0
    for b in loader:
        daily=b["daily"].to(device); intra=b["intraday"].to(device); sid=b["symbol_id"].to(device); y=b["future_returns"].to(device)
        nd=b["next_daily"].to(device); ni=b["next_intraday"].to(device)
        with torch.set_grad_enabled(train):
            out=model(daily,intra,sid)
            # The next-state target is encoded from the actual next 5m observation, but detached.
            # This is self-supervised dynamics learning, not a label from the future-return head.
            with torch.no_grad():
                next_regime = model.encode(nd, ni, sid)[1]
            nll=gaussian_nll(y,out["return_mu"],out["return_log_sigma"])
            dyn= torch.nn.functional.cross_entropy(out["next_regime_logits"], next_regime)
            entropy=soft_code_usage_entropy(out["regime_probs"])
            usage_entropy=code_usage_entropy(out["regime"], num_regimes)
            loss=(loss_cfg["return_weight"]*nll + loss_cfg["dynamics_weight"]*dyn
                  + loss_cfg["vq_weight"]*out["vq_loss"] - loss_cfg["entropy_weight"]*entropy)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward();
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                restarts += model.vq.restart_dead_codes(out["z_e"].detach())
        bs=y.size(0); n+=bs
        for k,v in [("loss",loss),("nll",nll),("vq",out["vq_loss"]),("entropy",entropy),("usage_entropy",usage_entropy),("dynamics_ce",dyn)]: totals[k]+=float(v.detach().cpu())*bs
    res={k:v/max(n,1) for k,v in totals.items()}
    if train: res["code_restarts"]=restarts
    return res


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/base.yaml"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_all(cfg["seed"]); dev=device_from(cfg)
    p=cfg["data"]["out_path"]; s=cfg["split"]
    raw_train=WorldModelDataset(p,s["train_start"],s["train_end"]); stats=fit_stats(raw_train); del raw_train
    train=WorldModelDataset(p,s["train_start"],s["train_end"],stats); val=WorldModelDataset(p,s["val_start"],s["val_end"],stats)
    model=MarketWorldModel(train.daily.shape[-1],train.intraday.shape[-1],len(train.symbols),train.returns.shape[-1],cfg["model"]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["train"]["lr"],weight_decay=cfg["train"]["weight_decay"])
    tr=DataLoader(train,batch_size=cfg["train"]["batch_size"],shuffle=True,num_workers=cfg["train"]["num_workers"])
    va=DataLoader(val,batch_size=cfg["train"]["batch_size"],shuffle=False,num_workers=cfg["train"]["num_workers"])
    best=float("inf"); bad=0; ck=Path(cfg["train"]["checkpoint"]); ck.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,cfg["train"]["epochs"]+1):
        tm=run_epoch(model,tr,opt,dev,cfg["loss"],cfg["model"]["num_regimes"],True)
        vm=run_epoch(model,va,None,dev,cfg["loss"],cfg["model"]["num_regimes"],False)
        print(json.dumps({"epoch":epoch,"train":tm,"val":vm}))
        if vm["nll"]<best:
            best=vm["nll"]; bad=0
            torch.save({"model":model.state_dict(),"cfg":cfg,"stats":stats,"symbols":train.symbols},ck)
        else:
            bad+=1
            if bad>=cfg["train"]["patience"]: break
    print(f"best_val_nll={best:.6f} checkpoint={ck}")

if __name__=="__main__": main()
