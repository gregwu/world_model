from __future__ import annotations
import argparse, contextlib, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from mwm.data import fit_stats, make_loader, open_dataset
from mwm.intraday_model import IntradayWorldModel
from mwm.losses import gaussian_nll, soft_code_usage_entropy
from mwm.train import device_from, to_device, autocast_for


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model(ds, cfg):
    return IntradayWorldModel(ds.intra_dim, len(ds.symbols), ds.returns.shape[-1], cfg["model"])


@torch.no_grad()
def next_regime(model, b, amp):
    was = model.training; model.eval()
    with autocast_for(b["intraday"].device, amp):
        r = model.encode(b["next_intraday"], b["symbol_id"])[1]
    model.train(was)
    return r


def cross_sectional_relative(y, timestamps):
    """Demean future returns within each exact timestamp in the batch.

    Random batches only approximate a full market cross-section, so this is an auxiliary loss, not the primary
    target. A timestamp-aware batch sampler is the next step if this probe proves useful.
    """
    target = y.clone()
    for t in torch.unique(timestamps):
        m = timestamps == t
        if m.sum() >= 2: target[m] -= target[m].mean(0, keepdim=True)
        else: target[m] = 0
    return target


def run_epoch(model, loader, opt, dev, cfg, train):
    model.train(train); amp=cfg["train"].get("amp"); lc=cfg["loss"]
    totals={k:0.0 for k in ("loss","nll","dyn","relative","entropy")}; n=0
    for b in loader:
        b=to_device(b,dev); y=b["future_returns"]
        nxt=next_regime(model,b,amp)
        with torch.set_grad_enabled(train), autocast_for(dev,amp):
            out=model(b["intraday"],b["symbol_id"])
            nll=gaussian_nll(y,out["return_mu"].float(),out["return_log_sigma"].float())
            dyn=F.cross_entropy(out["next_regime_logits"].float(),nxt)
            rel_target=cross_sectional_relative(y,b["timestamp_ns"])
            relative=F.smooth_l1_loss(out["relative_mu"].float(),rel_target)
            entropy=soft_code_usage_entropy(out["regime_probs"])
            loss=(lc["return_weight"]*nll + lc["dynamics_weight"]*dyn + lc["vq_weight"]*out["vq_loss"]
                  - lc["entropy_weight"]*entropy + lc.get("relative_weight",0.0)*relative)
        if train:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["train"].get("grad_clip",1.0)); opt.step()
        bs=y.size(0); n+=bs
        for k,v in (("loss",loss),("nll",nll),("dyn",dyn),("relative",relative),("entropy",entropy)):
            totals[k]+=float(v.detach())*bs
    return {k:v/max(n,1) for k,v in totals.items()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/alpaca_5m_pure.yaml"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_all(cfg["seed"]); dev=device_from(cfg); s=cfg["split"]; tc=cfg["train"]
    raw=open_dataset(cfg["data"]["out_path"],s["train_start"],s["train_end"])
    stats=fit_stats(raw)
    train=open_dataset(cfg["data"]["out_path"],s["train_start"],s["train_end"],stats)
    val=open_dataset(cfg["data"]["out_path"],s["val_start"],s["val_end"],stats,max_samples=tc.get("val_max_samples"),seed=cfg["seed"])
    tr=make_loader(train,tc["batch_size"],True,tc["num_workers"],tc.get("samples_per_epoch"))
    va=make_loader(val,tc["batch_size"],False,tc["num_workers"])
    model=build_model(train,cfg).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=tc["lr"],weight_decay=tc["weight_decay"])
    ck=Path(tc["checkpoint"]); ck.parent.mkdir(parents=True,exist_ok=True); best=float("inf"); bad=0
    print(json.dumps({"train_pairs":len(train),"val_pairs":len(val),"device":str(dev),"pure_5m":True}),flush=True)
    for epoch in range(1,tc["epochs"]+1):
        t=time.time(); tm=run_epoch(model,tr,opt,dev,cfg,True); vm=run_epoch(model,va,None,dev,cfg,False)
        print(json.dumps({"epoch":epoch,"seconds":round(time.time()-t,1),"train":tm,"val":vm}),flush=True)
        if vm["nll"]<best:
            best=vm["nll"]; bad=0
            torch.save({"model":model.state_dict(),"cfg":cfg,"stats":stats,"symbols":train.symbols,"epoch":epoch,"model_type":"pure_5m"},ck)
        else:
            bad+=1
            if bad>=tc["patience"]: break
    print(f"best_val_nll={best:.6f} checkpoint={ck}")

if __name__=="__main__": main()
