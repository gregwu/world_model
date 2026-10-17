from __future__ import annotations
import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import WorldModelDataset
from .model import MarketWorldModel
from .train import device_from


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",default="artifacts/best.pt"); ap.add_argument("--split",choices=["val","test"],default="test"); args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False); cfg=ck["cfg"]; s=cfg["split"]
    start,end=(s["val_start"],s["val_end"]) if args.split=="val" else (s["test_start"],s["test_end"])
    ds=WorldModelDataset(cfg["data"]["out_path"],start,end,ck["stats"]); dev=device_from(cfg)
    model=MarketWorldModel(ds.daily.shape[-1],ds.intraday.shape[-1],len(ck["symbols"]),ds.returns.shape[-1],cfg["model"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
    loader=DataLoader(ds,batch_size=cfg["train"]["batch_size"],shuffle=False)
    ys=[]; mus=[]; sigmas=[]; regimes=[]
    with torch.no_grad():
        for b in loader:
            out=model(b["daily"].to(dev),b["intraday"].to(dev),b["symbol_id"].to(dev))
            ys.append(b["future_returns"].numpy()); mus.append(out["return_mu"].cpu().numpy()); sigmas.append(np.exp(out["return_log_sigma"].cpu().numpy())); regimes.append(out["regime"].cpu().numpy())
    y=np.concatenate(ys); mu=np.concatenate(mus); sd=np.concatenate(sigmas); r=np.concatenate(regimes)
    mae=np.mean(np.abs(y-mu),axis=0); dir_acc=np.mean(np.sign(y)==np.sign(mu),axis=0)
    coverage=np.mean((y>=mu-1.645*sd)&(y<=mu+1.645*sd),axis=0)
    uniq,counts=np.unique(r,return_counts=True)
    result={"n":len(y),"mae":mae.tolist(),"directional_accuracy":dir_acc.tolist(),"p90_interval_coverage":coverage.tolist(),"regimes_used":int(len(uniq)),"top_regimes":sorted(zip(uniq.tolist(),counts.tolist()),key=lambda x:x[1],reverse=True)[:10]}
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
