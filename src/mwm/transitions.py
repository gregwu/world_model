from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import WorldModelDataset
from .model import MarketWorldModel
from .train import device_from


def infer_regimes(model, ds, device, batch_size):
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False)
    out=[]
    model.eval()
    with torch.no_grad():
        for b in loader:
            q=model(b["daily"].to(device),b["intraday"].to(device),b["symbol_id"].to(device))
            out.append(q["regime"].cpu().numpy())
    return np.concatenate(out)


def transition_matrix(regimes, symbol_ids, timestamps_ns, k):
    order=np.lexsort((timestamps_ns,symbol_ids)); r=regimes[order]; s=symbol_ids[order]
    mat=np.zeros((k,k),dtype=np.int64)
    for i in range(len(r)-1):
        if s[i]==s[i+1]: mat[r[i],r[i+1]]+=1
    probs=mat/mat.sum(axis=1,keepdims=True).clip(min=1)
    return mat,probs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",default="artifacts/best.pt"); ap.add_argument("--split",choices=["train","val","test"],default="train"); ap.add_argument("--out",default="artifacts/transitions.npz"); args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False); cfg=ck["cfg"]; s=cfg["split"]
    ranges={"train":(s["train_start"],s["train_end"]),"val":(s["val_start"],s["val_end"]),"test":(s["test_start"],s["test_end"])}
    ds=WorldModelDataset(cfg["data"]["out_path"],*ranges[args.split],ck["stats"]); dev=device_from(cfg)
    model=MarketWorldModel(ds.daily.shape[-1],ds.intraday.shape[-1],len(ck["symbols"]),ds.returns.shape[-1],cfg["model"]).to(dev); model.load_state_dict(ck["model"])
    regimes=infer_regimes(model,ds,dev,cfg["train"]["batch_size"])
    current_rows=ds.pair_rows
    mat,probs=transition_matrix(regimes,ds.symbol_id[current_rows],ds.timestamps_ns[current_rows],cfg["model"]["num_regimes"])
    np.savez_compressed(args.out,counts=mat,probs=probs)
    print(json.dumps({"out":args.out,"total_transitions":int(mat.sum()),"nonzero_edges":int((mat>0).sum())},indent=2))

if __name__=="__main__": main()
