from __future__ import annotations
import argparse, json
import numpy as np
import torch
from .data import make_loader
from .evaluate import load_checkpoint
from .train import to_device


def infer_regimes(model, ds, device, batch_size):
    loader=make_loader(ds,batch_size,False)
    out=[]
    model.eval()
    with torch.no_grad():
        for b in loader:
            b=to_device(b,device)
            out.append(model(b["daily"],b.get("intraday"),b["symbol_id"])["regime"].cpu().numpy())
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
    ck,cfg,ds,dev,model=load_checkpoint(args.checkpoint,args.split)
    regimes=infer_regimes(model,ds,dev,cfg["train"]["batch_size"])
    current_rows=ds.pair_rows
    mat,probs=transition_matrix(regimes,ds.symbol_id[current_rows],ds.timestamps_ns[current_rows],cfg["model"]["num_regimes"])
    np.savez_compressed(args.out,counts=mat,probs=probs)
    print(json.dumps({"out":args.out,"total_transitions":int(mat.sum()),"nonzero_edges":int((mat>0).sum()),
                      "self_transition_share":float(np.trace(mat)/max(mat.sum(),1))},indent=2))

if __name__=="__main__": main()
