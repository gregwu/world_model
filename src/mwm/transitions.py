from __future__ import annotations
import argparse, json
import numpy as np
import torch
from .data import make_loader
from .evaluate import load_checkpoint
from .train import to_device, encode_next_regime, autocast_for, run_model


def infer_regimes(model, ds, device, batch_size, amp=None, with_next=False):
    """Regime of each pair's current sample (aligned with ds.pair_rows), and optionally of its next sample."""
    loader=make_loader(ds,batch_size,False)
    cur=[]; nxt=[]
    model.eval()
    with torch.no_grad():
        for b in loader:
            b=to_device(b,device)
            with autocast_for(device,amp):
                cur.append(run_model(model,b)["regime"].cpu().numpy())
            if with_next: nxt.append(encode_next_regime(model,b,amp).cpu().numpy())
    cur=np.concatenate(cur)
    return (cur,np.concatenate(nxt)) if with_next else cur


def transition_matrix(current, nxt, k):
    """Counts of (regime_t, regime_{t+1}) over adjacent-bar pairs of the same symbol."""
    mat=np.zeros((k,k),dtype=np.int64)
    np.add.at(mat,(current,nxt),1)
    probs=mat/mat.sum(axis=1,keepdims=True).clip(min=1)
    return mat,probs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",default="artifacts/best.pt"); ap.add_argument("--split",choices=["train","val","test"],default="train"); ap.add_argument("--out",default="artifacts/transitions.npz")
    ap.add_argument("--max-samples",type=int,default=None,help="count a fixed random subset of pairs (large packs)")
    ap.add_argument("--final-test",action="store_true",help="allow the test split when the config sets split.test_frozen"); args=ap.parse_args()
    ck,cfg,ds,dev,model=load_checkpoint(args.checkpoint,args.split,args.max_samples,args.final_test)
    current,nxt=infer_regimes(model,ds,dev,cfg["train"]["batch_size"],cfg["train"].get("amp"),with_next=True)
    mat,probs=transition_matrix(current,nxt,cfg["model"]["num_regimes"])
    np.savez_compressed(args.out,counts=mat,probs=probs)
    print(json.dumps({"out":args.out,"total_transitions":int(mat.sum()),"nonzero_edges":int((mat>0).sum()),
                      "self_transition_share":float(np.trace(mat)/max(mat.sum(),1))},indent=2))

if __name__=="__main__": main()
