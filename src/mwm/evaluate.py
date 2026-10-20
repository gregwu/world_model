from __future__ import annotations
import argparse, json
import numpy as np
import torch
from .data import make_loader, open_dataset
from .train import device_from, build_model, to_device, encode_next_regime, autocast_for, run_model


def check_test_access(cfg, split, final_test=False):
    """Refuse the test split when split.test_frozen is set, unless this is the deliberate final OOS test."""
    if split=="test" and cfg["split"].get("test_frozen") and not final_test:
        raise SystemExit("the test split is frozen (split.test_frozen in the config); tune on --split val, "
                         "and pass --final-test only for the one final out-of-sample evaluation")


def load_checkpoint(path, split, max_samples=None, final_test=False):
    ck=torch.load(path,map_location="cpu",weights_only=False); cfg=ck["cfg"]; s=cfg["split"]
    check_test_access(cfg, split, final_test)
    ranges={"train":(s["train_start"],s["train_end"]),"val":(s["val_start"],s["val_end"]),"test":(s["test_start"],s["test_end"])}
    ds=open_dataset(cfg["data"]["out_path"],*ranges[split],ck["stats"],max_samples=max_samples,seed=cfg["seed"]); dev=device_from(cfg)
    if ck.get("model_type")=="pure_5m":  # v0.2 IntradayWorldModel (scripts/train_5m.py)
        from .intraday_model import IntradayWorldModel
        model=IntradayWorldModel(ds.intra_dim,len(ck["symbols"]),ds.returns.shape[-1],cfg["model"])
    else:
        model=build_model(ds,len(ck["symbols"]),cfg["model"])
    model=model.to(dev); model.load_state_dict(ck["model"]); model.eval()
    return ck, cfg, ds, dev, model


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",default="artifacts/best.pt"); ap.add_argument("--split",choices=["val","test"],default="test")
    ap.add_argument("--max-samples",type=int,default=None,help="evaluate a fixed random subset of pairs (large packs)")
    ap.add_argument("--final-test",action="store_true",help="allow the test split when the config sets split.test_frozen"); args=ap.parse_args()
    ck,cfg,ds,dev,model=load_checkpoint(args.checkpoint,args.split,args.max_samples,args.final_test)
    loader=make_loader(ds,cfg["train"]["batch_size"],False)
    ys=[]; mus=[]; sigmas=[]; regimes=[]; nexts=[]; next_preds=[]
    with torch.no_grad():
        for b in loader:
            b=to_device(b,dev)
            with autocast_for(dev,cfg["train"].get("amp")):
                out=run_model(model,b)
            ys.append(b["future_returns"].cpu().numpy()); mus.append(out["return_mu"].float().cpu().numpy()); sigmas.append(np.exp(out["return_log_sigma"].float().cpu().numpy()))
            regimes.append(out["regime"].cpu().numpy()); next_preds.append(out["next_regime_logits"].float().argmax(-1).cpu().numpy())
            nexts.append(encode_next_regime(model,b,cfg["train"].get("amp")).cpu().numpy())
    y=np.concatenate(ys); mu=np.concatenate(mus); sd=np.concatenate(sigmas); r=np.concatenate(regimes)
    nxt=np.concatenate(nexts); nxt_pred=np.concatenate(next_preds)
    mae=np.mean(np.abs(y-mu),axis=0); dir_acc=np.mean(np.sign(y)==np.sign(mu),axis=0)
    # Unchanged prices (y == 0) can never match a nonzero predicted sign; on 5m bars they are common, so also report
    # accuracy over moves only, and how often the price did not move.
    moved=y!=0; zero_share=1-moved.mean(axis=0)
    dir_acc_moves=np.array([np.mean(np.sign(y[moved[:,h],h])==np.sign(mu[moved[:,h],h])) for h in range(y.shape[1])])
    up_share_moves=np.array([np.mean(y[moved[:,h],h]>0) for h in range(y.shape[1])])
    coverage=np.mean((y>=mu-1.645*sd)&(y<=mu+1.645*sd),axis=0)
    uniq,counts=np.unique(r,return_counts=True)
    result={"n":len(y),"mae":mae.tolist(),"directional_accuracy":dir_acc.tolist(),
            "zero_return_share":zero_share.tolist(),"directional_accuracy_moves":dir_acc_moves.tolist(),
            "up_share_moves":up_share_moves.tolist(),"predicted_up_share":np.mean(mu>0,axis=0).tolist(),"p90_interval_coverage":coverage.tolist(),
            "regimes_used":int(len(uniq)),"top_regimes":sorted(zip(uniq.tolist(),counts.tolist()),key=lambda x:x[1],reverse=True)[:10],
            "next_regime_accuracy":float(np.mean(nxt_pred==nxt)),"persistence_accuracy":float(np.mean(r==nxt))}
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
