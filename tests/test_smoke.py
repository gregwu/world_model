import torch
from mwm.model import MarketWorldModel

def test_forward():
    cfg=dict(d_model=32,nhead=4,num_layers=1,dim_feedforward=64,dropout=0.1,latent_dim=16,num_regimes=8,stock_embedding_dim=8,dynamics_hidden=32,return_hidden=32)
    m=MarketWorldModel(12,12,10,4,cfg)
    out=m(torch.randn(3,40,12),torch.randn(3,64,12),torch.tensor([0,1,2]))
    assert out["return_mu"].shape==(3,4)
    assert out["next_regime_logits"].shape==(3,8)
