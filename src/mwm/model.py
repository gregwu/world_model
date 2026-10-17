from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SequenceEncoder(nn.Module):
    def __init__(self, in_dim, d_model, nhead, num_layers, ff, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.pos(self.proj(x))
        x = self.encoder(x)
        return self.norm(x[:, -1])


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)
        self.beta = beta

    def forward(self, z):
        # squared Euclidean distance to codebook
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1).unsqueeze(0)
        )
        ids = dist.argmin(dim=1)
        z_q = self.embedding(ids)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss
        z_st = z + (z_q - z).detach()
        return z_st, ids, vq_loss, dist


class MarketWorldModel(nn.Module):
    def __init__(self, daily_dim: int, intra_dim: int, num_symbols: int, num_horizons: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.daily_enc = SequenceEncoder(daily_dim, d, cfg["nhead"], cfg["num_layers"], cfg["dim_feedforward"], cfg["dropout"])
        self.intra_enc = SequenceEncoder(intra_dim, d, cfg["nhead"], cfg["num_layers"], cfg["dim_feedforward"], cfg["dropout"])
        self.stock_emb = nn.Embedding(num_symbols, cfg["stock_embedding_dim"])
        fusion_in = d * 2 + cfg["stock_embedding_dim"]
        self.to_latent = nn.Sequential(
            nn.Linear(fusion_in, cfg["dynamics_hidden"]), nn.GELU(), nn.LayerNorm(cfg["dynamics_hidden"]),
            nn.Linear(cfg["dynamics_hidden"], cfg["latent_dim"]),
        )
        self.vq = VectorQuantizer(cfg["num_regimes"], cfg["latent_dim"])
        dyn_in = cfg["latent_dim"] + cfg["stock_embedding_dim"]
        self.next_regime = nn.Sequential(
            nn.Linear(dyn_in, cfg["dynamics_hidden"]), nn.GELU(),
            nn.Linear(cfg["dynamics_hidden"], cfg["num_regimes"]),
        )
        ret_in = cfg["latent_dim"] + cfg["stock_embedding_dim"]
        self.return_head = nn.Sequential(
            nn.Linear(ret_in, cfg["return_hidden"]), nn.GELU(),
            nn.Linear(cfg["return_hidden"], num_horizons * 2),
        )
        self.num_horizons = num_horizons

    def encode(self, daily, intraday, symbol_id):
        sd = self.daily_enc(daily)
        si = self.intra_enc(intraday)
        se = self.stock_emb(symbol_id)
        z_e = self.to_latent(torch.cat([sd, si, se], dim=-1))
        z_q, regime_id, vq_loss, dist = self.vq(z_e)
        return z_q, regime_id, vq_loss, dist, se

    def forward(self, daily, intraday, symbol_id):
        z, regime, vq_loss, dist, se = self.encode(daily, intraday, symbol_id)
        x = torch.cat([z, se], dim=-1)
        next_logits = self.next_regime(x)
        ret = self.return_head(x).view(-1, self.num_horizons, 2)
        mu = ret[..., 0]
        log_sigma = ret[..., 1].clamp(-6.0, 3.0)
        return {
            "z": z,
            "regime": regime,
            "vq_loss": vq_loss,
            "regime_dist": dist,
            "next_regime_logits": next_logits,
            "return_mu": mu,
            "return_log_sigma": log_sigma,
        }
