from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

from .model import VectorQuantizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class WindowEncoder(nn.Module):
    """Encode one causal 5m horizon. The last token summarizes the window."""
    def __init__(self, in_dim: int, d_model: int, nhead: int, layers: int, ff: int, dropout: float, max_len: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.enc(self.pos(self.proj(x)))[:, -1])


class IntradayWorldModel(nn.Module):
    """Pure 5-minute hierarchical market model.

    One 455-bar input is viewed at three causal scales:
      short   = last 12 bars  (~1h)
      session = last 78 bars  (~1 regular session)
      multi   = full history  (~5-6 sessions)

    The three views are fused into a continuous latent, quantized into a learned market/stock regime,
    and used for probabilistic multi-horizon returns, market-neutral residual returns, and next-state dynamics.
    No daily features are consumed.
    """
    def __init__(self, intra_dim: int, num_symbols: int, num_horizons: int, cfg: dict):
        super().__init__()
        d = cfg.get("d_model", 128)
        nhead = cfg.get("nhead", 4); layers = cfg.get("num_layers", 2)
        ff = cfg.get("dim_feedforward", 256); dropout = cfg.get("dropout", 0.1)
        self.short_bars = cfg.get("short_bars", 12)
        self.session_bars = cfg.get("session_bars", 78)
        self.multi_bars = cfg.get("multi_bars", 455)
        self.short_enc = WindowEncoder(intra_dim, d, nhead, layers, ff, dropout, self.short_bars)
        self.session_enc = WindowEncoder(intra_dim, d, nhead, layers, ff, dropout, self.session_bars)
        self.multi_enc = WindowEncoder(intra_dim, d, nhead, layers, ff, dropout, self.multi_bars)
        emb_dim = cfg.get("stock_embedding_dim", 32)
        self.stock_emb = nn.Embedding(num_symbols, emb_dim)
        latent_dim = cfg.get("latent_dim", 64)
        hidden = cfg.get("dynamics_hidden", 128)
        self.fuse = nn.Sequential(nn.Linear(3*d + emb_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, latent_dim))
        self.vq = VectorQuantizer(cfg.get("num_regimes", 64), latent_dim,
                                  temperature=cfg.get("vq_temperature", 0.1),
                                  dead_threshold=cfg.get("vq_dead_threshold", 1e-3))
        head_in = latent_dim + emb_dim
        self.return_head = nn.Sequential(nn.Linear(head_in, cfg.get("return_hidden", 128)), nn.GELU(), nn.Linear(cfg.get("return_hidden", 128), num_horizons*2))
        self.relative_head = nn.Sequential(nn.Linear(head_in, cfg.get("return_hidden", 128)), nn.GELU(), nn.Linear(cfg.get("return_hidden", 128), num_horizons))
        self.next_latent = nn.Sequential(nn.Linear(head_in, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))
        nn.init.zeros_(self.next_latent[-1].weight); nn.init.zeros_(self.next_latent[-1].bias)
        self.dynamics_log_scale = nn.Parameter(torch.tensor(math.log(1.0/cfg.get("vq_temperature", 0.1))))
        self.num_horizons = num_horizons

    def encode(self, intraday, symbol_id):
        if intraday.size(1) < self.multi_bars:
            raise ValueError(f"need at least {self.multi_bars} 5m bars, got {intraday.size(1)}")
        x = intraday[:, -self.multi_bars:]
        hs = self.short_enc(x[:, -self.short_bars:])
        hd = self.session_enc(x[:, -self.session_bars:])
        hm = self.multi_enc(x)
        se = self.stock_emb(symbol_id)
        z_e = self.fuse(torch.cat([hs, hd, hm, se], -1))
        z, regime, vq_loss, dist, probs = self.vq(z_e)
        return z, regime, vq_loss, dist, probs, z_e, se

    def forward(self, intraday, symbol_id):
        z, regime, vq_loss, dist, probs, z_e, se = self.encode(intraday, symbol_id)
        x = torch.cat([z, se], -1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            raw = self.return_head(x.float()).view(-1, self.num_horizons, 2)
            relative_mu = self.relative_head(x.float())
        zd = F.normalize(z_e, dim=-1).detach()
        z_next = F.normalize(zd + self.next_latent(torch.cat([zd, se.detach()], -1)), dim=-1)
        next_logits = -(2.0 - 2.0*z_next @ self.vq.codebook().detach().t()) * self.dynamics_log_scale.exp()
        return {
            "z": z, "z_e": z_e, "regime": regime, "vq_loss": vq_loss,
            "regime_dist": dist, "regime_probs": probs,
            "next_regime_logits": next_logits,
            "return_mu": raw[..., 0], "return_log_sigma": raw[..., 1].clamp(-6.0, 3.0),
            "relative_mu": relative_mu,
        }
