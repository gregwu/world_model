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
    """Cosine VQ (latents and codes on the unit sphere) with EMA usage tracking and dead-code restarts."""

    def __init__(self, num_codes: int, dim: int, beta: float = 0.25, temperature: float = 0.1,
                 usage_decay: float = 0.99, dead_threshold: float = 1e-3):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, dim)
        nn.init.normal_(self.embedding.weight)
        self.beta = beta
        self.temperature = temperature
        self.usage_decay = usage_decay
        self.dead_threshold = dead_threshold
        self.register_buffer("usage", torch.full((num_codes,), 1.0 / num_codes))

    def codebook(self):
        return F.normalize(self.embedding.weight, dim=-1)

    def forward(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):  # fp32: bf16 is too coarse for near-tied codes
            return self._forward(z.float())

    def _forward(self, z):
        z = F.normalize(z, dim=-1)
        codes = self.codebook()
        # squared Euclidean distance on the unit sphere = 2 - 2*cos
        dist = 2.0 - 2.0 * z @ codes.t()
        ids = dist.argmin(dim=1)
        z_q = codes[ids]
        codebook_loss = F.mse_loss(z_q, z.detach())
        commit_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss
        z_st = z + (z_q - z).detach()
        # Soft assignments give the code-usage entropy a gradient; hard argmin ids do not.
        probs = torch.softmax(-dist / self.temperature, dim=-1)
        if self.training:
            with torch.no_grad():
                frac = torch.bincount(ids, minlength=self.usage.numel()).float() / ids.numel()
                self.usage.mul_(self.usage_decay).add_(frac, alpha=1 - self.usage_decay)
        return z_st, ids, vq_loss, dist, probs

    @torch.no_grad()
    def restart_dead_codes(self, z_e) -> torch.Tensor:
        """Re-seed codes whose EMA usage fell below threshold with random encoder outputs. Call after opt.step().
        Mask-based so it never forces a host sync; returns the number of restarted codes as a device tensor."""
        dead = self.usage < self.dead_threshold
        pick = torch.randint(0, z_e.size(0), (dead.numel(),), device=z_e.device)
        fresh = F.normalize(z_e[pick], dim=-1) + 0.01 * torch.randn_like(z_e[pick])
        self.embedding.weight.copy_(torch.where(dead[:, None], fresh, self.embedding.weight))
        self.usage.copy_(torch.where(dead, torch.full_like(self.usage, 1.0 / self.usage.numel()), self.usage))
        return dead.sum()


class MarketWorldModel(nn.Module):
    def __init__(self, daily_dim: int, intra_dim: int, num_symbols: int, num_horizons: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        if daily_dim <= 0 and intra_dim <= 0:
            raise ValueError("need daily and/or intraday inputs")
        # daily_dim == 0 builds a pure-intraday model; intra_dim == 0 a daily-only one
        self.daily_enc = (SequenceEncoder(daily_dim, d, cfg["nhead"], cfg["num_layers"], cfg["dim_feedforward"], cfg["dropout"])
                          if daily_dim > 0 else None)
        self.intra_enc = (SequenceEncoder(intra_dim, d, cfg["nhead"], cfg["num_layers"], cfg["dim_feedforward"], cfg["dropout"])
                          if intra_dim > 0 else None)
        self.stock_emb = nn.Embedding(num_symbols, cfg["stock_embedding_dim"])
        fusion_in = d * ((self.daily_enc is not None) + (self.intra_enc is not None)) + cfg["stock_embedding_dim"]
        self.to_latent = nn.Sequential(
            nn.Linear(fusion_in, cfg["dynamics_hidden"]), nn.GELU(), nn.LayerNorm(cfg["dynamics_hidden"]),
            nn.Linear(cfg["dynamics_hidden"], cfg["latent_dim"]),
        )
        self.vq = VectorQuantizer(cfg["num_regimes"], cfg["latent_dim"],
                                  temperature=cfg.get("vq_temperature", 0.1),
                                  dead_threshold=cfg.get("vq_dead_threshold", 1e-3))
        dyn_in = cfg["latent_dim"] + cfg["stock_embedding_dim"]
        self.next_regime = nn.Sequential(
            nn.Linear(dyn_in, cfg["dynamics_hidden"]), nn.GELU(),
            nn.Linear(cfg["dynamics_hidden"], cfg["num_regimes"]),
        )
        # "latent" dynamics: predict the next latent as z + residual(z, stock) and score it against the current
        # codebook. Zero-initialized residual => starts exactly at persistence, and logits stay tied to the codebook
        # geometry, so code re-seeding/drift doesn't scramble what the head has learned (unlike id-space logits).
        self.dynamics_head = cfg.get("dynamics_head", "logits")
        if self.dynamics_head == "latent":
            self.next_latent = nn.Sequential(
                nn.Linear(dyn_in, cfg["dynamics_hidden"]), nn.GELU(),
                nn.Linear(cfg["dynamics_hidden"], cfg["latent_dim"]),
            )
            nn.init.zeros_(self.next_latent[-1].weight); nn.init.zeros_(self.next_latent[-1].bias)
            # Learnable inverse temperature: lets the head express uncertainty by flattening its distribution
            # instead of moving z_next toward an average of next codes (which flips the argmax away from "stay").
            self.dynamics_log_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.get("vq_temperature", 0.1))))
        # "gated" dynamics: p(next) = g * p(stay in current regime) + (1 - g) * softmax(move logits), with a
        # per-sample learned gate initialized to ~0.95. Starts at persistence; id-space drift only affects "move".
        if self.dynamics_head == "gated":
            self.stay_gate = nn.Sequential(
                nn.Linear(dyn_in, cfg["dynamics_hidden"]), nn.GELU(),
                nn.Linear(cfg["dynamics_hidden"], 1),
            )
            nn.init.zeros_(self.stay_gate[-1].weight); nn.init.constant_(self.stay_gate[-1].bias, 3.0)
        ret_in = cfg["latent_dim"] + cfg["stock_embedding_dim"]
        self.return_head = nn.Sequential(
            nn.Linear(ret_in, cfg["return_hidden"]), nn.GELU(),
            nn.Linear(cfg["return_hidden"], num_horizons * 2),
        )
        self.num_horizons = num_horizons

    def encode(self, daily, intraday, symbol_id):
        parts = []
        if self.daily_enc is not None:
            parts.append(self.daily_enc(daily))
        if self.intra_enc is not None:
            parts.append(self.intra_enc(intraday))
        se = self.stock_emb(symbol_id)
        z_e = self.to_latent(torch.cat([*parts, se], dim=-1))
        z_q, regime_id, vq_loss, dist, probs = self.vq(z_e)
        return z_q, regime_id, vq_loss, dist, se, z_e, probs

    def forward(self, daily, intraday, symbol_id):
        z, regime, vq_loss, dist, se, z_e, probs = self.encode(daily, intraday, symbol_id)
        x = torch.cat([z, se], dim=-1)
        if self.dynamics_head == "latent":
            # Detached inputs: the dynamics loss trains only this head. Otherwise it also moves the encoder, which
            # shifts the current regime (and so the target) while the head chases it.
            # Anchor on the continuous latent: its nearest code is by definition the assigned regime, so a zero
            # residual is exact persistence. (z_q can sit next to near-duplicate re-seeded codes and miss it.)
            zd = F.normalize(z_e, dim=-1).detach()
            z_next = F.normalize(zd + self.next_latent(torch.cat([zd, se.detach()], dim=-1)), dim=-1)
            next_logits = -(2.0 - 2.0 * z_next @ self.vq.codebook().detach().t()) * self.dynamics_log_scale.exp()
        elif self.dynamics_head == "gated":
            # Detached inputs: the dynamics loss trains only the head, not the encoder or codebook.
            xd = torch.cat([F.normalize(z_e, dim=-1).detach(), se.detach()], dim=-1)
            # Exact persistence (lightly smoothed one-hot). softmax(-dist/T) is too diffuse here: neighbouring codes
            # are often only a few hundredths apart in cosine distance, so it spreads "stay" over several regimes.
            k = dist.size(-1)
            p_stay = F.one_hot(regime, k).float() * (1 - 1e-2) + 1e-2 / k
            p_move = torch.softmax(self.next_regime(xd).float(), dim=-1)
            g = torch.sigmoid(self.stay_gate(xd).float())
            next_logits = torch.log(g * p_stay + (1 - g) * p_move + 1e-8)  # log-probs; cross_entropy accepts them as logits
        else:
            next_logits = self.next_regime(x)
        # fp32 even under bf16 autocast: predicted mean returns are ~1e-4, and bf16 rounded up to 17% of them to
        # exactly 0, which broke sign-based metrics (a zero sign never matches the realized sign).
        with torch.autocast(device_type=x.device.type, enabled=False):
            ret = self.return_head(x.float()).view(-1, self.num_horizons, 2)
        mu = ret[..., 0]
        log_sigma = ret[..., 1].clamp(-6.0, 3.0)
        return {
            "z": z,
            "regime": regime,
            "vq_loss": vq_loss,
            "regime_dist": dist,
            "regime_probs": probs,
            "z_e": z_e,
            "next_regime_logits": next_logits,
            "return_mu": mu,
            "return_log_sigma": log_sigma,
        }
