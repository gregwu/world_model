from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def gaussian_nll(y, mu, log_sigma):
    inv_var = torch.exp(-2.0 * log_sigma)
    return (0.5 * (y - mu).pow(2) * inv_var + log_sigma + 0.5 * math.log(2 * math.pi)).mean()


def code_usage_entropy(regime_ids: torch.Tensor, num_regimes: int):
    counts = torch.bincount(regime_ids, minlength=num_regimes).float()
    p = counts / counts.sum().clamp_min(1.0)
    entropy = -(p * torch.log(p.clamp_min(1e-8))).sum()
    return entropy / math.log(num_regimes)
