from __future__ import annotations

import math

import torch


def normalized_topk_entropy(raw_logits: torch.Tensor, k: int) -> float:
    """Entropy over the raw top-k logits, before any sampling-time transformation."""
    if raw_logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(raw_logits.shape)}")
    k = min(int(k), int(raw_logits.numel()))
    if k < 2:
        raise ValueError("Entropy top-k must be at least 2")
    top_values = torch.topk(raw_logits.float(), k=k, dim=-1).values
    probabilities = torch.softmax(top_values, dim=-1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()
    return float((entropy / math.log(k)).item())
