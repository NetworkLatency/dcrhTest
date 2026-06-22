import math

import torch

from dcrh.runtime.transformers.entropy import normalized_topk_entropy


def test_uniform_topk_entropy_is_one():
    logits = torch.zeros(20)
    value = normalized_topk_entropy(logits, 20)
    assert abs(value - 1.0) < 1e-6


def test_peaked_entropy_is_small():
    logits = torch.tensor([20.0, 0.0, 0.0, 0.0])
    assert normalized_topk_entropy(logits, 4) < 1e-4
