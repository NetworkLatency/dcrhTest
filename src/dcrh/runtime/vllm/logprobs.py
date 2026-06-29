from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_items(logprobs: Any) -> list[Any]:
    if logprobs is None:
        return []
    if isinstance(logprobs, Mapping):
        return list(logprobs.values())
    if isinstance(logprobs, Sequence) and not isinstance(logprobs, (str, bytes)):
        return list(logprobs)
    if isinstance(logprobs, Iterable):
        return list(logprobs)
    return []


def raw_topk_values(logprobs: Any, k: int) -> list[float]:
    """Extract rank-ordered raw logit/logprob values from a vLLM logprobs object.

    vLLM returns logprob records through a container whose exact Python type may
    vary between entrypoints. In raw_logits mode the numeric field is still named
    logprob by the public output object, so this helper treats it as an opaque
    score and normalizes it downstream.
    """
    if k < 1:
        raise ValueError("k must be positive")
    entries: list[tuple[int | None, float]] = []
    for item in _as_items(logprobs):
        rank = _field(item, "rank")
        value = _field(item, "logprob")
        if value is None:
            value = _field(item, "value")
        if value is None:
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
        parsed_rank = None if rank is None else int(rank)
        entries.append((parsed_rank, float(value)))
    if not entries:
        raise ValueError("No numeric entries were found in the vLLM logprobs object")
    if all(rank is not None for rank, _ in entries):
        entries.sort(key=lambda pair: int(pair[0]))
    else:
        entries.sort(key=lambda pair: pair[1], reverse=True)
    return [value for _, value in entries[:k]]


def top2_margin_from_raw_values(values: Sequence[float]) -> float:
    """Return p_top1 - p_top2 after softmax over the supplied raw scores."""
    if len(values) < 2:
        raise ValueError("At least two raw values are required")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    probs = probs / probs.sum()
    top2 = np.sort(probs)[-2:]
    return float(top2[-1] - top2[-2])


def top2_margin_from_vllm_logprobs(logprobs: Any, k: int) -> float:
    return top2_margin_from_raw_values(raw_topk_values(logprobs, k))
