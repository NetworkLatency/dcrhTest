from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REGION_NAMES = ("A", "O", "P", "C")


@dataclass(slots=True)
class TpmMargin:
    """TPM margin: local action clarity from raw next-token logits."""

    top1_token_id: int
    top2_token_id: int
    p_top1: float
    p_top2: float
    tpm_margin: float
    top1_token_text: str | None = None
    top2_token_text: str | None = None


@dataclass(slots=True)
class RegionSpans:
    """Non-overlapping token spans for MDRV A/O/P/C route regions."""

    A: tuple[int, int]
    O: tuple[int, int]
    P: tuple[int, int]
    C: tuple[int, int]
    prefix_token_len: int
    input_ids: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return {name: getattr(self, name) for name in REGION_NAMES}


def compute_tpm_from_logits(raw_logits: torch.Tensor, tokenizer: Any | None = None) -> TpmMargin:
    """Compute M_i = p_top1 - p_top2 from full-vocabulary raw LM logits."""
    import torch

    if raw_logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(raw_logits.shape)}")
    if raw_logits.numel() < 2:
        raise ValueError("TPM requires at least two logits")
    probabilities = torch.softmax(raw_logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probabilities, k=2, dim=-1)
    top1_id = int(top_ids[0].item())
    top2_id = int(top_ids[1].item())
    p_top1 = float(top_probs[0].item())
    p_top2 = float(top_probs[1].item())
    top1_text = None
    top2_text = None
    if tokenizer is not None:
        top1_text = tokenizer.decode(
            [top1_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        top2_text = tokenizer.decode(
            [top2_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    return TpmMargin(
        top1_token_id=top1_id,
        top2_token_id=top2_id,
        p_top1=p_top1,
        p_top2=p_top2,
        tpm_margin=p_top1 - p_top2,
        top1_token_text=top1_text,
        top2_token_text=top2_text,
    )


def token_is_content_token(token_id: int, token_text: str, tokenizer: Any) -> bool:
    """Return true for non-whitespace, non-special, non-pure-punctuation tokens."""
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if int(token_id) in special_ids:
        return False
    stripped = token_text.strip()
    if not stripped:
        return False
    return not all(unicodedata.category(char).startswith("P") for char in stripped)


def find_first_content_token(
    token_ids: Sequence[int],
    tokenizer: Any,
) -> tuple[int, int, str] | None:
    """Find the first content token in a continuation after a boundary."""
    for offset, token_id in enumerate(token_ids):
        text = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if token_is_content_token(int(token_id), text, tokenizer):
            return offset, int(token_id), text
    return None


def _encode_piece(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    return list(encoded.get("input_ids", []))


def build_region_spans_from_token_ids(
    prompt_ids: Sequence[int],
    chunk_ids: Sequence[Sequence[int]],
    input_ids: Sequence[int] | None = None,
) -> RegionSpans:
    """Build A/O/P/C spans from exact prompt and generated chunk token ids."""
    prompt_len = len(prompt_ids)
    start = prompt_len
    chunk_spans: list[tuple[int, int]] = []
    for ids in chunk_ids:
        end = start + len(ids)
        chunk_spans.append((start, end))
        start = end
    prefix_token_len = start
    if input_ids is None:
        input_ids = [*prompt_ids, *(token for ids in chunk_ids for token in ids)]

    if not chunk_spans:
        empty = (prefix_token_len, prefix_token_len)
        return RegionSpans(
            A=(0, prompt_len),
            O=empty,
            P=empty,
            C=empty,
            prefix_token_len=prefix_token_len,
            input_ids=tuple(int(x) for x in input_ids),
        )

    c_span = chunk_spans[-1]
    p_span = chunk_spans[-2] if len(chunk_spans) >= 2 else (c_span[0], c_span[0])
    if len(chunk_spans) >= 3:
        o_span = (chunk_spans[0][0], chunk_spans[-3][1])
    else:
        o_span = (p_span[0], p_span[0])

    return RegionSpans(
        A=(0, prompt_len),
        O=o_span,
        P=p_span,
        C=c_span,
        prefix_token_len=prefix_token_len,
        input_ids=tuple(int(x) for x in input_ids),
    )


def build_region_spans(prompt_text: str, chunks: Sequence[str], tokenizer: Any) -> RegionSpans:
    """Build A/O/P/C spans from separately tokenized prompt and step chunks.

    The caller should pass chunks exactly as the router treats them; for MDRV's
    first version the trailing "\\n\\n" delimiter belongs to the current chunk C.
    """
    prompt_ids = _encode_piece(tokenizer, prompt_text)
    chunk_ids = [_encode_piece(tokenizer, chunk) for chunk in chunks]
    prompt_len = len(prompt_ids)
    chunk_lengths = [len(ids) for ids in chunk_ids]
    input_ids = tuple(token_id for piece in (prompt_ids, *chunk_ids) for token_id in piece)

    start = prompt_len
    chunk_spans: list[tuple[int, int]] = []
    for length in chunk_lengths:
        end = start + length
        chunk_spans.append((start, end))
        start = end
    prefix_token_len = start

    if not chunk_spans:
        empty = (prefix_token_len, prefix_token_len)
        return RegionSpans(
            A=(0, prompt_len),
            O=empty,
            P=empty,
            C=empty,
            prefix_token_len=prefix_token_len,
            input_ids=input_ids,
        )

    c_span = chunk_spans[-1]
    if len(chunk_spans) >= 2:
        p_span = chunk_spans[-2]
    else:
        p_span = (c_span[0], c_span[0])

    if len(chunk_spans) >= 3:
        o_span = (chunk_spans[0][0], chunk_spans[-3][1])
    else:
        o_span = (p_span[0], p_span[0])

    return RegionSpans(
        A=(0, prompt_len),
        O=o_span,
        P=p_span,
        C=c_span,
        prefix_token_len=prefix_token_len,
        input_ids=input_ids,
    )


def _boundary_attention_row(attention: torch.Tensor) -> torch.Tensor:
    weights = attention.float()
    if weights.ndim == 4:
        # [batch, heads, query, key]
        weights = weights[0, :, -1, :]
    elif weights.ndim == 3:
        # [heads, query, key] or [batch, query, key].
        weights = weights[:, -1, :]
    elif weights.ndim == 2:
        # [heads, key]
        pass
    elif weights.ndim == 1:
        return weights
    else:
        raise ValueError(f"Unsupported attention shape: {tuple(attention.shape)}")
    return weights.mean(dim=0)


def compute_attention_route(
    attention: torch.Tensor,
    region_spans: RegionSpans | Mapping[str, tuple[int, int]],
    exclude_positions: Sequence[int] = (),
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compute length-normalized A/O/P/C route density from one boundary query row."""
    import torch

    spans = region_spans.as_dict() if isinstance(region_spans, RegionSpans) else dict(region_spans)
    row = _boundary_attention_row(attention)
    key_len = int(row.numel())
    excluded = {int(x) for x in exclude_positions}
    densities: dict[str, float] = {}
    for name in REGION_NAMES:
        start, end = spans[name]
        start = max(0, min(int(start), key_len))
        end = max(start, min(int(end), key_len))
        positions = [pos for pos in range(start, end) if pos not in excluded]
        if not positions:
            densities[name] = 0.0
        else:
            index = torch.tensor(positions, dtype=torch.long, device=row.device)
            densities[name] = float(row.index_select(0, index).mean().item())

    denom = sum(max(0.0, value) for value in densities.values())
    if denom <= epsilon:
        return {name: 0.0 for name in REGION_NAMES}
    return {name: max(0.0, densities[name]) / denom for name in REGION_NAMES}


def _normalized_distribution(values: Sequence[float], epsilon: float) -> list[float]:
    clipped = [max(0.0, float(x)) for x in values]
    total = sum(clipped)
    if total <= epsilon:
        return [0.0 for _ in clipped]
    return [x / total for x in clipped]


def compute_jsd_route_velocity(
    route: Sequence[float] | Mapping[str, float],
    previous_route: Sequence[float] | Mapping[str, float],
    epsilon: float = 1e-12,
) -> float:
    """Compute V_i = JSD(r_i, r_{i-1}) / log(2) using natural logs."""
    if isinstance(route, Mapping):
        p_values = [route[name] for name in REGION_NAMES]
    else:
        p_values = list(route)
    if isinstance(previous_route, Mapping):
        q_values = [previous_route[name] for name in REGION_NAMES]
    else:
        q_values = list(previous_route)
    if len(p_values) != len(q_values):
        raise ValueError("Route distributions must have the same length")

    p = _normalized_distribution(p_values, epsilon)
    q = _normalized_distribution(q_values, epsilon)
    if sum(p) <= epsilon and sum(q) <= epsilon:
        return 0.0
    m = [(x + y) * 0.5 for x, y in zip(p, q)]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        total = 0.0
        for a, b in zip(left, right):
            if a <= epsilon:
                continue
            total += a * math.log(a / max(b, epsilon))
        return total

    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(jsd / math.log(2.0))
