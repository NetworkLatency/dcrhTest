from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Iterable

import torch


BACKEND_NAME = "dcrh_sdpa_probe"
_REGISTERED = False


@dataclass(slots=True)
class ProbeSummary:
    grounding: float
    observed_layers: int
    observed_heads: int
    probe_seconds: float


@dataclass(slots=True)
class GroundingCollector:
    """Collects only question-attention mass; no attention matrix is retained."""

    selected_layers: frozenset[int]
    question_start: int
    question_end: int
    sink_positions: tuple[int, ...]
    head_chunk_size: int = 4
    epsilon: float = 1e-6
    profile_with_cuda_sync: bool = False
    _sum: float = field(init=False, default=0.0)
    _count: int = field(init=False, default=0)
    _layers_seen: set[int] = field(init=False, default_factory=set)
    _probe_seconds: float = field(init=False, default=0.0)
    active: bool = field(init=False, default=False)

    def begin_token(self) -> None:
        self._sum = 0.0
        self._count = 0
        self._layers_seen.clear()
        self._probe_seconds = 0.0
        self.active = True

    @staticmethod
    def _sync_if_requested(tensor: torch.Tensor, enabled: bool) -> None:
        if enabled and tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

    def observe(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
    ) -> None:
        if not self.active or module.layer_idx not in self.selected_layers:
            return
        if query.shape[0] != 1 or query.shape[2] != 1:
            raise RuntimeError(
                "Grounding probing currently requires batch_size=1 and one-token decode forwards."
            )
        key_length = int(key.shape[-2])
        if self.question_end > key_length:
            raise RuntimeError(
                f"Question span [{self.question_start}, {self.question_end}) exceeds key length {key_length}. "
                "This usually indicates a sliding-window cache on the selected layer."
            )

        self._sync_if_requested(query, self.profile_with_cuda_sync)
        started = time.perf_counter()

        num_q_heads = int(query.shape[1])
        groups = int(getattr(module, "num_key_value_groups", 1))
        valid_sink = [p for p in self.sink_positions if 0 <= p < key_length]
        sink_tensor = (
            torch.tensor(valid_sink, dtype=torch.long, device=query.device)
            if valid_sink
            else None
        )
        valid_key_count = key_length - len(valid_sink)
        question_count = self.question_end - self.question_start
        if valid_key_count <= 0 or question_count <= 0:
            raise RuntimeError("Invalid question or non-sink key count for grounding computation")

        base_rate = min(
            1.0 - self.epsilon,
            max(self.epsilon, question_count / valid_key_count),
        )
        base_logit = math.log(base_rate / (1.0 - base_rate))

        for head_start in range(0, num_q_heads, self.head_chunk_size):
            head_end = min(num_q_heads, head_start + self.head_chunk_size)
            q_chunk = query[:, head_start:head_end]
            kv_indices = (
                torch.arange(head_start, head_end, device=query.device, dtype=torch.long)
                // groups
            )
            k_chunk = key.index_select(1, kv_indices)

            scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scaling
            scores = scores.float()
            if attention_mask is not None:
                mask = attention_mask[:, :, :, :key_length]
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                else:
                    scores = scores + mask.to(dtype=scores.dtype)

            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            question_mass = weights[
                ..., self.question_start : self.question_end
            ].sum(dim=-1)
            if sink_tensor is None:
                valid_mass = torch.ones_like(question_mass)
            else:
                sink_mass = weights.index_select(-1, sink_tensor).sum(dim=-1)
                valid_mass = (1.0 - sink_mass).clamp_min(self.epsilon)

            relative_mass = (question_mass / valid_mass).clamp(
                min=self.epsilon, max=1.0 - self.epsilon
            )
            grounding = torch.logit(relative_mass) - base_logit
            self._sum += float(grounding.sum().item())
            self._count += grounding.numel()

            del scores, weights, question_mass, relative_mass, grounding

        self._layers_seen.add(int(module.layer_idx))
        self._sync_if_requested(query, self.profile_with_cuda_sync)
        self._probe_seconds += time.perf_counter() - started

    def end_token(self) -> ProbeSummary:
        self.active = False
        missing = self.selected_layers.difference(self._layers_seen)
        if missing:
            raise RuntimeError(
                f"Grounding probe did not observe selected layers: {sorted(missing)}"
            )
        if self._count == 0:
            raise RuntimeError("Grounding probe collected no attention heads")
        return ProbeSummary(
            grounding=self._sum / self._count,
            observed_layers=len(self._layers_seen),
            observed_heads=self._count,
            probe_seconds=self._probe_seconds,
        )


def central_layers(num_hidden_layers: int, width: int = 4) -> list[int]:
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be positive")
    width = max(1, min(width, num_hidden_layers))
    start = max(0, (num_hidden_layers - width) // 2)
    return list(range(start, start + width))


def validate_selected_layers(layers: Iterable[int], num_hidden_layers: int) -> list[int]:
    unique = sorted(set(int(x) for x in layers))
    if not unique:
        raise ValueError("At least one attention layer must be selected")
    for layer in unique:
        if layer < 0 or layer >= num_hidden_layers:
            raise ValueError(
                f"Attention layer {layer} is outside [0, {num_hidden_layers})"
            )
    return unique


def register_probe_attention_backend() -> None:
    """Register an SDPA backend that computes G online during one-token decoding."""
    global _REGISTERED
    if _REGISTERED:
        return

    import os
    import transformers
    from packaging.version import Version

    supported = Version("4.57.6")
    installed = Version(transformers.__version__)
    if installed != supported and os.environ.get("DCRH_ALLOW_UNTESTED_TRANSFORMERS") != "1":
        raise RuntimeError(
            f"DCRH pins transformers=={supported}; found {installed}. "
            "Install the pinned local wheel, or set DCRH_ALLOW_UNTESTED_TRANSFORMERS=1 at your own risk."
        )

    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask

    def dcrh_sdpa_probe(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        collector: GroundingCollector | None = kwargs.pop(
            "dcrh_grounding_collector", None
        )
        output, _ = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
        if collector is not None and collector.active:
            collector.observe(
                module=module,
                query=query,
                key=key,
                attention_mask=attention_mask,
                scaling=float(scaling if scaling is not None else module.scaling),
            )
        return output, None

    AttentionInterface.register(BACKEND_NAME, dcrh_sdpa_probe)
    AttentionMaskInterface.register(BACKEND_NAME, sdpa_mask)
    _REGISTERED = True
