from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

import torch

from ...core.router_signals import REGION_NAMES


BACKEND_NAME = "dcrh_sdpa_probe"
_REGISTERED = False


@dataclass(slots=True)
class RouteSummary:
    route_distribution: dict[str, float]
    observed_layer: int
    observed_heads: int
    probe_seconds: float


@dataclass(slots=True)
class RouteCollector:
    """Collects MDRV A/O/P/C route density from one full-attention layer."""

    route_layer: int
    region_spans: Mapping[str, tuple[int, int]]
    exclude_positions: tuple[int, ...] = ()
    head_chunk_size: int = 4
    epsilon: float = 1e-12
    profile_with_cuda_sync: bool = False
    _density_sums: dict[str, float] = field(init=False)
    _heads_seen: int = field(init=False, default=0)
    _layer_seen: bool = field(init=False, default=False)
    _probe_seconds: float = field(init=False, default=0.0)
    active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._density_sums = {name: 0.0 for name in REGION_NAMES}

    def begin_boundary(self) -> None:
        self._density_sums = {name: 0.0 for name in REGION_NAMES}
        self._heads_seen = 0
        self._layer_seen = False
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
        if not self.active or module.layer_idx != self.route_layer:
            return
        if query.shape[0] != 1:
            raise RuntimeError("Route probing currently requires batch_size=1")

        key_length = int(key.shape[-2])
        max_span_end = max(int(end) for _, end in self.region_spans.values())
        if max_span_end > key_length:
            raise RuntimeError(
                f"Route span end {max_span_end} exceeds attention key length {key_length}"
            )

        self._sync_if_requested(query, self.profile_with_cuda_sync)
        started = time.perf_counter()

        num_q_heads = int(query.shape[1])
        groups = int(getattr(module, "num_key_value_groups", 1))
        # MDRV only needs the boundary query row: the last query position.
        query_row = query[:, :, -1:, :]
        mask_row = None
        if attention_mask is not None:
            mask_row = attention_mask[:, :, -1:, :key_length]
        excluded = {
            position
            for position in self.exclude_positions
            if 0 <= int(position) < key_length
        }

        for head_start in range(0, num_q_heads, self.head_chunk_size):
            head_end = min(num_q_heads, head_start + self.head_chunk_size)
            q_chunk = query_row[:, head_start:head_end]
            kv_indices = (
                torch.arange(head_start, head_end, device=query.device, dtype=torch.long)
                // groups
            )
            k_chunk = key.index_select(1, kv_indices)

            scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scaling
            scores = scores.float()
            if mask_row is not None:
                if mask_row.dtype == torch.bool:
                    scores = scores.masked_fill(~mask_row, torch.finfo(scores.dtype).min)
                else:
                    scores = scores + mask_row.to(dtype=scores.dtype)

            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            heads_in_chunk = head_end - head_start
            for name in REGION_NAMES:
                start, end = self.region_spans[name]
                start = int(start)
                end = int(end)
                positions = [pos for pos in range(start, end) if pos not in excluded]
                if not positions:
                    continue
                index = torch.tensor(positions, dtype=torch.long, device=weights.device)
                density = weights.index_select(-1, index).mean(dim=-1)
                self._density_sums[name] += float(density.sum().item())
            self._heads_seen += heads_in_chunk

            del scores, weights

        self._layer_seen = True
        self._sync_if_requested(query, self.profile_with_cuda_sync)
        self._probe_seconds += time.perf_counter() - started

    def end_boundary(self) -> RouteSummary:
        self.active = False
        if not self._layer_seen:
            raise RuntimeError(f"Route probe did not observe layer {self.route_layer}")
        if self._heads_seen <= 0:
            raise RuntimeError("Route probe collected no attention heads")
        densities = {
            name: max(0.0, value / self._heads_seen)
            for name, value in self._density_sums.items()
        }
        denom = sum(densities.values())
        if denom <= self.epsilon:
            route = {name: 0.0 for name in REGION_NAMES}
        else:
            route = {name: densities[name] / denom for name in REGION_NAMES}
        return RouteSummary(
            route_distribution=route,
            observed_layer=self.route_layer,
            observed_heads=self._heads_seen,
            probe_seconds=self._probe_seconds,
        )


def select_mdrv_route_layer(config) -> int | None:
    """Select the last full-attention layer for MDRV route probing.

    This follows the safeguard order requested for Qwen3-family models and never
    intentionally selects a sliding-window layer.
    """
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        return None

    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        candidates = [
            index
            for index, layer_type in enumerate(list(layer_types)[:num_layers])
            if layer_type == "full_attention"
        ]
        return max(candidates) if candidates else None

    use_sliding_window = getattr(config, "use_sliding_window", None)
    sliding_window = getattr(config, "sliding_window", None)
    if use_sliding_window is False or sliding_window is None:
        return num_layers - 1

    max_window_layers = getattr(config, "max_window_layers", None)
    if max_window_layers is None:
        return None
    index = min(num_layers, int(max_window_layers)) - 1
    return index if index >= 0 else None


def register_probe_attention_backend() -> None:
    """Register an SDPA backend that can collect one MDRV route row."""
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
        route_collector: RouteCollector | None = kwargs.pop(
            "dcrh_route_collector", None
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
        if route_collector is not None and route_collector.active:
            route_collector.observe(
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
