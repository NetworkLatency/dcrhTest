from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _torch():
    import torch

    return torch


@dataclass(slots=True)
class PrefillRecord:
    role: str
    purpose: str
    tokens: int
    base_prompt_tokens: int
    shared_text_tokens: int
    control_tokens: int
    seconds: float
    estimated_live_kv_bytes: int
    attention_qk_elements_upper_bound: int


@dataclass(slots=True)
class TransitionRecord:
    name: str
    payload: dict[str, Any]
    wall_time: float


class CostLedger:
    """Tracks actual recomputation and discarded work; no KV snapshots are assumed."""

    def __init__(self, record_attention_work_proxy: bool = True) -> None:
        self.record_attention_work_proxy = record_attention_work_proxy
        self.prefills: list[PrefillRecord] = []
        self.transitions: list[TransitionRecord] = []
        self.counters: defaultdict[str, float] = defaultdict(float)
        self.per_role: dict[str, defaultdict[str, float]] = {
            "slm": defaultdict(float),
            "llm": defaultdict(float),
        }
        self.per_purpose: dict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.gpu_memory: dict[str, dict[str, int]] = {}
        self._started = time.perf_counter()

    def _add(self, role: str, purpose: str, key: str, value: float) -> None:
        self.counters[key] += value
        self.per_role[role][key] += value
        self.per_purpose[purpose][key] += value

    def record_prefill(
        self,
        role: str,
        purpose: str,
        tokens: int,
        base_prompt_tokens: int,
        shared_text_tokens: int,
        control_tokens: int,
        seconds: float,
        estimated_live_kv_bytes: int,
        attention_qk_elements_upper_bound: int,
    ) -> None:
        record = PrefillRecord(
            role=role,
            purpose=purpose,
            tokens=int(tokens),
            base_prompt_tokens=int(base_prompt_tokens),
            shared_text_tokens=int(shared_text_tokens),
            control_tokens=int(control_tokens),
            seconds=float(seconds),
            estimated_live_kv_bytes=int(estimated_live_kv_bytes),
            attention_qk_elements_upper_bound=int(attention_qk_elements_upper_bound),
        )
        self.prefills.append(record)
        self._add(role, purpose, "prefill_calls", 1)
        self._add(role, purpose, "prefill_tokens", tokens)
        self._add(role, purpose, "prefill_seconds", seconds)
        self._add(role, purpose, "control_prefill_tokens", control_tokens)
        if self.record_attention_work_proxy:
            self._add(
                role,
                purpose,
                "prefill_attention_qk_elements_upper_bound",
                attention_qk_elements_upper_bound,
            )
        self.per_role[role]["max_estimated_live_kv_bytes"] = max(
            self.per_role[role]["max_estimated_live_kv_bytes"],
            float(estimated_live_kv_bytes),
        )

    def record_sampled_token(self, role: str, purpose: str) -> None:
        self._add(role, purpose, "generated_tokens", 1)

    def record_decode_forward(
        self,
        role: str,
        purpose: str,
        sequence_length: int,
        seconds: float,
        probe_seconds: float,
        estimated_live_kv_bytes: int,
        attention_qk_elements_upper_bound: int,
        probe_qk_elements: int,
        estimated_probe_score_buffer_bytes: int,
    ) -> None:
        self._add(role, purpose, "decode_forward_calls", 1)
        self._add(role, purpose, "decode_seconds", seconds)
        self._add(role, purpose, "probe_seconds", probe_seconds)
        self.per_role[role]["max_estimated_live_kv_bytes"] = max(
            self.per_role[role]["max_estimated_live_kv_bytes"],
            float(estimated_live_kv_bytes),
        )
        if self.record_attention_work_proxy:
            self._add(
                role,
                purpose,
                "decode_attention_qk_elements_upper_bound",
                attention_qk_elements_upper_bound,
            )
            self._add(
                role,
                purpose,
                "probe_attention_qk_elements",
                probe_qk_elements,
            )
            self.per_role[role]["max_estimated_probe_score_buffer_bytes"] = max(
                self.per_role[role]["max_estimated_probe_score_buffer_bytes"],
                float(estimated_probe_score_buffer_bytes),
            )

    def mark_discarded_generation(
        self,
        role: str,
        reason: str,
        tokens: int,
        characters: int = 0,
        decode_seconds: float = 0.0,
        probe_seconds: float = 0.0,
    ) -> None:
        self.counters["discarded_generated_tokens"] += int(tokens)
        self.counters[f"discarded_{reason}_tokens"] += int(tokens)
        self.counters[f"discarded_{reason}_characters"] += int(characters)
        self.counters[f"discarded_{reason}_decode_seconds"] += float(decode_seconds)
        self.counters[f"discarded_{reason}_probe_seconds"] += float(probe_seconds)
        self.per_role[role]["discarded_generated_tokens"] += int(tokens)
        self.per_role[role][f"discarded_{reason}_tokens"] += int(tokens)
        self.per_role[role][f"discarded_{reason}_decode_seconds"] += float(
            decode_seconds
        )
        self.per_role[role][f"discarded_{reason}_probe_seconds"] += float(
            probe_seconds
        )

    def mark_wasted_prefill(self, role: str, purpose: str, tokens: int, seconds: float) -> None:
        self.counters["wasted_prefill_tokens"] += int(tokens)
        self.counters["wasted_prefill_seconds"] += float(seconds)
        self.per_role[role]["wasted_prefill_tokens"] += int(tokens)
        self.per_role[role]["wasted_prefill_seconds"] += float(seconds)
        self.per_purpose[purpose]["wasted_prefill_tokens"] += int(tokens)
        self.per_purpose[purpose]["wasted_prefill_seconds"] += float(seconds)

    def record_rollback_tradeoff(
        self,
        slm_discarded_tokens: int,
        llm_actual_prefill_tokens: int,
        llm_detection_point_prefill_tokens: int,
        llm_actual_prefill_qk_elements_upper_bound: int,
        llm_detection_point_qk_elements_upper_bound: int,
        llm_actual_estimated_kv_bytes: int,
        llm_detection_point_estimated_kv_bytes: int,
    ) -> None:
        saved = int(llm_detection_point_prefill_tokens) - int(llm_actual_prefill_tokens)
        self.counters["rollback_slm_discarded_tokens"] += int(slm_discarded_tokens)
        self.counters["llm_upgrade_actual_prefill_tokens"] += int(llm_actual_prefill_tokens)
        self.counters["llm_detection_point_counterfactual_prefill_tokens"] += int(
            llm_detection_point_prefill_tokens
        )
        self.counters["rollback_llm_prefill_tokens_saved"] += saved
        self.counters["llm_upgrade_actual_prefill_qk_elements_upper_bound"] += int(
            llm_actual_prefill_qk_elements_upper_bound
        )
        self.counters[
            "llm_detection_point_counterfactual_prefill_qk_elements_upper_bound"
        ] += int(llm_detection_point_qk_elements_upper_bound)
        self.counters["rollback_llm_prefill_qk_elements_saved_upper_bound"] += int(
            llm_detection_point_qk_elements_upper_bound
            - llm_actual_prefill_qk_elements_upper_bound
        )
        self.counters["llm_upgrade_actual_estimated_live_kv_bytes"] = max(
            self.counters["llm_upgrade_actual_estimated_live_kv_bytes"],
            int(llm_actual_estimated_kv_bytes),
        )
        self.counters[
            "llm_detection_point_counterfactual_estimated_live_kv_bytes"
        ] = max(
            self.counters[
                "llm_detection_point_counterfactual_estimated_live_kv_bytes"
            ],
            int(llm_detection_point_estimated_kv_bytes),
        )

    def transition(self, name: str, **payload: Any) -> None:
        self.transitions.append(
            TransitionRecord(name=name, payload=payload, wall_time=time.perf_counter())
        )

    def capture_gpu_memory(self, devices: list[str]) -> None:
        torch = _torch()
        for device in sorted(set(devices)):
            if not device.startswith("cuda") or not torch.cuda.is_available():
                continue
            d = torch.device(device)
            self.gpu_memory[device] = {
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(d)),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(d)),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(d)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(d)),
            }

    def summary(self) -> dict[str, Any]:
        total_generated = int(self.counters.get("generated_tokens", 0))
        discarded = int(self.counters.get("discarded_generated_tokens", 0))
        result = {
            "wall_seconds": time.perf_counter() - self._started,
            "counters": dict(self.counters),
            "generated_committed_tokens_estimate": max(0, total_generated - discarded),
            "per_role": {k: dict(v) for k, v in self.per_role.items()},
            "per_purpose": {k: dict(v) for k, v in self.per_purpose.items()},
            "prefills": [asdict(x) for x in self.prefills],
            "transitions": [asdict(x) for x in self.transitions],
            "gpu_memory": self.gpu_memory,
            "kv_policy": "live_session_only_no_snapshots",
        }
        return result


def synchronize_if_needed(device: torch.device, enabled: bool) -> None:
    torch = _torch()
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(devices: list[str]) -> None:
    torch = _torch()
    if not torch.cuda.is_available():
        return
    for device in sorted(set(devices)):
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
