"""vLLM worker protocol and output parsing helpers."""

from .logprobs import (
    raw_topk_values,
    top2_margin_from_raw_values,
    top2_margin_from_vllm_logprobs,
)
from .protocol import (
    VllmSamplingRequest,
    WorkerCommand,
    WorkerEndEvent,
    WorkerEnvelope,
    WorkerErrorEvent,
    WorkerGenerateRequest,
    WorkerTokenEvent,
    command_from_json_line,
    event_line,
    to_json_line,
)

__all__ = [
    "VllmSamplingRequest",
    "WorkerCommand",
    "WorkerEndEvent",
    "WorkerEnvelope",
    "WorkerErrorEvent",
    "WorkerGenerateRequest",
    "WorkerTokenEvent",
    "command_from_json_line",
    "event_line",
    "raw_topk_values",
    "top2_margin_from_raw_values",
    "top2_margin_from_vllm_logprobs",
    "to_json_line",
]
