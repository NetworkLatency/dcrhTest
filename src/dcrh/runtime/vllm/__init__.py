"""vLLM worker protocol and output parsing helpers."""

from .logprobs import (
    normalized_entropy_from_raw_values,
    normalized_entropy_from_vllm_logprobs,
    raw_topk_values,
)
from .protocol import (
    ProbeSpan,
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
    "ProbeSpan",
    "VllmSamplingRequest",
    "WorkerCommand",
    "WorkerEndEvent",
    "WorkerEnvelope",
    "WorkerErrorEvent",
    "WorkerGenerateRequest",
    "WorkerTokenEvent",
    "command_from_json_line",
    "event_line",
    "normalized_entropy_from_raw_values",
    "normalized_entropy_from_vllm_logprobs",
    "raw_topk_values",
    "to_json_line",
]
