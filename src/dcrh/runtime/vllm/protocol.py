from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


ProbeMode = Literal["none", "transformers_replay", "vllm_attention"]
WorkerCommandKind = Literal["generate", "abort", "shutdown"]
WorkerEventKind = Literal["token", "end", "error", "ready"]


@dataclass(slots=True)
class VllmSamplingRequest:
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    repetition_penalty: float
    seed: int
    max_tokens: int
    logprobs: int = 20
    logprobs_mode: str = "raw_logits"


@dataclass(slots=True)
class ProbeSpan:
    question_start: int
    question_end: int
    sink_positions: list[int]


@dataclass(slots=True)
class WorkerGenerateRequest:
    request_id: str
    role: str
    question: str
    shared_text: str
    control_text: str
    purpose: str
    sampling: VllmSamplingRequest
    probe_mode: ProbeMode = "none"
    enable_grounding: bool = False
    seed_key: str | None = None


@dataclass(slots=True)
class WorkerCommand:
    kind: WorkerCommandKind
    request: WorkerGenerateRequest | None = None
    request_id: str | None = None


@dataclass(slots=True)
class WorkerTokenEvent:
    request_id: str
    token_id: int
    text_piece: str
    entropy: float
    grounding: float | None
    token_index: int
    sequence_length_after_token: int
    decode_seconds: float
    probe_seconds: float


@dataclass(slots=True)
class WorkerEndEvent:
    request_id: str
    end_reason: str
    generated_tokens: int
    generated_text: str


@dataclass(slots=True)
class WorkerErrorEvent:
    request_id: str | None
    error_type: str
    error: str


@dataclass(slots=True)
class WorkerEnvelope:
    kind: WorkerEventKind
    payload: dict[str, Any]


def to_json_line(payload: Any) -> str:
    return json.dumps(asdict(payload), ensure_ascii=False, separators=(",", ":")) + "\n"


def command_from_json_line(line: str) -> WorkerCommand:
    raw = json.loads(line)
    request = raw.get("request")
    parsed_request = None
    if request is not None:
        sampling = VllmSamplingRequest(**request["sampling"])
        parsed_request = WorkerGenerateRequest(
            **{k: v for k, v in request.items() if k != "sampling"},
            sampling=sampling,
        )
    return WorkerCommand(
        kind=raw["kind"],
        request=parsed_request,
        request_id=raw.get("request_id"),
    )


def event_line(kind: WorkerEventKind, payload: Any) -> str:
    return to_json_line(WorkerEnvelope(kind=kind, payload=asdict(payload)))
