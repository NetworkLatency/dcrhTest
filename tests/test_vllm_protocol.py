from dataclasses import dataclass

from dcrh.runtime.vllm import (
    VllmSamplingRequest,
    WorkerCommand,
    WorkerGenerateRequest,
    command_from_json_line,
    normalized_entropy_from_raw_values,
    raw_topk_values,
    to_json_line,
)


@dataclass
class FakeLogprob:
    logprob: float
    rank: int


def test_vllm_logprobs_use_rank_not_mapping_order():
    values = {
        4: FakeLogprob(logprob=0.1, rank=3),
        2: FakeLogprob(logprob=3.0, rank=1),
        3: FakeLogprob(logprob=1.0, rank=2),
    }
    assert raw_topk_values(values, 2) == [3.0, 1.0]


def test_entropy_from_raw_values_is_normalized():
    entropy = normalized_entropy_from_raw_values([1.0, 1.0, 1.0, 1.0], k=4)
    assert abs(entropy - 1.0) < 1e-12


def test_worker_command_round_trip():
    request = WorkerGenerateRequest(
        request_id="req-1",
        role="slm",
        question="q",
        shared_text="",
        control_text="",
        purpose="unit",
        sampling=VllmSamplingRequest(
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            seed=123,
            max_tokens=16,
        ),
        probe_mode="vllm_attention",
        enable_grounding=True,
    )
    line = to_json_line(WorkerCommand(kind="generate", request=request))
    parsed = command_from_json_line(line)
    assert parsed.kind == "generate"
    assert parsed.request is not None
    assert parsed.request.request_id == "req-1"
    assert parsed.request.sampling.logprobs_mode == "raw_logits"
