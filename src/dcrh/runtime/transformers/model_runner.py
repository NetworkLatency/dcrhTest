from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention_backend import (
    BACKEND_NAME,
    RouteCollector,
    RouteSummary,
    register_probe_attention_backend,
    select_mdrv_route_layer,
)
from ...config import CostConfig, GenerationConfig, ModelConfig, PromptConfig, SignalConfig
from ...core.router_signals import (
    RegionSpans,
    TpmMargin,
    compute_tpm_from_logits,
)
from ...core.costs import CostLedger, synchronize_if_needed
from ...prompt import PromptBuilder, PromptEncoding
from .sampling import SamplingParameters, TokenSampler


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _normalize_eos_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(x) for x in value}


@dataclass(slots=True)
class SessionPrefillInfo:
    purpose: str
    tokens: int
    seconds: float
    base_prompt_tokens: int
    shared_text_tokens: int
    control_tokens: int


@dataclass(slots=True)
class TokenObservation:
    token_id: int
    text_piece: str
    decode_seconds: float
    sequence_length_after_token: int


@dataclass(slots=True)
class StepResult:
    observation: TokenObservation
    ended: bool
    end_reason: str | None


@dataclass(slots=True)
class BoundaryState:
    next_token_logits: torch.Tensor
    input_ids: torch.LongTensor
    prefix_token_len: int
    tpm_margin: TpmMargin
    route_summary: RouteSummary | None
    attention_available: bool
    attention_unavailable_reason: str | None


class LocalQwen3Runner:
    """One local Qwen3 model with MDRV boundary replay support."""

    def __init__(
        self,
        role: str,
        model_cfg: ModelConfig,
        prompt_cfg: PromptConfig,
        generation_cfg: GenerationConfig,
        signal_cfg: SignalConfig,
        cost_cfg: CostConfig,
    ) -> None:
        register_probe_attention_backend()
        self.role = role
        self.model_cfg = model_cfg
        self.generation_cfg = generation_cfg
        self.signal_cfg = signal_cfg
        self.cost_cfg = cost_cfg
        self.thinking_end_marker = (
            prompt_cfg.thinking_end_marker if prompt_cfg.enable_thinking else ""
        )
        self.device = torch.device(model_cfg.device)
        self.dtype = resolve_dtype(model_cfg.dtype)

        local_path = Path(model_cfg.path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {local_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(local_path),
            local_files_only=True,
            trust_remote_code=model_cfg.trust_remote_code,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(local_path),
            local_files_only=True,
            trust_remote_code=model_cfg.trust_remote_code,
            torch_dtype=self.dtype,
            attn_implementation=BACKEND_NAME,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        if getattr(self.model.config, "model_type", None) != "qwen3":
            raise ValueError(
                f"The core implementation currently supports Qwen3 dense CausalLMs; got "
                f"model_type={getattr(self.model.config, 'model_type', None)!r}."
            )
        self.model.set_attn_implementation(BACKEND_NAME)

        self.route_layer = select_mdrv_route_layer(self.model.config)

        self.prompt_builder = PromptBuilder(
            tokenizer=self.tokenizer,
            system_prompt=prompt_cfg.system_prompt,
            enable_thinking=prompt_cfg.enable_thinking,
            sink_prefix_tokens=prompt_cfg.sink_prefix_tokens,
        )
        self.eos_ids = _normalize_eos_ids(getattr(self.model.generation_config, "eos_token_id", None))
        self.eos_ids |= _normalize_eos_ids(getattr(self.tokenizer, "eos_token_id", None))

    @property
    def name(self) -> str:
        return self.model_cfg.name or Path(self.model_cfg.path).name

    def model_metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "path": str(Path(self.model_cfg.path).resolve()),
            "backend": self.model_cfg.backend,
            "probe_mode": self.model_cfg.probe_mode,
            "model_type": self.model.config.model_type,
            "num_hidden_layers": int(self.model.config.num_hidden_layers),
            "num_attention_heads": int(self.model.config.num_attention_heads),
            "num_key_value_heads": int(self.model.config.num_key_value_heads),
            "head_dim": int(getattr(self.model.config, "head_dim")),
            "mdrv_route_layer": self.route_layer,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "transformers_config": json.loads(self.model.config.to_json_string()),
        }

    def estimate_kv_bytes(self, sequence_length: int) -> int:
        bytes_per_value = torch.tensor([], dtype=self.dtype).element_size()
        return int(
            2
            * int(self.model.config.num_hidden_layers)
            * int(self.model.config.num_key_value_heads)
            * int(getattr(self.model.config, "head_dim"))
            * bytes_per_value
            * int(sequence_length)
        )

    def estimate_prefill_qk_elements_upper_bound(self, sequence_length: int) -> int:
        """Dense-attention upper bound; this is a work proxy rather than a FLOP count."""
        return int(
            int(self.model.config.num_hidden_layers)
            * int(self.model.config.num_attention_heads)
            * int(sequence_length)
            * int(sequence_length)
        )

    def estimate_decode_qk_elements_upper_bound(self, sequence_length: int) -> int:
        return int(
            int(self.model.config.num_hidden_layers)
            * int(self.model.config.num_attention_heads)
            * int(sequence_length)
        )

    def build_prompt(
        self,
        question: str,
        shared_text: str = "",
        control_text: str = "",
    ) -> PromptEncoding:
        return self.prompt_builder.build(
            question=question,
            shared_text=shared_text,
            control_text=control_text,
            device=self.device,
        )

    def forward_boundary_state(
        self,
        prefix_text: str,
        region_spans: RegionSpans | None = None,
        collect_attention_route: bool = False,
    ) -> BoundaryState:
        """Replay a pre-action prefix and expose MDRV TPM plus optional route."""
        if region_spans is not None and region_spans.input_ids:
            input_ids = torch.tensor(
                [list(region_spans.input_ids)],
                dtype=torch.long,
                device=self.device,
            )
        else:
            encoded = self.tokenizer(
                prefix_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
        if input_ids.numel() == 0:
            raise ValueError("Boundary prefix tokenization produced no tokens")

        route_collector: RouteCollector | None = None
        unavailable_reason: str | None = None
        if collect_attention_route:
            if self.route_layer is None:
                unavailable_reason = "no_full_attention_layer"
            elif region_spans is None:
                unavailable_reason = "region_spans_missing"
            else:
                route_collector = RouteCollector(
                    route_layer=self.route_layer,
                    region_spans=region_spans.as_dict(),
                    exclude_positions=(int(input_ids.shape[-1]) - 1,),
                    head_chunk_size=self.signal_cfg.attention_head_chunk_size,
                    epsilon=self.signal_cfg.route_epsilon,
                    profile_with_cuda_sync=self.signal_cfg.profile_route_with_cuda_sync,
                )
                route_collector.begin_boundary()

        synchronize_if_needed(
            self.device, self.cost_cfg.synchronize_cuda_for_timing
        )
        try:
            with torch.inference_mode():
                outputs = self._forward(
                    input_ids=input_ids,
                    use_cache=False,
                    dcrh_route_collector=route_collector,
                )
        except Exception as exc:
            if route_collector is None:
                raise
            unavailable_reason = f"attention_unavailable:{type(exc).__name__}:{exc}"
            route_collector = None
            with torch.inference_mode():
                outputs = self._forward(
                    input_ids=input_ids,
                    use_cache=False,
                )
        synchronize_if_needed(
            self.device, self.cost_cfg.synchronize_cuda_for_timing
        )

        route_summary = None
        if route_collector is not None:
            try:
                route_summary = route_collector.end_boundary()
            except Exception as exc:
                unavailable_reason = f"attention_unavailable:{type(exc).__name__}:{exc}"

        logits = outputs.logits[0, -1].detach()
        return BoundaryState(
            next_token_logits=logits,
            input_ids=input_ids.detach(),
            prefix_token_len=int(input_ids.shape[-1]),
            tpm_margin=compute_tpm_from_logits(logits, tokenizer=self.tokenizer),
            route_summary=route_summary,
            attention_available=route_summary is not None,
            attention_unavailable_reason=unavailable_reason,
        )

    def create_session(
        self,
        question: str,
        shared_text: str,
        control_text: str,
        purpose: str,
        ledger: CostLedger,
        seed_key: str | None = None,
    ) -> "GenerationSession":
        encoding = self.build_prompt(question, shared_text, control_text)
        material = "\x1f".join(
            [
                str(self.generation_cfg.seed),
                self.role,
                purpose,
                question,
                seed_key or "",
            ]
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        seed %= 2**63 - 1
        return GenerationSession(
            runner=self,
            prompt=encoding,
            purpose=purpose,
            ledger=ledger,
            seed=seed,
        )

    def unload(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class GenerationSession:
    """A live session owns one transient KV cache. It never creates cache snapshots."""

    def __init__(
        self,
        runner: LocalQwen3Runner,
        prompt: PromptEncoding,
        purpose: str,
        ledger: CostLedger,
        seed: int,
    ) -> None:
        self.runner = runner
        self.prompt = prompt
        self.purpose = purpose
        self.ledger = ledger
        self.generated_ids: list[int] = []
        self.decode_seconds_by_token: list[float] = []
        self._past_key_values = None
        self._next_logits: torch.Tensor | None = None
        self._closed = False
        self._prefill_info: SessionPrefillInfo | None = None
        self._thinking_end_seen = False
        self._marker_buffer = ""
        params = SamplingParameters(
            do_sample=runner.generation_cfg.do_sample,
            temperature=runner.generation_cfg.temperature,
            top_p=runner.generation_cfg.top_p,
            top_k=runner.generation_cfg.top_k,
            presence_penalty=runner.generation_cfg.presence_penalty,
            repetition_penalty=runner.generation_cfg.repetition_penalty,
        )
        self.sampler = TokenSampler(
            vocab_size=int(runner.model.config.vocab_size),
            device=runner.device,
            parameters=params,
            seed=seed,
        )
        # Presence/repetition penalties apply to the full text seen by this session.
        self.sampler.seen[self.prompt.input_ids[0].unique()] = True
        self._prefill()

    @property
    def prefill_info(self) -> SessionPrefillInfo:
        if self._prefill_info is None:
            raise RuntimeError("Session has not been prefilled")
        return self._prefill_info

    @property
    def total_sequence_length(self) -> int:
        return self.prompt.total_tokens + len(self.generated_ids)

    def _forward(self, **kwargs):
        try:
            return self.runner.model(logits_to_keep=1, **kwargs)
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            return self.runner.model(**kwargs)

    def _prefill(self) -> None:
        synchronize_if_needed(self.runner.device, self.runner.cost_cfg.synchronize_cuda_for_timing)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self._forward(
                input_ids=self.prompt.input_ids,
                use_cache=True,
            )
        synchronize_if_needed(self.runner.device, self.runner.cost_cfg.synchronize_cuda_for_timing)
        seconds = time.perf_counter() - started
        self._past_key_values = outputs.past_key_values
        self._next_logits = outputs.logits[0, -1].detach()
        kv_bytes = self.runner.estimate_kv_bytes(self.prompt.total_tokens)
        self.ledger.record_prefill(
            role=self.runner.role,
            purpose=self.purpose,
            tokens=self.prompt.total_tokens,
            base_prompt_tokens=self.prompt.base_prompt_tokens,
            shared_text_tokens=self.prompt.shared_text_tokens,
            control_tokens=self.prompt.control_tokens,
            seconds=seconds,
            estimated_live_kv_bytes=kv_bytes,
            attention_qk_elements_upper_bound=(
                self.runner.estimate_prefill_qk_elements_upper_bound(
                    self.prompt.total_tokens
                )
            ),
        )
        self._prefill_info = SessionPrefillInfo(
            purpose=self.purpose,
            tokens=self.prompt.total_tokens,
            seconds=seconds,
            base_prompt_tokens=self.prompt.base_prompt_tokens,
            shared_text_tokens=self.prompt.shared_text_tokens,
            control_tokens=self.prompt.control_tokens,
        )

    def step(self) -> StepResult:
        if self._closed:
            raise RuntimeError("Cannot step a closed generation session")
        if self._next_logits is None:
            raise RuntimeError("No next-token logits are available")
        if self.total_sequence_length >= int(self.runner.model.config.max_position_embeddings):
            dummy = TokenObservation(
                token_id=-1,
                text_piece="",
                decode_seconds=0.0,
                sequence_length_after_token=self.total_sequence_length,
            )
            return StepResult(dummy, ended=True, end_reason="max_context")

        token_id = self.sampler.sample(self._next_logits)
        self.generated_ids.append(token_id)
        self.decode_seconds_by_token.append(0.0)
        self.ledger.record_sampled_token(self.runner.role, self.purpose)
        piece = self.runner.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        sequence_length = self.total_sequence_length

        if token_id in self.runner.eos_ids:
            observation = TokenObservation(
                token_id=token_id,
                text_piece=piece,
                decode_seconds=0.0,
                sequence_length_after_token=sequence_length,
            )
            return StepResult(observation, ended=True, end_reason="eos")

        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.runner.device)
        synchronize_if_needed(self.runner.device, self.runner.cost_cfg.synchronize_cuda_for_timing)
        started = time.perf_counter()
        forward_kwargs = dict(
            input_ids=token_tensor,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
        with torch.inference_mode():
            outputs = self._forward(**forward_kwargs)
        synchronize_if_needed(self.runner.device, self.runner.cost_cfg.synchronize_cuda_for_timing)
        seconds = time.perf_counter() - started
        self._past_key_values = outputs.past_key_values
        self._next_logits = outputs.logits[0, -1].detach()
        self.decode_seconds_by_token[-1] = seconds
        self.ledger.record_decode_forward(
            role=self.runner.role,
            purpose=self.purpose,
            sequence_length=sequence_length,
            seconds=seconds,
            estimated_live_kv_bytes=self.runner.estimate_kv_bytes(sequence_length),
            attention_qk_elements_upper_bound=(
                self.runner.estimate_decode_qk_elements_upper_bound(sequence_length)
            ),
        )
        observation = TokenObservation(
            token_id=token_id,
            text_piece=piece,
            decode_seconds=seconds,
            sequence_length_after_token=sequence_length,
        )
        thinking_end = False
        marker = self.runner.thinking_end_marker
        if marker and not self._thinking_end_seen:
            keep = max(64, len(marker) * 2)
            self._marker_buffer = (self._marker_buffer + piece)[-keep:]
            if marker in self._marker_buffer:
                self._thinking_end_seen = True
                thinking_end = True
        return StepResult(
            observation,
            ended=thinking_end,
            end_reason="thinking_end" if thinking_end else None,
        )

    def generated_text(self, up_to_token: int | None = None) -> str:
        ids = self.generated_ids if up_to_token is None else self.generated_ids[:up_to_token]
        return self.runner.tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generated_cost_slice(
        self, start_token: int = 0, end_token: int | None = None
    ) -> float:
        end = len(self.generated_ids) if end_token is None else int(end_token)
        start = max(0, int(start_token))
        end = max(start, min(end, len(self.generated_ids)))
        return float(sum(self.decode_seconds_by_token[start:end]))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._past_key_values = None
        self._next_logits = None
        gc.collect()
        if self.runner.device.type == "cuda":
            torch.cuda.empty_cache()
