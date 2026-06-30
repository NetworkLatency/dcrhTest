from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


DTypeName = Literal["bfloat16", "float16", "float32"]
ModelBackend = Literal["transformers", "vllm_worker"]
ProbeMode = Literal["transformers", "transformers_replay", "vllm_attention", "none"]
DataFormat = Literal["auto", "jsonl", "json", "csv", "tsv", "parquet", "hf_disk"]
BenchmarkDataset = Literal["aime24", "aime25", "gpqa", "gpqa_diamond"]
VerifierName = Literal[
    "none",
    "exact",
    "boxed",
    "gsm8k_numeric",
    "math",
    "aime",
    "gpqa",
    "benchmark",
    "auto",
]


@dataclass(slots=True)
class ModelConfig:
    path: str
    device: str
    dtype: DTypeName = "bfloat16"
    trust_remote_code: bool = False
    name: str | None = None
    backend: ModelBackend = "transformers"
    probe_mode: ProbeMode = "transformers"
    worker_endpoint: str | None = None


@dataclass(slots=True)
class ModelsConfig:
    slm: ModelConfig
    llm: ModelConfig


@dataclass(slots=True)
class DataConfig:
    path: str
    format: DataFormat = "auto"
    dataset: BenchmarkDataset | None = None
    split: str | None = None
    id_field: str = "id"
    question_field: str = "question"
    answer_field: str | None = "answer"
    start: int = 0
    limit: int | None = None


@dataclass(slots=True)
class PromptConfig:
    system_prompt: str = "You are a careful reasoning assistant."
    enable_thinking: bool = True
    takeover_cue: str = (
        "\n\nYou are continuing a reasoning process. The previous small model reasoning "
        "is trusted only up to the last provided step. Continue from that point and "
        "solve the problem. Do not assume any later discarded reasoning is correct. "
        "Put the final answer in \\boxed{}.\n\n"
    )
    sink_prefix_tokens: int = 1
    thinking_end_marker: str = "</think>"


@dataclass(slots=True)
class GenerationConfig:
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int = 1234
    max_initial_slm_tokens: int = 8192
    max_llm_finish_tokens: int = 8192
    max_final_answer_tokens: int = 2048


@dataclass(slots=True)
class SignalConfig:
    attention_head_chunk_size: int = 4
    route_epsilon: float = 1e-12
    profile_route_with_cuda_sync: bool = False


@dataclass(slots=True)
class ControllerConfig:
    alarm_threshold: float | None = None


@dataclass(slots=True)
class ProtocolConfig:
    run_mode: Literal["offline_replay", "online_collaboration"] = "offline_replay"
    route_discount: bool = True
    takeover_mode: Literal["rollback", "current"] = "rollback"
    attention_route_mode: Literal["last_layer_single_row", "none"] = "last_layer_single_row"


@dataclass(slots=True)
class CostConfig:
    synchronize_cuda_for_timing: bool = True
    record_cuda_memory: bool = True
    record_attention_work_proxy: bool = True


@dataclass(slots=True)
class EvaluationConfig:
    verifier: VerifierName = "auto"


@dataclass(slots=True)
class OutputConfig:
    directory: str = "outputs/core_run"
    overwrite: bool = False
    save_full_text: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    models: ModelsConfig
    data: DataConfig
    prompt: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(
        self,
        require_tau: bool = False,
        model_roles: tuple[str, ...] = ("slm", "llm"),
    ) -> None:
        valid_data_formats = {"auto", "jsonl", "json", "csv", "tsv", "parquet", "hf_disk"}
        if self.data.format not in valid_data_formats:
            raise ValueError(
                f"data.format must be one of {sorted(valid_data_formats)}"
            )
        valid_datasets = {"aime24", "aime25", "gpqa", "gpqa_diamond"}
        if self.data.dataset is not None and self.data.dataset not in valid_datasets:
            raise ValueError(
                f"data.dataset must be one of {sorted(valid_datasets)} when set"
            )
        valid_roles = {"slm", "llm"}
        requested_roles = tuple(str(role) for role in model_roles)
        unknown_roles = sorted(set(requested_roles).difference(valid_roles))
        if unknown_roles:
            raise ValueError(f"Unknown model role(s): {unknown_roles}")
        for role in requested_roles:
            model = getattr(self.models, role)
            if model.backend not in {"transformers", "vllm_worker"}:
                raise ValueError(f"models.{role}.backend must be 'transformers' or 'vllm_worker'")
            if model.probe_mode not in {
                "transformers",
                "transformers_replay",
                "vllm_attention",
                "none",
            }:
                raise ValueError(
                    f"models.{role}.probe_mode must be one of "
                    "['transformers', 'transformers_replay', 'vllm_attention', 'none']"
                )
            if model.backend == "transformers" and model.probe_mode not in {"transformers", "none"}:
                raise ValueError(
                    f"models.{role}.probe_mode={model.probe_mode!r} requires backend='vllm_worker'"
                )
            if model.backend == "transformers" and not Path(model.path).exists():
                raise FileNotFoundError(f"Local {role} model path does not exist: {model.path}")
            if model.backend == "vllm_worker" and model.worker_endpoint is None:
                raise ValueError(
                    f"models.{role}.worker_endpoint is required when backend='vllm_worker'"
                )
        if not Path(self.data.path).exists():
            raise FileNotFoundError(f"Local dataset path does not exist: {self.data.path}")
        if self.signals.attention_head_chunk_size < 1:
            raise ValueError("signals.attention_head_chunk_size must be >= 1")
        if self.signals.route_epsilon <= 0.0:
            raise ValueError("signals.route_epsilon must be positive")
        if self.prompt.enable_thinking and not self.prompt.thinking_end_marker:
            raise ValueError(
                "prompt.thinking_end_marker must be non-empty when thinking mode is enabled"
            )
        if self.generation.max_final_answer_tokens < 1:
            raise ValueError("generation.max_final_answer_tokens must be >= 1")
        if require_tau and self.controller.alarm_threshold is None:
            raise ValueError("Set controller.alarm_threshold as the MDRV tau.")
        if self.protocol.run_mode not in {"offline_replay", "online_collaboration"}:
            raise ValueError("protocol.run_mode must be 'offline_replay' or 'online_collaboration'")
        if self.protocol.takeover_mode not in {"rollback", "current"}:
            raise ValueError("protocol.takeover_mode must be 'rollback' or 'current'")
        if self.protocol.attention_route_mode not in {"last_layer_single_row", "none"}:
            raise ValueError(
                "protocol.attention_route_mode must be 'last_layer_single_row' or 'none'"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_model(cfg: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**cfg)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    models_raw = raw.get("models") or {}
    models = ModelsConfig(
        slm=_construct_model(models_raw["slm"]),
        llm=_construct_model(models_raw["llm"]),
    )
    cfg = ExperimentConfig(
        models=models,
        data=DataConfig(**raw["data"]),
        prompt=PromptConfig(**raw.get("prompt", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        signals=SignalConfig(**raw.get("signals", {})),
        controller=ControllerConfig(**raw.get("controller", {})),
        protocol=ProtocolConfig(**raw.get("protocol", {})),
        cost=CostConfig(**raw.get("cost", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
    return cfg


def save_resolved_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.as_dict(), f, allow_unicode=True, sort_keys=False)
