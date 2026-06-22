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
    selected_layers: list[int] | None = None
    reference_path: str | None = None
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
    repair_cue: str = (
        "\n\n[Expert repair]\n"
        "Re-examine the original problem from the trusted prefix above. "
        "Repair the reasoning and continue with a coherent derivation.\n\n"
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
    max_llm_repair_tokens: int = 4096
    max_llm_finish_tokens: int = 8192
    max_slm_active_tokens_after_handoff: int = 4096
    max_final_answer_tokens: int = 2048


@dataclass(slots=True)
class SignalConfig:
    entropy_top_k: int = 20
    entropy_quantile: float = 0.9
    attention_query_window: int = 32
    attention_head_chunk_size: int = 4
    token_mass: int | None = None
    pvalue_epsilon: float = 1e-6
    grounding_epsilon: float = 1e-6
    profile_probe_with_cuda_sync: bool = False


@dataclass(slots=True)
class ControllerConfig:
    alarm_threshold: float | None = None
    threshold_path: str | None = None
    ready_epsilon: float = 1e-8


@dataclass(slots=True)
class ProtocolConfig:
    max_repair_episodes: int = 1
    max_trial_attempts: int = 2
    trial_blocks: int = 2
    after_repair_budget: Literal["llm_finish", "keep_slm"] = "llm_finish"
    include_discarded_suffix_in_llm_prompt: bool = False
    llm_repair_monitor: Literal["h_only", "dual", "finish_directly"] = "h_only"


@dataclass(slots=True)
class CostConfig:
    synchronize_cuda_for_timing: bool = True
    record_cuda_memory: bool = True
    record_attention_work_proxy: bool = True


@dataclass(slots=True)
class CalibrationConfig:
    warmup_samples: int = 64
    rank_samples: int = 256
    budget_samples: int = 128
    target_alarm_rate: float = 0.2
    max_tokens_per_sample: int = 4096


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
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self, require_references: bool = False) -> None:
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
        for role, model in (("slm", self.models.slm), ("llm", self.models.llm)):
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
            if require_references and not model.reference_path:
                raise ValueError(f"models.{role}.reference_path is required for the core run")
            if require_references and model.reference_path and not Path(model.reference_path).exists():
                raise FileNotFoundError(
                    f"Reference file for {role} does not exist: {model.reference_path}"
                )
        if not Path(self.data.path).exists():
            raise FileNotFoundError(f"Local dataset path does not exist: {self.data.path}")
        if self.signals.entropy_top_k < 2:
            raise ValueError("signals.entropy_top_k must be >= 2")
        if not 0.0 < self.signals.entropy_quantile < 1.0:
            raise ValueError("signals.entropy_quantile must be in (0, 1)")
        if self.signals.attention_query_window < 1:
            raise ValueError("signals.attention_query_window must be >= 1")
        if self.signals.token_mass is not None and self.signals.token_mass < 1:
            raise ValueError("signals.token_mass must be >= 1 when provided")
        if self.signals.attention_head_chunk_size < 1:
            raise ValueError("signals.attention_head_chunk_size must be >= 1")
        if self.prompt.enable_thinking and not self.prompt.thinking_end_marker:
            raise ValueError(
                "prompt.thinking_end_marker must be non-empty when thinking mode is enabled"
            )
        if self.generation.max_final_answer_tokens < 1:
            raise ValueError("generation.max_final_answer_tokens must be >= 1")
        if self.protocol.trial_blocks != 2:
            raise ValueError(
                "The core method fixes protocol.trial_blocks=2 because delta entropy requires two observations."
            )
        if self.controller.alarm_threshold is None and self.controller.threshold_path is None:
            if require_references:
                raise ValueError(
                    "Set controller.alarm_threshold or controller.threshold_path for the core run."
                )
        if require_references and self.controller.threshold_path and not Path(self.controller.threshold_path).exists():
            raise FileNotFoundError(
                f"controller.threshold_path does not exist: {self.controller.threshold_path}"
            )
        if self.protocol.max_repair_episodes != 1:
            raise ValueError(
                "The released core protocol fixes max_repair_episodes=1; additional loops are validation work."
            )
        valid_llm_monitors = {"h_only", "dual", "finish_directly"}
        if self.protocol.llm_repair_monitor not in valid_llm_monitors:
            raise ValueError(
                f"protocol.llm_repair_monitor must be one of {sorted(valid_llm_monitors)}"
            )
        if not 0.0 < self.calibration.target_alarm_rate < 1.0:
            raise ValueError("calibration.target_alarm_rate must be in (0, 1)")

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
        calibration=CalibrationConfig(**raw.get("calibration", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
    return cfg


def save_resolved_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.as_dict(), f, allow_unicode=True, sort_keys=False)
