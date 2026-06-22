from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from tqdm import tqdm

from ..config import load_config, save_resolved_config
from ..evaluation.data import iter_examples
from ..evaluation.verifier import verify
from ..utils.io import (
    append_jsonl,
    ensure_directory,
    load_completed_ids,
    read_json,
    write_json,
)
from ..runtime.transformers.model_runner import LocalQwen3Runner
from ..utils.offline import force_offline_environment
from ..core.protocol import CoreProtocol
from ..core.reference import EmpiricalReference


def _resolve_threshold(cfg) -> float:
    if cfg.controller.alarm_threshold is not None:
        return float(cfg.controller.alarm_threshold)
    payload = read_json(cfg.controller.threshold_path)
    value = payload.get("threshold")
    if value is None:
        raise ValueError(
            f"Threshold file {cfg.controller.threshold_path} contains threshold=null; "
            "use the SLM threshold JSON, not the LLM reference report."
        )
    return float(value)




def _validate_reference(reference, runner, cfg, role: str) -> None:
    meta = reference.metadata
    if meta.get("role") != role:
        raise ValueError(f"Reference role mismatch: expected {role}, found {meta.get('role')}")
    model_meta = meta.get("model", {})
    saved_path = model_meta.get("path")
    current_path = str(Path(runner.model_cfg.path).resolve())
    if saved_path is not None and str(Path(saved_path).resolve()) != current_path:
        raise ValueError(
            f"{role} reference was calibrated for {saved_path!r}, not {current_path!r}. "
            "Rebuild the unlabeled reference for this local checkpoint."
        )
    checks = {
        "num_hidden_layers": int(runner.model.config.num_hidden_layers),
        "num_attention_heads": int(runner.model.config.num_attention_heads),
        "num_key_value_heads": int(runner.model.config.num_key_value_heads),
        "selected_layers": runner.selected_layers,
    }
    for key, current in checks.items():
        saved = model_meta.get(key)
        if saved is not None and saved != current:
            raise ValueError(
                f"{role} reference mismatch for {key}: saved={saved!r}, current={current!r}"
            )
    signal_checks = {
        "entropy_top_k": cfg.signals.entropy_top_k,
        "entropy_quantile": cfg.signals.entropy_quantile,
        "grounding_window": cfg.signals.attention_query_window,
        "attention_head_chunk_size": cfg.signals.attention_head_chunk_size,
        "pvalue_epsilon": cfg.signals.pvalue_epsilon,
        "grounding_epsilon": cfg.signals.grounding_epsilon,
    }
    for key, current in signal_checks.items():
        saved = meta.get(key)
        if saved is not None and saved != current:
            raise ValueError(
                f"{role} reference signal mismatch for {key}: saved={saved!r}, current={current!r}"
            )
    if cfg.signals.token_mass is not None and int(cfg.signals.token_mass) != int(
        reference.token_mass
    ):
        raise ValueError(
            f"{role} reference token_mass={reference.token_mass} but the current config "
            f"requests signals.token_mass={cfg.signals.token_mass}; rebuild the reference."
        )
    prompt_signature = {
        "system_prompt": cfg.prompt.system_prompt,
        "enable_thinking": cfg.prompt.enable_thinking,
        "sink_prefix_tokens": cfg.prompt.sink_prefix_tokens,
        "thinking_end_marker": cfg.prompt.thinking_end_marker,
    }
    if meta.get("prompt_signature") not in (None, prompt_signature):
        raise ValueError(f"{role} reference was built with a different prompt signature")
    generation_signature = {
        "do_sample": cfg.generation.do_sample,
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "top_k": cfg.generation.top_k,
        "presence_penalty": cfg.generation.presence_penalty,
        "repetition_penalty": cfg.generation.repetition_penalty,
        "seed": cfg.generation.seed,
    }
    if meta.get("generation_signature") not in (None, generation_signature):
        raise ValueError(f"{role} reference was built with a different generation signature")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline DCRH core experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    force_offline_environment()
    cfg = load_config(args.config)
    cfg.validate(require_references=True)
    output_dir = ensure_directory(cfg.output.directory)
    results_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    if cfg.output.overwrite:
        for path in (results_path, errors_path):
            if path.exists():
                path.unlink()
    save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    slm_reference = EmpiricalReference.load(cfg.models.slm.reference_path)
    llm_reference = EmpiricalReference.load(cfg.models.llm.reference_path)
    threshold = _resolve_threshold(cfg)

    slm = LocalQwen3Runner(
        role="slm",
        model_cfg=cfg.models.slm,
        prompt_cfg=cfg.prompt,
        generation_cfg=cfg.generation,
        signal_cfg=cfg.signals,
        cost_cfg=cfg.cost,
    )
    llm = LocalQwen3Runner(
        role="llm",
        model_cfg=cfg.models.llm,
        prompt_cfg=cfg.prompt,
        generation_cfg=cfg.generation,
        signal_cfg=cfg.signals,
        cost_cfg=cfg.cost,
    )
    _validate_reference(slm_reference, slm, cfg, "slm")
    _validate_reference(llm_reference, llm, cfg, "llm")
    write_json(
        output_dir / "run_metadata.json",
        {
            "alarm_threshold": threshold,
            "slm_reference_metadata": slm_reference.metadata,
            "llm_reference_metadata": llm_reference.metadata,
            "slm_model": slm.model_metadata(),
            "llm_model": llm.model_metadata(),
            "offline_only": True,
            "kv_policy": "live_session_only_no_snapshots",
        },
    )

    protocol = CoreProtocol(
        cfg=cfg,
        slm=slm,
        llm=llm,
        slm_reference=slm_reference,
        llm_reference=llm_reference,
        alarm_threshold=threshold,
    )
    completed = load_completed_ids(results_path)
    for example in tqdm(iter_examples(cfg.data), desc="core experiment"):
        if example.example_id in completed:
            continue
        try:
            result = protocol.run(example)
            row = result.as_dict(save_full_text=cfg.output.save_full_text)
            row["evaluation"] = verify(
                output=result.final_text,
                gold=example.answer,
                verifier=cfg.evaluation.verifier,
                dataset=cfg.data.dataset,
            )
            append_jsonl(results_path, [row])
        except Exception as exc:
            error = {
                "example_id": example.example_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, [error])
            if not args.continue_on_error:
                raise

    print(f"results={results_path}")
    if errors_path.exists():
        print(f"errors={errors_path}")


if __name__ == "__main__":
    main()
