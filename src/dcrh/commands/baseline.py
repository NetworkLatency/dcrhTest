from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from tqdm import tqdm

from ..config import load_config, save_resolved_config
from ..core.costs import CostLedger, reset_peak_memory
from ..evaluation.data import iter_examples
from ..evaluation.verifier import verify
from ..runtime.transformers.model_runner import LocalQwen3Runner
from ..utils.io import append_jsonl, ensure_directory, load_completed_ids, write_json
from ..utils.offline import force_offline_environment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local SLM-only or LLM-only baseline with the same decoder"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=["slm", "llm"], required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    force_offline_environment()
    cfg = load_config(args.config)
    cfg.validate(require_references=False)
    model_cfg = cfg.models.slm if args.role == "slm" else cfg.models.llm
    default_cap = (
        cfg.generation.max_initial_slm_tokens
        if args.role == "slm"
        else cfg.generation.max_llm_finish_tokens
    ) + cfg.generation.max_final_answer_tokens
    max_tokens = int(args.max_tokens or default_cap)
    if max_tokens < 1:
        raise ValueError("max-tokens must be positive")

    default_output = Path(cfg.output.directory) / f"{args.role}_only"
    output_dir = ensure_directory(args.output_dir or default_output)
    results_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    if cfg.output.overwrite:
        for path in (results_path, errors_path):
            if path.exists():
                path.unlink()
    save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    runner = LocalQwen3Runner(
        role=args.role,
        model_cfg=model_cfg,
        prompt_cfg=cfg.prompt,
        generation_cfg=cfg.generation,
        signal_cfg=cfg.signals,
        cost_cfg=cfg.cost,
    )
    write_json(
        output_dir / "run_metadata.json",
        {
            "mode": f"{args.role}_only",
            "max_tokens": max_tokens,
            "model": runner.model_metadata(),
            "grounding_probe_enabled": False,
            "offline_only": True,
            "kv_policy": "live_session_only_no_snapshots",
        },
    )

    completed = load_completed_ids(results_path)
    for example in tqdm(iter_examples(cfg.data), desc=f"{args.role}-only baseline"):
        if example.example_id in completed:
            continue
        try:
            device = str(runner.device)
            if cfg.cost.record_cuda_memory:
                reset_peak_memory([device])
            ledger = CostLedger(cfg.cost.record_attention_work_proxy)
            session = runner.create_session(
                question=example.question,
                shared_text="",
                control_text="",
                purpose=f"{args.role}_only",
                ledger=ledger,
                enable_grounding=False,
            )
            terminal_reason = "token_budget"
            for _ in range(max_tokens):
                step = session.step()
                if not step.ended or step.end_reason == "thinking_end":
                    continue
                terminal_reason = step.end_reason or "generation_end"
                break
            final_text = session.generated_text()
            session.close()
            if cfg.cost.record_cuda_memory:
                ledger.capture_gpu_memory([device])
            row = {
                "example_id": example.example_id,
                "question": example.question,
                "gold_answer": example.answer,
                "mode": f"{args.role}_only",
                "final_text": final_text if cfg.output.save_full_text else "",
                "terminal_reason": terminal_reason,
                "repair_episodes": 0,
                "trial_attempts": 0,
                "cost": ledger.summary(),
                "evaluation": verify(
                    output=final_text,
                    gold=example.answer,
                    verifier=cfg.evaluation.verifier,
                    dataset=cfg.data.dataset,
                ),
            }
            append_jsonl(results_path, [row])
        except Exception as exc:
            append_jsonl(
                errors_path,
                [
                    {
                        "example_id": example.example_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                ],
            )
            if not args.continue_on_error:
                raise

    print(f"results={results_path}")
    if errors_path.exists():
        print(f"errors={errors_path}")


if __name__ == "__main__":
    main()
