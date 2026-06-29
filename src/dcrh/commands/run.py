from __future__ import annotations

import argparse
import traceback

from tqdm import tqdm

from ..config import load_config, save_resolved_config
from ..evaluation.data import iter_examples
from ..evaluation.verifier import verify
from ..utils.io import (
    append_jsonl,
    ensure_directory,
    load_completed_ids,
    write_json,
)
from ..runtime.transformers.model_runner import LocalQwen3Runner
from ..utils.offline import force_offline_environment
from ..core.protocol import CoreProtocol


def _resolve_threshold(cfg) -> float:
    if cfg.controller.alarm_threshold is not None:
        return float(cfg.controller.alarm_threshold)
    raise ValueError("Set controller.alarm_threshold to the MDRV tau.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MDRV core experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    force_offline_environment()
    cfg = load_config(args.config)
    cfg.validate(require_tau=False)
    output_dir = ensure_directory(cfg.output.directory)
    results_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    if cfg.output.overwrite:
        for path in (results_path, errors_path):
            if path.exists():
                path.unlink()
    save_resolved_config(cfg, output_dir / "resolved_config.yaml")

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
    write_json(
        output_dir / "run_metadata.json",
        {
            "method": "MDRV",
            "tau": threshold,
            "run_mode": cfg.protocol.run_mode,
            "route_discount": cfg.protocol.route_discount,
            "takeover_mode": cfg.protocol.takeover_mode,
            "attention_route_mode": cfg.protocol.attention_route_mode,
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
