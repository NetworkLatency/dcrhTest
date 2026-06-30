from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tqdm import tqdm

from dcrh.config import load_config, save_resolved_config
from dcrh.core.costs import CostLedger, reset_peak_memory
from dcrh.core.scoring import MdrvTraceScorer
from dcrh.evaluation.data import iter_examples
from dcrh.runtime.transformers.model_runner import LocalQwen3Runner
from dcrh.utils.io import append_jsonl, ensure_directory, load_completed_ids, write_json
from dcrh.utils.offline import force_offline_environment
from tau_utils import summarize_tau_scores


def _parse_rhos(values: list[float]) -> list[float]:
    rhos = [float(value) for value in values]
    if not rhos:
        raise ValueError("At least one rho value is required")
    for rho in rhos:
        if rho <= 0.0 or rho > 1.0:
            raise ValueError("rho values must be in (0, 1]")
    return rhos


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_from_trace(
    example,
    trace,
    ledger: CostLedger,
    save_full_text: bool,
    save_per_step: bool,
) -> dict:
    per_step = trace.per_step
    max_step = (
        per_step[trace.max_router_score_step - 1]
        if trace.max_router_score_step is not None
        and trace.max_router_score_step <= len(per_step)
        else None
    )
    row = {
        "example_id": example.example_id,
        "question": example.question,
        "gold_answer": example.answer,
        "mode": "mdrv_tau_score",
        "score_definition": "S(x)=max_i R_i",
        "max_router_score": trace.max_router_score,
        "max_router_score_step": trace.max_router_score_step,
        "max_margin_drawdown": trace.max_margin_drawdown,
        "max_margin_drawdown_step": trace.max_margin_drawdown_step,
        "max_step": max_step,
        "terminal_reason": trace.terminal_reason,
        "num_slm_steps": len(trace.boundaries),
        "slm_generated_tokens": len(trace.generated_ids),
        "final_text": trace.slm_text if save_full_text else "",
        "attention_attempted_steps": sum(1 for step in per_step if step["G_i"] > 0.0),
        "attention_unavailable_steps": sum(
            1 for step in per_step if step["attention_unavailable_reason"] is not None
        ),
        "cost": ledger.summary(),
    }
    if save_per_step:
        row["per_step"] = per_step
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score calibration samples with S(x)=max_i R_i for MDRV tau selection"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rho", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--no-per-step", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    force_offline_environment()
    cfg = load_config(args.config)
    cfg.validate(require_tau=False, model_roles=("slm",))
    rhos = _parse_rhos(args.rho)

    default_output = Path(cfg.output.directory) / "tau_score"
    output_dir = ensure_directory(args.output_dir or default_output)
    results_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "tau_summary.json"
    if cfg.output.overwrite:
        for path in (results_path, errors_path, summary_path):
            if path.exists():
                path.unlink()
    save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    slm = LocalQwen3Runner(
        role="slm",
        model_cfg=cfg.models.slm,
        prompt_cfg=cfg.prompt,
        generation_cfg=cfg.generation,
        signal_cfg=cfg.signals,
        cost_cfg=cfg.cost,
    )
    write_json(
        output_dir / "run_metadata.json",
        {
            "method": "MDRV",
            "mode": "tau_score",
            "score_definition": "S(x)=max_i R_i",
            "rho_targets": rhos,
            "run_mode": cfg.protocol.run_mode,
            "route_discount": cfg.protocol.route_discount,
            "attention_route_mode": cfg.protocol.attention_route_mode,
            "slm_model": slm.model_metadata(),
            "offline_only": True,
            "kv_policy": "live_session_only_no_snapshots",
        },
    )

    scorer = MdrvTraceScorer(cfg=cfg, slm=slm)
    completed = load_completed_ids(results_path)
    rows = _load_existing_rows(results_path)
    scores = [
        float(row["max_router_score"])
        for row in rows
        if "max_router_score" in row
    ]

    for example in tqdm(iter_examples(cfg.data), desc="tau scoring"):
        if example.example_id in completed:
            continue
        try:
            device = str(slm.device)
            if cfg.cost.record_cuda_memory:
                reset_peak_memory([device])
            ledger = CostLedger(cfg.cost.record_attention_work_proxy)
            trace = scorer.score(
                example=example,
                ledger=ledger,
                tau=None,
                stop_on_takeover=False,
            )
            trace.session.close()
            if cfg.cost.record_cuda_memory:
                ledger.capture_gpu_memory([device])
            row = _row_from_trace(
                example=example,
                trace=trace,
                ledger=ledger,
                save_full_text=cfg.output.save_full_text,
                save_per_step=not args.no_per_step,
            )
            append_jsonl(results_path, [row])
            scores.append(float(row["max_router_score"]))
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

    summary = summarize_tau_scores(scores, rhos)
    write_json(summary_path, summary)
    print(f"results={results_path}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors_path.exists():
        print(f"errors={errors_path}")


if __name__ == "__main__":
    main()
