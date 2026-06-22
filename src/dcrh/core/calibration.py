from __future__ import annotations

import itertools
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ..config import ExperimentConfig, ModelConfig
from ..evaluation.data import Example, iter_examples
from ..runtime.transformers.model_runner import LocalQwen3Runner
from ..utils.io import write_json
from .costs import CostLedger
from .monitor import SessionMonitor
from .reference import EmpiricalReference


def _take_examples(cfg: ExperimentConfig, count: int) -> list[Example]:
    return list(itertools.islice(iter_examples(cfg.data), count))


def _run_plain_trace(
    runner: LocalQwen3Runner,
    example: Example,
    token_mass: int,
    cfg: ExperimentConfig,
    purpose: str,
    max_tokens: int,
    reference: EmpiricalReference | None = None,
    threshold: float | None = None,
    enable_grounding: bool = True,
) -> tuple[SessionMonitor, CostLedger]:
    ledger = CostLedger(cfg.cost.record_attention_work_proxy)
    session = runner.create_session(
        question=example.question,
        shared_text="",
        control_text="",
        purpose=purpose,
        ledger=ledger,
        enable_grounding=enable_grounding,
    )
    monitor = SessionMonitor(
        session=session,
        token_mass=token_mass,
        entropy_quantile=cfg.signals.entropy_quantile,
        grounding_window=cfg.signals.attention_query_window,
        reference=reference,
        threshold=threshold,
        pvalue_epsilon=cfg.signals.pvalue_epsilon,
        use_grounding_channel=enable_grounding,
        require_grounding=enable_grounding,
    )
    for _ in range(max_tokens):
        event = monitor.step(allow_alarm=False)
        if event.kind == "end":
            break
    session.close()
    return monitor, ledger


def calibrate_model(
    cfg: ExperimentConfig,
    role: str,
    output_reference: str | Path,
    output_threshold: str | Path,
) -> dict:
    if role not in {"slm", "llm"}:
        raise ValueError("role must be 'slm' or 'llm'")
    cfg.validate(require_references=False)
    needs_grounding_reference = role == "slm" or cfg.protocol.llm_repair_monitor == "dual"
    model_cfg: ModelConfig = cfg.models.slm if role == "slm" else cfg.models.llm
    runner = LocalQwen3Runner(
        role=role,
        model_cfg=model_cfg,
        prompt_cfg=cfg.prompt,
        generation_cfg=cfg.generation,
        signal_cfg=cfg.signals,
        cost_cfg=cfg.cost,
    )

    budget_count = cfg.calibration.budget_samples if role == "slm" else 0
    total_needed = (
        cfg.calibration.warmup_samples
        + cfg.calibration.rank_samples
        + budget_count
    )
    examples = _take_examples(cfg, total_needed)
    if len(examples) < total_needed:
        raise ValueError(
            f"Calibration requested {total_needed} examples but only {len(examples)} are available"
        )
    warmup_examples = examples[: cfg.calibration.warmup_samples]
    rank_start = cfg.calibration.warmup_samples
    rank_end = rank_start + cfg.calibration.rank_samples
    rank_examples = examples[rank_start:rank_end]
    budget_examples = examples[rank_end : rank_end + budget_count]

    atomic_lengths: list[int] = []
    calibration_costs: list[dict] = []
    huge_mass = 10**12
    for example in tqdm(warmup_examples, desc=f"{role} warmup atomic clock"):
        monitor, ledger = _run_plain_trace(
            runner=runner,
            example=example,
            token_mass=huge_mass,
            cfg=cfg,
            purpose=f"calibration_{role}_warmup",
            max_tokens=cfg.calibration.max_tokens_per_sample,
            enable_grounding=False,
        )
        atomic_lengths.extend(x for x in monitor.atomic_lengths if x > 0)
        calibration_costs.append(ledger.summary())
    if not atomic_lengths:
        raise RuntimeError(
            "No double-newline atomic boundaries were observed during warmup; "
            "the token-mass clock cannot be estimated."
        )
    empirical_token_mass = max(
        1, int(np.median(np.asarray(atomic_lengths, dtype=np.int64)))
    )
    token_mass = (
        int(cfg.signals.token_mass)
        if cfg.signals.token_mass is not None
        else empirical_token_mass
    )

    delta_mean: list[float] = []
    delta_quantile: list[float] = []
    grounding: list[float] = []
    rank_trajectory_counts: list[int] = []
    for example in tqdm(rank_examples, desc=f"{role} rank reference"):
        monitor, ledger = _run_plain_trace(
            runner=runner,
            example=example,
            token_mass=token_mass,
            cfg=cfg,
            purpose=f"calibration_{role}_rank",
            max_tokens=cfg.calibration.max_tokens_per_sample,
            enable_grounding=needs_grounding_reference,
        )
        previous = None
        for block in monitor.blocks:
            if needs_grounding_reference and block.grounding is None:
                raise RuntimeError(
                    "Dual-channel rank calibration requires grounding values. "
                    "Use a G-capable runner or run an entropy-only calibration path."
                )
            if block.grounding is not None:
                grounding.append(block.grounding)
            if previous is not None:
                eps = cfg.signals.pvalue_epsilon
                delta_mean.append(
                    math.log((block.entropy_mean + eps) / (previous.entropy_mean + eps))
                )
                delta_quantile.append(
                    math.log(
                        (block.entropy_quantile + eps)
                        / (previous.entropy_quantile + eps)
                    )
                )
            previous = block
        rank_trajectory_counts.append(len(monitor.blocks))
        calibration_costs.append(ledger.summary())

    if not delta_mean:
        raise RuntimeError(
            "Too few complete statistical blocks were collected for the empirical entropy reference."
        )
    if needs_grounding_reference and not grounding:
        raise RuntimeError(
            "Too few complete statistical blocks were collected for the empirical grounding reference."
        )

    metadata = {
        "method": "empirical_rank_reference",
        "role": role,
        "model": runner.model_metadata(),
        "token_mass": token_mass,
        "token_mass_source": (
            "configured" if cfg.signals.token_mass is not None else "warmup_median"
        ),
        "warmup_atomic_segments": len(atomic_lengths),
        "warmup_atomic_median": float(empirical_token_mass),
        "rank_samples": len(rank_examples),
        "rank_blocks_per_sample": rank_trajectory_counts,
        "channels": ["H", "G"] if needs_grounding_reference else ["H"],
        "entropy_top_k": cfg.signals.entropy_top_k,
        "entropy_quantile": cfg.signals.entropy_quantile,
        "pvalue_epsilon": cfg.signals.pvalue_epsilon,
        "grounding_window": (
            cfg.signals.attention_query_window if needs_grounding_reference else None
        ),
        "attention_head_chunk_size": (
            cfg.signals.attention_head_chunk_size if needs_grounding_reference else None
        ),
        "grounding_epsilon": (
            cfg.signals.grounding_epsilon if needs_grounding_reference else None
        ),
        "prompt_signature": {
            "system_prompt": cfg.prompt.system_prompt,
            "enable_thinking": cfg.prompt.enable_thinking,
            "sink_prefix_tokens": cfg.prompt.sink_prefix_tokens,
            "thinking_end_marker": cfg.prompt.thinking_end_marker,
        },
        "generation_signature": {
            "do_sample": cfg.generation.do_sample,
            "temperature": cfg.generation.temperature,
            "top_p": cfg.generation.top_p,
            "top_k": cfg.generation.top_k,
            "presence_penalty": cfg.generation.presence_penalty,
            "repetition_penalty": cfg.generation.repetition_penalty,
            "seed": cfg.generation.seed,
        },
        "kv_policy": "live_session_only_no_snapshots",
    }
    reference = EmpiricalReference(
        delta_mean=np.asarray(delta_mean),
        delta_quantile=np.asarray(delta_quantile),
        grounding=np.asarray(grounding, dtype=np.float64),
        token_mass=token_mass,
        metadata=metadata,
    )
    reference.save(output_reference)

    maxima: list[float] = []
    threshold: float | None = None
    if role == "slm":
        for example in tqdm(budget_examples, desc=f"{role} budget threshold"):
            monitor, ledger = _run_plain_trace(
                runner=runner,
                example=example,
                token_mass=token_mass,
                cfg=cfg,
                purpose=f"calibration_{role}_budget",
                max_tokens=cfg.calibration.max_tokens_per_sample,
                reference=reference,
                threshold=float("inf"),
            )
            max_score = 0.0
            if monitor.controller is not None:
                # Recompute to retain the maximum over the whole trajectory rather than only the final score.
                from .sequential import DualChannelController

                controller = DualChannelController(
                    reference=reference,
                    threshold=float("inf"),
                    pvalue_epsilon=cfg.signals.pvalue_epsilon,
                )
                for block in monitor.blocks:
                    update = controller.update(block, allow_alarm=False)
                    max_score = max(
                        max_score, update.entropy_score, update.grounding_score
                    )
            maxima.append(max_score)
            calibration_costs.append(ledger.summary())

        if not maxima:
            raise RuntimeError("No SLM budget trajectories were collected")
        quantile = 1.0 - cfg.calibration.target_alarm_rate
        threshold = max(
            1e-12, float(np.quantile(np.asarray(maxima), quantile, method="higher"))
        )
        threshold_payload = {
            "role": role,
            "threshold": threshold,
            "target_problem_alarm_rate": cfg.calibration.target_alarm_rate,
            "selection_quantile": quantile,
            "budget_max_scores": maxima,
            "reference_path": str(Path(output_reference).resolve()),
            "token_mass": token_mass,
            "note": "One scalar operating point; no decision surface is trained.",
        }
    else:
        threshold_payload = {
            "role": role,
            "threshold": None,
            "budget_max_scores": [],
            "reference_path": str(Path(output_reference).resolve()),
            "token_mass": token_mass,
            "note": "The LLM reference is used for settling; the core protocol does not use an LLM alarm threshold.",
            "channels": ["H", "G"] if needs_grounding_reference else ["H"],
        }
    write_json(output_threshold, threshold_payload)

    report = {
        "reference_path": str(Path(output_reference).resolve()),
        "threshold_path": str(Path(output_threshold).resolve()),
        "threshold": threshold,
        "token_mass": token_mass,
        "num_delta_mean": len(delta_mean),
        "num_delta_quantile": len(delta_quantile),
        "num_grounding": len(grounding),
        "model": runner.model_metadata(),
        "calibration_costs": calibration_costs,
    }
    report_path = Path(output_reference).with_suffix(".report.json")
    write_json(report_path, report)
    return report
