from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..config import ExperimentConfig
from ..evaluation.data import Example
from ..runtime.transformers.model_runner import LocalQwen3Runner
from .costs import CostLedger, reset_peak_memory
from .scoring import MdrvTraceScorer


@dataclass(slots=True)
class CoreRunResult:
    example_id: str
    question: str
    gold_answer: str | None
    final_text: str
    terminal_reason: str
    final_source: str
    triggered: bool
    trigger_step: int | None
    anchor_step: int | None
    discarded_steps: list[int]
    trigger_discarded_steps: list[int]
    num_slm_steps: int
    slm_generated_tokens: int
    llm_generated_tokens: int
    tau: float
    mode: str
    delimiter: str
    per_step: list[dict[str, Any]]
    cost: dict[str, Any]

    def as_dict(self, save_full_text: bool = True) -> dict[str, Any]:
        row = asdict(self)
        if not save_full_text:
            row["final_text"] = ""
        return row


class CoreProtocol:
    """MDRV: Margin Drawdown with Route Velocity rollback takeover."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        slm: LocalQwen3Runner,
        llm: LocalQwen3Runner,
        alarm_threshold: float,
    ) -> None:
        self.cfg = cfg
        self.slm = slm
        self.llm = llm
        self.tau = float(alarm_threshold)
        self.delimiter = "\n\n"
        if cfg.protocol.run_mode != "offline_replay":
            raise NotImplementedError(
                "MDRV currently implements offline_replay; online_collaboration is not wired yet."
            )

    def _decode(self, token_ids: Sequence[int]) -> str:
        return self.slm.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _continue_llm(
        self,
        example: Example,
        shared_text: str,
        ledger: CostLedger,
    ) -> tuple[str, int, str]:
        session = self.llm.create_session(
            question=example.question,
            shared_text=shared_text,
            control_text=self.cfg.prompt.takeover_cue,
            purpose="llm_mdrv_takeover",
            ledger=ledger,
        )
        terminal = "llm_finish_budget"
        for _ in range(self.cfg.generation.max_llm_finish_tokens):
            result = session.step()
            if not result.ended:
                continue
            if result.end_reason == "thinking_end":
                continue
            terminal = result.end_reason or "llm_generation_end"
            break
        text = shared_text + session.generated_text()
        generated_tokens = len(session.generated_ids)
        session.close()
        return text, generated_tokens, terminal

    def run(self, example: Example) -> CoreRunResult:
        devices = [str(self.slm.device), str(self.llm.device)]
        if self.cfg.cost.record_cuda_memory:
            reset_peak_memory(devices)
        ledger = CostLedger(self.cfg.cost.record_attention_work_proxy)

        trace = MdrvTraceScorer(self.cfg, self.slm).score(
            example=example,
            ledger=ledger,
            tau=self.tau,
            stop_on_takeover=True,
        )
        generated_ids = trace.generated_ids
        slm_text = trace.slm_text
        boundaries = trace.boundaries
        slm_terminal = trace.terminal_reason
        slm_session = trace.session
        triggered = trace.triggered
        trigger_step = trace.trigger_step
        anchor_step = trace.anchor_step

        final_source = "SLM"
        final_text = slm_text
        terminal_reason = slm_terminal
        discarded_steps: list[int] = []
        trigger_discarded_steps: list[int] = []
        llm_generated_tokens = 0

        if triggered and anchor_step is not None:
            anchor_boundary = boundaries[anchor_step - 1]
            if self.cfg.protocol.takeover_mode == "current" and trigger_step is not None:
                takeover_boundary = boundaries[trigger_step - 1]
                takeover_token_index = takeover_boundary.end_token_index
            else:
                takeover_token_index = anchor_boundary.end_token_index
            trusted_text = self._decode(generated_ids[:takeover_token_index])
            discarded_steps = [
                step.step_index
                for step in boundaries
                if step.end_token_index > takeover_token_index
            ]
            trigger_discarded_steps = list(range(anchor_step + 1, (trigger_step or anchor_step) + 1))
            decode_seconds = slm_session.generated_cost_slice(
                takeover_token_index,
                len(generated_ids),
            )
            ledger.mark_discarded_generation(
                role="slm",
                reason="mdrv_rollback_suffix",
                tokens=len(generated_ids) - takeover_token_index,
                characters=len(slm_text) - len(trusted_text),
                decode_seconds=decode_seconds,
            )
            ledger.transition(
                "mdrv_takeover",
                trigger_step=trigger_step,
                anchor_step=anchor_step,
                takeover_mode=self.cfg.protocol.takeover_mode,
                discarded_steps=discarded_steps,
                trigger_discarded_steps=trigger_discarded_steps,
                tau=self.tau,
            )
            final_text, llm_generated_tokens, terminal_reason = self._continue_llm(
                example=example,
                shared_text=trusted_text,
                ledger=ledger,
            )
            final_source = "LLM"

        slm_session.close()
        if self.cfg.cost.record_cuda_memory:
            ledger.capture_gpu_memory(devices)

        return CoreRunResult(
            example_id=example.example_id,
            question=example.question,
            gold_answer=example.answer,
            final_text=final_text,
            terminal_reason=terminal_reason,
            final_source=final_source,
            triggered=triggered,
            trigger_step=trigger_step,
            anchor_step=anchor_step,
            discarded_steps=discarded_steps,
            trigger_discarded_steps=trigger_discarded_steps,
            num_slm_steps=len(boundaries),
            slm_generated_tokens=len(generated_ids),
            llm_generated_tokens=llm_generated_tokens,
            tau=self.tau,
            mode=self.cfg.protocol.run_mode,
            delimiter=self.delimiter,
            per_step=trace.per_step,
            cost=ledger.summary(),
        )
