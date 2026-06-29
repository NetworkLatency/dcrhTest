from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..config import ExperimentConfig
from ..evaluation.data import Example
from ..runtime.transformers.model_runner import LocalQwen3Runner
from .boundaries import DoubleNewlineBoundaryDetector
from .costs import CostLedger, reset_peak_memory
from .router_signals import (
    REGION_NAMES,
    RegionSpans,
    build_region_spans_from_token_ids,
    find_first_content_token,
)
from .router_state import MdrvRouterState, MdrvStepRecord


@dataclass(slots=True)
class StepBoundary:
    step_index: int
    start_token_index: int
    end_token_index: int
    token_ids: list[int]
    text: str


@dataclass(slots=True)
class EffectiveBoundary:
    spans: RegionSpans
    skipped_token_ids: list[int]
    action_token_offset: int | None
    action_token_id: int | None
    action_token_text: str | None


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

    def _run_slm_trace(
        self,
        example: Example,
        ledger: CostLedger,
    ) -> tuple[list[int], str, str, list[StepBoundary], str, Any]:
        session = self.slm.create_session(
            question=example.question,
            shared_text="",
            control_text="",
            purpose="slm_mdrv_initial",
            ledger=ledger,
        )
        detector = DoubleNewlineBoundaryDetector()
        boundaries: list[StepBoundary] = []
        previous_boundary = 0
        terminal_reason = "slm_token_budget"

        for _ in range(self.cfg.generation.max_initial_slm_tokens):
            result = session.step()
            observation = result.observation
            if observation.token_id >= 0:
                token_index = len(session.generated_ids)
                boundary = detector.push(observation.text_piece, token_index)
                if boundary is not None:
                    token_ids = session.generated_ids[
                        previous_boundary : boundary.token_index
                    ]
                    boundaries.append(
                        StepBoundary(
                            step_index=len(boundaries) + 1,
                            start_token_index=previous_boundary,
                            end_token_index=boundary.token_index,
                            token_ids=[int(x) for x in token_ids],
                            text=self._decode(token_ids),
                        )
                    )
                    previous_boundary = boundary.token_index

            if not result.ended:
                continue
            if result.end_reason == "thinking_end":
                continue
            terminal_reason = result.end_reason or "slm_generation_end"
            break

        prompt_ids = [int(x) for x in session.prompt.input_ids.reshape(-1).tolist()]
        return (
            [int(x) for x in session.generated_ids],
            session.prompt.rendered_text,
            session.generated_text(),
            boundaries,
            terminal_reason,
            session,
        )

    def _effective_boundary(
        self,
        prompt_ids: Sequence[int],
        chunks: Sequence[StepBoundary],
        all_generated_ids: Sequence[int],
        boundary_end: int,
    ) -> EffectiveBoundary:
        continuation = [int(x) for x in all_generated_ids[boundary_end:]]
        found = find_first_content_token(continuation, self.slm.tokenizer)
        action_offset = None
        action_token_id = None
        action_token_text = None
        skipped: list[int] = []
        if found is not None:
            action_offset, action_token_id, action_token_text = found
            skipped = continuation[:action_offset]

        base_ids = [
            *prompt_ids,
            *(token for chunk in chunks for token in chunk.token_ids),
        ]
        effective_ids = [*base_ids, *skipped]
        spans = build_region_spans_from_token_ids(
            prompt_ids=prompt_ids,
            chunk_ids=[chunk.token_ids for chunk in chunks],
            input_ids=effective_ids,
        )
        return EffectiveBoundary(
            spans=spans,
            skipped_token_ids=skipped,
            action_token_offset=action_offset,
            action_token_id=action_token_id,
            action_token_text=action_token_text,
        )

    def _route_for(
        self,
        spans: RegionSpans,
        attention_enabled: bool,
    ) -> tuple[dict[str, float] | None, bool, str | None, int | None]:
        if not attention_enabled:
            return None, False, None, None
        state = self.slm.forward_boundary_state(
            prefix_text="",
            region_spans=spans,
            collect_attention_route=True,
        )
        route = (
            state.route_summary.route_distribution
            if state.route_summary is not None
            else None
        )
        layer = state.route_summary.observed_layer if state.route_summary is not None else None
        return route, state.attention_available, state.attention_unavailable_reason, layer

    def _step_record(
        self,
        boundary: StepBoundary,
        effective: EffectiveBoundary,
        prefix_token_len: int,
        tpm_state,
        route: dict[str, float] | None,
        route_velocity: float | None,
        bar_v: float | None,
        risk: float,
        margin_drawdown: float,
        segment_start: int | None,
        attention_available: bool,
        attention_reason: str | None,
        route_layer: int | None,
        takeover: bool,
    ) -> MdrvStepRecord:
        values = {name: None for name in REGION_NAMES}
        if route is not None:
            values.update({name: float(route[name]) for name in REGION_NAMES})
        tpm = tpm_state.tpm_margin
        return MdrvStepRecord(
            step_index=boundary.step_index,
            step_text=boundary.text,
            prefix_token_len=prefix_token_len,
            action_token_offset=effective.action_token_offset,
            skipped_action_tokens=len(effective.skipped_token_ids),
            action_token_id=effective.action_token_id,
            action_token_text=effective.action_token_text,
            M_i=tpm.tpm_margin,
            p_top1=tpm.p_top1,
            p_top2=tpm.p_top2,
            top1_token_id=tpm.top1_token_id,
            top1_token_text=tpm.top1_token_text,
            top2_token_id=tpm.top2_token_id,
            top2_token_text=tpm.top2_token_text,
            r_i_A=values["A"],
            r_i_O=values["O"],
            r_i_P=values["P"],
            r_i_C=values["C"],
            V_i=route_velocity,
            G_i=margin_drawdown,
            segment_start=segment_start,
            barV=bar_v,
            R_i=risk,
            attention_available=attention_available,
            attention_unavailable_reason=attention_reason,
            route_layer=route_layer,
            route_query_position=prefix_token_len - 1,
            takeover_decision_at_step=takeover,
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

        (
            generated_ids,
            _prompt_text,
            slm_text,
            boundaries,
            slm_terminal,
            slm_session,
        ) = self._run_slm_trace(example, ledger)
        prompt_ids = [int(x) for x in slm_session.prompt.input_ids.reshape(-1).tolist()]

        router = MdrvRouterState(
            tau=self.tau,
            route_discount=self.cfg.protocol.route_discount,
        )
        effective: list[EffectiveBoundary] = []
        records: list[MdrvStepRecord] = []
        triggered = False
        trigger_step: int | None = None
        anchor_step: int | None = None

        attention_mode = self.cfg.protocol.attention_route_mode
        attention_enabled = attention_mode == "last_layer_single_row"
        for boundary in boundaries:
            current_effective = self._effective_boundary(
                prompt_ids=prompt_ids,
                chunks=boundaries[: boundary.step_index],
                all_generated_ids=generated_ids,
                boundary_end=boundary.end_token_index,
            )
            effective.append(current_effective)
            tpm_state = self.slm.forward_boundary_state(
                prefix_text="",
                region_spans=current_effective.spans,
                collect_attention_route=False,
            )
            margin_update = router.update_margin(
                boundary.step_index,
                tpm_state.tpm_margin.tpm_margin,
            )

            route = None
            route_velocity = None
            bar_v = None
            risk = 0.0 if margin_update.margin_drawdown == 0.0 else margin_update.margin_drawdown
            attention_available = False
            attention_reason = None
            route_layer = None

            if margin_update.needs_attention and attention_enabled:
                baseline_route = None
                if margin_update.needs_baseline_route:
                    baseline_index = boundary.step_index - 2
                    if baseline_index >= 0:
                        (
                            baseline_route,
                            baseline_available,
                            baseline_reason,
                            _baseline_layer,
                        ) = self._route_for(
                            effective[baseline_index].spans,
                            attention_enabled=True,
                        )
                        if not baseline_available:
                            attention_reason = baseline_reason
                    else:
                        attention_reason = "baseline_route_missing"
                route, attention_available, current_reason, route_layer = self._route_for(
                    current_effective.spans,
                    attention_enabled=True,
                )
                if current_reason is not None:
                    attention_reason = current_reason
                if baseline_route is None and margin_update.needs_baseline_route:
                    attention_available = False
                route_velocity, bar_v, risk = router.route_discounted_risk(
                    route=route,
                    baseline_route=baseline_route,
                    attention_available=attention_available,
                )
            elif margin_update.needs_attention:
                route_velocity, bar_v, risk = router.route_discounted_risk(
                    route=None,
                    baseline_route=None,
                    attention_available=False,
                )

            takeover = risk >= self.tau and margin_update.margin_drawdown > 0.0
            records.append(
                self._step_record(
                    boundary=boundary,
                    effective=current_effective,
                    prefix_token_len=tpm_state.prefix_token_len,
                    tpm_state=tpm_state,
                    route=route,
                    route_velocity=route_velocity,
                    bar_v=bar_v,
                    risk=risk,
                    margin_drawdown=margin_update.margin_drawdown,
                    segment_start=margin_update.segment_start,
                    attention_available=attention_available,
                    attention_reason=attention_reason,
                    route_layer=route_layer,
                    takeover=takeover,
                )
            )
            if takeover:
                triggered = True
                trigger_step = boundary.step_index
                anchor_step = margin_update.segment_start
                break

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
            per_step=[asdict(record) for record in records],
            cost=ledger.summary(),
        )
