from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ..config import ExperimentConfig
from ..evaluation.data import Example
from .costs import CostLedger, reset_peak_memory
from .monitor import MonitorEvent, SessionMonitor
from .reference import EmpiricalReference
from .sequential import ControllerUpdate

if TYPE_CHECKING:
    from ..runtime.transformers.model_runner import GenerationSession, LocalQwen3Runner


@dataclass(slots=True)
class PhaseOutcome:
    kind: str
    session: GenerationSession
    monitor: SessionMonitor
    event: MonitorEvent | None
    reason: str


@dataclass(slots=True)
class CoreRunResult:
    example_id: str
    question: str
    gold_answer: str | None
    final_text: str
    terminal_reason: str
    repair_episodes: int
    trial_attempts: int
    cost: dict[str, Any]

    def as_dict(self, save_full_text: bool = True) -> dict[str, Any]:
        row = asdict(self)
        if not save_full_text:
            row["final_text"] = ""
        return row


def _block_payload(event: MonitorEvent, phase: str) -> dict[str, Any]:
    assert event.block is not None
    payload: dict[str, Any] = {
        "phase": phase,
        "block": asdict(event.block),
    }
    if event.update is not None:
        payload["controller"] = asdict(event.update)
    return payload


class CoreProtocol:
    """Detection -> rollback -> LLM repair -> SLM speculative handoff."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        slm: LocalQwen3Runner,
        llm: LocalQwen3Runner,
        slm_reference: EmpiricalReference,
        llm_reference: EmpiricalReference,
        alarm_threshold: float,
    ) -> None:
        self.cfg = cfg
        self.slm = slm
        self.llm = llm
        self.slm_reference = slm_reference
        self.llm_reference = llm_reference
        self.alarm_threshold = float(alarm_threshold)

    def _new_monitor(
        self,
        session: GenerationSession,
        reference: EmpiricalReference,
        use_grounding_channel: bool = True,
        require_grounding: bool | None = None,
    ) -> SessionMonitor:
        token_mass = (
            self.cfg.signals.token_mass
            if self.cfg.signals.token_mass is not None
            else reference.token_mass
        )
        return SessionMonitor(
            session=session,
            token_mass=token_mass,
            entropy_quantile=self.cfg.signals.entropy_quantile,
            grounding_window=self.cfg.signals.attention_query_window,
            reference=reference,
            threshold=self.alarm_threshold,
            pvalue_epsilon=self.cfg.signals.pvalue_epsilon,
            use_entropy_channel=True,
            use_grounding_channel=use_grounding_channel,
            require_grounding=require_grounding,
        )

    def _run_slm_active(
        self,
        session: GenerationSession,
        monitor: SessionMonitor,
        ledger: CostLedger,
        max_tokens: int,
        phase: str,
        honor_alarm: bool = True,
    ) -> PhaseOutcome:
        for _ in range(max_tokens):
            event = monitor.step(allow_alarm=honor_alarm)
            if event.kind == "block":
                ledger.transition("statistical_block", **_block_payload(event, phase))
                if honor_alarm and event.update is not None and event.update.alarm:
                    return PhaseOutcome("alarm", session, monitor, event, "dual_channel_alarm")
            elif event.kind == "end":
                return PhaseOutcome("end", session, monitor, event, event.end_reason or "end")
        return PhaseOutcome("budget", session, monitor, None, "token_budget")

    def _run_llm_until_ready(
        self,
        session: GenerationSession,
        ledger: CostLedger,
        max_tokens: int,
        phase: str,
    ) -> PhaseOutcome:
        monitor_mode = self.cfg.protocol.llm_repair_monitor
        use_grounding = monitor_mode == "dual"
        monitor = self._new_monitor(
            session,
            self.llm_reference,
            use_grounding_channel=use_grounding,
            require_grounding=use_grounding,
        )
        for _ in range(max_tokens):
            event = monitor.step(allow_alarm=False)
            if event.kind == "block":
                ledger.transition("statistical_block", **_block_payload(event, phase))
                assert monitor.controller is not None
                ready = (
                    monitor_mode != "finish_directly"
                    and monitor.controller.ready(
                        minimum_blocks=2,
                        epsilon=self.cfg.controller.ready_epsilon,
                    )
                )
                if ready:
                    reason = (
                        "llm_channels_settled"
                        if monitor_mode == "dual"
                        else "llm_entropy_settled"
                    )
                    return PhaseOutcome("ready", session, monitor, event, reason)
            elif event.kind == "end":
                return PhaseOutcome("end", session, monitor, event, event.end_reason or "end")
        return PhaseOutcome("budget", session, monitor, None, "llm_repair_budget")

    def _run_slm_trial(
        self,
        session: GenerationSession,
        ledger: CostLedger,
        phase: str,
    ) -> PhaseOutcome:
        monitor = self._new_monitor(session, self.slm_reference)
        max_tokens = self.cfg.generation.max_slm_active_tokens_after_handoff
        for _ in range(max_tokens):
            event = monitor.step(allow_alarm=False)
            if event.kind == "block":
                ledger.transition("statistical_block", **_block_payload(event, phase))
                assert monitor.controller is not None
                if monitor.controller.blocks_seen >= self.cfg.protocol.trial_blocks:
                    accepted = monitor.controller.ready(
                        minimum_blocks=self.cfg.protocol.trial_blocks,
                        epsilon=self.cfg.controller.ready_epsilon,
                    )
                    return PhaseOutcome(
                        "accepted" if accepted else "rejected",
                        session,
                        monitor,
                        event,
                        "slm_trial_safe" if accepted else "slm_trial_realarms",
                    )
            elif event.kind == "end":
                # A final answer produced during the trial is committed; no return decision remains.
                return PhaseOutcome("end", session, monitor, event, event.end_reason or "end")
        return PhaseOutcome("rejected", session, monitor, None, "slm_trial_budget")

    def _run_llm_finish(
        self,
        example: Example,
        shared_text: str,
        ledger: CostLedger,
        purpose: str,
    ) -> tuple[str, str]:
        session = self.llm.create_session(
            question=example.question,
            shared_text=shared_text,
            control_text="",
            purpose=purpose,
            ledger=ledger,
            enable_grounding=False,
        )
        text, reason = self._continue_live_session(
            session=session,
            base_shared_text=shared_text,
            max_tokens=self.cfg.generation.max_llm_finish_tokens,
            budget_reason="llm_finish_budget",
        )
        session.close()
        return text, reason

    def _continue_live_session(
        self,
        session: GenerationSession,
        base_shared_text: str,
        max_tokens: int,
        budget_reason: str,
    ) -> tuple[str, str]:
        """Complete the answer from an existing cache; a thinking delimiter is not EOS."""
        session.disable_grounding()
        reason = budget_reason
        for _ in range(max_tokens):
            result = session.step()
            if not result.ended:
                continue
            if result.end_reason == "thinking_end":
                continue
            reason = result.end_reason or "generation_end"
            break
        return base_shared_text + session.generated_text(), reason

    def _rollback_and_start_llm(
        self,
        example: Example,
        base_shared_text: str,
        outcome: PhaseOutcome,
        ledger: CostLedger,
        repair_episode: int,
    ) -> tuple[str, GenerationSession]:
        if outcome.event is None or outcome.event.update is None:
            raise RuntimeError("Rollback requires an alarm event")
        update: ControllerUpdate = outcome.event.update
        if update.onset_block is None:
            raise RuntimeError("Alarm has no estimated onset block")
        rollback_idx = outcome.monitor.rollback_token_index(update.onset_block)
        detection_idx = len(outcome.session.generated_ids)
        trusted_local = outcome.session.generated_text(rollback_idx)
        detected_local = outcome.session.generated_text(detection_idx)
        discarded_local = outcome.session.generated_text(detection_idx)[len(trusted_local) :]
        trusted_shared = base_shared_text + trusted_local
        detection_shared = base_shared_text + detected_local
        discarded_tokens = detection_idx - rollback_idx
        discarded_decode_seconds, discarded_probe_seconds = (
            outcome.session.generated_cost_slice(rollback_idx, detection_idx)
        )

        ledger.mark_discarded_generation(
            role="slm",
            reason="rollback_suffix",
            tokens=discarded_tokens,
            characters=len(discarded_local),
            decode_seconds=discarded_decode_seconds,
            probe_seconds=discarded_probe_seconds,
        )
        ledger.transition(
            "rollback",
            trigger=update.trigger,
            onset_block=update.onset_block,
            detection_block=update.block_index,
            rollback_generated_token_index=rollback_idx,
            detection_generated_token_index=detection_idx,
            discarded_slm_tokens=discarded_tokens,
            discarded_slm_decode_seconds=discarded_decode_seconds,
            discarded_slm_probe_seconds=discarded_probe_seconds,
            trusted_text_characters=len(trusted_shared),
        )

        repair_control = self.cfg.prompt.repair_cue
        if self.cfg.protocol.include_discarded_suffix_in_llm_prompt and discarded_local:
            repair_control += (
                "\n[Discarded provisional suffix; use only for diagnosis, do not copy it.]\n"
                + discarded_local
                + "\n[End discarded suffix]\n"
            )
        actual_encoding = self.llm.build_prompt(
            example.question, trusted_shared, repair_control
        )
        detection_encoding = self.llm.build_prompt(
            example.question, detection_shared, repair_control
        )
        ledger.record_rollback_tradeoff(
            slm_discarded_tokens=discarded_tokens,
            llm_actual_prefill_tokens=actual_encoding.total_tokens,
            llm_detection_point_prefill_tokens=detection_encoding.total_tokens,
            llm_actual_prefill_qk_elements_upper_bound=(
                self.llm.estimate_prefill_qk_elements_upper_bound(
                    actual_encoding.total_tokens
                )
            ),
            llm_detection_point_qk_elements_upper_bound=(
                self.llm.estimate_prefill_qk_elements_upper_bound(
                    detection_encoding.total_tokens
                )
            ),
            llm_actual_estimated_kv_bytes=self.llm.estimate_kv_bytes(
                actual_encoding.total_tokens
            ),
            llm_detection_point_estimated_kv_bytes=self.llm.estimate_kv_bytes(
                detection_encoding.total_tokens
            ),
        )
        outcome.session.close()

        llm_session = self.llm.create_session(
            question=example.question,
            shared_text=trusted_shared,
            control_text=repair_control,
            purpose=f"llm_upgrade_repair_{repair_episode}",
            ledger=ledger,
            enable_grounding=self.cfg.protocol.llm_repair_monitor == "dual",
        )
        # Only the fixed cue is committed to the public reasoning state. Optional diagnostic suffix is not.
        committed_repair_base = trusted_shared + self.cfg.prompt.repair_cue
        return committed_repair_base, llm_session

    def run(self, example: Example) -> CoreRunResult:
        devices = [str(self.slm.device), str(self.llm.device)]
        if self.cfg.cost.record_cuda_memory:
            reset_peak_memory(devices)
        ledger = CostLedger(self.cfg.cost.record_attention_work_proxy)
        repair_episodes = 0
        trial_attempts = 0
        shared_text = ""

        slm_session = self.slm.create_session(
            question=example.question,
            shared_text=shared_text,
            control_text="",
            purpose="slm_initial",
            ledger=ledger,
        )
        slm_monitor = self._new_monitor(slm_session, self.slm_reference)
        initial = self._run_slm_active(
            session=slm_session,
            monitor=slm_monitor,
            ledger=ledger,
            max_tokens=self.cfg.generation.max_initial_slm_tokens,
            phase="SLM_ACTIVE_INITIAL",
            honor_alarm=True,
        )
        if initial.kind != "alarm":
            if initial.reason == "thinking_end":
                final_text, terminal = self._continue_live_session(
                    session=slm_session,
                    base_shared_text=shared_text,
                    max_tokens=self.cfg.generation.max_final_answer_tokens,
                    budget_reason="slm_final_answer_budget",
                )
                reason = f"initial_slm_{terminal}"
            else:
                final_text = shared_text + slm_session.generated_text()
                reason = f"initial_slm_{initial.reason}"
            slm_session.close()
            if self.cfg.cost.record_cuda_memory:
                ledger.capture_gpu_memory(devices)
            return CoreRunResult(
                example_id=example.example_id,
                question=example.question,
                gold_answer=example.answer,
                final_text=final_text,
                terminal_reason=reason,
                repair_episodes=repair_episodes,
                trial_attempts=trial_attempts,
                cost=ledger.summary(),
            )

        repair_episodes += 1
        repair_base, llm_session = self._rollback_and_start_llm(
            example=example,
            base_shared_text=shared_text,
            outcome=initial,
            ledger=ledger,
            repair_episode=repair_episodes,
        )

        while True:
            llm_outcome = self._run_llm_until_ready(
                session=llm_session,
                ledger=ledger,
                max_tokens=self.cfg.generation.max_llm_repair_tokens,
                phase=f"LLM_REPAIR_{repair_episodes}",
            )
            if llm_outcome.kind == "end":
                if llm_outcome.reason == "thinking_end":
                    final_text, terminal = self._continue_live_session(
                        session=llm_session,
                        base_shared_text=repair_base,
                        max_tokens=self.cfg.generation.max_final_answer_tokens,
                        budget_reason="llm_final_answer_budget",
                    )
                    reason = f"llm_repair_{terminal}"
                else:
                    final_text = repair_base + llm_session.generated_text()
                    reason = f"llm_repair_{llm_outcome.reason}"
                llm_session.close()
                break
            if llm_outcome.kind == "budget":
                final_text, finish_reason = self._continue_live_session(
                    session=llm_session,
                    base_shared_text=repair_base,
                    max_tokens=self.cfg.generation.max_llm_finish_tokens,
                    budget_reason="llm_finish_after_repair_budget",
                )
                llm_session.close()
                reason = finish_reason
                break

            candidate_shared = repair_base + llm_session.generated_text()
            ledger.transition(
                "llm_ready_for_trial",
                llm_generated_tokens=len(llm_session.generated_ids),
                candidate_characters=len(candidate_shared),
            )
            # Enforce text-only handoff: drop the live LLM cache before SLM trial.
            llm_session.close()

            trial_attempts += 1
            trial_session = self.slm.create_session(
                question=example.question,
                shared_text=candidate_shared,
                control_text="",
                purpose=f"slm_trial_{trial_attempts}",
                ledger=ledger,
            )
            trial_outcome = self._run_slm_trial(
                session=trial_session,
                ledger=ledger,
                phase=f"SLM_TRIAL_{trial_attempts}",
            )

            if trial_outcome.kind == "end":
                if trial_outcome.reason == "thinking_end":
                    final_text, terminal = self._continue_live_session(
                        session=trial_session,
                        base_shared_text=candidate_shared,
                        max_tokens=self.cfg.generation.max_final_answer_tokens,
                        budget_reason="slm_trial_final_answer_budget",
                    )
                    reason = f"slm_trial_{terminal}"
                else:
                    final_text = candidate_shared + trial_session.generated_text()
                    reason = f"slm_trial_{trial_outcome.reason}"
                trial_session.close()
                break

            if trial_outcome.kind == "rejected":
                trial_text = trial_session.generated_text()
                trial_decode_seconds, trial_probe_seconds = (
                    trial_session.generated_cost_slice()
                )
                ledger.mark_discarded_generation(
                    role="slm",
                    reason="rejected_trial",
                    tokens=len(trial_session.generated_ids),
                    characters=len(trial_text),
                    decode_seconds=trial_decode_seconds,
                    probe_seconds=trial_probe_seconds,
                )
                ledger.mark_wasted_prefill(
                    role="slm",
                    purpose=trial_session.purpose,
                    tokens=trial_session.prefill_info.tokens,
                    seconds=trial_session.prefill_info.seconds,
                )
                ledger.transition(
                    "slm_trial_rejected",
                    attempt=trial_attempts,
                    discarded_trial_tokens=len(trial_session.generated_ids),
                    discarded_trial_decode_seconds=trial_decode_seconds,
                    discarded_trial_probe_seconds=trial_probe_seconds,
                    wasted_trial_prefill_tokens=trial_session.prefill_info.tokens,
                )
                trial_session.close()
                if trial_attempts >= self.cfg.protocol.max_trial_attempts:
                    final_text, finish_reason = self._run_llm_finish(
                        example,
                        candidate_shared,
                        ledger,
                        purpose="llm_finish_after_trial_limit",
                    )
                    reason = finish_reason
                    break
                # No LLM KV snapshot was retained; continuation requires full re-prefill.
                repair_base = candidate_shared
                llm_session = self.llm.create_session(
                    question=example.question,
                    shared_text=repair_base,
                    control_text="",
                    purpose=f"llm_resume_after_trial_reject_{trial_attempts}",
                    ledger=ledger,
                    enable_grounding=self.cfg.protocol.llm_repair_monitor == "dual",
                )
                continue

            # Accepted trial: keep this live SLM cache and continue from the provisional blocks.
            ledger.transition(
                "slm_trial_accepted",
                attempt=trial_attempts,
                committed_trial_tokens=len(trial_session.generated_ids),
            )
            accepted_base = candidate_shared
            active_outcome = self._run_slm_active(
                session=trial_session,
                monitor=trial_outcome.monitor,
                ledger=ledger,
                max_tokens=self.cfg.generation.max_slm_active_tokens_after_handoff,
                phase="SLM_ACTIVE_AFTER_HANDOFF",
                honor_alarm=True,
            )
            if active_outcome.kind != "alarm":
                if active_outcome.reason == "thinking_end":
                    final_text, terminal = self._continue_live_session(
                        session=trial_session,
                        base_shared_text=accepted_base,
                        max_tokens=self.cfg.generation.max_final_answer_tokens,
                        budget_reason="post_handoff_final_answer_budget",
                    )
                    reason = f"post_handoff_slm_{terminal}"
                else:
                    final_text = accepted_base + trial_session.generated_text()
                    reason = f"post_handoff_slm_{active_outcome.reason}"
                trial_session.close()
                break

            if self.cfg.protocol.after_repair_budget == "keep_slm":
                tail = self._run_slm_active(
                    session=trial_session,
                    monitor=trial_outcome.monitor,
                    ledger=ledger,
                    max_tokens=self.cfg.generation.max_slm_active_tokens_after_handoff,
                    phase="SLM_ACTIVE_IGNORING_SECOND_ALARM",
                    honor_alarm=False,
                )
                if tail.reason == "thinking_end":
                    final_text, terminal = self._continue_live_session(
                        session=trial_session,
                        base_shared_text=accepted_base,
                        max_tokens=self.cfg.generation.max_final_answer_tokens,
                        budget_reason="second_alarm_keep_slm_final_answer_budget",
                    )
                    reason = f"second_alarm_keep_slm_{terminal}"
                else:
                    final_text = accepted_base + trial_session.generated_text()
                    reason = f"second_alarm_keep_slm_{tail.reason}"
                trial_session.close()
                break

            # A second alarm does not start another repair-handoff loop. Roll back once and let LLM finish.
            second_base, second_llm = self._rollback_and_start_llm(
                example=example,
                base_shared_text=accepted_base,
                outcome=active_outcome,
                ledger=ledger,
                repair_episode=repair_episodes + 1,
            )
            final_text, reason = self._continue_live_session(
                session=second_llm,
                base_shared_text=second_base,
                max_tokens=self.cfg.generation.max_llm_finish_tokens,
                budget_reason="llm_finish_after_second_alarm_budget",
            )
            second_llm.close()
            break

        if self.cfg.cost.record_cuda_memory:
            ledger.capture_gpu_memory(devices)
        return CoreRunResult(
            example_id=example.example_id,
            question=example.question,
            gold_answer=example.answer,
            final_text=final_text,
            terminal_reason=reason,
            repair_episodes=repair_episodes,
            trial_attempts=trial_attempts,
            cost=ledger.summary(),
        )
