from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .boundaries import (
    AtomicBoundary,
    DoubleNewlineBoundaryDetector,
    StatisticalBlock,
    StatisticalBlockBuilder,
    TokenObservation,
)
from .reference import EmpiricalReference
from .sequential import ControllerUpdate, DualChannelController

if TYPE_CHECKING:
    from ..runtime.transformers.model_runner import GenerationSession


@dataclass(slots=True)
class MonitorEvent:
    kind: str
    observation: TokenObservation | None = None
    atomic_boundary: AtomicBoundary | None = None
    block: StatisticalBlock | None = None
    update: ControllerUpdate | None = None
    end_reason: str | None = None


class SessionMonitor:
    def __init__(
        self,
        session: "GenerationSession",
        token_mass: int,
        entropy_quantile: float,
        grounding_window: int,
        reference: EmpiricalReference | None = None,
        threshold: float | None = None,
        pvalue_epsilon: float = 1e-6,
        use_entropy_channel: bool = True,
        use_grounding_channel: bool = True,
        require_grounding: bool | None = None,
    ) -> None:
        self.session = session
        self.use_entropy_channel = bool(use_entropy_channel)
        self.use_grounding_channel = bool(use_grounding_channel)
        if require_grounding is None:
            require_grounding = self.use_grounding_channel
        self.detector = DoubleNewlineBoundaryDetector()
        self.builder = StatisticalBlockBuilder(
            token_mass=token_mass,
            entropy_quantile=entropy_quantile,
            grounding_window=grounding_window,
            require_grounding=bool(require_grounding),
        )
        self.atomic_lengths: list[int] = []
        self.blocks: list[StatisticalBlock] = []
        if reference is None:
            self.controller = None
        else:
            if threshold is None:
                raise ValueError("threshold is required when a reference is supplied")
            self.controller = DualChannelController(
                reference=reference,
                threshold=threshold,
                pvalue_epsilon=pvalue_epsilon,
                use_entropy=self.use_entropy_channel,
                use_grounding=self.use_grounding_channel,
            )

    def step(self, allow_alarm: bool = True) -> MonitorEvent:
        result = self.session.step()
        observation = result.observation
        if result.ended:
            # EOS/max-context has no complete G observation and is not used to close a block.
            return MonitorEvent(
                kind="end",
                observation=observation,
                end_reason=result.end_reason,
            )

        self.builder.add(observation)
        token_index = len(self.session.generated_ids)
        boundary = self.detector.push(observation.text_piece, token_index)
        if boundary is None:
            return MonitorEvent(kind="token", observation=observation)

        self.atomic_lengths.append(boundary.atomic_length)
        block = self.builder.on_atomic_boundary(boundary.token_index)
        if block is None:
            return MonitorEvent(
                kind="atomic_boundary",
                observation=observation,
                atomic_boundary=boundary,
            )

        self.blocks.append(block)
        update = None
        if self.controller is not None:
            update = self.controller.update(block, allow_alarm=allow_alarm)
        return MonitorEvent(
            kind="block",
            observation=observation,
            atomic_boundary=boundary,
            block=block,
            update=update,
        )

    def rollback_token_index(self, onset_block: int) -> int:
        for block in self.blocks:
            if block.block_index == onset_block:
                return block.start_token_index
        raise KeyError(f"No block with index {onset_block}")
