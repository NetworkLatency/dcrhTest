from __future__ import annotations

import math
from dataclasses import dataclass

from .boundaries import StatisticalBlock
from .reference import EmpiricalReference


@dataclass(slots=True)
class CUSUMState:
    cumulative: float = 0.0
    minimum: float = 0.0
    minimum_index: int = 0
    score: float = 0.0

    def update(self, increment: float, block_index: int) -> float:
        self.cumulative += float(increment)
        if self.cumulative < self.minimum:
            self.minimum = self.cumulative
            self.minimum_index = int(block_index)
        self.score = max(0.0, self.cumulative - self.minimum)
        return self.score

    @property
    def onset_block(self) -> int:
        return self.minimum_index + 1


@dataclass(slots=True)
class ControllerUpdate:
    block_index: int
    delta_mean: float | None
    delta_quantile: float | None
    p_mean: float | None
    p_quantile: float | None
    p_entropy: float | None
    p_grounding: float | None
    entropy_increment: float | None
    grounding_increment: float | None
    entropy_score: float
    grounding_score: float
    alarm: bool
    trigger: str | None
    onset_block: int | None


class DualChannelController:
    def __init__(
        self,
        reference: EmpiricalReference,
        threshold: float,
        pvalue_epsilon: float = 1e-6,
        use_entropy: bool = True,
        use_grounding: bool = True,
    ) -> None:
        if not use_entropy and not use_grounding:
            raise ValueError("At least one controller channel must be enabled")
        self.reference = reference
        self.threshold = float(threshold)
        self.pvalue_epsilon = float(pvalue_epsilon)
        self.use_entropy = bool(use_entropy)
        self.use_grounding = bool(use_grounding)
        self.entropy = CUSUMState()
        self.grounding = CUSUMState()
        self.previous_mean: float | None = None
        self.previous_quantile: float | None = None
        self.blocks_seen = 0

    def reset(self) -> None:
        self.entropy = CUSUMState()
        self.grounding = CUSUMState()
        self.previous_mean = None
        self.previous_quantile = None
        self.blocks_seen = 0

    def _increment(self, pvalue: float) -> float:
        p = min(1.0, max(self.pvalue_epsilon, float(pvalue)))
        return -math.log(p) - 1.0

    def update(self, block: StatisticalBlock, allow_alarm: bool = True) -> ControllerUpdate:
        self.blocks_seen += 1
        p_grounding = None
        grounding_increment = None
        if self.use_grounding:
            if block.grounding is None:
                raise RuntimeError("Grounding channel is enabled but the block has no G value")
            p_grounding = self.reference.grounding_pvalue(block.grounding)
            grounding_increment = self._increment(p_grounding)
            grounding_score = self.grounding.update(
                grounding_increment, block_index=block.block_index
            )
        else:
            grounding_score = self.grounding.score

        delta_mean = None
        delta_quantile = None
        p_mean = None
        p_quantile = None
        p_entropy = None
        entropy_increment = None

        if (
            self.use_entropy
            and self.previous_mean is not None
            and self.previous_quantile is not None
        ):
            eps = self.pvalue_epsilon
            delta_mean = math.log((block.entropy_mean + eps) / (self.previous_mean + eps))
            delta_quantile = math.log(
                (block.entropy_quantile + eps) / (self.previous_quantile + eps)
            )
            p_mean, p_quantile, p_entropy = self.reference.entropy_pvalue(
                delta_mean, delta_quantile
            )
            entropy_increment = self._increment(p_entropy)
            entropy_score = self.entropy.update(
                entropy_increment, block_index=block.block_index
            )
        else:
            entropy_score = self.entropy.score

        self.previous_mean = block.entropy_mean
        self.previous_quantile = block.entropy_quantile

        # The first statistical block is always cold-start only.
        can_alarm = allow_alarm and self.blocks_seen >= 2
        h_alarm = self.use_entropy and can_alarm and entropy_score >= self.threshold
        g_alarm = self.use_grounding and can_alarm and grounding_score >= self.threshold
        alarm = h_alarm or g_alarm
        trigger: str | None = None
        onset: int | None = None
        if alarm:
            if h_alarm and g_alarm:
                trigger = "HG"
                onset = min(self.entropy.onset_block, self.grounding.onset_block)
            elif h_alarm:
                trigger = "H"
                onset = self.entropy.onset_block
            else:
                trigger = "G"
                onset = self.grounding.onset_block

        return ControllerUpdate(
            block_index=block.block_index,
            delta_mean=delta_mean,
            delta_quantile=delta_quantile,
            p_mean=p_mean,
            p_quantile=p_quantile,
            p_entropy=p_entropy,
            p_grounding=p_grounding,
            entropy_increment=entropy_increment,
            grounding_increment=grounding_increment,
            entropy_score=entropy_score,
            grounding_score=grounding_score,
            alarm=alarm,
            trigger=trigger,
            onset_block=onset,
        )

    def ready(self, minimum_blocks: int = 2, epsilon: float = 1e-8) -> bool:
        entropy_ready = (not self.use_entropy) or self.entropy.score <= epsilon
        grounding_ready = (not self.use_grounding) or self.grounding.score <= epsilon
        return (
            self.blocks_seen >= minimum_blocks
            and entropy_ready
            and grounding_ready
        )
