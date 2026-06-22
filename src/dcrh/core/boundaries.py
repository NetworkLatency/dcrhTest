from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AtomicBoundary:
    token_index: int
    atomic_length: int


class DoubleNewlineBoundaryDetector:
    """Detects the start of a newline run of length >=2 from decoded token pieces."""

    def __init__(self) -> None:
        self._newline_run = 0
        self._emitted_in_current_run = False
        self._last_boundary_token = 0

    def push(self, token_piece: str, token_index: int) -> AtomicBoundary | None:
        emitted = False
        for char in token_piece:
            if char == "\n":
                self._newline_run += 1
                if self._newline_run >= 2 and not self._emitted_in_current_run:
                    emitted = True
                    self._emitted_in_current_run = True
            else:
                self._newline_run = 0
                self._emitted_in_current_run = False
        if not emitted:
            return None
        length = token_index - self._last_boundary_token
        self._last_boundary_token = token_index
        return AtomicBoundary(token_index=token_index, atomic_length=length)


@dataclass(slots=True)
class TokenObservation:
    token_id: int
    entropy: float
    grounding: float | None
    text_piece: str
    decode_seconds: float
    probe_seconds: float
    sequence_length_after_token: int


@dataclass(slots=True)
class StatisticalBlock:
    block_index: int
    start_token_index: int
    end_token_index: int
    token_count: int
    entropy_mean: float
    entropy_quantile: float
    grounding: float | None
    atomic_segments: int


class StatisticalBlockBuilder:
    """Closes a block only at a natural boundary after token mass reaches L0."""

    def __init__(
        self,
        token_mass: int,
        entropy_quantile: float,
        grounding_window: int,
        require_grounding: bool = True,
    ) -> None:
        if token_mass < 1:
            raise ValueError("token_mass must be positive")
        self.token_mass = int(token_mass)
        self.entropy_quantile_level = float(entropy_quantile)
        self.grounding_window = int(grounding_window)
        self.require_grounding = bool(require_grounding)
        self._block_start = 0
        self._observations: list[TokenObservation] = []
        self._atomic_segments = 0
        self._next_block_index = 1

    @property
    def pending_tokens(self) -> int:
        return len(self._observations)

    @property
    def block_start_token_index(self) -> int:
        return self._block_start

    def add(self, observation: TokenObservation) -> None:
        self._observations.append(observation)

    def on_atomic_boundary(self, end_token_index: int) -> StatisticalBlock | None:
        self._atomic_segments += 1
        if len(self._observations) < self.token_mass:
            return None
        if end_token_index != self._block_start + len(self._observations):
            raise RuntimeError(
                "Boundary token index is inconsistent with the current statistical block"
            )
        block = self._close(end_token_index)
        return block

    def _close(self, end_token_index: int) -> StatisticalBlock:
        import numpy as np

        entropies = np.asarray([x.entropy for x in self._observations], dtype=np.float64)
        grounding_values = [
            x.grounding for x in self._observations if x.grounding is not None
        ]
        if not grounding_values and self.require_grounding:
            raise RuntimeError("A statistical block contains no grounding observations")
        grounding_tail = grounding_values[-self.grounding_window :]
        grounding = (
            float(np.mean(np.asarray(grounding_tail, dtype=np.float64)))
            if grounding_tail
            else None
        )
        block = StatisticalBlock(
            block_index=self._next_block_index,
            start_token_index=self._block_start,
            end_token_index=end_token_index,
            token_count=len(self._observations),
            entropy_mean=float(entropies.mean()),
            entropy_quantile=float(
                np.quantile(entropies, self.entropy_quantile_level, method="linear")
            ),
            grounding=grounding,
            atomic_segments=self._atomic_segments,
        )
        self._next_block_index += 1
        self._block_start = end_token_index
        self._observations = []
        self._atomic_segments = 0
        return block
