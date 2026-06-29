from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AtomicBoundary:
    token_index: int
    atomic_length: int


class DoubleNewlineBoundaryDetector:
    """Detects a newline run of length >=2 from decoded token pieces.

    The emitted token index includes the token that completes the delimiter, so
    the first MDRV implementation treats the trailing "\n\n" as part of the
    current chunk.
    """

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
