from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class EmpiricalReference:
    delta_mean: np.ndarray
    delta_quantile: np.ndarray
    grounding: np.ndarray
    token_mass: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.delta_mean = np.sort(np.asarray(self.delta_mean, dtype=np.float64))
        self.delta_quantile = np.sort(np.asarray(self.delta_quantile, dtype=np.float64))
        self.grounding = np.sort(np.asarray(self.grounding, dtype=np.float64))
        if self.delta_mean.size == 0 or self.delta_quantile.size == 0:
            raise ValueError("Entropy reference distributions must be non-empty")
        if self.token_mass < 1:
            raise ValueError("Reference token_mass must be positive")

    @staticmethod
    def _upper_tail(sorted_values: np.ndarray, value: float) -> float:
        n = int(sorted_values.size)
        first_ge = int(np.searchsorted(sorted_values, value, side="left"))
        count_ge = n - first_ge
        return (1.0 + count_ge) / (n + 1.0)

    @staticmethod
    def _lower_tail(sorted_values: np.ndarray, value: float) -> float:
        n = int(sorted_values.size)
        count_le = int(np.searchsorted(sorted_values, value, side="right"))
        return (1.0 + count_le) / (n + 1.0)

    def entropy_pvalue(self, delta_mean: float, delta_quantile: float) -> tuple[float, float, float]:
        p_mean = self._upper_tail(self.delta_mean, delta_mean)
        p_quantile = self._upper_tail(self.delta_quantile, delta_quantile)
        p_combined = min(1.0, 2.0 * min(p_mean, p_quantile))
        return p_mean, p_quantile, p_combined

    def grounding_pvalue(self, grounding: float) -> float:
        if self.grounding.size == 0:
            raise RuntimeError("This empirical reference has no grounding distribution")
        return self._lower_tail(self.grounding, grounding)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(self.metadata, ensure_ascii=False)
        np.savez_compressed(
            path,
            delta_mean=self.delta_mean,
            delta_quantile=self.delta_quantile,
            grounding=self.grounding,
            token_mass=np.asarray([self.token_mass], dtype=np.int64),
            metadata=np.asarray([metadata_json]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmpiricalReference":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"][0]))
            return cls(
                delta_mean=data["delta_mean"],
                delta_quantile=data["delta_quantile"],
                grounding=data["grounding"],
                token_mass=int(data["token_mass"][0]),
                metadata=metadata,
            )
