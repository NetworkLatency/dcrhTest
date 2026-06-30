from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(slots=True)
class TauPoint:
    rho: float
    tau: float
    empirical_takeover_rate: float
    takeover_count: int
    sample_count: int

    @property
    def label(self) -> str:
        return f"tau_{int(round(self.rho * 100))}"

    def as_dict(self) -> dict:
        row = asdict(self)
        row["label"] = self.label
        return row


def tau_for_target_rate(scores: list[float], rho: float) -> TauPoint:
    if not scores:
        raise ValueError("scores must be non-empty")
    target = float(rho)
    if target <= 0.0 or target > 1.0:
        raise ValueError("rho must be in (0, 1]")

    ordered = sorted(float(score) for score in scores)
    n = len(ordered)
    index = min(n - 1, max(0, math.floor((1.0 - target) * n)))
    tau = ordered[index]
    takeover_count = sum(1 for score in ordered if score >= tau)
    return TauPoint(
        rho=target,
        tau=tau,
        empirical_takeover_rate=takeover_count / n,
        takeover_count=takeover_count,
        sample_count=n,
    )


def summarize_tau_scores(scores: Iterable[float], rhos: Iterable[float]) -> dict:
    values = [float(score) for score in scores]
    if not values:
        return {
            "sample_count": 0,
            "score_definition": "S(x)=max_i R_i",
            "scores": {},
            "tau_by_target_rate": [],
        }
    ordered = sorted(values)
    n = len(ordered)
    points = [tau_for_target_rate(ordered, rho).as_dict() for rho in rhos]
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    return {
        "sample_count": n,
        "score_definition": "S(x)=max_i R_i",
        "scores": {
            "min": ordered[0],
            "median": median,
            "max": ordered[-1],
        },
        "tau_by_target_rate": points,
    }
