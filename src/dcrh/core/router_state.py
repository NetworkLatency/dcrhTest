from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .router_signals import REGION_NAMES, TpmMargin, compute_jsd_route_velocity


@dataclass(slots=True)
class MdrvStepRecord:
    step_index: int
    step_text: str
    prefix_token_len: int
    action_token_offset: int | None
    skipped_action_tokens: int
    action_token_id: int | None
    action_token_text: str | None
    M_i: float
    p_top1: float
    p_top2: float
    top1_token_id: int
    top1_token_text: str | None
    top2_token_id: int
    top2_token_text: str | None
    r_i_A: float | None
    r_i_O: float | None
    r_i_P: float | None
    r_i_C: float | None
    V_i: float | None
    G_i: float
    segment_start: int | None
    barV: float | None
    R_i: float
    attention_available: bool
    attention_unavailable_reason: str | None
    route_layer: int | None
    route_query_position: int | None
    takeover_decision_at_step: bool


@dataclass(slots=True)
class MarginUpdate:
    step_index: int
    margin_drawdown: float
    segment_start: int | None
    needs_attention: bool
    needs_baseline_route: bool


class MdrvRouterState:
    """Margin drawdown router with lazy route-velocity discount."""

    def __init__(self, tau: float, route_discount: bool = True) -> None:
        self.tau = float(tau)
        self.route_discount = bool(route_discount)
        self.previous_margin: float | None = None
        self.margin_drawdown = 0.0
        self.last_reset_step = 0
        self.active_start_step: int | None = None
        self.route_velocities: list[float] = []
        self.previous_route: dict[str, float] | None = None

    def update_margin(self, step_index: int, tpm_margin: float) -> MarginUpdate:
        step = int(step_index)
        margin = float(tpm_margin)
        previous_drawdown = self.margin_drawdown

        if self.previous_margin is None:
            self.margin_drawdown = 0.0
        else:
            self.margin_drawdown = max(
                0.0,
                self.margin_drawdown + self.previous_margin - margin,
            )
        self.previous_margin = margin

        if self.margin_drawdown == 0.0:
            self.last_reset_step = step
            self.active_start_step = None
            self.route_velocities = []
            self.previous_route = None
            return MarginUpdate(
                step_index=step,
                margin_drawdown=0.0,
                segment_start=None,
                needs_attention=False,
                needs_baseline_route=False,
            )

        entering = previous_drawdown == 0.0
        if entering:
            self.active_start_step = step
            self.route_velocities = []
            self.previous_route = None

        return MarginUpdate(
            step_index=step,
            margin_drawdown=self.margin_drawdown,
            segment_start=self.active_start_step,
            needs_attention=True,
            needs_baseline_route=entering,
        )

    def route_discounted_risk(
        self,
        route: Mapping[str, float] | None,
        baseline_route: Mapping[str, float] | None = None,
        attention_available: bool = True,
    ) -> tuple[float | None, float | None, float]:
        if self.margin_drawdown == 0.0:
            return None, None, 0.0
        if not self.route_discount or not attention_available or route is None:
            return None, None, self.margin_drawdown

        previous = baseline_route if baseline_route is not None else self.previous_route
        if previous is None:
            self.previous_route = {name: float(route[name]) for name in REGION_NAMES}
            return None, None, self.margin_drawdown

        route_velocity = compute_jsd_route_velocity(route, previous)
        self.route_velocities.append(route_velocity)
        self.previous_route = {name: float(route[name]) for name in REGION_NAMES}
        bar_v = sum(self.route_velocities) / len(self.route_velocities)
        risk = self.margin_drawdown * (1.0 - bar_v)
        return route_velocity, bar_v, risk


def tpm_fields(tpm: TpmMargin) -> dict:
    return {
        "M_i": tpm.tpm_margin,
        "p_top1": tpm.p_top1,
        "p_top2": tpm.p_top2,
        "top1_token_id": tpm.top1_token_id,
        "top1_token_text": tpm.top1_token_text,
        "top2_token_id": tpm.top2_token_id,
        "top2_token_text": tpm.top2_token_text,
    }
