from dcrh.core.router_state import MdrvRouterState


def test_margin_drawdown_rises_on_consecutive_margin_losses():
    router = MdrvRouterState(tau=1.0)
    first = router.update_margin(1, 0.8)
    second = router.update_margin(2, 0.6)
    third = router.update_margin(3, 0.4)
    assert first.margin_drawdown == 0.0
    assert second.margin_drawdown > 0.0
    assert third.margin_drawdown > second.margin_drawdown


def test_margin_drawdown_recovers_to_zero():
    router = MdrvRouterState(tau=1.0)
    router.update_margin(1, 0.8)
    router.update_margin(2, 0.6)
    third = router.update_margin(3, 0.9)
    assert third.margin_drawdown == 0.0
    assert third.needs_attention is False


def test_anchor_is_first_low_margin_boundary_not_last_reset():
    router = MdrvRouterState(tau=0.1)
    router.update_margin(1, 0.8)
    update = router.update_margin(2, 0.6)
    assert update.margin_drawdown > 0.0
    assert update.segment_start == 2
    assert update.needs_baseline_route


def test_tpm_only_risk_equals_margin_drawdown():
    router = MdrvRouterState(tau=0.1, route_discount=True)
    router.update_margin(1, 0.8)
    update = router.update_margin(2, 0.6)
    _, bar_v, risk = router.route_discounted_risk(
        route=None,
        baseline_route=None,
        attention_available=False,
    )
    assert update.margin_drawdown == risk
    assert bar_v is None
