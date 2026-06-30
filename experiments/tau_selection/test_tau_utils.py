from tau_utils import summarize_tau_scores, tau_for_target_rate


def test_tau_for_target_rate_uses_upper_tail_budget():
    point = tau_for_target_rate([0.1, 0.2, 0.3, 0.4, 0.5], rho=0.2)
    assert point.tau == 0.5
    assert point.takeover_count == 1
    assert point.empirical_takeover_rate == 0.2


def test_tau_summary_labels_budget_points():
    summary = summarize_tau_scores([0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.4])
    labels = [row["label"] for row in summary["tau_by_target_rate"]]
    assert labels == ["tau_20", "tau_40"]
    assert summary["score_definition"] == "S(x)=max_i R_i"
