from dcrh.core.costs import CostLedger


def test_explicit_waste_accounting():
    ledger = CostLedger()
    ledger.record_prefill(
        "slm", "slm_trial_1", 100, 10, 90, 0, 1.0, 2000, 123456
    )
    ledger.mark_wasted_prefill("slm", "slm_trial_1", 100, 1.0)
    ledger.mark_discarded_generation(
        "slm", "rejected_trial", 20, decode_seconds=2.0, probe_seconds=0.5
    )
    summary = ledger.summary()
    assert summary["counters"]["wasted_prefill_tokens"] == 100
    assert summary["counters"]["discarded_rejected_trial_tokens"] == 20
    assert summary["counters"]["discarded_rejected_trial_decode_seconds"] == 2.0
    assert summary["kv_policy"] == "live_session_only_no_snapshots"
