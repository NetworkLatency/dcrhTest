from dcrh.core.costs import CostLedger


def test_explicit_mdrv_discard_accounting():
    ledger = CostLedger()
    ledger.record_prefill(
        "slm", "slm_mdrv_initial", 100, 10, 90, 0, 1.0, 4096, 123456
    )
    ledger.mark_discarded_generation(
        "slm",
        "mdrv_rollback_suffix",
        20,
        characters=40,
        decode_seconds=2.0,
    )
    summary = ledger.summary()
    assert summary["counters"]["discarded_generated_tokens"] == 20
    assert summary["counters"]["discarded_mdrv_rollback_suffix_tokens"] == 20
    assert summary["counters"]["discarded_mdrv_rollback_suffix_characters"] == 40
    assert summary["counters"]["discarded_mdrv_rollback_suffix_decode_seconds"] == 2.0
    assert summary["kv_policy"] == "live_session_only_no_snapshots"
