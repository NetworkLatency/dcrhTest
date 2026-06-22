# Core state machine and accounting invariants

```text
SLM_ACTIVE
  | persistent H or G alarm
  | estimate onset and discard suffix
  v
LLM_REPAIR -- both LLM channels settle --> SLM_TRIAL
     ^                                      |
     | rejected trial; full LLM re-prefill  | two blocks safe
     +--------------------------------------+----> SLM_ACTIVE
```

## Invariants

1. Decisions occur only at a completed double-newline boundary.
2. Atomic boundaries store text/token offsets and scalar statistics, never KV tensors.
3. A model switch always creates a new full-prefix prefill.
4. The H and G CUSUM states are independent; no weighted or multiplicative cross-signal fusion is used.
5. Correctness labels are never used to define G, choose layers, choose sinks, or make online transitions.
6. Escalation is an OR event; return is an AND event.
7. Every generated token is counted even if later discarded.
8. Every rejected-trial prefill is counted as wasted prefill.
9. The published core permits one repair–handoff episode; later failure is delegated to LLM completion.
