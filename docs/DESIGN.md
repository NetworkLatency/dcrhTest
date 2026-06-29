# MDRV Design Notes

```text
SLM trace
  |
  | replay completed "\n\n" boundaries
  v
TPM margin drawdown
  |
  | drawdown > 0
  v
optional A/O/P/C route velocity discount
  |
  | risk >= tau
  v
rollback to active segment anchor -> LLM takeover
```

## Invariants

1. Controller decisions are made only at completed `"\n\n"` boundaries.
2. The delimiter is part of the current chunk `C` in this first implementation.
3. The main TPM query position is the effective pre-action state, not the first
   generated content token after the boundary.
4. Effective pre-action state skips over post-boundary whitespace, special
   tokens, and pure punctuation before the first content token.
5. TPM uses full-vocabulary raw logits and computes `p_top1 - p_top2`.
6. Attention route is lazy: it is attempted only when margin drawdown is
   positive.
7. Route attention is never computed from a layer identified as sliding-window.
8. Route mass is region-length normalized before becoming an `A/O/P/C`
   distribution.
9. The attention query self-position is excluded from route regions.
10. If route attention is unavailable, risk falls back to TPM-only drawdown and
    the step records the unavailability reason.
11. The rollback anchor is the first low-margin boundary in the active drawdown
    segment.
12. A model switch creates a fresh full-prefix prefill. No boundary KV snapshot
    is stored or transferred.
13. Correct answers are used only by the offline verifier after generation, not
    by routing decisions.

## Region Semantics

At boundary `i`, generated chunks are partitioned as:

- `A`: base prompt.
- `O`: older reasoning chunks before the previous chunk.
- `P`: previous chunk.
- `C`: current chunk, including the closing `"\n\n"` delimiter.

For the first and second chunks, missing `O` or `P` regions are represented as
empty spans. Empty regions receive zero route mass.

## Fallback Behavior

MDRV remains runnable when attention route is disabled or unavailable. In that
case:

```text
R_i = G_i
```

where `G_i` is margin drawdown. The result row still logs TPM, drawdown, and
`attention_available=false`.
