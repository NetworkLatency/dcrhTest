# Tau Selection Utility

This folder is an experiment-side utility, not part of the public `dcrh`
package API.

It estimates MDRV routing thresholds from unlabeled SLM traces:

```text
S(x) = max_i R_i
Pr(S(x) >= tau) ~= rho
```

Run from the repository root:

```bash
python experiments/tau_selection/score_tau.py \
  --config configs/qwen3_1p7b_4b_core.yaml \
  --rho 0.1 0.2 0.3
```

Outputs:

```text
<output.directory>/tau_score/results.jsonl
<output.directory>/tau_score/tau_summary.json
```

Use `--no-per-step` to keep only per-sample max scores and reduce log size.

This tool imports the core `MdrvTraceScorer` so it stays aligned with the main
router implementation, but the CLI and tau-budget reporting live here on
purpose and can be excluded from a future public release.
