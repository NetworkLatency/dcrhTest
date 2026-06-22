# Project Structure

This repository keeps the algorithmic controller separate from runtime backends,
benchmark adapters, and command entrypoints. New code should import from the
subpackages shown below rather than relying on root-level module aliases.

## Top-level directories

```text
configs/    Example experiment configuration files.
docs/       Design notes, quickstarts, and structure documents.
examples/   Tiny local dataset examples.
scripts/    Thin command wrappers for local source-tree execution.
server/     Service launchers and runtime environment templates.
src/dcrh/   Python package.
tests/      Unit tests.
```

Generated directories such as `dist/`, `__pycache__/`, `.pytest_cache/`,
`outputs/`, `references/`, and `server/vllm_logs/` are excluded by `.gitignore`.

## Python package layers

```text
src/dcrh/
  config.py              Experiment config dataclasses and validation.
  prompt.py              Chat prompt rendering shared by runtimes.

  commands/
    run.py               Core experiment command.
    calibrate.py         Reference and threshold calibration command.
    baseline.py          Single-model baseline command.
    summarize.py         Result summary command.
    threshold.py         Threshold extraction helper.

  core/
    protocol.py          DCRH state machine: rollback, repair, handoff.
    monitor.py           Boundary detector + statistical block monitor.
    sequential.py        H/G CUSUM state and channel updates.
    boundaries.py        Atomic newline boundaries and token-mass blocks.
    reference.py         Empirical H/G reference distributions.
    calibration.py       Unlabeled reference and threshold construction.
    costs.py             Cost and timing ledger.

  evaluation/
    data.py              Generic local dataset loader.
    benchmarks.py        AIME/GPQA prompt and gold adapters.
    verifier.py          Answer checking utilities.

  runtime/
    transformers/
      model_runner.py    Current local Transformers Qwen3 runner.
      attention_backend.py
                          Current Transformers SDPA grounding hook.
      entropy.py         Raw-logit entropy utility.
      sampling.py        Local sampling policy.
    vllm/
      protocol.py        dcrh_vllm_worker message schema.
      logprobs.py        vLLM raw top-k entropy parsing helpers.

  utils/
    io.py                JSON/JSONL and directory helpers.
    offline.py           Hugging Face offline environment helpers.
```

The root package intentionally stays small: shared configuration and prompt
rendering live at the top level, while controller, evaluation, runtime, command,
and utility code live in their own subpackages.

## Runtime modes

- `transformers`: current end-to-end runnable reference path.
- `vllm_worker`: planned two-process worker path.
- `transformers_replay`: planned slow validation lane for vLLM-generated text.
- `vllm_attention`: planned fast SLM-side G probe inside vLLM attention.

The stock launcher in `server/start_vllm_pair.sh` starts two ordinary vLLM
OpenAI-compatible services for endpoint and latency checks. It does not expose
the SLM attention `G` side channel by itself.

## Refactor boundary

Do not put CUSUM, rollback, SLM/LLM state transitions, benchmark verification,
or result aggregation inside vLLM worker code. The worker layer should stream
token observations and optional probe scalars; the controller remains in
`dcrh.core.protocol` and `dcrh.core.monitor`.
