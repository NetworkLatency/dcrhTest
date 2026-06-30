# Project Structure

The repository is organized around a small MDRV controller, local runtime
backends, benchmark utilities, and command entrypoints.

## Top Level

```text
configs/    Example MDRV experiment configuration.
docs/       Design notes, quickstart, and structure guide.
examples/   Tiny local dataset examples.
experiments/ Local experiment utilities that are not part of the package API.
scripts/    Thin wrappers for source-tree execution.
server/     vLLM service launcher templates for future worker integration.
src/dcrh/   Python package.
tests/      Unit tests.
```

Generated directories such as `dist/`, `__pycache__/`, `.pytest_cache/`,
`outputs/`, and `server/vllm_logs/` are excluded by `.gitignore`.

## Python Package

```text
src/dcrh/
  config.py              Experiment config dataclasses and validation.
  prompt.py              Chat prompt rendering and prompt token accounting.

  commands/
    run.py               MDRV experiment command.
    baseline.py          Single-model baseline command.
    summarize.py         JSONL result summary command.

  core/
    protocol.py          Offline MDRV trace replay, rollback, and takeover.
    scoring.py           SLM trace generation and boundary scoring.
    router_signals.py    TPM, content-token filtering, A/O/P/C routes.
    router_state.py      Margin drawdown and route-velocity risk state.
    boundaries.py        Double-newline boundary detector.
    costs.py             Prefill, decode, discard, and memory ledger.

  evaluation/
    data.py              Generic local dataset loader.
    benchmarks.py        AIME/GPQA prompt and gold adapters.
    verifier.py          Answer checking utilities.

  runtime/
    transformers/
      model_runner.py    Local Qwen3 runner and boundary replay interface.
      attention_backend.py
                          Transformers SDPA route-collector backend.
      sampling.py        Local sampling policy.
    vllm/
      protocol.py        Reserved worker message schema.
      logprobs.py        vLLM raw score parsing helpers.

  utils/
    io.py                JSON/JSONL and directory helpers.
    offline.py           Hugging Face offline environment helpers.
```

## Ownership Boundary

- `core/` owns MDRV decisions and result semantics.
- `runtime/` owns model execution and optional signal collection.
- `evaluation/` owns data adaptation and final answer checking only.
- `commands/` wire configuration, iteration, output files, and error handling.

New routing logic should live in `core/`; backend-specific tensor work should
stay in `runtime/`.

Tau-selection and other paper-specific utilities should live under
`experiments/` so they can be managed or removed independently from the package.
Current local experiment utilities include `experiments/tau_selection/` and
`experiments/calibration_dataset/`.
