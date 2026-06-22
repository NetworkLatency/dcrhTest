# DCRH Core: Dual-Channel Rollback–Handoff Collaboration

This package implements the first complete core experiment for a local **Qwen3-1.7B SLM + Qwen3-4B LLM** pair. It is deliberately narrow:

1. The SLM is the default reasoning model.
2. Predictive-entropy evidence and question-attention evidence are accumulated in two independent sequential channels.
3. A persistent alarm estimates an onset block and rolls the SLM text back to that natural boundary.
4. The LLM re-prefills the complete trusted text prefix and repairs it until its entropy channel settles by default.
5. The LLM cache is discarded. The SLM re-prefills the complete LLM prefix and generates two provisional statistical blocks.
6. A safe trial is committed; an unsafe trial is discarded and the LLM must re-prefill before resuming.

There is **no model-pair decision-surface training**, no answer-level online verifier, and no KV-cache transfer between models.

For the repository map and refactor boundaries, see `docs/PROJECT_STRUCTURE.md`.
The state-machine invariants live in `docs/DESIGN.md`; the Chinese offline
quickstart is `docs/QUICKSTART_CN.md`.

## What is implemented

- Strict offline loading with `local_files_only=True` and Hugging Face offline environment variables.
- Raw-logit top-k predictive entropy, evaluated before penalties, temperature, top-k, or top-p.
- Online question-attention statistic `G` from selected middle layers without materializing or retaining full attention matrices.
- Natural `"\n\n"` checkpoints plus a token-mass statistical clock.
- Unlabeled empirical rank calibration for each model.
- Independent Page-CUSUM-style channels for entropy instability and anchoring loss.
- Change-onset rollback.
- LLM entropy-monitored repair and two-block SLM speculative handoff.
- Optional runtime/probe configuration fields for the planned two-process vLLM worker path.
- Detailed recomputation, rollback, rejected-trial, prefill, decode, latency, memory, and work-proxy accounting.
- Resume-safe JSONL output.
- Explicit handling of the Qwen3 `</think>` transition: monitoring stops at the end of slow thinking and the same live model/cache completes the final answer.

## Deliberate memory policy

The implementation stores **text and scalar statistics only** at natural boundaries. It never stores a KV snapshot at each boundary.

A `GenerationSession` owns one live transient cache. When control moves between models, the old session is closed and the next model re-prefills the complete text prefix. In particular:

- SLM → LLM upgrade: full LLM prefill of `[prompt, trusted prefix]`;
- LLM → SLM trial: full SLM prefill of `[prompt, LLM-repaired prefix]`;
- rejected SLM trial → LLM resume: full LLM prefill again;
- accepted SLM trial: the live trial cache is retained only for immediate continuation, not as a checkpoint archive.

The result files state `"kv_policy": "live_session_only_no_snapshots"`.

## Signal definitions

### Predictive entropy

For the raw next-token logits, before any decoding transformation, the code takes the top `K=20` logits, renormalizes them, and computes normalized Shannon entropy. Within a statistical block it stores the mean and q90. Between adjacent blocks it computes log changes:

```text
d_mean = log((mean_t + eps) / (mean_{t-1} + eps))
d_q90  = log((q90_t  + eps) / (q90_{t-1}  + eps))
```

The two tail probabilities are combined by a Bonferroni union. They are not manually weighted.

### Structural question anchoring

During each one-token decode forward, the custom attention backend computes only the attention mass allocated to the original user-question token span in the selected full-attention layers. Special tokens and a configurable initial sink prefix are removed from the denominator. The question mass is corrected for the question's token share using a log-odds enrichment statistic:

```text
G = logit(question_attention / non_sink_attention)
    - logit(question_tokens / visible_non_sink_tokens)
```

The backend reduces this value immediately over heads and selected layers. It retains one scalar per generated token, not an attention tensor. At a statistical boundary, `G` is the mean over the last 32 generated tokens.

`G` is a structural context-use statistic. It is not an answer-correctness, hallucination, or causal-faithfulness score.

## Statistical clock

Every decoded `"\n\n"` creates an atomic text checkpoint. Several short atomic segments are merged until their cumulative generated-token count reaches `L0`; the next natural boundary closes a statistical block.

`L0` is estimated separately for each model as the median non-empty atomic-segment length on an unlabeled warmup subset. It is saved inside that model's reference file.

## Sequential controller

Each model has three empirical unlabeled distributions:

- entropy-mean log change;
- entropy-q90 log change;
- absolute `G`.

At runtime these give an upper-tail entropy p-value and a lower-tail anchoring p-value. Each channel uses the centered increment

```text
x = -log(p) - 1
```

and an unreflected cumulative sum whose distance from its historical minimum is the current Page-CUSUM score. The SLM escalates when either channel crosses the single threshold `B`. The historical minimum identifies the estimated onset block used for rollback.

The first statistical block initializes entropy and cannot trigger an escalation. Its anchoring increment is retained, so a later alarm can still estimate onset at block 1 and roll back to the prompt boundary.

## LLM repair and SLM trial

- **Escalation:** entropy alarm **OR** anchoring alarm.
- **LLM ready:** by default, the LLM entropy score has returned to zero after at least two blocks.
- **SLM trial accepted:** both SLM trial channel scores remain zero after exactly two statistical blocks.

Two SLM trial blocks are structural, not tuned: a dynamic entropy change cannot be defined from one block.

The released core fixes one repair–handoff episode. A later SLM alarm causes rollback followed by LLM completion, rather than another unbounded collaboration loop.

Set `protocol.llm_repair_monitor: dual` only for an ablation that also computes LLM-side `G`. Set `finish_directly` to skip the repair-settling handshake and let the LLM complete after rollback.

## vLLM worker path

The default implementation is still the local Transformers reference runner. The configuration now reserves the vLLM worker route:

```yaml
models:
  slm:
    backend: vllm_worker
    probe_mode: vllm_attention
    worker_endpoint: tcp://127.0.0.1:9101
  llm:
    backend: vllm_worker
    probe_mode: none
    worker_endpoint: tcp://127.0.0.1:9102
```

The intended architecture is two external `dcrh_vllm_worker` processes wrapping vLLM engines, plus the existing DCRH controller. Token text and raw-logit entropy travel through the worker stream. SLM-side `G` is either produced by a vLLM attention probe backend (`vllm_attention`) or by a slow Transformers replay validation lane (`transformers_replay`). LLM repair defaults to `probe_mode: none` because the default protocol uses H-only repair settling.

The repository currently includes the worker event protocol and vLLM logprob parsing utilities. The actual vLLM attention probe backend remains a separate implementation step; until it is attached, `vllm_attention` should be treated as a declared target mode rather than a supported production runner.

### Start a matched stock vLLM pair

For endpoint, latency, and future worker integration checks, `server/start_vllm_pair.sh` starts separate SLM and LLM OpenAI-compatible vLLM servers with DCRH-friendly defaults:

```bash
PYTHON_BIN=/path/to/vllm-env/bin/python \
SLM_MODEL=/models/Qwen3-1.7B \
LLM_MODEL=/models/Qwen3-4B \
SLM_CUDA_DEVICE=0 \
LLM_CUDA_DEVICE=1 \
SLM_PORT=8101 \
LLM_PORT=8102 \
bash server/start_vllm_pair.sh
```

Or copy `server/vllm_pair.env.example` to `server/vllm_pair.env`, edit local paths, then:

```bash
set -a
source server/vllm_pair.env
set +a
bash server/start_vllm_pair.sh
```

The script is modeled after the prior TestRou launcher: it creates logs under `server/vllm_logs`, checks `/health`, writes role-specific PID files, refuses unhealthy occupied ports unless `FORCE_RESTART=1`, and prints:

```text
SLM_BASE_URL=http://127.0.0.1:8101/v1
LLM_BASE_URL=http://127.0.0.1:8102/v1
```

By default it adds `--logprobs-mode raw_logits --max-logprobs 20`, because DCRH's vLLM path needs raw top-k values for entropy. This stock launcher does not expose the SLM attention `G` probe; that still requires the future `dcrh_vllm_worker` or vLLM probe backend side channel.

## Compatibility

The selective attention backend is pinned to:

- Python 3.10+
- PyTorch 2.4+
- `transformers==4.57.6`
- dense Qwen3 causal language models with selected `full_attention` layers
- batch size 1

The exact Transformers pin is intentional because the internal attention-interface signature is version-sensitive. Set `DCRH_ALLOW_UNTESTED_TRANSFORMERS=1` only after testing a different version.

## Offline installation

On a networked machine with a compatible CUDA/PyTorch setup, build a wheel directory:

```bash
python -m pip download -r requirements.txt -d wheelhouse
python -m pip wheel --no-deps . -w wheelhouse
```

Copy the project, the `wheelhouse` directory, both model directories, and the dataset to the offline server. Then install without an index:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links ./wheelhouse -r requirements.txt
pip install --no-index --no-deps ./dist/dcrh_core-0.1.0-py3-none-any.whl
```

For a CUDA server, ensure the copied PyTorch wheel matches the installed driver/CUDA environment. The package never downloads a model, tokenizer, config, generation config, or dataset.

## Local dataset formats

### Generic rows

```json
{"id":"sample-1","question":"...","answer":"42"}
```

Field names are configurable. `data.format: auto` infers `jsonl`, `json`, `csv`,
`tsv`, `parquet`, or `hf_disk` from the path when possible.

### AIME and GPQA adapters

Set `data.dataset` to one of:

- `aime24`
- `aime25`
- `gpqa`
- `gpqa_diamond`

For `aime24` and `aime25`, rows may use `problem`, `question`, or `prompt` for
the problem text, and `answer`, `solution`, or `target` for gold. For GPQA,
rows may either contain explicit `A`/`B`/`C`/`D` choices and a letter answer, or
the common `Correct Answer` plus `Incorrect Answer 1..3` fields. The latter is
rendered with the correct answer as option `A`, matching the prior TestRou
runner.

With `evaluation.verifier: auto`, AIME uses math-answer normalization and GPQA
uses choice-letter matching.

### Hugging Face dataset saved to disk

Set `data.format: hf_disk`, point `data.path` to a directory created with `save_to_disk`, and install the optional `datasets` dependency from the local wheelhouse.

Use disjoint calibration and evaluation files. Correct answers are not read by signal calibration, but reusing questions would contaminate evaluation protocol selection.

## Configuration

Copy and edit:

```text
configs/qwen3_1p7b_4b_core.yaml
```

All paths should be absolute local paths. A two-GPU setup is the default example:

- Qwen3-1.7B on `cuda:0`;
- Qwen3-4B on `cuda:1`.

Both models may share one sufficiently large GPU. The code does not currently implement model swapping from CPU/disk between states.

## Run the core experiment

### 1. Build the SLM reference and choose `B`

```bash
python scripts/calibrate.py \
  --config configs/qwen3_1p7b_4b_core.yaml \
  --role slm \
  --reference-out /local/references/qwen3_1p7b_reference.npz \
  --threshold-out /local/references/qwen3_1p7b_threshold.json
```

The calibration workflow uses three disjoint slices of the configured local data:

- warmup slice: estimate token mass;
- rank slice: estimate empirical signal distributions;
- budget slice: choose one scalar threshold from a target problem-level alarm rate.

### 2. Build the LLM reference

```bash
python scripts/calibrate.py \
  --config configs/qwen3_1p7b_4b_core.yaml \
  --role llm \
  --reference-out /local/references/qwen3_4b_reference.npz \
  --threshold-out /local/references/qwen3_4b_unused_threshold.json
```

For the LLM role, calibration skips the budget-threshold phase. The emitted JSON records `threshold: null`; the empirical reference and token mass are used for repair settling.

### 3. Run

Update the reference and threshold paths in the YAML, then run:

```bash
python scripts/run.py --config configs/qwen3_1p7b_4b_core.yaml
```

The command resumes by sample ID if `results.jsonl` already exists. Add `--continue-on-error` to record failures and continue.

### 4. Summarize

```bash
python scripts/summarize.py \
  --results /local/outputs/qwen3_1p7b_4b_core/results.jsonl \
  --output /local/outputs/qwen3_1p7b_4b_core/summary.json
```

### 5. Run matched local baselines

The package also provides SLM-only and LLM-only runs using the same raw-logit decoder and cost ledger, with the G probe disabled:

```bash
python scripts/baseline.py \
  --config configs/qwen3_1p7b_4b_core.yaml \
  --role slm \
  --max-tokens 16384

python scripts/baseline.py \
  --config configs/qwen3_1p7b_4b_core.yaml \
  --role llm \
  --max-tokens 16384
```

By default, outputs are written under `<output.directory>/slm_only` and `<output.directory>/llm_only`.

## Cost accounting

Each sample contains a complete `cost` object.

### Actual prefill records

Every prefill record includes:

- model role and purpose;
- total prefix tokens;
- base-prompt, shared-text, and control-token counts;
- synchronized wall time;
- estimated live K+V bytes at that prefix length.
- dense-attention QK-element upper bound for the prefill.

Purposes distinguish:

- `slm_initial`;
- `llm_upgrade_repair_*`;
- `slm_trial_*`;
- `llm_resume_after_trial_reject_*`;
- LLM completion fallbacks.

### Explicit wasted work

The counters include:

- `discarded_rollback_suffix_tokens`;
- `discarded_rejected_trial_tokens`;
- decode/probe seconds attributable to both discarded suffixes;
- `wasted_prefill_tokens` and seconds for rejected SLM trials;
- `rollback_slm_discarded_tokens`;
- actual LLM upgrade-prefill tokens;
- counterfactual LLM prefill tokens at the later detection point;
- LLM prefill tokens saved by rolling back earlier.

Thus the rollback trade-off is visible: an earlier rollback discards more already-generated SLM text but gives the LLM a shorter prefill.

### Work proxies

- `prefill_attention_qk_elements_upper_bound`;
- `decode_attention_qk_elements_upper_bound`;
- `probe_attention_qk_elements`, which counts the explicitly recomputed G-score elements;
- `max_estimated_probe_score_buffer_bytes`, based on one float32 head chunk.

The first two are dense-attention upper bounds rather than FLOP estimates. The probe counter follows the actual selected-layer/head/key dimensions. Use measured latency as the primary systems metric.

### Probe timing

Total prefill/decode times are synchronized when `cost.synchronize_cuda_for_timing: true`. Isolated `probe_seconds` is exact only when `signals.profile_probe_with_cuda_sync: true`; enabling that option synchronizes inside selected layers and materially perturbs throughput. For publication-quality overhead, compare complete runs with and without the probe backend rather than relying only on micro-timing.

## Output files

```text
resolved_config.yaml
run_metadata.json
results.jsonl
errors.jsonl             # only if a sample fails
```

Each result records the final text, terminal route, number of repair/trial events, verifier output, all controller transitions, all prefills, discarded work, timing, work proxies, and GPU peak allocation/reservation.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The included tests cover entropy, empirical tails, CUSUM onset, token-mass boundaries, waste accounting, the selective Qwen3 attention backend, and a local-only tiny random Qwen3 model. They do not establish the scientific validity of `H`, `G`, rollback, or handoff on real pretrained checkpoints.

## Scope and limitations

- Only open-weight local Qwen3 models are supported in the released core.
- Stable but incorrect and well-anchored SLM reasoning may never alarm.
- The LLM empirical reference is estimated on ordinary local problem prompts, while repair prefixes are a shifted distribution.
- Standard self-attention weights are used as structural allocation statistics, not causal explanations.
- The exact `"\n\n"` boundary is detected from decoded token pieces; models that do not emit paragraph boundaries will rarely be monitored.
- No batching, distributed serving engine, quantization, CPU model swapping, or black-box API LLM is implemented.
- The vLLM worker protocol is scaffolded, but the vLLM attention backend and IPC transport are not yet wired into `dcrh-run`.
- The generic boxed/numeric verifier is a convenience, not a substitute for benchmark-native evaluation.
- Safety token caps are engineering safeguards, not learned decision parameters.

The intended first experiment is not to claim a universal collaboration controller. It tests whether two functionally distinct internal signals can support a complete, training-free detection–rollback–repair–handoff loop while exposing all recomputation and discarded-work costs.
