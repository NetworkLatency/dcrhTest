# MDRV Core

This repository implements the current core method for a local
**Qwen/Qwen3-1.7B SLM + Qwen/Qwen3-4B LLM** pair. The Python package and CLI
names remain `dcrh`/`dcrh-run` for environment compatibility, but the default
controller is now **MDRV**: Margin Drawdown with Route Velocity.

There is no route training, answer-level online verifier, KV-cache transfer, or
per-boundary KV snapshot. A live generation session owns one transient cache;
model switches are implemented by full-prefix prefill.

## Method

1. The SLM generates an initial reasoning trace with the normal sampler.
2. The controller records completed `"\n\n"` boundaries. The delimiter token(s)
   are counted inside the current chunk `C`.
3. For each completed boundary, the SLM replays the effective pre-action prefix
   and computes TPM from full-vocabulary raw logits:

   ```text
   M_i = p_top1 - p_top2
   ```

   If whitespace, special tokens, or pure punctuation appear after the boundary,
   the effective prefix includes them and the TPM query position becomes the
   token immediately before the first content token.

4. The router maintains margin drawdown. Attention route is computed lazily only
   while drawdown is positive.
5. Route attention is taken from one full-attention layer and one query row. Raw
   attention mass is first normalized by region length, then normalized into a
   four-dimensional distribution over `A/O/P/C`.
6. If attention is unavailable, the route discount is skipped and the router
   falls back to TPM-only risk while logging `attention_unavailable`.
7. When risk reaches `controller.alarm_threshold` (`tau`), MDRV rolls back to
   the first low-margin boundary in the active drawdown segment. The LLM then
   continues from the trusted prefix with `prompt.takeover_cue`.

## Attention Layer Safeguard

The Transformers route backend never intentionally reads a sliding-window layer.
Layer selection follows this order:

1. If `config.layer_types` exists, choose the largest index whose value is
   `"full_attention"`.
2. Else if `use_sliding_window` is false or `sliding_window` is `None`, choose
   the last layer.
3. Else choose `min(num_hidden_layers, max_window_layers) - 1`.
4. If no full-attention layer can be identified, route attention is disabled for
   that model and MDRV uses TPM-only risk.

For the current Qwen3-1.7B and Qwen3-4B targets this resolves to the last layer.

## Configuration

Copy and edit:

```text
configs/qwen3_1p7b_4b_core.yaml
```

Important fields:

- `models.slm.path`, `models.llm.path`: absolute local model directories.
- `models.*.device`: usually `cuda:0` and `cuda:1`.
- `controller.alarm_threshold`: MDRV `tau`.
- `protocol.route_discount`: enable route-velocity risk discount.
- `protocol.takeover_mode`: `rollback` or `current`.
- `protocol.attention_route_mode`: `last_layer_single_row` or `none`.
- `signals.attention_head_chunk_size`: memory bound for route probe chunks.

The active schema is intentionally small: MDRV uses `controller.alarm_threshold`
as the routing threshold and computes boundary signals directly during replay.

## Run

```bash
python scripts/check_offline.py --config configs/qwen3_1p7b_4b_core.yaml
python scripts/run.py --config configs/qwen3_1p7b_4b_core.yaml
```

The command writes:

```text
resolved_config.yaml
run_metadata.json
results.jsonl
errors.jsonl
```

Use `--continue-on-error` to record per-sample failures and continue.

Single-model baselines use the same prompt builder, sampler, and cost ledger:

```bash
python scripts/baseline.py --config configs/qwen3_1p7b_4b_core.yaml --role slm
python scripts/baseline.py --config configs/qwen3_1p7b_4b_core.yaml --role llm
```

Summaries:

```bash
python scripts/summarize.py \
  --results /path/to/outputs/results.jsonl \
  --output /path/to/outputs/summary.json
```

## Output Signals

Each MDRV step records:

- TPM fields: `M_i`, `p_top1`, `p_top2`, top token ids/text.
- Boundary content-token metadata: `action_token_*` and skipped token count.
- Route fields: `r_i_A`, `r_i_O`, `r_i_P`, `r_i_C`, `V_i`, `barV`.
- Router fields: `G_i`, `segment_start`, `R_i`, takeover decision.
- Attention availability and selected route layer.

Cost accounting records real prefills, generated tokens, decode time, dense
attention work proxies, rollback-discarded SLM suffixes, and GPU memory peaks.

## Compatibility

- Python 3.10+
- PyTorch 2.4+
- `transformers==4.57.6`
- Dense local Qwen3 causal language models
- Batch size 1 in the current local runner

Set `DCRH_ALLOW_UNTESTED_TRANSFORMERS=1` only after validating a different
Transformers version, because the attention backend depends on internal
attention-interface signatures.
