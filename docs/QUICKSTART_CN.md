# MDRV Core 离线运行说明

当前项目默认方法是 MDRV。包名和脚本名仍保留 `dcrh` / `dcrh-run`，用于兼容已有运行环境。

## 1. 准备目录

离线服务器上建议准备：

```text
/work/models/Qwen3-1.7B/
/work/models/Qwen3-4B/
/work/data/evaluation.jsonl
/work/dcrh_core/
/work/wheelhouse/
```

模型目录必须包含本地 `config.json`、tokenizer、generation config 和权重文件。代码会使用 Hugging Face 离线模式和 `local_files_only=True`，不会主动下载模型或数据。

安装示例：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links /work/wheelhouse -r requirements.txt
pip install --no-index --no-deps /work/dcrh_core/dist/dcrh_core-0.1.0-py3-none-any.whl
```

## 2. 数据格式

通用 JSONL 每行一个样本：

```json
{"id":"math-0001","question":"题目文本","answer":"42"}
```

字段名可以在 YAML 中通过 `data.id_field`、`data.question_field`、`data.answer_field` 修改。AIME/GPQA 适配器可以通过 `data.dataset` 启用。

## 3. 修改配置

复制 `configs/qwen3_1p7b_4b_core.yaml`，至少修改：

```yaml
models:
  slm:
    path: /work/models/Qwen3-1.7B
    device: cuda:0
  llm:
    path: /work/models/Qwen3-4B
    device: cuda:1

data:
  path: /work/data/evaluation.jsonl

controller:
  alarm_threshold: 0.15

output:
  directory: /work/outputs/mdrv_core
```

`controller.alarm_threshold` 就是 MDRV 的 `tau`。第一版的边界信号在 replay 时直接计算，只需要在配置中设置这个阈值。

运行前检查：

```bash
python scripts/check_offline.py \
  --config /work/configs/evaluation.yaml
```

## 4. 核心实验

```bash
python scripts/run.py --config /work/configs/evaluation.yaml
```

中断后再次执行会按 `example_id` 跳过已完成样本。需要记录错误并继续时：

```bash
python scripts/run.py \
  --config /work/configs/evaluation.yaml \
  --continue-on-error
```

输出目录包含：

```text
resolved_config.yaml
run_metadata.json
results.jsonl
errors.jsonl
```

## 5. 基线与汇总

SLM-only / LLM-only：

```bash
python scripts/baseline.py \
  --config /work/configs/evaluation.yaml \
  --role slm \
  --max-tokens 16384

python scripts/baseline.py \
  --config /work/configs/evaluation.yaml \
  --role llm \
  --max-tokens 16384
```

汇总：

```bash
python scripts/summarize.py \
  --results /work/outputs/mdrv_core/results.jsonl \
  --output /work/outputs/mdrv_core/summary.json
```

## 6. 结果字段

每个样本会记录：

- `triggered`、`trigger_step`、`anchor_step`：MDRV 是否接管以及接管位置；
- `discarded_steps`：回滚后丢弃的 SLM chunk；
- `per_step`：每个边界的 TPM、route、drawdown、risk 和 attention 可用性；
- `cost.prefills`：真实 prefill 调用；
- `cost.counters.discarded_mdrv_rollback_suffix_tokens`：回滚丢弃的 SLM token；
- `cost.kv_policy: live_session_only_no_snapshots`：确认不保存边界 KV 快照。

## 7. 运行前建议

先用少量样本冒烟测试，确认：

1. 配置可以加载，模型和数据路径均为本地路径；
2. 结果中能看到 `"\n\n"` 边界对应的 `per_step`；
3. `controller.alarm_threshold` 已设置；
4. 如果 attention 不可用，`per_step` 中会记录原因，并回退到 TPM-only；
5. 输出目录可以断点续跑。
