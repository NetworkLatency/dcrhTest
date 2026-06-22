# DCRH Core 离线运行说明

该版本只实现第一轮核心实验：Qwen3-1.7B 作为 SLM，Qwen3-4B 作为 LLM；模型均冻结，不训练路由器。系统只保存文本、token offset 和标量统计，不保存边界 KV 快照。

## 1. 环境与目录

建议在有网络的机器准备依赖 wheel，然后整体复制到离线服务器。离线服务器上的目录可以类似：

```text
/work/models/Qwen3-1.7B/
/work/models/Qwen3-4B/
/work/data/calibration.jsonl
/work/data/evaluation.jsonl
/work/dcrh_core/
/work/wheelhouse/
```

模型目录必须包含本地 `config.json`、tokenizer、generation config 和权重分片。代码会设置 Hugging Face 离线环境变量，并对模型/tokenizer 使用 `local_files_only=True`。

安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links /work/wheelhouse -r requirements.txt
pip install --no-index --no-deps /work/dcrh_core/dist/dcrh_core-0.1.0-py3-none-any.whl
```

运行前检查：

```bash
python scripts/check_offline.py \
  --config /work/configs/evaluation.yaml
```

## 2. 数据格式

JSONL 每行一个样本：

```json
{"id":"math-0001","question":"题目文本","answer":"42"}
```

校准集不读取答案，但应与评测集按题目划分，避免同题复用。校准和评测建议分别准备 YAML，只修改 `data.path` 与 `output.directory`。

## 3. 修改配置

复制 `configs/qwen3_1p7b_4b_core.yaml`，将以下字段改成绝对本地路径：

```yaml
models:
  slm:
    path: /work/models/Qwen3-1.7B
    device: cuda:0
  llm:
    path: /work/models/Qwen3-4B
    device: cuda:1

data:
  path: /work/data/calibration.jsonl

output:
  directory: /work/outputs/dcrh_core
```

两张 GPU 是推荐配置。单卡时可把两个 `device` 都设为 `cuda:0`，但两个模型权重会同时驻留；代码当前不做 CPU/磁盘换入换出。

## 4. 无标签校准

### SLM：建立经验分布并选择一个阈值 B

```bash
python scripts/calibrate.py \
  --config /work/configs/calibration.yaml \
  --role slm \
  --reference-out /work/references/qwen3_1p7b_reference.npz \
  --threshold-out /work/references/qwen3_1p7b_threshold.json
```

### LLM：建立自身 H/G 经验分布

```bash
python scripts/calibrate.py \
  --config /work/configs/calibration.yaml \
  --role llm \
  --reference-out /work/references/qwen3_4b_reference.npz \
  --threshold-out /work/references/qwen3_4b_unused_threshold.json
```

LLM 不使用升级阈值；第二个 JSON 仅记录校准信息。把两个 reference 路径和 SLM threshold 路径写回评测 YAML。

## 5. 核心实验

```bash
python scripts/run.py --config /work/configs/evaluation.yaml
```

中断后再次执行会按 `example_id` 跳过已完成样本。单样本失败默认终止；需要记录错误并继续时增加：

```bash
python scripts/run.py \
  --config /work/configs/evaluation.yaml \
  --continue-on-error
```

汇总：

```bash
python scripts/summarize.py \
  --results /work/outputs/dcrh_core/results.jsonl \
  --output /work/outputs/dcrh_core/summary.json
```

同一解码器下的 SLM-only / LLM-only 基线：

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

基线关闭 G probe，但仍记录真实 prefill、decode、KV 估计和峰值显存。

## 6. 成本统计字段

每个结果的 `cost` 对象包含：

- `prefills`：每次真实 prefill 的角色、用途、token 数、耗时、估计实时 K+V 字节数；
- `discarded_rollback_suffix_tokens`：变点确认前已经生成、回滚后丢弃的 SLM token；
- `discarded_rollback_suffix_decode_seconds`：上述 token 对应的实测 decode forward 时间；
- `discarded_rejected_trial_tokens`：接管试运行失败后丢弃的 SLM token；
- `wasted_prefill_tokens/seconds`：失败 trial 已支付但没有进入最终轨迹的 SLM prefill；
- `llm_upgrade_actual_prefill_tokens`：LLM 在回滚点 `[0, τ̂]` 上的实际 prefill 长度；
- `llm_detection_point_counterfactual_prefill_tokens`：若不回滚、从报警点升级时的反事实长度；
- `rollback_llm_prefill_tokens_saved`：回滚带来的 LLM prefill token 节省；
- `probe_attention_qk_elements`：为计算 G 在选定层显式重算的 QK 元素数；
- `gpu_memory`：每张 CUDA 设备的峰值 allocated/reserved memory；
- `kv_policy: live_session_only_no_snapshots`：确认不存在边界 KV 归档。

`prefill_attention_qk_elements_upper_bound` 和 `decode_attention_qk_elements_upper_bound` 是稠密注意力工作量上界，不是 FLOPs。论文中的系统结论应以同步端到端时延、prefill token、生成 token 和峰值显存为主。

## 7. 运行前建议

先把校准规模改小，使用 8–16 道题完成端到端冒烟测试；确认：

1. 能观察到 `"\n\n"` 原子边界；
2. reference 文件内三组分布非空；
3. `results.jsonl` 中确实出现 SLM、LLM、trial 的 prefill 记录；
4. 回滚样本的实际 LLM prefill 小于报警点反事实 prefill；
5. rejected trial 的 prefill 和生成开销被计入 waste 字段。

完成后再恢复正式校准规模。
