你现在在我的代码仓库中工作。请先阅读当前项目结构、README、已有的模型加载、推理、评测、日志代码，然后实现一个科研原型方法，不要重构整个项目，不要替换已有评测接口，不要强行引入新的部署框架。

研究背景：
我正在实现一个 small-large reasoning model collaboration 方法。默认由 SLM 生成推理过程；当 SLM 的推理路径出现可疑状态转移时，不是 early exit，也不是直接从当前 token 继续，而是让 LLM 从一个较早的可信 step 回滚式接管。

当前实验组合：
- SLM: Qwen3-1.7B
- LLM: Qwen3-4B
- step delimiter: "\n\n"
- 当前方法是科研原型，优先保证信号定义、日志可解释、ablation 容易做；不需要生产级推理吞吐优化。

请实现的方法暂名为 MDRV: Margin Drawdown with Route Velocity。

一、总体任务

请实现一个可插入现有推理流程的 collaboration router。它应该支持：

1. SLM 默认按 step 生成推理内容。
2. 每遇到自然 transition boundary，也就是生成到 "\n\n" 后，记录一个 step。
3. 在该边界处计算 TPM:
   M_i = p_top1 - p_top2
   其中 p_top1 和 p_top2 是 SLM 在当前边界预测下一 token 时 top-1 与 top-2 token 的 softmax probability。
4. 在同一边界处计算 attention route:
   r_i = [r_i^A, r_i^O, r_i^P, r_i^C]
   它表示当前边界 query 对四个区域的信息路由分布。
5. 用 TPM drawdown 计算 SLM 局部选择清晰度的未恢复下降。
6. 用 route velocity 解释 TPM drawdown 是否伴随信息路由变化。
7. 当接管风险超过 tau 时，丢弃 drawdown segment 内的 SLM 后缀，让 Qwen3-4B 从最近可信 prefix 继续完成。
8. LLM 接管后直接生成最终答案；第一版不做 LLM→SLM 交还。
9. 输出最终答案和完整日志。

二、不要实现的东西

请不要实现以下内容：

- 不要实现 ASAG 的 attention entropy variation。
- 不要实现 early exit。
- 不要实现 intermediate answer probing。
- 不要实现 answer stability。
- 不要实现 verifier。
- 不要实现 jump prompt。
- 不要实现 logits injection。
- 不要实现 PRM / reward model。
- 不要实现 speculative decoding 的 token acceptance / rejection。
- 不要新增多个阈值来判断 high/low margin、high/low attention、连续异常次数。
- 不要把 TPM 解释成答案置信度、正确率估计或 final answer confidence。

本方法只关心：
TPM = SLM 对下一推理动作的局部选择清晰度。
Attention route velocity = transition boundary 处信息路由是否变化。

三、建议的代码组织

请尽量按现有项目风格实现。若项目已有类似 runner / engine / evaluator，请接入已有结构。若没有，请新增最少的模块。建议模块如下，但可以根据项目实际情况调整：

1. model_adapters.py 或现有模型文件中新增：
   - SLMAdapter
   - LLMAdapter
   - 需要暴露：
     - generate_text(...)
     - forward_boundary_state(...)
   - forward_boundary_state(prefix_text) 应返回：
     - next_token_logits
     - optional attention weights
     - tokenized input ids
     - token spans needed by route computation

2. router_signals.py:
   - compute_tpm_from_logits(logits)
   - build_region_spans(prompt_text, chunks, tokenizer)
   - compute_attention_route(attentions, region_spans)
   - compute_jsd_route_velocity(r_i, r_prev)

3. router_state.py:
   - 保存每个 step 的 M_i, r_i, V_i, G_i, segment_start, barV, R_i。
   - 实现 TPM drawdown 和 takeover decision。

4. collaborative_generator.py:
   - 主入口：
     collaborative_generate(problem, config)
   - 负责调用 SLM、检测 "\n\n" step、计算信号、触发 LLM rollback takeover、输出结果。

5. logging:
   - 每个样本输出一个 JSONL record。
   - 日志比速度更重要。

四、TPM 计算定义

在第 i 个 step 结束后，当前 prefix 为：

prompt + T_1 + "\n\n" + ... + T_i + "\n\n"

请用 SLM 对该 prefix 做一次 forward，取最后一个位置的 logits 作为“预测下一 token”的 logits。

从 logits 计算 softmax probability，取 top-2：

p_top1 = top-1 probability
p_top2 = top-2 probability
M_i = p_top1 - p_top2

注意：
- M_i 只表示下一推理动作的局部选择清晰度。
- 不要把 M_i 命名成 confidence。
- 不要把 M_i 作为答案正确率。
- 日志中记录 top1 token id/text、top2 token id/text、p_top1、p_top2、M_i。
- 默认使用 raw LM logits。若项目已有 generation scores，也可以加一个可选 source 字段，但主方法以 raw LM logits 为准。

五、Attention route 计算定义

在同一个 boundary forward 中，尝试读取 SLM 的 attention。

优先使用：
- 最后一层 attention。
- 对所有 heads 求平均。
- 只取 boundary query row，也就是 prefix 最后一个 token 对历史所有 key tokens 的 attention。
- 不要计算整个 step 的 attention map。
- 不要用 ASAG 的 last-four-layer entropy/window-32 方案作为主实现。

将历史 token 分成四个区域：

A: 原题、系统提示、固定 instruction、question 部分。
O: 较早 reasoning chunks，即 T_1 到 T_{i-2}。
P: 前一个 chunk，即 T_{i-1}。
C: 当前 chunk，即 T_i。

请实现 region span 构造：
- 用 tokenizer 对 prompt 和各 chunk 重新编码来建立 token span。
- 保证四个区域的 token span 不重叠。
- 当 i=1 时，O 和 P 为空。
- 当 i=2 时，O 为空，P=T_1，C=T_2。
- 空区域 route 设为 0。

区域 route 建议使用 length-normalized density，而不是 raw mass：

对区域 Z：
u_i^Z = mean attention over tokens in Z

然后：
r_i^Z = u_i^Z / sum_Z u_i^Z

最终：
r_i = [r_i^A, r_i^O, r_i^P, r_i^C]

注意：
- r_i 只表示信息路由分布。
- 不要把 r_i 解释成正确性。
- 不要把 attention 当作因果解释。
- 只允许使用一个很小的 epsilon 处理除零或 log(0)，epsilon 是数值稳定项，不作为可调超参数。
- 若当前模型/后端不能返回 attention，则允许降级为 TPM-only 模式：记录 attention_unavailable=true，并令 route discount 不生效。

六、Route velocity 定义

从第二个 step 开始计算：

V_i = JSD(r_i, r_{i-1}) / log(2)

要求：
- 使用 Jensen-Shannon divergence。
- 使用自然对数时除以 log(2)，使 V_i 在 [0, 1] 附近。
- V_i 只表示从上一个 boundary 到当前 boundary 的信息路由变化幅度。
- 不要使用窗口均值作为主方法。
- 不要新增 W。

七、TPM drawdown 定义

定义：

G_1 = 0

对 i >= 2：

G_i = max(0, G_{i-1} + M_{i-1} - M_i)

含义：
G_i 是从最近一次清晰状态以来，SLM 下一推理动作局部选择清晰度的未恢复下降。

实现细节：
- 当 G_i == 0 时，更新 last_reset_step = i。
- 当前 drawdown segment start 为 s_i = last_reset_step。
- 若 G_i > 0，则可疑 segment 是 T_{s_i+1}, ..., T_i。
- 可信 prefix 是 prompt + T_1 ... T_{s_i}。

八、Route velocity 对接管风险的作用

Attention 不进入 CUSUM，不累计为异常证据。它只折扣或解释 TPM drawdown。

在当前 drawdown segment 内计算：

barV_{s_i:i} = average of V_k for k in s_i+1 ... i

然后：

R_i = G_i * (1 - barV_{s_i:i})

触发条件：

R_i >= tau

其中 tau 是第一版唯一在线判定阈值。

解释：
- 高 G_i + 高 barV：TPM 清晰度下降，但信息路由在变化，可能是合理探索或解法切换，降低接管风险。
- 高 G_i + 低 barV：TPM 清晰度下降且信息路由变化弱，更像不确定且停滞，提高接管风险。
- G_i 低：不接管。
- final answer 是否正确不由 router 判断。

九、LLM rollback takeover

当 R_i >= tau 时：

1. 只允许触发一次 LLM takeover。
2. anchor step = s_i。
3. trusted_prefix = prompt + T_1 ... T_{s_i}
4. discard_suffix = T_{s_i+1} ... T_i
5. 用 LLM 从 trusted_prefix 继续生成最终答案。
6. 默认不要把 discard_suffix 传给 LLM。
7. 不要从当前完整 SLM prefix 继续，因为这会把 LLM 锚定到可疑后缀。

LLM 接管 prompt 可以保守地加入一段固定 instruction，但不要引入复杂 meta-reasoning。建议形式：

"You are continuing a reasoning process. The previous small model reasoning is trusted only up to the last provided step. Continue from that point and solve the problem. Do not assume any later discarded reasoning is correct. Put the final answer in \\boxed{}."

若现有项目已有统一 system prompt，请用项目风格整合，不要破坏 evaluator。

十、在线模式与降级模式

第一版可以支持两种模式：

A. online_collaboration
- SLM 实际 step-wise 生成。
- 每生成到 "\n\n" 就计算信号。
- 触发时立即 rollback 给 LLM。
- 这是主方法目标。

B. offline_replay
- 先让 SLM 正常生成完整输出。
- 再按 "\n\n" 分 chunk。
- replay 每个 boundary prefix，计算 M_i、r_i、V_i、G_i、R_i。
- 根据 tau 找到理论 takeover anchor。
- 可选择再调用 LLM 从 anchor prefix 生成。
- 这个模式用于科研调试、信号可视化和 ablation；允许作为先实现版本。

若 online step-wise generation 在当前框架中成本较高，请先实现 offline_replay + 可选 LLM rerun。不要因为 online 复杂而修改方法定义。

十一、日志要求

请为每个 sample 输出 JSONL。至少包含：

- sample_id
- problem/question
- model names
- tau
- mode: online_collaboration 或 offline_replay
- delimiter
- final_source: SLM 或 LLM
- triggered: true/false
- trigger_step
- anchor_step
- discarded_steps
- num_slm_steps
- slm_generated_tokens
- llm_generated_tokens
- final_output
- per_step records:
  - step_index
  - step_text
  - prefix_token_len
  - M_i
  - p_top1
  - p_top2
  - top1_token_id/text
  - top2_token_id/text
  - r_i_A, r_i_O, r_i_P, r_i_C
  - V_i
  - G_i
  - segment_start
  - barV
  - R_i
  - attention_available
  - takeover_decision_at_step

日志是科研归因的核心。不要只输出最终 accuracy。

十二、必要 ablation hooks

请保留下列简单开关，不要新增复杂策略：

1. route_discount=true/false
   - true: R_i = G_i * (1 - barV)
   - false: R_i = G_i
   用于验证 attention route velocity 是否有用。

2. takeover_mode="rollback" 或 "current"
   - rollback: LLM 从 trusted_prefix 接管。
   - current: LLM 从当前完整 SLM prefix 接管。
   用于验证 rollback anchor 是否有用。
   主方法默认 rollback。

3. attention_route_mode="last_layer_single_row" 或 "none"
   - none 表示 TPM-only。
   - 不要把 entropy 作为第三种主模式。

4. run_mode="online_collaboration" 或 "offline_replay"

十三、测试与 sanity check

请添加少量轻量测试或 debug checks，不需要完整评测：

- 给定 M 序列 [0.8, 0.6, 0.4]，G 应连续上升。
- 给定 M 序列 [0.8, 0.6, 0.9]，G 应恢复到 0。
- JSD(r, r) 应为 0。
- r_i 四个分量应非负，且和接近 1。
- i=1、i=2 时空区域不会报错。
- 触发 takeover 时 anchor_step 应等于最近一次 G=0 的 step。
- attention 不可用时可以继续 TPM-only，不要崩溃。

十四、实现优先级

优先级从高到低：

1. 离线 replay 版信号计算和日志。
2. TPM-only router。
3. attention route velocity。
4. rollback takeover 调用 LLM。
5. online step-wise generation。
6. ablation hooks。

不要为了吞吐优化、batch 推理、KV cache 复用、vLLM 集成而阻塞核心方法。当前目标是研究验证，不是生产部署。

十五、保持方法边界

请在代码注释和变量命名中保持以下语义：

- M_i / tpm_margin: local action clarity
- G_i / margin_drawdown: unresolved clarity loss
- r_i / route_distribution: information route over A/O/P/C
- V_i / route_velocity: route change
- R_i / takeover_risk: collaboration routing score
- tau: takeover budget threshold

不要使用 answer_confidence、correctness_score、stability_score 这类容易误导的命名。

完成后，请汇报：
- 修改了哪些文件。
- 新增了哪些入口函数。
- 如何运行一个最小样例。
- 日志字段在哪里查看。
- 哪些部分是降级实现，哪些部分仍是 TODO。