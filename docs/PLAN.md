# rolloutscope —— Agent rollout 开放 trace 格式 + 跨框架诊断器

> 立项调研与实施规划
> 版本：v0.1 草案 ｜ 日期：2026-10-19

---

## 1. 结论先行

**值得做，但必须换一个定位。**

原方案的表述是「定义一个统一的 `AgentTrace` schema，再写 verl/TRL 适配器」。调研后发现：**schema 这一层已经有人在做，而且做得比我们好**——OpenTelemetry 的 GenAI 语义约定正在成为事实标准，OpenInference / Langfuse / Phoenix / Traceloop 都在往这个方向收敛。如果我们把项目的主卖点放在「又一个统一格式」上，大概率会变成第 6 个没人用的格式。

真正没人做、且痛点真实存在的，是**schema 之上的诊断层**：

> 现有工具全都在回答「模型生成了什么」（trace viewer），
> 没有一个在回答「这批 rollout 里有多少是白跑的、白跑的是哪一类、下一步该改什么」。

所以定位调整为：

> **rolloutscope = 给 RL rollout 做体检的 linter。**
> 输入是你训练时已经落盘的 rollout 文件（verl parquet / TRL completions / 任意 jsonl / 甚至 OTel spans），
> 输出是一份带证据链的诊断报告：零优势组占多少、熵在哪一步塌、哪些样本疑似 reward hacking、浪费了多少算力。
> `AgentTrace` schema 是内部实现和互操作层，不是卖点。

一句话电梯陈述：**「ruff 之于 Python，就是 rolloutscope 之于 RL rollout。」**

---

## 2. 调研证据

### 2.1 verl：已有 trace，但锁死在第三方后端

verl 文档里已有 `rollout_trace` 功能（`actor_rollout_ref.rollout.trace.backend=weave|mlflow|trackio`），提供 `rollout_trace_op` 装饰器和 `rollout_trace_attr` 来标记轨迹入口，能按 `step / sample_index / rollout_n / validate` 过滤。

这说明两件事：

- **「verl 没有 trace」是错的**。原方案里「verl 用 Weave，所以数据散架」这个前提只对了一半——verl 是主动把数据推给 Weave/MLflow/Trackio，而不是没有数据。
- **真正的痛点在成本和可分析性**。官方文档自己写着：Weave 免费额度 1GB/月，而训练期间「每天产生几十 GB trace 数据」。也就是说，**大部分人要的不是更好的可视化，而是「不要把所有 token 都塞进 SaaS」**。这恰恰支持「本地文件 + 离线诊断」的路线。

### 2.2 TRL：已经有 `frac_reward_zero_std`，但只有一个标量

TRL 的 `GRPOTrainer` 会记录 `frac_reward_zero_std`（reward 标准差为 0 的 prompt 比例），以及 `completions/mean_length`、`entropy`、`clipped_ratio` 等指标，还支持 `log_completions` / `rollout_func` / reward 函数里的 `log_extra`、`log_metric`。

这是最强的需求验证：**主流框架已经把「零优势组」当成一等公民指标了**。但它止步于一个聚合标量：

- 不知道是「全对」还是「全错」（两者对训练的处置完全不同：前者该淘汰，后者该降难度或加 hint）；
- 没有样本级归因，点不进去看是哪道题；
- 没有跨 step 的分解趋势，看不出「从第几步开始退化成刷简单题」；
- 没有换算成「浪费了多少 GPU 小时」。

我们的价值就是把这个标量展开成**可归因、可下钻、可换算成本**的诊断。

### 2.3 rLLM：自己的三层结构，没有 OTel 出口

rLLM 用 `Episode → Trajectory → Step` 三层组织数据（`Trajectory(name, steps)`、`Episode(trajectories, artifacts)`），Model Gateway 自动捕获 token ids + logprobs，trace 存 sqlite，自带 rLLM UI。

结论：每个框架都在「自己那一套结构 + 自己那个 UI」里闭环，**跨框架的公共分析层是空白**。这也印证了原方案的判断，只是缺一个能让人愿意接入的理由——我们的理由是「不用改训练代码，读你已有的文件就行」。

### 2.4 OTel GenAI 语义约定：有 agent/tool span，但**没有 RL 语义**

已核对 `open-telemetry/semantic-conventions-genai` 仓库（状态仍为 Development）。关键 span 与属性：

| 场景 | `gen_ai.operation.name` | 关键属性 |
| --- | --- | --- |
| Agent 调用 | `invoke_agent` | `gen_ai.agent.name`、`gen_ai.conversation.id` |
| 模型调用 | `chat` / `generate_content` | `gen_ai.request.model`、`gen_ai.response.model`、`gen_ai.usage.input_tokens` / `output_tokens`、`gen_ai.input.messages`、`gen_ai.output.messages`、`gen_ai.response.finish_reasons` |
| 工具执行 | `execute_tool` | `gen_ai.tool.name`、`gen_ai.tool.call.id`、`gen_ai.tool.call.arguments`、`gen_ai.tool.call.result` |

**缺口很明确**：这套约定能描述「一次 agent 运行发生了什么」，但完全没有 RL 训练需要的维度——reward、advantage、group id（GRPO 的分组）、training step、rollout index、是否被截断、policy/ref logprob。这些恰恰是「为什么不涨」的全部信息。

这给了我们一个干净的设计原则：**schema 不另起炉灶，而是「OTel GenAI 的超集」**——trace 主体 100% 用 OTel 字段，RL 维度用 `rolloutscope.*` 扩展属性挂上去，并把这组扩展属性整理成一份可以提给 OTel 社区的提案。这样既不重复造轮子，又有实质贡献点。

### 2.5 竞品与同类项目

| 类别 | 代表 | 它回答的问题 | 缺口 |
| --- | --- | --- | --- |
| Trace viewer / LLM observability | Weave、Langfuse、Arize Phoenix、LangSmith、AgentOps、MLflow | 这次调用输入什么、输出什么、花了多少 token | 不做 RL 语义，不做「这一批 rollout 的健康度」 |
| 训练指标面板 | wandb、Trackio、TRL/verl 自带 metrics | loss/entropy/length 的曲线 | 聚合指标，无样本级归因，跨框架不可比 |
| 框架自带 UI | rLLM UI、verl trace | 单次轨迹回放 | 各自闭环，不能横向对比 |
| 学术侧 | 论文里的 "Zero-Advantage Collapse"（如 arXiv 2606.08815、DEEPO 等） | 提出概念并给算法解 | 没有工程化工具，没人把它做成 `pip install` |

GitHub 检索 `agent rollout trace diagnostics` / `grpo zero advantage analysis` 均为 0 结果，`rl rollout diagnostics` 仅 2 个无关小项目。**没有直接竞品。**

### 2.6 立项判断

| 维度 | 评估 |
| --- | --- |
| 需求真实性 | 强。verl 文档自曝 trace 成本、TRL 已内置零优势指标、论文已命名 Zero-Advantage Collapse |
| 差异化 | 强。诊断层空白，且「读已有文件」的接入成本远低于「改训练代码接 SDK」 |
| 可行性 | 高。纯 Python + pandas，无 GPU，一晚上能出可截图 demo |
| 风险 | 中。规则是启发式，容易被质疑「这不就是几个 if」；OTel 约定仍是 Development，字段会漂移 |
| 面试价值 | 高。能把「全对组为什么没梯度」「reward hacking 怎么在数据里现形」讲成故事，比背公式有说服力 |

**Go。** 但第一版必须砍到「诊断引擎 + 一个能截图的报告」，schema 和适配器只做到够用。

---

## 3. 产品设计

### 3.1 三条设计原则

1. **零侵入（zero-instrumentation）**
   不要求用户改训练代码、不要求重跑训练。读你已经有的东西：verl `save_to_disk` 出来的 parquet、TRL 的 `trainer_state.json` + completions 导出、任意 jsonl、OTLP spans。**这是能不能被用起来的分水岭。**

2. **schema 是 OTel 的超集，不是替代品**
   所有 trace 主体字段与 OTel GenAI 一一对应（`gen_ai.*`）；RL 专有维度用 `rolloutscope.*` 扩展。提供 `to_otel()` / `from_otel()`，任何已经用 OpenInference / Langfuse / Traceloop 埋点的 agent，导出的 span 直接能进诊断。

3. **每条诊断必须可行动**
   输出不是一个分数，而是：`现象 → 量化 → 证据样本 → 可能原因 → 建议动作 → 估算浪费`。不能落到「改什么」的告警等于噪音。

### 3.2 用户路径

```bash
pip install rolloutscope

# 1) 先看 demo（30 秒出报告，用来判断值不值得接）
rolloutscope demo

# 2) 扫自己已有的产物（自动识别格式）
rolloutscope report ./checkpoints/exp1/rollout_dump --out report.html

# 3) CI / 训练脚本里跑，超阈值就 fail
rolloutscope scan ./rollout_dump --fail-under 60 --json out.json

# 4) 导出成 OTel，接你现有的观测平台
rolloutscope convert ./rollout_dump --to otel --out spans.jsonl
```

### 3.3 数据模型（`AgentTrace`）

```text
TraceBatch                    一次分析的范围（一个 step / 一个 dump / 一次实验）
└── AgentTrace                一次 rollout（= 一条轨迹 = GRPO 组里的一个样本）
    ├── linkage: run_id, framework, step, group_id, rollout_index, prompt_id
    ├── prompt: str | None
    ├── steps: list[Step]     一次 LLM 调用 = 一个 chat span
    │   ├── index, text, finish_reason, model
    │   ├── token_usage: {input, output, cached_input}
    │   ├── logprobs: list[float] | None   （用于熵/NLL 估计）
    │   └── tool_calls: list[ToolCall]     （= execute_tool span）
    │       └── name, call_id, arguments, result, error, duration_ms
    ├── reward: float | None
    ├── advantage: float | None
    ├── reward_components: dict[str, float]
    └── status: ok | invalid | error  +  invalid_reason
```

OTel 映射（`to_otel`）：

```text
invoke_agent {agent}                      gen_ai.agent.name, gen_ai.conversation.id
├── chat {model}                          gen_ai.usage.*, gen_ai.input.messages, gen_ai.response.finish_reasons
│   └── execute_tool {tool}               gen_ai.tool.name / call.id / call.arguments / call.result
└── chat {model} ...
```

RL 扩展属性（本项目提案，全部 `rolloutscope.` 前缀，避免污染 `gen_ai.` 命名空间）：

| 属性 | 类型 | 含义 |
| --- | --- | --- |
| `rolloutscope.schema.version` | string | schema 版本，如 `0.1` |
| `rolloutscope.framework` | string | `verl` / `trl` / `rllm` / `otel` / `generic` |
| `rolloutscope.train.step` | int | 训练步 |
| `rolloutscope.group.id` | string | GRPO 分组键（同一 prompt） |
| `rolloutscope.rollout.index` | int | 组内第 n 个采样 |
| `rolloutscope.reward` | double | 标量 reward |
| `rolloutscope.advantage` | double | 组内归一化后的 advantage |
| `rolloutscope.reward.components` | string(json) | 多奖励函数的分解 |
| `rolloutscope.status` | string | `ok` / `invalid` |
| `rolloutscope.invalid.reason` | string | `truncated` / `tool_format` / `empty` / `tool_loop` |

### 3.4 诊断规则（核心资产）

每条规则都有：**明确的数学定义 + 可配置阈值 + 证据样本 + 修复建议**。阈值全部可在 `rolloutscope.toml` 覆盖。

#### R1 `zero-advantage` —— 零优势组（GRPO 白跑）

对训练步 $s$、分组 $g$（同一 prompt 的 $n$ 个采样），reward 为 $r_1,\dots,r_n$：

$$\sigma_g=\operatorname{std}(r_1..r_n),\qquad \hat A_i=\frac{r_i-\bar r}{\sigma_g+\epsilon}$$

- **零优势组**：$\sigma_g < \tau_\sigma$（默认 $10^{-6}$），此时所有 $\hat A_i \approx 0$，该组对梯度贡献为 0。
- 进一步二分：
  - **饱和组（saturated）**：$\min_i r_i \ge r_{\text{pass}}$ —— 全对，题目太简单，继续采样纯浪费；
  - **坍缩组（collapsed）**：$\max_i r_i < r_{\text{pass}}$ —— 全错，题目太难，梯度为 0 但也没提供学习信号。
- 输出：`zero_adv_ratio`、`saturated_ratio`、`collapsed_ratio`、**浪费 token 数**、按吞吐换算的 **GPU 小时 / 成本估算**。
- 为什么重要（面试话术）：GRPO 的梯度是 $\sum_i \hat A_i \nabla \log \pi_\theta(y_i)$，当组内 reward 全等时 $\hat A_i=0$，这一组 $n$ 条 rollout 的前向 + reward 计算全部作废，但算力已经花掉了。**这就是「训练不涨但 loss 很稳」的典型形态。**

#### R2 `entropy-collapse` —— 熵坍缩

- 若提供了每 token logprob：$H_t \approx -\log p_\theta(y_t)$（严格说是 NLL/交叉熵，是熵的上界估计；报告里标注 `proxy`）。
- 若输入含 `entropy` 标量指标（TRL `trainer_state.json` 里有）：直接用，标注 `exact`。
- 判定：以首 3 步中位数为基线 $H_0$，当连续 $k\ge2$ 步的步内中位数满足 $H_s < \alpha H_0$（默认 $\alpha=0.6$）判为坍缩，报告**首次越界步**。
- 附加：同一步内 trace 间熵的方差过低（$\text{CV}<0.05$）→ 输出同质化，即使熵值没塌也提示。

#### R3 `length-collapse` —— 输出长度坍缩

- 步内输出长度中位数 $L_s$；$L_s / L_0 < \beta$（默认 $0.6$）连续 2 步 → 坍缩。
- 步内长度 $\text{CV} < 0.05$ → 退化成固定长度（常见于模型学会「凑够某个模式就停」）。
- 与 R2 联合判断：熵塌 + 长度塌同时出现，严重度升为 critical（这是典型的 mode collapse 前兆）。

#### R4 `reward-hacking` —— 疑似刷分

对步 $s$ 内样本 $i$：设步内 reward 的 $q_{90}$ 为高分线、长度中位数 $L_s$、工具调用中位数 $T_s$。

$$\text{flag}_i = \big[r_i \ge q_{90}\big] \wedge \Big[(\ell_i < 0.35 L_s)\ \vee\ (T_s \ge 2 \wedge c_i \le 0.2 T_s)\Big]$$

- 即「分很高，但明显比同批短，或者几乎不调工具」。
- 需要支撑度：步内样本数 $\ge 20$ 且候选 $\ge 5$ 才出结论，否则只提示 `insufficient_data`。
- 同时算 **lift**：$P(\text{high reward}\mid \text{short}) / P(\text{high reward}\mid \text{not short})$，lift > 1.5 说明「短」和「高分」正相关，是刷分的有力旁证。
- 只在「工具型任务」启用工具维度（全局工具调用中位数 $\ge 1$ 才认为任务需要工具），避免对纯文本任务误报。

#### R5 `invalid-rollout` —— 无效 rollout

| 子类型 | 判定 |
| --- | --- |
| `truncated` | `finish_reason ∈ {length, max_tokens}`，或 output tokens ≥ `max_tokens` 的 98% |
| `empty` | output tokens = 0 或 text 全空白 |
| `tool_format` | 工具参数不是合法 JSON / 缺必填字段 / 工具名不在白名单（若提供） |
| `tool_error` | 工具返回带 error 标记 |
| `tool_loop` | 连续 3 次相同 `(tool_name, arguments)` |

- 输出各子类占比 + 无效 rollout 消耗的 token 占比。
- 工具格式错误率高通常意味着 **chat template 与工具调用格式不匹配**，这是最容易修、收益最大的一类。

#### R6 `group-difficulty` —— 难度结构与有效 rollout 率（招牌指标）

- 组通过率 $p_g = \frac{1}{n}\sum_i \mathbb{1}[r_i \ge r_{\text{pass}}]$
- 分桶：$p_g=1$ 饱和 / $p_g=0$ 坍缩 / $0<p_g<1$ 有效
- **有效 rollout 率 ERR** $= \frac{|\{g: 0<p_g<1\}|}{|G|}$
- **梯度信号密度** $= \frac{\sum_i |\hat A_i|}{\sum_i \text{tokens}_i / 1000}$ —— 每千 token 换来多少梯度信号，是比 loss 更能反映「这步训练值不值」的指标。
- 按 step 画堆叠柱状图：**这是整份报告里最能回答「为什么不涨」的一张图**。

#### 健康分（Health Score）

$$\text{score} = \text{clip}\Big(100 - \big[30\cdot r_{zero} + 25\cdot r_{invalid} + 20\cdot \min(r_{hack}/0.2,1) + 15\cdot f_{entropy} + 10\cdot f_{length}\big], 0, 100\Big)$$

权重可配，报告里明写「这是启发式评分，不是绝对标准」，避免被当成伪科学。

### 3.5 报告形态

单文件 HTML，**零外部依赖**（不引 CDN、不引 JS 库，图表是 Python 直接生成的 inline SVG），可以离线打开、可以直接截图、可以丢进 PR 当附件。

结构：

1. 头部：实验名、框架、trace/group 数、**健康分**、生成时间
2. KPI 卡：ERR、零优势组占比、无效占比、估算浪费
3. 图 1（招牌）：按 step 的「饱和 / 坍缩 / 有效」堆叠柱 + ERR 折线
4. 图 2：输出长度中位数 + p10–p90 带 + 熵曲线（双轴）
5. 图 3：reward vs length 散点，reward hacking 样本高亮
6. 诊断清单：按严重度排序，每条含定义、量化、证据样本 id、建议
7. 样本下钻：3–5 条典型轨迹的对话/工具时间线（trace 部分的价值在这里体现）

---

## 4. 实施规划

### M0 骨架与 demo（本次交付）

- [x] `AgentTrace` / `Step` / `ToolCall` dataclass + JSON 序列化
- [x] OTel 双向映射（`to_otel` / `from_otel`，含 OTLP JSON 解析）
- [x] 适配器：verl（parquet/jsonl）、TRL（trainer_state + completions）、通用 JSONL、格式自动探测
- [x] 规则 R1–R6 + 健康分
- [x] 离线 HTML 报告（纯 SVG）
- [x] CLI：`demo` / `scan` / `report` / `convert`
- [x] 合成 demo 数据（带设计好的病理，让 6 条规则全部触发）
- [x] pytest：schema 往返、OTel 往返、规则在构造数据上的判定、适配器 fixture

### M1 真实数据校准（1–2 周）

- 用 1 个真实 verl 小实验 + 1 个 TRL GRPO 小实验跑通，校准阈值
- 修适配器对真实字段的假设（verl 的 `uid/index`、`rm_scores`、`response_mask`；TRL completions 的实际落盘格式）
- 加 `rolloutscope diff A.json B.json`：两次实验的体检对比

### M2 生态接入（2–4 周）

- 发布 PyPI，补 README 截图、5 分钟 quickstart
- 提供 `rolloutscope.otel` 的 OTel SpanProcessor，训练时实时采集（可选，非必需）
- 写一篇「为什么全对组没有梯度」的博客，把 R1 的定义和实测数据放进去
- 把 `rolloutscope.*` 扩展属性整理成 OTel semconv 提案 issue

### M3 深度（可选）

- 规则可插拔（entry points）+ 自定义规则 DSL
- 与 wandb / Trackio 的 metric 对齐（同一实验里同时看曲线和体检）
- 多实验横向对比报告（哪个超参组合的 ERR 最高）

### 验收标准

| 项 | 标准 |
| --- | --- |
| 安装 | `pip install rolloutscope` 后 `rolloutscope demo` 一条命令出报告，无 GPU、无网络 |
| 依赖 | 运行期仅 `pandas`（parquet 需要 `pyarrow`，可选） |
| 性能 | 10 万条 trace 的扫描 < 30s（纯 pandas 向量化，不逐条循环） |
| 可解释 | 每个 finding 都能点进具体 trace id |
| 可测试 | 规则层单测覆盖率 > 80%，全部用构造数据，不依赖外部服务 |

---

## 5. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 「这不就是几个 if 判断」 | 被认为没技术含量 | 每条规则给数学定义 + 文献引用（Zero-Advantage Collapse、entropy collapse）+ 在真实实验上给出「命中率 vs 人工标注」的校准结果；把 lift / 置信区间做进去 |
| OTel GenAI 约定仍是 Development，字段会变 | 适配层腐化 | schema 版本号 + 映射表集中在 `otel.py` 一个文件；升级只改一处 |
| 各框架数据格式频繁变动 | 适配器失效 | 适配器只依赖最小字段集，缺失字段降级而不是报错；用真实 dump 做 fixture 回归 |
| 没人用 | 项目死掉 | 先做「demo 30 秒出图」，README 首屏放报告截图；优先在 verl/TRL 的 issue/discussion 里以「诊断工具」身份出现，而不是「又一个 trace 格式」 |
| 阈值不通用 | 误报/漏报 | 全部阈值可配；报告里区分 `critical / warn / info`，低置信度只出 info |

---

## 6. 与面试叙事的结合

这个项目能撑起的技术叙事（按深度排序）：

1. **GRPO 为什么会在「全对组」上白跑**：从 $\hat A_i = (r_i-\bar r)/\sigma$ 出发，说明 $\sigma=0$ 时梯度恒为 0，且这不是实现 bug 而是算法结构性问题；再讲工程上怎么度量（R1）、怎么处置（饱和组降采样 / 坍缩组加 hint / 动态基线）。
2. **指标会骗人，数据不会**：loss 平稳、reward 缓升，但 ERR 从 0.7 掉到 0.2 —— 模型在把简单题做对、难题放弃，报告里能直接看到。
3. **观测的成本模型**：verl 官方说每天几十 GB trace，Weave 免费额度 1 GB —— 为什么「全量 trace 上传 SaaS」不可行，为什么本地聚合 + 采样下钻是正解。
4. **标准与实现的边界**：OTel 能描述 agent 行为但描述不了 RL 语义，扩展属性怎么设计才不污染上游命名空间。

---

## 7. 命名

`rolloutscope`（PyPI 可用，已核实）。备选：`rollout-lens`、`agentrollout`、`rolloutdiag`（均可用）；`agenttrace` / `rldoctor` / `tracelint` 已被占用。

选 `rolloutscope` 的理由：`scope` 同时有「范围」和「内窥镜」两层意思，和「把一次训练剖开看」的定位贴合，且不与现有包冲突。
