# LITERATURE.md — Probing LLM Internal States

覆盖 `baseline/` 下的 21 个 baseline，外加做 "unified probe" 的几条相关工作。

**分类标准只有一个：从模型里读什么信号（signal source）。** aggregator、supervision、
query token 位置、做什么用，都作为卡片里的字段写，不另开分组。

每张卡三行：
- **引用**：作者、venue、arXiv。
- **做什么**：读什么 → 算什么 → 输出什么。
- **对我们**：extraction 覆盖情况 + 能在 fusion 里当什么 expert。

完整 5 轴 taxonomy 在附录 A。

---

## 2. 方法卡片

### 2.1 Residual / hidden state

**Geometry of Truth** — Marks & Tegmark, COLM 2024 Spotlight ([arXiv:2310.06824](https://arxiv.org/abs/2310.06824)).
- 做什么：true/false 陈述的中层残差 → mass-mean / LR 线性探针 → 真值方向。
- 对我们：`input_last_token_hidden` 完整覆盖，直接 LR 即可。fusion 里是"residual LR expert"的代表。

**SAPLMA / correctness-model-internals** — Azaria & Mitchell, Findings-EMNLP 2023 ([arXiv:2304.13734](https://arxiv.org/abs/2304.13734)).
- 做什么：prompt 末 token 的中层 hidden → PCA + LR → 生成前的 correctness 预测。
- 对我们：`input_last_token_hidden` 够。fusion 里的"pre-generation correctness expert"。

**SEP — Semantic Entropy Probes** — Kossen et al., [arXiv:2406.15927](https://arxiv.org/abs/2406.15927) (NeurIPS-24 SafeGenAI Workshop).
- 做什么：生成末 token 的 hidden → LR → 近似 semantic entropy（原本要多次采样才能算）。
- 对我们：`gen_last_token_hidden` 够。fusion 里的"post-generation uncertainty expert"。

**LID — Local Intrinsic Dimension** — Yin et al., ICML 2024 ([arXiv:2402.18048](https://arxiv.org/abs/2402.18048)).
- 做什么：生成末 token 的全层 hidden → MLE 估计局部内在维度 → 无监督真实性分数。
- 对我们：`gen_last_token_hidden` 够。fusion 里的"manifold-geometry expert"（非线性 aggregator）。

**Chain-of-Embedding** — Wang et al., ICLR 2025 ([arXiv:2410.13640](https://arxiv.org/abs/2410.13640)).
- 做什么：某个固定位置的全层 hidden 当轨迹 → 逐层 magnitude + angle 统计 → 4 个标量。
- 对我们：`gen_mean_pool_hidden` 够。fusion 里的"trajectory expert"，原论文只做二分类 AUROC，我们可以扩成多分类 LR。

**LLM Knowledge-Boundary Perception** — Ni et al., ACL 2025 Main ([arXiv:2502.11677](https://arxiv.org/abs/2502.11677)).
- 做什么：生成 token 的 hidden（位置可选 first/last/avg/min-prob，层可选 mid/last/all）→ MLP → knowledge-boundary 分数。
- 对我们：`gen_last_token_hidden` + `gen_mean_pool_hidden` 够，当前项目的 "KB MLP" baseline。

**STEP — Step-level Trace Evaluation & Pruning** — [arXiv:2601.09093](https://arxiv.org/abs/2601.09093) (preprint, 2026-01).
- 做什么：CoT 生成中每个 `\n\n` 步骤边界的 hidden → 轻量 MLP 打分 → 在线剪枝低质量 trace。
- 对我们：`gen_step_boundary_hidden` 够，但粒度是 step 级，做 sample 级分类要聚合。

**EigenScore / INSIDE** — Chen et al., ICLR 2024 ([arXiv:2402.03744](https://arxiv.org/abs/2402.03744)).
- 做什么：同一 prompt 采 10 次 generation 的中层 hidden → 响应协方差的特征值 → 无监督幻觉分数。
- 对我们：我们只采了 1 次，分数退化；要复现得补 9 次采样。

**SAPLMA-with-SAE / SAE Entities** — Ferrando et al., ICLR 2025 ([arXiv:2411.14257](https://arxiv.org/abs/2411.14257)).
- 做什么：实体 token 位置的残差过 SAE → 稀疏 latent → 挑出"known-entity"/"unknown-entity"特异的维度。
- 对我们：`input_last_token_hidden` 是 SAE 输入，但 Qwen/Mistral/Llama 没公开 SAE；只能切到 LlamaScope 用 Llama-3.1-8B 复现。

### 2.2 Attention

**SAT Probe / Mechanistic Error Probe** — Yuksekgonul et al., ICLR 2024 ([arXiv:2309.15098](https://arxiv.org/abs/2309.15098)).
- 做什么：prompt 末 token 的 per-head attention 回看 constraint span → `‖attn·V·W_o‖` 按 head 展平 → LR 预测 constraint 是否满足。
- 对我们：`input_attn_value_norms` 基本够（差一个 W_o 投影 + constraint span 定位）。fusion 里的"attention-flow expert"。

**ITI / Honest LLaMA** — Li et al., NeurIPS 2023 Spotlight ([arXiv:2306.03341](https://arxiv.org/abs/2306.03341)).
- 做什么：per-head attention output 上训 truthfulness LR → 把真值方向在 inference 时加回去。
- 对我们：`input_per_head_activation` 就是这个口径，LR 即可。做 detection 用就是"per-head truth expert"，做 editing 用是另一回事。

### 2.3 Residual + attention 联合读出

这三篇读了多种 signal，按 signal source 无法归到单一 bin。

**LLM-Check** — Sriramanan et al., NeurIPS 2024 ([OpenReview LYx4w3CAgy](https://openreview.net/forum?id=LYx4w3CAgy)).
- 做什么：平行三路——(1) attention kernel 的 eigen-spectrum，(2) 残差的 centered SVD，(3) token 概率的 perplexity + logit entropy；每路独立给分，不做学习式融合。
- 对我们：三路都能部分复现——`input_attn_stats[...,2]` = diag_logmean、`gen_last_token_hidden` 做一个生成末层 SVD、`gen_logit_stats_last.entropy`；PPL 因为只存了末 token logit 只能降级。

**ICR Probe** — Zhang et al., ACL 2025 Main Long ([arXiv:2507.16488](https://arxiv.org/abs/2507.16488)).
- 做什么：把每层残差更新拆成 FFN 贡献 + self-attention 贡献，top-k/p 池化后喂小 probe。
- 对我们：**不够**。需要每生成步 × 每层的完整 attention 矩阵 + 每层 hidden；我们只存了标量统计和末层。只能做降级版。

**Gnosis** — Ghadiri & Niu, [arXiv:2512.20578](https://arxiv.org/abs/2512.20578) (preprint, 2025-12).
- 做什么：在冻结 backbone 上新增 `_should_stop` head，读 full raw attention（FFT + CNN）+ last-layer hidden + token probs → 标量 correctness。
- 对我们：**不够**。依赖 raw attention maps（体积太大没存），且 head 绑定 backbone，要复现得在 Qwen2.5-7B 上重训一次。

### 2.4 Logits

**DoLa — Decoding by Contrasting Layers** — Chuang et al., ICLR 2024 ([arXiv:2309.03883](https://arxiv.org/abs/2309.03883)).
- 做什么：decoding 时把晚期层和早期层的 logits 相减再 softmax；不是 probe，是 decoding 策略。
- 对我们：在线方法，离线 feature 没法 replay；不进 fusion，作为对照参考。

### 2.5 Multi-sample consistency / routing

两篇都把内部信号用作"要不要触发外部动作"的门控。

**SeaKR** — Yao et al., ACL 2025 Oral (top 2.9%) ([arXiv:2406.19215](https://arxiv.org/abs/2406.19215)).
- 做什么：多次采样的 FFN 内部激活 → eigen-score → 阈值触发 RAG 检索 + snippet 重排 + 推理策略选择。
- 对我们：多采样没做，只能在 1-sample 下退化成 "SeaKR-inspired" score probe；原口径复现要补 10× 采样。

**Self-Routing RAG (SR-RAG)** — Wu et al., [arXiv:2504.01018](https://arxiv.org/abs/2504.01018) (preprint).
- 做什么：一次 forward 内生成"选哪个知识源 + 答案"；内部 uncertainty 作为路由依据。
- 对我们：代码未开源；不纳入复现。

### 2.6 权重编辑类（不是 detector，列作参照）

**ROME** — Meng et al., NeurIPS 2022 ([arXiv:2202.05262](https://arxiv.org/abs/2202.05262))；**MEMIT** — Meng et al., ICLR 2023 Oral ([arXiv:2210.07229](https://arxiv.org/abs/2210.07229)).
- 做什么：causal tracing 定位事实存在哪层 MLP → rank-one 权重更新（MEMIT 把 ROME 扩到上千条并发编辑）。
- 对我们：范式不同，不做 probe，作为 probing → editing 的参考坐标，不进 fusion。

---

## 3. "有没有人做过 unified probe"

### 3.1 Surveys

- **Representation Engineering Survey** — Wehner et al., [arXiv:2502.17601](https://arxiv.org/abs/2502.17601) (2025-02)。probe + steering 的规范化 survey，不做 benchmark。
- **Hallucination Survey** — [arXiv:2510.06265](https://arxiv.org/abs/2510.06265) (2025-10)。把检测分成 retrieval- / uncertainty- / embedding- / learning- / self-consistency-based 五家；"embedding-based" ≈ 我们的 2.1 + 2.2。

### 3.2 真正试图 unify 的工作

- **LLM-Check** (§2.3)：三源但无学习融合；breadth 有，unification 没有。
- **Gnosis** (§2.3)：learned fusion，但只 2 源、绑 backbone、label 只有 correctness。
- **HaluNet** — [arXiv:2512.24562](https://arxiv.org/abs/2512.24562) (2025-12)：semantic embedding + logprob + entropy 三路学习融合，但**全是输出侧信号**，没碰 attention / SAE / 跨层轨迹，只在 QA 上评估。
- **UniFact — Towards Unification of HD and FV** — [arXiv:2512.02772](https://arxiv.org/abs/2512.02772) (2025-12)：不是模型，是个评估框架；结论是 **no paradigm dominates**，hybrid 始终最强——这是做 fusion 的直接经验依据。
- **Neural Probe-Based Hallucination Detection** — [arXiv:2512.20949](https://arxiv.org/abs/2512.20949) (2025-12)：跨 task 的神经 probe，但**单一 signal source**。
- **Cross-Layer Attention Probing (CLAP)** — [arXiv:2509.09700](https://arxiv.org/abs/2509.09700) (2025)：跨层 + 跨 token 的 attention probe，但仍**单一 signal source**。

### 3.3 "在哪个 token 上 probe"这条支线

上面 §2 的 baseline 几乎都在 prompt-last 或 gen-last 读。另一条线在质疑这个选择：

- **LLMs Know More Than They Show** — Orgad et al., [arXiv:2410.02707](https://arxiv.org/abs/2410.02707) (preprint)。truthfulness 信号集中在 "exact answer tokens" 上，位置选错就会丢信号。
- **HaMI — Adaptive Token Selection** — Niu et al., NeurIPS 2025 ([arXiv:2504.07863](https://arxiv.org/abs/2504.07863))。MIL 框架下**学出最佳 probe 位置**。
- **First Hallucination Tokens Are Different** — [arXiv:2507.20836](https://arxiv.org/abs/2507.20836) (2025)。第一个幻觉 token 的 hidden/logit 和其他 token 显著不同。
- **Real-Time Entity Probes** — [arXiv:2509.03531](https://arxiv.org/abs/2509.03531) (2025)。对每个实体 token 在线 probe，长文本生成中实时标注。
- **ACT-ViT** — [arXiv:2510.00296](https://arxiv.org/abs/2510.00296) (2025)。把整个 `(layer, token)` 激活张量当图像喂 ViT。

### 3.4 Gap

- 没有 fusion 覆盖全部 signal source（residual + attention + MLP + logit + SAE + multi-sample）。
- 没有 fusion 把 **query-token 位置** 当一等变量；§3.3 的工作说明"位置选择"和"signal 选择"同等重要。
- 没有跨模型家族 / 跨任务的单一权重 fusion；UniFact 的 benchmark 已经说明 hybrid 最强。

这就是 `IDEA_REPORT.md` 的 Multi-View Expert-Library Stacking 想占的位置。

---

## 4. 快速查表

| Baseline | Signal source | Aggregator | Venue | 我们能复现? |
|---|---|---|---|---|
| Geometry of Truth | residual | LR / mass-mean | COLM 2024 Spot. | ✅ |
| SAPLMA / correctness-internals | residual | PCA + LR | Findings-EMNLP 2023 | ✅ |
| SEP | residual (gen-last) | LR | arXiv 2406.15927 (NeurIPS-24 Wksp) | ✅ |
| LID | residual (gen-last) | intrinsic-dim | ICML 2024 | ✅ |
| Chain-of-Embedding | residual trajectory | trajectory stats | ICLR 2025 | ✅ |
| KB-Perception | residual (gen) | MLP | ACL 2025 Main | ✅ |
| STEP | residual (step boundary) | scorer MLP | arXiv 2601.09093 | ⚠️ 需 step→sample 聚合 |
| EigenScore | residual (multi-sample) | eigenvalue spectrum | ICLR 2024 | ⚠️ 缺 10× 采样 |
| SAE Entities | SAE latents on residual | latent selection | ICLR 2025 | ⚠️ Llama 可，Qwen/Mistral 无公开 SAE |
| SAT Probe | attention | attention-flow LR | ICLR 2024 | ✅（差 W_o + span 定位）|
| ITI | per-head attention | LR + steering | NeurIPS 2023 Spot. | ✅ |
| LLM-Check | attn + residual + logit | eigen / entropy | NeurIPS 2024 | ⚠️ PPL 降级 |
| ICR Probe | residual update decomp | top-k/p probe | ACL 2025 Long | ❌ 缺 per-step per-layer raw |
| Gnosis | residual + raw attn | trained head | arXiv 2512.20578 | ❌ 需 raw attn + 重训 head |
| DoLa | intermediate logits | layer contrast (decoding) | ICLR 2024 | ❌ 在线方法 |
| SeaKR | FFN internal + samples | eigen + routing | ACL 2025 Oral | ⚠️ 缺多采样 |
| SR-RAG | uncertainty + routing token | end-to-end | arXiv 2504.01018 | ❌ 无代码 |
| ROME / MEMIT | MLP hidden (causal trace) | rank-one edit | NeurIPS 2022 / ICLR 2023 Oral | ❌ 不是 probe |

Unification 尝试（§3.2）：

| Paper | 融合了什么 | 不足 |
|---|---|---|
| LLM-Check | attn + residual + logit | 无学习融合 |
| Gnosis | residual + attn | 2 源，绑 backbone |
| HaluNet (2512.24562) | embedding + logprob + entropy | 输出侧 only；QA only |
| UniFact (2512.02772) | HD ∪ FV（benchmark） | 结论是 no single dominates |
| Neural Probe (2512.20949) | residual only | 单 source |
| CLAP (2509.09700) | attention only | 单 source |

---

## 附录 A：完整 taxonomy（5 条独立轴）

上面 §2 只用了**轴 1**做分组，其他四轴进了卡片字段。完整的 5 轴如下——在 fusion 设计里，每一轴都对应一个选择。

**轴 1 — Signal source**：residual / attention / MLP / logit / SAE / multi-sample。

**轴 2 — Query-token position**：prompt-last / gen-last / answer-specific（entity / exact
answer / first hallu token）。§3.3 那条支线的论点就是这一轴。

**轴 3 — Aggregator**：linear probe / MLP / 几何统计（eigen, intrinsic-dim, trajectory）/
contrastive decoding / steering / SAE 稀疏分解。

**轴 4 — Supervision label**：correctness / truthfulness / semantic entropy /
knowledge-boundary / constraint-satisfaction / no-sup。

**轴 5 — Application**：detect / decide (RAG routing, pruning) / edit (ITI, ROME) /
decode (DoLa)。

经验观察：方法之间互不兼容主要来自轴 4 和轴 5（label 和应用场景不同），不是轴 1。
所以 fusion 的正确对象是**同一轴-4 标签**下跨轴-1 的异质 expert 组合。
