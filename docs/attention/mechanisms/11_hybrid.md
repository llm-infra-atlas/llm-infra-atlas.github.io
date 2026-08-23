# 11 · 混合模式：层比例、NoPE 与正反证据

> 到 2026 年，几乎没有 frontier 模型是「纯」的：要么是稀疏层与全局层交替（Gemma 3 5:1、GPT-OSS 1:1、Llama 4 3:1），要么是线性层与全 attention 层交替（Kimi K3 3:1、Qwen3.5 3:1、MiniMax-01 7:1、Nemotron-H 约 8%）。本篇讲这些比例从哪来、怎么排布、为什么 NoPE 突然变得重要，以及一件很多中文材料不谈的事：MiniMax 在 230B 规模上试过 hybrid 然后退回了全 attention，并公开写了理由。
>
> **前置**：[`03`](./03_sparse_static.md) §4（Gemma/GPT-OSS/Llama 4 的层间交替）、[`09`](./09_linear_kda_kimi.md)（KDA + MLA 的 3:1）、[`02`](./02_position_and_stability.md) §3.2（NoPE）。统一记号见 [Attention 机制](./README.md) §2。
>
> 论文：MiniMax-M2 [arXiv:2605.26494](https://arxiv.org/abs/2605.26494)、Kimi K3 [arXiv:2607.24653](https://arxiv.org/abs/2607.24653)、Kimi Linear [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)、Falcon-H1 [arXiv:2507.22448](https://arxiv.org/abs/2507.22448)、Nemotron-H [arXiv:2504.03624](https://arxiv.org/abs/2504.03624)、Jamba [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)。

---

## 1. 为什么需要混合

```
纯 sparse：  KV cache 仍随 L 增长（要留着供打分）
             表达力上限 = full attention（只做选择，不做压缩）

纯 linear：  状态 O(1)，但 finite-state capacity 硬约束
             copying / 精确检索任务上理论弱于 Transformer（Jelassi et al. 2024）

纯 full：    O(L²) 计算 + O(L) cache，1M 上下文不可行
```

混合的逻辑很朴素：**让大多数层便宜，少数层保留全部能力**。少数全局层负责跨全上下文的精确检索，多数便宜层负责局部建模和信息压缩。

Kimi Linear 消融里那一行纯 MLA（0:1）val PPL 5.77 对 3:1 的 5.65 值得注意——混合不只是省钱，它本身让质量更好。这和 [`04`](./04_sparse_trainable.md) §7 里 NSA 超过 full attention 是同一类现象：稀疏/压缩约束是一种有益的归纳偏置。

## 2. 两种混合拓扑

### 2.1 层间交替（layer-wise interleaving）

这是主流做法。

```
[ cheap, cheap, cheap, full ] × N        ← 3:1
```

Kimi Linear 选它的理由很实际："its superior **infrastructure simplicity and training stability**"——每层只有一种算子，TP/PP 切分、CUDA graph 捕获、prefix cache 管理都不用为「层内两种算子」做特例。

**一条被两代 Kimi 明确写下的规则：最末层必须是全局层。**

- Kimi Linear 27 层：`full_attn_layers = [4, 8, 12, 16, 20, 24, **27**]`，即 $[K,K,K,M] \times 6 + [K,K,M]$
- Kimi K3 93 层："An additional Gated MLA layer is placed at **the end of the backbone**, ensuring that the **final layer always performs global attention**."

原因在于：最后一层的输出直接进 LM head，如果它只有一个有损压缩的状态可用，任何需要精确回看的预测都无法完成。

Gemma 3 反向选择了首层是 local（"starting with a local layer as the first layer"）——首尾两端的选择是不对称的，这个细节值得留意。

### 2.2 层内并行（intra-layer parallel）

代表是 Falcon-H1 与 Hymba。

```
        ┌── attention ──┐
x ──────┤               ├──► concat / average ──► W_o
        └── SSM ────────┘
```

**Falcon-H1** 用 **concat**，所以两路的通道数可以不等——实测最优约 1/8 的通道给 attention。**Hymba** 用 **average**，这要求两路维度相同、无法调比例——Falcon-H1 论文明确指出这是其缺陷。

concat 与 average 是一个真实的设计差别：concat 让「比例」成为一个连续可调的超参（且 `W_o` 能学到怎么融合），average 把它锁死在 1:1。

Falcon-H1 还用了极高的 RoPE base（$b \approx 10^{11}$）把位置编码推到 near-NoPE（[`02`](./02_position_and_stability.md) §4）——层内并行的模型同样需要解决「两种位置信号冲突」的问题（§4）。

## 3. 各模型的层比例

| 模型 | 便宜层 | 全局层 | 比例 | 排布细节 |
|---|---|---|---|---|
| **Kimi Linear 48B-A3B** | KDA | MLA(NoPE) | **3:1** | 27 层 = 20 + 7，$[K,K,K,M] \times 6 + [K,K,M]$，末层 M |
| **Kimi K3 2.8T-A104B** | KDA | Gated MLA(NoPE) | **3:1** | 93 层 = 69 + 24，末层 M |
| **Qwen3-Next-80B-A3B** | Gated DeltaNet | Gated Attention | **3:1** | 48 层 = $12 \times (3\ \text{GDN} + 1\ \text{Attn})$ |
| **Qwen3.5**（397B-A17B 等） | Gated DeltaNet | Gated Attention | **3:1** | HF `layer_types` 是逐层字符串列表 |
| **MiniMax-Text-01 / M1** | Lightning Attention-2 | softmax GQA-8 | **7:1** | 80 层 |
| **MiniMax-M2** | —— | **全 full attention** | —— | **回退**，见 §5 |
| **Nemotron-H 8B** | Mamba-2 | GQA-8 | **4 / 52 层 ≈ 8%** | **NoPE**；hidden 4096，SSM state 128 |
| **Nemotron-H 56B** | Mamba-2 | GQA | **10 / 118 层** | hidden 8192，SSM state 256 |
| **Nemotron Nano 2 (12B-v2)** | Mamba-2 | GQA 40q/8kv | **6 / 62 层** | 28 FFN + 28 Mamba2 + 6 attn |
| **Jamba 52B-A12B** | Mamba-1 | attention | **1:7**（`a:m`, block `l=8`） | 4 个 Jamba block；MoE 每 2 层，16 experts top-2；256K |
| **IBM Granite 4.0-H** | Mamba-2 | transformer | **9:1** | Micro 3B dense / Tiny 7B-A1B / Small 32B-A9B |
| **Falcon-H1** | Mamba-2 | attention | **层内并行**，≈1/8 通道 | concat；0.5B–34B |
| **Hymba** | Mamba | attention | **层内并行 1:1**（average） | 无法调比例 |
| **Gemma 2 27B** | SWA(4096) | global(8192) | **1:1** | 每隔一层 |
| **Gemma 3 27B** | SWA(1024) | global | **5:1** | **首层 local** |
| **GPT-OSS 120b** | SWA(128) | full | **1:1** | + per-head learned sink |
| **Llama 4 Scout** | chunked(8192)+RoPE | full+**NoPE** | **3:1** | `no_rope_layers=[1,1,1,0,…]` |
| **Gated DeltaNet-H1 / H2** | GDN（+Mamba2） | SWA | 交替 | H2 = Mamba2→GDN→SWA |
| **RWKV-7 Goose** | RWKV-7 | **无** | 100% linear | 0.1B–2.9B（+7.2B/13.3B） |

### 3:1 的实测依据

3:1 这个比例从哪来？Kimi Linear 的消融是最直接的依据（[`09`](./09_linear_kda_kimi.md) §5 那张表）：

| 比例 | Train PPL | Val PPL | 原文解读 |
|---|---|---|---|
| 0:1（纯 MLA） | 9.45 | 5.77 | "**performed poorly**" |
| 1:1 | 9.29 | 5.66 | "maintained a similar validation loss but at the cost of **increased inference overhead**" |
| **3:1** | **9.23** | **5.65** | 最优 |
| 7:1 | 9.23 | 5.70 | "comparable training loss but led to **significantly worse validation** performance" |
| 15:1 | 9.34 | 5.82 | —— |

注意 3:1 和 7:1 的 train PPL 完全一样（9.23）而 val PPL 差 0.05。这是过拟合式的信号：7:1 下全局层太少，模型在训练分布上够用，泛化就不够了。

Qwen 的官方消融也落在同一处："we find that **Gated DeltaNet offers stronger in-context learning ability than commonly used methods like Sliding Window Attention or Mamba2**. When we mix Gated DeltaNet with standard attention at a **3:1 ratio** …, the model **consistently outperforms any monolithic architecture**."

注意这句话里的两层结论：第一，GDN 比 SWA 和 Mamba2 都强（三种「便宜层」的横向对比）；第二，3:1 混合优于任何纯架构。

### Mamba 系与 KDA 系的比例差异

Nemotron-H 只有 8% 的层是 attention，Jamba 只有 12.5%，而 Kimi/Qwen 都是 25%。一个直观的猜想是：Mamba-2 是标量衰减、表达力弱于 KDA/GDN，所以更依赖全局层来弥补？恰恰相反——比例低说明它更能靠自己撑住。更可能的解释是目标场景不同：Nemotron-H/Jamba 优化的是通用长文本吞吐；Kimi K3/Qwen3.5 要撑住 long-CoT reasoning 和 agentic 任务，那些任务对精确回看的要求高得多（这也正是 §5 里 MiniMax 批评的核心）。

这一点我没有找到直接的一手论据，标为推测。但它与 §5 的证据方向一致。

## 4. NoPE 在混合架构中的作用

这是本篇最值得单独讲的一节，因为它把 [`02`](./02_position_and_stability.md)、[`01`](./01_basics_head_sharing.md) §4.5、[`09`](./09_linear_kda_kimi.md) §4 三处伏笔一次收拢。

### 两种位置信号的冲突

Kimi Linear 的诊断（原文）：

> "In Kimi Linear (RoPE), the global attention layer carries a **strong, explicit relative positional signal**, while the linear attention contributes a **weaker, implicit positional inductive bias**. This **mismatch yields an overemphasis on short-range order in the global layer**, which benefits short contexts but makes the model **less flexible when adapting mid-training to extended contexts**."

实测代价（Table 5，128k）：RULER **84.3（NoPE）对 78.8（RoPE）**。

### 解法与附带收益

既然 KDA 本身就是「数据相关的乘性位置编码」（[`09`](./09_linear_kda_kimi.md) §4），就让它独揽位置编码的责任，全局层一律 NoPE：

```
① MLA + NoPE ⇒ 推理时退化成【纯 MQA】       （[01] §4.5：吸收后 n_kv=1；NoPE 去掉最后那个 d_h^R）
② 长上下文训练不需要 RoPE 调参               （免 frequency base tuning、免 YaRN）
```

第一条是纯粹的附带收益：压缩后 $d_c = 512$（原来 $d_c + d_h^R = 576$），而且吸收路径上再没有任何位置相关的矩阵卡在中间。

### 三家的做法对照

| 模型 | 便宜层带位置吗 | 全局层 | 机制 |
|---|---|---|---|
| **Kimi Linear / K3** | ✓ KDA 的乘性衰减 | **NoPE** | 衰减即位置 |
| **Llama 4** | ✓ chunked + RoPE | **NoPE** + inference-time attn temperature scaling | 局部层管短程顺序 |
| **Nemotron-H** | ✓ Mamba-2 的衰减 | **NoPE** | 同 Kimi |
| **Falcon-H1** | ✓ Mamba-2 | RoPE base $\approx 10^{11}$（**near-NoPE**） | 用极大 base 逼近 NoPE |
| **MiniMax-Text-01** | ✓ Lightning 的 $\lambda$ | **partial RoPE 0.5** | 折中：一半维度带位置 |
| **Qwen3-Next** | ✓ GDN 的衰减 | **partial RoPE 0.25** | 折中，只 64/256 维带 RoPE |
| Gemma 3 | SWA + RoPE θ=10k | RoPE θ=1M + linear scaling | **两种都带**，靠 θ 区分尺度 |

```
一条清晰的规律：
    「便宜层已经提供位置信息」的架构，全局层倾向于 NoPE 或 partial RoPE；
    「便宜层也是 softmax attention」的架构（Gemma 3），两种层都带 RoPE，靠 θ 区分尺度。
```

为什么 NoPE 在纯 Transformer 里没流行、在混合架构里流行了：[`02`](./02_position_and_stability.md) §3.2 的 NoPE 论文只在约 107M 规模验证过。纯 decoder 靠 causal mask 确实有隐式顺序，但那个信号很弱。**混合架构提供了一个强得多的替代来源**——线性层的乘性衰减是显式、可学习、数据相关的 recency 信号。所以 NoPE 不是「不要位置编码」，而是把位置编码从 attention 挪到了 recurrence 里。

## 5. MiniMax-M2 的回退

这是一场没有定论的争论，也是很多中文材料不谈的部分，但它是这个领域最有价值的负面结果。

MiniMax-Text-01 是 linear attention 第一次上到 456B（7:1 hybrid）。而 MiniMax-M2 回退到了全 attention，并公开写了理由（[arXiv:2605.26494](https://arxiv.org/abs/2605.26494) 与 [HF 博客](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)）：

> "M2 adopts **full multi-head attention across all layers**, departing from the hybrid design used in MiniMax-Text-01… Despite the theoretical appeal of efficient attention mechanisms, we found **no variant that reliably matches full attention quality in production settings** spanning reasoning, coding, and agent tasks."

### 四条具体理由

**① 基准饱和造成的误判。** 开发 Text-01 时业界还在评 MMLU/BBH/MATH/LongBench（现已饱和）："From the perspective of a year ago, a hybrid of Lightning Attention and Full Attention looked just as good as pure full attention… **Not quite. The price paid became obvious at a larger scale**: the model had clear deficits in **complex, multi-hop reasoning tasks**."

**② attention 模式被破坏（最技术性的一条）。** 把预训练好的 full attention 模型转成 Lightning Attention hybrid 后，长上下文 agentic 任务性能显著下降。归因于 hybrid 无法维持预训练中形成的关键 attention 模式——retrieval heads、induction heads、长程一致性机制。"They tried detecting critical heads and keeping only those as FA, but they weren't able to reliably identify and retain all the patterns."

**③ 代理指标不可靠。** "We developed proxy metrics for this specific weakness and iterated until the hybrid model seemed to match MHA. But **does that proxy metric still correlate with real-world downstream performance at an even larger scale**? Are there other hidden weaknesses? Who knows."

**④ 基建成熟度与交叉点问题。**

- "many of them are **memory-bound — even during training**. Without extreme IO optimization, you're basically leaving a huge amount of GPU FLOPs on the table."
- 交叉点："Linear attention has linear compute complexity and constant memory usage. That means there's a crossover point where it becomes more efficient than full attention. **In theory, that point lies at a few thousand tokens — which isn't particularly long for today's large models.**"（暗示实际交叉点远高于理论值。对照 [`09`](./09_linear_kda_kimi.md) §6 算出的 $T \approx 496$ 理论交叉点——这正是 MiniMax 说「理论上几千 token」时指的那个量级，而它的批评是实际值高得多。）
- 数值精度：linear attention 对低精度远比 full attention 敏感（对照 [`08`](./08_linear_delta_rule.md) §4 那张 RWKV-7 状态范围图——状态数值稳定性是这条路的硬前提）。
- prefix caching、speculative decoding 的基建不成熟。

M2 的 SWA 实验也失败了："During pre-training, all variants showed degraded performance on retrieval, multi-hop reasoning, and in-context learning tasks… After SFT, the gap became more pronounced specifically at long context: on benchmarks exceeding 32K context (agent tasks and complex long-context evaluations), **SWA variants performed significantly worse**."

MiniMax 自己的中文表述比论文更平衡：「一直在做，但是在工业系统里真的打过 Full Attention 还有些距离」——也就是说，问题不是「线性 attention 有根本缺陷」，而是「在当前基建下的工业级规模上还没有取胜」。

### 正方证据

| 证据 | 内容 |
|---|---|
| **Kimi K3 上到 2.8T-A104B** | KDA:Gated MLA = 3:1，1M 上下文，全 NoPE，已发布 |
| **Qwen3.5 上到 397B-A17B** | GDN:Gated Attention = 3:1，从 Qwen3-Next 预览到主线生产 |
| **Kimi Linear 的 RL 阶段结论** | "in reasoning-intensive long-form generation under RL, we empirically observe that **Kimi Linear performs significantly better than MLA**"，且差距随训练拉大 |
| Nemotron-H / Granite 4.0-H / Falcon-H1 | 多家在中等规模上稳定发布 |

### 分歧的可能原因：从头训练与事后转换

我的判断是，关键变量不是规模，而是「从头训」还是「转换」。把两边的做法并排：

```
MiniMax：把【已经预训练好的 full attention 模型】转换成 hybrid
         ⇒ 破坏了预训练中形成的 retrieval head / induction head 结构（它自己的理由 ②）

Kimi：   from scratch 训练 hybrid
         ⇒ 模型从第一步就在 hybrid 约束下组织信息流，
           不存在「已形成的 attention 模式被破坏」这回事
```

这很可能是两家结论分歧的核心原因。而且这个解释与 [`04`](./04_sparse_trainable.md) §6.2 里 NSA 的论据结构完全一致——NSA 也发现「post-hoc 上稀疏」不如「原生训练稀疏」，理由同样是 dense 预训练形成的 retrieval head 经不起改造。

两条路线在同一个问题上得到了同一个答案：机制改动要么从头训，要么就得付出结构被破坏的代价。

时间线也重要：MiniMax-M2 的判断早于 Kimi K3 的 2.8T 验证。这场争论还没结束，但证据的天平在 2026 年偏向了「从头训的 hybrid 可行」。

## 6. 决策表

什么约束下选什么方案：

| 你的约束 | 建议 | 理由 |
|---|---|---|
| 上下文 ≤ 32K，追求极致质量 | **纯 full attention + GQA/MLA** | 交叉点之前 linear 不划算（[`09`](./09_linear_kda_kimi.md) §6）；FA2/3 在短序列上仍快于 linear kernel（[06 · Flash Linear Attention](../fa/06_flash_linear_attention.md) §9 的实测） |
| 128K，内存受限，不想改训练流程 | **静态层间交替（Gemma 3 式 5:1 SWA）** | 几乎免费的 8× 内存；质量对窗口和比例都不敏感（[`03`](./03_sparse_static.md) §4） |
| 128K+，有 dense frontier checkpoint 想复用 | **DSA 式蒸馏稀疏** | 2.1B + 943.7B token 适配即可持平，且能降价 50%（[`05`](./05_sparse_dsa_frontier.md) §4） |
| 1M，从头预训练，要极致 KV cache | **linear hybrid 3:1 + NoPE** | Kimi Linear 0.88 GiB @128K vs 全 MLA 8.58 GiB；末层留全局 |
| 1M，从头预训练，要极致检索精度 | **CSA/HCA 式压缩+稀疏** | 保留完整历史（虽然压缩过），表达力上限不受 finite-state 约束 |
| 已有 full attention 模型，想事后改 hybrid | ⚠️ **谨慎** | MiniMax-M2 的教训（§5 理由②） |
| 中小规模、纯线性 | RWKV-7 / 纯 GDN | RWKV-7 有复杂度类层面的表达力保证 |

## 7. 三条路线的对照

作为全章收束，把三条路线最终对照如下：

| | ① 基础（head sharing） | ② sparse | ③ linear |
|---|---|---|---|
| **改什么** | KV 的表示 | mask（选择性跳过） | 递归（有损压缩） |
| **每 token 状态** | $2 h_{\mathrm{kv}} d_h \to d_c + d_h^R$ | 不变（仍需全 cache） | **$d_k d_v$，与 $L$ 无关** |
| **decode 计算** | $O(Ld)$ | $O(kd)$ | **$O(d_k d_v)$** |
| **表达力上限** | = full attention | = full attention（只做选择） | **可超过**（delta rule + 非对角转移） |
| **主要风险** | MQA 的微调不稳定 | 打分器质量 / post-hoc 剪枝损失 | finite-state capacity / 低精度敏感 |
| **代表** | MLA（DeepSeek-V2/V3、Kimi K2） | NSA → DSA → CSA/HCA | GLA → GDN → **KDA**（Kimi Linear/K3） |
| **共同结局** | 都进了 hybrid | 都进了 hybrid | 都进了 hybrid |

最后一行是全章的结论。2026 年的实际情况是：MLA 提供了廉价的全局层、sparse 和 linear 各自提供了廉价的多数层，而 NoPE 让它们能干净地拼在一起。没有哪条路线单独赢了——赢的是「按层混合」这个想法本身。

而三条路线共享同一个更深的教训，值得作为本章的结尾：**机制改动必须从预训练第一步就在场。** NSA 的原生训练论据、DSA 的两阶段蒸馏、MiniMax 的转换失败、Kimi 的从头训成功——四份证据指向同一个方向。attention 机制不是一个可以事后替换的算子，它塑造了模型内部信息流的全部结构。

---

回到 [Attention 机制](./README.md) 看完整的地图与账本表。kernel 侧的实现在 [05 · Flash Sparse Attention](../fa/05_flash_sparse_attention.md)（稀疏）与 [06 · Flash Linear Attention](../fa/06_flash_linear_attention.md)（线性，逐行对齐 [[fla:]]）。
