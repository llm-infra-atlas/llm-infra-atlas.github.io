# 01 · 前沿开源模型架构速览

本文按模型的结构变化整理开放权重模型：先统一参数、上下文与发布口径，再比较主要 family，最后把新增机制连接到 attention、MoE 和 serving 章节。模型名称升级不一定意味着骨架改变；同一个 family 里的 Flash 版本也可能采用独立架构。

**核验日期：2026-10-28。** 最新增补以官方 model card、技术报告与发布仓库为依据；历史表保留用于架构对照，不作为实时性能榜单。API 产品、开放 checkpoint、base/instruct、RL/distillation 版本分别标注，不从托管服务的上下文或能力反推开放权重配置。

---

## 0. 参数口径

读参数时先分清三个量，后面所有表都按这套口径：

| 量 | 它决定什么 | 不是什么 |
|---|---|---|
| **total params** | 权重占显存、EP 切分规模 | 单 token 算力 |
| **active params** | 单 token 的 matmul 量、decode 延迟下限 | 「模型有多大」 |
| **context** | KV / 压缩状态的长度开销 | 训练时一定见过这么长 |

MoE 写成 `E=256, k=8, shared=1`：每层 256 个 routed expert，每 token 选 8 个，外加始终激活的 shared expert。`A` 记法（如 `235B-A22B`）表示 total–activated。

截至本次核验，值得对照的设计方向包括：

1. 大模型经常使用细粒度 MoE，但 total、active 与权重显存仍是不同指标；n-gram/Engram 等条件记忆还引入额外参数口径。
2. 长上下文通过 sparse、linear、窗口、序列压缩与跨层 KV 共享等多种方法降本。固定压缩率和 sparse selection 通常只降低系数，KV 仍可随长度线性增长。
3. Hybrid 不只意味着 linear/full 交替，也包括 linear/sparse、window/global，以及分成 producer/consumer 两段的 causal encoder-decoder。
4. MTP 与其他 speculative decoder 越来越常见，但有些模型采用独立训练的 draft，不应统一记成 backbone 的一层 MTP。
5. Thinking 与 agentic 能力的提升可能主要来自 post-training；GLM-5.3 就明确沿用 5.2 的 base。
6. 发布 dtype 与训练 dtype 分开记录；FP4 权重、FP8 激活和 FP4 KV 分别作用于不同内存与计算路径。

## 近期发布与结构变化

下面是相对原有 2026-08 快照核实到的更新。其余 family 的经典结构保留在后续表中。

| 发布 / checkpoint | 已核实的结构或口径 | 本仓库的阅读入口 |
|---|---|---|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 552B backbone，另有 196B Engram；40 层分成 20+20 的 CED；prefill/decode 激活分别 8B/16B；CSA2、跨层 KV/index 共享、原生图文 | [YOCO 与 CED](./attention/mechanisms/12_cross_layer_kv_yoco.md)、[CSA2](./attention/mechanisms/05_sparse_dsa_frontier.md) |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 125B / 6B，另计 51B n-gram embedding 与 4B MTP；48 层，GDN:QSA=3:1；gated residual | [hybrid](./attention/mechanisms/11_hybrid.md)、[QSA](./attention/mechanisms/05_sparse_dsa_frontier.md) |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | 官方明确与 5.2 使用同一 base，提升来自 post-training | 本文 GLM family；不重复发明一套骨架 |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 独立的 `glm5_next` 原生多模态 checkpoint；与 5.3 的 `glm_moe_dsa` 区分 | 部署前按各自 config 与官方卡片核对 attention/cache 类型 |
| [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) | 70 层，60 SWA + 10 global；384 routed experts、top-8；speculative decoder 为 5 层 SWA；另有 MOPD checkpoint | 本文 MiMo family、[投机解码](./speculative_decoding/README.md) |

截至核验日，DeepSeek 还发布了 [Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)，官方将其标为 V4.1-Flash 的 minor update；这里保留架构名并单独注明 checkpoint，不据此推断又有一套新骨架。相应的 [DeepInfer 推理策略](./serving/09_deepinfer.md)已补入 serving 章节。

这里不把尚未发布的后继名称作为已发布模型。Qwen 官方把 Flash-Next 称为后继架构的 experimental preview，这不等于已经发布了完整的 Qwen4 family。

---

## 1. 代表架构对照

下表保留原有代表模型以对照家族骨架，不按性能排名，也不把较新的小激活版本强行替换较大的前代。2026-08 之后的发布见上表与各 family 补充。

| Family | 代表模型 | Total / Active | 层 | Attention | MoE | Context | 相对前代的关键变化 |
|---|---|---|---|---|---|---|---|
| Kimi | **K3** | 2.8T / 104B | 93 | 69 KDA + 24 Gated MLA；AttnRes | 896 routed, k=16, shared=2（LatentMoE） | 1M | 从全面采用 MLA 换成 hybrid 线性attention，同时更深更稀 |
| Qwen | **3.8-2.4T-A95B** | 2.4T / 95B | 92 | 3:1 Gated DeltaNet / Gated Attention | 512, k=10+1 shared | 262K（可扩 ~1M） | 第一次把 Qwen-Max 级开源；骨架沿 3.5 |
| DeepSeek | **V4-Pro** | 1.6T / 49B | 61 | 交错 CSA / HCA；mHC residual | 384 routed, k=6, shared=1；前 3 层 Hash routing | 1M | 弃 MLA，上压缩+稀疏 hybrid；去掉 node-limited routing |
| GLM | **5.2** | 753B / ~40B | ~80 | DSA + IndexShare（每 4 层共享 indexer） | 256, k=8（5 代骨架） | 1M | 5 代上 DSA；5.2 把 indexer 做成跨层共享以撑 1M |
| MiniMax | **M3** | ~428B / ~23B | — | MiniMax Sparse Attention (MSA) | 128 routed, k=4, shared=1 | 1M | 从 M2 的满attention回到可扩展稀疏attention；原生多模态 |
| MiMo | **V2.5-Pro** | 1.02T / 42B | 70 | 60 SWA + 10 全局；GQA | 384, k=8 | 1M | 3 层 MTP；hybrid 窗口/全局 |
| Hy | **Hy3 preview** | 295B / 21B | 80+1 MTP | GQA 64/8 | 192, k=8 | 256K | 同激活量级里偏「矮胖」；自带 MTP |
| Llama | **4 Maverick** | 400B / 17B | — | GQA + early fusion 多模态 | 128 routed + shared, k=1 | 1M | 家族第一次 MoE；Scout 用 16E / 10M 窗口 |
| OpenAI | **gpt-oss-120b** | 117B / 5.1B | 36 | 交替 SWA-128 / full；GQA 64/8 | 128, k=4 | 128K (YaRN) | GPT-2 以来首个开源权重推理模型 |

> DeepSeek **V4-Flash**（284B / 13B）不是另一套理念，而是 V4-Pro 的小激活对照版本，它与 Pro 共享主要设计路线。Gemma 4 不进这张「大模型」表，它面向**本地 / 单机**部署，见 §9。

---

## 2. DeepSeek：V3 到 V4

官方：V3 [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)；V3.2 [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)；V4 [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) / [HF collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)。

DeepSeek 这条线几乎定义了 2025–2026 开源 MoE 的默认骨架：MLA + DeepSeekMoE（细粒度 routed + shared）+ aux-loss-free bias + MTP。本仓库 [`ep`](./parallel/05_ep/README.md) 里贯穿的 `H=7168, E=256, k=8` 就是 V3 量级。

### 2.1 家族表

| | V3 / V3.1 / R1 | V3.2 | V4-Flash | V4-Pro |
|---|---|---|---|---|
| Total / Active | 671B / 37B | 同左 | 284B / 13B | **1.6T / 49B** |
| 层（含结构） | 3 dense + 58 MoE + 1 MTP | 同左 | 43 | 61 |
| Hidden | 7168 | 7168 | 4096 | 7168 |
| Attention | **MLA**（128 heads） | MLA + **DSA**（lightning indexer, top-k=2048） | **CSA / HCA** hybrid；前 2 层 SWA | **CSA / HCA** hybrid；前 2 层 HCA |
| MoE | 256 routed + 1 shared, k=8；sigmoid 打分；**group-limited**（`group_topk=4`） | 同左 | 256 routed + 1 shared, k=6；前 3 层 Hash routing | 384 routed + 1 shared, k=6；前 3 层 Hash routing |
| Residual | 普通 identity | 同左 | **mHC**（`hc_mult=4`） | **mHC**（`hc_mult=4`） |
| Context | 128K | 128K | **1M** | **1M** |
| 精度 | FP8 训练 | 同左 | expert **FP4** + 其余 FP8 | 同左 |
| 预训练 token | 14.8T | 续训 DSA ~0.94T | 32T | 33T |

**R1** 不是新骨架：权重形状与 V3 相同，差别在 GRPO / 长 CoT post-training。V3.1 相对 V3 也几乎只改数据与对齐。

### 2.2 架构跳变

```
V3/V3.1:  MLA（低秩 KV） + DeepSeekMoE + node-limited top-k
    │
    ▼  continued pretrain
V3.2:    同上 + DSA（indexer 先挑 token，再对子集做 MLA）
    │
    ▼  换 attention / residual，MoE 只微调
V4:      CSA(先按 m=4 压 KV，再 DSA) 交错 HCA(按 m'≈128 重压、不再稀疏)
         + mHC 把 residual 扩成 4 路流形约束
         + 去掉 node-limited；affinity 改为 Sqrt(Softplus)
```

官方给出了 1M 上下文下明确的开销数字：V4-Pro 相对 V3.2，单 token decode FLOPs 约为 **27%**、KV 约为 **10%**；Flash 进一步降到 FLOPs 约 **10%**、KV 约 **7%**。CSA 还保留一条未压缩的 sliding-window 分支补充局部信息，并带 attention sink。

MoE 侧 V4 仍是 DeepSeekMoE，但 expert 变宽（`moe_intermediate_size=3072`，V3 为 2048）、每 token 少激活 2 个、前几层改 Hash routing。MTP 配置与 V3 相同（1 层）。并行含义：V3 的 `group_limited_topk` 是拓扑参数（见 [集合通信：原语、算法、NCCL 实现与拓扑映射](./hpc/04_collectives.md)）；V4 明确**去掉了跨 node 数约束**，靠并行策略把效率补回来。

---

### 2.3 V4.1 的变化

V4.1 不是简单扩大 V4：全局 KV 从 causal encoder 的边界投影生成，CSA2 进一步规定哪些层共享 KV、哪些层重算 top-k。它把「生成 KV 的计算深度」与「使用 KV 的层数」部分解耦，因此 prefill 和 decode 的激活参数量不再相同。SWA 仍保留逐层局部依赖，需要 bounded replay；不能把整个 decoder 当成无状态读出层。细节见 [YOCO 与 CED](./attention/mechanisms/12_cross_layer_kv_yoco.md)。

Engram 则引入由 token/n-gram 索引的条件记忆。与 MoE 选择并执行 FFN 不同，条件记忆主要查询 embedding 表，再与 hidden state 融合；它增加的参数不能直接按同规模 dense matmul 计算成本估算，也不能因访问稀疏就忽略其存储与搬运成本。参数分项及训练设计见 [V4.1 报告](https://arxiv.org/abs/2609.19969)。

## 3. Kimi：K2 到 K3

官方：K2 [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) / [GitHub](https://github.com/MoonshotAI/Kimi-K2)；K3 [arXiv:2607.24653](https://arxiv.org/abs/2607.24653) / [HF](https://huggingface.co/moonshotai/Kimi-K3)。

K2 的经典做法是在 DeepSeek-V3 骨架基础上把稀疏度再提一档：同样采用 MLA 与 DeepSeekMoE 风格的结构，但 expert 更多、attention 头更少。K3 于 2026-07 发布，总参数约 2.8T。

| | K2 / K2.5 / K2.7 Code | **K3** |
|---|---|---|
| Total / Active | 1.04T / 32.6B | **2.78T / 104.2B** |
| 层 | 61（1 dense + 60 MoE） | 93（1 dense） |
| Hidden | 7168 | 7168 |
| Attention | 61× MLA，64 heads | **69 KDA + 24 Gated MLA**（块内 3:1，末层必全局）；96 heads |
| 深度通路 | 普通 residual | **AttnRes**（伪 query 对 embedding + 前块输出做加权） |
| MoE | 384 routed, k=8, shared=1；expert hidden 2048 | **Stable LatentMoE**：896 routed, k=16, shared=2；latent 宽 3584；expert hidden 3072 |
| 激活 | SwiGLU | **SiTU-GLU**（对 SwiGLU 两支做 tanh soft-cap） |
| Context | 128K（K2.7 Code 256K） | **1M** |
| 多模态 | K2.5 起接 MoonViT | 原生；MoonViT-V2 401M / 27 层 |
| 优化器 | MuonClip | Muon + per-head orthogonalization |
| 发布精度 | block-FP8 | QAT：**MXFP4** expert 权重 / MXFP8 激活 |
| MTP | 1 层 | 1 层 |

K3 的 MoE 有一个和 V3 / K2 不同的形状：**LatentMoE**。shared expert 仍走满宽 $d=7168$；routed expert 先把 token 投到 $\ell=3584$ 再计算，因此 896 个 expert 中只激活 16 个这样的稀疏度在计算上才是可行的。负载均衡从 aux-loss-free 的固定步长 bias 换成 **Quantile Balancing**。

Attention 块布局和 Qwen3.5 / 3.8 是同一类 3:1 hybrid（线性层管长序列，周期性全局层管检索）：

```
K3 block:   [KDA → LatentMoE] ×3 → [Gated MLA → LatentMoE] ×1
Qwen3.5/3.8: [Gated DeltaNet → MoE] ×3 → [Gated Attention → MoE] ×1
```

K2.5 / K2.6 / K2.7 的权重形状仍是 K2 这一档（1T-A32B），差别在数据、多模态和对齐，可以不当作新骨架对待。

---

## 4. Qwen：3 到 3.8

官方：Qwen3 [blog](https://qwenlm.github.io/blog/qwen3/) / [HF Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)；Qwen3.5 [HF 397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)；Qwen3.8 [HF 2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)。

Qwen 的跳变不在「更深的 MLA MoE」，而在 **3.5 起换成 Gated DeltaNet hybrid**。3.8 是第一次把 Max 级（2.4T）开源；托管的 Qwen3.8-Max 在这套权重上加了视觉、非 thinking、默认 1M。

| | Qwen3-235B-A22B | Qwen3.5-397B-A17B | **Qwen3.8-2.4T-A95B** |
|---|---|---|---|
| Total / Active | 235B / 22B | 397B / 17B | **2.4T / 95B** |
| 层 | 94 | 60 | 92 |
| Hidden | — | 4096 | 8192 |
| Layout | 标准 Transformer + GQA | `15 × (3× GDN→MoE + 1× Gated Attn→MoE)` | `23 × (3× GDN→MoE + 1× Gated Attn→MoE)` |
| 线性attention | — | GDN：64 V / 16 QK，head 128 | GDN：128 V / 16 QK，head 128 |
| 全局attention | GQA 64Q / 4KV | Gated Attn 32Q / 2KV，head 256，RoPE dim 64 | Gated Attn 64Q / 4KV，head 256，RoPE dim 64 |
| MoE | 128 / k=8 | 512 / **10 routed + 1 shared**；expert inter 1024 | 512 / **10+1**；expert inter 2048 |
| Vocab（padded） | — | 248,320 | 248,320 |
| Context | 32K native / 128K YaRN | 262,144 / 可扩 ~1,010,000 | 同左 |
| MTP | — | multi-step | multi-step |
| 开源权重形态 | 文本；thinking 可关 | 含视觉 encoder | **文本-only，thinking 强制开** |

同代还有 **Qwen3.8-27B** 这种 dense 小旗舰（256K），以及更早的 Qwen3 dense 谱（0.6B–32B）和 Qwen3-30B-A3B。理解 3.8 的关键是：激活从 3.5 的 17B 跳到 95B，expert 池仍是 512，变化的是层数、hidden 和 expert 宽度。

---

### 4.1 Flash-Next 的另一条扩展轴

Flash-Next 保留 3:1 的 GDN hybrid，但全局层改成 Qwen Sparse Attention（QSA）：以 4-token micro-block 为选择单位，预算为 512 blocks，即 2048 tokens；它不是把全局层也改成 recurrent state。原生 context 为 262,144，可扩展至 1M，开放 checkpoint 与托管 Flash 的默认窗口分开记录。

其 51B n-gram embedding 独立于 125B 主模型和 4B MTP，适合研究「增加知识容量却不同比例增加每 token matmul」的路线。Gated Residual 进一步改变深度信息的读写方式。上述定义均来自 [官方 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)，不能仅按名称把它当成 2.4T Qwen3.8 的等比例缩小版。

## 5. GLM：4.5 到 5.3

官方：GLM-4.5 [arXiv:2508.06471](https://arxiv.org/abs/2508.06471)；GLM-5 [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)；GLM-5.2 [Z.ai blog](https://z.ai/blog/glm-5.2) / [NIM card](https://docs.api.nvidia.com/nim/reference/z-ai-glm-5.2)。

GLM 自己的设计哲学和 DeepSeek / Kimi 相反：**宁可更深、hidden 更窄**（4.5 论文原话：deeper models exhibited better reasoning）。5 代接过 DeepSeek 的 DSA，5.2 再把 indexer 做成跨层共享，把窗口推到 1M。

| | GLM-4.5 | GLM-4.5-Air | GLM-5 / 5.1 | **GLM-5.2** |
|---|---|---|---|---|
| Total / Active | 355B / 32B | 106B / 12B | 744B / 40B | **753B / ~40B** |
| 层 | 3 dense + 89 MoE + 1 MTP | 1+45+1 MTP | 80（为减小 EP 通信而减层） | 同 5 代骨架 + IndexShare |
| Hidden | 5120 | 4096 | — | — |
| Attention | **GQA** 96Q / 8KV，partial RoPE，QK-Norm | GQA 96/8 | **MLA-256** + **DSA**（indexer top-k=2048） | DSA + **IndexShare**：每 4 层共用一个 indexer |
| MoE | 160 routed + 1 shared, k=8；sigmoid + loss-free bias | 128 + 1, k=8 | 256 experts, k=8 | 同 5 代 |
| Context | 128K | 128K | ~200K | **1M** |
| 预训练 token | 23T | — | 28.5T | mid-train 起带 IndexShare（128K） |

4.5 相对 V3 / K2 的对照（官方 Table 1）值得记住：同样约 32B 激活，GLM 用 **96 层量级、hidden 5120、160 expert**，而 V3 是 61 层 / 7168 / 256 expert。5 代把 expert 扩到 256、层数压回 80，并在 mid-train 之后用很小的 token 预算（官方写 20B）把 MLA 续训成 DSA。

5.2 的 IndexShare：每 4 个 transformer 层共享一个轻量 indexer，只在这 4 层的第一层算 top-k，后 3 层复用。官方数字是 1M 上 indexer 相关 FLOPs 降约 **2.9×**。KV 体积并不按同样比例下降——长上下文的瓶颈从算力转移到 KV 容量，这是 serving 侧的关键事实。

---

### 5.1 5.3 与 5.3-Flash

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) 的官方卡片明确写明沿用 GLM-5.2 的 base，改进来自 post-training。因此上表中的 5.2 骨架可继续作为这一分支的参考，版本升级不能自动算作 attention 架构升级。5.3-Flash 则是独立发布的原生多模态 `glm5_next` 模型，应按[其自身卡片](https://huggingface.co/zai-org/GLM-5.3-Flash)与配置核对；两个 checkpoint 的许可也应分别查看，不沿用 family 的旧标签。

## 6. MiniMax：Text-01 到 M3

官方：Text-01 / 早期 MiniMax [HF docs](https://huggingface.co/docs/transformers/main/model_doc/minimax)；M2 [arXiv:2605.26494](https://arxiv.org/abs/2605.26494) / [GitHub](https://github.com/MiniMax-AI/MiniMax-M2)；M3 [GitHub](https://github.com/MiniMax-AI/MiniMax-M3) / MSA [arXiv:2606.13392](https://arxiv.org/abs/2606.13392)。

这条线的特点是在稀疏与满 attention 之间来回调整：早期用 Lightning Attention 换长上下文，M2 认为质量不够又回到满 attention，M3 通过自研的 MSA 重新把稀疏做对。

| | MiniMax-Text-01 | M2 | **M3** |
|---|---|---|---|
| Total / Active | 456B / 45.9B | 230B / 10B | **~428B / ~23B** |
| Attention | 每 7 层 Lightning + 1 层 softmax | **全层 GQA**（48Q / 8KV） | **MSA**（GQA 骨架上的 block-sparse；论文实验 block=128, keep=16） |
| MoE | 32 experts（较粗） | 256, k=8 | 128 routed + 1 shared, k=4（config 口径） |
| 层 | 80 | 62 | — |
| Context | 训 1M / 推 4M | ~196K | **1M** |
| 其他 | — | MTP；「mini activations」 | 原生图/视频；官方称 1M 上相对 M2 prefill ~9×、decode ~15×、单 token 算力 ~1/20 |

M2 把激活压到 10B，明确服务 agent 的 plan–act–verify 环。M3 重新提高激活参数量、换用 MSA，并把多模态从第一阶段混训进去。MSA 不是 MLA 那种低秩 KV，而是**在 GQA 上按块选 KV**。

---

## 7. Llama：3.1 405B 与 Llama 4

官方：[Llama 4 herd blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) / [MODEL_CARD](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md)。3.1 405B 是开源 dense 的经典上限，Llama 4 是 Meta 第一次公开 MoE。

| | Llama 3.1 405B（经典 dense） | 4 Scout (17B×16E) | **4 Maverick (17B×128E)** |
|---|---|---|---|
| Total / Active | 405B / 405B | 109B / 17B | **400B / 17B** |
| MoE | 无 | 16 routed + shared，**top-1** | 128 routed + shared，**top-1** |
| 层结构 | dense | 交替 dense / MoE | 同左 |
| 多模态 | 后接 | **early fusion** 原生图文 | 同左 |
| Context | 128K | **10M** | 1M |
| 预训练 token | ~15T 量级 | ~40T | ~22T |
| 发布 | 2024-07 | 2025-04 | 2025-04 |

Scout 和 Maverick **激活同为 17B**，所以单 token 算力接近；差别在 expert 池（16 对 128）和窗口。路由是 dropless token-choice，每层**只进 1 个 routed + 1 个 shared**，比 DeepSeek / Qwen 的 k=8–16 粗一个数量级。Behemoth（~2T / 288B active）是教师模型，权重未发布。

到 2026-08，Llama 4 在 Arena / AA 上已不是开源第一档，但仍是**西方开源 MoE + 超长窗**的参照系。

---

## 8. gpt-oss：OpenAI 的开源推理模型

官方：[Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/) / model card [arXiv:2508.10925](https://arxiv.org/abs/2508.10925)。2025-08 发布，Apache 2.0。它的价值不在于 2026 年的榜单排名，而在于回答了「OpenAI 开源了一个什么样的 Transformer」这份常识。

| | gpt-oss-20b | **gpt-oss-120b** |
|---|---|---|
| Total / Active | 20.9B / 3.6B | **116.8B / 5.1B** |
| 层 | 24 | 36 |
| Residual 宽 | 2880 | 2880 |
| MoE | 32 experts, k=4，SwiGLU | 128 experts, k=4 |
| Attention | 交替 **banded SWA (128)** / full；GQA 64Q / 8KV；head dim 64；**attention sink** | 同左 |
| PE | RoPE + YaRN → 131,072 | 同左 |
| 发布量化 | MXFP4（20b ~16GB，120b 单卡 80GB 可跑） | 同左 |

骨架更接近 GPT-3（Pre-LN、交替局部 / 全局attention）而不是 MLA / DSA 系。5.1B 激活主要降低每 token 计算；权重能装入单卡主要依赖 MXFP4 等存储方式，不能由 active 参数量推断权重显存。

---

## 9. Gemma 4：本地 / 单机开源族

官方：[arXiv:2607.02770](https://arxiv.org/abs/2607.02770)。Apache 2.0；dense 加一个小 MoE。其定位不是和 K3 / V4 竞争总参数规模，而是**单机可部署的多模态 + thinking**。

| | E2B | E4B | 12B | 26B-A4B | 31B |
|---|---|---|---|---|---|
| 形态 | dense + per-layer embed | 同左 | dense，**encoder-free** | **MoE 26B / 3.8B active** | dense |
| 视觉 / 音频 | ViT 150M + USM 305M | 同左 | 无独立 encoder，patch/chunk 直投 | ViT 550M | ViT 550M |
| Attention | 本地 SWA : 全局 = 4:1 | 5:1 | 5:1；全局层 **K 复用为 V** | 同左 | 同左 |
| PE | 局部 RoPE；全局 p-RoPE (p=0.25) | 同左 | 同左 | 同左 | 同左 |
| 其他 | thinking；MTP drafter；QAT | 同左 | 同左 | 同左 | 同左 |

全局 KV 相对朴素 GQA 还能再压缩一截（官方写到 37.5%），靠的是「全局层 values=keys + p-RoPE」。12B 把 550M ViT / 305M 音频 encoder 换成一次大 matmul，是为了减少碎片化显存，而不是为了提升精度指标。

---

## 10. 其余主流开源族

### 10.1 Hy3（腾讯混元）

官方：[README](https://github.com/Tencent-Hunyuan/Hy3-preview)。2026-04 开源。

| | Hy3 preview |
|---|---|
| Total / Active | 295B / 21B（另计 MTP 3.8B） |
| 层 | 80 + 1 MTP |
| Hidden / FFN | 4096 / 13312 |
| Attention | GQA 64Q / 8KV，head dim 128 |
| MoE | 192 experts, k=8 |
| Context / Vocab | 256K / 120,832 |
| 精度 | BF16 |

和 V4 / K3 比，Hy3 没有采用 hybrid / 压缩attention，窗口停在 256K；其特点是**以 21B 激活参数量达到接近 32–37B 激活对手模型的水平**，以及配套的推理栈（vLLM / SGLang + MTP）。

### 10.2 MiMo（小米）

官方：[MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro)；最新增补以 [V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) 为准。

| | V2.5 | **V2.5-Pro** |
|---|---|---|
| Total / Active | 310B / 15B | **1.02T / 42B** |
| 层 | 48（1 dense + 47 MoE） | 70（1 + 69 MoE） |
| Hidden | 4096 | 6144 |
| Attention | 39 SWA + 9 全局 | **60 SWA + 10 全局**；GQA 128/8；QK 192 / V 128 |
| SWA 窗 | 128 | 128 |
| MoE | 256, k=8 | 384, k=8；expert inter 2048 |
| MTP | 3 层 | **3 层** |
| Context | 1M | 1M |

和 Qwen / Kimi 的 3:1 线性 / 全局不同，MiMo 是**窗口attention为主、少量全局层**（约 6:1），更接近 Gemma / gpt-oss 的 SWA 传统，但把 MTP 做成 3 层原生模块，而不是 1 层的附属结构。

V2.6-Pro-RL 仍采用 70 层、60 SWA + 10 global、hidden 6144、384 experts / top-8 的主干；其官方表把 speculative decoder 列为 **5 层 SWA、window 1024**，不能继续照抄 V2.5 的 3 层 MTP。V2.6 同时发布 [Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) 等蒸馏版本，需要把 backbone、post-training 路径和 draft 配置分别比较。

### 10.3 Mixtral

[Mixtral 8×7B](https://arxiv.org/abs/2401.04088)（2023-12）是现代开源 MoE 的参照原点：8 个粗 expert、**top-2**、约 47B total / 约 13B active、32K。2026 的旗舰（k=8–16、数百 expert、shared expert、MLA / DSA / 线性attention）几乎都在与它相反的方向上演进——更细、更稀、attention 不再是满 GQA。新模型不必再列 Mixtral 的分数，但「MoE 是稀疏 FFN、attention 仍 dense」这个默认形态就是从它开始确立的。

---

## 11. 术语速查

| 词 | 一句话 | 往哪看 |
|---|---|---|
| **MLA** | 把 KV 压进低秩 latent，decode 用 MQA 模式共享一份压缩 KV | [Attention 机制](./attention/mechanisms/README.md)；V3 是范本 |
| **DSA** | lightning indexer 先 top-k token，再只对子集做attention；复杂度 $O(Lk)$ | V3.2 / GLM-5 |
| **CSA / HCA** | 先沿序列把每 $m$ / $m'$ 个 KV 压成一条，CSA 再 DSA，HCA 只做重压 | V4 |
| **Gated DeltaNet / KDA** | 线性attention（递推状态，KV 不随 $L$ 涨）；KDA 是带 channel-wise forget 的 delta-rule | Qwen3.5/3.8、K3 |
| **MSA** | GQA 上的 block-sparse：按块选 KV，不是低秩压缩 | MiniMax-M3 |
| **SWA** | 固定窗口局部attention，通常和少量全局层交替 | gpt-oss、Gemma 4、MiMo |
| **mHC** | 把 residual 扩成 $n$ 路，并把混合矩阵约束到双随机流形，防止深栈爆炸 | V4；原文 [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) |
| **AttnRes** | 每层用学到的伪 query 对 embedding 和前块输出做加权，深度上可「回看」 | K3 |
| **DeepSeekMoE** | 细粒度 routed + 少量 shared；aux-loss-free bias；常配 group-limited top-k | [01 · Router 与 Dispatch 前的 Preprocess](./parallel/05_ep/01_router_and_preprocess.md) |
| **LatentMoE** | routed expert 走降维 latent，shared 走满宽；才能支撑 896×16 的规模 | K3 |
| **MTP** | 训练时多步预测，推理当 draft head | 几乎所有 2025+ 旗舰 |
| **YOCO / CED** | 边界 hidden 生成供上层使用的 KV，减少跨层重复与 prompt 计算 | [跨层 KV 共享](./attention/mechanisms/12_cross_layer_kv_yoco.md) |
| **IndexShare** | 多层复用同一个 DSA indexer 的 top-k | GLM-5.2 |

---

## 12. 来源

历史榜单入口（不是本次架构更新依据）：[Artificial Analysis · Large Open Source](https://artificialanalysis.ai/models/open-source/large)、[LMArena](https://arena.ai/leaderboard)。逐条数字以各节链接的 model card / 技术报告为准；V4-Flash 的层数 / hidden / expert 数对齐官方 `config.json` 口径（与 Pro 的 [HF config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base/blob/main/config.json) 同一套字段）。

主要报告：

- DeepSeek-V3 [2412.19437](https://arxiv.org/abs/2412.19437)；V3.2 [2512.02556](https://arxiv.org/abs/2512.02556)；V4 [2606.19348](https://arxiv.org/abs/2606.19348)
- Kimi K2 [2507.20534](https://arxiv.org/abs/2507.20534)；K3 [2607.24653](https://arxiv.org/abs/2607.24653)
- Qwen3 blog；Qwen3.5 / 3.8 HuggingFace model card
- GLM-4.5 [2508.06471](https://arxiv.org/abs/2508.06471)；GLM-5 [2602.15763](https://arxiv.org/abs/2602.15763)；GLM-5.2 [z.ai/blog/glm-5.2](https://z.ai/blog/glm-5.2)
- MiniMax-M2 [2605.26494](https://arxiv.org/abs/2605.26494)；MSA [2606.13392](https://arxiv.org/abs/2606.13392)
- Llama 4 model card；gpt-oss [2508.10925](https://arxiv.org/abs/2508.10925)；Gemma 4 [2607.02770](https://arxiv.org/abs/2607.02770)
- Hy3 [GitHub](https://github.com/Tencent-Hunyuan/Hy3-preview)

下一篇：继续读 [Attention 机制](./attention/mechanisms/README.md) 看生产模型里的 attention 变体；MoE 数据通路见 [Expert Parallelism (EP)](./parallel/05_ep/README.md)。
