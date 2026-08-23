# 04 · 融合与 connector

[`03`](./03_classic_vlms.md) 里介绍的每一家模型都选了自己的一种接法。本篇要把这些接法收敛成一个统一的设计空间：projector 到底在做什么，三种 fusion 方式对 attention 形状和 KV 分别有什么影响，以及训练和推理里都会遇到的那个把 embedding scatter 进序列的实现细节。

读这一篇之前，最好先看过 [`00`](./00_definitions.md) 里关于③projector 的定义，以及 [`02`](./02_encoders.md) 里 $\mathbf{Z}\in\mathbb{R}^{N\times D}$ 的来源，下面会直接用到这两个概念。

---

## 1. Connector = 对齐 + 注入

```
encoder out [N × D_v]  ──projector──►  [N × D_llm]  ──fusion──►  进入 LLM 的某处
                       (对齐宽度/语义)              (拼进序列 or 旁挂 cross-attn)
```

Projector 通常非常轻量（一层线性层或者两层 MLP），但在 LLaVA 的 Stage 1 里，它却是唯一被解冻训练的部分。ModServe 的实测数据显示，connector 占总参数量不到 0.1%，占 TTFT 也不到 0.4%，所以在 infra 层面上从来不会单独把它解耦出来，而是和 LLM 放在一起处理（详见 [`06`](./06_heterogeneity_and_disaggregation.md)）。真正决定 KV 和 prefill 开销的其实是 fusion 的方式。

Megatron 里的 `MultimodalProjector`（[[megatron-lm:megatron/core/models/vision/multimodal_projector.py#L15]]）：

```python
# multimodal_projector.py:47
if self.projector_type == "mlp":
    self.encoder = MLP(config, submodules, input_size=input_size)      # 两层 + GELU
elif self.projector_type == "affine":
    self.encoder = submodules.linear_fc1(input_size, config.hidden_size, ...)
```

LLaVA 用的是 affine，LLaVA-1.5 换成了 mlp。Q-Former 和 Perceiver 属于「重 connector」：它们先把 $N$ 压缩成 $M$，再做投影，具体见 [`03`](./03_classic_vlms.md)。

---

## 2. 三种融合范式

### 2.1 Decoder-only early fusion：视觉 token 当文本 token 用

这是目前的主流做法（LLaVA、Qwen-VL、InternVL、DeepSeek-VL2 都是这样）。Projector 输出的 $N$ 个向量直接占用序列位置，整条序列走标准的因果 self-attention：

```
input:  [<text> <text> <IMG×N> <text> ...]
        └────── 长度 S = T + N，标准 causal attention ──────┘
```

$$
\mathbf{A}=\mathrm{softmax}\Bigl(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_h}}+\mathbf{M}_{\mathrm{causal}}\Bigr)\in\mathbb{R}^{S\times S},
\qquad S=T+N.
$$

这带来的后果是：

- 复杂度是 $O((T+N)^2)$。当动态分辨率让 $N$ 涨到 3000 以上时，视觉部分会主导整个 prefill 的开销。ModServe 的实测数字是，在同样尺寸下，decoder-only 的 prefill 大约比 cross-attn 方案贵 10 倍。
- 全部 $N$ 个视觉 token 都会进入 KV cache：

$$
\mathrm{KV}_{\mathrm{bytes}}=2\cdot L\cdot(T+N)\cdot n_{\mathrm{kv}}\,d_h\cdot\mathrm{dtype}.
$$

- 它的好处是不需要改动 LLM 的结构，而且视觉和文本能在每一层里深度交互。这正是它能够成为默认选择的原因。

### 2.2 Cross-attention fusion：视觉信息旁挂

Flamingo 和 Llama 3.2 Vision 走的是这条路。主序列里只有文本，长度是 $T$；层与层之间插入 gated cross-attn，Q 来自文本，K/V 来自视觉（Flamingo 用的是 64 个 Perceiver token）。具体公式见 [`03 §2.1`](./03_classic_vlms.md)。

这样做的后果是：self-attention 仍然是 $O(T^2)$，但需要另外维护一份 image-side 的 KV；精度通常会略低一些（ModServe 的实测大约低 5 个百分点）。LM 主干在这种方案下通常保持冻结，这会影响训练时 PP 的负载分配，详见 [`06`](./06_heterogeneity_and_disaggregation.md) 里关于 Cornstarch 的部分。

### 2.3 离散统一 / 解耦统一

![Chameleon：VQ tokenizer 把图变成离散 token，与文本共用一个 transformer](assets/arxiv/2405.09818_chameleon.png)

> 图：图像经过 VQ 进入词表，理解和生成都变成 next-token 预测。在 infra 层面，整条流水线又变回了同构的 transformer，图像 token 也能像文本一样复用 KV。（Chameleon Team 2024, teaser；[arXiv:2405.09818](https://arxiv.org/abs/2405.09818)）

- **Chameleon**：codebook 大小 8192，$512^2\to 1024$ 个 token，词表大小 65536；从零开始训练；用 QK-Norm 和 z-loss 来稳定训练过程。
- **Emu3**：完全用 next-token 预测，理解和生成都不依赖 diffusion。
- **Janus**：理解走连续路径（SigLIP），生成走离散路径（VQ），共享同一个自回归 Transformer，详见 [`03 §7`](./03_classic_vlms.md)。

统一范式让多模态退化成了一条纯粹的 token 序列，因此可以直接享受 LLM serving 那套现成的基础设施，比如 KV cache 和 continuous batching。代价是要么承受 codebook 带来的信息损失，要么像 Janus 那样维护两座独立的塔。

---

## 3. 对照表

| 维度 | decoder-only 拼接 | cross-attention | 离散统一 |
|---|---|---|---|
| 视觉在哪 | 主序列 | cross-attn 的 K/V | 主序列（离散 id） |
| 主序列长度 | $T+N$（$N$ 主导） | $T$ | $T+N_{\mathrm{vq}}$ |
| self-attn | $(T+N)^2$ | $T^2$ + xattn | $(T+N)^2$ |
| 视觉进 KV？ | 是，全部 $N$ | image-side KV | 是，和文本无异 |
| 改 LLM 结构 | 否 | 是（插层） | 否（需 VQ） |
| 代表 | LLaVA / Qwen-VL / InternVL | Flamingo / Llama 3.2 V | Chameleon / Emu3 / Janus 生成支路 |

开源生态目前以 decoder-only 为绝对主流，后面几篇讨论 infra 问题时也会默认按这条路径来展开：image token 会进入 KV，prefill 会被 $N$ 拉长。

---

## 4. Scatter：把占位符换成 embedding

Decoder-only 方案里反复出现一个工程动作：先在序列里用占位符标记图像所在的位置，等到 forward 的时候，再按照这些位置把 encoder 的输出写进去。

### 4.1 训练侧：Megatron 的 `_preprocess_data`

`LLaVAModel`（[[megatron-lm:megatron/core/models/multimodal/llava_model.py#L57]]）同时持有 vision、projection、language 三部分。它把 `image_token_index` 定义为 `-200`（[[megatron-lm:megatron/core/models/multimodal/llava_model.py#L49]]）。`_preprocess_data`（[[megatron-lm:megatron/core/models/multimodal/llava_model.py#L482]]）会把 `-200` 替换成对应的 image embedding，同时把对应的 label 改成 `-100`、loss_mask 置为 0。

Docstring 里明确处理了 PP 切分下的情况：只有 `pre_process=True` 的第一个 LM chunk 会去改 input embedding，只有 `post_process=True` 的最后一个 chunk 会去改 label。`add_encoder` / `add_decoder`（[[megatron-lm:megatron/core/models/multimodal/llava_model.py#L116-L117]]）决定了当前这个 rank 具体要构造哪一部分，这正是 [`06`](./06_heterogeneity_and_disaggregation.md) 里讲的解耦方案的最小实现单元。

### 4.2 推理侧：SGLang 与 vLLM

SGLang 的 `embed_mm_inputs`（[[sglang:python/sglang/srt/managers/mm_utils.py#L901]]）做的是 masked scatter：先用 `embed_tokens` 得到文本 embedding，再按照 mask 原地覆盖掉图像所在的位置。这个流程只在 prefill 阶段、且 batch 里确实含有多模态输入时才会执行；decode 阶段会直接短路，因为图像信息已经在 KV 里了。

vLLM 的 `_merge_multimodal_embeddings`（[[vllm:vllm/model_executor/models/utils.py#L479]]）也是类似的逻辑：

```python
inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
```

三家实现在结构上是一致的：用占位 token 标记位置，encoder 算出 embedding，再按照 mask 或者 index 把它写进去。encoder 和 backbone 之间的强耦合，其实就集中在这一个张量上，这也是为什么它们后来能被拆分成独立服务的物理原因。

---

## 5. 小结

Projector 负责对齐宽度，在 infra 层面基本可以忽略；真正决定序列长度和 KV 开销的是 fusion 的选择。目前的主流做法是用「序列变长」换取实现上的简单和精度上的优势。而占位符到 embedding 这一步替换操作，正是 encoder 之所以能被独立解耦出来的物理依据。

---

接下来是 [05 · 生成器：diffusion / DiT / 自回归图像](./05_generation.md)。如果模型还需要真正「画出」图像，第三条流水线要处理的就是这个问题：走的是扩散路线，还是自回归路线。
