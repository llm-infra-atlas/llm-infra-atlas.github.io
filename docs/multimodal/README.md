# 多模态 —— 架构视角

这一组文档回答的是模型层面的问题：像素和波形是怎么变成 token 的、视觉塔用什么目标训出来、各家 VLM 怎么把视觉接进 LLM、需要生成图像时走的是扩散还是自回归。它和 [推理服务：从单请求推理到 SLO-aware 集群](../serving/README.md)、[大规模训练的并行策略 —— 总览](../parallel/README.md) 分工不同，可以对照着看：

| | 本章（多模态架构） | serving / 并行策略 |
|---|---|---|
| 视角 | 架构 / 算法 | infra / 调度与并行 |
| 核心对象 | 对比损失、patch token、connector、融合范式、去噪 ODE | EPD 解耦、packing、`cu_seqlens`、encoder cache |
| 变长 | $N = HW/P^2$ 从哪来、为什么逐样本波动 | 按 token 而不是按样本做负载均衡 |

## 前置知识

本章假设读者具备以下背景：

- 知道 decoder-only Transformer 与 next-token prediction；相关的训练视角可对照 [训练全景：从数据到权重更新](../train/00_overview.md)。
- 熟悉 scaled dot-product attention 的 shape；不熟悉的可以先读 [Attention 总览](../attention/README.md)。
- 对比损失、ViT 的 patchify、diffusion 的前向/反向过程这几块知识，正文里会分别给出定义，不需要提前准备。

---

## 0. 整体图景

现代多模态模型大体上可以拆成三段：一个（或多个）模态 encoder，一个负责对齐的 connector，一个 LLM backbone，再加上一个可选的生成器。Encoder 把像素或波形压缩成一串 $N\times D$ 的 embedding；connector 把这串向量的宽度对齐到 LLM 的 hidden size；至于这串向量最终以什么方式和文本交互——是直接拼进主序列，还是旁挂在 cross-attention 上——则由融合方式决定。这里的 $N$ 由分辨率或时长决定，逐样本会剧烈波动，后面所有关于变长的讨论都从这里发端。

画成一条三段流水线大致是这样：

```
   ┌───────────────┐      ┌────────────────────────┐      ┌──────────────────┐
   │ modality       │     │ LLM backbone            │      │ modality          │
   │ encoder(s)     │ ──► │ (Transformer decoder)   │ ──► │ generator         │
   │ ViT / Whisper  │ emb │  prefill + decode       │ tok │ diffusion / AR    │
   └───────────────┘      └────────────────────────┘      └──────────────────┘
   像素/音频 → token          自回归理解 + 生成 text          可选：渲染回像素
```

三段各自的计算特征差别很大，这一点是后面 `06`–`08` 三篇 infra 讨论的共同起点：

| 段 | 典型算子 | 算力画像 | 一次请求跑几遍 | KV |
|---|---|---|---|---|
| **encoder** | ViT / 双向 attention，窄而浅 | compute-bound，一次性 | 1 | 无 |
| **backbone · prefill** | 大 GEMM、causal attention | compute / 带宽混合 | 1 | 写 |
| **backbone · decode** | GEMV、逐 token | memory-bound、串行 | N 步 | 读+追加 |
| **generator** | DiT / U-Net，双向 | compute-bound、迭代 | 20–1000 | 无 |

由于这四段的计算特征差别如此之大，没有任何单一的并行配置或 batch 大小能让它们同时被喂饱，训练和推理系统最终都会走向按 stage 解耦的方案。

全章还有两条相互独立的演化轴，会在多篇文档里反复出现：

```
轴 1 · 视觉怎么变成 LLM 能读的东西
  CLIP 双塔对比 ──► 冻结塔 + 可训 connector（Flamingo / BLIP-2 / LLaVA）
                 ──► 解冻 ViT + 动态分辨率（InternVL / Qwen2-VL）

轴 2 · 视觉信息在哪和文本交互
  decoder-only 拼接（LLaVA 家族，主流）
  cross-attention 旁挂（Flamingo / Llama 3.2 Vision）
  离散统一 token（Chameleon / Emu3）；理解/生成解耦（Janus）
```

---

## 1. 这组文档怎么读

下面这张表给出了每篇文档大致覆盖的内容和对应的关键锚点，方便按需跳读：

| 文件 | 内容 | 锚点 |
|---|---|---|
| `README.md`（本文） | 整体图景、两条演化轴、与 serving/parallel 的分工、阅读顺序 | —— |
| [00 · 术语定义：encoder / projector / fusion / VLM](./00_definitions.md) | 词汇辨析：VLM / encoder / projector / fusion / vision token / tile；三个都叫 projection 的东西 | —— |
| [01 · 对比预训练](./01_contrastive_pretrain.md) | 视觉塔从哪来：CLIP 对称 InfoNCE、温度、batch 大小；SigLIP 的 pairwise sigmoid；为什么几乎所有开源 VLM 都从这里起步 | CLIP Fig 1；SigLIP Eq. sigmoid |
| [02 · Encoder：像素/音频到变长 token](./02_encoders.md) | encoder 内部：ViT patchify 的数学、token 数 $N=HW/P^2$、变分辨率（NaViT / AnyRes / 动态 tile）、压缩（pixel-shuffle / Perceiver / Q-Former）、Whisper 与视频 | [[megatron-lm:megatron/core/models/vision/clip_vit_model.py#L115]] |
| [03 · 经典 VLM](./03_classic_vlms.md) | 按设计缺口串起 Flamingo、BLIP-2、LLaVA/1.5、InternVL、Qwen2-VL、DeepSeek-VL2、Janus，每节讲架构加训练配方 | 各论文 Fig 1 |
| [04 · 融合与 connector](./04_fusion_and_connectors.md) | 融合设计空间：三种范式对 attention 形状与 KV 的影响；embedding 怎么被 scatter 进序列 | [[megatron-lm:megatron/core/models/multimodal/llava_model.py#L482]]；sglang `mm_utils.py` |
| [05 · 生成器：diffusion / DiT / 自回归图像](./05_generation.md) | 生成器的数学与算力画像：DDPM / CFG / DDIM / latent diffusion / DiT / flow matching / 自回归图像 | DDPM Eq.14；DiT adaLN-Zero |
| [06 · 异构与 stage 解耦](./06_heterogeneity_and_disaggregation.md) | infra 问题之一：模型异构如何推导出 stage 解耦（DistTrain / EPD） | `add_encoder/add_decoder` |
| [07 · 变长输入与负载均衡](./07_variable_length_load_balancing.md) | infra 问题之二：变长的 $N$ 如何通过 packing、重排、token-balanced CP 摊平 | `greedy_knapsack` |
| [08 · 冗余、缓存与显存](./08_caching_redundancy_memory.md) | infra 问题之三：encoder 是纯函数，因此可以按 `mm_hash` 缓存；image token 会撑大 KV | `EncoderCacheManager` |

比较自然的读法是：先读本文建立三段流水线和两条轴的印象，再读 [`00`](./00_definitions.md) 把常用词的边界分清楚，接着读 [`01`](./01_contrastive_pretrain.md) 弄清视觉塔的训练目标，然后读 [`02`](./02_encoders.md) 把 $N$ 的公式过一遍，读 [`03`](./03_classic_vlms.md) 看各家沿着谱系怎么把视觉接进 LLM，读 [`04`](./04_fusion_and_connectors.md) 把三种融合方式收进同一张设计表，最后读 [`05`](./05_generation.md) 看生成任务在哪里分叉。`06`–`08` 三篇是把前面的架构事实推到 infra 上的自然结果，建议放在读完 `01`–`04` 之后再看。

如果只关心算法、暂时不想碰 infra，读完 `00`–`05` 就已经完整；`06`–`08` 里每个设计动机其实都能追溯回 `01` 的对比损失、`02` 的 token 数量公式，以及 `04` 的融合范式，需要的时候再回来查也不迟。

---

## 2. 一组贯穿全文的数字

后面出现的各种 shape 和「贵不贵」的判断，基本都是从这张表里的几个具体模型量级引出来的：

| | CLIP ViT-L/14 | LLaVA-1.5 | Qwen2-VL | InternVL 1.5（单 tile） |
|---|---|---|---|---|
| 分辨率 | $224^2$ / $336^2$ | $336^2$ | 原生，像素预算内任意 | $448^2$ tile，最多 40 |
| patch $P$ | 14 | 14 | 14 | 14 |
| 未压缩 $N$ | 256 / 576 | 576 | $HW/14^2$ | 1024 / tile |
| 压缩后 | —— | 576 | $\div 4$（2×2 merge） | 256（pixel-shuffle $r{=}2$） |
| connector | 对比空间的线性投影 | 两层 MLP | MLP merge | MLP |
| 融合 | 双塔（不做生成） | decoder-only 拼接 | decoder-only 拼接 | decoder-only 拼接 |

---

## 3. 代码映射表

想直接对照代码的话，可以从下面几个入口开始：

| 主题 | 文件 | 看什么 |
|---|---|---|
| 多模态模型装配 | [[megatron-lm:megatron/core/models/multimodal/llava_model.py#L57]] | `LLaVAModel`：vision_model + projector + language_model；`add_encoder/add_decoder` |
| image embedding 注入 | [[megatron-lm:megatron/core/models/multimodal/llava_model.py#L482]] | `_preprocess_data`：`image_token_index=-200` 占位替换成 image embedding |
| projector | [[megatron-lm:megatron/core/models/vision/multimodal_projector.py#L15]] | `mlp` 或 `affine` |
| vision encoder | [[megatron-lm:megatron/core/models/vision/clip_vit_model.py#L26]] | conv1 patchify + class token + position embedding |
| scatter（推理） | sglang [[sglang:python/sglang/srt/managers/mm_utils.py#L1022]]；vLLM [[vllm:vllm/model_executor/models/utils.py#L479]] | masked scatter / `_merge_multimodal_embeddings` |

---

## 4. 参考代码与论文

后面几篇会反复引用同一批参考代码，上游都固定在具体 commit（代码链接带 `#Lx-Ly`；Megatron pin 在 `e03878b5f`）：

- [[megatron-lm:megatron/core/models/multimodal/llava_model.py]] —— `LLaVAModel`：vision + projector + LLM
- [[megatron-lm:megatron/core/models/vision/clip_vit_model.py]] —— `CLIPViTModel` 的 patchify
- [[megatron-lm:megatron/core/models/vision/multimodal_projector.py]] —— `MultimodalProjector`
- [[sglang:python/sglang/srt/managers/mm_utils.py]]、[[vllm:vllm/model_executor/models/utils.py]] —— embedding 的 scatter 实现

也会按阅读顺序陆续出现下面这些代表性论文：

- Dosovitskiy et al., *ViT*, 2020. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Radford et al., *CLIP*, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Zhai et al., *SigLIP*, 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- Alayrac et al., *Flamingo*, 2022. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- Li et al., *BLIP-2*, 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- Liu et al., *LLaVA / LLaVA-1.5*, 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485), [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)
- Chen et al., *InternVL 1.5*, 2024. [arXiv:2404.16821](https://arxiv.org/abs/2404.16821)
- Wang et al., *Qwen2-VL*, 2024. [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
- Wu et al., *DeepSeek-VL2*, 2024. [arXiv:2412.10302](https://arxiv.org/abs/2412.10302)
- Chen et al., *Janus / Janus-Pro*, 2024–2025. [arXiv:2410.13848](https://arxiv.org/abs/2410.13848), [arXiv:2501.17811](https://arxiv.org/abs/2501.17811)
- Ho et al., *DDPM*, 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)；Peebles & Xie, *DiT*, 2022. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)；Esser et al., *SD3*, 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)

---

接下来的一篇是 [00 · 术语定义：encoder / projector / fusion / VLM](./00_definitions.md)，先把 encoder、projector、fusion、tile 这几个经常被混用的词厘清楚，后面的讨论才有共同的语言。
