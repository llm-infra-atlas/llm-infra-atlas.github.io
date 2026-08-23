# 00 · 术语定义：encoder / projector / fusion / VLM

多模态文档里有几个词被反复重载：encoder 有时指整座视觉塔，有时指某个线性层；projection 在三个地方出现，含义各不相同。本篇先把这些术语的定义统一起来，后面的章节才能无障碍地使用它们。读完之后，你应该能在一张架构图里准确指出哪一段是 encoder、哪一段是 projector；能说清楚 vision token 和文本 token 的区别；也能分清 tile 和 native dynamic resolution 并不是一回事。至于对比损失、ViT 的公式细节和各家的训练配方，会分别放在 `01`–`03` 里展开。

---

## 1. 从一次图文请求说起

设想用户丢来一张图和一句「这是什么」。一次 decoder-only VLM 的前向大致是这样的：

```
image  ──►  vision encoder  ──►  Z ∈ R^{N×D_v}
text   ──►  tokenizer + embed  ──►  T ∈ R^{L×D_llm}

Z  ──►  projector  ──►  H_v ∈ R^{N×D_llm}
H_v 和 T 按某种 fusion 合成一条序列
        │
        ▼
  LLM.forward()  ──►  assistant text
```

「多模态」这个词底下其实叠了三件不同的事情：

1. **编码**：把像素变成 $N$ 个连续向量，这是 encoder 的工作。
2. **对齐**：这 $N$ 个向量的宽度和语义要能被 LLM 读懂，这是 projector（或者叫 connector）的工作。
3. **融合**：它们在 LLM 的哪一层、以什么样的 attention 形状和文本交互，这是 fusion 要回答的问题。

后面会看到的所有架构分歧，追根溯源都是这三步里的某一步换了不同的做法。

---

## 2. 三个都叫「projection / encoder」的东西

一个容易让人混淆的地方是，projection 和 encoder 这两个词在多模态语境里其实被重载了三次，指代三种不同的东西。本篇先给出定义，具体的数学细节留到 [`02`](./02_encoders.md)。

| # | 名字 | 在哪 | 干什么 | 形状 |
|---|---|---|---|---|
| ① | **patch embedding** | ViT 第一层 | 每个像素 patch 线性映射成 $D$ 维 | $\mathbb{R}^{P^2 C}\to\mathbb{R}^{D}$ |
| ② | **modality encoder** | 整座视觉/音频塔 | patchify + ① + $L$ 层 Transformer | 图像 $\to\mathbf{Z}\in\mathbb{R}^{N\times D}$ |
| ③ | **projector / connector** | encoder 之后、LLM 之前 | 把 $D$ 对齐到 $D_{\mathrm{llm}}$ | $\mathbb{R}^{D}\to\mathbb{R}^{D_{\mathrm{llm}}}$ |

如果有人问「图里哪部分是 encoder」，答案是②——也就是整座视觉塔，包括 patchify 在内，而不是某一个线性层。ViT 论文顶部那个用于分类的 MLP Head，以及③里的 projector，都不属于 encoder 的范畴。

值得一提的是，Megatron 的 `MultimodalProjector` 内部把 MLP 子模块命名成了 `self.encoder`（[[megatron-lm:megatron/core/models/vision/multimodal_projector.py#L47]]）。这只是一个变量命名上的巧合，并不是说它在流水线意义上真的是 encoder。

还有几个经常一起出现的近义词，放在一起看会更清楚：

| 词 | 指什么 |
|---|---|
| **vision tower** | ②的别名，强调的是「那座被冻住或者解冻的预训练视觉模型」 |
| **connector / adapter / aligner** | ③的别名。LLaVA 用线性层或者两层 MLP；Flamingo 用 Perceiver Resampler；BLIP-2 用 Q-Former |
| **resampler** | 一类特殊的③：输出长度 $M$ **与输入 $N$ 无关**（Perceiver / Q-Former） |

---

## 3. Vision token 不是文本 token

文本 token 是词表上的一个离散 id，它的 embedding 来自一张查找表。Vision token 则完全不同：它是 encoder 直接吐出来的**连续向量**，没有对应的词表，因此不能像文本 token 那样直接做 next-token 分类。

这就带来一个问题：VLM 必须在流水线的某个地方，把这种连续的视觉表示接进本质上离散的语言模型。目前主要有三条路：

| 路 | 视觉在 LLM 里长什么样 | 代表 |
|---|---|---|
| **连续拼接** | $N$ 个 $\mathbb{R}^{D_{\mathrm{llm}}}$ 向量直接占用序列位置，和文本 embedding 一视同仁 | LLaVA、Qwen2-VL、InternVL |
| **连续旁挂** | 不进入主序列，只作为 cross-attention 的 K/V | Flamingo、Llama 3.2 Vision |
| **先离散化** | 用 VQ tokenizer 把图片编码成 codebook id，真正变成词表里的 token | Chameleon、Emu3；Janus 的生成支路 |

「image token 数 $N$」这个说法，在前两条路径里指的是连续向量的个数；只有在第三条路径里，它才是离散 id 的个数。从 infra 账本的角度看，两者都会占用序列长度，但只有走离散这条路的 token，才能像文本 token 一样在 decode 阶段复用 KV。

---

## 4. VLM / MLLM / omni / any-to-any

这几个术语经常被混着用，这里给出本仓库统一采用的用法：

| 词 | 本仓库的用法 |
|---|---|
| **VLM**（vision-language model） | 输入含图（可以再加文本），输出是文本。LLaVA、Qwen2-VL 都属于这一类 |
| **MLLM** | VLM 的近义词，更强调「大」和「能跟随指令」，本仓库不做进一步区分 |
| **omni** | 同一套模型能吃图像、音频、视频，输出仍以文本为主，比如 Qwen2-Audio、Qwen2.5-Omni |
| **any-to-any** | 输入和输出都可以是多种模态：既能看图问答，又能文生图，代表是 Chameleon、Emu3、Janus |

理解（understanding）和生成（generation）这两个任务对视觉表示的粒度要求并不一样：理解更需要语义信息，生成则需要像素级的细节。如果硬要让一个 encoder 同时服务这两个目标，往往会两头都做得不够好，这正是 Janus 选择拆开理解和生成两个 encoder 的动机，具体可以看 [`03`](./03_classic_vlms.md)。

---

## 5. Tile、动态分辨率、packing

传统做法是把图片强行 squish 到 $224^2$ 或 $336^2$ 的固定分辨率：高分辨率的图会丢细节，低分辨率的图又浪费了不必要的算力。解决这个问题主要有两类思路，值得注意的是它们并不是同一件事，不应该混为一谈：

| | **tiling / AnyRes** | **native dynamic resolution** |
|---|---|---|
| 做法 | 把图切成若干个固定大小的 tile，再额外加一张 thumbnail | ViT 直接按原图尺寸出 patch，序列长度随 $H,W$ 变化 |
| 单 tile 的 $N$ | 是个常数（比如 InternVL 一个 $448^2$ 的 tile 会产出 256 个 token） | 正比于 $HW/P^2$ |
| 总 token | tile 数乘以每个 tile 的 $N$ | 一张图对应一条变长序列 |
| 代表 | LLaVA-NeXT AnyRes、InternVL、DeepSeek-VL2 | NaViT、Qwen2-VL |

**packing** 则是另外一件事：它是把多张图（或者多个样本）的 patch 装进同一条序列里，再用 block-diagonal 的 attention mask 或者 `cu_seqlens` 挡住跨样本的 attention。NaViT 的 patch-n-pack，以及训练 dataloader 里常见的 knapsack 装箱，本质上都是在做 packing。它解决的问题是「变长的样本怎么才能填满一张 GPU」，而不是「一张图怎样在切分后仍然保持长宽比」，这两者不要弄混。

---

## 6. 训练阶段的常用叫法

下面这几个阶段名会在 [`03`](./03_classic_vlms.md) 里反复出现，这里先把它们的语义对齐一遍：

| 阶段 | 冻什么 | 训什么 | 数据 |
|---|---|---|---|
| **对比预训练** | 不冻结（从零训）或者只锁住图像塔（LiT 的做法） | 双塔 | 图文对，在 batch 内部做检索 |
| **alignment / feature alignment** | 通常冻结 encoder 和 LLM | 只训 connector | caption / 图文对 |
| **visual instruction tuning** | 通常仍然冻结 encoder | connector + LLM（或者再解冻 ViT） | 指令-回答数据，常常包含 GPT-4 合成的对话 |
| **联合 / 持续预训练** | 视具体配方而定 | 往往会解冻 ViT | 混合 OCR、VQA、视频、纯文本等多种数据 |

LLaVA 原论文把前两行分别对应 Stage 1 和 Stage 2。后续的工作在这个基础上把 Stage 1 拉长、把 ViT 解冻、再加上动态分辨率，但阶段的分工始终没有变：先让视觉向量能被 LLM 读懂，再教会它按照指令说话。

---

接下来是 [01 · 对比预训练](./01_contrastive_pretrain.md)。几乎所有开源 VLM 用的视觉塔都来自 CLIP 或 SigLIP，下一篇会先把对比损失的定义式，以及「为什么需要超大 batch」这个问题讲清楚。
