# 08 · 冗余、缓存与显存

解耦（[`06`](./06_heterogeneity_and_disaggregation.md)）和均衡（[`07`](./07_variable_length_load_balancing.md)）之后，image token 还会直接引出另外两个问题：冗余（同一张图被反复编码、同一段前缀被反复 prefill）和显存（$N$ 个视觉 token 撑大了 KV cache，也撑大了训练时的激活）。本篇要讲的就是对应的解法：跨请求的 embedding 缓存与去重、prefix cache 遇到 image token 时该怎么处理、以及训练侧 encoder 激活显存和 recompute 的做法。

这两个问题追根溯源其实是同一件事：[`02`](./02_encoders.md) 里讲过，视觉 token 数 $N$ 本身很大，而且 encoder 是一个确定性的纯函数——同一张图输入进去，永远得到同一组 embedding。「确定性纯函数加上输出很大」恰好是缓存的理想场景，同时也是显存压力的来源。

---

## 0. 两条线索：冗余与显存

encoder 是纯函数，所以可缓存；token 很多，所以耗显存：

```
encoder(image) = embedding    ← 确定性、无状态 ⟹ 同图必同 emb ⟹ 跨请求/跨样本可缓存(去冗余)
embedding ∈ ℝ^{N×D}, N 大     ⟹ 进 KV(推理) / 进激活(训练) ⟹ 显存压力
```

本篇有两条线索贯穿始终：冗余这条线索关心的是同一张图在多轮对话、多个请求、热门图片里被反复编码的问题，解法是缓存 encoder 的输出（§1）；以及同一段前缀被反复 prefill 的问题，解法是 prefix cache 或者 radix cache（§2）。显存这条线索关心的是 $N$ 个视觉 token 占用 KV（推理场景，§3）以及占用激活（训练场景，§4）的问题。

---

## 1. 跨请求复用 encoder 输出

encoder 是一个 compute-bound 的大头（[`06 §3.1`](./06_heterogeneity_and_disaggregation.md) 里提到过，最高能占到 TTFT 的 79%）。如果同一张图反复出现——比如多轮对话里的同一张图、系统图标、热门 meme——重新计算就是纯粹的浪费。解法很直接：用图像内容的 hash 作为 key，缓存 encoder 的输出，命中之后就是零编码成本。

### 1.1 SGLang：`mm_hash` 与 `MultiModalStaticCache`

SGLang 的做法是这样的：`data_hash` / `tensor_hash`（[[sglang:python/sglang/srt/managers/mm_utils.py#L1207]]）对像素张量（或者它的特征）用 SHA-256 取前 8 字节来算出 `mm_hash`；GPU 张量走的是 triton 实现的 `gpu_tensor_hash`，CPU 则走增量式的 hashlib。一条请求里的多张图各自有自己的 hash，再通过 `combine_hashes`（[[sglang:python/sglang/srt/disaggregation/multimodal_cache.py#L18]]，实现是 `hash(tuple(mm_hashes))`）合成一个组合 key。

缓存本身是 `MultiModalStaticCache`（[[sglang:python/sglang/srt/disaggregation/multimodal_cache.py#L76]]），是一个 LRU 风格的 `OrderedDict`：`get` 命中之后会调用 `move_to_end`，`set` 的时候如果超过 `max_size`（由 `SGLANG_VLM_CACHE_SIZE_MB` 控制，默认 4096MB）就会 `popitem(last=False)` 淘汰最旧的条目。

这个缓存由 encode server 持有（[[sglang:python/sglang/srt/disaggregation/encode_server.py#L293]]），所以能够跨请求生效——这可以看作 EPD 解耦把 encoder 独立出来之后顺手得到的一个好处：既然 encoder 已经是一个独立的服务，它的输出缓存自然就是 server 级别的，能被所有请求共享。

### 1.2 vLLM：`MultiModalHasher` 与 `EncoderCacheManager`

vLLM（v1 引擎）把同样的思想做成了一等公民，分成两个部分。

`MultiModalHasher`（[[vllm:vllm/multimodal/hasher.py#L50]]）负责算 hash，默认用 blake3（也可以选 sha256 或者 sha512，用于满足 FIPS 合规要求），对多模态 item 序列化之后求出 `mm_hash`。这个 hash 同时也是 §2 里 prefix cache key 的一部分。

`EncoderCacheManager`（[[vllm:vllm/v1/core/encoder_cache_manager.py#L17]]）负责按 `mm_hash` 跨请求复用 encoder embedding。它维护一张 `cached: mm_hash → {引用它的 request id}` 的表，`check_and_update_cache`（[[vllm:vllm/v1/core/encoder_cache_manager.py#L94]]）在命中时直接复用，并把这个条目从 `freeable` 集合里移出；一旦引用计数归零，条目就进入 `freeable`，只有在分配显存需要腾地方时才会被真正淘汰——这是一种按引用计数决定去留的策略，而不是定时的 LRU。

在协议层，vLLM 原生的 EPD 方案（[`06 §4.2`](./06_heterogeneity_and_disaggregation.md)）做得更彻底：请求只带 image hash、不带像素，EC Connector 根据 hash 去查 Embedding Cache（对应 `ECConnectorBase.has_cache_item`），一旦命中就完全跳过 encoder 和上传步骤。这相当于把「去重」从服务内部的优化提升成了 API 层面的契约。

SGLang 的 `MultiModalStaticCache`（纯 LRU）和 vLLM 的 `EncoderCacheManager`（引用计数加按需淘汰）其实是同一个目标下的两种淘汰策略：前者是「最近最少用就扔」，后者是「只要还有请求引用就绝不扔」。后者更贴合 serving 的场景——毕竟一张图在它所属请求的整个生命周期里都不应该被误淘汰。

这里有必要区分两种容易混淆的「缓存」（EPDServe 论文里的 MMBlockManager 属于前者）：一种是单请求的传输缓冲区，把 encode 出来的 embedding 暂存起来，等着传给 prefill，命中率对它没有意义，因为它本来就不是为了复用而设计的；另一种是跨请求复用缓存（也就是 SGLang 和 vLLM 上面这套方案），按内容 hash 跨请求复用，命中就意味着零编码成本。只有后者才是真正在消除冗余，生产系统普遍都做了后者，而早期的学术系统大多只有前者。

---

## 2. prefix cache 遇到 image token

文本 serving 里的 prefix cache（或者 radix cache，见 [`serving`](../serving/README.md)）是按 token 前缀来复用 KV 的。多模态请求的前缀里夹着 image token，这会带来两个变化。

第一个变化是前缀匹配的粒度问题：image token 在 `input_ids` 里其实是一个占位 token（参考 [`04 §4`](./04_fusion_and_connectors.md) 里的 `image_token_index`），但两张不同的图完全有可能占用同样的占位 token id。所以前缀匹配不能只看 token id，必须把 `mm_hash` 也纳入前缀 key 的一部分——否则会把「prompt 相同但图片不同」的两条请求错误地当成可以复用 KV 的同一条请求。这是多模态 radix cache 和纯文本实现之间最大的差异。

第二个变化是复用带来的价值更大了：在多轮视觉对话里，同一张图配上同样的系统 prompt，会构成一段相当长的公共前缀（毕竟一张图本身就要占几百到上千个 token），命中一次能省下的 prefill 开销远远超过纯文本场景。

概括起来，image token 让 prefix cache 的 key 从单纯的「token 序列」升级成了「token 序列加上内容 hash」。这和 §1 讲的 embedding 缓存其实是两层不同的缓存：一层缓存 encoder 的输出，省的是编码开销；另一层缓存 LLM 的 KV，省的是 prefill 开销。多模态场景下这两层缓存都需要，而且都必须把 `mm_hash` 当成一等公民对待。

---

## 3. 推理显存：image token 与 KV cache

decoder-only 的融合方式（[`04 §2.1`](./04_fusion_and_connectors.md)）把 $N$ 个视觉 token 放进了主序列，这也就意味着它们会全部进入 KV cache。一张动态分辨率的图可能有 3000 个以上的 token，一段视频可能有上万个 token，对 KV 预算的冲击是数量级上的：

```
KV 占用 ∝ (T_text + N_visual) × layers × 2 × d_kv × dtype
                    └─ N 可达数千~上万, 常常主导整个 KV ─┘
```

这带来几个直接的后果。首先是 batch 大小会被 KV 限制：能并发处理的请求数会被含图请求的 $N$ 拉低。EPDServe 把 encoder 权重从 prefill/decode 池里剥离出去（[`06 §3.2`](./06_heterogeneity_and_disaggregation.md)），腾出来的显存正好可以还给 KV，实测能让 KV 容量放大 2.2 倍。其次是 cross-attention 融合方式下的显存账不一样：视觉 token 不进入主序列，只出现在 cross-attn 的 K/V 一侧（[`04 §2.2`](./04_fusion_and_connectors.md)），主序列的 KV 会比较短，但需要单独维护一份 image-side 的 KV。这也是 ModServe 对两种融合方式采用不同路由指标（按总 token 还是按纯文本 token）的原因。最后，生成器本身没有 KV：diffusion generator（[`05 §0`](./05_generation.md)）是双向的，也不需要 KV，它的显存压力体现在激活上（每一步都要对全量 latent token 算出中间结果），而不是 KV 上——这再一次印证了四段流水线各自的显存模型也是不一样的。

---

## 4. 训练显存：encoder 激活与 recompute

训练侧的 image token 不会进入 KV（训练本身没有 KV cache 这个概念），但会进入激活，而且这个问题会被 [`07 §0`](./07_variable_length_load_balancing.md) 里提到的「激活峰值方差约等于均值」这个现象进一步放大。具体有两个战场。

### 4.1 encoder 激活显存：BigMac 的 O(1)

[`06 §1.4`](./06_heterogeneity_and_disaggregation.md) 讲过 BigMac 面对的核心矛盾：如果想让解耦后的 encoder 保持算力高效，就必须保留所有 microbatch 的 encoder 激活，显存占用是 $O(M/P)$，一旦 batch 变大就会 OOM。BigMac 用依赖安全的嵌套流水线把需要保留的激活压缩到了「3 个 encoder 单元加 1 个 generator」，也就是 $O(1)$，并且在不牺牲时序表现的前提下让显存能够随着 batch 大小稳定增长。这是「变长激活」和「解耦」这两个约束共同逼出来的设计。

### 4.2 自适应 recompute

变长样本的激活峰值本来就参差不齐（长视频样本的激活可能非常大），如果按照最坏情况对全程都开启 activation recompute，在短样本上其实就是纯粹的浪费。OmniBal 的 Balanced Adaptive Re-Computation（[`07 §3`](./07_variable_length_load_balancing.md)）根据每个 partition 实际的显存余量自适应地决定要开多少 recompute，把闲置的显存换成节省下来的计算时间，比起一刀切的全量 recompute 要更省时间。

训练显存的这两条线——encoder 激活的 $O(1)$ 方案和自适应 recompute——本质上都是在对抗同一个问题：变长带来的激活峰值不可预测。这和推理侧「image token 撑大 KV」这个问题其实是同一个根源（$N$ 大且变长）在训练和推理两侧的不同表现形式。

---

## 5. 小结

```
            视觉 token 数 N 大  +  encoder 是确定性纯函数
                    │                         │
        ┌───────────┴──────────┐    ┌────────┴─────────┐
     显存压力                  冗余(可缓存)
        │                                   │
  推理: image token 进 KV          encoder 输出按 mm_hash 跨请求复用(§1)
   → EPD 剥离权力还显存(§3)         前缀 KV 复用要带 mm_hash(§2)
  训练: 进激活, 峰值方差大
   → BigMac O(1) + 自适应 recompute(§4)
```

三个问题——[`06`](./06_heterogeneity_and_disaggregation.md) 讲的异构、[`07`](./07_variable_length_load_balancing.md) 讲的变长、本篇讲的冗余与显存——合起来就是多模态 infra 的全部骨架，而且每一个都能追溯回知识底座里的两张表：[README §0](./README.md) 的「四种算力画像」，以及 [`02 §6`](./02_encoders.md) 的「token 数量标度律」。

---

## 6. 全章回顾

```
算法底座                              infra 解法(按问题类型, 训推合并)
─────────                            ──────────────────────────────
01 对比预训练 / 02 encoder: N 怎么来   06 异构 → 解耦(DistTrain/EPD/...)
03 经典 VLM / 04 fusion: N 怎么进 LLM  07 变长 → 负载均衡(packing/重排/CP/调度)
05 gen: 迭代/双向/无 KV                08 冗余/显存 → 缓存/去重/recompute
```

把全章压缩成一句话：多模态把「token 数」从原本由 tokenizer 决定的可控量，变成了由分辨率或时长决定的、逐样本剧烈波动的随机量；同时把原本单一同构的 transformer，变成了四种算力画像迥异的算子流水线。infra 的所有应对措施，归结起来就是三件事：解耦（让每一段独立配置资源）、均衡（按 token 而不是按样本分配，并且尊重图片或帧作为不可分割的原子）、缓存（利用确定性 encoder 输出去消除冗余，做 image-aware 的 KV 复用）。而且训练和推理共用的是同一套思路。

---

本章到这里就结束了。可以回到 [`README`](./README.md) 重新看一遍全景图和代码映射表，也可以横向对接 [大规模训练的并行策略 —— 总览](../parallel/README.md)（了解解耦之后各段并行的具体细节）、[推理服务：从单请求推理到 SLO-aware 集群](../serving/README.md)（了解 EPD 之上的 P-D 分离与 KV 管理）、[00 · Roofline model：性能上界的两道天花板](../hpc/00_roofline_model.md)（从 roofline 的角度理解四种算力画像）。
