# 07 · 变长输入与负载均衡

[`06`](./06_heterogeneity_and_disaggregation.md) 里的解耦方案解决了模型异构的问题。但 [`02`](./02_encoders.md) 留下的那个逐样本剧烈波动的 token 数 $N$（图片 66 到 3136、音频数百到上万、视频上万）依然存在，这就是数据异构。本篇要讲的是怎么把这种波动的负载摊平：训练侧有 sequence packing、数据重排、OmniBal 的三方面均衡、token 均衡 CP，推理侧则有 chunked prefill 和按 image-token 负载做调度。

这里有一个核心判断需要先说清楚：在纯文本的世界里，「按样本数均分到各 rank」基本上等价于「按算力均分」，因为句子长度的方差是有限的；但到了多模态的世界里，这个近似彻底失效了——一条视频样本的算力可能是一条文本样本的 1000 倍。所以多模态场景下的负载均衡必须按 token（也就是算力）而不是按样本来切分。

---

## 0. 变长 N 带来的负载不均

把波动的 $N$ 代入并行训练的场景，会在几个维度上制造出负载不均（OmniBal 和 DistTrain 都做过实测）：

```
DP 维(intra-batch):   rank0 抽到几条文本(轻)   rank1 抽到几条长视频(重)
                      → rank0 早早算完, 卡在 all-reduce 干等 rank1   ← 快 rank 空转

PP 维(inter-microbatch): microbatch_a 全是短样本(encoder 快)
                         microbatch_b 含长视频(encoder 慢)
                      → 1F1B 流水线节奏被最慢的 microbatch 打乱   ← 气泡

显存维:               长样本的激活峰值远超短样本
                      → 要么按最坏情况保守开 recompute(浪费), 要么 OOM
```

OmniBal（[arXiv:2407.20761](https://arxiv.org/abs/2407.20761)）的 profiling 给出了非常直观的数字：单次 forward 的耗时是 85±93 ms，显存占用是 39±23 GB——方差几乎和均值一样大。DP-0 可能要空等 DP-1 大约 760 ms。这种「方差约等于均值」的负载情形，在纯文本训练里是很少见到的。

三种不均对应三种不同的解法：DP 维的不均靠 packing 加数据重排来解决（§1、§2）；PP 维和显存维的不均靠均衡的模型切分加自适应 recompute 来解决（§3）；单条样本本身超长的情况则靠 CP 来解决（§4）。推理侧其实是同一组问题在另一个形态下的重现（§5）。

---

## 1. sequence packing：把变长样本装箱进定长序列

最基础的一招来自 NaViT 的 patch-n-pack（详见 [`02 §2.3`](./02_encoders.md)）：把多条短样本装进一条固定长度的序列里，从而消灭 padding 带来的浪费。

### 1.1 装箱问题

Megatron 的多模态 dataloader 直接用贪心装箱来解决这个问题（[[megatron-lm:examples/multimodal/dataset_helpers.py#L103]] 的 `greedy_knapsack`，配合 `search_for_fit:95` 做二分查找）：

```python
# dataset_helpers.py:103 greedy_knapsack  贪心 + 二分: 尽量多塞样本进 max_capacity 的"背包"
sorted_item_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples)))  # 按长度排
while sorted_item_sizes:
    current_knapsack = []; remaining = max_capacity
    while (idx := search_for_fit(sorted_item_sizes, remaining)) != -1:  # 二分找能塞的最大样本
        remaining -= sorted_item_sizes[idx]
        current_knapsack.append(sorted_samples.pop(idx)); sorted_item_sizes.pop(idx)
    knapsacks.append(current_knapsack)
```

每条样本的「长度」定义为文本 token 数加上 image token 数（也就是 [`02 §6`](./02_encoders.md) 里的 $N$），打包之后的结果是 `ImageTaskSamplePacked`（[[megatron-lm:examples/multimodal/dataset_helpers.py#L50]]），带有 `cu_lengths` 字段记录每个子样本的累积长度。相关的开关是 `packing_seq_length` 和 `packing_buffer_size`（[[megatron-lm:examples/multimodal/dataset_helpers.py#L159]]）。

### 1.2 `cu_seqlens` / THD

被打包进同一条序列的多个样本不能互相做 attention。实现上靠的不是普通的 mask，而是给 FlashAttention 喂一种变长格式（THD：Total tokens, Heads, Dim），再配合 `cu_seqlens`（累积序列长度边界）。Megatron 的 `PackedSeqParams`（[[megatron-lm:megatron/core/packed_seq_params.py#L10]]）就是承载这组参数的地方：

```python
# packed_seq_params.py:10
@dataclass
class PackedSeqParams:
    qkv_format: str = None        # 'thd'
    cu_seqlens_q: Tensor = None   # [0, 5, 7, 11, ...] 每个子样本的边界
    cu_seqlens_q_padded: Tensor = None
    ...
```

在多模态 CP 场景下，负责构造这组参数的是 [[megatron-lm:megatron/core/models/multimodal/context_parallel.py#L69]] 里的 `get_packed_seq_params`：

```python
# context_parallel.py:69 (精简自源码)
cu_seqlens = torch.arange(0, (batch+1)*img_seq_len, step=img_seq_len, ...)
if cp_size > 1 and (padding_needed or use_packed_sequence):
    cu_seqlens_padded = ...   # CP 下还要给 padded 版本
packed_seq_params = PackedSeqParams(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                                    qkv_format='thd', ...)
```

这其实是 [Attention —— 总览](../attention/README.md) 里那套变长 attention 接口在多模态场景下的直接复用：`cu_seqlens` 告诉 kernel 序列在哪些位置被截断，每一段只在自己内部做 attention。packing 加上 `cu_seqlens` 组合起来，就能在摊平变长负载的同时不破坏语义，这是变长训练能够成立的基础。

### 1.3 packing 的代价

装箱之后，每一条「打包序列」的长度仍然可能不完全相等，因为贪心装箱并不保证能把背包完美填满，残余的 padding 依然存在。而且被打包进同一条序列的样本数并不固定，导致每个 packed sample 里的图片数量、attention 分段数都不一样，这相当于把数据异构从「样本级」推到了「packed-sample 级」，问题并没有被消除，只是变得更细粒度了。所以在 packing 之上，还需要叠加数据重排（见 §2）。

---

## 2. 数据重排：调整梯度累积顺序，削平 straggler

DistTrain（[`06 §1.1`](./06_heterogeneity_and_disaggregation.md)）的「disaggregated data reordering」抓住了一个安全的前提：在一个 iteration 内部，梯度累积的顺序是可以交换的，重新排列样本顺序并不会改变最终的收敛结果。既然如此，就可以放心地重排样本顺序来削平 straggler。具体分两层：

intra-microbatch 层面（DP 组间均衡）把样本均匀分配到各个 DP 组，使得每一组的总 token 数（近似等于算力）尽量接近。这本质上是经典的 multiway number partitioning（多路数划分，属于 NP-hard 问题）：DistTrain 的做法是把样本按大小降序排列，每个样本贪心地塞进当前负载最低的 DP 组（也就是 LPT 贪心），近似比小于 4/3，复杂度是 `O(n log n + m·n)`。

inter-microbatch 层面（1F1B 气泡最小化）决定的是 microbatch 送入的顺序。把最小的 microbatch 放在最前面，在尾部保留 `p−1` 个最小的，中间用 best-fit 填充，这样可以避免 1F1B 的 warmup/cooldown 阶段被过大的 microbatch 打乱节奏。复杂度是 `O(l·(l+p))`。

这里有一个关键性质需要强调：重排只改变顺序，并不改变样本集合，所以 loss 逐 bit 不变。这意味着「负载均衡」变成了一个纯粹的调度问题，可以完全放在 CPU producer 节点上异步完成（DistTrain 就把预处理和重排都交给了 CPU 节点，让它和 GPU 训练过程 overlap 起来）。这大概是「按 token 而不是按样本均分」这个原则最干净的一种落地方式。

---

## 3. OmniBal：把均衡同时做在数据、模型、重算三处

OmniBal（ICML 2025）认为只均衡数据是不够的，PP 的切分方式和 recompute 的开关也需要一起均衡，具体做了三件事：

| 平衡 | 解什么 | 机制 | 效果 |
|---|---|---|---|
| **Balanced Dynamic Mini-Batch** | DP 组间 token 不均 | 迭代采样加过滤，在 image/text token 上限下动态组 batch | padding 比例从 0.31 降到 0（不到 1 分钟完成） |
| **Balanced Model Partitioning** | PP stage 算力不均 | 搜索式 pipeline 切分，最小化各 stage forward 时间的方差 | —— |
| **Balanced Adaptive Re-Computation** | 显存按最坏情况保守分配 | 按 partition 自适应开启 recompute，回收闲置显存 | —— |

最终结果是：InternVL-Chat 在部分数据集和模型上取得了约 1.8 倍（最高 3.5 倍）的提升；从 64 台扩展到 512 台 H100 时，并行效率从 85% 提升到了 95%。

OmniBal 的 Balanced Model Partitioning 和 Cornstarch 的 frozen-aware PP（[`06 §1.3`](./06_heterogeneity_and_disaggregation.md)）其实是同一个问题的两个侧面：PP stage 的边界不能简单按层数均分，而要按真实算力（包含变长因素和冻结因素）均分。而 Balanced Adaptive Re-Computation 则呼应了 BigMac 的 $O(1)$ 激活方案，两者都在应对「变长带来的激活峰值」这个显存难题。

---

## 4. context parallel：当单条样本本身就超长

packing 和数据重排处理的是「很多条样本该怎么分配」这个问题；但当一条长视频或者长音频样本本身（参考 [`02 §4-5`](./02_encoders.md)，单条样本可以达到上万甚至十几万 token）已经放不下单张卡时，就需要靠 CP（context parallel）把这一条序列切到多张卡上。多模态场景下的 CP 比纯文本的 CP 要难在两个地方：

第一是切分点必须对齐样本或者帧的边界，不能把一张图的 patch 拦腰切到两个 CP rank 上，否则位置编码（M-RoPE）和 tubelet 结构都会乱掉。Megatron 的 `context_parallel.py` 专门提供了 `split_to_context_parallel_ranks_dynamic_res`（[[megatron-lm:megatron/core/models/multimodal/context_parallel.py#L313]]）和 `_compute_tubelet_aware_split_points`（[[megatron-lm:megatron/core/models/multimodal/context_parallel.py#L233]]）来做 tubelet 感知的切分，并用 all-to-all 在 CP 组间重新分布（[[megatron-lm:megatron/core/models/multimodal/context_parallel.py#L225]]）。

第二是每个 CP rank 的 token 负载要保持均衡。在变长且非因果的 attention 场景下，简单地把序列等分给各个 rank 会导致算力明显不均。Cornstarch 的 token workload-balanced CP（[`06 §1.3`](./06_heterogeneity_and_disaggregation.md)）以 128-token 的块为粒度跑 ILP（用贪心的 LPT 方法求解），必要时还会再切子块，以此把负载摊平。BigMac 的 Decoupled CP（各个模块使用不同的 CP 组，通过 all-to-all 做转换）解决的是「encoder 和 LLM 理想的 CP 度不一样」这个问题。

概括起来，CP 在多模态场景下从「均匀切分长序列」升级成了「边界感知加负载感知的切分长序列」。底层用的仍然是 [Context Parallelism (CP) —— Infra 视角深入](../parallel/04_cp/README.md) 里那套 ring 或者 all-to-all attention，只是切分逻辑因为变长和位置编码而变得更复杂了。

---

## 5. 推理侧的变长问题

推理阶段没有「梯度累积顺序」可以调整，但变长的 $N$ 制造出的负载不均，换了一副面孔出现在调度、batching、路由这几个环节里。

### 5.1 chunked prefill 遇到 image token

纯文本场景下的 chunked prefill 可以把 prompt 按任意 token 边界切成 chunk，从而限制单步的算力消耗。但 image embedding 是一个整体——一张图对应的 $N$ 个 token 必须作为一个整体被 scatter 进序列（参考 [`04 §4.2`](./04_fusion_and_connectors.md)）。SGLang 的 `_get_chunked_prefill_embedding`（[[sglang:python/sglang/srt/managers/mm_utils.py#L623]]）专门处理这个细节：图片对应的 item 在 processor 阶段已经按图切好了，chunk 的时候按照图的 offset 去取对应的那段 embedding，避免把一张图切碎。

vLLM 的做法则是把 encoder 的执行本身变成一份可调度的算力预算。`MultiModalBudget`（[[vllm:vllm/multimodal/encoder_budget.py#L44]]）计算出 `encoder_compute_budget` 和 `encoder_cache_size`，scheduler 的 `_try_schedule_encoder_inputs`（[[vllm:vllm/v1/core/sched/scheduler.py#L1280]]）在 `max_num_encoder_input_tokens`（[[vllm:vllm/v1/core/sched/scheduler.py#L218]]）这个上限内决定这一步要编码哪些图，编好的 embedding 交给 `EncoderCacheManager`（[`08 §1.2`](./08_caching_redundancy_memory.md)）管理。这相当于把 encode 这件事当成和 prefill chunk 并列的另一类受预算约束的工作，避免某个 encode 很重的请求独占整个 step 的算力。

这是一个典型的「多模态打破了文本侧优雅假设」的例子：文本 token 可以任意切分，image token 却不行。调度器必须把「一张图是一个不可分割的原子」这个约束纳入进来，并单独为 encode 这件事划出一份算力预算。

### 5.2 按 image-token 负载调度与路由

ModServe（[`06 §3.3`](./06_heterogeneity_and_disaggregation.md)）把训练侧「按 token 而不是按样本均衡」的思路原样搬到了推理场景：路由按 image-token 负载来决定，新请求会被发到当前 image-token 负载最低的实例，而不是简单地按请求数轮询，因为一条多图请求可能顶得上几十条文本请求。decoder-only 模型按总 pending token 路由、cross-attn 模型按纯文本 token 路由，这直接对应了 [`04 §3`](./04_fusion_and_connectors.md) 里两种融合方式在序列长度上的差异。token 感知的自动扩缩策略也是一样，副本数按 token 吞吐来算，而不是按 QPS 来算。

EPDServe 的 IRP（[`06 §3.2`](./06_heterogeneity_and_disaggregation.md)）则是推理侧针对「单样本切分」的解法：把一条请求的图像 patch 切分到多个 encode worker 上并行处理，和训练侧用 CP 切长序列的思路异曲同工。

---

## 6. 小结

把「变长负载均衡」用到的各种工具按照「均衡的对象是什么」归纳成一张表：

| 均衡对象 | 训练侧 | 推理侧 |
|---|---|---|
| 多样本 → DP/实例 | 数据重排（LPT 多路划分） | 按 image-token 路由（ModServe） |
| 多样本 → 一条序列 | sequence packing（knapsack + `cu_seqlens`） | chunked prefill（图为原子） |
| 单超长样本 → 多卡 | 边界+负载感知 CP（Cornstarch/BigMac） | IRP 切 patch 到多 encoder（EPDServe） |
| PP stage / 显存 | 均衡切分 + 自适应 recompute（OmniBal） | 每段独立 TP + 自动扩缩 |

贯穿全篇的原则可以概括成一句话：多模态的负载均衡，做的始终是「按 token（也就是算力）而不是按样本均分」，同时「尊重 image 或者帧作为不可分割的原子」。前者是因为 $N$ 的方差几乎和均值一样大，后者是因为视觉 token 在编码和位置编码上本身就是一个整体。

---

接下来是 [08 · 冗余、缓存与显存](./08_caching_redundancy_memory.md)。解耦加均衡之后，还剩下两个由 image token 直接引出的问题：同一张图被反复编码造成的冗余，以及 image token 撑大 KV 和激活造成的显存压力。下一篇会讲跨请求的 embedding 缓存与去重、prefix cache 遇到 image token 时的处理方式，以及训练侧 encoder 的激活显存问题。
