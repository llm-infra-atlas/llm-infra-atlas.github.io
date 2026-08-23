# 系统侧负载均衡：EPLB、LPLB、UltraEP 与 MoonEP

> 读这篇之前，最好已经读过 [01 · Router 与 Dispatch 前的 Preprocess](./01_router_and_preprocess.md)，了解 `routing_map` 和 top-k 的含义，也读过 [02 · Dispatch：permute、all-to-all、buffer 分配](./02_dispatch.md)，熟悉 EP 的 all-to-all 通信是怎么发生的。本文要讨论的是另一层面的问题：router 已经选完 expert 之后，怎么让物理 GPU 上的 token 数和 expert 权重的布局始终保持可计算、可扩展。

这里有一个值得先厘清的边界：前面几篇讲的 aux loss 和 expert bias，解决的是算法侧「该把 token 选给哪个 expert」的问题；而这一章要解决的是系统侧的问题，也就是 expert 该放在哪张卡上、遇到热点 expert 时能不能复制、token 是否需要重新路由。这两类问题不应该被混进同一个 loss 里去解决。

## 1. 统一问题定义

设一个 EP group 有 $R$ 个 rank、$E$ 个 logical expert。router 给出每个 token 的 top-$k$ expert，记本 batch 中 expert $e$ 的 token 数为 $T_e$。如果不做系统侧处理，rank $r$ 的工作量是

$$
L_r = \sum_{e\in\mathcal{E}_r} T_e,
$$

其中 $\mathcal{E}_r$ 是 rank $r$ 上的 local experts；step time 近似由 $\max_r L_r$ 决定。系统侧负载均衡允许两类动作：

1. **复制（replication）**：把热点 expert 的权重放到额外的 physical slot；这是低频、会涉及权重同步的动作。
2. **重路由（reroute）**：同一 logical expert 的 token 在可用 replica 之间重新分配；这是高频、不能改变模型语义的动作。

目标不是让 router 的统计完全均匀，而是最小化由通信与 grouped GEMM 共同决定的 critical path：

$$
\min \; \max_r\; \mathrm{cost}_r(\text{local tokens},\text{remote tokens},\text{GEMM shape}).
$$

## 2. EPLB：周期性专家重放置

EPLB（[[eplb:]]）的输入是一段时间窗口内统计到的 expert workload，输出是 physical expert 到 rank/node 的 placement。整个过程大致可以分成三步：先做 **balanced packing**，按 workload 从大到小把 expert 放进多个 pack，用类似 LPT（最长处理时间优先）的贪心策略近似最小化最大 pack 的负载；接着做 **replicate**，优先复制那些造成最大 bottleneck 的热点 expert，直到用完显存或 replica 数量的预算；最后做 **placement**，在 node 或 NVLink group 这一级错开摆放，避免同一个热点 expert 的多个 replica 挤在同一个通信域里。

EPLB 输出的是下一段时间要使用的 physical layout，而不是针对当前这个 batch 算出来的 token map。所以它适合解决那种持续存在的长期热度问题，不太适合追逐单个 microbatch 里的随机噪声；搬运权重本身是有成本的，这也意味着它没办法像 token reroute 那样做到每一层都重新排布。

## 3. LPLB：固定专家布局下的 per-batch 重路由

LPLB（[[lplb:]]）假定 EPLB 已经准备好 topology，并且每个 redundant expert 只连接两个 logical owners。令可移动 token 类别为 $j$，它只能在两个相邻 rank 之间分配，变量 $x_j \in [0, 1]$ 表示送往其中一个 replica 的比例。以 $z$ 表示最大 rank load，核心约束是

$$
\begin{aligned}
\min_{x,z}\quad & z \\
\text{s.t.}\quad & L_r(x) \le z,\quad r=0,\ldots,R-1,\\
& 0\le x_j\le 1.
\end{aligned}
$$

kernel 在单个 SM 上执行 affine-scaling interior-point 迭代，得到分数解之后再按 token 粒度落实为具体分配。它不搬运 expert 权重，只改变 dispatch 的目标，因此可以按 batch 运行；代价是 topology 和 replica 数量需要预先固定。

## 4. UltraEP：实时专家均衡的完整 runtime

UltraEP 的 [[ultraep:README.md#L7-L15|README]] 明确了它与 EPLB/LPLB 的差别：输入不是滞后的历史统计，而是每层、每个 microbatch 的 post-gating load；规划、通信和 replica buffer 都在 GPU-native runtime 内完成，控制面和数据面都提供高性能的算子实现，关键路径开销极低（约 300 微秒）。

UltraEP 将 MoE 专家负载均衡变成了一项实时执行的系统能力，并且在 MoE 训练的实际生产中落地。在 106B 到 671B 参数的主流 MoE 上，真实动态负载下的训推吞吐平均达到 force-balanced 理想性能的 94.3%。相较于业界 SOTA 训推框架（Megatron-LM/SGLang）平均提升 1.49 倍，基本消除了专家负载不均。

一层的关键顺序是：

```text
router(logical routing)
  → update_placement(exact load)
  → weight_sync(master → replica)
  → reroute(logical → physical)
  → dispatch → grouped GEMM → combine
  → restore replica weights
  → async grad_reduce(replica → master)
```

UltraEP 的 replica buffer 跨 layer 复用，因此显存开销由每 rank 的 redundant slot 数决定，而不是 `num_layers × replicas`。

## 5. MoonEP：静态形状、无 host sync

MoonEP 的目标不是尽量把负载摊平，而是构造性地做到完全均衡：每个 rank 每层接收且仅接收固定的 $S \times K$ 个 token（[[moonep:README.md#L3-L9|README]]），前提是每个 EP rank 预留的冗余专家 slot 数等于其固有的 local expert 数。通信 buffer 和 grouped GEMM 因此都是静态形状；运行时不再需要把动态的 `recv_count` 拷回 host 再 `torch.empty`，host sync 也随之消失。

实现上，GPU planner 负责选择 dynamic redundant experts，dispatch 之前预取权重，再用 symmetric memory 把 token 直接写到按 expert 分组的最终位置。planner 保证每个 rank 正好接收满 $S \times K$ 个 token；weight prefetch 加上静态的 $[E+B, H, H']$ weight tensor 使 GEMM 的形状可以预先确定；zero-copy view 则避免了重新分配。

关键数据契约（[[moonep:README.md#L43-L59|README]]）：

- MoonEP 与训练/推理框架之间的契约是：**每个 expert projection 对应一个连续的 symmetric-memory 权重张量 $[E+B, H, H']$，外加 planner 生成的 `cu_seqlens`**。group GEMM 只按行索引寻址 expert；`cu_seqlens[E+B]`（由 `dispatch` 返回）选择当前 step 中哪些 expert 行参与计算。
- 行 $[0, E)$ 是所有 rank 的 local expert，每 rank $E/R$ 行；每一段在物理上就是 home rank 的参数内存，通过 symmetric memory 映射到所有 rank。
- 行 $[E, E+B)$ 是本地 prefetch slot，由 `buffer.prefetch_weight` 填充；planner 通过 `cu_seqlens` 把被复制 expert 的 token 段指向这些 slot。其物理内存来自所有 layer 共享的全局 pool，因此额外开销是总共 $B$ 份 expert 权重，而不是每层 $B$ 份。
- 训练必须取 $B = E/R$；推理（只做 prefetch、无梯度）允许 $B < E/R$，推荐 $B$ 取 3–4。

到这里，EP 全流程逻辑相关的这一组文档就讲完了。如果还想继续往通信和计算算子内部深入，可以转到 [MoE 架构](../../moe/README.md)一章：先看 [05 · Grouped GEMM 与 Expert 计算](../../moe/05_grouped_gemm.md)，了解 expert 计算本身是怎么写成一个高效 kernel 的；再看 [06 · DeepEP：V1 (legacy/NVSHMEM) 与 V2 (elastic/NCCL Gin)](../../moe/06_deepep.md)，把 dispatch/combine 背后的实现细节彻底展开；最后是 [07 · MegaMoE：把 MoE forward 融成单个 kernel](../../moe/07_megamoe.md)，讲怎么把整条链路融合进一个 kernel。
