# 09 · DeepInfer 的推理策略

前几篇分别讨论 batching、cache、P/D 分离与并行。本篇以 DeepSeek 开源的 DeepInfer 为案例，把这些机制放回同一条请求路径：模型结构决定需要保存什么，硬件与通信决定怎样分组，调度器再决定每一步给哪些请求分配多少计算。

资料核对日期为 **2026-10-28**，代码与随仓库文档固定到 commit `553741557`。这一版本面向 DeepSeek-V4.1-Flash，默认示例使用 Flash-1017 checkpoint，支持 NVIDIA SM100 与 Huawei Ascend 950。它开放的是离线推理引擎；完整 KV cache 服务、E/P/D 服务编排和跨 DP replica 路由仍需外部系统。下文明确区分已开放的引擎能力与官方介绍的生产系统设计。出处见 [[deepinfer:README.md|DeepInfer 发布说明]]。

## 1. 从模型结构理解系统设计

V4.1 的 MoE、压缩 sparse attention 与 CED 跨层结构，使系统成本与普通 dense Transformer 有明显差别。MoE 需要分布式 expert dispatch；attention 需要管理不同类型的历史状态；后半部 decoder 能复用 encoder 的表示，又给 prefill 裁剪创造了条件。模型背景见 [YOCO 与跨层 KV 复用](../attention/mechanisms/12_cross_layer_kv_yoco.md)。

![DeepSeek 官方 EPD 推理系统架构](assets/deepinfer_architecture.png)

> 图：DeepSeek 官方系统总览。黑色箭头表示请求流，蓝色箭头表示 KV 流；独立存储让计算实例之外也能保留历史。该图描述完整生产系统，并不表示图中所有服务都包含在本次开源引擎内。来源：[[deepinfer:doc/inference_deployment_qa.md|官方 Q&A]]。

图中有两个独立的负载均衡层。外层把不同请求交给不同实例或 attention rank；内层 EPLB 则调整 expert 的物理布局，避免同一个 MoE step 被最忙的 expert rank 拖住。请求数量均衡，不能保证 expert 工作均衡。

## 2. Attention DP 与 Wide EP

### 两种切分服务于不同计算

Attention DP 让不同 GPU 处理不同请求，各自读取这些请求的历史 KV。进入 MoE 后，token 按 router 结果发往持有 expert 的 GPU，再把结果 combine 回原 rank。这意味着 attention 的数据归属与 expert 权重的归属不同，但可以共用同一个并行组。

对一个简化账本，设 routed expert 总权重大小为 $W_e$，EP 度为 $p$，batch 中路由足够分散，则每个 rank 驻留的非冗余 expert 权重约为 $W_e/p$。在小 batch decode 中，读取权重的代价很突出；扩大 EP 可以减少每 rank 的 expert 权重负担，并汇集来自更多 attention rank 的 token，使 expert GEMM 更有效率。

这不是吞吐随 $p$ 必然线性增长：dispatch/combine 的消息规模、拓扑、热点 expert、冗余副本和 collective 尾部都会进入关键路径。若总请求量固定而 EP 不断扩大，每个 expert 得到的 token 太少，也可能得不偿失。

### Prefill 与 decode 选择不同规模

官方 Q&A 描述的生产部署使用较小的 prefill EP/DP 组和较大的 decode Wide-EP 组；其典型范围分别为 2–8 与 64–128。该部署当时不使用 TP/CP，而引擎 README 另外列出了 balanced CP 能力。**支持某种并行方式与生产选择某种并行方式，是不同层面的信息。**

Prefill 本来就有较多 token，可以用较小组获得足够的 GEMM 规模，并让多个实例独立接收请求。Decode 每请求每轮 token 少，更依赖跨请求聚合和权重分摊。选择并行度时，应固定总 GPU 数、输入输出长度和 SLO，比较完整服务的 goodput，而非只看单实例峰值。

### EPLB 用空间换热点缓冲

EPLB 在每张 GPU 上预留少量 redundant expert slot，把热点 expert 复制到多个位置，再在这些副本间分配路由。负载统计使用滑动窗口，权重重排以较慢周期执行；这样既能跟踪分布变化，也避免每轮迁移大块权重。

可以用以下量观察 expert 侧失衡：

$$
\rho=\frac{\max_r T_r}{\frac1p\sum_rT_r},
$$

其中 $T_r$ 为 rank 的 expert 阶段时间。redundant slot 越多，调整空间越大，但常驻权重也越多。Prefill 与 decode 的 token 统计噪声、batch 规模不同，窗口和重排周期应分别配置。这里的原则是让慢速的权重布局适应稳定负载趋势，由快速的 token 路由吸收短时波动。

## 3. 以计算预算决定 prefill chunk

固定 token budget 容易把工作量不同的请求当成一样。处理 $q$ 个新 token 时，历史长度 $c$ 会改变 attention 代价；同样一个 4K chunk，接在短前缀与长前缀后面，执行时间可能不同。

DeepSeek 的生产调度以估计的执行时间作为 quota，结合 context 长度分层，并根据预算求 chunk 大小。为了理解这种调度，可用下面的示意模型：

$$
\widehat T(q,c)=a q+b q c+\gamma q^2+\delta.
$$

第一项代表近似逐 token 的 projection/FFN，第二项代表读取既有历史，第三项代表新 token 之间的交互，最后一项代表固定成本。**这是解释用的模型，不是公开实现的具体拟合式或系数**；CSA 等 sparse/compressed attention 的实际曲线还取决于选择预算、压缩率、算子与硬件，需要 profile。

调度问题变成

$$
q^*=\max\{q:\widehat T(q,c)\le B_{\mathrm{remain}},\quad q\le q_{\mathrm{token\ cap}}\}.
$$

这样同一个时间预算下，较昂贵的上下文自然得到较短 chunk。并行组还要协调各 rank 的可用预算，否则最快的 rank 提前结束，仍需等待最慢的 rank。这个例子把 [chunked prefill](./02_batching_and_chunked_prefill.md) 从「限制 token 数」推进到「限制关键路径时间」。

## 4. Decode 路由的两个负载维度

Decode 不能只看请求数量。两个 GPU 各有 40 条请求，但一边平均历史 2K，另一边 128K，attention 读取量可能差很多；只看 KV 总长度也不够，因为每步 projection、MoE、采样与调度成本还受并发数影响。

一个直观的路由状态是

$$
x_r=(n_r,\ell_r),\qquad
\ell_r=\sum_{i\in\mathrm{requests}(r)}L_i.
$$

$n_r$ 衡量每步并发，$\ell_r$ 粗略衡量历史读取压力。官方策略同时考虑二者；短请求可以补充并发不足但历史较重的 rank，长请求则优先找历史负担较小的 rank。对压缩或 sparse 模型，实际字节数/选中位置数比原始长度更接近真实成本，长度是便于调度的代理量。

还应把预计剩余生成长度与可用 HBM 作为约束。路由追求的是之后一段时间内的均衡，仅让准入时的数字相等，无法避免长尾请求逐渐聚集。

## 5. Cache 按状态性质拆分

V4.1 的 cache 不只有一种。官方系统把可按 block 累积的历史与固定窗口状态分别管理：

| 状态 | 作用 | 官方系统中的存放方式 |
|---|---|---|
| Block KV，包括 main KV 与 indexer key | 支持历史检索与跨请求前缀复用 | 运行时在 HBM，持久副本在独立 3FS 存储集群 |
| SWA state | 保留最近窗口的局部历史 | 运行时在 HBM，跨请求复用的副本在 host memory，通过目录定位 |
| 可重建的 decoder SWA | 为 decode 提供局部状态 | prefill 后以 bounded replay 重建，无需作为完整持久前缀保存 |

独立共享存储使不同计算实例都能读取同一个 prefix，从而弱化 prefix affinity 与负载均衡之间的冲突；同时可以分别扩容计算与存储。代价是 cache hit 增加了 metadata 查询、存储读取与 H2D，命中率本身不再足以衡量收益。

对一次命中，粗略判断条件为

$$
T_{\mathrm{match}}+T_{\mathrm{fetch}}+T_{\mathrm{swap\ in}}
<T_{\mathrm{recompute}}.
$$

真实时间还要扣除与计算重叠的部分。官方实现把远端读取与逐层 H2D 分开流水化：后层 KV 尚在传输时，前层可以开始计算。需要先完成的 prefix match 仍在路由关键路径上，不能把所有 cache 开销都算成可隐藏。

这是一种针对其模型和基础设施的存储组织，不意味着所有系统都应移除 CPU cache 或使用同样的介质。相关权衡见 [分层 cache](./04_prefix_and_hierarchical_cache.md) 与 [P/D 传输](./05_disaggregation_and_kv_transfer.md)。

## 6. Bounded replay 为什么是近似

原始 YOCO 的 cross-decoder 读取共享历史，生成首个 token 时可跳过许多 prompt 位置的后半部计算。V4.1 的 decoder 还有 SWA：若要启动 decode，就需要这些层最近窗口的 KV。由于每一层又依赖前一层的局部历史，严格重建会沿层数扩大所需的输入范围。

若有 $D$ 层 causal window，窗口包含当前位置且宽度为 $w$，一个位置的理论感受野可增长到

$$
1+D(w-1).
$$

重建整个末尾窗口还可能需要更长的输入后缀。具体长度取决于要恢复的状态与层间结构，不能只凭「窗口为 128」就断言只算 128 个 token 一定精确。

Bounded replay 主动截断这种依赖。设只重放从 $s$ 开始的后缀，则 replay 中位置 $i$ 的 SWA 范围变成

$$
[\max(s,i-w+1),i].
$$

对于 replay 开头的 query，原本可见但位于 $s$ 之前的 key 被省略，因此得到的是近似 SWA state。已有 block KV 则按其复用规则保留；不能为了重建 SWA 无意中覆盖它。这个策略以可控的后缀工作量换取 prefill 加速，需要模型质量验证。

引擎默认 `compact` 模式还会缩短实际执行的 token 轴，保留的后缀需要覆盖 SWA window 与 speculative backtracking；README 给出的默认保留长度为 140。这个工程 buffer 长度与 Q&A 用于说明 replay 的 128-token 窗口不是同一个量。

另外，[[deepinfer:examples/yoco/README.md|mask/compact 等价验证]]检查的是两种执行方式在同一 replay 规则下是否一致；它不能证明 bounded replay 与不截断的完整模型计算等价。

## 7. DSpark 分配 verification 预算

固定长度 speculative decoding 会给每个请求验证同样多的 draft token。但不同请求、不同 draft 位置的接受概率不同：如果某请求的第一个候选已很不可靠，再验证它后面的多个候选，收益往往很低。

DSpark 使用 confidence 来分配 variable-length verification。下面从期望收益解释这件事。令 $a_{i,j}$ 是请求 $i$ 在前 $j-1$ 个 draft 均被接受条件下，第 $j$ 个也被接受的概率；验证 $k_i$ 个 draft 时，理想化的平均前进长度为

$$
E_i(k_i)=1+\sum_{j=1}^{k_i}\prod_{u=1}^{j}a_{i,u}.
$$

其中 1 表示标准 verify 过程产生的纠正或额外 token；暂不计 EOS、长度上限与协议变体。第 $j$ 个 draft 的边际价值是前 $j$ 个一起成功的概率。如果 confidence 已表示这种累计概率，就不应再做一次连乘。

例如前三个条件接受概率为 0.9、0.8、0.2，对应的额外收益为 0.9、0.72、0.144。第三个位置的价值明显低于另一条高置信请求的第一个位置。调度还必须满足前缀约束：不能跳过前两个位置，独立验证第三个。

对一个 batch，可以把优化目标写成

$$
\max_{\{k_i\}}
\frac{\sum_iE_i(k_i)}
{T_{\mathrm{step}}(\{k_i\},\mathrm{batch},\mathrm{hardware})}.
$$

分母不能简单视为 verification token 数的线性函数：batch shape、GEMM 利用率、attention 与 collective 都会改变执行时间。官方策略结合 confidence 与 profile 的吞吐曲线选择 verification 前缀。因此「验证更少」不一定更快，尤其在小 batch 或算子利用率尚未饱和时。

Confidence 只决定把验证算力花在哪里；最终接受/拒绝仍由 speculative decoding 的验证规则决定。更完整的分布保持条件见 [speculative decoding](../speculative_decoding/README.md)。

## 8. Async scheduler 如何处理依赖

同步 decode 的 CPU 调度、GPU forward/sample、CPU 后处理首尾串联，会在小步长下暴露明显空隙。异步调度让 CPU 提前准备下一步，GPU 持续执行已排好的工作。

```text
GPU:       forward N → sample N → forward N+1 → sample N+1
CPU:       schedule N+1 / launch   post-process N / prepare next
dependency: token / confidence / finished flags must become valid before use
```

关键是区分可以提前决定的信息与必须等待的信息。比如 DSpark 中，CPU 准备下一 batch 时，上一轮 confidence 可能还没有回来；官方设计先固定总预算，再由 GPU 根据有效 confidence 分配各请求长度。不能把异步调度解释成 CPU 提前知道了尚未生成的 token。

结束请求回收、KV slot 重用、speculative rollback 与 structured-output 状态更新也有类似依赖。优化目标是减少不必要的 host 同步，同时保留这些正确性约束。

## 9. 输出约束与可重复性

Structured output 把 grammar 编译与逐 token 执行分开：编译得到 automaton，运行时根据当前状态生成合法 token mask，再参与 sampling。这样避免每轮重新处理完整 grammar；但 mask 会改变当前目标分布，draft 与 verify 也必须遵守一致的约束与状态推进规则。

Batch-invariant inference 与固定随机种子解决不同问题。前者约束 batch composition 改变时的模型数值行为，后者约束抽样随机流。改变归约顺序造成的 logit 微差，可能改变采样结果；只固定 seed 不一定足够。

这一版本 README 对 deterministic sampling 给出的组合是：在 NVIDIA 上启用 `DEEPINFER_BATCH_INVARIANT=1`，关闭 variable-length MTP，并为请求提供固定 `seq_seed`。这些是特定版本与支持平台下的约定，不能扩展成跨任意硬件、任意算子的逐 bit 保证。

## 10. 怎样评价这些策略

Wide EP 应同时报告总 GPU 数、拓扑、并发与每用户速度；cache 应报告非缓存输入吞吐、命中长度分布和读取延迟；bounded replay 应比较质量与状态误差；DSpark 应报告验证长度、接受长度和最终 goodput。把多项机制一起打开后，只看总 token/s，无法判断哪一项真正解决了瓶颈。

大 EP 组还扩大了故障域：一个设备故障可能让整个通信组停止。生产系统需要容量冗余、请求重路由和状态恢复，不能把单个算子或离线引擎的速度直接等同于线上可用吞吐。

官方 benchmark 的 prefill 与 decode 使用不同 GPU 数量与工作负载，阅读曲线时应保留这些条件。本仓库对这里的机制做原理分析，不把官方测量外推为其他硬件或服务规模下的收益保证。

## 11. 参考入口

| 内容 | 固定版本来源 |
|---|---|
| 开放范围、支持平台与功能开关 | [[deepinfer:README.md]] |
| 生产并行、路由、cache、DSpark、异步调度 | [[deepinfer:doc/inference_deployment_qa.md]] |
| 离线推理与 benchmark 参数 | [[deepinfer:examples/README.md]] |
| YOCO mask/compact 的验证范围 | [[deepinfer:examples/yoco/README.md]] |

以上原理可以分别回看 [batching](./02_batching_and_chunked_prefill.md)、[prefix cache](./04_prefix_and_hierarchical_cache.md)、[P/D 分离](./05_disaggregation_and_kv_transfer.md) 和 [调度与 overlap](./06_scheduling_parallelism_and_overlap.md)。DeepInfer 的案例说明这些决策需要共同设计：模型省下的计算，只有在状态管理、数据传输和调度都配合时，才会转化为服务收益。
