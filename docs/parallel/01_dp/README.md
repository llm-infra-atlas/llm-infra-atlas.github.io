# Data Parallelism（DP）、ZeRO 与 FSDP

DP 是最普适的一种并行：多卡各持一份完整的模型副本，各自用不同的 data shard 训练。本章顺着这个直觉往下延伸——从最朴素的 gradient all-reduce 出发，一路推到 ZeRO-1/2/3 在显存和通信之间的权衡、Megatron 的 distributed optimizer，以及 FSDP 逐层做 all-gather/reshard 的机制，最后用 Megatron-LM 与 PyTorch 的实现核对这些机制。ZeRO 到底在分片什么、通信为什么是对称的，都会先讲清楚定义，再讲工程上怎么实现。

## 前置知识

- 知道最朴素的 data parallelism 直觉：多卡各持一份完整的模型副本，各自用不同的 data shard 训练。
- 知道 `all-reduce` 的语义；不熟悉的话可以先看 [集合通信：原语、算法、NCCL 实现与拓扑映射](../../hpc/04_collectives.md)。
- 知道 autograd 的基本语义（forward 建图、backward 沿图反传梯度）；见 [03 · autograd：引擎、自定义 Function、hooks、checkpoint](../../torch/03_autograd.md)。
- 知道 Adam 与混合精度训练的基本组成（bf16 参数/梯度、fp32 master、optimizer states `m`/`v`），§1 的显存账本会直接用到这些量。

---

## 1. 朴素 DP 的显存代价

最朴素的 data parallelism 是这样运作的：每张卡持有一份完整的模型副本，各自处理自己的 micro-batch，backward 算完梯度后，对梯度做一次 **all-reduce** 求平均，再各自用这份相同的梯度去更新参数——这样一来，各卡上的参数在整个训练过程中始终保持一致。

```mermaid
flowchart LR
    subgraph R0["rank0 (完整副本)"]
        f0["fwd/bwd on batch0"] --> g0["grad0"]
    end
    subgraph R1["rank1 (完整副本)"]
        f1["fwd/bwd on batch1"] --> g1["grad1"]
    end
    g0 & g1 -->|"all-reduce (求和/平均)"| G["相同的 grad"]
    G --> u["各 rank 用相同 grad 更新 → 参数仍一致"]
```

问题在于，每张卡都要存下完整的参数、梯度和 optimizer states。对 Adam 加混合精度训练来说，按参数元素数 $P$ 来算，单个参数的显存账本是这样的：

| 项 | 每元素字节 | 全模型字节 |
|---|---|---|
| 参数（bf16） | 2B | $2P$ |
| 梯度（bf16） | 2B | $2P$ |
| fp32 master 参数 | 4B | $4P$ |
| Adam `m`（fp32） | 4B | $4P$ |
| Adam `v`（fp32） | 4B | $4P$ |
| **合计** | | **$16P$** |

朴素 DP 下每张卡都要背下这 $16P$ 的显存开销，其中 $12P$（optimizer states 加 fp32 master）纯粹是冗余复制的结果：$N$ 张卡存了 $N$ 份完全相同的东西。ZeRO 想解决的问题正是这里，把这些冗余按 DP 维度切开，分散到各卡上。

## 2. ZeRO 频谱：三级显存分片

ZeRO（Rajbhandari et al., *ZeRO*, 2019, [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)）把这件事分成三级来做，逐步把三类状态分片到 $N=\mathrm{DP}$ 张卡：

```mermaid
flowchart LR
    DDP["朴素DP/DDP\n每卡 16P\n(全复制)"]
      -->|"切 optimizer states"| Z1["ZeRO-1\n2P+2P+12P/N"]
    Z1 -->|"再切 gradients"| Z2["ZeRO-2\n2P+2P/N+12P/N"]
    Z2 -->|"再切 parameters"| Z3["ZeRO-3 / FSDP\n16P/N"]
```

![ZeRO per-device memory consumption across Pos/Pg/Pp stages](assets/arxiv/1910.02054_zero_memory.png)

> 图：ZeRO 论文的经典显存账本图（记号 $\Psi$=参数量、$K$=optimizer state 系数，Adam 混精下 $K=12$、基线每参数 $2+2+K=16$ 字节、$N_d$=DP 度）。$P_{\mathrm{os}}$（ZeRO-1，切 optimizer states）→ $P_{\mathrm{os+g}}$（ZeRO-2，再切 gradients）→ $P_{\mathrm{os+g+p}}$（ZeRO-3/FSDP，再切 parameters），每卡显存从 $16\Psi$ 一路降到 $16\Psi/N_d$。这张图与下表逐行对应。（Rajbhandari et al. 2019, Fig 1；[arXiv:1910.02054](https://arxiv.org/abs/1910.02054)）

下面这张表把每一级切分的对象、每卡显存和通信量放在一起，方便对照：

| | 切什么 | 每卡显存（Adam 混精） | 通信/步（同精度元素量，省略有限 N 系数） | 对应实现 |
|---|---|---|---|---|
| **DDP** | 无（全复制） | $16P$ | all-reduce grad = **$2P$** | torch DDP, Megatron DDP (`no_shard`) |
| **ZeRO-1** | optimizer states + master | $4P + 12P/N$ | RS grad + AG param = **$2P$** | **Megatron `DistributedOptimizer`** |
| **ZeRO-2** | + gradients | $2P + 2P/N + 12P/N$ | RS grad + AG param = **$2P$** | DeepSpeed ZeRO-2 |
| **ZeRO-3** | + parameters | **$16P/N$** | RS grad + 2× AG param = **$3P$** | **FSDP** (torch / Megatron-FSDP) |

上表的显存只统计长期模型状态，通信按参数与梯度同精度、每步一次 forward/backward、ZeRO-3 在 forward 后 reshard 的理想模型统计。ZeRO 阶段描述的是状态的存储方式，不由某一次 collective 的名字决定。尤其是 reduce-scatter 后只使用梯度 shard，并不能证明完整 grad buffer 已经释放。

## 3. 通信量的口径

令 $b_p,b_g$ 分别为参数和梯度通信的每元素字节数，$c_N=(N-1)/N$。按 ring 风格模型，每个 rank 每步发送的字节数为

$$
\begin{aligned}
V_{\mathrm{DDP}}&=2c_NPb_g,\\
V_{\mathrm{ZeRO\text{-}1/2}}&=c_NP(b_g+b_p),\\
V_{\mathrm{ZeRO\text{-}3}}&=c_NP(b_g+2b_p).
\end{aligned}
$$

仅在参数、梯度同精度时，前两项相等，第三项为它们的 1.5 倍。若参数为 BF16、规约梯度为 FP32，比例就不同；gradient accumulation、保留参数和 activation recompute 还会改变 collective 次数。「通信量相同」也不保证耗时相同，因为 bucket 数、启动延迟与 overlap 都会改变关键路径。完整推导及具体数字见 [ZeRO](./02_zero_and_distributed_optimizer.md)。

## 4. DP 在并行体系中的位置

```
world = DP × CP × TP × PP   ;  DP 在最外层(跨机)
```

DP 不是孤立存在的，它要和 TP、PP、CP、EP 这些其它并行维度耦合在一起。下面这张表梳理了几种主要的耦合关系：

| 耦合 | 要点 |
|---|---|
| **DP × CP** | 梯度规约域是 `dp_cp`（CP 在梯度上像 DP，见 [04 · Megatron-LM 实现：切分、传参、RoPE、hybrid CP、与其它并行的协同](../04_cp/04_megatron_cp_integration.md)）。Megatron 用 `data_parallel_group(with_context_parallel=True)` |
| **DP × TP** | TP 的梯度本来就切开（每卡不同权重分片），DP 在 TP 之上对「相同分片」做规约。`CUDA_DEVICE_MAX_CONNECTIONS=1` 让 TP/DP 通信错峰（[04 · TP/SP 的通信-计算 overlap 与工程优化](../02_tp_sp/04_overlap_and_optimizations.md)） |
| **DP × PP** | PP 各 stage 是不同的 DP 副本组；DDP 的 bucketing 在非 first stage / interleaved 的后续 chunk 会调整（[[megatron-lm:megatron/core/distributed/distributed_data_parallel.py#L105]]）|
| **DP × EP** | MoE 的 expert 有独立的 `expert_data_parallel_group`；expert 参数的 DP 规约域不同于 attention（见 [EP](../05_ep/README.md)） |

## 5. Megatron 的统一抽象：连续 buffer 与 `main_grad`

Megatron 并不会把每个参数的 `.grad` 单独拿去做 all-reduce，那样会产生成百上千个小 kernel，效率很差。它的做法是用一块连续的 grad buffer（`_ParamAndGradBuffer`，定义在 `param_and_grad_buffer.py` 里）把同 dtype 的所有梯度拼接在一起，再按 bucket 分块通信。配合 TP 侧的 `gradient_accumulation_fusion`（详见[01 · ColumnParallelLinear / RowParallelLinear 与核心 autograd](../02_tp_sp/01_linear_layers.md)），wgrad 会直接累加进这块 buffer 的 fp32 视图 `main_grad` 里，不需要额外的加法 kernel。这样一来，DP 的梯度同步就变成了对这一整块连续 buffer 做 reduce-scatter 或 all-reduce：通信颗粒更大、kernel 数量更少，也更容易和计算 overlap 起来。这正是 01 要讲的内容。

```
[param0.grad][param1.grad]...[paramK.grad]   ← 一块连续 grad buffer (按 dtype)
 └────────── bucket 0 ──────────┘└─ bucket1 ─┘
   每个 bucket 填满就触发一次异步 reduce-scatter (overlap 进 backward)
```

---

## 6. 一组贯穿全文的数字（70B、DP=64）

```
P = 70e9            参数量
DP = 64
Adam 混合精度: 每参数 16 字节(全复制) → 单卡 1120 GB(!!)  ← 朴素 DP 根本放不下
```

- 朴素 DP：optimizer 与 master 合计 $12P$，约 840 GB/卡，显然放不下。
- ZeRO-1（distributed optimizer）：optimizer 部分降到 $12P/64$，约 13 GB/卡；但参数与梯度仍有 $4P$，约 280 GB，因此还需要配合 TP/PP 进一步切分。
- ZeRO-3/FSDP：全部状态合计 $16P/64$，约 17.5 GB/卡，代价是每层 forward 都要 all-gather 该层参数。

这组数字没有计入 activation 与完整 unit 的瞬时参数。70B 既可采用 ZeRO-1 × TP × PP，也可以采用 FSDP/HSDP 或与 TP/CP 的组合；选择取决于网络、batch、序列长度与单层大小，不能由总参数量单独决定。

---

## 这组文档怎么读

下面这张表列出阅读路径。先理解数据与状态如何分布，再看执行优化与实现对照：

| 文件 | 内容 | 对应代码 |
|---|---|---|
| `README.md`（本文） | DP 全景：显存账本、ZeRO-1/2/3 频谱、通信量口径、DDP↔ZeRO↔FSDP 对应、并行维度 | 配置类 |
| [01 · Megatron DDP：连续 buffer 与通信 overlap](./01_ddp_and_overlap.md) | Megatron DDP：连续 grad buffer、`main_grad`/fp32 累加、bucket + `register_grad_ready` hook、`overlap_grad_reduce`、all-reduce vs reduce-scatter | `distributed_data_parallel.py`, `param_and_grad_buffer.py` |
| [02 · ZeRO 显存账本与 Megatron DistributedOptimizer](./02_zero_and_distributed_optimizer.md) | ZeRO-1/2/3 显存账本推导；Megatron `DistributedOptimizer`：optimizer state 分片、grad reduce-scatter、param all-gather、`overlap_param_gather` | `distrib_optimizer.py`, `param_and_grad_buffer.py:351/556` |
| [03 · FSDP1 与 FSDP2](./03_fsdp.md) | FSDP 的 full-shard 模式：逐层 all-gather 参数→算→reshard 释放→反向再 all-gather→reduce-scatter 梯度；PyTorch FSDP2 `fully_shard` 的 unshard/reshard/post_backward、stream prefetch；Megatron-FSDP | torch FSDP2, `fsdp/` |
| [[atlas:docs/parallel/01_dp/dp_lab.ipynb]] | 纯 torch 手写 **DDP / ZeRO-1 / ZeRO-3(FSDP)** 三种，用 gloo 本地多进程跑通 all-reduce / reduce-scatter / all-gather，逐元素对齐单进程 reference | —— |

读的顺序建议是：先读本文建立显存账本和 ZeRO 频谱的整体框架，再看 01 里 DDP 的 bucket 与 overlap 工程实现，然后是 02 的 optimizer state 分片，接着 03 讲 FSDP 如何把参数也分片，最后动手做一遍 lab，把三种方式都亲手实现一遍。

## 参考代码

参考代码（commit `e03878b5f` / torch 2.8）：

- [[megatron-lm:megatron/core/distributed/distributed_data_parallel.py]] —— Megatron DDP（grad bucketing + overlap）
- [[megatron-lm:megatron/core/distributed/param_and_grad_buffer.py]] —— 连续 grad/param buffer、`start/finish_grad_sync`、`start_param_sync`
- [[megatron-lm:megatron/core/optimizer/distrib_optimizer.py]] —— `DistributedOptimizer`（ZeRO-1：optimizer state 分片）
- [[megatron-lm:megatron/core/distributed/distributed_data_parallel_config.py]] —— `overlap_grad_reduce`/`overlap_param_gather`/`use_distributed_optimizer`/`data_parallel_sharding_strategy`
- PyTorch FSDP2：[[pytorch:torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py]]（`unshard`/`reshard`/`post_backward`）、`_fsdp_collectives.py`（`foreach_all_gather`/`foreach_reduce`）

---

讲完了显存账本和 ZeRO 的整体频谱，下一个自然的问题是：Megatron DDP 具体怎么用连续 buffer、bucket 和 grad-ready hook，把梯度的 all-reduce/reduce-scatter 藏进 backward 的计算里？这就是[01 · Megatron DDP：连续 buffer 与通信 overlap](./01_ddp_and_overlap.md)要讲的内容。
