# 大规模训练的并行策略总览

> 这一组文档从 infra 的视角，把大规模 LLM 训练里用到的各种并行维度逐一讲清楚。对每一种并行，都会说清楚它切分的是什么、通信是怎么发生的、前向和反向之间如何对称、以及它在显存和带宽之间做了怎样的取舍，并逐段对齐 Megatron-LM（训练）、PyTorch FSDP、DeepEP 等真实代码。
>
> 在整站的知识地图里，这一组文档处于「空间维」的位置：[GPU 集群与网络](../hpc/README.md) 那组文档讲清了硬件、网络与集合通信的地基，本章回答的是模型和 batch 怎么切到成千上万张卡上；切好之后，一个 iteration 从取数据到权重更新、落盘的完整生命周期，则是 [训练系统：一个 iteration 的完整生命周期](../train/README.md) 那组文档的内容。

## 前置知识

每个维度需要的前置知识不太一样，具体见各自子目录的 README；共通的要求如下，正文用到更细节的前置概念时会就地补上定义。

- 熟悉 matmul 与 autograd 的基本用法；可先读 [02 · 计算 op：matmul / einsum / reduction / gather-scatter / SDPA](../torch/02_compute_ops.md) 与 [03 · autograd：引擎、自定义 Function、hooks、checkpoint](../torch/03_autograd.md)。
- 知道集合通信的基本语义（all-reduce / all-gather / reduce-scatter / all-to-all）；见 [集合通信：原语、算法、NCCL 实现与拓扑映射](../hpc/04_collectives.md)。
- 熟悉 transformer 的层结构（attention / MLP / LayerNorm）。

## 五个并行维度

训练里常用的并行维度一共有五个，下表先给一个横向对照：

| 维度 | 切什么 | 主通信原语 | 通信域 | 文档 |
|---|---|---|---|---|
| **DP / ZeRO / FSDP** | batch + optimizer/grad/param 分片 | all-reduce / RS+AG | `data_parallel_group`（含 CP） | [Data Parallelism (DP) / ZeRO / FSDP](./01_dp/README.md) |
| **TP / SP** | 单层权重的 hidden/head 维 + activation 的 seq 维 | all-reduce / all-gather / reduce-scatter | NVLink 域（机内） | [Tensor Parallelism (TP) 与 Sequence Parallelism (SP)](./02_tp_sp/README.md) |
| **PP** | 层（深度维） | P2P send/recv | `pipeline_model_parallel_group` | [Pipeline Parallelism (PP)](./03_pp/README.md) |
| **CP** | attention 的 sequence 维 | ring P2P / all-to-all | `context_parallel_group` | [Context Parallelism (CP)](./04_cp/README.md) |
| **EP**（MoE） | expert | all-to-all | `expert_parallel_group` | [Expert Parallelism (EP)](./05_ep/README.md) |

## 维度的组合方式

这五个维度并不是互斥的选项，而是可以叠加使用的正交切分。下面是一种典型的组合方式：

```
world_size = DP × CP × TP × PP            (× EP 在 MoE 里复用 DP×TP 的切分)

rank 排布(典型, 内→外):  TP → CP → PP → DP
                         └NVLink┘   └──── IB(跨机) ────┘
```

- TP 放在最内层，也就是机内的 NVLink 域：它的通信量大、频率高，对带宽最敏感。
- CP 沿序列维切分：长上下文场景下，把 attention 里 `O(s²)` 的开销摊到多张卡上。
- PP 跨机扩展层数：P2P 通信是低频小包，即使走 IB 也扛得住，但会带来 bubble。
- DP 放在最外层，用来扩大 batch：ZeRO-1 / FSDP 进一步把 optimizer 状态和参数也切分开，从而节省显存。
- EP 在 MoE 里把 expert 切开，dispatch 和 combine 都通过 all-to-all 通信完成。生产级 MoE 往往把 TP 收到 1（或者只留给 attention），把机内带宽让给 EP；原因见 [Tensor Parallelism (TP) 与 Sequence Parallelism (SP)](./02_tp_sp/README.md) §7。

## 贯穿所有文档的两条主线

这一组文档会反复回到两个共同的主题，帮助在不同并行维度之间建立类比。

1. **forward / backward 的通信对称性**：每种并行的反向都是前向的镜像，比如 TP 里 `f`/`g` 共轭算子的对应关系、CP/EP 里 all-to-all 的反向仍然是 all-to-all、MoE 里 dispatch 的反向正是 combine、FSDP/SP 里 all-gather 的反向是 reduce-scatter。只要把前向通信写对，autograd 就会自动给出正确的反向通信。
2. **通信-计算 overlap 与 `CUDA_DEVICE_MAX_CONNECTIONS=1`**：几乎所有并行策略的性能，最终都取决于能不能把集合通信藏进计算时间里，这需要异步发起通信，并用单一硬件队列保证通信按预期的顺序被发射。DualPipe / combined-1F1B 是把这套 overlap 思路做到极致的代表。

## 每个目录的结构

并行相关的每个子目录都遵循同样的组织方式：

```
<dim>/
├── README.md              全景 + 阅读指引 + 代码映射表
├── 0N_*.md                逐主题深入(逐行对齐 Megatron/torch 代码)
├── _build_nb.py           生成 lab notebook 的脚本
└── <dim>_lab.ipynb        纯 torch + 真 torch.distributed(gloo) 的可运行 lab
```

每个 lab 都能在 Mac 的 CPU 上跑起来：用 `gloo` 后端加上 `mp.spawn` 在本地起多个进程模拟多卡环境，构造一个小规模的 test case，亲手实现对应并行策略的前向和反向，并和单进程的 reference 实现逐元素对齐。lab 里的关键代码块都能对应到正文的具体章节。

## 建议阅读顺序

如果不知道从哪里开始，建议按下面的顺序读：

1. 先读 [Data Parallelism (DP) / ZeRO / FSDP](./01_dp/README.md)，这是最普适的一种并行：几乎所有训练任务都会用到 DP，而 ZeRO/FSDP 的显存账本也是后面几个维度的基础。
2. 接着读 [Tensor Parallelism (TP) 与 Sequence Parallelism (SP)](./02_tp_sp/README.md)，理解 `f`/`g` 共轭算子是搞懂层内通信的关键；MoE 时代 TP 为什么逐渐被弃用，也在这一篇里讲到。
3. 然后是 [Pipeline Parallelism (PP)](./03_pp/README.md)，讲清楚 bubble、显存和通信之间的三角关系。
4. 再读 [Context Parallelism (CP)](./04_cp/README.md)，对比 ring 和 Ulysses 两种做法，面向的是长上下文场景。
5. 最后是 [Expert Parallelism (EP)](./05_ep/README.md)，EP 的核心是 all-to-all 通信，也把前面四个维度串到了一起。
