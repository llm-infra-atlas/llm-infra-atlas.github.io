# Profiling：性能与显存的观测和分析

这一章要回答一个很具体的问题：模型慢了、显存爆了，不要凭感觉去猜，而是要去测量。测什么、用什么工具测、测出来的图和表该怎么读，就是接下来三篇的内容。

## 前置知识

本章假设读者具备以下背景。如果读到某处发现这些概念还不够用，正文会就地补上最小限度的定义。

- 熟悉一次训练 loop 的构成（forward / backward / optimizer step）。
- 知道 CUDA kernel 是异步执行的：CPU 只负责把 kernel 推进队列就返回，真正的耗时发生在 GPU 自己的时间线上（这一点在 [05 · CUDA 执行模型：stream / event / 显存分配 / AMP](../torch/05_cuda_streams_memory_amp.md) §1 讲过）。
- 知道 caching allocator 的基本行为，也就是 `allocated` 与 `reserved` 的区别（同见该文 §4）。
- [`03`](./03_nsight_compute.md) 那一篇额外要求读过 [00 · Roofline model：性能上界的两道天花板](../hpc/00_roofline_model.md)，因为 ncu 的 SpeedOfLight 分析，本质上就是把单个 kernel 放到 roofline 图上去看它落在哪个位置。

---

## 0. 三个观测维度

这一章的工具可以按「观测对象」分成三类：**时间线**、**显存**、**kernel**。时间线这一类里有两件工具——torch.profiler 看到的是「哪个 op 花了多少时间」，站在框架语义这一层；nsys 看到的是「整条时间线上 CPU 和 GPU 到底谁在等谁」，站在系统这一层。它们读的是同一个系统，区别只在放大倍数，所以放在 [`01`](./01_timeline_tracing.md) 一篇里讲。显存这一类回答的是「显存被谁占着、峰值曲线长什么样」，是 allocator 状态这一层的观测，见 [`02`](./02_memory_profiling.md)。kernel 这一类的 ncu 回答的是「这一个 kernel 为什么跑不满硬件」，是微架构这一层，见 [`03`](./03_nsight_compute.md)。排查问题的关键就是选对维度、再从粗到细逐级下钻，而不是把所有工具从头到尾各用一遍。

```
观测对象：同一个训练 step                       开销        观测窗
┌─────────────────────────────────────────────────────────────────┐
│ 时间线   torch.profiler  op 级:aten::mm、nccl:all_reduce 各花   │  中         几个 step
│                          多久,带 input shape / python stack      │
│          nsys            系统级:CPU 线程、CUDA API、kernel、     │  低~中      秒级~分钟级
│                          memcpy、NVTX range 的完整时间线         │
│ 显存     memory snapshot allocator 级:每个 segment/block 被谁   │  低         任意时刻
│                          分配、峰值曲线、碎片化程度              │
│ kernel   ncu             微架构级:单个 kernel 的 SOL/occupancy/ │  高(数十倍) 单个 kernel
│                          warp stall、在 roofline 上的位置        │
└─────────────────────────────────────────────────────────────────┘
         放大倍数递增 ▶（越微观,开销越大,观测窗必须越小）
```

### 0.1 时间线三问

时间线类的工具读的其实都是同一样东西，也就是 CPU 与 GPU 的两条时间线。不妨先用一张 ASCII 图把它画出来，后面 [`01`](./01_timeline_tracing.md) 的内容其实就是在读这张图的某一个局部：

```
CPU 线程      │ launch│launch│launch│      │launch│ ... │launch│   (异步,发完就返回)
              ▼       ▼      ▼      ▼      ▼             ▼
GPU queue   ┌─────────────────────────────────────────────────────┐
stream 7    │ fwd_gemm │ fwd_gemm │ fwd_attn │ ncclAllReduce │ ... │   ← kernel 依次执行
stream 8    │          │ ncclAllToAll (EP dispatch, 与计算 overlap) │   ← 另一条 stream
copy engine │      H2D │                          │ D2H (ckpt offload)│
```

读懂这条时间线，归根结底只需要问三个问题，记住它们基本上就能看懂所有 profiler 的输出：

1. **GPU 有没有空转**，也就是 stream 上有没有空洞——空转说明 CPU 没有及时提交 kernel，常见原因是 launch-bound、某个同步点，或者 dataloader 太慢。
2. **GPU 在跑什么**，也就是每个 kernel 占了多长时间——耗时最长的一批 kernel，就是接下来要优化的目标。
3. **该并行的部分有没有真正并行**，也就是计算 stream 与通信 stream、copy engine 之间有没有重叠——如果没有重叠，说明 comm-compute overlap 没有做好（这也呼应 [大规模训练的并行策略 —— 总览](../parallel/README.md) 里反复强调的第二条主线）。

### 0.2 工具速查

如果只是想知道「我想搞清楚 X，该用哪个工具」，可以直接查下面这张表：

| 问题 | 工具 | 看哪里 |
|---|---|---|
| 哪个 op / kernel 最贵？ | torch.profiler | `key_averages().table` / trace（[`01`](./01_timeline_tracing.md) §3） |
| GPU 是不是在等 CPU？ | torch.profiler / nsys | trace 上 GPU 行的空洞（[`01`](./01_timeline_tracing.md) §4 / §10） |
| 通信和计算 overlap 了吗？ | nsys | 多条 stream 的 kernel 是否重叠（[`01`](./01_timeline_tracing.md) §10） |
| 显存峰值是多少、被谁占了？ | torch.cuda.memory | `memory_stats` / `memory_summary`（[`02`](./02_memory_profiling.md) §2） |
| OOM 是真不够还是碎片化？ | memory snapshot + memory_viz | 时间线上 reserved vs allocated（[`02`](./02_memory_profiling.md) §3-4） |
| 某一步突然变慢 / 周期性尖刺？ | nsys | 长窗口 timeline 找异常段（[`01`](./01_timeline_tracing.md) §10） |
| 这个 kernel 为什么慢？换 kernel 值不值？ | ncu | SpeedOfLight + roofline + Memory/Compute Workload Analysis（[`03`](./03_nsight_compute.md)） |
| 训练吞吐 / MFU 达没达标？ | 不用 profiler | MFU 定义见 [00 · Roofline model：性能上界的两道天花板](../hpc/00_roofline_model.md) §6，profiler 只负责解释「为什么没达标」 |

---

## 1. 阅读顺序

这组文档一共三篇，读之前不妨先看一眼它们各自覆盖什么、彼此怎么衔接：

| 文件 | 内容 | 代码锚点 |
|---|---|---|
| `README.md`（本文） | 三个观测维度的直观理解、被观测的时间线、工具速查、Megatron 代码映射 | —— |
| [01 · 时间线 tracing：torch.profiler 与 Nsight Systems](./01_timeline_tracing.md) | **时间维度**：torch.profiler 的打点原理（RecordFunction → Kineto → CUPTI）、参数全解、`schedule` 状态机、`key_averages` 表与 chrome trace 怎么读、五类典型瓶颈、分布式集成；nsys 的系统级时间线、`--capture-range` 框窗口、NVTX 标注、GUI/`nsys stats`/sqlite 三种读法、timeline 判读清单 | [[megatron-lm:megatron/training/training.py#L3356]] |
| [02 · 显存 profiling：allocator 模型、snapshot 与 OOM 排查](./02_memory_profiling.md) | **显存维度**：训练显存的构成式、caching allocator 的 segment/block/pool 模型与碎片化的精确定义、`memory_stats` 字段字典、`memory_snapshot` + `_record_memory_history` + memory_viz 工作流、OOM 排查决策树、`PYTORCH_CUDA_ALLOC_CONF` | [[megatron-lm:megatron/training/training.py#L2597]] |
| [03 · Nsight Compute（ncu）：kernel 级微架构分析](./03_nsight_compute.md) | **kernel 维度**：SOL 两面墙与 roofline 的对应、`--set/--section/--metrics`、kernel replay 与计数器权限、从「哪个 kernel 慢」到「为什么慢」的标准流程 | [00 · Roofline model：性能上界的两道天花板](../hpc/00_roofline_model.md) |

建议的阅读顺序是：先读本文，建立「三个观测维度 + 时间线三问」这套框架；接着读 [`01`](./01_timeline_tracing.md)，学会最便宜的第一刀——多数问题到这一步就已经能定位方向；然后读 [`02`](./02_memory_profiling.md) 解决显存类问题，包括 OOM、峰值、碎片，这部分相对自成一体，可以和 `01` 并行读；[`03`](./03_nsight_compute.md) 只有在真的要写或者要换 kernel 时才会用到，适合放在最后读。

之所以把 torch.profiler 和 nsys 合在一篇里，是因为实际排查问题时走的顺序是开销从低到高、观测窗从大到小，而这两件工具共享同一套方法论：框住观测窗、打上语义、按三问读时间线、做量化统计。torch.profiler 只需要在训练脚本里加十几行代码，就能拿到带语义的瓶颈分布；nsys 需要在命令行外面包一层，但换来的是完整的系统级时间线；ncu 会把目标 kernel 重放几十遍，慢上一个数量级，所以只值得花在已经锁定的一两个 kernel 身上。

---

## 2. Megatron 里的集成

参考代码与事实来源均来自上游固定 commit（代码链接带 `#Lx-Ly`，Megatron pin 在 commit `e03878b5f`）：

- [[megatron-lm:megatron/training/training.py]]：训练 loop 里对 torch.profiler、nsys、memory snapshot 三件工具的完整集成，可以当作「工业界怎么用这些工具」的标准答案来读。
- [[megatron-lm:megatron/training/config/common_config.py]]：`ProfilingConfig`，是所有 profiling 开关的权威定义。
- [[megatron-lm:megatron/core/utils.py]]：`nvtx_range_push/pop` 的实现，展示了 NVTX 标注具体是怎么插入到框架代码里的。

Megatron 是「工业界怎么用这些工具」的标准答案，本章各篇会反复引用它的实现。所有开关都集中在 `ProfilingConfig`（[[megatron-lm:megatron/training/config/common_config.py#L25-L67]]），训练 loop 里的接线在 [[megatron-lm:megatron/training/training.py]]：

| 工具 | 开关 | 触发 | 停止 | 产物 |
|---|---|---|---|---|
| torch.profiler | `--profile --use-pytorch-profiler` | `schedule(wait=profile_step_start-1, warmup=1, active=...)`（[[megatron-lm:megatron/training/training.py#L3356-L3363]]） | schedule 自然结束 | 每 rank 一份 chrome trace（[[megatron-lm:megatron/training/training.py#L3352-L3355]]） |
| nsys | `--profile`（不加 `--use-pytorch-profiler`） | `cudaProfilerStart()` + `emit_nvtx`（[[megatron-lm:megatron/training/training.py#L3410-L3412]]） | `cudaProfilerStop()`（[[megatron-lm:megatron/training/training.py#L2933-L2935]]） | nsys-rep（需配合 `nsys profile -c cudaProfilerApi` 启动，命令模板见 [[megatron-lm:megatron/training/config/common_config.py#L30-L33]]） |
| memory snapshot | `--record-memory-history` | 常开记录 | 每 `log_interval` 在 last rank dump（[[megatron-lm:megatron/training/training.py#L2597-L2603]]） | `snapshot.pickle`，拖进 memory_viz 看 |
| NVTX 标注 | `--nvtx-ranges` | `configure_nvtx_profiling(True)`（[[megatron-lm:megatron/training/training.py#L3406-L3407]]） | profile 结束自动关（[[megatron-lm:megatron/training/training.py#L2924-L2926]]） | 给 nsys/ torch trace 加语义层 |

这套集成里有几个值得留意的设计，后文会展开讲：

- **只 profile 少数 rank**（`--profile-ranks`，[[megatron-lm:megatron/training/config/common_config.py#L53]]）：几千卡的任务不可能全量录 trace，所以 `profile_ranks=[]` 之外的 rank 上 profiler 根本不会启动（[[megatron-lm:megatron/training/training.py#L3341-L3344]]）。
- **只录稳定态的两步**（`profile_step_start=10, profile_step_end=12` 是默认值）：这样可以避开 warmup、cudnn autotune、编译期，让录到的 trace 真正代表稳态。
- **memory snapshot 只在 last rank dump**（[[megatron-lm:megatron/training/training.py#L2598]]）：各 rank 的显存行为通常是对称的，录一份就足够代表。
- **NVTX 可以动态开关**（`configure_nvtx_profiling`，[[megatron-lm:megatron/core/utils.py#L2432-L2439]]）：标注本身有开销，所以平时是关掉的，只有在要 profile 的窗口内才打开——[[megatron-lm:megatron/core/transformer/transformer_layer.py#L651]]（`self_attention`）、`:804`（`mlp`）等处的 `nvtx_range_push`，平时其实是空操作。

---

## 3. 与其他章节的关系

- [08 · torch.compile / profiler](../torch/08_compile_profiler.md) §2 给过 torch.profiler 30 行的快速上手和「现象 → 手段」速查表；本章可以看作它的展开版：补上了原理、完整参数、trace 判读方法论，以及 nsys、ncu、显存这三件它没有展开的工具。
- [05 · CUDA 执行模型：stream / event / 显存分配 / AMP](../torch/05_cuda_streams_memory_amp.md) §4 讲了 caching allocator 的基本行为；[`02`](./02_memory_profiling.md) 把它展开成一套可测量、可判读的完整模型。
- [00 · Roofline model：性能上界的两道天花板](../hpc/00_roofline_model.md) 里的 roofline 是 [`03`](./03_nsight_compute.md) 的理论底座：ncu 的 SpeedOfLight 分析做的事情，就是把 kernel 的实测值放到那两道天花板下面去读。
- [大规模训练的并行策略 —— 总览](../parallel/README.md) 的两条主线，是读 trace 时的直接依据：反向通信是前向的镜像，所以 trace 上 bwd 段的通信 kernel 应该和 fwd 段对称；comm-compute overlap 有没有做到，在 nsys 里看两条 stream 是否重叠就能判断。

---

讲完这些，下一个自然的问题是从哪里下手最省事——答案在 [01 · 时间线 tracing：torch.profiler 与 Nsight Systems](./01_timeline_tracing.md)：十几行代码就能拿到「哪个 op 慢、GPU 有没有在等 CPU」的答案，是这一整套工作流里最便宜的第一刀。
