# PyTorch 操作 —— LLM Infra / 算法框架开发常用 API 总结

> 这是一组面向框架开发的 PyTorch 底层 API 参考。阅读它只需要熟悉 PyTorch 张量的基本用法，并写过一次简单的训练 loop。内容按主题整理写 Megatron、vLLM、SGLang、FSDP 这类框架时真正高频用到的底层 torch 操作，逐一讲每个 op 的语义、签名要点、容易出错的地方，以及在大模型组件里的典型用法；需要用到某个定义时会就地补齐，而不会默认读者已经知道。
>
> 代码示例默认已经 `import torch`、`import torch.nn.functional as F`、`import torch.distributed as dist`。

---

## 0. torch 的分层结构

写框架时接触到的 torch 能力，大致可以分成四层，从下到上依次是：

```
┌──────────────────────────────────────────────────────────────┐
│  4. 编译 / 图层      torch.compile, CUDA Graph, profiler        │  ← 把下面三层「打包加速」
├──────────────────────────────────────────────────────────────┤
│  3. autograd 层      Tensor.backward, autograd.Function, hooks │  ← 前反向、自定义算子的反向
├──────────────────────────────────────────────────────────────┤
│  2. op / 算子层      mm, einsum, gather, scatter, SDPA, F.*    │  ← 实际计算
│     分布式 op        all_reduce, all_gather, reduce_scatter... │  ← 跨卡计算
├──────────────────────────────────────────────────────────────┤
│  1. 内存 / 张量层    Storage, stride, view, contiguous, dtype  │  ← 数据怎么躺在显存里
└──────────────────────────────────────────────────────────────┘
        ↓ 全部最终落到 ↓
   CUDA stream / event / caching allocator / CUDA VMM（执行与显存的实际载体）
```

放到具体系统里看，这四层各有自己的出现位置：内存层是一切算子、Megatron 连续 grad buffer、推理侧 KV cache 排布的共同地基；op 层是 attention、MoE router、采样这些组件的实现载体；distributed 层是 DP/TP/PP/EP 各并行维的通信落点；autograd 层承载自定义算子的反向与 gradient checkpointing；编译层则出现在训练的 full-layer / full-iteration CUDA graph 和推理 decode 的 batch-size 分桶 graph 里。本章各篇与这四层一一对应，§1 的文件表会给出具体映射。

全文贯穿三条主线，建议先记住：

1. **Tensor 是 Storage 上的一个 view**。一个 `Tensor` 本质上只是 `(storage, offset, shape, stride)` 这个四元组。`view`、`transpose`、`permute`、`expand`、`[...]` 切片这些操作大多不拷贝数据，只是改了 stride；只有 `contiguous`、`clone`，以及有时候的 `reshape` 才会真正拷贝一份新内存。写算子的时候，先分清楚手上拿到的是 view 还是 copy，是第一位要确认的事情（详见 01）。

2. **forward 里有什么 op，backward 就有它对应的镜像操作**。autograd 会自动把 forward 的计算图反过来走一遍。手写算子时，需要做的就是在 `autograd.Function.backward` 里实现这个镜像（详见 03）。分布式通信里也是同样的道理：`all_gather` 的反向是 `reduce_scatter`，`all_reduce` 的反向则是 identity（详见 04）。

3. **性能优化的核心，是把 kernel launch 和访存的开销藏起来**。CUDA 本身是异步的：Python 侧调用一个 `op()`，其实只是把对应的 kernel 塞进 stream 队列就立刻返回了。CUDA Graph 消除的是重复 launch 的开销，stream 用来做计算和通信的 overlap，`torch.compile` 做的是算子融合——这些手段本质上都是在减少 launch 次数、提高每次 kernel 里的有效计算量（详见 05/07/08）。显存这一侧对应的是 caching allocator 加 CUDA VMM：目标是减少同步、减少碎片，让显存段可以按页粒度伸缩（详见 06）。

---

## 1. 这组文档怎么读

这是一份面向框架开发的 torch 底层 API 参考，不是入门教程。每一篇聚焦一类底层能力，给出对应的 API、语义，以及在 LLM 场景下的具体用法。下面这张表按阅读顺序列出了各篇的内容和关键词：

| 文件 | 内容 | 关键词 |
|---|---|---|
| `README.md`（本文） | 全景：torch 各层能力的整体划分，以及一张按需求查 op 的速查表 | —— |
| [01 · Tensor 与内存布局：view / stride / contiguous / dtype](./01_tensor_memory_layout.md) | Tensor / Storage / stride / view vs copy / contiguous / `as_strided` / dtype / device / memory_format —— 一切算子的基础 | view, stride, contiguous |
| [02 · 计算 op：matmul / einsum / reduction / gather-scatter / SDPA](./02_compute_ops.md) | matmul 家族（`mm/bmm/matmul/einsum/addmm`）、reduction、broadcasting、indexing / `gather` / `scatter` / `index_*`、`cat/stack/split`、`topk/sort/cumsum`、SDPA、LLM 高频 fused op | matmul, gather/scatter, einsum, SDPA |
| [03 · autograd：引擎、自定义 Function、hooks、checkpoint](./03_autograd.md) | autograd 引擎、`requires_grad`、`backward` vs `grad`、自定义 `autograd.Function`（含 `setup_context`）、saved tensors、各种 hook、`no_grad/inference_mode`、gradient checkpointing、`vmap/grad`（functorch） | custom Function, checkpoint, hooks |
| [04 · torch.distributed：通信原语、process group、DeviceMesh](./04_distributed.md) | `init_process_group`、ProcessGroup / sub-group、所有 collective（`all_reduce/all_gather/reduce_scatter/all_to_all/broadcast/p2p`）、async work、`ReduceOp`、`DeviceMesh`、functional collectives、NCCL 调优 | collective, P2P, DeviceMesh |
| [05 · CUDA 执行模型：stream / event / 显存分配 / AMP](./05_cuda_streams_memory_amp.md) | device/stream/event、`synchronize`、caching allocator 观测 API、pinned memory、`autocast` / AMP / `GradScaler` | stream, autocast |
| [06 · Caching Allocator 与 CUDA VMM](./06_caching_allocator.md) | caching allocator 三级结构、碎片化、CUDA VMM（VA/PA 解耦）、`expandable_segments`、`PYTORCH_ALLOC_CONF` / `MemPool` / `cudaMallocAsync` | allocator, VMM, expandable segments |
| [07 · CUDA Graph](./07_cuda_graph.md) | CUDA Graph capture / replay 的原理与硬性约束；训练里的 full-layer graph（Megatron `local` 自研 fwd+bwd 双图 / `full_iteration` / TE）；推理 decode 的 batch-size 分桶 graph（SGLang） | CUDA Graph, capture/replay, full-layer, decode |
| [08 · torch.compile / profiler](./08_compile_profiler.md) | `torch.compile`（mode / dynamic / custom op / guards、自动套 CUDA Graph）、`torch.profiler` / nsys | compile, profiler |

建议按这个顺序读：01（内存模型是后面一切的基础）→ 02（计算 op）→ 03（autograd，写自定义算子必读）→ 04（分布式）→ 05（stream / 异步）→ 06（allocator / VMM）→ 07/08（graph / compile）。

---

## 2. 常用需求速查表

下面按高频需求反查该用哪个 op，详细语义见对应章节。

### 形状与内存（→ [01](./01_tensor_memory_layout.md)）

| 想干什么 | 用 | 注意 |
|---|---|---|
| 加 / 去掉维度 | `unsqueeze` / `squeeze`，或 `x[None]` / `x[:, 0]` | view，不拷贝 |
| 重排维度 | `transpose(a,b)` / `permute(...)` / `movedim` | view，**结果非 contiguous** |
| 改形状 | `view`（要求 contiguous）/ `reshape`（必要时拷贝）/ `flatten` | `reshape` 更安全，`view` 更快且能暴露 bug |
| 广播到目标形状 | `expand`（view，stride=0）/ `broadcast_to` | 不要 `repeat`（会拷贝），除非真要独立内存 |
| 合并 / 拆分 batch 维 | `view(-1, ...)` / `reshape` | flatten token：`x.view(-1, h)` |
| 手动构造 sliding window / 重叠块 | `as_strided` / `unfold` | 危险但强大，零拷贝造重叠 view |
| 保证内存连续（喂给 kernel 前） | `contiguous()` / `contiguous(memory_format=...)` | 已连续则 no-op |

### 计算（→ [02](./02_compute_ops.md)）

| 想干什么 | 用 |
|---|---|
| 普通 GEMM | `torch.matmul` / `@` / `F.linear`（带 bias，权重要转置语义） |
| batched GEMM | `torch.bmm` / `matmul`（广播 batch 维） |
| 任意爱因斯坦求和（attention score、MoE 等） | `torch.einsum("...", a, b)` |
| GEMM + bias 融合 | `torch.addmm` / `baddbmm` |
| 按索引取行（embedding、MoE gather token） | `F.embedding` / `index_select` / `x[idx]` |
| 按索引散射写回（MoE combine、scatter add） | `scatter_` / `scatter_add_` / `index_add_` / `index_copy_` |
| 沿某维按 index 收集（top-k routing） | `gather` |
| top-k / argmax（router、采样） | `topk` / `argmax` / `argsort` |
| softmax / logsumexp（数值稳定） | `F.softmax` / `torch.logsumexp` / `F.log_softmax` |
| attention（flash 路径） | `F.scaled_dot_product_attention` |
| RMSNorm / LayerNorm | `F.rms_norm` / `F.layer_norm`（或手写） |
| 激活 | `F.silu` / `F.gelu` / SwiGLU 手写 |
| 前缀和 / 累积（MoE offset、序列打包） | `cumsum` / `cumprod` |

### autograd（→ [03](./03_autograd.md)）

| 想干什么 | 用 |
|---|---|
| 写一个有自定义反向的算子 | `class X(torch.autograd.Function)` + `forward/backward`（或 `setup_context`） |
| 只前向、不建图（推理 / 评估） | `with torch.inference_mode():`（比 `no_grad` 更彻底） |
| 局部断开梯度 | `x.detach()` / `with torch.no_grad():` |
| 拿梯度但不写进 `.grad` | `torch.autograd.grad(out, inputs)` |
| 省激活显存换重算 | `torch.utils.checkpoint.checkpoint` |
| 在反向时看 / 改梯度 | `tensor.register_hook` / `module.register_full_backward_hook` |
| 向量化（per-sample grad、jacobian） | `torch.func.vmap` / `grad` / `jacrev` |

### 分布式（→ [04](./04_distributed.md)）

| 想干什么 | 用 |
|---|---|
| 求和 / 平均梯度（DP/TP） | `dist.all_reduce(t, op=ReduceOp.SUM)` |
| 收集分片（FSDP unshard、SP all-gather） | `dist.all_gather_into_tensor` |
| 散射求和（ZeRO grad、SP reduce-scatter） | `dist.reduce_scatter_tensor` |
| MoE dispatch / combine | `dist.all_to_all_single` / `all_to_all` |
| PP 层间传激活 | `dist.batch_isend_irecv`（避免死锁） |
| 广播权重 / 同步随机种子 | `dist.broadcast` |
| 描述多维并行拓扑 | `DeviceMesh` + functional collectives |

### 性能（→ [05](./05_cuda_streams_memory_amp.md) / [06](./06_caching_allocator.md) / [07](./07_cuda_graph.md) / [08](./08_compile_profiler.md)）

| 想干什么 | 用 |
|---|---|
| compute / comm overlap | 多 `torch.cuda.Stream` + `event` 同步 |
| 消除小 kernel 的 launch 开销（decode / 训练 full-layer） | CUDA Graph capture / replay（[07](./07_cuda_graph.md)）|
| 混合精度训练 | `torch.autocast` + `GradScaler`（fp16）/ 裸 autocast（bf16） |
| 看显存峰值 / 找泄漏 | `torch.cuda.memory_summary` / `_record_memory_history` + snapshot（机制见 [06](./06_caching_allocator.md)） |
| 变长 batch / 碎片化 OOM | `PYTORCH_ALLOC_CONF=expandable_segments:True`（[06](./06_caching_allocator.md)） |
| H2D 拷贝重叠 | pinned memory（`pin_memory=True`）+ `non_blocking=True` |
| 算子融合、降图开销 | `torch.compile(model)`（[08](./08_compile_profiler.md)）|
| 定位瓶颈 | `torch.profiler.profile` + Chrome trace（[08](./08_compile_profiler.md)）|

---

## 3. 版本与查证约定

- 文档以 **PyTorch 2.x**（2.3+）的稳定 API 为准，明确标注哪些是 2.x 新增（`F.scaled_dot_product_attention`、`DeviceMesh`、functional collectives、`torch.compile`、`torch.func`、`inference_mode`、`F.rms_norm`、`expandable_segments` / `MemPool`）。
- 凡是「in-place」「是否拷贝」「是否可微」这类性质，文中会显式标注，因为这正是框架 bug 的高发区。
- 所有 collective 的 forward/backward 对偶、in-place 语义，和并行策略里的实现是同一套，遇到具体并行场景请交叉参考 [并行策略总览](../parallel/README.md)。

下一篇：[01 · Tensor 与内存布局：view / stride / contiguous / dtype](./01_tensor_memory_layout.md)，先讲清 Tensor 到底是什么、什么时候拷贝、什么时候不拷贝——这是写任何底层算子的基础。
