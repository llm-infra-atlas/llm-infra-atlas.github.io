# 08 · torch.compile / profiler

> 这一篇讲两个用来"打包加速"的特性：`torch.compile` 会自动做算子融合和图优化（并且可以自动套用 CUDA Graph），profiler 则用来定位性能瓶颈在哪里。CUDA Graph 本身已经在 [07](./07_cuda_graph.md) 里单独讲过（包括训练里的 full-layer graph 和推理里的 decode graph），这一篇专注讲 compile 和 profiler 各自的原理、用法，以及容易踩的坑。

---

## 1. torch.compile：自动图优化与算子融合

### 1.1 一行接入

```python
model = torch.compile(model)             # 就这一行；之后正常用
# 或编译单个函数
@torch.compile
def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
```

`torch.compile` 具体做的事情是：先用 TorchDynamo 把 Python 字节码追踪成一张 FX 图（遇到无法追踪的部分就"graph break"，退回到 eager 模式执行），再用 AOTAutograd 同时生成前向和反向两张图，最后交给 Inductor 后端去生成融合后的 Triton（GPU 上）或 C++（CPU 上）kernel。它带来的收益主要来自两方面：一是算子融合，把一长串 element-wise 或归一化 op 融合成一个 kernel，省下访存和 launch 的开销；二是降低了 Python 本身的调用开销。

### 1.2 关键参数

```python
torch.compile(
    model,
    mode="default",          # 见下表
    dynamic=False,           # 是否允许动态 shape（变长序列设 None/True）
    fullgraph=False,         # True=不允许 graph break（强制整图，便于发现问题）
    backend="inductor",      # 默认后端
)
```

| mode | 行为 | 用途 |
|---|---|---|
| `"default"` | 平衡编译时间与性能 | 通用 |
| `"reduce-overhead"` | **自动用 CUDA Graph** 降 launch 开销（见 [07](./07_cuda_graph.md)）| 小 batch / decode |
| `"max-autotune"` | 充分 autotune（含 Triton GEMM 模板），编译慢 | 固定 shape 的极致性能 |

> `mode="reduce-overhead"` 内部做的事情，其实就是把编译出来的图再套上一层 CUDA Graph（前向和反向各录一张，用静态 buffer 加 replay 的方式执行），省去了手写 capture 的麻烦。它对静态 buffer、固定 shape 的要求，以及重新编译会让图失效这些性质，本质上和 [07](./07_cuda_graph.md) 讲的完全一样。多数训练和推理场景其实不需要手写 capture，只有在需要精细控制的场合（比如推理引擎要维护多个 batch-size 对应的图池）才值得手写。

### 1.3 常见问题

- graph break：一旦遇到依赖数据取值的控制流（比如 `if x.sum() > 0`）、`.item()`、不支持的 op，或者 print 这类调用，Dynamo 就会断开当前的图、退回 eager 执行，融合带来的收益也会打折扣。可以用 `fullgraph=True` 强制在断图时直接报错，方便定位问题；也可以用 `TORCH_LOGS="graph_breaks"` 查看具体在哪里断开的。
- 重编译（guard 失效）：compile 会针对输入的 shape、dtype、常量值这些信息设置 guard，一旦某次调用不满足 guard 的条件，就会触发重新编译。变长的 shape 会导致反复重编译——解决办法要么是设 `dynamic=True`，让它直接编译出支持动态 shape 的版本，要么是把 shape 通过 bucketing 限制成有限的几种。如果怀疑重编译次数过多，用 `TORCH_LOGS="recompiles"` 可以看到具体原因。
- 首次调用慢：编译本身有一次性的 warmup 开销，从几秒到几十秒不等，所以第一次调用会比较慢，之后才会体现出编译带来的加速。做性能测量时要记得排除掉这第一次。
- 自定义 kernel 需要注册：手写的 CUDA op 需要通过 `torch.library.custom_op`（2.4+）注册给 compile，让它认识这个算子，否则同样会触发 graph break。

```python
# 注册自定义 op 让 torch.compile 能追踪（而非 graph break）
@torch.library.custom_op("mylib::fused_rmsnorm", mutates_args=())
def fused_rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return my_cuda_kernel(x, w)

@fused_rmsnorm.register_fake          # 给 compile 提供 shape 推导（不真算）
def _(x, w):
    return torch.empty_like(x)
```

### 1.4 和分布式的配合

functional collectives（见 [04 第 6 节](./04_distributed.md#6-devicemesh-functional-collectives)）因为是函数式的，能被 compile 追踪并参与调度，从而实现通信和计算的自动重叠；而传统的 in-place `dist.*` 调用会导致 graph break。像 torchtitan 这样的新框架，用的正是 compile、functional collective 和 DeviceMesh 三者组合起来的方案。

---

## 2. torch.profiler：定位瓶颈

优化性能不应该靠猜。profiler 能直接告诉你时间到底花在哪里、程序是 CPU-bound 还是 GPU-bound、哪个 kernel 最耗时。这里只是一个 30 行左右的快速上手，更完整的内容——包括原理、参数全解、trace 的判读方法论、nsys/ncu/显存分析——见 [Profiling：性能与显存的观测和分析](../profiling/README.md)。

### 2.1 基本用法

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,        # 记录输入 shape（看是否触发慢 kernel）
    profile_memory=True,       # 记录显存分配
    with_stack=True,           # 记录 Python 调用栈（定位到代码行）
) as prof:
    for _ in range(10):
        model(x)
        prof.step()

# 看汇总表
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# 导出 Chrome trace（chrome://tracing 或 perfetto.dev 打开看时间线）
prof.export_chrome_trace("trace.json")
```

### 2.2 用 schedule 跳过 warmup

长时间训练时，通常只想 profile 训练稳定之后的那几步：

```python
my_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)
with profile(activities=[...], schedule=my_schedule,
             on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")) as prof:
    for step in range(10):
        train_step()
        prof.step()            # 必须每步调，驱动 schedule 状态机
```

调度按照 `wait`（跳过不记录）、`warmup`（热身但不记录）、`active`（正式记录）的顺序循环。这样可以避免把编译或者 cudnn autotune 的 warmup 时间也算进最终的统计里。

### 2.3 怎么读 trace

在 Chrome trace 的时间线上，可以关注这几种典型模式：

- 如果 GPU 那条流（CUDA）上出现大片空白，说明程序是 CPU-bound 或者 launch-bound，可以考虑上 CUDA Graph（见 [07](./07_cuda_graph.md)）或者 `torch.compile`，也可以想办法减少 CPU 侧的 Python 开销和同步点。
- 如果某个 kernel 占满了整条时间线且耗时很长，说明瓶颈在计算本身，可以看看能不能做算子融合、换成 flash 系列的 kernel，或者调整精度。
- 如果通信 kernel（比如 nccl）和计算是串行排队的，说明 overlap 没有做好，需要检查 stream 和异步通信的用法（见 [04 第 3 节](./04_distributed.md#3-overlap)）。
- 如果 GPU 那条流被同步点频繁打断，通常要去找 `.item()`、`.cpu()`，或者其他依赖数据取值的 op。

### 2.4 配合 NVTX / nsys

如果需要更细粒度的信息，比如 kernel 内部的执行情况、SM 占用率、访存模式，就要用 NVIDIA Nsight Systems（`nsys`）。可以在代码里打上 NVTX range，给 timeline 加上语义标注：

```python
with torch.cuda.nvtx.range("attention"):
    out = F.scaled_dot_product_attention(q, k, v)
# nsys profile -o report python train.py，再用 Nsight 看
```

`torch.cuda.nvtx.range_push` / `range_pop`，或者对应的 context manager，可以给 trace 打上语义标签，方便定位"到底是哪一段业务逻辑慢"。

---

## 3. 三者的配合方式

CUDA Graph（[07](./07_cuda_graph.md)）、`torch.compile`、profiler 各自负责一段，合在一起就构成了一条相对固定的排查流程：

| 现象（从 profiler 看） | 手段 |
|---|---|
| GPU 空隙多、小 kernel 密集（decode） | `torch.compile(mode="reduce-overhead")` 或手写 CUDA Graph（[07](./07_cuda_graph.md)）|
| element-wise / norm 一大串小 kernel | `torch.compile`（Inductor 自动融合成 Triton kernel） |
| 固定 shape、想把 GEMM 调到极限 | `torch.compile(mode="max-autotune")` |
| 通信没和计算重叠 | 手动 stream + async collective / functional collective + compile |
| 显存峰值过高 | memory snapshot 定位 + activation checkpointing（[03 第 6 节](./03_autograd.md#6-gradient-checkpointing)） |

大致的排查顺序是：先用 profiler 定位瓶颈所在，如果是 launch-bound 就上 compile 或者 CUDA Graph；如果是算力或者访存 bound，就想办法融合算子或者换 kernel、调精度；如果是通信 bound，就去做 overlap。不先定位就盲目套用某种优化手段，往往不但没有效果，反而可能让程序变得更慢。

---

## 4. 全篇主线回顾

回到 [README](./README.md) 里提到的三条主线，到这里应该都能串起来了：

1. Tensor 是 storage 上的一个 view（[01](./01_tensor_memory_layout.md)）：这决定了哪些 op 是零拷贝的，哪些 kernel 能直接吃到这块内存而不需要额外转换。
2. forward 里的 op 决定了 backward 的对偶（[03](./03_autograd.md) 的 autograd、[04](./04_distributed.md) 的通信对偶）：这是写自定义算子和实现并行通信共用的一套统一框架；CUDA Graph（[07](./07_cuda_graph.md)）则把这个对偶关系固化成了 fwd 图和 bwd 图两张静态图。
3. 性能的核心是减少 launch、把访存和通信藏起来（[05](./05_cuda_streams_memory_amp.md) 的 stream 与异步执行、[06](./06_caching_allocator.md) 的 allocator 与 VMM、[07](./07_cuda_graph.md) 的 CUDA Graph、[08](./08_compile_profiler.md) 的 compile 与 profiler）：这是几乎所有加速手段共同追求的目标。

把这三条主线和 [并行策略](../parallel/README.md) 里讲的并行实现、[FlashAttention](../attention/fa/README.md) 里讲的 IO-aware kernel 对照起来看，就构成了一套相对完整的 LLM infra 底层能力图谱。
