# 05 · CUDA 执行模型：stream / event / 显存分配 / AMP

> 前几篇讲的所有 op，最终都要落到 CUDA 上以异步的方式执行。这一篇讲的是 GPU 执行的真实模型：kernel 是怎么排队的、stream 怎么用来做 overlap、caching allocator 有哪些观测 API、混合精度又是怎么配置的。至于 allocator 的三级结构、碎片化的成因、CUDA VMM 与 `expandable_segments` 的细节，会单独放到 [06](./06_caching_allocator.md) 里展开。这些内容共同决定了一个框架的性能上限。

---

## 1. CUDA 的异步执行模型

在 Python 侧调用一个 op（比如 `y = x @ w`）时，CPU 只是把对应的 kernel 塞进 stream 队列就立刻返回了，GPU 在后台慢慢把它执行完。也就是说，CPU 和 GPU 是并行推进的，两者互不等待。

```python
y = model(x)        # 立即返回，kernel 还在 GPU 上排队/执行
loss = y.sum()      # 继续往队列里塞，CPU 不等 GPU
print(loss.item())  # .item() 触发同步！CPU 在这里阻塞等 GPU 算完
```

理解了这一点，有几条推论是框架开发时必须记在心里的：

- 计时必须先同步：要在 `torch.cuda.synchronize()` 之后再读时间，否则测到的只是"把 kernel 塞进队列"所花的时间，而不是真实的计算耗时。
- 任何把数据读回 CPU 的操作都会构成一个同步点，比如 `.item()`、`.cpu()`、`.tolist()`、`print(tensor)`、`bool(tensor)`、`.nonzero()`（因为它的输出形状依赖数据本身）。在热路径上，每一个这样的同步点都会让 GPU 的流水线打一次嗝。
- OOM 报错给出的调用栈可能并不准确。因为执行是异步的，错误往往要等到后面某个同步点才会真正暴露出来。想定位到真正出错的 op，可以用 `CUDA_LAUNCH_BLOCKING=1` 强制 kernel 同步执行（仅用于调试，正式运行不要开）。

```python
# 正确计时
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(100): y = model(x)
torch.cuda.synchronize(); t1 = time.perf_counter()   # 必须同步！
```

---

## 2. Stream：并发执行的通道

CUDA 的 stream 本质上是一个 kernel 队列：同一个 stream 内的 kernel 严格按照提交顺序执行，而不同 stream 上的 kernel 只要硬件资源足够，就可以并发执行。如果不特别指定，所有操作默认都在 default stream 上进行。

所谓 overlap，本质上就是把可以并行的工作分别放到不同的 stream 上，让它们在 GPU 上同时跑起来。其中最典型的场景就是计算和通信的 overlap：

```python
s_comm = torch.cuda.Stream()

# 在通信 stream 上发起 all-reduce
with torch.cuda.stream(s_comm):
    dist.all_reduce(grad, async_op=True)

# 同时在 default stream 上做别的计算
other = compute_next_layer()

# 同步：让 default stream 等通信 stream 完成（见下节 event）
torch.cuda.current_stream().wait_stream(s_comm)
```

这里有一个跨 stream 使用内存时容易踩的坑：caching allocator 是按 stream 来跟踪每个张量的生命周期的。如果一个张量在 stream A 上分配，却被 stream B 使用，就必须用 `record_stream` 显式告诉 allocator 这件事，否则 allocator 可能在 stream B 还在用这块内存的时候，就把它当作空闲内存回收并复用给别的分配，造成数据损坏：

```python
buf = torch.empty(..., device='cuda')      # 在 default stream 分配
with torch.cuda.stream(s_comm):
    dist.all_gather_into_tensor(buf, shard)
    buf.record_stream(s_comm)               # 告知 allocator buf 也被 s_comm 用了
```

这个坑在手写通信 overlap 的代码里非常隐蔽，往往只是偶发的数据错误，不容易复现。FSDP 和 Megatron 内部对此都做了小心的处理。

---

## 3. Event：跨 stream 同步与精确计时

`torch.cuda.Event` 是插在某个 stream 里的一个标记点，主要用途有两个：让一个 stream 等待另一个 stream 到达某个特定位置，或者精确测量 GPU 上的执行时间。

```python
# 跨 stream 同步：B 等 A 的某个点
ev = torch.cuda.Event()
with torch.cuda.stream(s_a):
    do_work_a()
    ev.record(s_a)              # 在 A 上打标记
with torch.cuda.stream(s_b):
    s_b.wait_event(ev)          # B 等到 A 的标记点才继续
    do_work_b()

# GPU 计时（比 synchronize + perf_counter 更精确，只量 GPU 时间）
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
model(x)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)    # 毫秒
```

`stream.wait_stream(other)` 是一个常用的快捷方式，作用是让当前 stream 等待 other stream 上当前已提交的所有工作完成。stream 和 event 这两个概念，是 CUDA Graph、通信 overlap、双缓冲 prefetch 等更高层机制共同依赖的底层积木。

---

## 4. Caching allocator：显存是怎么管理的

PyTorch 并不会每次都直接调用 `cudaMalloc` / `cudaFree`，那样的开销很大而且会引入同步。它内部维护了一个 caching allocator：第一次申请内存时，会一次性向系统申请一大块（`cudaMalloc`），之后 `free` 掉的内存不会真的还给系统，而是进入缓存池以便复用。由此带来两个直接后果：

- `del tensor` 或者变量出了作用域之后，对应的显存会回到 PyTorch 自己的缓存池，而不是还给操作系统。所以用 `nvidia-smi` 看到占用没有下降，是完全正常的现象。
- 只有调用 `torch.cuda.empty_cache()`，才会把缓存池里的空闲块真正还给系统。这个操作一般只在需要把显存让给别的进程时才用，平时训练不需要调用它，而且它本身会引入一次同步。

排查 OOM 或者显存泄漏时，下面这几个显存观测 API 很常用：

```python
torch.cuda.memory_allocated()       # 当前张量实际占用
torch.cuda.max_memory_allocated()   # 峰值（找峰值在哪一步）
torch.cuda.memory_reserved()        # allocator 向 OS 要的总量（含缓存）
torch.cuda.reset_peak_memory_stats()# 重置峰值统计，分段测量
print(torch.cuda.memory_summary())  # 人类可读的详细分块报告
```

如果想搞清楚显存究竟被谁吃掉了，显存历史快照是最终的排查手段：

```python
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... 跑一段训练 ...
torch.cuda.memory._dump_snapshot("snap.pickle")
# 上传到 https://pytorch.org/memory_viz 可视化每块内存的分配栈
```

还有一个常见现象是碎片化：跑长序列或者变长 batch 时，经常会出现 reserved 远大于 allocated、但依然 OOM 的情况。根本原因在于 `cudaMalloc` 出来的各个段彼此之间无法合并，再加上段内部被存活的 block 占住之后留下的空洞越来越多。碎片化的准确定义、CUDA VMM 怎么把虚拟地址和物理地址拆开，以及 `PYTORCH_ALLOC_CONF=expandable_segments:True` 的完整机制，留到 [06](./06_caching_allocator.md) 详细讲。现在训练大模型，打开 expandable segments 基本已经是标配做法。

---

## 5. Pinned memory 与异步 H2D 拷贝

普通的 CPU 内存（pageable 内存）要拷贝到 GPU 时，必须先经过一次中转，而且这个过程无法真正和 kernel 执行重叠。pinned（page-locked）内存则可以被 GPU 直接 DMA 访问，配合 `non_blocking=True` 就能实现 H2D 拷贝和计算的重叠：

```python
# DataLoader 里开 pin_memory，拿到的 batch 在 pinned 内存
loader = DataLoader(ds, pin_memory=True)

# 异步拷到 GPU：拷贝在后台进行，CPU 不等
x = batch.to('cuda', non_blocking=True)   # 仅对 pinned 源 + 不同 device 有效
# ... 这之间可以做点 CPU 工作 / 上一个 batch 的计算 ...
```

这里有几个要点要记住：

- `non_blocking=True` 只有在源内存是 pinned 时才真正生效；如果源是普通的 pageable 内存，它会静默退化为同步拷贝，不会报错但也不会有加速效果。
- 一个经典的 prefetch 模式是：在一个独立的 copy stream 上用 `non_blocking` 方式提前拷贝下一个 batch，让这个拷贝和当前 batch 的计算重叠起来，从而把数据加载的延迟隐藏掉。
- D2H 拷贝（也就是 `.cpu()`）同样可以异步进行，但要注意，读取拷回 CPU 的结果之前必须先同步。

---

## 6. 混合精度（AMP）：autocast + GradScaler

LLM 训练通常用 bf16 或 fp16 做实际计算，同时用 fp32 保存主权重和 optimizer state。torch 的 AMP 机制会自动决定每个 op 该用哪种精度。

### 6.1 autocast：自动选精度

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    out = model(x)          # matmul/conv 等自动用 bf16，归一化/softmax/loss 留 fp32
    loss = F.cross_entropy(out, y)
# 反向在 autocast 区外，梯度按各 op 前向的 dtype 走
loss.backward()
```

autocast 内部维护着一张 op 白名单：像 matmul、linear、attention 这类计算量大、数值上也比较鲁棒的 op 会被转成低精度；而 reduction、norm、softmax、loss、exp 这类对精度比较敏感的 op 则会保留在 fp32 上计算。正因为如此，不需要手动调用 `.half()` 把整个模型转成低精度——那样会把本不该降精度的部分也一起降了，容易带来数值问题。

### 6.2 GradScaler：只有 fp16 需要

fp16 的动态范围比较小，很小的梯度值容易直接下溢成 0。`GradScaler` 的做法是把 loss 乘上一个较大的 scale 再做反向传播，这样梯度会按同样的比例放大，从而避免下溢；到 optimizer step 之前再把梯度除回去，同时自动跳过出现 inf 或 nan 的那些 step：

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast('cuda', dtype=torch.float16):
    loss = model(x)
scaler.scale(loss).backward()          # 放大 loss 再反向
scaler.unscale_(optimizer)             # （可选）裁剪前要先 unscale
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)                 # 内部 unscale + 检查 inf/nan + step
scaler.update()                        # 动态调整 scale
```

bf16 因为动态范围和 fp32 相同，不会出现下溢问题，所以不需要 `GradScaler`。这也是大模型训练默认选择 bf16 的重要原因之一——整个训练流程可以更简单。

| | fp16 | bf16 |
|---|---|---|
| 动态范围 | 小（易溢出/下溢） | 同 fp32 |
| 需要 GradScaler | **是** | 否 |
| 精度（尾数） | 略高 | 略低 |
| 适用 | 老硬件、推理 | **训练默认** |

### 6.3 FP8

`torch.float8_e4m3fn` / `e5m2` 配合 per-tensor 或 per-row 的 scaling，可以用来做 FP8 的 GEMM 和通信，是 Hopper 架构之后训练和推理提速的重要手段。torch 原生对 FP8 的支持还比较有限，生产环境一般会用 TransformerEngine 或者 torchao 提供的 FP8 recipe。相比 bf16，FP8 需要额外管理 scaling factor 和 amax 历史，工程复杂度更高一些。

---

## 7. cuDNN / cuBLAS 后端开关

下面这几个全局开关会同时影响数值结果和性能：

```python
# TF32：用 tensor core 加速 fp32 matmul（牺牲少量精度），A100+ 默认开
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# cuDNN autotuner：固定 shape 时自动选最快 kernel（变长 shape 反而变慢）
torch.backends.cudnn.benchmark = True

# bf16 GEMM 的累加精度（reduced precision reduction）
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

`cudnn.benchmark=True` 在输入尺寸固定的场景（比如典型的 CNN 训练）下能帮你选到更快的 kernel；但 LLM 场景里序列长度经常变化，每出现一个新的 shape 都要重新做一次 autotune，反而会拖慢速度。通常的权衡做法是配合 bucketing，把实际出现的 shape 限制在有限的几种之内。

---

## 8. 设备管理杂项

```python
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()
torch.cuda.set_device(local_rank)        # 每进程绑卡
with torch.cuda.device(1): ...           # 临时切设备
torch.cuda.get_device_properties(0)      # SM 数、显存、算力
torch.cuda.synchronize()                 # 等当前设备所有 stream 完成
```

这一篇提到的显存"池子"，下一篇会被进一步拆开成 segment、block、pool 三层结构，讲清楚碎片化到底是怎么产生的、为什么有三种情况都合并不了，以及 CUDA VMM 和 `expandable_segments` 的完整工作机制，见 [06 · Caching Allocator 与 CUDA VMM](./06_caching_allocator.md)；再往后是 [07 · CUDA Graph](./07_cuda_graph.md)，讲 CUDA Graph 怎么把地址固定在一个 private mempool 上，以及 [08 · torch.compile / profiler](./08_compile_profiler.md) 里的 compile 和 profiler。
