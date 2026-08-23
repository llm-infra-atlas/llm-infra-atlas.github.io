# 01 · Tensor 与内存布局：view / stride / contiguous / dtype

> 内存布局是所有底层算子的基础。写 LLM 框架时，很多看起来莫名其妙的 bug 和性能坑，追根溯源都出在对同几个问题的误判上：这个张量到底是 view 还是 copy、它的内存连不连续、stride 具体是多少。接下来我们把 `Tensor` 拆开，看看它真实的样子。

---

## 1. Tensor 的本质：`(Storage, offset, shape, stride)`

一个 `torch.Tensor` 并不直接持有数据，它只是一段一维 `Storage`（连续内存块）上的**一个带步长的视图**：

```
Tensor = (storage, storage_offset, shape, stride, dtype, device)
```

- **storage**：底层连续的一维内存（`x.untyped_storage()`）。多个 tensor 可以共享同一个 storage。GPU 上这段内存来自 caching allocator 的一个 block，并不是每次都真的去调用一次 `cudaMalloc`（见 [06](./06_caching_allocator.md)）。
- **storage_offset**：在 storage 里从第几个元素开始（`x.storage_offset()`）。
- **shape**：逻辑形状（`x.shape`）。
- **stride**：每个维度往前走一步，需要在 storage 里跳过几个元素（`x.stride()`）。理解 stride，是理解一切 view 操作的钥匙。

逻辑下标 $(i, j, k)$ 对应到 storage 里的物理位置，换算方式是：

$$
\text{offset} + i \cdot \text{stride}[0] + j \cdot \text{stride}[1] + k \cdot \text{stride}[2]
$$

```python
x = torch.arange(12).reshape(3, 4)
x.shape            # torch.Size([3, 4])
x.stride()         # (4, 1)   —— 行优先：走一行跳 4，走一列跳 1
x.storage_offset() # 0
x.data_ptr()       # storage 首地址（+offset 后的元素地址）
```

之所以要强调这一点，是因为 transpose 只是交换 stride、并不移动数据，切片只是改 offset 和 shape，广播则是把某个维度的 stride 设成 0。把这几件事想清楚，你就能预判哪些操作是零拷贝的，哪些 kernel 可以直接吃到这段内存而不用先做转换。

```python
xt = x.t()          # transpose
xt.stride()         # (1, 4)  —— 只是把 stride 反过来，storage 没变
xt.data_ptr() == x.data_ptr()   # True，共享内存
```

---

## 2. view vs copy：哪些操作不拷贝

下面这张表大概是本篇最关键的一部分。表里的 view 类操作返回的张量和原张量共享同一个 storage，改动其中一个会直接影响另一个。

| 操作 | 是否拷贝 | stride 变化 | 说明 |
|---|---|---|---|
| `view(shape)` | **否** | 重新计算 | 要求原张量 contiguous（或兼容），否则报错 |
| `reshape(shape)` | **可能** | —— | 能 view 就 view，不能就 copy。更安全但行为不透明 |
| `transpose(a,b)` / `t()` | 否 | 交换两维 | 结果**非 contiguous** |
| `permute(dims)` | 否 | 重排 | 结果通常非 contiguous |
| `movedim` / `swapaxes` | 否 | 重排 | 同上 |
| `squeeze` / `unsqueeze` | 否 | 增删一个大小为 1 的维 | `x[None]` / `x[:, None]` 等价 |
| `expand(shape)` | 否 | 被扩维 stride=0 | **零拷贝广播**，多个逻辑元素指向同一物理位置，只读用 |
| `broadcast_to` | 否 | 同 expand | expand 的别名式 API |
| `narrow` / 基础切片 `x[2:5]` | 否 | 改 offset + shape | 连续切片是 view |
| `x[mask]`（bool 索引） | **是** | —— | advanced indexing 一律拷贝 |
| `x[idx]`（int tensor 索引） | **是** | —— | 同上 |
| `flatten` | 可能 | —— | 连续则 view，否则 copy |
| `contiguous()` | 可能 | 变 row-major | 已连续则返回自身（no-op），否则拷贝 |
| `clone()` | **是** | 保留 stride（默认） | 显式深拷贝，仍在计算图里（可微） |
| `detach()` | 否 | —— | 共享数据，但脱离计算图（见 03） |
| `to(dtype)` / `float()` | 视情况 | —— | dtype 变了必拷贝；dtype/device 都不变则返回自身 |

结合这张表，写框架代码时有几条基本原则值得记住：

- 拿到上游传来的张量，准备写进去（in-place）或者喂给只认连续内存的 kernel 之前，先想清楚它到底是不是 view、连不连续。
- `expand` 出来的张量不要做 in-place 写，因为多个逻辑位置共享着同一块内存，写一个就会污染一整片。真要写，先 `clone()` 一份。
- 当一个函数返回的是 view 时，调用方对返回值做的 in-place 修改会"穿透"回原张量，这是隐蔽 bug 一个常见的来源。

```python
# 经典坑：transpose 后 view 报错
x = torch.randn(2, 3, 4)
x.transpose(1, 2).view(2, -1)        # RuntimeError: view size is not compatible...
x.transpose(1, 2).reshape(2, -1)     # OK，reshape 会在必要时拷贝
x.transpose(1, 2).contiguous().view(2, -1)  # OK，先连续化再 view
```

---

## 3. contiguous：连续性的定义与判断

一个张量是 contiguous（row-major / C-order）的，当且仅当它的 stride 恰好是"行优先紧密排列"的：从最后一维起，$\text{stride}[-1] = 1$，$\text{stride}[i] = \text{stride}[i+1] \cdot \text{shape}[i+1]$。

```python
x = torch.randn(3, 4)
x.is_contiguous()              # True
x.t().is_contiguous()          # False —— transpose 破坏了连续性
x.t().contiguous().is_contiguous()  # True，付出一次拷贝
```

kernel 为什么会在意这件事？因为很多手写的 CUDA kernel、第三方算子（包括部分 cuBLAS 路径）都假设输入是连续的，靠 `data_ptr + 线性偏移` 去访存。一旦喂进去一个非连续的张量，要么直接报错，要么更麻烦——静默地读错数据。`view` 之所以要求输入连续，本质上也是因为它需要保证"逻辑上的 reshape 等于物理上仍是同一段紧密内存"。

性能上也有取舍要考虑：`contiguous()` 在张量已经连续时是 no-op，不产生任何开销；但在非连续时会触发一次 memcpy。所以不要无脑地到处加 `contiguous()`，那样反而会引入不必要的拷贝，正确的做法是只在即将进入要求连续内存的边界时才加这一步。

```python
# attention 里的典型模式：reshape head 维度
q = q.view(b, s, n_head, head_dim).transpose(1, 2)  # [b, n_head, s, head_dim]，非连续
# SDPA 能吃非连续，但若喂给自定义 kernel，往往要：
q = q.contiguous()
```

---

## 4. `as_strided` / `unfold`：零拷贝构造重叠 view

`as_strided(size, stride, offset)` 允许直接手动指定 `(shape, stride, offset)`，是所有 view 操作的底层原语。用它可以构造出"重叠窗口"，也就是让多个逻辑元素映射到重叠的物理区间，从而零拷贝地实现 sliding window：

```python
x = torch.arange(8.)
# 造 5 个长度为 4、步长为 1 的滑动窗口，零拷贝
w = x.as_strided(size=(5, 4), stride=(1, 1))
# w[i] = x[i:i+4]
```

`unfold(dim, size, step)` 是一个更安全的封装，常用于卷积式取块或者序列分块：

```python
x = torch.arange(10)
x.unfold(0, size=4, step=2)   # shape [4, 4]，重叠窗口
```

不过要留意其中的风险：`as_strided` 不会检查越界，可能读到 storage 之外的内存；在重叠 view 上做 in-place 写或者跑 autograd 也容易出错。框架里偶尔会用它来做 KV cache 的环形缓冲视图、attention mask 的对角带等结构，但用之前务必清楚自己在做什么。

---

## 5. LLM 中常用的 dtype

| dtype | 位宽 | 范围 / 精度 | LLM 用途 |
|---|---|---|---|
| `float32` | 32 | 标准 | master weight、optimizer state、loss/归一化的累加 |
| `bfloat16` | 16 | 指数同 fp32，尾数少 | **训练默认计算精度**，不需要 loss scaling |
| `float16` | 16 | 范围小，精度略高于 bf16 | 老硬件 / 推理，需 `GradScaler` 防下溢 |
| `float8_e4m3fn` / `e5m2` | 8 | FP8 | H100+ 上的 GEMM / 通信，配 scaling |
| `int8` / `uint8` | 8 | 整数 | 权重量化（GPTQ/AWQ 反量化前的存储） |
| `int32` / `int64` | 32/64 | 整数 | index（`gather/scatter` 的 index 必须是 int64）、token id |
| `bool` | 1（实际 1 字节） | —— | attention mask、padding mask |

几个需要记住的要点：

- 混合精度训练的核心思路是：参数的主副本（master）用 fp32 保存，前反向计算用 bf16 或 fp16，optimizer 的更新则发生在 fp32 上。细节见 [05 的 AMP 部分](./05_cuda_streams_memory_amp.md)。
- bf16 和 fp16 该怎么选：bf16 的指数位和 fp32 相同，动态范围大，训练时几乎不需要 loss scaling；fp16 的范围小，梯度容易下溢，因此需要配合 `GradScaler`。大模型训练默认用 bf16。
- index 的 dtype 有硬性要求：`gather/scatter/index_select` 的索引张量必须是 `int64`（也就是 `long`），传 int32 会直接报错。
- 想查某个 dtype 的具体性质，用 `torch.finfo(torch.bfloat16)`（浮点类型）或 `torch.iinfo(torch.int64)`（整数类型），它们会给出 `.max/.min/.eps` 这些字段。

```python
torch.finfo(torch.bfloat16).max    # 3.39e38（和 fp32 同量级）
torch.finfo(torch.float16).max     # 65504（容易溢出！）
```

还有一点容易被忽略：类型提升（type promotion）会在混合 dtype 运算时自动把结果提升到"更宽"的类型。这在写数值稳定的归一化逻辑时要格外小心，比如想让累加发生在 fp32 里，就必须显式调用 `.float()`，不能指望类型提升帮你做对。

---

## 6. device 与张量创建

```python
x = torch.empty(4, 8, device='cuda', dtype=torch.bfloat16)   # 不初始化，最快
y = torch.zeros_like(x)        # 继承 shape/dtype/device，最省心
z = torch.empty_like(x)        # 同上但不初始化
w = torch.randn(4, 8, device='cuda')
```

| 创建方式 | 说明 | 框架用途 |
|---|---|---|
| `empty` / `empty_like` | 不初始化内存，**最快** | 预分配 buffer、马上要被写满的输出张量 |
| `zeros` / `ones` / `full` | 填固定值 | 梯度累加 buffer、mask |
| `*_like(x)` | 继承 `x` 的 shape/dtype/device/layout | 写算子时构造输出张量的**首选** |
| `torch.arange` / `linspace` | 序列 | position id、RoPE 频率 |
| `from_numpy` / `as_tensor` | 共享 numpy 内存（零拷贝） | 数据加载 |
| `torch.frombuffer` | 从 bytes 零拷贝建张量 | 反序列化、mmap 权重 |

性能方面还有几个要点：

- 写自定义算子时，用 `torch.empty_like(input)` 来构造输出张量，避免不必要的 `zeros` 调用——零初始化本身也是一次 kernel，白白多花一次开销。
- `tensor.new_empty` / `new_zeros` 会自动继承原张量的 device 和 dtype，同时允许你指定新的 shape，比手写 `device=x.device, dtype=x.dtype` 简洁很多。
- `torch.empty(..., device='meta')` 会创建所谓的 **meta device** 张量：它只有形状信息，没有真实内存。这在 FSDP 或者大模型"先建图算出各处 shape、再实例化"的延迟初始化场景中很常用，做法是先在 `torch.device('meta')` 上初始化模型，再用 `to_empty` 把它搬到真实设备上。

```python
# 大模型常见：在 meta device 上构造模型（不占显存），再分片实例化
with torch.device('meta'):
    model = build_huge_model()      # 0 显存，只有 shape/dtype
# 之后 FSDP / 手动 shard 时才真正分配
```

---

## 7. memory_format：channels_last

`contiguous(memory_format=torch.channels_last)` 会把一个 NCHW 张量在物理内存上按 NHWC 排布。这主要是为 CNN 服务的，纯 transformer 结构的 LLM 基本用不到。之所以在这里提一句，是想说明 `contiguous` 其实带一个 `memory_format` 参数，不要误以为只存在一种"连续"。写 LLM 相关代码时，你只需要关心默认的 `torch.contiguous_format`（也就是 row-major）即可。

---

## 8. in-place 操作

带下划线后缀的操作是 in-place 的（比如 `add_`、`mul_`、`relu_`、`copy_`、`zero_`），它们直接修改原张量，不会分配新内存：

```python
x.add_(y)          # x += y，原地
x.mul_(0.5)
buf.copy_(src)     # 把 src 内容拷进 buf（设备/dtype 可不同，会转换）
grad.zero_()       # 清零，复用 buffer
```

这类操作的好处很直接：省显存、省一次分配，框架里做梯度累加、buffer 复用时都大量依赖它们。但代价也要认清楚：

1. 它会破坏 autograd 的假设。如果一个张量在 forward 里被 in-place 修改了，而它的原始值恰好是反向所需要的，就会报出 `a leaf Variable that requires grad is being used in an in-place operation`，或者出现版本计数错误——autograd 是靠 version counter 来检测这类情况的。
2. 不能在叶子张量（也就是 `requires_grad=True` 的参数）上随意做 in-place 写，除非是在 `no_grad` 环境下（比如 optimizer 的 step）。
3. `copy_` 是一个高频原语，用来把数据从 src 拷贝到 dst，可以跨 device（H2D/D2H）、跨 dtype，配合 `non_blocking=True` 还能做异步拷贝（见 05）。

```python
# optimizer step 的典型 in-place（在 no_grad 下，复用 param 内存）
with torch.no_grad():
    param.add_(grad, alpha=-lr)     # param -= lr * grad，原地更新
```

---

## 9. 小结：内存自检清单

拿到一个张量、准备写底层逻辑之前，不妨把下面几个问题过一遍：

1. 它是 view 还是独立内存？我对它做 in-place 写会不会穿透到别处？（可以用 `x._base is not None` 判断它是不是 view）
2. 它连续吗？我要喂的 kernel 要求连续吗？如果要求连续，就在边界处调用 `contiguous()`。
3. dtype 对吗？index 是 int64 吗？累加的地方要不要转成 `.float()`？
4. device 对吗？跨设备运算会直接报错，构造输出张量时用 `*_like` 可以自动对齐。
5. 我要构造的输出张量该用 `empty_like` 还是 `zeros_like`？如果马上就会被写满，用 empty 更划算。

讲完了内存和 tensor 的底层结构，下一个自然的问题是：在这套内存模型之上，LLM 里真正高频使用的计算 op 长什么样？这就是 [02 · 计算 op：matmul / einsum / reduction / gather-scatter / SDPA](./02_compute_ops.md) 要讲的内容，包括 matmul 家族、einsum、gather/scatter 以及 SDPA。
