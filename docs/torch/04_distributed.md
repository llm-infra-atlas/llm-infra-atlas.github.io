# 04 · torch.distributed：通信原语、process group、DeviceMesh

> 多卡训练和推理里，所有跨卡数据流最终都建立在 `torch.distributed` 的 collective 之上。这一篇会把每个通信原语的语义、in-place 性质、可微对偶，以及它们在 LLM 并行里各自对应的用途讲清楚，是 [并行策略总览](../parallel/README.md) 整个目录赖以成立的底层 API 基础。

---

## 1. 初始化与 process group

```python
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",          # GPU 用 nccl；CPU/调试用 gloo
    init_method="env://",    # 从环境变量读 MASTER_ADDR/PORT/RANK/WORLD_SIZE
)
rank = dist.get_rank()                    # 全局 rank
world = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)         # 每进程绑一张卡（关键！）
```

| 概念 | 含义 |
|---|---|
| **backend** | `nccl`（GPU collective，唯一生产选择）/ `gloo`（CPU，Mac 上跑 lab）/ `mpi` |
| **rank** | 进程全局编号 `[0, world)` |
| **world_size** | 总进程数 = 总 GPU 数（通常 1 进程 1 卡） |
| **local_rank** | 本机内编号，用来 `set_device` |
| **ProcessGroup (PG)** | 一组参与 collective 的进程；默认是全局 PG，可建子组 |

启动时一般用 `torchrun --nproc_per_node=8 train.py`，它会自动设好 `RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*` 这些环境变量。

多维并行（TP×PP×DP×……）之所以能实现，靠的是把全局 rank 切分成多个子 process group，每个 collective 只在对应的子组内部通信；`new_group` 就是构造这些子组的接口，可以说是并行维度真正的物理载体。

```python
# 8 卡分成 TP=2 × DP=4：构造 4 个 TP 组
# rank 排布决定哪些卡在同一组（影响 NVLink 局部性）
tp_group = dist.new_group(ranks=[0, 1])   # 在 rank 0,1 上调用才有效
# 实际框架用 parallel_state 统一构造所有维度的 group
```

> 各并行维度的 group 具体怎么排布、为什么 TP 要放在 NVLink 域最内层，见 [Tensor Parallelism（TP）与 Sequence Parallelism（SP）](../parallel/02_tp_sp/README.md) 第 4 节。

---

## 2. collective 原语总览

这是本篇的核心内容。需要先明确一个共同的默认行为：每个 collective 默认都会就地修改或者写入传入的张量，并且默认是同步的，也就是会阻塞直到通信完成。下表中"LLM 用途"一列对应的是 [并行策略总览](../parallel/README.md) 里的各个维度。

| 原语 | 语义 | 输出 | LLM 用途 |
|---|---|---|---|
| `all_reduce(t, op)` | 各 rank 的 `t` 按 op 规约，结果**所有 rank 都拿到** | in-place 改 `t` | TP 求和、DDP 梯度平均 |
| `reduce(t, dst, op)` | 规约结果只给 `dst` rank | in-place（仅 dst 有效） | 较少单独用 |
| `broadcast(t, src)` | 把 `src` 的 `t` 广播给所有 rank | in-place | 同步初始权重/随机种子 |
| `all_gather_into_tensor(out, t)` | 收集各 rank 的 `t` 拼成 `out` | 写 `out` | FSDP unshard、SP all-gather |
| `reduce_scatter_tensor(out, t, op)` | 规约后按 rank 切片分发 | 写 `out` | ZeRO 梯度、SP reduce-scatter |
| `all_to_all_single(out, t)` | 每 rank 把数据切 N 份分发给 N 个 rank | 写 `out` | **MoE dispatch/combine**、Ulysses CP |
| `all_to_all(out_list, in_list)` | tensor list 版（变长） | 写 list | 变长 MoE dispatch |
| `gather(t, gather_list, dst)` | 收集到单个 dst（不规约） | dst 的 list | 收集 metric/日志 |
| `scatter(t, scatter_list, src)` | src 把 list 分发出去 | 写 `t` | 分发数据 |
| `barrier()` | 同步点，所有 rank 到齐才继续 | —— | checkpoint、计时对齐 |

### 2.1 `all_reduce`

```python
# TP: 把各卡的部分和求和（row-parallel linear 出口）
dist.all_reduce(y, op=dist.ReduceOp.SUM, group=tp_group)
# DDP 梯度平均：SUM 后除以 world（NCCL 无原生 AVG 的旧版本）
dist.all_reduce(grad, op=dist.ReduceOp.SUM); grad /= world
```

`ReduceOp` 支持 `SUM` / `AVG`（NCCL 新版支持）/ `MAX` / `MIN` / `PRODUCT`。这里有一个踩过的坑要提醒：gloo 后端不支持 `reduce_scatter` 的 `AVG`，在 Mac 上跑 lab 时得改成先 `SUM` 再手动除（详见 lab 约定）。

### 2.2 `all_gather_into_tensor` / `reduce_scatter_tensor`（推荐的连续版）

新版 API 优先推荐 `*_tensor` 形式，输入输出都是单块连续张量，比老的 list 版本（`all_gather` / `reduce_scatter`）要少很多拷贝：

```python
# all_gather: 每 rank 有 [k, h] 的分片，拼成 [world*k, h]
shard = torch.randn(k, h, device='cuda')
full = torch.empty(world * k, h, device='cuda')
dist.all_gather_into_tensor(full, shard, group=pg)

# reduce_scatter: 输入 [world*k, h]，规约后每 rank 拿 [k, h]
inp = torch.randn(world * k, h, device='cuda')
out = torch.empty(k, h, device='cuda')
dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM, group=pg)
```

这里有一个关键恒等式：`all_reduce = reduce_scatter + all_gather`。两者的通信量是相同的，但把 all_reduce 拆成这两步之后，就可以让其中一部分和计算重叠起来，这正是 SP 和 ZeRO 能够工作的物理基础（见 [DP](../parallel/01_dp/README.md)、[TP/SP](../parallel/02_tp_sp/README.md)）。

### 2.3 `all_to_all`：MoE 的核心原语

每个 rank 把自己手里的数据切成 N 份，第 i 份发给 rank i，同时也收来自所有其他 rank 发来的那一份。这正是 MoE expert parallel 里 dispatch 阶段要做的事情：把 token 按照它被路由到的 expert 所在的 rank 重新分配。

```python
# 等长版：每 rank 发/收等量
out = torch.empty_like(inp)
dist.all_to_all_single(out, inp, group=ep_group)

# 变长版：每 rank 发给/收自不同 rank 的数量不同（MoE 实际场景）
dist.all_to_all_single(
    out, inp,
    output_split_sizes=out_splits,   # 从每个 rank 收多少
    input_split_sizes=in_splits,     # 发给每个 rank 多少
    group=ep_group,
)
```

> MoE 里两次 all-to-all（dispatch 和 combine）、变长 split 的计算方式，以及和 grouped GEMM 之间的衔接，是 [EP](../parallel/05_ep/README.md) 这一篇的主线。DeepEP 这类库做的事情，本质上就是在优化这两次 all-to-all 通信。

---

## 3. 异步通信与 overlap

每个 collective 只要传 `async_op=True`，就会立即返回一个 `Work` 句柄，通信在后台进行，中间可以插入其他计算，最后再调用 `wait()` 等它完成：

```python
handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
# ... 这里做别的计算（如下一层的 wgrad），和通信 overlap ...
handle.wait()                    # 用 grad 前必须等通信完成
grad /= world
```

几个需要理解的要点：

- overlap 的本质是通信 kernel 和计算 kernel 在不同的 CUDA stream 上并发执行。`async_op` 配合 `wait()`，再加上 stream 的调度，才能真正实现 compute 和 comm 的重叠。
- Megatron 靠设置 `CUDA_DEVICE_MAX_CONNECTIONS=1` 来强制通信 kernel 先于计算 kernel 发射，从而确保延迟真的被藏起来了（见 [04 · TP/SP 的通信-计算 overlap 与工程优化](../parallel/02_tp_sp/04_overlap_and_optimizations.md)）。
- 在 `wait()` 返回之前，绝对不能读写那块参与通信的张量，否则会出现数据竞争，读到的是通信过程中的中间状态。

---

## 4. P2P：点对点通信（PP 的传输层）

pipeline parallel 用 P2P 通信在相邻的 stage 之间传递激活和梯度：

```python
# 阻塞版
dist.send(tensor, dst=next_rank)
dist.recv(tensor, src=prev_rank)

# 非阻塞版
req = dist.isend(tensor, dst=next_rank)
req.wait()

# 批量版（PP 1F1B 必须用这个，否则死锁！）
ops = [
    dist.P2POp(dist.isend, send_tensor, next_rank),
    dist.P2POp(dist.irecv, recv_tensor, prev_rank),
]
reqs = dist.batch_isend_irecv(ops)
for r in reqs: r.wait()
```

这里有一个 PP 里经典的死锁陷阱：如果交错的 send/recv 用的是各自独立的阻塞 `send`/`recv`，相邻的两个 stage 就可能同时在等对方先接收，结果就是双方一起卡死。正确的做法是用 `batch_isend_irecv` 把一组 send/recv 一次性提交，这也正是 Megatron 里 `send_forward_recv_backward` 的实现方式。这一条在 [PP](../parallel/03_pp/README.md) 的 lab 里已经验证过。

---

## 5. 可微 collective：通信的 forward/backward 对偶

裸的 `dist.*` 调用是不可微的，因为它们直接就地修改张量，autograd 根本不知道发生了什么。要让通信参与反向传播，就必须把它包成一个 `autograd.Function`（见 [03 第 3.3 节](./03_autograd.md#33-llm)）。各个原语的反向对偶关系如下：

| forward collective | backward collective | 直觉 |
|---|---|---|
| `all_reduce` (SUM) | identity（梯度直接传） | sum 的导数是 1 |
| `all_gather` | `reduce_scatter` | gather 的逆是 scatter，梯度要规约 |
| `reduce_scatter` | `all_gather` | 互为对偶 |
| `all_to_all` | `all_to_all`（split 互换） | 自对偶，方向反过来 |
| `broadcast` (from src) | `reduce` (to src) | 复制的反向是求和回源 |
| identity (copy `f`) | `all_reduce` | Megatron `f` 算子 |

这张表可以说是理解所有并行通信的一把统一钥匙：只要确定了 forward 在哪里通信、通信的是什么，backward 应该在哪里通信、用哪种原语，就由这张表里的对偶关系自动确定了。Megatron 的 `f`/`g`、FSDP 的 unshard/reduce、SP 的 all-gather/reduce-scatter，全都是这张表的具体实例。

```python
class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = torch.empty(world*x.shape[0], *x.shape[1:], device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out
    @staticmethod
    def backward(ctx, g):
        out = torch.empty_like(g.chunk(world)[0])
        dist.reduce_scatter_tensor(out, g.contiguous(), group=ctx.group)  # 对偶
        return out, None
```

---

## 6. DeviceMesh 与 functional collectives

老式的手动 `new_group` 管理多维并行既繁琐又容易出错。2.x 引入了 `DeviceMesh` 来描述多维设备拓扑，配合 DTensor 和 functional collectives，构成了 FSDP2 和 TP API 的底座。

```python
from torch.distributed.device_mesh import init_device_mesh

# 8 卡组织成 2D mesh：dp=2, tp=4
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
tp_group = mesh["tp"].get_group()     # 取某一维的 ProcessGroup
dp_mesh = mesh["dp"]
```

functional collectives（位于 `torch.distributed._functional_collectives`）是函数式的：它们返回一个新张量，而不是就地修改输入，这样就可以被 `torch.compile` 追踪：

```python
import torch.distributed._functional_collectives as fc
y = fc.all_reduce(x, "sum", group)           # 返回新张量，不改 x
y = fc.all_gather_tensor(x, gather_dim=0, group=mesh["tp"])
```

这套设计为什么重要，主要有两个原因：

- 因为是函数式的，它能被 `torch.compile` 捕获，并和其他计算一起融合、调度。传统的 in-place collective 对编译器来说是图里的一个"黑盒副作用"，很难重排；而 functional 版本让编译器有能力自动安排 comm 和 compute 的重叠。
- `DeviceMesh` 是 FSDP2（`fully_shard`）、TP API（`parallelize_module`）、DTensor 共用的统一拓扑描述，新一代框架（比如 torchtitan）已经全面转向使用它。

> FSDP2 基于 DeviceMesh 和 DTensor 的具体实现细节，见 [03 · FSDP（ZeRO-3）：逐层 all-gather 与 reshard](../parallel/01_dp/03_fsdp.md)。

---

## 7. DTensor：带分片语义的张量

`DTensor` 把"这个张量在 mesh 上是怎么分片的"作为元数据直接带在张量上（用 `Shard(dim)`、`Replicate`、`Partial` 这些 placement 来描述），需要哪种 collective 由 placement 自动推导出来。它是 FSDP2 和 TP 高层 API 的内部表示。手写并行代码时通常还是直接用裸的 collective，但读 FSDP2 或者 torchtitan 的源码时会大量遇到 DTensor，认得它是必要的：

```python
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
dt = distribute_tensor(big_tensor, mesh["tp"], [Shard(0)])  # 沿 dim0 切到 tp 维
full = dt.full_tensor()                                      # 触发 all_gather 还原
```

---

## 8. 调试与排错清单

| 症状 | 常见原因 |
|---|---|
| hang / 卡住不动 | 某个 rank 没进同一个 collective（分支不一致）；或 PP send/recv 死锁（用 `batch_isend_irecv`） |
| `NCCL timeout` | 通信不对齐、某 rank 提前退出、watchdog 超时（`TORCH_NCCL_BLOCKING_WAIT=1` 调试） |
| 数值不对 | `wait()` 前就用了异步通信的张量；或 reduce 后忘了除 world |
| 偶发 illegal memory access | 通信张量非 contiguous / 跨 stream 没同步 |
| 各卡结果发散 | 初始权重没 broadcast 对齐；dropout RNG 不一致 |

排查时常用的环境变量有：`NCCL_DEBUG=INFO`（查看 NCCL 拓扑和通道信息）、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`、`CUDA_DEVICE_MAX_CONNECTIONS=1`（做 overlap 时必需）、`NCCL_P2P_DISABLE` / `NCCL_IB_DISABLE`（排查互联问题时使用）。

通信和计算最终都要跑在 CUDA stream 上，下一篇就顺着这条线往下看：stream 和 event 怎么配合做 overlap、allocator 的观测 API 怎么用、autocast 又是怎么实现混合精度的，这些都在 [05 · CUDA 执行模型：stream / event / 显存分配 / AMP](./05_cuda_streams_memory_amp.md) 里；显存分配的底层机制（CUDA VMM、expandable segments）则留到 [06](./06_caching_allocator.md) 单独展开。
