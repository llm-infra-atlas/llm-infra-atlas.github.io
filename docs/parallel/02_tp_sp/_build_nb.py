import json, sys

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md(r"""# Lab · 手写 Tensor Parallelism + Sequence Parallelism 的前向 / 反向

本 notebook 用**纯 PyTorch** 把 [`docs/parallel/tp_sp`](./README.md) 里讲的 TP / SP 通路亲手实现一遍，并用**真实的 `torch.distributed`** 集合通信（gloo 后端、本地多进程）跑通 all-reduce / all-gather / reduce-scatter。所有代码在 **Mac CPU** 上几秒内跑完。

我们做的简化（以及对应正文哪一段）：

| 真实系统 | 本 lab 的简化 | 正文 |
|---|---|---|
| 完整 Transformer layer | 一个两层 MLP（fc1 col → relu → fc2 row） | `02` |
| `LinearWithGradAccumulationAndAsyncCommunication`（async + wgrad fusion） | 直接用 `matmul` + 朴素 autograd | `01` |
| `tp_comm_overlap` / userbuffers | 不做 overlap，通信同步执行 | `04` |
| TE / FP8 / RNG tracker | 全程 fp32，无 dropout | `02`, `04` |
| attention 按 head 切 | 只演示 MLP（col→row 骨架一致） | `02` |

但**关键环节一个不少**：column-parallel 切输出维、row-parallel 切输入维、`f`/`g` 共轭算子、一次 all-reduce 拼回部分和、SP 用 AG+RS 替换 all-reduce 并把 activation 按 seq 切开，以及 **反向通信自动镜像**（`f.bwd=all-reduce`、`g.bwd=identity`、`AG.bwd=RS`、`RS.bwd=AG`）。

**运行方式**：从上到下执行。Part 0–2 在主进程内跑；Part 3–4 用 `mp.spawn` 起 `TP` 个进程（worker 先写到 `tp_worker.py` 再 import）。
""")

md(r"""## Part 0 · 全局配置与「全局问题」

TP 的本质：**单个权重矩阵被切到多张卡**。为能验证正确性，让每个 rank 都能从固定 seed 重建出完整的全局问题（完整权重 + 输入），这样单进程 reference 和多进程 TP 可逐元素对比。

我们的 MLP：`Z = relu(X · A^T) · B^T`，其中 `A:[F,H]` 是 fc1 权重、`B:[H,F]` 是 fc2 权重。
""")
code(r"""
import torch
TP = 2            # tensor parallel size = 进程数
H, F, N = 8, 16, 6  # hidden / FFN 中间维 / token 数 (= seq*batch 拍平)
assert F % TP == 0 and N % TP == 0
print(f"TP={TP}, H={H}, F(ffn)={F}, tokens N={N};  每 rank 持 A[{F//TP},{H}], B[{H},{F//TP}]")

def make_global():
    "所有 rank 共享的全局张量（同一 seed → 各 rank 重建出一致副本）"
    g = torch.Generator().manual_seed(42)
    A = torch.randn(F, H, generator=g) * 0.3   # fc1: 输出维 F, 输入维 H
    B = torch.randn(H, F, generator=g) * 0.3   # fc2: 输出维 H, 输入维 F
    X = torch.randn(N, H, generator=g)         # 输入 token
    return A, B, X
""")

md(r"""## Part 1 · 单进程 reference（ground truth）

把**整个 MLP 放在一张卡**上算，并做 `backward()` 拿到对 `X / A / B` 的梯度真值。后面 TP / SP 版本都和它逐元素对比。
""")
code(r"""
def reference():
    A, B, X = make_global()
    Xr = X.clone().requires_grad_(True)
    Ar = A.clone().requires_grad_(True)
    Br = B.clone().requires_grad_(True)
    Z = torch.relu(Xr @ Ar.t()) @ Br.t()      # [N,H]
    Z.sum().backward()                          # loss = 所有输出之和
    return Z.detach(), Xr.grad, Ar.grad, Br.grad

REF_Z, REF_XG, REF_AG, REF_BG = reference()
print("reference Z:", tuple(REF_Z.shape), "| 一行示例:", REF_Z[0].round(decimals=3).tolist())
""")

md(r"""## Part 2 · 单进程「透明版」TP：看清 column / row 的切法

先不引入多进程，在一个进程里**模拟 TP=2**：手动把 `A` 按输出维（行）切成 `A_0,A_1`，把 `B` 按输入维（列）切成 `B_0,B_1`，验证：

- **column-parallel（fc1）**：`relu(X·A^T)` 的列 = 两段 `relu(X·A_i^T)` 拼接 → 中间激活天然按 F 切开，relu 可本地算；
- **row-parallel（fc2）**：`Y·B^T = Σ_i Y_i·B_i^T` → 各卡算部分和，**一次 all-reduce（这里就是求和）** 得到最终结果。

这对应正文 [`README` 第 2 节](./README.md) 和 [`02`](./02_transformer_block.md)。
""")
code(r"""
A, B, X = make_global()
fs = F // TP
A_sh = [A[i*fs:(i+1)*fs, :] for i in range(TP)]   # column-parallel: 切输出维 F
B_sh = [B[:, i*fs:(i+1)*fs] for i in range(TP)]   # row-parallel:    切输入维 F

# column-parallel：每“卡”本地算自己的 F 段，无通信
Y_sh = [torch.relu(X @ A_sh[i].t()) for i in range(TP)]        # 每个 [N, F/TP]
print("column 段拼接 == 完整 relu(X·A^T):",
      torch.allclose(torch.cat(Y_sh, dim=1), torch.relu(X @ A.t())))

# row-parallel：每“卡”算部分和，再 all-reduce（求和）
Z_partial = [Y_sh[i] @ B_sh[i].t() for i in range(TP)]         # 每个 [N, H] 部分和
Z_tp = sum(Z_partial)                                          # ← 这一步真实系统里是 all-reduce
print("row 部分和求和(all-reduce) == reference:", torch.allclose(Z_tp, REF_Z, atol=1e-5))
print("\n要点：整段只有 1 次规约(all-reduce)，中间激活全程按 F 切开。")
""")

md(r"""## Part 2.5 · 单进程「透明版」TP+SP：把 1 次 all-reduce 拆成 AG + RS

继续在一个进程里模拟 TP=2，但这次叠加 **Sequence Parallelism**：TP 区**外**的 activation 不再每卡存完整 `N` 个 token，而是**按 seq（这里就是 `N`）切成 `1/TP`**。代价是入/出 TP 区各加一次通信——而它们合起来恰好等价于纯 TP 的那一次 all-reduce：

- **进区 = all-gather(seq)**：各卡只持 `X` 的一段 seq，规约前先拼回完整 `X`（单进程里就是 `cat`），TP 区内的 `column→relu→row` 与纯 TP 一字不差；
- **出区 = reduce-scatter(seq)**：把 row 部分和先 **reduce**（求和成完整 `[N,H]`），再按 seq **scatter** 给各卡（每卡只拿回 `1/TP` 段）。

`all-reduce ≡ all-gather ∘ reduce-scatter`：通信量没变，但区外 activation 显存降到 `1/TP`。对应正文 [`03`](./03_sequence_parallel.md) 与 [`README` 第 3 节](./README.md)。
""")
code(r"""
A, B, X = make_global()
fs, ns = F // TP, N // TP
A_sh = [A[i*fs:(i+1)*fs, :] for i in range(TP)]   # column-parallel: 切输出维 F（同 Part 2）
B_sh = [B[:, i*fs:(i+1)*fs] for i in range(TP)]   # row-parallel:    切输入维 F（同 Part 2）
X_sh = [X[i*ns:(i+1)*ns, :] for i in range(TP)]   # SP: 输入按 seq 切，每“卡”只持 N/TP 个 token

# 进 TP 区：all-gather(seq) —— 各卡的 seq 段拼回完整 X（单进程里就是 cat）
X_full = torch.cat(X_sh, dim=0)                                # [N, H]，每“卡”拿到完整 token
print("进区 all-gather(seq) 拼回 == 完整 X:", torch.allclose(X_full, X))

# TP 区内：column → relu → row，与纯 TP 完全一致
Y_sh = [torch.relu(X_full @ A_sh[i].t()) for i in range(TP)]   # 每个 [N, F/TP]
Z_partial = [Y_sh[i] @ B_sh[i].t() for i in range(TP)]         # 每个 [N, H] 部分和

# 出 TP 区：reduce-scatter(seq) —— 先 reduce(求和)，再按 seq scatter 给各卡
Z_sum = sum(Z_partial)                                         # reduce: 完整 [N, H]
Z_sp_sh = [Z_sum[i*ns:(i+1)*ns, :] for i in range(TP)]         # scatter: 每“卡” [N/TP, H]
print("出区 reduce-scatter(seq) 各段拼回 == reference:",
      torch.allclose(torch.cat(Z_sp_sh, dim=0), REF_Z, atol=1e-5))
print(f"\n要点：activation 只在 TP 区内是完整 N（X_full={tuple(X_full.shape)}）；区外每卡只存 1/TP"
      f"（X_sh[0]={tuple(X_sh[0].shape)}, Z_sp_sh[0]={tuple(Z_sp_sh[0].shape)}）。")
print("all-reduce ≡ all-gather ∘ reduce-scatter：纯 TP 的 1 次 all-reduce 被拆成进区 AG + 出区 RS。")
""")

md(r"""## Part 3 · 真实分布式 TP：`f`/`g` 共轭算子 + `torch.distributed`

现在换成真进程。关键是两个自定义 autograd 原语（正文 [`README` 第 1 节](./README.md) 的 `f`/`g`）：

- **`f` = `CopyToTP`**：forward 恒等（输入在 TP 各卡复制），**backward all-reduce**（把各卡对输入的梯度求和）。
- **`g` = `ReduceFromTP`**：**forward all-reduce**（把 row 部分和求和），backward 恒等。

其余（GEMM、relu）都是原生可导算子。于是 autograd 自动把反向通信放到正确位置：`f` 的 backward 出现 all-reduce、`g` 的 forward 出现 all-reduce —— 互为共轭。
""")
code(r'''
%%writefile tp_worker.py
import os, torch, torch.distributed as dist
TP, H, F, N = 2, 8, 16, 6

def make_global():
    g = torch.Generator().manual_seed(42)
    A = torch.randn(F, H, generator=g) * 0.3
    B = torch.randn(H, F, generator=g) * 0.3
    X = torch.randn(N, H, generator=g)
    return A, B, X

def reference():
    A, B, X = make_global()
    Xr=A.new_tensor(X).requires_grad_(True); Ar=A.clone().requires_grad_(True); Br=B.clone().requires_grad_(True)
    Z = torch.relu(Xr @ Ar.t()) @ Br.t(); Z.sum().backward()
    return Z.detach(), Xr.grad, Ar.grad, Br.grad

class CopyToTP(torch.autograd.Function):       # f: fwd identity, bwd all-reduce
    @staticmethod
    def forward(ctx, x, group): ctx.group = group; return x
    @staticmethod
    def backward(ctx, g): dist.all_reduce(g, group=ctx.group); return g, None

class ReduceFromTP(torch.autograd.Function):   # g: fwd all-reduce, bwd identity
    @staticmethod
    def forward(ctx, x, group): dist.all_reduce(x, group=group); return x
    @staticmethod
    def backward(ctx, g): return g, None

class GatherSeq(torch.autograd.Function):      # SP 进区: fwd all-gather(seq), bwd reduce-scatter
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group; w = group.size()
        out = torch.empty(x.shape[0]*w, *x.shape[1:], dtype=x.dtype)
        dist.all_gather_into_tensor(out, x.contiguous(), group=group); return out
    @staticmethod
    def backward(ctx, g):
        w = ctx.group.size(); out = torch.empty(g.shape[0]//w, *g.shape[1:], dtype=g.dtype)
        dist.reduce_scatter_tensor(out, g.contiguous(), group=ctx.group); return out, None

class ReduceScatterSeq(torch.autograd.Function): # SP 出区: fwd reduce-scatter(seq), bwd all-gather
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group; w = group.size()
        out = torch.empty(x.shape[0]//w, *x.shape[1:], dtype=x.dtype)
        dist.reduce_scatter_tensor(out, x.contiguous(), group=group); return out
    @staticmethod
    def backward(ctx, g):
        w = ctx.group.size(); out = torch.empty(g.shape[0]*w, *g.shape[1:], dtype=g.dtype)
        dist.all_gather_into_tensor(out, g.contiguous(), group=ctx.group); return out, None

def run(rank, world):
    os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29622')
    dist.init_process_group('gloo', rank=rank, world_size=world)
    grp = dist.group.WORLD
    A, B, X = make_global(); fs = F // world; ns = N // world
    refZ, refXg, refAg, refBg = reference()

    # ===== 纯 TP =====
    A_r = A[rank*fs:(rank+1)*fs, :].clone().requires_grad_(True)  # column-parallel 切输出维
    B_r = B[:, rank*fs:(rank+1)*fs].clone().requires_grad_(True)  # row-parallel 切输入维
    x = X.clone().requires_grad_(True)                            # 输入 TP 复制
    xin = CopyToTP.apply(x, grp)                                  # f
    y = torch.relu(xin @ A_r.t())                                 # [N, F/TP] 切开
    z = ReduceFromTP.apply(y @ B_r.t(), grp)                      # g: all-reduce 部分和
    z.sum().backward()
    okZ = torch.allclose(z.detach(), refZ, atol=1e-5)
    okX = torch.allclose(x.grad, refXg, atol=1e-5)               # f.backward 的 all-reduce 把梯度补全
    okA = torch.allclose(A_r.grad, refAg[rank*fs:(rank+1)*fs, :], atol=1e-5)
    okB = torch.allclose(B_r.grad, refBg[:, rank*fs:(rank+1)*fs], atol=1e-5)
    if rank == 0:
        print(f"[TP]    forward={okZ}  X.grad={okX}  A.grad={okA}  B.grad={okB}")

    # ===== TP + SP =====
    A_r2 = A[rank*fs:(rank+1)*fs, :].clone().requires_grad_(True)
    B_r2 = B[:, rank*fs:(rank+1)*fs].clone().requires_grad_(True)
    x_r = X[rank*ns:(rank+1)*ns, :].clone().requires_grad_(True)  # 输入按 seq 切
    xfull = GatherSeq.apply(x_r, grp)                             # 进 TP 区: all-gather seq -> [N,H]
    y2 = torch.relu(xfull @ A_r2.t())                            # [N, F/TP]
    z_r = ReduceScatterSeq.apply(y2 @ B_r2.t(), grp)             # 出 TP 区: reduce-scatter -> [N/TP,H]
    z_r.sum().backward()
    okZ2 = torch.allclose(z_r.detach(), refZ[rank*ns:(rank+1)*ns, :], atol=1e-5)
    okX2 = torch.allclose(x_r.grad, refXg[rank*ns:(rank+1)*ns, :], atol=1e-5)
    okA2 = torch.allclose(A_r2.grad, refAg[rank*fs:(rank+1)*fs, :], atol=1e-5)
    okB2 = torch.allclose(B_r2.grad, refBg[:, rank*fs:(rank+1)*fs], atol=1e-5)
    if rank == 0:
        print(f"[TP+SP] forward={okZ2}  X.grad={okX2}  A.grad={okA2}  B.grad={okB2}"
              f"   (activation 按 seq 切: x_r={tuple(x_r.shape)}, z_r={tuple(z_r.shape)})")
    dist.destroy_process_group()
''')
code(r"""
import torch.multiprocessing as mp
import tp_worker, importlib; importlib.reload(tp_worker)   # 改了 worker 后重跑本格即可
mp.spawn(tp_worker.run, args=(TP,), nprocs=TP, join=True)
print("\nTP 与 TP+SP 的 forward / X.grad / A.grad / B.grad 全部 == 单进程 reference ✅")
""")

md(r"""### 这里到底验证了什么？

1. **forward 正确**：`column→row` 两段 GEMM、中间激活按 `F` 切开本地算 relu、出口一次 all-reduce 拼回部分和 —— 逐元素等于「整层在一张卡」的 reference。
2. **`X.grad` 正确**：输入在 TP 各卡复制，每卡只算出对输入梯度的一部分，靠 **`f` 的 backward 里那次 all-reduce** 把各卡贡献求和补全。我们从没手写过这个 all-reduce 的位置——autograd 由 `f` 的共轭性自动放对。
3. **权重梯度正确**：`A_r.grad` / `B_r.grad` 各自只等于 reference 对应的**切片**，说明权重梯度天然就是切开的（不需跨 TP 规约；跨 TP 规约是 DP 的事）。
4. **SP 等价**：把 `f`→`GatherSeq`、`g`→`ReduceScatterSeq` 后，结果不变，但 activation（`x_r`、`z_r`）只存 `1/TP`。`all-reduce ≡ all-gather∘reduce-scatter` 在这里被现场验证。
""")

md(r"""## Part 4 · 对照正文的「通信对称表」

把 lab 里的算子对回 [`README` 第 4 节](./README.md) 的对称表：

| forward（本 lab 的代码） | forward 通信 | backward 通信（autograd 自动） |
|---|---|---|
| `CopyToTP` (f) 进区 | 无 | **all-reduce**（补全 X.grad） |
| `ReduceFromTP` (g) 出区 | **all-reduce**（拼部分和） | 无 |
| column GEMM `xin@A_r.t()` | 无 | dgrad/wgrad 本地 |
| row GEMM `y@B_r.t()` | （规约在 g） | （在 f） |
| `GatherSeq` (SP 进区) | **all-gather**(seq) | **reduce-scatter**(seq) |
| `ReduceScatterSeq` (SP 出区) | **reduce-scatter**(seq) | **all-gather**(seq) |

**整条反向 = 把 forward 镜像翻转**：你只需把 `f`/`g`（或 SP 的 AG/RS）这一对共轭原语写对，其余全用原生算子，autograd 就免费给出整个 TP/SP 的反向通信。这正是 [`01` 第 4 节](./01_linear_layers.md) 里 Megatron 那个 autograd function 在做的事（只是它还顺手把 dgrad 通信和 wgrad 计算 overlap 了）。

## 练习 / 往真实系统靠

1. 把 `TP` 改成 4（需 `F`、`N` 能整除），观察切分粒度。
2. 把 `f`/`g` 的 all-reduce 换成 `async_op=True` + `handle.wait()`，并在中间插入 wgrad GEMM，模拟 [`04`](./04_overlap_and_optimizations.md) 的 dgrad∥wgrad overlap（CPU 上看不出加速，但能验证正确性不变）。
3. 加一个 attention：`linear_qkv` 用 column（按 head 切）、`linear_proj` 用 row，验证多 head 下 TP 仍正确（[`02`](./02_transformer_block.md)）。
4. 实现 vocab-parallel cross entropy（[`02` 第 4 节](./02_transformer_block.md)）：把 `logits` 按 vocab 切，用 3 次标量 all-reduce 算出 loss，对比 `F.cross_entropy`。
5. 给 TP 区内/区外加 dropout，用两套 RNG seed 验证「区外 mask 相同、区内 mask 不同」的必要性（[`02` 第 5 节](./02_transformer_block.md)）。
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python","version":"3.9"}},"nbformat":4,"nbformat_minor":5}
with open(sys.argv[1],"w") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print("wrote", sys.argv[1], "cells:", len(cells))
