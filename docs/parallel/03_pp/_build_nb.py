import json, sys

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md(r"""# Lab · 手写 Pipeline Parallelism：GPipe 与 1F1B

本 notebook 用**纯 PyTorch** 把 [`docs/parallel/pp`](./README.md) 的两种核心调度亲手实现一遍，用**真·跨 stage P2P**（gloo `batch_isend_irecv`）把一个多层 MLP 按层切成 stage 流水跑通**前向 + 反向**，逐元素对齐单进程 reference，并打印 pipeline 时序、量化 bubble。Mac CPU 几秒跑完。

实现对应正文：

| 实现 | 调度 | 显存 | 正文 |
|---|---|---|---|
| **GPipe** | 全 forward → 全 backward | O(m) | [`01`](./01_gpipe_1f1b.md) |
| **1F1B** | warmup → forward/backward 交替 → cooldown | **O(p)** | [`01`](./01_gpipe_1f1b.md) |

简化：每个 stage = `relu(x·Wᵀ+b)`，fp32，无 TP/recompute/overlap。但**关键点一个不少**：micro-batch 流水、跨 stage 用 `forward_step`/`backward_step` 手动搭反向、**combined `send_forward_recv_backward`（避免交错 send/recv 死锁）**、1F1B 的 warmup 数 = `p-rank-1`、以及「相同 bubble、不同显存」。

**运行方式**：从上到下执行；worker 写入 `pp_worker.py` 后由 `mp.spawn` 起 `P`(=stage 数) 个进程。
""")

md(r"""## Part 0 · 全局配置与「全局问题」

PP 把**层**切到多卡。每个 rank（= 一个 stage）持有一层 `relu(x·Wᵀ+b)`，从固定 seed 重建全局权重，便于和单进程 reference 对比。数据只在 first stage、target 只在 last stage（真实 PP 就是这样）。
""")
code(r"""
import torch
P, D, M, MB = 3, 4, 4, 2     # stage 数(=world) / 隐藏维 / micro-batch 数 / 每 micro-batch 样本数
print(f"PP stages p={P}, micro-batches m={M};  GPipe/1F1B bubble=(p-1)/(m+p-1)={(P-1)/(M+P-1):.1%}")
print(f"GPipe activation 显存 ∝ m={M}; 1F1B ∝ p={P}  ← 这就是 1F1B 的全部价值")

def make_global():
    g = torch.Generator().manual_seed(3)
    W = [torch.randn(D,D,generator=g)*0.3 for _ in range(P)]
    b = [torch.randn(D,generator=g)*0.1 for _ in range(P)]
    X = torch.randn(M*MB,D,generator=g); Y = torch.randn(M*MB,D,generator=g)
    return W,b,X,Y
""")

md(r"""## Part 1 · 单进程 reference

把所有 stage 串起来跑完整 batch，`backward()` 得到每个 stage 的权重梯度真值。
""")
code(r"""
def stage_fwd(x,W,b): return torch.relu(x@W.t()+b)
def reference():
    W,b,X,Y = make_global()
    Ws=[w.clone().requires_grad_(True) for w in W]; bs=[bb.clone().requires_grad_(True) for bb in b]
    h=X
    for r in range(P): h=stage_fwd(h,Ws[r],bs[r])
    loss=((h-Y)**2).mean(); loss.backward()
    return loss.detach(), [w.grad for w in Ws], [bb.grad for bb in bs]
REFL, REFWG, REFBG = reference()
print("reference loss:", REFL.item())
""")

md(r"""## Part 2 · 关键原语：手动 `forward_step` / `backward_step` + combined P2P

PP 的反向跨越进程边界，不能靠一张 autograd 图。每个 stage：

- **forward_step**：收上游 activation `x`（`requires_grad`）→ 算 `y=stage(x)` → 发给下游；把 `(x,y)` 入队。
- **backward_step**：收下游回传的 `dy`（last stage 则从 loss 求）→ `torch.autograd.backward(y, dy)` → 这一步**同时**给出本 stage 权重梯度（累加）和 `x.grad` → 把 `x.grad` 发回上游。

为避免「我等你收、你等我收」的交错死锁，用 **combined `batch_isend_irecv`** 把「发 forward」和「收 backward」合成一次原子操作 —— 这正是 Megatron 的 `send_forward_recv_backward` / `send_backward_recv_forward`（[`01` 第 3 节](./01_gpipe_1f1b.md)）。
""")

md(r"""## Part 3 & 4 · GPipe 与 1F1B（真分布式）

worker 同时实现两种调度，各自验证梯度并回收每个 stage 的执行顺序字符串（用来「画」流水线）。
""")
code(r'''
%%writefile pp_worker.py
import os, torch, torch.distributed as dist
P, D, M, MB = 3, 4, 4, 2

def make_global():
    g=torch.Generator().manual_seed(3)
    W=[torch.randn(D,D,generator=g)*0.3 for _ in range(P)]; b=[torch.randn(D,generator=g)*0.1 for _ in range(P)]
    return W,b,torch.randn(M*MB,D,generator=g),torch.randn(M*MB,D,generator=g)
def stage_fwd(x,W,b): return torch.relu(x@W.t()+b)
def reference():
    W,b,X,Y=make_global(); Ws=[w.clone().requires_grad_(True) for w in W]; bs=[bb.clone().requires_grad_(True) for bb in b]
    h=X
    for r in range(P): h=stage_fwd(h,Ws[r],bs[r])
    ((h-Y)**2).mean().backward(); return [w.grad for w in Ws],[bb.grad for bb in bs]

def sendrecv(send_t, send_dst, recv_shape, recv_src):
    "combined P2P: 同时发与收, 避免交错 send/recv 死锁 (= Megatron send_*_recv_*)"
    ops=[]; buf=None
    if send_t is not None: ops.append(dist.P2POp(dist.isend, send_t.contiguous(), send_dst))
    if recv_shape is not None:
        buf=torch.empty(recv_shape); ops.append(dist.P2POp(dist.irecv, buf, recv_src))
    if ops:
        for req in dist.batch_isend_irecv(ops): req.wait()
    return buf

def run(rank, world):
    os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29688')
    dist.init_process_group('gloo', rank=rank, world_size=world)
    W,b,X,Y=make_global(); Wr=W[rank].clone().requires_grad_(True); br=b[rank].clone().requires_grad_(True)
    refWG,refBG=reference(); is_first,is_last=rank==0,rank==world-1; shape=(MB,D)

    def data_in(i): return X[i*MB:(i+1)*MB].clone().requires_grad_(True)
    def recv_forward(i):
        if is_first: return data_in(i)
        return sendrecv(None,None,shape,rank-1).requires_grad_(True)
    def bwd(x,y,gy,i):
        if is_last: (((y-Y[i*MB:(i+1)*MB])**2).mean()/M).backward()
        else: torch.autograd.backward(y,gy)
        return x.grad

    def gpipe():
        Wr.grad=None; br.grad=None; ins,outs,order=[],[],[]
        x=recv_forward(0)
        for i in range(M):                              # 全 forward
            y=stage_fwd(x,Wr,br)
            if not is_last: sendrecv(y,rank+1,None,None)
            ins.append(x); outs.append(y); order.append(f"F{i}")
            x=recv_forward(i+1) if i+1<M else None
        for i in reversed(range(M)):                    # 全 backward
            gy=None if is_last else sendrecv(None,None,shape,rank+1)
            xg=bwd(ins[i],outs[i],gy,i)
            if not is_first: sendrecv(xg,rank-1,None,None)
            order.append(f"B{i}")
        return order

    def one_f_one_b():
        Wr.grad=None; br.grad=None; ins,outs,order=[],[],[]
        nwarm=min(world-rank-1,M); nrem=M-nwarm; done=0
        x=recv_forward(0)
        for i in range(nwarm):                          # warmup: 只 forward
            y=stage_fwd(x,Wr,br)
            if not is_last: sendrecv(y,rank+1,None,None)
            ins.append(x); outs.append(y); order.append(f"F{i}"); x=recv_forward(i+1)
        for k in range(nrem):                           # steady: F/B 交替
            i=nwarm+k; y=stage_fwd(x,Wr,br); ins.append(x); outs.append(y); order.append(f"F{i}")
            gy=None if is_last else sendrecv(y,rank+1,shape,rank+1)   # send fwd + recv bwd (combined)
            xg=bwd(ins[done],outs[done],gy,done); order.append(f"B{done}")
            last=(k==nrem-1)
            if is_first: x=None if last else data_in(i+1)
            elif last: sendrecv(xg,rank-1,None,None); x=None
            else: x=sendrecv(xg,rank-1,shape,rank-1).requires_grad_(True)  # send bwd + recv fwd
            done+=1
        while done<M:                                   # cooldown: 排空 backward
            gy=None if is_last else sendrecv(None,None,shape,rank+1)
            xg=bwd(ins[done],outs[done],gy,done); order.append(f"B{done}")
            if not is_first: sendrecv(xg,rank-1,None,None)
            done+=1
        return order

    og=gpipe(); ok1=torch.allclose(Wr.grad,refWG[rank],atol=1e-5) and torch.allclose(br.grad,refBG[rank],atol=1e-5)
    r1=torch.tensor([1.0 if ok1 else 0.0]); dist.all_reduce(r1)
    o1=one_f_one_b(); ok2=torch.allclose(Wr.grad,refWG[rank],atol=1e-5) and torch.allclose(br.grad,refBG[rank],atol=1e-5)
    r2=torch.tensor([1.0 if ok2 else 0.0]); dist.all_reduce(r2)
    gg=[None]*world; dist.all_gather_object(gg,' '.join(og)); g1=[None]*world; dist.all_gather_object(g1,' '.join(o1))
    if rank==0:
        print(f"[GPipe] grads==reference: {int(r1.item())}/{world}")
        for r in range(world): print(f"   stage{r}: {gg[r]}")
        print(f"\n[1F1B]  grads==reference: {int(r2.item())}/{world}")
        for r in range(world): print(f"   stage{r} (warmup={min(world-r-1,M)}): {g1[r]}")
    dist.destroy_process_group()
''')
code(r"""
import torch.multiprocessing as mp
import pp_worker, importlib; importlib.reload(pp_worker)
mp.spawn(pp_worker.run, args=(P,), nprocs=P, join=True)
""")

md(r"""### 读懂这张时序图

`Fi` = micro-batch i 的 forward，`Bi` = 它的 backward。看 **1F1B** 的输出：

- **stage0 (warmup=2)**：`F0 F1 F2 B0 F3 B1 B2 B3` —— 先填 2 个 forward（把流水线填到下游），再进入 `F/B` 交替，最后排空。它要存到 **3 个** micro-batch 的 activation（warmup+1=p）。
- **stage2 (warmup=0)**：`F0 B0 F1 B1 F2 B2 F3 B3` —— 立刻 forward+backward 交替，只需存 **1 个** micro-batch 的 activation。

对比 **GPipe**：每个 stage 都是 `F0 F1 F2 F3 B3 B2 B1 B0` —— 所有 forward 做完才 backward，于是**每个 stage 都得存全部 m=4 个 micro-batch 的 activation**。

**两者梯度逐元素相同**（都 == reference），bubble 也相同；唯一区别是 **1F1B 把 first stage 的 activation 从 m 个降到 p 个** —— 这正是 [`01` 第 2 节](./01_gpipe_1f1b.md) 的核心结论。
""")

md(r"""## Part 5 · bubble 量化

用「格子数」估算 bubble：每个 stage 在 warmup/cooldown 有空闲格。理论 bubble fraction = `(p-1)/(m+p-1)`。
""")
code(r"""
def bubble_fraction(p, m, v=1):
    # interleaved: 把流水线细化 v 倍, bubble 降 1/v
    return (p-1)/(v*m + p - 1)
for (p,m,v) in [(P,M,1),(P,M*4,1),(8,32,1),(8,32,2),(8,32,4)]:
    tag = "interleaved" if v>1 else "1F1B/GPipe"
    print(f"p={p}, m={m}, v={v} ({tag}): bubble ≈ {bubble_fraction(p,m,v):.1%}")
print("\n→ 增大 m 或 interleaved(v>1) 都能压 bubble; 这对应 README 第 2-3 节与 02 的 interleaved。")
""")

md(r"""## 练习 / 往真实系统靠

1. 把 `P`/`M` 调大，观察 1F1B 的 warmup 数与 activation 份数随 stage 变化（first stage 最苦，[`03` 第 1 节](./03_overlap_and_memory.md)）。
2. 实现 **interleaved 1F1B**：每个 rank 持有 2 个不连续 chunk（如 stage0 = layer 0,3），warmup 数改成 `(p-rank-1)*2+(v-1)*…`（[`02` 第 1 节](./02_interleaved_zerobubble_dualpipe.md)），数 bubble 是否降到 ~1/v。
3. 实现 **zero-bubble** 的 split-backward：把 `bwd` 拆成 `B_act`（算 `x.grad`，要回传）和 `B_wgt`（算 `W.grad`，可延后），把 `B_wgt` 塞进 cooldown 的空格（[`02` 第 2 节](./02_interleaved_zerobubble_dualpipe.md)）。
4. 给 P2P 加 **overlap**：用独立 stream 让 stage 边界传输和本 stage 计算并行（[`03` 第 2 节](./03_overlap_and_memory.md)）。
5. 在每个 stage 内叠加 **TP**（把 `W` 再按列/行切到子进程），验证 PP×TP 组合（[`../tp_sp`](../02_tp_sp/)）。
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python","version":"3.9"}},"nbformat":4,"nbformat_minor":5}
with open(sys.argv[1],"w") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print("wrote", sys.argv[1], "cells:", len(cells))
