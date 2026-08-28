import json, sys

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md(r"""# Lab · 手写 Context Parallelism：Ring Attention 与 Ulysses

本 notebook 用**纯 PyTorch** 把 [`docs/parallel/cp`](./README.md) 的两条 CP 主线亲手实现一遍：

1. **Ring Attention**（`cp_comm_type=p2p`，[`01`](./01_ring_attention.md)）：online softmax + **真·环形 P2P**（`batch_isend_irecv`），逐块累加 attention，逐元素对齐 full causal attention；
2. **Ulysses**（`cp_comm_type=a2a`，[`02`](./02_ulysses_a2a.md)）：两次 **all-to-all** 在「seq 切 ↔ head 切」之间切换，本地算完整 attention，**前向 + 反向**都对齐 reference；
3. **zigzag 负载均衡**（[`README` 第 4 节](./README.md)）：演示朴素切分的负载不均，以及 Megatron `get_pretrain_batch_on_this_cp_rank` 的 `2*cp` 块对称分配。

全部在 **Mac CPU**（gloo）几秒跑完。简化：单精度、小 `S/heads/d`、ring 用连续切分（causal mask 更直观）、不接 RoPE。但**算法关键点一个不少**：online softmax 累加、KV 环形传递与 compute overlap、causal 下的 src/dst mask、a2a 的转置语义、以及 `a2a.backward == a2a`。
""")

md(r"""## Part 0 · 全局配置与「全局问题」

CP 把 **sequence 维**切到多卡。每个 rank 从固定 seed 重建完整序列的 Q/K/V，便于和单卡 full attention 对比。
""")
code(r"""
import math, torch
CP, S, Hh, D = 2, 8, 2, 4      # cp size / seq len / heads / head_dim
L, h = S // CP, Hh // CP        # 每 rank 的 token 数 / Ulysses 下每 rank 的 head 数
assert S % CP == 0 and Hh % CP == 0
print(f"CP={CP}, seq S={S}, heads={Hh}, head_dim={D};  每 rank: {L} token(全 head) / Ulysses 下 {h} head(全 seq)")

def make_global():
    g = torch.Generator().manual_seed(7)
    Q = torch.randn(S, Hh, D, generator=g)
    K = torch.randn(S, Hh, D, generator=g)
    V = torch.randn(S, Hh, D, generator=g)
    return Q, K, V
""")

md(r"""## Part 1 · 单进程 reference：完整 causal attention

ground truth：把整段序列放一张卡，标准 `softmax(QKᵀ/√d + causal_mask)·V`。
""")
code(r"""
def full_attn(Q, K, V):
    q, k, v = Q.permute(1,0,2), K.permute(1,0,2), V.permute(1,0,2)   # [H,S,D]
    s = q @ k.transpose(-1,-2) / math.sqrt(D)
    s = s.masked_fill(~torch.tril(torch.ones(S,S)).bool(), float('-inf'))
    return (torch.softmax(s, -1) @ v).permute(1,0,2)                 # [S,H,D]

Q0,K0,V0 = make_global()
REF = full_attn(Q0,K0,V0)
print("reference out:", tuple(REF.shape), "| token0 head0:", REF[0,0].round(decimals=3).tolist())
""")

md(r"""## Part 2 · zigzag 负载均衡（[`README` 第 4 节](./README.md)）

causal mask 下，靠后的 token attend 范围更大。朴素地把序列连续切成 `cp` 段，**最后一个 rank 工作量是第一个的数倍**。Megatron 的解法：切成 `2*cp` 块，rank `r` 取块 `r` 和块 `2*cp-1-r`（一前一后），让每个 rank 工作量大致相等。下面用「每个 rank 需要算的 KV 块数」量化这个不均衡。
""")
code(r"""
def attended_blocks_naive(cp):
    # 连续切: rank r 持有第 r 段; causal 下需 attend 第 0..r 段 → r+1 块
    return [r+1 for r in range(cp)]

def attended_blocks_zigzag(cp):
    # 2*cp 块, rank r 取 (r, 2cp-1-r); 块 j 需 attend 0..j 共 j+1 个 sub-block
    return [ (r+1) + (2*cp-1-r+1) for r in range(cp) ]   # 两块各自的 attend 量之和

for cp in (2,4,8):
    nv, zz = attended_blocks_naive(cp), attended_blocks_zigzag(cp)
    print(f"cp={cp}:  naive 各 rank 工作量={nv} (max/min={max(nv)/min(nv):.1f}x)   "
          f"zigzag={zz} (max/min={max(zz)/min(zz):.2f}x)")
print("\n→ naive 最大/最小工作量比随 cp 线性恶化; zigzag 把它压到接近 1，这就是 utils.py:2308 的动机。")
""")
code(r"""
# Megatron get_pretrain_batch_on_this_cp_rank 的切分下标 (utils.py:2359)，单进程演示每个 rank 拿哪些 token
def zigzag_indices(cp, rank, seq=S):
    chunks = torch.arange(seq).view(2*cp, seq//(2*cp))
    idx = torch.tensor([rank, 2*cp - rank - 1])
    return chunks.index_select(0, idx).reshape(-1).tolist()
for r in range(CP):
    print(f"rank{r} 持有 token: {zigzag_indices(CP, r)}   (块 {r} 和块 {2*CP-1-r})")
""")

md(r"""## Part 3 · Ring Attention：online softmax + 真·环形 P2P（[`01`](./01_ring_attention.md)）

每个 rank 固定自己的 `Q_i`，让 `(K,V)` 沿环 `0→1→…→cp-1→0` 流动；转 `cp` 步后每个 query 见过所有 KV。每步用 **online softmax** 累加（FlashAttention 算子的跨卡版），并用 **`batch_isend_irecv`** 让「下一块 KV 的传输」与「当前块的 attention 计算」overlap。

causal mask 规则（连续切分，源块 `src` vs 本块 `rank`）：`src<rank` 全 attend、`src==rank` 块内 causal、`src>rank` 整块跳过（省一半计算）。
""")
code(r'''
%%writefile ring_worker.py
import os, math, torch, torch.distributed as dist
CP, S, Hh, D = 2, 8, 2, 4
L = S // CP

def make_global():
    g = torch.Generator().manual_seed(7)
    return (torch.randn(S,Hh,D,generator=g), torch.randn(S,Hh,D,generator=g), torch.randn(S,Hh,D,generator=g))

def ref_attn():
    Q,K,V = make_global(); q,k,v = Q.permute(1,0,2),K.permute(1,0,2),V.permute(1,0,2)
    s = q@k.transpose(-1,-2)/math.sqrt(D)
    s = s.masked_fill(~torch.tril(torch.ones(S,S)).bool(), float('-inf'))
    return (torch.softmax(s,-1)@v).permute(1,0,2)

def flash_update(O, m, l, q, k, v, causal):
    "online softmax 累加一块 KV。q:[H,Lq,D] k,v:[H,Lk,D]  O:[H,Lq,D] m,l:[H,Lq]"
    s = q @ k.transpose(-1,-2) / math.sqrt(D)            # [H,Lq,Lk]
    if causal:
        Lq, Lk = s.shape[-2], s.shape[-1]
        s = s.masked_fill(~torch.tril(torch.ones(Lq,Lk)).bool(), float('-inf'))
    m_new = torch.maximum(m, s.amax(-1))                 # running max
    p     = torch.exp(s - m_new.unsqueeze(-1))
    alpha = torch.exp(m - m_new)                         # 旧状态的修正系数
    l = alpha*l + p.sum(-1)
    O = alpha.unsqueeze(-1)*O + p @ v
    return O, m_new, l

def run(rank, world):
    os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29633')
    dist.init_process_group('gloo', rank=rank, world_size=world)
    Q,K,V = make_global()
    q_i = Q[rank*L:(rank+1)*L].permute(1,0,2).contiguous()                                # [H,L,D]
    kv  = torch.stack([K[rank*L:(rank+1)*L], V[rank*L:(rank+1)*L]]).permute(0,2,1,3).contiguous()  # [2,H,L,D]
    O = torch.zeros(Hh,L,D); m = torch.full((Hh,L), float('-inf')); l = torch.zeros(Hh,L)
    for step in range(world):
        src = (rank - step) % world
        nxt = torch.empty_like(kv)
        reqs = dist.batch_isend_irecv([                      # KV 沿环异步流动
            dist.P2POp(dist.isend, kv.clone(), (rank+1)%world),
            dist.P2POp(dist.irecv, nxt,        (rank-1)%world),
        ])
        if src <= rank:                                      # src>rank 的块整块跳过(causal)
            O,m,l = flash_update(O,m,l, q_i, kv[0], kv[1], causal=(src==rank))
        for r in reqs: r.wait()                              # 等本步 KV 到位
        kv = nxt
    out = (O / l.unsqueeze(-1)).permute(1,0,2)               # 归一化 → [L,H,D]
    ref = ref_attn()[rank*L:(rank+1)*L]
    ok = torch.allclose(out, ref, atol=1e-5)
    t = torch.tensor([1.0 if ok else 0.0]); dist.all_reduce(t)
    if rank==0: print(f"[ring] forward 全 {world} rank == reference: {int(t.item())}/{world}  ✅" if t.item()==world else f"[ring] MISMATCH {int(t.item())}/{world}")
    dist.destroy_process_group()
''')
code(r"""
import torch.multiprocessing as mp
import ring_worker, importlib; importlib.reload(ring_worker)
mp.spawn(ring_worker.run, args=(CP,), nprocs=CP, join=True)
""")

md(r"""**看清了什么**：ring 的每个 rank 只持有 `1/cp` 的 KV，靠 `cp` 步环形 P2P + online softmax 把完整 attention 算出来，且 KV 传输与 attention 计算 overlap（`batch_isend_irecv` 发出后立刻算当前块）。`src>rank` 整块跳过 = causal 省一半计算，也正是负载不均的来源（Part 2）。
""")

md(r"""## Part 4 · Ulysses：两次 all-to-all（[`02`](./02_ulysses_a2a.md)），前向 + 反向

与 ring 相反：不搬 KV，而是用 all-to-all 把 layout 从「seq 切（每 rank `L` token、全 `H` head）」转成「head 切（每 rank 全 `S` token、`h=H/cp` head）」，本地算**完整序列**的 attention，再 a2a 切回。

只需一个 autograd 原语 `A2A`：等分 all-to-all，**backward 还是 all-to-all**（与 [`../ep`](../05_ep/) 的 dispatch/combine 对称性同构）。于是整条 Ulysses 前向 + 反向都由 autograd 自动给出 —— 我们顺带验证 `Q.grad` 也跨进程对齐 reference。
""")
code(r'''
%%writefile uly_worker.py
import os, math, torch, torch.distributed as dist
CP, S, Hh, D = 2, 8, 2, 4
L, h = S // CP, Hh // CP

def make_global():
    g = torch.Generator().manual_seed(7)
    return (torch.randn(S,Hh,D,generator=g), torch.randn(S,Hh,D,generator=g), torch.randn(S,Hh,D,generator=g))

def full_attn(Q,K,V):
    q,k,v = Q.permute(1,0,2),K.permute(1,0,2),V.permute(1,0,2)
    s = q@k.transpose(-1,-2)/math.sqrt(D)
    s = s.masked_fill(~torch.tril(torch.ones(S,S)).bool(), float('-inf'))
    return (torch.softmax(s,-1)@v).permute(1,0,2)

def reference():
    Q,K,V = make_global(); Qr = Q.clone().requires_grad_(True)
    o = full_attn(Qr,K,V); o.sum().backward()
    return o.detach(), Qr.grad

class A2A(torch.autograd.Function):     # 等分 all_to_all；backward 还是 all_to_all
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group; out = torch.empty_like(x.contiguous())
        dist.all_to_all_single(out, x.contiguous(), group=group); return out
    @staticmethod
    def backward(ctx, g):
        out = torch.empty_like(g.contiguous()); dist.all_to_all_single(out, g.contiguous(), group=ctx.group); return out, None

def seq2head(x, grp):   # [L,H,D] (seq切) -> [S,h,D] (head切)
    x = x.view(L, CP, h, D).permute(1,0,2,3).reshape(CP*L, h, D)
    return A2A.apply(x, grp).reshape(S, h, D)

def head2seq(y, grp):   # [S,h,D] -> [L,H,D]
    y = A2A.apply(y.reshape(CP*L, h, D), grp)
    return y.reshape(CP, L, h, D).permute(1,0,2,3).reshape(L, Hh, D)

def run(rank, world):
    os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29644')
    dist.init_process_group('gloo', rank=rank, world_size=world)
    grp = dist.group.WORLD
    Q,K,V = make_global()
    q = Q[rank*L:(rank+1)*L].clone().requires_grad_(True)        # [L,H,D] seq-split
    k = K[rank*L:(rank+1)*L].clone(); v = V[rank*L:(rank+1)*L].clone()
    qh, kh, vh = seq2head(q,grp), seq2head(k,grp), seq2head(v,grp)   # → [S,h,D] head-split
    qq,kk,vv = qh.permute(1,0,2), kh.permute(1,0,2), vh.permute(1,0,2)
    s = qq@kk.transpose(-1,-2)/math.sqrt(D)
    s = s.masked_fill(~torch.tril(torch.ones(S,S)).bool(), float('-inf'))
    oh = (torch.softmax(s,-1)@vv).permute(1,0,2)                    # [S,h,D] 本地完整 attention
    o = head2seq(oh, grp)                                          # → [L,H,D]
    o.sum().backward()
    refO, refQg = reference()
    okO = torch.allclose(o.detach(), refO[rank*L:(rank+1)*L], atol=1e-5)
    okG = torch.allclose(q.grad,     refQg[rank*L:(rank+1)*L], atol=1e-5)
    t = torch.tensor([1.0 if (okO and okG) else 0.0]); dist.all_reduce(t)
    if rank==0: print(f"[ulysses] forward+backward 全 {world} rank == reference: {int(t.item())}/{world}  ✅" if t.item()==world else f"[ulysses] MISMATCH {int(t.item())}/{world}")
    dist.destroy_process_group()
''')
code(r"""
import uly_worker, importlib; importlib.reload(uly_worker)
mp.spawn(uly_worker.run, args=(CP,), nprocs=CP, join=True)
""")

md(r"""## Part 5 · 对照与小结

| | Ring（Part 3） | Ulysses（Part 4） |
|---|---|---|
| 搬什么 | KV chunk 沿环 P2P | head 维 all-to-all |
| 通信次数 | `cp` 步 P2P（可与 compute overlap） | 2×3 次 a2a（不可 overlap）|
| 本地计算 | online softmax 逐块累加 | 完整 attention（`h` 个 head）|
| head 约束 | 无 | `cp ≤ heads` 且整除（本 lab `cp=2 ≤ heads=2`）|
| 反向 | 再转一圈（本 lab 只验前向） | `A2A.backward==A2A`，autograd 自动（已验 Q.grad）|

两者都把 `O(s²)` 单卡 attention 摊到 `cp` 张卡，且每卡只存 `1/cp` 的 Q/K/V。对应正文 [`README` 第 3 节](./README.md) 的对比表。

## 练习 / 往真实系统靠

1. 把 ring 的连续切分换成 **zigzag**（Part 2 的下标），重写 causal mask 逻辑（src/dst 块的真实 position 区间），验证负载更均衡且结果不变。
2. 给 ring 实现 **backward**（再沿环传 `dK,dV`），验证 `Q/K/V.grad` 对齐 reference。
3. 把 Ulysses 的 `cp` 调到 `heads`（如 `cp=heads=2` 时每 rank 1 个 head），再试 `cp>heads` 观察为何报错（head 不够切）。
4. 实现 **`a2a+p2p` 分层**（[`02` 第 4 节](./02_ulysses_a2a.md)）：把 `cp` 拆成内层 a2a + 外层 ring 两级 group。
5. 把本地 attention 换成调用 `torch.nn.functional.scaled_dot_product_attention`，对齐工业实现。
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python","version":"3.9"}},"nbformat":4,"nbformat_minor":5}
with open(sys.argv[1],"w") as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print("wrote", sys.argv[1], "cells:", len(cells))
