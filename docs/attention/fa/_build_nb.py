import json, sys

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)})

md(r"""# Lab · 手写 FlashAttention（Mac CPU / 纯 PyTorch）

本 notebook 把 [`docs/attention/fa`](./README.md) 的**算法核心**亲手实现一遍——不依赖任何 CUDA kernel，纯 torch、纯 CPU，几秒跑完。重点是把三个不变量（[`01`](./01_io_awareness_online_softmax.md)）做到**逐元素正确**：

1. **Tiling + online softmax forward**：Q 切行块、KV 切列块，外层循环遍历 KV 块，维护 running `(m, l, O)`，用 `α = exp(m_old - m_new)` 修正旧累加——**从不物化 `[N, N]` 的 `S/P`**；
2. **LSE**：循环结束存 `LSE = m + log(l)`，既是归一化因子也是 backward 的统计量；
3. **Recomputation backward**：只用 `Q,K,V,O,LSE,dO` **重算** `P`，求 `dQ,dK,dV`，逐元素对齐 `torch.autograd`；
4. **causal** 变体 + KV 块裁剪；
5. 量化「不物化 `[N,N]`」省下的**显存**（这是 FA 的全部意义）。

> 这是 CPU 教学实现，**刻意**用 Python 双重 for 循环显式展开 tiling，对应 GPU 上「一个 CTA 一个 Q 块、内层循环 KV 块」。它**不快**（真正的加速来自把块放进 SRAM + Tensor Core + 异步，见 [`02`](./02_fa2_parallelism.md)/[`03`](./03_fa3_hopper_async.md)），但算法不变量一个不少。
""")

md(r"""## Part 0 · 配置与 ground truth

单 head 视角（多 head 只是再加一层 batch 维，不影响算法）。reference 用 PyTorch 的标准 attention（物化 `[N,N]`）作为 ground truth。
""")
code(r"""
import math, torch
torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

N, Nk, d = 128, 128, 32      # seqlen_q / seqlen_k / head_dim（小尺寸，CPU 秒级）
scale = 1.0 / math.sqrt(d)

Q = torch.randn(N,  d, dtype=torch.float64)   # float64 让数值对比更干净
K = torch.randn(Nk, d, dtype=torch.float64)
V = torch.randn(Nk, d, dtype=torch.float64)

def ref_attention(Q, K, V, causal=False, scale=scale):
    # 标准实现：完整物化 S=[N,Nk] 和 P=[N,Nk]
    S = (Q @ K.transpose(-1, -2)) * scale           # [N, Nk]  ← O(N^2) 中间矩阵
    if causal:
        qi = torch.arange(S.shape[0])[:, None]
        ki = torch.arange(S.shape[1])[None, :]
        S = S.masked_fill(ki > qi, float('-inf'))
    P = torch.softmax(S, dim=-1)                    # [N, Nk]  ← 又一个 O(N^2)
    return P @ V, S                                  # [N, d]

O_ref, _ = ref_attention(Q, K, V)
print("reference O:", tuple(O_ref.shape), "| O[0,:4] =", O_ref[0, :4].tolist())
""")

md(r"""## Part 1 · FlashAttention forward：online softmax 逐块累加

完全对应 [`01` 第 3 节](./01_io_awareness_online_softmax.md)的伪代码与 [`flash_attn_triton.py:212-258`](../../../references/flash-attention/flash_attn/flash_attn_triton.py#L212-L258)。关键：内层循环里 `S_block` / `P_block` 是 `[Br, Bc]` 的小块，**用完即弃**；`O` 累加器全程不除 `l`，循环结束才一次性归一（这就是 FA2 的「延迟 rescale」，[`02` 第 1 节](./02_fa2_parallelism.md)）。
""")
code(r"""
def flash_forward(Q, K, V, Br=32, Bc=32, causal=False, scale=scale):
    N, d = Q.shape
    Nk = K.shape[0]
    O = torch.zeros(N, d, dtype=Q.dtype)
    LSE = torch.zeros(N, dtype=Q.dtype)
    max_block_elems = 0      # 追踪「片上同时持有的最大块」元素数 → 证明 O(Br*Bc) 而非 O(N^2)

    for i in range(0, N, Br):                          # 外层：遍历 Q 行块（GPU 上是不同 CTA 并行）
        qi = Q[i:i+Br]                                  # [br, d]
        br = qi.shape[0]
        m_i = torch.full((br,), float('-inf'), dtype=Q.dtype)   # running max
        l_i = torch.zeros(br, dtype=Q.dtype)                    # running sum(exp)
        o_i = torch.zeros(br, d, dtype=Q.dtype)                 # running 加权 V 累加（未归一）

        # causal：只需遍历到「包含本 Q 块对角线」的 KV 块为止（[02] 第 4 节的块裁剪）
        j_end = min(Nk, i + br) if causal else Nk
        for j in range(0, j_end, Bc):                  # 内层：遍历 KV 列块
            kj = K[j:j+Bc]; vj = V[j:j+Bc]              # [bc, d]
            s = (qi @ kj.transpose(-1, -2)) * scale     # [br, bc]  ← GEMM-I，停在「寄存器」
            if causal:                                  # 块内逐元素 mask（全局下标）
                qidx = (i + torch.arange(br))[:, None]
                kidx = (j + torch.arange(kj.shape[0]))[None, :]
                s = s.masked_fill(kidx > qidx, float('-inf'))
            max_block_elems = max(max_block_elems, s.numel())

            # ---- online softmax 更新（[01] 第 2 节）----
            m_new = torch.maximum(m_i, s.max(dim=-1).values)     # 新 running max
            p = torch.exp(s - m_new[:, None])                    # [br, bc] 未归一概率
            alpha = torch.exp(m_i - m_new)                       # 旧累加的修正因子
            l_i = alpha * l_i + p.sum(dim=-1)
            o_i = alpha[:, None] * o_i + p @ vj                  # GEMM-II：修正旧 O 再加新块
            m_i = m_new

        o_i = o_i / l_i[:, None]                        # 一次性归一（延迟 rescale）
        O[i:i+Br] = o_i
        LSE[i:i+Br] = m_i + torch.log(l_i)              # 存给 backward
    return O, LSE, max_block_elems

O_fa, LSE_fa, mbe = flash_forward(Q, K, V, Br=32, Bc=32)
err = (O_fa - O_ref).abs().max().item()
print(f"max|O_flash - O_ref| = {err:.2e}   (应 ~1e-15，float64 机器精度)")
print(f"片上同时持有的最大 S/P 块 = {mbe} 元素 (= Br*Bc = 32*32)，与 N 无关 → O(N) 显存")
assert err < 1e-10
""")

md(r"""**验证不变量①**：逐元素对齐标准实现（误差在机器精度），且**片上最大块恒为 `Br*Bc`，与 `N` 无关**。把 `N` 放大、`Br/Bc` 不变，标准实现的 `S` 是 `N²`、FA 的块仍是 `Br*Bc`——这就是 `O(N²)→O(N)` 显存的来源（Part 4 量化）。

顺带验证「online softmax 对块顺序不敏感」（[`01` 第 2 节](./01_io_awareness_online_softmax.md)）：把 KV 打乱顺序，结果不变（这正是 Ring Attention 能跨卡轮转 KV 的基础）。
""")
code(r"""
def flash_forward_shuffled(Q, K, V, Br=32, Bc=32, scale=scale):
    perm = torch.randperm(K.shape[0])
    O, _, _ = flash_forward(Q, K[perm], V[perm], Br, Bc, causal=False)
    return O

O_shuf = flash_forward_shuffled(Q, K, V)
print(f"max|O_shuffled - O_ref| = {(O_shuf - O_ref).abs().max().item():.2e}  → KV 块顺序无关 ✓")
""")

md(r"""## Part 2 · causal 变体

causal 下 query `i` 只 attend `key ≤ i`。实现上：①外层只遍历到「含对角线」的 KV 块（块裁剪，省一半计算）②对角块内逐元素 mask。对齐 reference 的 causal。
""")
code(r"""
O_ref_c, _ = ref_attention(Q, K, V, causal=True)
O_fa_c, LSE_c, _ = flash_forward(Q, K, V, Br=32, Bc=32, causal=True)
err_c = (O_fa_c - O_ref_c).abs().max().item()
print(f"max|O_flash_causal - O_ref_causal| = {err_c:.2e}")
assert err_c < 1e-10
print("causal 对齐 ✓")
""")

md(r"""## Part 3 · Recomputation backward

forward 只存了 `O(N)` 的 `O, LSE`，**没存 `P`**。backward 用 `Q,K,V,O,LSE,dO` **重算** `P = exp(S·scale - LSE)`（[`01` 第 5 节](./01_io_awareness_online_softmax.md)），再走 softmax 的 Jacobian 求 `dQ,dK,dV`：

```
D  = rowsum(dO ⊙ O)                 # [N]
dV = Pᵀ dO ;  dP = dO Vᵀ
dS = P ⊙ (dP - D) · scale
dQ = dS K ;  dK = dSᵀ Q
```

注意循环结构：`dQ` 按 Q 块本地累加；`dK,dV` 需跨 Q 块累加（对应 [`02` 第 5 节](./02_fa2_parallelism.md)「backward 固定 KV、沿 Q 循环」的归约方向）。
""")
code(r"""
def flash_backward(Q, K, V, O, LSE, dO, Br=32, Bc=32, causal=False, scale=scale):
    N, d = Q.shape; Nk = K.shape[0]
    dQ = torch.zeros_like(Q); dK = torch.zeros_like(K); dV = torch.zeros_like(V)
    D = (dO * O).sum(dim=-1)                          # [N]
    for i in range(0, N, Br):
        qi = Q[i:i+Br]; doi = dO[i:i+Br]
        Di = D[i:i+Br]; Li = LSE[i:i+Br]
        br = qi.shape[0]
        dqi = torch.zeros(br, d, dtype=Q.dtype)
        j_end = min(Nk, i + br) if causal else Nk
        for j in range(0, j_end, Bc):
            kj = K[j:j+Bc]; vj = V[j:j+Bc]
            s = (qi @ kj.transpose(-1, -2)) * scale
            if causal:
                qidx = (i + torch.arange(br))[:, None]
                kidx = (j + torch.arange(kj.shape[0]))[None, :]
                s = s.masked_fill(kidx > qidx, float('-inf'))
            p = torch.exp(s - Li[:, None])            # 重算 P（masked 处 exp(-inf)=0）
            dV[j:j+Bc] += p.transpose(-1, -2) @ doi   # 跨 Q 块累加
            dp = doi @ vj.transpose(-1, -2)
            ds = (p * (dp - Di[:, None])) * scale
            dqi += ds @ kj
            dK[j:j+Bc] += ds.transpose(-1, -2) @ qi   # 跨 Q 块累加
        dQ[i:i+Br] = dqi
    return dQ, dK, dV

# ground truth：让 torch.autograd 对标准实现求导
for causal in (False, True):
    Qa = Q.clone().requires_grad_(True); Ka = K.clone().requires_grad_(True); Va = V.clone().requires_grad_(True)
    Oa, _ = ref_attention(Qa, Ka, Va, causal=causal)
    dO = torch.randn_like(Oa)
    Oa.backward(dO)

    O_f, LSE_f, _ = flash_forward(Q, K, V, causal=causal)
    dQ, dK, dV = flash_backward(Q, K, V, O_f, LSE_f, dO, causal=causal)
    eq = max((dQ-Qa.grad).abs().max().item(), (dK-Ka.grad).abs().max().item(), (dV-Va.grad).abs().max().item())
    print(f"causal={causal!s:5}  max|grad_flash - grad_autograd| = {eq:.2e}")
    assert eq < 1e-9
print("\nbackward（recomputation）逐元素对齐 autograd ✓  —— 全程未物化 [N,N] 的 P")
""")

md(r"""## Part 4 · 量化「不物化 `[N,N]`」省的显存

FA 的全部意义。比较：标准实现的峰值中间显存（`S` + `P` 两个 `[N,Nk]`） vs FA 的峰值（只有 `[Br,Bc]` 的块 + `O(N)` 的 `O,LSE`）。这里只**计算字节数**、不真的分配大张量。
""")
code(r"""
def mem_bytes(N, Nk, d, Br, Bc, dtype_bytes=2):  # bf16=2B
    std_intermediate = 2 * N * Nk * dtype_bytes                   # S + P，都是 [N,Nk]
    fa_intermediate  = (Br * Bc + N) * dtype_bytes                # 一个块 + LSE（O 是必有的输出）
    return std_intermediate, fa_intermediate

print(f"{'N':>8} {'标准 S+P':>16} {'FlashAttn':>14} {'省下':>10}")
for N in (1024, 8192, 32768, 131072):
    std, fa = mem_bytes(N, N, 128, 128, 128)
    ratio = std / fa
    print(f"{N:>8} {std/2**20:>13.1f} MB {fa/2**10:>11.1f} KB {ratio:>8.0f}x")
print("\n标准实现的中间显存随 N^2 爆炸；FA 的块恒为 Br*Bc → 这就是 4K→128K 长上下文能跑的根本原因。")
""")

md(r"""## 小结

| 不变量 | 本 lab 验证 | 对应文档/代码 |
|---|---|---|
| tiling：不物化 `[N,N]` | Part 1：片上最大块 = `Br*Bc`，与 N 无关；Part 4：显存 `O(N²)→O(N)` | [`01`](./01_io_awareness_online_softmax.md) §3 |
| online softmax：流式算对、块序无关 | Part 1：对齐 reference + 打乱 KV 不变 | [`01`](./01_io_awareness_online_softmax.md) §2，`triton:212` |
| 延迟 rescale | Part 1：循环内 `o_i` 不除 `l`，末尾一次归一 | [`02`](./02_fa2_parallelism.md) §1 |
| causal 块裁剪 | Part 2：只遍历含对角线的 KV 块 | [`02`](./02_fa2_parallelism.md) §4 |
| recomputation + LSE backward | Part 3：用 `LSE` 重算 `P`，对齐 autograd | [`01`](./01_io_awareness_online_softmax.md) §5 |

真正的 FlashAttention 在此之上做的，全是「同一个算法怎么映射到 GPU 才快」：把块放进 SRAM、用 Tensor Core 算两个 GEMM、用 TMA/WGMMA 异步化、用 warp-specialization + ping-pong 让访存和计算重叠（[`02`](./02_fa2_parallelism.md)/[`03`](./03_fa3_hopper_async.md)）。算法不变量，就是你在这个 notebook 里手写的这些。
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = sys.argv[1] if len(sys.argv) > 1 else "fa_lab.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {out} with {len(cells)} cells")
