# 06 · Flash Linear Attention

> 本篇讲 linear attention 的公式如何变成 Triton kernel，与机制侧的数学推导互补。阅读本篇需要 [01 · IO-awareness、online softmax 与 tiling](./01_io_awareness_online_softmax.md)（tiling、SRAM/HBM 层级、recomputation backward）与 [06 · linear 路线（一）：kernel trick、RNN 等价与三种计算形式](../mechanisms/06_linear_foundation.md) §3（chunkwise 的三行公式）作为前置；机制侧的数学见 [07 · linear 路线（二）：衰减机制的演进](../mechanisms/07_linear_decay_gating.md)–[`09`](../mechanisms/09_linear_kda_kimi.md)。
>
> 对照代码：上游固定版本 [[fla:]]，commit `81091cc6`，package version `0.5.2`（[[fla:fla/__init__.py#L12]]）。**"flash linear attention" 这个名字出自 GLA 论文** [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) §3。

---

## 0. 与 FlashAttention 的对照

| | FlashAttention | Flash Linear Attention |
|---|---|---|
| 要避免物化的东西 | $[N, N]$ 的 $S/P$ | 每步的 $[d_k, d_v]$ hidden state（$L$ 份） |
| 手段 | tiling + online softmax | **chunkwise** + 只在 chunk 边界物化状态 |
| 外层循环 | 一个 CTA 固定 Q 块、遍历 KV 块 | 一个 CTA 固定 state 的 $[BK, BV]$ 分块、**遍历时间** |
| backward | 用 $Q, K, V, O, \mathrm{LSE}$ **重算** $S/P$ | 用 $Q, K, V, g$ **重算** chunk states |
| 不变量 | online softmax 的 $m, l, O$ 三元组 | $O = QS + ((QK^{\top}) \odot M)\,\tilde{V}$，$S$ store-before-update |

**两边是同一套 IO-aware 思路的两个实例**，连「用算力换显存」的这笔交换都一样。差别在于 linear attention 没有 softmax，所以不需要 online 归一化，但多了一个跨 chunk 的串行依赖——这是它全部工程复杂度的来源。

## 1. 一个 op 由哪几个 kernel 拼起来

`simple_gla` 是最干净的入口：它没有写任何 Triton 代码，整个 forward 就是两个共享 kernel 的编排。

[[fla:fla/ops/simple_gla/chunk.py#L31-L58]]

```python
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=False,
        state_v_first=state_v_first,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )
    return o, ht
```

**这正是 GLA 论文 Algorithm 1 的 materialization 版本**：先串行扫一遍，把所有 chunk state 写进 HBM，再让所有 chunk 并行算输出。

依赖图（`simple_gla` / `gla` / `retention` / `lightning_attn` 共用；delta-rule 族多两级）：

```
        ┌─ chunk_local_cumsum ──────► g（对数域局部累积衰减，预乘 RCP_LN2）
        │      utils/cumsum.py:428
        │
衰减族 ─┼─ chunk_fwd_h ─────────────► h[B,NT,H,K,V]（HBM）+ ht
        │      common/chunk_h.py:306
        │
        └─ chunk_fwd_o ─────────────► o
               common/chunk_o.py:543

delta 族在 chunk_fwd_h 之前多插两级：
        ├─ solve_tril / kkt_solve ──► A = (I + strictLower(βKKᵀ⊙Γ))⁻¹
        │      utils/solve_tril.py:355 或 gated_delta_rule/chunk_fwd.py:40（融合版）
        └─ recompute_w_u ───────────► W = A(Γ⊙K), U = A·V
               gated_delta_rule/wy_fast.py:43
   然后走 chunk_delta_h.py:687 而不是 chunk_h.py:306
```

顺带一个能说明「共享 kernel」复用程度的例子：`lightning_attn` 完全没有自己的实现，它就是 `simple_gla` 加一个固定的逐头斜率：

[[fla:fla/ops/lightning_attn/chunk.py#L74-L80]]

```python
    g_gamma = -(8 / H * (1 - layer_idx / num_layers)) * q.new_tensor(range(H), dtype=torch.float)
    return chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        g_gamma=g_gamma,
```

[[fla:fla/ops/]] 下有 39 个族，但真正的 kernel 集中在 [[fla:fla/ops/common/]] 和 [[fla:fla/ops/utils/]] 里那几个，靠 `tl.constexpr` 开关（`USE_G`/`USE_GK`/`USE_GV`/`USE_G_GAMMA`）区分——见 §9。

## 2. inter-chunk 状态递归：`chunk_fwd_kernel_h`

[[fla:fla/ops/common/chunk_h.py#L36]]。这个 kernel 实现 $S_{[i+1]} = \mathrm{Diag}(\gamma) \cdot S_{[i]} + K_{[i]}^{\top} V_{[i]}$，是整个 op 里唯一带跨 chunk 串行依赖的部分。

### 累加器常驻寄存器

[[fla:fla/ops/common/chunk_h.py#L84-L84]]

```python
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
```

`b_h` 是 fp32 累加器，在整个 kernel 生命周期内都驻留在寄存器里，除了下面指定的 store 点之外从不落 HBM。`BK=BV=64` 时它占 16 KiB 寄存器。

### 时间循环，以及 store-before-update

[[fla:fla/ops/common/chunk_h.py#L95-L115]]

```python
    for i_t in range(NT):
        i_s = i_t // NTS
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_k = k + (bos*H + i_h) * K + o_k[:, None] + o_t[None, :] * (H*K)
        p_v = v + (bos*H + i_h) * V + o_t[:, None] * (H*V) + o_v[None, :]

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
        if STATE_V_FIRST:
            p_h = h + o_h + o_v[:, None] * K + o_k[None, :]
            m_h = (o_v[:, None] < V) & (o_k[None, :] < K)
        else:
            p_h = h + o_h + o_k[:, None] * V + o_v[None, :]
            m_h = (o_k[:, None] < K) & (o_v[None, :] < V)

        if i_t % NTS == 0:
            tl.store(p_h, (tl.trans(b_h) if STATE_V_FIRST else b_h).to(p_h.dtype.element_ty), mask=m_h)
        # [BK, BT]
        b_k = tl.load(p_k, mask=(o_k[:, None] < K) & m_t[None, :], other=0.0)
        # [BT, BV]
        b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
```

注意 `:110-111` 的 store 发生在更新之前：写进 `h[i_t]` 的是**进入** chunk `i_t` 时的状态，也就是数学上的 $S_{[t]}$——正是 `chunk_fwd_o` 算 $Q_{[t]} S_{[t]}$ 时需要的那个。这与 [06 · linear 路线（一）：kernel trick、RNN 等价与三种计算形式](../mechanisms/06_linear_foundation.md) §3.3 代码注释里那句「`S` 的更新在输出之后」是同一件事。**写反了不会报错，只会静默算成 off-by-one-chunk。**

`p_k` 的构造有个细节值得点出：`o_k[:, None] + o_t[None, :] * (H*K)` —— 行索引走 $K$（连续，stride 1）、列索引走时间，于是 `b_k` 直接以 $[BK, BT]$ 的布局到手，不需要 `tl.trans`，正好是 `tl.dot` 算 $K^{\top} V$ 想要的布局。

### 衰减，然后一次 `tl.dot`

[[fla:fla/ops/common/chunk_h.py#L118-L124]]

```python
        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_h *= exp2(b_g_last)
            b_v = (b_v * exp2(b_g_last - b_g)[:, None]).to(b_v.dtype)
```

对照 [07 · linear 路线（二）：衰减机制的演进](../mechanisms/07_linear_decay_gating.md) §3.5 的 `ssd_chunk`：`b_h *= exp2(g_last)` 就是 $S \cdot \gamma^C$，`b_v * exp2(g_last − g)` 就是把贡献衰减到 chunk 末尾。`USE_GK`（通道级，GLA 用）在 `:132-139` 沿 $K$ 轴缩放并预乘 `b_k`；`USE_GV` 在 `:142-149` 沿 $V$ 轴。

整个循环的计算就是这一行：

[[fla:fla/ops/common/chunk_h.py#L151-L151]]

```python
        b_h += tl.dot(b_k, b_v)
```

### tiling：grid 中不含时间维

[[fla:fla/ops/common/chunk_h.py#L336-L336]]

```python
    def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
```

program id 是 `(i_k, i_v, i_nh)`（[[fla:fla/ops/common/chunk_h.py#L65]]）。**每个 CTA 拥有 state 矩阵的一个 $[BK, BV]$ 分块，串行走完整个时间轴。** 并行度 $= N \cdot H \cdot \lceil K/BK \rceil \cdot \lceil V/BV \rceil$。

`B=1, T=8192, H=96, K=V=128, BK=BV=64` 时是 `1·96·2·2 = 384` 个 CTA——足够填满一张 GB200，所以这个「没有序列并行」的 kernel 实际上不是瓶颈。这也解释了为什么 GLA 论文的 non-materialization 版本在 batch 大时够用（[06 · linear 路线（一）：kernel trick、RNN 等价与三种计算形式](../mechanisms/06_linear_foundation.md) §3.4）。

片上驻留（`BK=BV=BT=64`）：

| 对象 | shape | 在哪 | 字节 |
|---|---|---|---|
| `b_h` | $[BK, BV]$ fp32 | **寄存器**（累加器） | 16 KiB |
| `b_k` | $[BK, BT]$ bf16 | SRAM → 寄存器 | 8 KiB |
| `b_v` | $[BT, BV]$ bf16 | SRAM → 寄存器 | 8 KiB |
| `b_g`/`b_gk`/`b_gv` | $[BT]$ 或 $[BK, BT]$ | 寄存器 | ≤ 8 KiB |

`b_k`/`b_v` 还要乘 `num_stages`（2–4），因为 Triton 会做软件流水。

### 块大小：`BT` 不参与 autotune

- **`BT` 是调用方传进来的 `chunk_size`**，默认 64（[[fla:fla/ops/common/chunk_h.py#L317]]），且必须是 2 的幂。`gla` 会自适应挑选：

[[fla:fla/ops/gla/chunk.py#L1358-L1358]]

```python
        chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))
```

- **`BK`/`BV` 参与 autotune**，候选集合取决于 SRAM 容量：

[[fla:fla/ops/common/chunk_h.py#L16-L34]]

```python
BKV_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G', 'USE_GK', 'USE_GV', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
```

共 96 个搜索点（2×2×4×3），key 挂在 `BT` 和几个门类型 flag 上。

### `split_size`：materialization 与否之间的连续插值

`:70-76` 和 `:110` 实现了一个 state-checkpointing 旋钮：`BS`（split size）是 `BT` 的倍数，状态每 `NTS = BS/BT` 个 chunk 才存一次。`split_size=None` 时 `BS == BT`、`NTS == 1`、每个 chunk 都存——就是经典的 materialization 版。设大了就用「输出 kernel 里重算」来换 HBM 占用。

> **GLA 论文的 materialization / non-materialization 二选一，在今天的 `fla` 里变成了这个连续旋钮。** 论文的 non-materialization 版本已经没有独立代码路径了，最接近的就是 `split_size` 和 `C=1` 的 `fused_recurrent`。

### varlen

[[fla:fla/ops/common/chunk_h.py#L67-L76]]

```python
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
```

`T` 被声明成 `do_not_specialize=['T']`（[[fla:fla/ops/common/chunk_h.py#L35]]），正是为了让「按序列覆写 `T`」不触发重编译。这套 `cu_seqlens` 语义与 FA 的 varlen 完全一致（[04 · FA4（CuTeDSL）与工程接口](./04_fa4_cutedsl_and_api.md) §4）。

## 3. intra-chunk 输出：`chunk_fwd_kernel_o`

[[fla:fla/ops/common/chunk_o.py#L56]]。这一步实现 chunkwise 的那一行公式（inter 为跨 chunk 项，intra 为块内二次项）：

$$
O_{[i]} = \underbrace{Q_{[i]}\, S_{[i]}}_{\text{inter}} + \underbrace{((Q_{[i]} K_{[i]}^{\top}) \odot M)\, V_{[i]}}_{\text{intra}}
$$

和 `chunk_fwd_h` 正好反过来：时间维进了 grid，状态维变成内循环。

[[fla:fla/ops/common/chunk_o.py#L80-L80]]

```python
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
```

[[fla:fla/ops/common/chunk_o.py#L566-L566]]

```python
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HV)
```

**每个 CTA 固定一个 `(chunk, V-tile, batch×head)`，所有 chunk 完全并行。** 这就是 materialization 的全部回报：Pass 1 把 `h[i]` 写进 HBM 之后，Pass 2 再没有任何跨 chunk 依赖。

### 两次 `tl.dot`，对应公式的两项

[[fla:fla/ops/common/chunk_o.py#L101-L152]]

```python
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    ...
    for i_k in range(tl.cdiv(K, BK)):
        ...
        if STATE_V_FIRST:
            b_o += tl.dot(b_q, tl.trans(b_h))
        else:
            b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)
    ...
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    ...
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
```

- `b_o += Q @ h`：inter-chunk，`h` 就是 §2 存下来的、进入本 chunk 时的 $S_{[i]}$。
- `b_A += Q @ K`：intra-chunk 的 $[BT, BT]$ attention 矩阵，只存在于寄存器中，从不落 HBM。
- `tl.where(o_t[:, None] >= o_t[None, :])`：因果下三角（含对角）。
- 最后 `b_A @ V` 把 intra 部分加进输出。

`b_A` 是 $[64, 64]$ fp32，即 16 KiB，正好是 SRAM/寄存器友好的那一档——这就是 [06 · linear 路线（一）：kernel trick、RNN 等价与三种计算形式](../mechanisms/06_linear_foundation.md) §3 把 `C=64` 定为默认值的物理理由。

### `exp(g_i − g_j)` 是差分，不是连乘

衰减在输出侧只做这一次相对修正（[[fla:fla/ops/common/chunk_o.py#L132-L142]]）：

```
b_o  *= exp2(g)                  # inter：把旧状态衰减到当前 query 的时刻
b_A  *= exp2(g[:, None] − g[None, :])   # intra：相对衰减 γ^{j→i}
```

`g` 进 kernel 之前已经被 `chunk_local_cumsum(..., scale=RCP_LN2)` 预乘过（见下）。所以 `exp2(g_i − g_j)` 用一次减法和一次 `exp2` 就给出 $\gamma^{j \to i}$——这正是 [07 · linear 路线（二）：衰减机制的演进](../mechanisms/07_linear_decay_gating.md) §3.5 那句「`exp(g_i − g_j)` 是差分」。`USE_G_GAMMA`（RetNet / Lightning 的固定逐头斜率）走同一条公式，只是 `g = g_gamma * (1..BT)` 当场构造。

### 预乘 `RCP_LN2` 的原因

[[fla:fla/ops/utils/constant.py#L10]]：`RCP_LN2 = 1.4426950216`（$= 1/\ln 2$）。恒等式：

$$
\exp(x) = \exp_2(x / \ln 2) = \exp_2(x \cdot \text{RCP\_LN2})
$$

GPU 的 SFU 上 `exp2` 是硬件指令，`exp` 要多一次换底。`simple_gla` 在进 `chunk_fwd_h/o` 之前一次性做完预乘（[[fla:fla/ops/simple_gla/chunk.py#L171-L181]]）：

[[fla:fla/ops/simple_gla/chunk.py#L171-L181]]

```python
        # Pre-scale by RCP_LN2 so downstream kernels can use exp2 directly.
        if g is not None:
            g = chunk_local_cumsum(
                g,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        if g_gamma is not None:
            g_gamma = g_gamma * RCP_LN2
```

`chunk_local_cumsum`（[[fla:fla/ops/utils/cumsum.py#L428]]）按 chunk 做对数域前缀和：chunk 内 $g_t \leftarrow \sum_{s \in \text{chunk},\, s \le t} g_s$，chunk 之间不累加——跨 chunk 的衰减由 `chunk_fwd_h` 里那次 `b_h *= exp2(g_last)` 负责。$[B, T, H]$ 走标量路径，$[B, T, H, K]$ 走向量路径（GLA / KDA 的通道级门）。

> FA 的 Triton kernel 用的是同一招：`softmax_scale` 预乘 `RCP_LN2` 之后全程 `exp2`（[01 · IO-awareness、online softmax 与 tiling](./01_io_awareness_online_softmax.md) §4）。两边独立发展出来的工程选择完全同构。

## 4. `solve_tril`：在 SRAM 内求 `(I + 严格下三角)` 的逆

delta-rule 族在 `chunk_fwd_h` 之前多两级，第一级是求

$$
A = (I + \mathrm{strictLower}(\beta\, K K^{\top} \odot \Gamma))^{-1}
$$

这就是 WY / UT 的那一步（[08 · linear 路线（三）：delta rule 与 DPLR 统一框架](../mechanisms/08_linear_delta_rule.md) §2.2）。`C=64` 时这是一个 $64 \times 64$ 下三角求逆，复杂度 $O(C^3)$，但完全放得进 SRAM。

[[fla:fla/ops/utils/solve_tril.py#L355]] 的公开 API 只收 `A: [B, T, H, BT]`，$BT \in \{16, 32, 64\}$，返回 $(I + A)^{-1}$。实现拆成两层。

### 16×16 块内：前向代入

[[fla:fla/ops/utils/solve_tril.py#L38|solve_tril_16x16_kernel]] 把对角 $16 \times 16$ 块留在寄存器里做 Gauss 前向代入：

[[fla:fla/ops/utils/solve_tril.py#L78-L86]]

```python
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
```

`for i in range(2, 16)` 是沿对角往下扫：第 `i` 行吸收已经求完的前 `i` 行。16 这个宽度不是随便挑的——再大寄存器装不下，再小 $O(C^3)$ 的常数项会变差。

### 16 到 32 到 64：分块 Schur 归并

`merge_16x16_to_32x32_inverse_kernel`（[[fla:fla/ops/utils/solve_tril.py#L108]]）和再上一档 32→64，用分块三角阵的 Schur 补：

$$
\begin{bmatrix} I + A_{00} & 0 \\ A_{10} & I + A_{11} \end{bmatrix}^{-1} = \begin{bmatrix} A_{00}^{-1} & 0 \\ -A_{11}^{-1} A_{10} A_{00}^{-1} & A_{11}^{-1} \end{bmatrix}
$$

**这个归并是精确的，不是迭代近似。** 所以 `BT=64` 的逆 = 16 个对角 $16 \times 16$ 代入 + 若干次 $16 \times 16$ GEMM，全程在 SRAM 内完成。

GDN 后来把「算 $\beta K K^{\top}$ + `solve_tril`」融进一个 kernel（[[fla:fla/ops/gated_delta_rule/chunk_fwd.py#L40]]）：把 64 再切成 4 个 `BC=16` 的 sub-chunk，10 个下三角 $[BC, BC]$ 块全部驻留寄存器，算完 $K K^{\top}$ 立刻做代入和块归并，中间的 $A$ 不再落 HBM。这是同一套 $16 \times 16$ 骨架的融合版。

## 5. WY 表示：`W = A(Γ⊙K)`，`U = A·V`

求完 `A` 之后，[[fla:fla/ops/gated_delta_rule/wy_fast.py#L255|recompute_w_u_fwd]] 做两次 GEMM（kernel [[fla:fla/ops/gated_delta_rule/wy_fast.py#L43]]）：

[[fla:fla/ops/gated_delta_rule/wy_fast.py#L82-L106]]

```python
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)
    ...
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)
```

对照 [08 · linear 路线（三）：delta rule 与 DPLR 统一框架](../mechanisms/08_linear_delta_rule.md) §2.2：$U = A(\beta \odot V)$ 是「被 Householder 修正过的新 value」，$W = A(\beta \odot \Gamma \odot K)$ 是「用来从旧状态里把将要被替换的方向读出来」的左乘向量。后面的 `chunk_delta_h` 只认 `(W, U)`，不再碰原始的 $\beta, A$。

函数名叫 `recompute_*` 不是笔误：backward 默认不保存 `W, U`，反向时用同一份 `A` 再算一遍——这又是 FA 那笔「用算力换显存」的交换。只有 `disable_recompute=True` 时才把它们 `save_for_backward`。

编排在 [[fla:fla/ops/gated_delta_rule/chunk.py#L70-L80]]：

```
g = chunk_local_cumsum(g, scale=RCP_LN2)
w, u, A = chunk_gated_delta_rule_fwd_intra(...)   # 融合 kkt + solve_tril + WY
h, v_new, ht = chunk_gated_delta_rule_fwd_h(k, w, u, g, ...)
o = chunk_fwd_o(q, k, v_new, h, g, ...)           # 复用衰减族的输出 kernel
```

`v_new` 是被 delta rule 改写过的 value（$U - WS$），所以输出侧可以继续走 §3 那个 `chunk_fwd_o`，不必为 delta 再写一份 intra-chunk。

## 6. `chunk_delta_h` 与 `blockdim64`

[[fla:fla/ops/common/chunk_delta_h.py#L58]]，`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`。名字里的 `blockdim64` 不是 chunk size，而是**把 $K$ 维拆成最多 4 条固定 64 宽的 slab，全部驻留寄存器**：

[[fla:fla/ops/common/chunk_delta_h.py#L99-L114]]

```python
    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        ...
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
```

衰减族的 `chunk_fwd_h` 可以把 `(BK, BV)` 切给不同 CTA（grid 含 `⌈K/BK⌉`），但 delta 不行：$\tilde{V} = U - WS$ 这一步要沿整个 $d_k$ 维缩并——$W$ 的一行要和完整的 $S$ 做点积，才能读出「这个 key 当前映射到什么」。$K$ 维不能再切给别的 CTA，否则每个 CTA 只看见 $S$ 的一条带，点积是错的。

这就是 [08 · linear 路线（三）：delta rule 与 DPLR 统一框架](../mechanisms/08_linear_delta_rule.md) §2.5 说的「state-to-state dependencies」的硬件后果：`K ≤ 256` 被写进 assert（[[fla:fla/ops/common/chunk_delta_h.py#L712]]），`K=128` 时同时活着 `b_h1 + b_h2` 两条 64 宽 slab。

### 循环体：先读后写，和衰减族同一不变量

精简自 [[fla:fla/ops/common/chunk_delta_h.py#L184-L294]]：

```python
tl.store(p_h1, b_h1, mask=m_h1)            # ① store-before-update
b_v = tl.dot(b_w, b_h1)                    # ② W @ S  → 读出旧值
b_v = tl.load(p_v) - b_v                   # ③ U − W S（kernel 的 v 其实是 u）
b_h1 *= b_g_last                           # ④ 先衰减旧状态
b_h1 += tl.dot(b_k, b_v)                   # ⑤ 再写入残差
```

wrapper 把 `u` 绑到 kernel 的 `v` 参数上（[[fla:fla/ops/common/chunk_delta_h.py#L725]]）。逐步对齐 [08 · linear 路线（三）：delta rule 与 DPLR 统一框架](../mechanisms/08_linear_delta_rule.md) §1.3 的「读出 → 残差 → 写入」：

```
v_old = Wᵀ S          # 这个 key 现在映射到什么
S     ← α · S
S     ← S + Kᵀ (U − v_old)
```

`USE_GK`（KDA 的通道级衰减）在 `:259-286` 沿 $K$ 轴对每条 slab 分别 `*= exp2(gk_last)`——这是 GDN 的标量门换成 KDA 的 $\mathrm{Diag}(\alpha)$ 时唯一多出来的那几行。

grid 退化成 $(\lceil V/BV \rceil \cdot N \cdot HV,)$（[[fla:fla/ops/common/chunk_delta_h.py#L722]]）：并行度只剩 $V$ 维和 batch×head，$K$ 维被吃进寄存器。这是 delta 家族相对衰减家族必须付出的固定开销。

## 7. backward：时间倒序扫描与 `dg` 的闭式

和 FA 一样：forward 不存中间量，backward 重算。差别是重算的对象从 $S/P$ 换成了 chunk states。

### 状态梯度倒序扫描

[[fla:fla/ops/common/chunk_h.py#L242|chunk_bwd_kernel_dh]] 的时间循环是

[[fla:fla/ops/common/chunk_h.py#L242-L242]]

```python
    for i_t in range(NT - 1, -1, -1):
```

前向 $S_{i+1} = \mathrm{Diag}(\gamma)\, S_i + K_i^{\top} V_i$ 的伴随是后缀和：$dS_i$ 从序列末尾往回传，每步先乘衰减再加本 chunk 的 $dO$ 贡献。这就是 [06 · linear 路线（一）：kernel trick、RNN 等价与三种计算形式](../mechanisms/06_linear_foundation.md) §1 说的「前向是前缀和、对 $K$/$V$ 的反向是后缀和」——和 FA backward「固定 KV 块、循环 Q 块」是同一种方向反转。

### `dg` 不必物化 `L×d×d`

GLA 论文给出 $d\log\alpha$ 的闭式（[07 · linear 路线（二）：衰减机制的演进](../mechanisms/07_linear_decay_gating.md) §4.5；第二行是后缀和）：

$$
\begin{aligned}
d\log b_t &= q_t \odot dq_t - k_t \odot dk_t \\
d\log \alpha_t &= \sum_{t \le i \le L} d\log b_i
\end{aligned}
$$

`chunk_bwd_kernel_dqkwg` 里就是第一行（[[fla:fla/ops/common/chunk_o.py#L314]]）：

[[fla:fla/ops/common/chunk_o.py#L314-L322]]

```python
        b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)

        p_dg = dg + o_t * HV
        # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
        b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_qk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_qk)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_t)
```

$q \odot dq - k \odot dk$ 在输出 kernel 的反向里顺带算完，chunk 末尾那个 `b_dg_last` 接住来自后续 chunk 的 $\langle h, dh \rangle$（状态对衰减的贡献）。真正的 reverse-cumsum 被拆到另一个 kernel——注释写明是 Triton 编译器的问题，不是算法需要。

`save_for_backward` 存 `q, k, v, g, h`（以及 delta 族的 `A`），不存 `b_A` 和逐步的 $S_t$。chunk states `h` 若开了 `split_size` 还可以更稀，反向时在输出 kernel 里从最近的 checkpoint 重算。

## 8. `fused_recurrent`：decode 对应 `C = 1`

训练和 prefill 走 chunkwise；decode 的 `seqlen_q` 往往不超过 64，再切 chunk 只会白白付出启动开销。layer 里的切换是硬编码的（[[fla:fla/layers/kda.py#L212]]）：

```
mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
```

[[fla:fla/ops/common/fused_recurrent.py#L30]] 是字面意义上的逐步 RNN：

[[fla:fla/ops/common/fused_recurrent.py#L103-L130]]

```python
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        ...
        if STATE_V_FIRST:
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], axis=1)
        else:
            b_h += b_k[:, None] * b_v[None, :]
            b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        p_q += (-1 if REVERSE else 1) * H*K
        ...
```

和 `chunk_fwd_h` 的对照：

| | `chunk_fwd_h` | `fused_recurrent` |
|---|---|---|
| 时间步 | 一个 chunk（`BT=64`） | **一个 token** |
| 状态物化 | 每 chunk / 每 `split_size` 写 HBM | **不写中间状态**，只在结束时可选写 `ht` |
| 输出 | 另起 `chunk_fwd_o` | **融合在同一循环**：先更新 $S$ 再 $o = S^{\top} q$ |
| 并行 | $N \cdot H \cdot \lceil K/BK \rceil \cdot \lceil V/BV \rceil$ | 同左，但 $T$ 必须串行 |
| 指针 | 每步重算 offset | 指针 `+= ± stride`，`REVERSE` 给 backward 用 |

`lightning_attn` / `retention` / `simple_gla` 的 decode 全部走这条共享路径，只是 `USE_G` / `USE_G_GAMMA` 不同。KDA 有自己的 `fused_recurrent`（[[fla:fla/ops/kda/fused_recurrent.py]]），因为多了 $\beta$ 和 $\mathrm{Diag}(\alpha)$，但循环骨架一样。

这就是 [06 · linear 路线（一）：kernel trick、RNN 等价与三种计算形式](../mechanisms/06_linear_foundation.md) §3.3 那张三形式表的生产落地：`C=64` 训练，`C=1` 推理。

## 9. 族谱、开关与速度

[[fla:fla/ops/]] 下大约 40 个目录，真正的 Triton 源集中在 `common/` + `utils/`。族与族之间靠 `tl.constexpr` 开关区分，编译期消掉，运行时零分支：

| flag | 语义 | 谁用 |
|---|---|---|
| `USE_G` | 头级标量衰减 `g: [B,T,H]` | simple_gla、GDN、Mamba-2 |
| `USE_G_GAMMA` | 数据无关的逐头斜率 `g_gamma: [H]` | RetNet、Lightning |
| `USE_GK` | 通道级衰减 `gk: [B,T,H,K]` | **GLA、KDA** |
| `USE_GV` | 沿 V 轴的衰减 | 少数跨 V 门控变体 |
| `STATE_V_FIRST` | state 存 $[V, K]$ 还是 $[K, V]$ | FlashKDA 强制 `True`（CUTLASS 布局） |
| `IS_VARLEN` | `cu_seqlens` 覆盖 `T` | 全族 |

再往上叠的是「要不要 delta」：衰减族走 `chunk_h` + `chunk_o`；delta 族在前面插 `solve_tril` + `wy_fast`，状态更新换 `chunk_delta_h`，输出仍复用 `chunk_o`。

### 通道级门控的开销：二级分块

GLA / KDA 的 `USE_GK` 让 intra-chunk 的 $A_{ij,d} = \exp(g_{id} - g_{jd})$ 带上 $d$ 下标，不能从 matmul 中提出来（[07 · linear 路线（二）：衰减机制的演进](../mechanisms/07_linear_decay_gating.md) §4.2）。`fla` 的对策是把 `BT=64` 再切成 `BC=16` 的 sub-chunk，对应三个 kernel（[[fla:fla/ops/gla/chunk.py]]）：

| kernel | 干什么 | tensor core？ |
|---|---|---|
| `chunk_gla_fwd_A_kernel_intra_sub_inter`（[[fla:fla/ops/gla/chunk.py#L53]]） | sub-chunk 对 $i > j$，半精度 matmul | ✓ |
| `..._intra_sub_intra`（[[fla:fla/ops/gla/chunk.py#L134]]） | **对角 sub-chunk**，逐位置对、log 空间 fp32 | ✗ |
| `..._intra_sub_intra_split/merge`（[[fla:fla/ops/gla/chunk.py#L209]]） | $K$ 太大时再切一刀 | 对角仍 ✗ |

`chunk_simple_gla` / `chunk_retention` 没有这条路径：标量 $g$ 可以整块乘到 $QK^{\top}$ 上，一次 `tl.dot` 结束。这就是通道级门控带来的额外开销——`chunk_gla` 相对它们慢 1.4–2.3×（`fla` 在 GB200 上的对照；量级随 $T, H, K$ 浮动，但方向稳定）。KDA 用 `a=b=k` 把二级 chunk 矩阵从 4 个减到 2 个（[09 · linear 路线（四）：KDA 与 Kimi Linear / K3](../mechanisms/09_linear_kda_kimi.md) §3）；Kimi K3 再用 `g_min = −5` 把对角路径也拉回 tensor core。

### 短序列上 FA 仍然更快

linear 的渐近优势在 $L \gg d$ 时才显现。`T ≲ 2K` 时 FA2/FA3 的 Tensor Core 利用率更高，chunkwise 还要付状态扫描和 $C \times C$ 求逆的固定成本。这是 [11 · 混合模式：层比例、NoPE 与正反证据](../mechanisms/11_hybrid.md) §5 MiniMax 回退全attention时引用的那条工程事实：理论交叉点在几百 token，**实测交叉点被 memory-bound 推到几千**。hybrid 把全attention层留在关键位置，部分原因就是不想在短上下文上交这份额外的税。

| 场景 | 该调谁 |
|---|---|
| 训练 / 长 prefill，衰减族 | `chunk_*`（`simple_gla` / `retention` / `gla`） |
| 训练 / 长 prefill，delta 族 | `chunk_gated_delta_rule` / `chunk_kda` |
| decode / `q_len ≤ 64` | `fused_recurrent_*` |
| 短序列质量优先 | **FA2/3/4**，不要为了线性而线性 |

## 10. KDA / FlashKDA

KDA 的数学在 [09 · linear 路线（四）：KDA 与 Kimi Linear / K3](../mechanisms/09_linear_kda_kimi.md)。kernel 侧它是 GDN 路径的一次特化，不是新的递归。

[[fla:fla/ops/kda/chunk_fwd.py#L20]] 的编排：

```
g = kda_gate_chunk_cumsum(...) or chunk_local_cumsum(..., scale=RCP_LN2)
w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(...)     # 融合 inter + solve_tril
h, v_new, ht = chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g, ...)
o = chunk_gla_fwd_o_gk(q, v_new, g, Aqk, h, ...)      # 走 GLA 的通道级输出
```

和 GDN 的三处不同：

1. **门是通道级且可带下界。** `kda_gate_chunk_cumsum`（[[fla:fla/ops/kda/gate.py]]）把 `A_log`、`dt_bias`、softplus 融进 cumsum。Kimi K3 的 `lower_bound=-5` 就在这里（[[fla:fla/ops/kda/gate.py#L62]] 的默认值）：$g = -5 \cdot \sigma(e^{A} \cdot x)$，累积衰减有下界，$1/\Gamma$ 不再爆炸，对角 sub-chunk 可以重新走半精度 matmul。
2. **intra 只产 `Aqk, Akk` 两份**（[[fla:fla/ops/kda/chunk_intra.py#L43]] 的 fused inter+solve），通用 DPLR 要四份。这就是 `a=b=k` 绑定带来约 2× 收益的来源。
3. **输出复用 GLA 的 `chunk_gla_fwd_o_gk`**，因为衰减带着 $K$ 维。

### FlashKDA 推理后端

[[fla:fla/ops/kda/backends/flash_kda.py]] 是 Moonshot 的推理后端（[FlashKDA](https://github.com/MoonshotAI/FlashKDA)），`priority=3`，`FLA_FLASH_KDA` 控制开关。它把推理期的多个 kernel 收成一次 CUTLASS 调用。verifier（[[fla:fla/ops/kda/backends/flash_kda.py#L41]]）把路径收得很窄：

- 只允许 `torch.inference_mode()`，训练仍走 Triton
- `bf16`，`K=V=128`，不允许 GVA（`HV == H`）
- 必须 `use_gate_in_kernel` + `use_qk_l2norm_in_kernel` + `use_beta_sigmoid_in_kernel`——L2Norm / σ(β) / gate 全部融进 kernel，调用方传裸张量
- 必须 `state_v_first=True`、`safe_gate=True`
- 不支持 CP（`cp_context is not None` 直接拒绝）

`fla` 自己的 Triton 路径仍然是可微、可 varlen、可 CP 的参考实现；FlashKDA 是「生产 decode / prefill 把启动开销和中间 HBM 往返砍掉」的特化。KDA 的 Context Parallelism 在 [[fla:fla/ops/cp/]]，按序列维切 chunk、用 `chunk_gated_delta_rule_fwd_h_pre_process` 做跨 rank 的状态传递（[[fla:fla/ops/kda/chunk_fwd.py#L81]]）——state 固定大小，所以 prefix caching 和 CP 都比 softmax KV 便宜。

---

## 11. 小结

| 不变量 | FlashAttention | Flash Linear Attention |
|---|---|---|
| 不物化 | $[N, N]$ 的 $S/P$ | 逐步 $[d_k, d_v]$ state |
| tiling | Q 块固定，扫 KV | state 的 $[BK, BV]$ 固定，扫时间 |
| 数值 | online softmax 的 $m, l, O$ | store-before-update 的 $S_{[i]}$ + 块内 $A$ |
| 衰减 / 稳定 | 减 row-max；`exp2` | 对数域 cumsum；预乘 `RCP_LN2`；`exp2` |
| backward | 重算 $P$，存 LSE | 重算 chunk states，`dg` 走闭式 |
| 额外复杂度 | 无（块之间可交换） | **跨 chunk 串行**；delta 还不能切 $K$ |

读任何一份 `fla` kernel，先找 `b_h` 的 store 点，再找 `exp2(g_i − g_j)`，再看 grid 里有没有时间维——三件事对上了，公式就对上了。

下一篇：回到 [Attention 机制](../mechanisms/README.md) 看三条路线的开销对比；稀疏侧的 flash 变体在上一篇 [05 · Flash Sparse Attention](./05_flash_sparse_attention.md)。lab 仍覆盖 softmax FA 的 online softmax 不变量：[[atlas:docs/attention/fa/fa_lab.ipynb]]。
