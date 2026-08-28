# 09 · linear 路线（四）：KDA 与 Kimi Linear / K3

> KDA 在数学上只比 Gated DeltaNet 改了一处：把标量 $\alpha_t$ 换成向量 $\mathrm{Diag}(\alpha_t)$。真正的贡献在工程上——把这个改动带来的算子开销从通用 DPLR 的水平压回 delta rule 的水平，办法是把 DPLR 的 $a$ 和 $b$ 都绑定到 $k$。这条从表达力需求到数值精度问题再到算子设计的因果链，是纯算法论文里看不到的东西，也是本篇的重点。
>
> **前置**：[`07`](./07_linear_decay_gating.md) §4（GLA 的通道级门、$1/\Gamma$ 溢出与二级分块）、[`08`](./08_linear_delta_rule.md)（delta rule、WY/UT、DPLR 统一）。统一记号见 [Attention 机制](./README.md) §2。
>
> 论文：Kimi Linear [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)、Kimi K3 [arXiv:2607.24653](https://arxiv.org/abs/2607.24653)。代码：[[fla:fla/ops/kda/]]、[[fla:fla/layers/kda.py]]。

---

## 1. KDA 的递归

原文 Eq. 1（本章列向量约定，与论文一致）：

$$
\begin{aligned}
S_t &= \underbrace{( I - \beta_t k_t k_t^{\top} )}_{\text{delta: exact replacement}}\ \cdot\ \underbrace{\mathrm{Diag}(\alpha_t)}_{\text{channel-wise decay}}\ \cdot\ S_{t-1} + \beta_t k_t v_t^{\top} \\
o_t &= S_t^{\top} q_t, \qquad S_t \in \mathbb{R}^{d_k \times d_v}, \qquad \alpha_t \in (0,1)^{d_k}
\end{aligned}
$$

配置中 $d_k = d_v = 128$，即每 head 有 128×128 = 16384 个状态数。

与 GDN 的唯一差别是 $\alpha_t$ 从标量变成 $\mathrm{Diag}(\alpha_t)$。原文的动机：

> "While GDN, similar to Mamba2, employs a **coarse head-wise forget gate**, KDA introduces a **channel-wise variant in which each feature dimension maintains an independent forgetting rate**, akin to Gated Linear Attention (GLA). This fine-grained design enables **more precise regulation of the finite-state RNN memory**."

⚠️ 乘序在这里变得关键。KDA 是先衰减、后 delta（$(I - \beta k k^{\top})$ 在最左）。因为 $\mathrm{Diag}(\alpha_t)$ 不是标量，与 Householder 项不可交换；GDN 的 $\alpha_t$ 是标量，所以没有这个问题。

```python
def kda_recurrent(q, k, v, g, beta, scale):
    """g: [B,T,H,K] 对数【通道级】衰减；beta: [B,T,H]。decode 路径。"""
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]); o = torch.zeros_like(v)
    for t in range(T):
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        S = S * g[:, t].exp()[..., None]                          # Diag(α_t) S   ← 沿 K 维（行）
        v_old = torch.einsum('bhk,bhkv->bhv', kt, S)              # 在【已衰减】的状态上读
        S = S + bt[..., None, None] * kt.unsqueeze(-1) * (vt - v_old).unsqueeze(-2)
        o[:, t] = torch.einsum('bhk,bhkv->bhv', q[:, t] * scale, S)
    return o

# 与 fla/ops/kda/naive.py:12 的 naive_recurrent_kda 交叉验证（fp32）：rel_err = 6.8e-7 ✓
```

## 2. Chunkwise 算法

对应原文 §3.1，共五步。记号：$\gamma^{i \to j} := \prod_{k=i}^{j} \alpha^k$，简写 $\gamma^r := \gamma^{1 \to r}$；$\mathcal{A}_{ij} = \gamma^i / \gamma^j \in \mathbb{R}^{C \times C}$；$\Gamma^{i \to j} \in \mathbb{R}^{C \times d_k}$ 是把 $\gamma^i \dots \gamma^j$ 按行堆叠。

**Step 1 — 部分展开（Eq. 2）**：

$$
\begin{aligned}
S_{[t]}^r &= P_{[t]}^r\, S_{[t]}^0 + H_{[t]}^r \\
P_{[t]}^r &= \prod_{i \le r} (I - \beta^i k^i (k^i)^{\top})\, \mathrm{Diag}(\alpha^i) \\
H_{[t]}^r &= \sum_{i \le r} \Big( \prod_{j=i+1}^{r} (I - \beta^j k^j (k^j)^{\top})\, \mathrm{Diag}(\alpha^j) \Big)\, \beta^i k^i (v^i)^{\top}
\end{aligned}
$$

**Step 2 — WY 表示（Eq. 3）**，遵循 Comba 的 `P` 形式，以省掉一次额外矩阵求逆：

$$
\begin{aligned}
P_{[t]}^r &= \mathrm{Diag}(\gamma^r) - \sum_{i \le r} \mathrm{Diag}(\gamma^{i \to r})\, k^i (w^i)^{\top} \\
H_{[t]}^r &= \sum_{i \le r} \mathrm{Diag}(\gamma^{i \to r})\, k^i (u^i)^{\top}
\end{aligned}
$$

**Step 3 — UT transform（Eq. 6–7）**：

$$
\begin{aligned}
M_{[t]} &= [ I + \mathrm{StrictTril}( \mathrm{Diag}(\beta) \cdot (\Gamma^{1 \to C} \odot K) \cdot (K / \Gamma^{1 \to C})^{\top} ) ]^{-1} \mathrm{Diag}(\beta) \\
W_{[t]} &= M_{[t]}\, (\Gamma^{1 \to C} \odot K_{[t]}), \qquad U_{[t]} = M_{[t]}\, V_{[t]}
\end{aligned}
$$

注意其中的 $K / \Gamma^{1 \to C}$，即除以累积衰减。这就是 [`07`](./07_linear_decay_gating.md) §4.2 里 GLA 的同一个数值问题，KDA 继承了它。§7 会讲 Kimi K3 如何彻底解决。

**Step 4 — 状态更新（Eq. 8）**：

$$
S_{[t+1]} = \mathrm{Diag}(\gamma^C)\, S_{[t]} + (\Gamma^{i \to C} \odot K_{[t]})^{\top}\, (U_{[t]} - W_{[t]} S_{[t]})
$$

**Step 5 — 输出（Eq. 9）**：

$$
O_{[t]} = \underbrace{(\Gamma^{1 \to C} \odot Q_{[t]})\, S_{[t]}}_{\text{inter-chunk}} \;+\; \underbrace{\mathrm{Tril}\big( (\Gamma^{1 \to C} \odot Q_{[t]})\, (K_{[t]} / \Gamma^{1 \to C})^{\top} \big)}_{\text{intra-chunk}}\; \underbrace{(U_{[t]} - W_{[t]} S_{[t]})}_{\text{pseudo value}}
$$

> **$\mathrm{Tril}$ 含对角线**，这一点 K3 论文明确解释了原因："the diagonal is retained because **each output reads the state after the current-token update**." —— 因为 $o_t = S_t^{\top} q_t$ 用的是更新后的 $S_t$，所以 query $t$ 要能看到 key $t$ 自己。

### 完整实现

```python
def kda_chunk(q, k, v, g, beta, scale, C):
    """g: [B,T,H,K] 对数通道级衰减。与 kda_recurrent 数学恒等。"""
    B, T, H, K = q.shape
    V, NT = v.shape[-1], T // C
    qc, kc, vc = (x.transpose(1, 2).reshape(B, H, NT, C, -1) for x in (q, k, v))
    bc = beta.transpose(1, 2).reshape(B, H, NT, C)
    gc = g.transpose(1, 2).reshape(B, H, NT, C, K).cumsum(-2)          # 局部 cumsum
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).triu(0)
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()   # ← 含对角

    def pair(x, y, gx):
        """A_ij = Σ_d x_id y_jd exp(gx_id − gx_jd)。通道级衰减【无法】提出 matmul，
        所以这里必须逐 j 循环 —— 这就是 GLA/KDA「二级分块」在数学上的根源。"""
        out = torch.zeros(B, H, NT, C, C, dtype=q.dtype, device=q.device)
        for j in range(C):
            out[..., j] = ((x * (gx - gx[..., j:j+1, :]).exp()) * y[..., j:j+1, :]).sum(-1)
        return out

    Akk  = pair(kc, kc, gc) * bc[..., None]                            # (Γ⊙K)(K/Γ)^T · diag(β)
    Amat = -Akk.masked_fill(strict, 0)
    for i in range(1, C):                                              # 前向代入求逆
        Amat[..., i, :i] = Amat[..., i, :i] + (Amat[..., i, :, None] * Amat[..., :, :i]).sum(-2)
    Amat = Amat + torch.eye(C, dtype=q.dtype, device=q.device)
    Tmat = Amat * bc[..., None, :]
    W, U = Tmat @ (kc * gc.exp()), Tmat @ vc                           # Eq. 7
    Aqk = pair(qc * scale, kc, gc) * causal                            # Eq. 9 的 intra 项

    S = q.new_zeros(B, H, K, V); o = torch.zeros_like(vc)
    for n in range(NT):
        qi, ki, gi = qc[:, :, n] * scale, kc[:, :, n], gc[:, :, n]
        v_tilde = U[:, :, n] - W[:, :, n] @ S                          # pseudo value
        o[:, :, n] = (qi * gi.exp()) @ S + Aqk[:, :, n] @ v_tilde      # Eq. 9
        g_last = gi[:, :, -1:]
        S = S * g_last.transpose(-1, -2).exp() + \
            (ki * (g_last - gi).exp()).transpose(-1, -2) @ v_tilde     # Eq. 8
    return o.reshape(B, H, T, V).transpose(1, 2)

# 实测（float64, C=16）：
#   rel_err(kda_chunk, kda_recurrent)                     = 2.4e-15
#   vs fla/ops/kda/naive.py:12  naive_recurrent_kda (fp32) = 6.8e-7
#   vs fla/ops/kda/naive.py:69  naive_chunk_kda     (fp32) = 2.5e-6    ✓
```

把这段代码和 [`08`](./08_linear_delta_rule.md) §3 的 `gdn_chunk` 并排看，差别只有两处：第一，`gc` 多了一个 `K` 维（`cumsum(-2)` 而不是 `cumsum(-1)`）；第二，$\Gamma \odot K K^{\top}$ 那个 $[C, C]$ 矩阵变成了 `pair()` 里的循环。**第二处就是全部的性能代价。**

## 3. `a = b = k` 绑定与算子加速

这是 KDA 相对 GLA / GDN / RWKV-7 唯一真正的算法创新，它带来了约 2 倍的算子加速。

### 3.1 KDA 是 DPLR 的一个受限变体

把 Eq. 1 重写：

$$
S_t = (\mathrm{Diag}(\alpha_t) - \beta_t k_t k_t^{\top}\, \mathrm{Diag}(\alpha_t))\, S_{t-1} + \beta_t k_t v_t^{\top}
$$

对照 [`08`](./08_linear_delta_rule.md) §5 的通用 DPLR $S_t = (D - a_t b_t^{\top})\, S_{t-1} + k_t v_t^{\top}$：

$$
D = \mathrm{Diag}(\alpha_t), \qquad a_t = \beta_t k_t, \qquad b_t = k_t \odot \alpha_t
$$

**核心技巧是把 $a$ 和 $b$ 都绑定到 $k$**（原文 "By binding both variables `a` and `b` to `k`"）。因为 $\alpha_t$ 被两者共享，可以像 Eq. 1 那样把它提出来，变成「GLA 式细粒度乘性衰减 + DeltaNet 式 Householder 变换」的干净两段式，而不是一个纠缠在一起的低秩项。

（[`08`](./08_linear_delta_rule.md) §5 的 DPLR 归约表实测验证了这个对应关系，`rel_err = 5.2e-16`。）

### 3.2 收益：两处算子削减

问题根源：Eq. 9 的 intra-chunk 项含 $K / \Gamma^{1 \to C}$，即除以累积衰减的倒数，会数值溢出。GLA 的解法是 log 域加二级分块（16 token 一个 sub-chunk）全精度，代价是无法使用半精度 matmul（[`07`](./07_linear_decay_gating.md) §4.4）。

| 改进点 | 通用 DPLR | **KDA** | 收益 |
|---|---|---|---|
| 二级 chunk 矩阵个数 | 需要 4 个：$A_{ab}, A_{ak}, A_{qb}, A_{qk}$ | **只需 2 个：$A_{qk}, A_{kk}$** | "removes the need for **two** secondary chunking steps" |
| inter-chunk / output 阶段 | $o_1 = A_{qk} v$；$o_2 = A_{qb}(u + wS)$；$o_3 = (q \odot g)\, S$；状态更新 **两次** `S += …` | $o = (q \odot g)\, S + A_{qk}(u - wS)$；状态更新 **一次** | 额外消除约 **3 次矩阵乘法** |

原文结论：

> "As a result, the operator efficiency of KDA improves by roughly **100% compared to the DPLR formulation**." 以及 "we further benchmark the kernel speed in Fig. 2, showing that KDA achieves nearly **2× the speed of DPLR** for sequence lengths up to 64k."

在上面的 `kda_chunk` 代码里能直接看到这个收益：`pair()` 只被调用两次（`Akk` 和 `Aqk`），而通用 DPLR 需要四次；内层循环里 `o[:, :, n]` 只有一行、`S` 只更新一次。如果 $a$ 和 $b$ 是自由的，就必须额外计算 `Aab`、`Aak`、`Aqb`，并把状态更新拆成两项。

而且 KDA "remains more consistent with the classical delta rule"——相对通用 DPLR 还有理论上的额外优势（回到 [`08`](./08_linear_delta_rule.md) §1.3 那个「一步 SGD」的解释）。

### 3.3 与 RWKV-7 的对比

Kimi Linear Table 6 的两行（并行形式，省略归一化项与 $\beta_t$）只差一列：

```
RWKV-7:  ( (Q⊙Γ)(K/Γ)ᵀ ⊙ M ) · ( I + (K̂⊙Γ^{0→t−1})( (K̃⊙B)/Γ )ᵀ ⊙ M⁻ )^{-1} V
                                        └─ K̂ 与 K̃⊙B 是【两个不同的张量】─┘
KDA:     ( (Q⊙Γ)(K/Γ)ᵀ ⊙ M ) · ( I + (K⊙Γ)(K/Γ)ᵀ ⊙ M⁻ )^{-1} V
                                        └─ K 出现【两次】，同一个张量 ─┘
```

这就是「绑定 $a = b = k$」在并行形式上的直观体现，也是省掉两次二级分块的根源。RWKV-7 换来的是「移除键与写入键可以不同」的表达力（[`08`](./08_linear_delta_rule.md) §4），KDA 换来的是 2 倍的算子速度。这是一个明确的取舍点。

## 4. KDA 作为可学习的位置编码

这是 Kimi Linear 中最有洞察力的一节（§6.1），也是「为什么 MLA 层可以用 NoPE」的全部理由。

RoPE 的广义形式：

$$
s_{t,i} = q_t^{\top} \Big( \prod_{j=i+1}^{t} R_j \Big)\, k_i
$$

$R_j$ 是块对角旋转矩阵。由正交性 $R_{t-i} = R_t^{\top} R_i$，绝对位置可分别施加于 $q, k$ 并自动转成相对位置（[`02`](./02_position_and_stability.md) §2）。

gated delta rule 有完全对应的形式（Eq. 12）：

$$
o_t = \sum_{i \le t} \Big( q_t^{\top}\, \underbrace{\Big( \prod_{j=i+1}^{t} A_j\, (I - \beta_j k_j k_j^{\top}) \Big)}_{\text{same structure as RoPE's } \prod R_j}\, k_i \Big)\, v_i
$$

因此，**GDN/KDA 是一种「数据相关、可学习的乘性位置编码」**，它放松了 RoPE 的正交性约束，所以 "can be potentially more powerful"。

把本章几处伏笔串起来，完整的逻辑链如下：

```
Mamba-2（[07] §3.2）:   "L 是一个 data-dependent positional mask，替换了 heuristic 位置编码"
RoPE（[02] §2）:        正交旋转 ⇒ 严格相对；自带弱 long-term decay
KDA（本节）:             Π A_j (I − βkkᵀ) 与 Π R_j 同构，但数据相关、非正交 ⇒ 更强
                                    ↓
Kimi Linear 的决定:     既然 KDA 层已经是「主要的 position-aware 算子」，
                        MLA 层就不需要位置编码 ⇒ NoPE
                                    ↓
两个附带收益:            ① MLA + NoPE 在推理时退化成【纯 MQA】（[01] §4.5）
                        ② 长上下文训练不需要 RoPE 调参（frequency base / YaRN）
```

原文：

> "In Kimi Linear, we apply **NoPE to all full attention (MLA) layers**. This design **delegates the entire responsibility for encoding positional information and recency bias to the KDA layers**. KDA is thus established as the **primary position-aware operator**."
>
> "First, **NoPE enables their conversion to the highly-efficient pure Multi-Query Attention (MQA) during inference.** Second, it **simplifies long-context training, as it obviates the need for RoPE parameter adjustments, such as frequency base tuning or methods like YaRN**."

### 消融实验

消融证明这不是纸上推理。Table 5（128k 长上下文）：

| 模型 | RULER | MRCR | HELMET-ICL | RepoQA | Avg. |
|---|---|---|---|---|---|
| MLA | 81.3 | 22.6 | 88.0 | 63.0 | 52.2 |
| GDN-H | 80.5 | 23.9 | 85.5 | 63.0 | 51.2 |
| Kimi Linear (**RoPE**) | 78.8 | 22.0 | 88.0 | 66.5 | 51.8 |
| **Kimi Linear (NoPE)** | **84.3** | **29.6** | **90.0** | **68.5** | **54.5** |

NoPE 与 RoPE 在 RULER 上是 84.3 对 78.8，相差 5.5 分，是长上下文的决定性因素。原文的解释很有说服力：

> "In Kimi Linear (RoPE), the global attention layer carries a strong, explicit relative positional signal, while the linear attention contributes a weaker, implicit positional inductive bias. This **mismatch yields an overemphasis on short-range order in the global layer**, which benefits short contexts but makes the model **less flexible when adapting mid-training to extended contexts**."

两个位置信号互相冲突了。短上下文时 RoPE 的强信号有帮助（表里 RoPE 版短上下文分数相当），但要扩到 128K 时它成了束缚。

同方向的工作：Falcon-H 用极高 base（`b ≈ 10^11`）把位置编码推到 near-NoPE 状态；SwanGPT 交替 RoPE 层与 NoPE 全 attention 层（[`02`](./02_position_and_stability.md) §4 表格最后两行）。

## 5. 神经参数化与输出门

对应原文 §4。

$$
\begin{aligned}
q_t^h,\ k_t^h &= \mathrm{L2Norm}\big( \mathrm{Swish}( \mathrm{ShortConv}( W_{q/k}^h\, x_t ) ) \big) &&\in \mathbb{R}^{d_k} \\
v_t^h &= \mathrm{Swish}\big( \mathrm{ShortConv}( W_v^h\, x_t ) \big) &&\in \mathbb{R}^{d_v} \\
\alpha_t^h &= f( W_\alpha^{\uparrow} W_\alpha^{\downarrow}\, x_t ) &&\in [0,1]^{d_k} \quad \text{(low-rank)} \\
\beta_t^h &= \mathrm{Sigmoid}( W_\beta^h\, x_t ) &&\in [0,1]
\end{aligned}
$$

- **$d_k = d_v = 128$**（全部实验）
- $\alpha$ 走**低秩投影**（rank = head dim），衰减函数 $f$ 与 GDN/Mamba 相同
- **L2Norm 施于 $q, k$**：保证特征值稳定性（对照 [`08`](./08_linear_delta_rule.md) §1.2 的特征值分析——$\|k\|$ 影响 Householder 项的谱）
- ShortConv kernel size = **4**

**输出门（Eq. 10）**：

$$
o_t = W_o\, \big( \mathrm{Sigmoid}(W_g^{\uparrow} W_g^{\downarrow} x_t) \odot \mathrm{RMSNorm}(\mathrm{KDA}(q_t, k_t, v_t, \alpha_t, \beta_t)) \big)
$$

（顺序：KDA → 逐头 RMSNorm → ⊙ sigmoid 门 → $W_o$。）

即 `FusedRMSNormGated` 的语义。层的连线在 [[fla:fla/layers/kda.py#L248-L307]] 能逐行对上。

消融（Table 1）：「sigmoid 而非 swish」是一个实测结论，不是偏好问题。

| 配置 | Training PPL ↓ | Validation PPL ↓ |
|---|---|---|
| **默认（Sigmoid 输出门，3:1）** | **9.23** | **5.65** |
| w/o output gate | 9.25 | 5.67 |
| w/ **swish** output gate（GDN 的选择） | 9.43 | **5.81** |
| w/o convolution layer | 9.29 | 5.70 |
| hybrid **0:1**（纯 MLA） | 9.45 | 5.77 |
| hybrid 1:1 | 9.29 | 5.66 |
| hybrid **3:1** | **9.23** | **5.65** |
| hybrid 7:1 | 9.23 | 5.70 |
| hybrid 15:1 | 9.34 | 5.82 |

Sigmoid 门明显优于 Swish 门（5.65 对 5.81），与 Gated Attention 论文（[arXiv:2505.06708](https://arxiv.org/abs/2505.06708)）的结论一致。GDN 用的是 SiLU/swish 门，Kimi 明确换成了 Sigmoid，并把 GDN-H baseline 也统一改成 Sigmoid 以求公平。输出门用低秩参数化以保证参数量公平对比，且明确目的是 "**alleviating the Attention Sink**"（引 Gated Attention 论文）——详见 [`10`](./10_gating.md) §3。

## 6. Kimi Linear 的配置

| 项 | 值 |
|---|---|
| 层数 | **27**（= **20 KDA + 7 MLA**） |
| `full_attn_layers` | `[4, 8, 12, 16, 20, 24, 27]` ⇒ 模式 `[K,K,K,M]×6 + [K,K,M]`，**末层是 MLA** |
| hidden | 2304 |
| KDA | 32 头 × head_dim 128，`short_conv_kernel_size = 4` |
| MLA | `mla_use_nope: true`，`kv_lora_rank: 512`，`qk_nope_head_dim: 128`，`qk_rope_head_dim: 64` |
| MoE | 256 routed + 1 shared，top-8；`moe_intermediate_size: 1024`；第 1 层 dense |
| 参数 | **48B total / 3B active** |
| 上下文 | **1,048,576（1M）** |

⚠️ 实际是 27 层而不是精确的 3:1（20.25 : 6.75），且末层是 MLA——保证最后一层做全局 attention。这个设计选择在 Kimi K3 里被明确写成一条规则（§7）。

训练：对照实验用 **1.4T tokens**（K2 预训练语料子集），4096 上下文，MuonClip 优化器，WSD 调度，`lr = 1.1e-3`，global batch 32M tokens；最终发布 checkpoint 用 **5.7T tokens**。

Scaling law：**Kimi Linear 的计算效率约为 MLA 的 1.16 倍**（MLA: `2.3092·C^{−0.0536}`，Kimi Linear: `2.2879·C^{−0.0527}`）。

结果层级（原文的总结）：

```
预训练 / SFT 阶段：  Kimi Linear  >  GDN-H  >  MLA
长上下文阶段：       Kimi Linear  >  MLA    >  GDN-H     ← GDN-H 掉队
RL 阶段：            Kimi Linear  >  MLA               且差距随训练拉大
```

Base（1.4T）：MMLU-Pro **51.0**（vs MLA 47.2、GDN-H 47.9）；MMLU 73.8；BBH 72.9。

合成任务（§5.1，2 层 2 头 head dim 128）：Palindrome / MQAR / Stack(LIFO, 64 独立栈)。KDA 在从 256 到 2048 的全部长度上准确率最高，MQAR/Palindrome 上收敛显著快于 GDN。Mamba2 在所有任务上全部失败——原文归因："a typical linear attention that uses only multiplicative decay and **lacks a delta rule**." 这是 [`08`](./08_linear_delta_rule.md) §3 那个「门控与 delta rule 正交」论点的又一次确认。

### 效率：两组数字的区别

| | 512k | 1M |
|---|---|---|
| Prefill 相对 MLA | **2.3×** | **2.9×** |
| Decode TPOT（bs=1） | 1.8× | **2.2×** |
| Decode TPOT（大 batch） | —— | **6.3×**（1.84ms vs MLA 11.48ms） |

原文的分析给出了一个简洁的上界：

> "For our hybrid model, as sequence length increases, the **I/O-bounded decoding time approaches a maximum hybrid efficiency ratio of 3:1** compared to full attention… by eliminating the need for a large, linear-scaling KV cache, Kimi Linear is able to **reallocate memory resources to support larger batch sizes**."

```
┌──────────────────────────────────────────────────────────────────────────┐
│  2.3× / 2.2× 是【单序列】的算力/IO 加速（受 3:1 层比的上界约束，理论 4×）    │
│  6.3× 需要靠省下的 KV cache 换【更大 batch】才能达到                        │
│  —— 写文档/汇报时千万不要混淆这两个数字                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

FLOPs（原文 §6.3，单头，$d_h = 128$，$C = 64$）：

$$
\begin{aligned}
\mathrm{FLOPs}_{\mathrm{KDA}}(T) &= 6 T d_h^2 + 3 T C d_h + T C^2 = 126976 \cdot T \\
\mathrm{FLOPs}_{\mathrm{Attn}}(T) &= 2 T^2 d_h = 256 \cdot T^2
\end{aligned}
$$

两式相等解出交叉点 $T \approx 496$。

约 500 token 之后 KDA 的 FLOPs 就更少。这与 MiniMax 所说的「理论交叉点在几千 token」数量级一致；MiniMax 的批评是实际交叉点因 memory-bound 而远高于此（[`11`](./11_hybrid.md) §5）。

推理策略：prefill 用 FLOP-intensive 的 chunk kernel；autoregressive generation 切到 recurrent kernel。这就是 [`06`](./06_linear_foundation.md) §3 那张三形式表在生产环境中的落地。

## 7. Kimi K3：KDA 扩展到 2.8T 规模

[arXiv:2607.24653](https://arxiv.org/abs/2607.24653)。这是 KDA 从 48B 研究模型扩展到 2.8T 生产模型的验证，也是 [`11`](./11_hybrid.md) §5 那场「hybrid 是否可行」争论中最重的一份证据。

| 维度 | Kimi K2 | **Kimi K3** |
|---|---|---|
| 总 / 激活参数 | ~1T / 32.6B | **2.8T / 104B** |
| 层数 | 61 | **93** |
| attention层组成 | **61 MLA** | **69 KDA + 24 Gated MLA** |
| attention 机制 | MLA + RoPE/YaRN | **Hybrid KDA–Gated MLA，全 NoPE** |
| attention头数 | 64 | 96 |
| 专家数 / 激活 / shared | 384 / 8 / 1（DeepSeekMoE） | **896 / 16 / 2（Stable LatentMoE）** |
| 训练上下文 | 128K | **1M** |
| 激活函数 | SwiGLU | **SiTU-GLU** |
| 优化器 | Muon | **Per-Head Muon** |
| 量化 | FP8 | **MXFP4 权重 / MXFP8 激活（QAT）** |

架构要点：**3:1 KDA : Gated MLA**，每 block = 3 KDA + 1 Gated MLA，且——

> "An additional Gated MLA layer is placed at **the end of the backbone**, ensuring that the **final layer always performs global attention**."

——这与 Kimi Linear 27 层布局是同一个设计哲学，这次写成了明确规则。**全 NoPE**："Unlike Kimi K2 and Kimi K2.5, Kimi K3 follows the hybrid design of Kimi Linear and applies No Position Encoding (NoPE) to **all** MLA layers."

另有 **AttnRes（Attention Residuals）**：把 attention 思想用到深度维——每层用可学习 pseudo-query $w$ 对 embedding 及所有先前 block 输出算 attention 权重 $\alpha$，选择性检索。Block AttnRes 把开销从 $O(Nd)$ 降到 $O(N_b d)$（$N_b$ 为 block 数）。声称相对 K2 有约 2.5 倍的整体 scaling efficiency 提升。

### K3 对 KDA 的两处改进

#### 改进一：lower-bounded decay

这一改进彻底消除了二级分块的对角路径，也是 [`07`](./07_linear_decay_gating.md) §4.4 那个遗留问题的最终解决。

问题回顾：Eq. 9 的 $1/\Gamma^{1 \to C}$ 会无界增长直至溢出。Kimi Linear 的对策是 log 域加 16-token 二级 tile，其中对角 tile 仍需逐位置对计算（就是上面 `pair()` 那个循环），是 intra-chunk 的主要瓶颈。

K3 换掉了从 decay logits $z_t^h$ 到 log-decay $g_t^h$ 的映射：

```
Kimi Linear（沿用 GDN/Mamba-2）:   g_t^h = −e^{A_h} · Softplus(z_t^h)  ∈ (−∞, 0)^{d_k}
                                                                        ↑ 无下界

┌──────────────────────────────────────────────────────────────────────────────┐
│  Kimi K3:   g_t^h = g_min · Sigmoid( e^{A_h} z_t^h )  ∈ (g_min, 0)^{d_k}      │
│             α_t^h = exp(g_t^h) ∈ ( e^{g_min}, 1 )^{d_k},    g_min = −5 固定    │
└──────────────────────────────────────────────────────────────────────────────┘
```

$A_h$ 是可学习的逐头 log-scale（初始化 $A_h = 0$）。

数值分析（原文逐步推导的那条链）：

```
g_min = −5           ⇒  每个 retention factor α_{t,j}^h > e^{−5} ≈ 6.7e−3
16-token tile        ⇒  tile 上累积 log-decay ∈ (−80, 0)
                     ⇒  倒数缩放因子 < e^{80}
                     ⇒  【落在 BF16 的动态范围内】（BF16 max ≈ 3.4e38 ≈ e^{88.7}）
                     ⇒  对角 tile 与非对角 tile 都能用 dense Tensor Core matmul
                     ⇒  彻底消除逐位置对的对角路径
```

这条因果链是本篇中心论点的最终形态：

```
想要通道级衰减（表达力）
    → 累积衰减带 d 下标，无法提出 matmul（[07] §4.2）
    → 1/Γ 溢出，只能 log 域 + 二级分块，对角块丢掉 tensor core（[07] §4.4）
    → KDA 绑定 a=b=k，二级 chunk 矩阵 4→2，省 3 次 matmul，~2×（本篇 §3）
    → K3 给衰减加下界 g_min=−5，缩放因子落进 BF16 范围，对角块也能上 tensor core（本节）
```

四步走了三年，每一步都是「表达力需求引出数值问题、数值问题推动算子设计」。

#### 改进二：full-rank output gate

Kimi Linear 用低秩 $W_g^{\uparrow} W_g^{\downarrow}$；K3 改为输入相关的全秩投影：

$$
y_t = W_o\, [\, \mathrm{Sigmoid}(W_g x_t) \odot \mathrm{RMSNorm}(\tilde{o}_t) \,]
$$

（KDA 层。）

同一个全秩门也加到 MLA 层上（"Gated MLA"，Eq. 7）：

$$
y_t = W_o\, [\, \mathrm{Sigmoid}(W_g x_t) \odot \tilde{o}_t \,]
$$

（MLA 层，注意没有 RMSNorm。）

> "This gate allows each token to **modulate the channels read from global attention**."

为什么 MLA 版本不要 RMSNorm：softmax 输出本身已是凸组合、方差可控；linear attention 输出的动态范围则需要归一化。这个不对称在 [`10`](./10_gating.md) §4 会再讲。

其他系统工程：**FlashKDA** 专用 CUDA kernel、**KDA Context Parallelism**（跨设备切分序列维）、**state-aware prefix caching**（利用 KDA 固定状态易于传输复用的特性）。上下文课程：预训练从 8K 到 64K，cooldown 阶段从 256K 到 1M。训练时 flash attention 输出保持 **FP32** 以修正舍入偏差。kernel 侧见 [06 · Flash Linear Attention](../fa/06_flash_linear_attention.md) §10。

## 8. Kimi 家族的 attention 机制

这几个模型的 attention 经常被混淆，这里逐个核实：

| 模型 | attention | 说明 |
|---|---|---|
| **Kimi K1.5**（[arXiv:2501.12599](https://arxiv.org/abs/2501.12599)） | **未公开** | Kimi Linear 只在 **RL 算法**层面引用它（"we use the same algorithm as in K1.5"），未涉及attention架构。**不要断言 K1.5 的attention类型。** |
| **Kimi K2 / K2.5**（[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)） | **全 MLA，无线性attention** | 复用 DeepSeek-V3 建模代码（`architectures: DeepseekV3ForCausalLM`），61 层全 MLA，RoPE θ=50k + YaRN，128K。创新在 **MuonClip** 与 MoE 稀疏度，不在attention |
| **Kimi Linear 48B-A3B** | **KDA : MLA(NoPE) = 3:1** | 研究/验证模型，1M |
| **Kimi K3 2.8T-A104B** | **KDA : Gated MLA(NoPE) = 3:1** | 生产模型，1M |
| **MoBA** | 稀疏，不是线性 | 属于 [`04`](./04_sparse_trainable.md)。"已部署支撑 Kimi 的长上下文请求"——但那是 serving 侧，不是 K2 这个**模型**的架构 |

> 值得留意的一点：Moonshot 同时押注了两条路线——MoBA（稀疏，部署在 serving）和 KDA（线性，写进模型架构）。而它发布的旗舰模型 K2 用的是纯 MLA，K3 才切到 KDA。**研究架构、serving 优化、发布模型是三条独立的时间线。**

## 9. 稀疏与线性的对比

Kimi Linear §7.1 对两条路线的对比值得完整引用——它是 [Attention 机制](./README.md) §1 那句核心论断的原文来源：

> "Sparse attention tends to **retrieve fine-grained historical information more effectively**, but this advantage comes at the cost of **storing the entire KV cache** for token selection… Moreover, sparse attention performs **only information selection**, and **its theoretical expressive upper bound remains that of full attention**. In contrast, linear attention, grounded in the principle of '**compression as intelligence**', enables generalization with a fixed-size state and, when combined with the **Delta learning rule**, can achieve **theoretically stronger expressive capacity**."

三个论点，各自的出处：

| 论点 | 出处 |
|---|---|
| 稀疏必须保留全部 KV cache | [Attention 机制](./README.md) §7 的账本表；[`05`](./05_sparse_dsa_frontier.md) §8 结尾 |
| 稀疏的表达力上限 = full attention（它只做选择） | 定义上如此 |
| linear + delta rule 理论上可以更强 | RWKV-7 的复杂度类结果（[`08`](./08_linear_delta_rule.md) §4）：单层解 $S_5$、常数层识别所有正则语言、严格超越 $\mathrm{TC}^0$ |

## 10. 小结

| 项 | KDA 的答案 |
|---|---|
| 递归 | $S_t = (I - \beta_t k_t k_t^{\top})\, \mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^{\top}$ |
| 相对 GDN | 标量 $\alpha_t$ → 通道级 $\mathrm{Diag}(\alpha_t)$ |
| 相对通用 DPLR | 约束 $a = \beta k$、$b = \alpha \odot k$ ⇒ 二级 chunk 矩阵 4→2、省 3 次 matmul ⇒ **~2×** |
| 相对 RWKV-7 | 放弃「移除键 ≠ 写入键」的表达力，换 2× 速度 |
| 位置编码 | KDA 本身就是数据相关的乘性位置编码 ⇒ MLA 层可以 NoPE ⇒ **纯 MQA + 免 YaRN** |
| 输出门 | Sigmoid（**不是** swish；实测 5.65 vs 5.81），低秩（K3 改全秩并推广到 MLA） |
| 数值 | Kimi Linear：log 域 + 16-token 二级分块；**K3：$g_{\min} = -5$ 下界 ⇒ 全部 tile 上 tensor core** |
| 规模验证 | Kimi Linear 48B-A3B（1M）→ **Kimi K3 2.8T-A104B（1M）** |

---

下一篇：[10 · 门控](./10_gating.md) —— 「门」在本章出现了太多次：decay gate、write gate $\beta$、output gate、NSA 的分支 gate、Gated MLA。它们是三种不同的东西，值得一次讲清，顺便解释输出门为什么能把 attention sink 从 46.7% 降到 4.8%。
