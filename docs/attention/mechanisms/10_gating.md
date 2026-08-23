# 10 · 门控

> 「门」这个词在本章已经出现了五次，指的却是三种不同的东西：$\alpha_t$（衰减门，作用于状态转移）、$\beta_t$（写入门，作用于新写入量）、output gate（作用于层输出）；此外还有 NSA 的分支门。把它们混为一谈会得出错误的结论——尤其容易误以为 output gate 是 linear attention 的专属组件，而事实恰恰相反：它是在 softmax attention 上被系统验证有效之后，才回流到 linear attention 的。
>
> **前置**：[`07`](./07_linear_decay_gating.md)（decay gate 谱系）、[`08`](./08_linear_delta_rule.md) §1.3（$\beta_t$ 即学习率）、[`02`](./02_position_and_stability.md) §5.5（softmax 分母恒正，attention 无权「不看」）。
>
> 论文：Gated Attention [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)（Qwen 团队，NeurIPS 2025）。

---

## 1. 三种门控

```
                        作用对象           作用维度      出现在哪
① decay / forget gate   状态转移矩阵 A_t    【时间】维    只在 linear attention
   α_t                  S_t = A_t S_{t-1}+…              RetNet→Mamba-2→GLA→GDN→KDA→RWKV-7

② write gate / 学习率   新写入的量          【时间】维    只在 delta rule 族
   β_t                  + β_t k_t v_tᵀ                   DeltaNet→GDN→KDA；RWKV-7 的 a_t（通道级 ICLR）

③ output gate           层的输出            【通道】维    linear 与 softmax attention 【通用】
   σ(W_g x_t) ⊙ o_t                                      RetNet→GLA→GDN→KDA / Qwen3-Next→Kimi K3

（④ NSA 的分支门 g^cmp/g^slc/g^win 是 output gate 的一个变种：
   对同一层的三路输出各给一个独立 sigmoid，见 [04] §5）
```

为什么必须区分：①作用于时间维，控制「记住多久」；③作用于通道维，控制「读出哪些通道」。它们的梯度路径、参数量、失效模式全都不同。①只在有 recurrent state 的机制里有意义；③在任何 attention 里都有意义。

统一的叙事线索是有的，而且值得讲：LSTM/GRU 的门控思想在 Transformer 时代被拆成两半重新发现——遗忘门回到了 linear attention 的状态更新里，输出门回到了 attention 层的输出上。但这个故事只有在把两者分开讲之后才成立。

## 2. 衰减门与写入门

这两个门已经在 [`07`](./07_linear_decay_gating.md) 和 [`08`](./08_linear_delta_rule.md) 详细讲过，这里只做粒度上的收束：

| 粒度 | 衰减门 $\alpha_t$ | 代表 |
|---|---|---|
| 固定标量，逐头 | $\gamma$ 超参，不学 | RetNet、Lightning Attention |
| 数据相关标量 | $\exp(-\mathrm{softplus}(x_t W_\gamma) \cdot \exp(a))$ | Mamba-2、Gated DeltaNet |
| **数据相关通道向量** | $\sigma(x_t W_{\alpha 1} W_{\alpha 2})^{1/\tau}$ | GLA、RWKV-6、**KDA** |
| 通道向量 + 有下界 | $g_{\min} \cdot \sigma(e^{A_h} z_t)$，$g_{\min}=-5$ | **Kimi K3** |
| 矩阵（全 $d_k \times d_v$） | $\exp(-(1^{\top} \alpha_t) \odot \exp(A))$ | Mamba-1 |

写入门 $\beta_t$ 的粒度演进短一些：DeltaNet/GDN/KDA 都是标量 $\sigma(W_\beta x_t)$；RWKV-7 是通道向量 $a_t$（"in-context learning rate"）——从测试时回归的视角看，这是「逐通道的学习率」，对应 Adam 式的 per-parameter 学习率。

`fla` 里两个门的参数化都能逐行对上：`A_log [HV]` + `dt_bias [HV*K]`（KDA，[[fla:fla/layers/kda.py#L174-L185]]）、`A_log [HV]` + `dt_bias [HV]`（GDN，[[fla:fla/layers/gated_deltanet.py#L151-L168]]）。注意 KDA 的 `dt_bias` 多了 `K` 维——就是通道级的那一维。

## 3. output gate

output gate 的历史是从 RetNet 的一个局部补丁开始，最终变成一个系统性发现。

### 3.1 谱系

它的祖先是 RetNet 的 $(\mathrm{swish}(X W_G) \odot \mathrm{GroupNorm}(Y)) W_O$（[`07`](./07_linear_decay_gating.md) §1），当时的动机很局部：各头 $\gamma$ 不同导致输出方差统计不同，必须逐头归一化，于是顺手加了一个 gate。之后它被 GLA 沿用、被 GDN 沿用，直到 Qwen 团队系统研究了它，并推广到纯 softmax attention。

```
RetNet (2023)  swish gate + GroupNorm  ─┐
GLA (2024)     swish gate + 逐头 LN     ├─► 都被当成「linear attention 的一个补丁」
GDN (2024)     SiLU gate + Norm        ─┘
                                          │
arXiv:2505.06708 (2025)  ───────────────► 证明它对【纯 softmax attention】也有效，
                                          且给出机理：非线性 + 稀疏性
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────┐
Qwen3-Next / Qwen3.5「Gated Attention」               Kimi Linear / K3「Gated MLA」
（sigmoid gate on SDPA output）                       （sigmoid gate，KDA 层带 RMSNorm，MLA 层不带）
```

### 3.2 门的位置消融

论文测了五个位置（Fig 1）：

| 位置 | 施加处 |
|---|---|
| **$G_1$** | **SDPA 输出之后**（concat 之前，逐头） ⭐ |
| $G_2$ | value 投影之后 |
| $G_3$ | key 投影之后 |
| $G_4$ | query 投影之后 |
| $G_5$ | 最终 dense 输出层之后 |

通用形式 $Y' = Y \odot \sigma(X W_\theta)$，$X$ 取 pre-normalization 之后的 hidden states。

主结果（Table 1，15B MoE，3.5T tokens，节选）：

| 变体 | 形状 | 额外参数(M) | Test PPL ↓ | MMLU |
|---|---|---|---|---|
| **(5) SDPA Elementwise $G_1$** | $n \times q \times d_k$ | 201 | **5.761** | **60.82** |
| (6) v Elementwise $G_2$ | $n \times k \times d_k$ | 25 | 5.820 | 59.17 |
| (7) k Elementwise $G_3$ | $n \times k \times d_k$ | 25 | 6.016 | 59.18 |
| (8) q Elementwise $G_4$ | $n \times q \times d_k$ | 201 | 5.981 | 58.74 |
| (9) Dense Output $G_5$ | $n \times d_{\mathrm{model}}$ | 100 | 6.017 | 59.41 |
| **(10) SDPA Headwise $G_1$** | $n \times q$ | **1.6** | 5.792 | 60.05 |
| (11) v Headwise $G_2$ | $n \times q$ | **0.2** | 5.808 | 59.32 |
| (12) SDPA Head-**Shared** $G_1$ | $n \times d_k$ | 201 | 5.801 | 60.06 |

四条结论：

1. **$G_1$（SDPA 输出）与 $G_2$（value）最有效**，PPL 降超过 0.2、MMLU 涨约 2 点；$G_1$ 优于 $G_2$。
2. **Head-Specific 至关重要。** 看行 10：headwise 门在 15B 模型上只加 **1.6M** 参数（elementwise 的 1/125）就拿到了大部分收益。而一旦跨头共享（行 12 对 10、行 13 对 11），收益明显下降。原文："**As long as different heads receive distinct gating scores**, the granularity of gating and the choice of activation function have relatively minor impacts."
3. **乘性优于加性。**
4. **门控改善训练稳定性**：3.5T token 训练下 "largely reducing the loss spike"，允许更大 LR 和 batch。baseline 在提高 max LR 时直接发散；加 sandwich norm 能恢复收敛但改善微乎其微，而加门控后提高 LR 能带来明显提升。

第 2 条最有实用价值：想要 output gate 的好处，只需要 per-head 一个标量，参数量可以忽略不计。

```python
class GatedAttentionOutput(torch.nn.Module):
    """G1 位置的 output gate。headwise=True 时每 head 只有一个标量门（参数量 ≈ 0）。"""
    def __init__(self, d_model, n_h, d_h, headwise=False):
        super().__init__()
        self.n_h, self.d_h, self.headwise = n_h, d_h, headwise
        out = n_h if headwise else n_h * d_h
        self.g_proj = torch.nn.Linear(d_model, out, bias=False)

    def forward(self, attn_out, x):
        """attn_out: [B, T, n_h, d_h]（SDPA 输出，concat 之前）；x: [B, T, d_model]（pre-norm 之后）"""
        B, T, _, _ = attn_out.shape
        g = torch.sigmoid(self.g_proj(x))
        g = g.view(B, T, self.n_h, 1) if self.headwise else g.view(B, T, self.n_h, self.d_h)
        return attn_out * g          # ⊙ 之后再 concat → W_O
```

### 3.3 机理一：非线性与低秩瓶颈

$W_v$ 和 $W_O$ 是两个连续的线性层，可以合并成一个低秩线性变换（秩不超过 $d_h$）。在 $G_1$ 或 $G_2$ 引入门控，就是在两个线性映射之间插入非线性：

```
不加门：  v → W_v → (attention 的凸组合) → W_O        ≈ 一个低秩线性映射
加门：    v → W_v → (凸组合) → ⊙σ(W_g x) → W_O        ← Linear → 非线性 → Linear 的三明治
```

附带发现（Table 3 行 5）：对每个 head 输出单独做 RMSNorm——几乎零额外参数——也能显著降 PPL。这解释了为什么 GLA/GDN/KDA 的 `FusedRMSNormGated` 里那个 norm 不是可有可无的装饰。

Qwen3-Next 官方博客的说法也是这个角度："output gating … **to reduce low-rank issues in attention**"。

### 3.4 机理二：稀疏性与 attention sink

这是全篇最精彩的结果，也是 [`02`](./02_position_and_stability.md) §5.5 那个问题的第四个答案：加门控后，投向第一个 token 的平均 attention 从 46.7% 降到 4.8%。

$G_1$ 的 elementwise 门给 SDPA 输出引入强输入相关稀疏性。Fig 2 的数据：

| | 投向第一个 token 的平均 attention |
|---|---|
| baseline | **46.7%** |
| 加门控后 | **4.8%** |
| 第 21 层 baseline | **83%** |
| 第 21 层 加门控 | **4%** |

为什么有效：回顾 [`03`](./03_sparse_static.md) §2.1 的根因——softmax 的分母恒正，模型「没什么可看」时必须把概率质量放到某处，于是训练出一个 sink token 来承接。**门控提供了一个真正的出口**：直接把输出通道置零，无需通过 sink token 中转。

把四个答案并排看，这一节的位置就清楚了：

| 方案 | 怎么让 attention 有权「不看」 | 改了什么 |
|---|---|---|
| softmax₁ / Zero Sink | 分母加一个固定的 $1$ | 改 softmax |
| learned sink logit（GPT-OSS、DeepSeek-V4） | 分母加一个可学习的 $\exp(l_h)$ | 改 softmax |
| sigmoid attention | 彻底不归一化 | 改 softmax |
| **output gate** | **在输出侧给一个乘性的零通道** | **不改 softmax** |

最后一个方案的工程优势很明显：**它发生在 attention kernel 之外**，对 FlashAttention 完全透明——和 [`02`](./02_position_and_stability.md) §5.3 里「QK-Norm 胜过 softcap」是同一个理由。

论文明确承诺开源 "attention-sink-free models"。其他卖点：mitigates "**massive activation**"、"enhances long-context extrapolation"、RULER 上 "**over 10 points**" 的增益。

### 3.5 sigmoid 与 swish 的对比

| 来源 | 结论 |
|---|---|
| RetNet / GLA / GDN | 用 **swish / SiLU** |
| Gated Attention 论文 | 用 **sigmoid**（且指出激活函数选择「影响相对较小」，只要 head-specific） |
| **Kimi Linear 消融**（Table 1） | **sigmoid 5.65 vs swish 5.81** —— 差 0.16 val PPL |

Kimi 明确从 GDN 的 swish 换成了 sigmoid，并把 GDN-H baseline 也统一改成 sigmoid 以求公平（[`09`](./09_linear_kda_kimi.md) §5）。

关于 sigmoid 为什么更好，一个合理的解释是：$\mathrm{sigmoid} \in (0, 1)$ 是一个纯粹的衰减/选通算子；$\mathrm{swish}(x) = x \cdot \sigma(x)$ 在 $x$ 大时接近 $x$，是一个放大算子，会把输出的动态范围也一起改变。既然后面还有 $W_o$，放大是多余的，而且和前面的 RMSNorm 相冲突。（这是我的解读，论文只给了数字。）

## 4. Normalized gating：`FusedRMSNormGated` 的两种形态

linear attention 的层输出一般是这个形式：

```python
o_raw  = LinearAttn(q, k, v, alpha, beta)       # [B, T, H, d_v]
o_norm = RMSNorm(o_raw, per_head=True)          # 逐头归一化（多头方差统计不同，必须分开）
gate   = Sigmoid(W_g @ x)                        # Kimi Linear 低秩；Kimi K3 全秩
y      = W_o @ (gate * o_norm)
```

这就是 `fla` 里的 `FusedRMSNormGated`，KDA 层的连线在 [[fla:fla/layers/kda.py#L307]]（`activation="sigmoid"`），GDN 在 [[fla:fla/layers/gated_deltanet.py#L355-L359]]（`use_gate` 为真时用 `FusedRMSNormGated`，否则退化成普通 `RMSNorm`）。

⚠️ Kimi K3 的 Gated MLA 版本没有 RMSNorm：

$$
\begin{aligned}
\text{KDA:} \qquad y &= W_o \left[ \sigma(W_g x) \odot \mathrm{RMSNorm}(\tilde{o}) \right] \\
\text{Gated MLA:} \qquad y &= W_o \left[ \sigma(W_g x) \odot \tilde{o} \right]
\end{aligned}
$$

原因在于：softmax 输出本身已经是 value 的凸组合，方差可控；linear attention 的输出是未归一化的加权和（[`06`](./06_linear_foundation.md) §4.3 提到现代实现全部丢弃归一化项），动态范围需要 norm 来约束。**这个不对称直接来自「丢掉了 softmax 归一化」这个决定。**

## 5. NSA 的分支门

[`04`](./04_sparse_trainable.md) §5 的那个门是 output gate 的一个变种：它对同一层的三路输出各给一个独立 sigmoid，然后求和。

$$
o_t^* = g^{\mathrm{cmp}} \cdot o^{\mathrm{cmp}} + g^{\mathrm{slc}} \cdot o^{\mathrm{slc}} + g^{\mathrm{win}} \cdot o^{\mathrm{win}}
$$

三个 gate 是三个独立 sigmoid，不是 softmax。和 $G_1$ 的差别：$G_1$ 是「一路输出的逐通道选通」，NSA 的是「多路输出的逐路加权」。共同点是都作用在层输出、通道/分支维，都用 sigmoid，都不归一化——都是在给模型「关掉某条路」的权力。

DeepSeek-V4 的 CSA/HCA 里这个门消失了（改成 concatenation + grouped output projection），但 learned attention sink 出现了（[`05`](./05_sparse_dsa_frontier.md) §6.3）——同一个「让 attention 有权不看」的需求，在不同架构里以不同形式得到满足。

## 6. 小结

| 门 | 作用对象 | 维度 | 谁有 | 粒度演进 | 激活 |
|---|---|---|---|---|---|
| **decay / forget** $\alpha_t$ | 状态转移 $A_t$ | 时间 | 只有 linear attention | 固定标量→数据相关标量→**通道向量**→带下界 | $\exp(-\mathrm{softplus}(\cdot))$；K3 $g_{\min} \cdot \sigma(\cdot)$ |
| **write** $\beta_t$ | 新写入量 | 时间 | 只有 delta rule 族 | 标量（DeltaNet/GDN/KDA）→ 通道向量（RWKV-7 的 $a_t$） | sigmoid |
| **output** | 层输出 | **通道** | **linear 与 softmax 都有** | head-shared → **head-specific**（关键）→ elementwise | **sigmoid 优于 swish** |
| NSA 分支门 | 三路输出 | 分支 | NSA | —— | 三个独立 sigmoid |

需要记住的三条：

1. **decay gate 和 output gate 是两种东西**，不要因为都叫「门」就混起来。前者是 RNN 的遗忘门回归，后者是 attention 的通道选通。
2. **output gate 不是 linear attention 的专属**——它是在 softmax attention 上被系统验证后才回流的。Qwen3-Next 同时用了 Gated DeltaNet（衰减门）和 Gated Attention（输出门），而且用在不同类型的层上；Kimi K3 更极端：KDA 层有衰减门加输出门，Gated MLA 层只有输出门。
3. **head-specific 是 output gate 的关键**，粒度和激活函数都次要。per-head 一个标量就能拿到大部分收益，参数量可忽略。

---

下一篇：[11 · 混合模式：层比例、NoPE 与正反证据](./11_hybrid.md) —— 混合模式：3:1 这个比例从哪来、层间混合与层内并行的差别、NoPE 为什么是关键，以及 Kimi K3 与 MiniMax-M2 这场「hybrid 是否可行」的正反方证据。
