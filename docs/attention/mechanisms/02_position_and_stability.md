# 02 · 位置编码与数值稳定

> 本篇接续 [`01`](./01_basics_head_sharing.md)，尤其依赖其中 §4.3 decoupled RoPE 那一段代数；本篇沿用的统一记号见 [Attention 机制](./README.md) §2。
>
> 位置编码和 logit 的数值范围表面上不属于「attention 机制」本身，但后面几乎每一个设计都被它们卡住过：RoPE 阻断了 MLA 的矩阵吸收（01 §4.3）；NoPE 让 MLA 退化成纯 MQA，并省掉了 YaRN 调参（09 §5）；softmax 分母恒为正，这一点催生了 attention sink，进而催生了 StreamingLLM 和 GPT-OSS 的 learned sink（03 §2）；MLA 吸收模式下 key 不物化，所以 QK-Norm 用不了，Kimi 只能改从权重侧做 clip（本篇 §5.4）。正因为这几条线都要用到本篇的结论，这一篇需要放在 sparse 与 linear 两条路线之前。
>
> 全文中 RoPE / ALiBi / NoPE / YaRN / QK-Norm / softcap / attention sink / logit / entropy collapse 这些术语一律保留英文，不做翻译。

---

## 1. 位置信息进入 attention 的四种方式

位置信息可以在四个不同的地方注入 attention，下表把它们并排列出来：

```
                     加在哪里                     进 KV cache 吗    可否与投影交换
① 绝对 APE (sin/learned)  token embedding 上           ——             ——（在 attention 之外）
② 相对 bias (T5, ALiBi)   logit 上加一个 b_{i-j}       否             是（不动 K/V）
③ 旋转 (RoPE)             q, k 各自被旋转              **是**（k 已旋转）  **否** ← MLA 的麻烦来源
④ 无 (NoPE)               不加                         ——             ——
```

真正决定后面工程细节的是第三列和第四列。`②` 不碰 K/V，所以对任何 KV 压缩方案都友好；`③` 把位置信息烙进了 $k$，于是 $k$ 在 cache 里的形态被固定住，$W^{UK}$ 也就再也提不出来了。

## 2. RoPE：旋转、相对性与两种排布

Su et al. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)。核心（Eq. 14–15）：把 $q$/$k$ 的 $d$ 维切成 $d/2$ 个二维平面，第 $i$ 个平面按角度 $m\theta_i$ 旋转（$m$ 是绝对位置）：

$$
\begin{aligned}
f_{q,k}(x_m, m) &= R_{\Theta,m}^{d}\, W_{q,k}\, x_m, \qquad
R_{\Theta,m}^{d} = \operatorname{blockdiag}\big(R_1, \dots, R_{d/2}\big) \\
R_i &= \begin{bmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{bmatrix}, \qquad
\Theta = \{\, \theta_i = \mathrm{base}^{-2(i-1)/d} \,\}, \quad \mathrm{base} = 10000
\end{aligned}
$$

**相对位置性质（Eq. 16）** 是 RoPE 存在的全部理由：因为 $R$ 是正交矩阵，满足 $(R_m)^{\top} R_n = R_{n-m}$，两个 token 的内积就只依赖它们的相对位置：

$$
q_m^{\top} k_n = (R_{\Theta,m} W_q x_m)^{\top} (R_{\Theta,n} W_k x_n) = x_m^{\top} W_q^{\top} R_{\Theta,\, n-m}\, W_k x_n
$$

等号右端的旋转矩阵只依赖 $n-m$。原文强调了这一点："$R_{\Theta}^d$ is an **orthogonal matrix**, which ensures stability during the process of encoding position information." 正交性同时保证了两件事：相对性成立，并且 $\|Rx\| = \|x\|$，也就是旋转不改变 q/k 的模长，logit 的尺度不会因为位置不同而变化。

**高效实现见 Eq. 34**，写代码时可以直接照这个来：

```python
def rope(x, pos, base=10000.0):
    """x: [..., T, d] with d even.  pos: [T] (long).  Interleaved 排布（论文 Eq. 34）。"""
    half = x.shape[-1] // 2
    freq = base ** (-torch.arange(half, dtype=x.dtype, device=x.device) / half)
    ang  = pos[:, None].to(x.dtype) * freq[None, :]          # [T, half]
    cos  = ang.cos().repeat_interleave(2, -1)                # [T, d]
    sin  = ang.sin().repeat_interleave(2, -1)
    x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + x_rot * sin

# 实测（float64）：把同一对 (x, y) 放在 (m,n) = (3,7),(10,14),(0,4),(25,29) 上，
# q_m^T k_n 的最大离散度 = 8.9e-16 —— 只依赖 offset，与绝对位置无关。✓
```

> ⚠️ **两种排布，数学等价但权重不通用。** 论文 Eq. 34 是 **interleaved**（$x_{2i}$ 与 $x_{2i+1}$ 配对）；HuggingFace 的 Llama 实现是 **half-split**（$x_i$ 与 $x_{i+d/2}$ 配对）。两者只差一个维度置换，单独训练都对，但**拿 interleaved 的权重跑 half-split 的 kernel 会静默降质**。DeepSeek-V3.2 的 repo 明确记录了这个坑：indexer 的 RoPE 要 non-interleaved 而 MLA 的 RoPE 要 interleaved，2025-11-17 之前的 demo 代码搞错了、静默掉点。

**Long-term decay（Eq. 35–37）**：把打分写成复数形式 $\mathrm{Re}\big[\sum_i q_{[2i:2i+1]}\, k^{*}_{[2i:2i+1]}\, e^{i(m-n)\theta_i}\big]$，经过 Abel 变换可以得到一个上界，它会随 $|m-n|$ 增大而衰减。这意味着 RoPE 天生带有一个**弱的 recency bias**：距离越远，打分的上界越小。这一点会在 [`09`](./09_linear_kda_kimi.md) §4 变成关键论据——**linear attention 里的乘性衰减扮演了同样的角色，而且它是数据相关、可学习的**，比 RoPE 的固定衰减更灵活。

## 3. ALiBi / NoPE / partial RoPE

### 3.1 ALiBi

ALiBi（[arXiv:2108.12409](https://arxiv.org/abs/2108.12409)）直接在 logit 上加一个与距离成正比的负偏置：$\mathrm{softmax}(q_i K^{\top} + m \cdot [-(i-1), \dots, -2, -1, 0])$。

三个容易记错的细节（原文）：①"we do **not** add position embeddings at any point in the network"；②脚注 10："The ALiBi bias is **not** multiplied by the $\sqrt{d_k}$ scaling factor"；③slopes **固定不训练**，$n$ 头时是以 $2^{-8/n}$ 为首项、同值为公比的几何数列（8 头：$\{2^{-1}, \dots, 2^{-8}\}$）。作者试过让 slope 可学，"did **not** yield strong extrapolation results"（还慢 3%）。

ALiBi 属于「加在 logit 上」的第 `②` 类，**不碰 K/V**，所以它和任何 KV 压缩方案都兼容。它后来没有流行开的原因是长上下文检索能力偏弱，本质上它是一个硬编码的指数 recency prior，距离一远就几乎看不到东西了。

### 3.2 NoPE

NoPE（[arXiv:2305.19466](https://arxiv.org/abs/2305.19466)）是完全不加任何位置编码。这背后的关键洞察，原文引用了 Tsai et al. 2019 的说法："encoder-only Transformers, such as BERT, become **bag-of-words** models in the absence of positional encoding. However, **decoder-only Transformers with causal attention mask are not permutation invariant**." 换句话说，**causal mask 本身就已经编码了顺序信息**：token $t$ 能看到多少个前面的 token，这件事本身就是一种位置信息。

论文给出的实证结论是："**NoPE outperforms other explicit positional encoding methods**"，而且"They achieve this **without computing additional terms in the attention mechanism**"，也就是不需要额外计算量。理论上 NoPE 可以表示绝对和相对位置编码，但经验上它的行为更接近 T5 式的相对编码。

从 attention 分布的观察（Fig 4）还能看到一个有意思的规律：**NoPE 和 T5 相对位置编码会鼓励模型同时关注远处和近处；ALiBi 只关注最近的位置；而 RoPE 和 APE 没有特定的距离偏好**。

需要留意的是，该论文的实验规模只有约 107M 参数，这是它最大的局限。**NoPE 真正被大规模验证，是在 hybrid 架构里**——比如 Llama 4 的 NoPE global 层、Nemotron-H、Kimi Linear 里全部的 MLA 层，原因是这些架构里总有别的层（局部 RoPE 层或线性 attention 层）负责短程顺序信息，NoPE 层不需要单独承担这个责任。这也是 [`11`](./11_hybrid.md) §4 要展开讨论的主题。

### 3.3 partial RoPE

partial RoPE 只旋转每个 head 的前一小段维度：例如 head_dim 256、rope_dim 64 时，前 64 维带 RoPE、后 192 维 NoPE。

Qwen3-Next 官方给出的理由是："RoPE is partial — applied to only the first 64 of the 256 dimensions … to keep performance from dropping on inputs longer than what was seen during training."

这背后的机制可以这样理解：施加了 RoPE 的维度，一旦序列长度超出训练时见过的范围，旋转角就会进入模型从未见过的区间而失效；而没有施加 RoPE 的维度**只按内容匹配、与距离无关**，不会随序列变长而崩溃。所以只保留 25% 的维度带位置信息、其余 75% 不带，相当于把「可能崩溃的维度」限制在很小的一部分里。

实测配置：

| Model | `partial_rotary_factor` | RoPE dims / head_dim |
|---|---|---|
| GLM-4.6 | **0.5** | 64 / 128 |
| Qwen3-Next-80B | **0.25** | **64 / 256** |
| MiniMax-Text-01 | `rotary_dim: 64` | 64 / 128 |
| DeepSeek-V2/V3（MLA） | —— | 64 / 192（decoupled，机制不同：那 64 维是**独立**的 $k^{R}$，不是 $k^{C}$ 的前缀） |
| DeepSeek-V4 | —— | 最后 64 维 + 输出侧 $-t$ 反旋 |

> **DeepSeek-V4 的一个巧思值得单独提**：因为 KV entry 同时充当 value，绝对位置会从 value 路径泄漏到输出。V4 的解法是对输出 $o_{t,i}$ 再施加位置 $-t$ 的 RoPE，把泄漏的绝对位置转回相对位置（[arXiv:2606.19348](https://arxiv.org/abs/2606.19348)）。

## 4. YaRN：长上下文外推

如果 RoPE 是在 4K 长度上训练的，却要跑到 128K，直接外推会崩溃，原因同样是旋转角进入了训练时未见过的区间。YaRN（[arXiv:2309.00071](https://arxiv.org/abs/2309.00071)）把此前的一系列做法统一成了两个函数：$g(m)$ 负责位置变换，$h(\theta_d)$ 负责频率变换。这里 $s = L'/L$ 是 scale factor，$\lambda_d = 2\pi/\theta_d = 2\pi b^{2d/|D|}$ 是第 $d$ 维的波长：

| 方法 | $g(m)$ | $h(\theta_d)$ | 备注 |
|---|---|---|---|
| **PI**（位置插值） | $m/s$ | $\theta_d$ | "blind"：均匀拉伸所有维度 |
| **NTK-aware** | $m$ | $b'^{-2d/\lvert D\rvert}$，其中 $b' = b \cdot s^{\lvert D\rvert/(\lvert D\rvert-2)}$ | 只改 base；有 out-of-bound 外推 |
| **NTK-by-parts** | $m$ | $(1-\gamma(r_d))\cdot \theta_d / s + \gamma(r_d)\cdot \theta_d$ | "targeted"：**分频带** |
| **Dynamic NTK** | 推理时按当前 seqlen 动态更新 $s$ | 同 NTK-aware | 免微调最佳；**必须缓存 pre-RoPE 的 k** |
| **YaRN** | NTK-by-parts + attention scaling | 同上 | 综合最优 |

**ramp function（Eq. 18）**，$r(d) = L/\lambda_d$：

$$
\gamma(r) = \begin{cases}
0, & r < \alpha \\
1, & r > \beta \\
(r-\alpha)/(\beta-\alpha), & \text{otherwise}
\end{cases}
$$

其中 $r < \alpha$ 对应低频（波长 ≥ 上下文长度），取 $\gamma = 0$ 即只插值、不外推；$r > \beta$ 对应高频（波长 ≪ 上下文长度），取 $\gamma = 1$ 即完全不插值；中间地带线性过渡。

三条设计原则（原文逐字）：
- "if the wavelength $\lambda$ is **much smaller** than the context size $L$, we **do not interpolate**"
- "if the wavelength $\lambda$ is **equal to or bigger** than the context size $L$, we want to **only interpolate and avoid any extrapolation**"
- "dimensions in-between can have a bit of both"

这样设计背后的直觉是：高频维度在训练窗口内已经转过很多整圈，它学到的是「相邻 token 之间的细粒度顺序」，插值会破坏这种细粒度信息；低频维度在训练窗口内还没转完一圈，它编码的是「大致有多远」这种粗粒度信息，直接外推会让它进入完全没见过的旋转角度。LLaMA 族模型推荐的取值是 $\alpha=1, \beta=32$。

**YaRN 还额外做了 attention scaling（Eq. 21–22）**：

$$
\mathrm{softmax}\big(q_m^{\top} k_n \,/\, (t\sqrt{|D|})\big), \qquad \sqrt{1/t} = 0.1 \ln s + 1
$$

这是 LLaMA / Llama 2 的取值。这里有一个很漂亮的实现技巧："we can instead use a '**length scaling**' trick which scales both $q_m$ and $k_n$ by a constant factor $\sqrt{1/t}$ by simply **scaling the complex RoPE embeddings** by the same amount. With this, YaRN can effectively alter the attention mechanism **without modifying its code** … **zero overhead**." 也就是把温度直接折进预计算好的 cos/sin 表里，不需要改动任何 attention 代码。

训练成本也很低："only **400 training steps**, representing approximately **0.1% of the model's original pre-training corpus**."

**DeepSeek-V2 的变体常数不一样，容易抄错**："YaRN was specifically applied to the **decoupled shared key $k_t^{R}$** as it is responsible for carrying RoPE… we set the scale $s$ to **40**, $\alpha$ to 1, $\beta$ to 32… due to our distinct attention mechanism, we **adjust the length scaling factor**… $\sqrt{t} = 0.0707 \ln s + 1$"（注意 **0.0707 而非 0.1**，且写成 $\sqrt{t}$ 而非 $\sqrt{1/t}$）。长上下文训练只在 **32K** 做了 1000 步，128K NIAH 仍好。

各模型的 `rope_theta` / scaling（HF config）：

| Model | `rope_theta` | `rope_scaling` |
|---|---|---|
| Llama-3-70B | 500,000 | null（8K） |
| Llama-4-Scout | 500,000 | `llama3`, factor 16 |
| Qwen3 | 1,000,000 | null |
| Qwen3-Next-80B | **10,000,000** | null（partial 0.25） |
| DeepSeek-V3 | 10,000 | `yarn`, factor **40**, β_fast 32, β_slow 1, orig 4096 |
| Kimi-K2 | 50,000 | `yarn`, factor **32**, β_fast 1.0, β_slow 1.0 |
| gpt-oss-120b | 150,000 | `yarn`, factor 32（**只作用于 dense 层**） |
| Gemma-3-27B | global 1,000,000 / **local 10,000** | `linear` factor 8.0（**只作用于 global 层**） |
| Grok-2 | **208,533,496** | β_fast 8, β_slow 1, factor 16 |
| Falcon-H1 | **$\approx 10^{11}$** | —— **故意推到近 NoPE** |

> 表格最后两行其实是同一个想法的极端形式：**当 base 大到一定程度，所有维度的波长都远超上下文长度，RoPE 就实际上退化成了「几乎不加位置」**。Kimi Linear 的论文里也提到了这个技巧："**Falcon-H** uses an unconventionally high base frequency (e.g., $b \approx 10^{11}$) to push its positional encoding to a **near-NoPE state**."

## 5. 数值稳定：logit 爆炸与四种应对方法

### 5.1 病理：attention entropy collapse

[arXiv:2303.06296](https://arxiv.org/abs/2303.06296)（Zhai et al., Apple）把这个现象和背后的理论都讲得很清楚：

> 现象："we track the **attention entropy** for each attention head during the course of training, which is a **proxy for model sharpness**… low attention entropy is accompanied by high training instability, which can take the form of oscillating loss or divergence. We denote the pathologically low attention entropy … as **entropy collapse**."
>
> 理论："we **prove a tight lower bound on the attention entropy**, which **decreases exponentially fast with the growth of the spectral norm of the attention matrix logits**."

论文里的因果实验很有说服力：把 attention logits 的温度从 1 降到 0.1，**如果在 warmup 期干预**，entropy 会掉到接近 0，Hessian 的最大奇异值超过稳定阈值，训练随之发散；**如果在 warmup 之后再干预**，模型能恢复，但精度会更低。

ViT-22B（[arXiv:2302.05442](https://arxiv.org/abs/2302.05442)）的实测结果印证了这一点："we observed **divergent training loss after a few thousand steps**… for models with around **8B parameters**… It was caused by **extremely large values in attention logits**, which lead to (almost one-hot) attention weights with near-zero entropy." Appendix B 进一步指出："Without normalization, attention logits quickly grow to **over 50000** in magnitude."

把这些观察串起来，就是一条完整的因果链：logit 谱范数增长，导致 entropy 下界指数下降，进而触发 entropy collapse，最终训练发散。下面要介绍的四种手段，分别作用在这条因果链的不同环节上。

### 5.2 QK-Norm：约束 logit 的谱范数

ViT-22B 的做法（Gilmer et al. 2023）：

$$
\mathrm{softmax}\big((1/\sqrt{d})\; \mathrm{LN}(X W^Q)\; \mathrm{LN}(X W^K)^{\top}\big)
$$

⚠️ **ViT-22B 用的是 LayerNorm，且不是 per-head。** 后来 LLM 普遍改成 **per-head RMSNorm**，且**在 RoPE 之前**施加：

```python
class QKNormAttention(torch.nn.Module):
    """per-head RMSNorm on q and k, applied BEFORE RoPE — Qwen3 / Gemma 3 / OLMo 2 的做法。"""
    def __init__(self, d, n_h, d_h, eps=1e-6):
        super().__init__()
        self.n_h, self.d_h = n_h, d_h
        self.q_proj = torch.nn.Linear(d, n_h * d_h, bias=False)
        self.k_proj = torch.nn.Linear(d, n_h * d_h, bias=False)
        self.q_norm = torch.nn.RMSNorm(d_h, eps=eps)      # [d_h]，所有 head 共享参数、逐 head 归一
        self.k_norm = torch.nn.RMSNorm(d_h, eps=eps)

    def forward(self, x, pos):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_h, self.d_h))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_h, self.d_h))
        q, k = rope(q.transpose(1, 2), pos), rope(k.transpose(1, 2), pos)   # norm 之后才 rope
        ...
```

**为什么这样做有效**：RMSNorm 把每个 head 的 $q$、$k$ 的 RMS 都固定成 1，于是 $|q^{\top} k| \le \|q\| \, \|k\| = d_h$，logit 被硬性夹在 $O(d_h)$ 量级，logits 矩阵的谱范数因此有界，根据 §5.1 的下界，entropy 就不会 collapse。它还有一个副作用：学习率可以开得更大——"With the QK-normalization, the higher **1e-3** learning rate remains stable"。

各家落法：

| Model | 形式 |
|---|---|
| **OLMo 2** | **RMSNorm** on key/query projections + reordered norm + z-loss(1e-5)。表格里 OLMo-0424 是 "Clip to 8"、OLMo 2 是 "QK-Norm" |
| **Qwen3** | **per-head RMSNorm on Q and K, before RoPE**；同时 `attention_bias=False`（取代 Qwen2 的 QKV-bias）。权重名 `self_attn.q_norm.weight`，shape `[head_dim]` |
| **Gemma 3** | "**we replace the soft-capping of Gemma 2 with QK-norm**"，per-head RMSNorm(128) |
| **Llama 4** | `use_qk_norm: true` |
| **GLM-4.5/4.6** | "We also incorporate **QK-Norm** to stabilize the range of attention logits"（GLM-4.5-**Air** 是 No） |
| **DeepSeek-V3 / Kimi K2** | **No** —— 见 §5.4 |

**Qwen3-Next 还发现了 QK-Norm 自身的一个问题**：可学习的 scale 在训练中会"tend to increase abnormally"。它的解法是改成学习一个增量 $\Delta$，用 $(1+\Delta)$ 作为 scale，并对 norm weight 加 weight decay，这就是 **Zero-Centered RMSNorm**。

### 5.3 Softcap：把 logit 压进固定区间

Gemma 2（[arXiv:2408.00118](https://arxiv.org/abs/2408.00118)）："We cap logits … such that the value of the logits stays between $-\mathrm{soft\_cap}$ and $+\mathrm{soft\_cap}$"：

$$
\mathrm{logits} \leftarrow \mathrm{soft\_cap} \cdot \tanh(\mathrm{logits} \,/\, \mathrm{soft\_cap})
$$

Gemma-2-27B 实测值：`attn_logit_softcapping: 50.0`、`final_logit_softcapping: 30.0`。Grok-2 也用（且方向相反：attn 30 / final 50 / router 30）。

Gemma 3 后来把 softcap 去掉、换成了 QK-Norm，主要原因是 softcap 与 FlashAttention 不兼容：$\tanh$ 需要作用在完整的 logit 矩阵上，而 FlashAttention 从不物化完整的 $S$。FlashAttention 的变通做法是把 softcap 预乘进 scale 塞进 kernel（[03 · FA3：Hopper 上的异步化与 overlap](../fa/03_fa3_hopper_async.md) §5），但这限制了 kernel 的选择空间。相比之下，QK-Norm 作用在 $q$/$k$ 上，**发生在 attention kernel 之外**，对 kernel 完全透明，这正是它最终胜出的工程理由。

### 5.4 QK-Clip / MuonClip：MLA 吸收模式下的权重侧方案

**MLA 为什么不能用 QK-Norm**：因为吸收模式下 $k^{C}$ 根本不物化（key 本身就是 latent），没有一个具体的 $k$ 张量可以归一化。Kimi K2（[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)）给出了一个作用在权重侧的解法。

per-head 的 max logit（在一个 batch $B$ 上统计）：

$$
S_{\max}^h = \frac{1}{\sqrt{d}}\, \max_{X \in B}\, \max_{i,j}\; Q_i^h (K_j^h)^{\top}
$$

然后**按 head** 缩放权重：$\gamma_h = \min(1,\, \tau / S_{\max}^h)$，且**对 MLA 只 clip unshared 分量**：

| 分量 | 缩放 |
|---|---|
| $q^{C}$, $k^{C}$（head-specific） | 各乘 $\sqrt{\gamma_h}$ |
| $q^{R}$（head-specific rotary） | 乘 $\gamma_h$ |
| $k^{R}$（**跨 head 共享**） | **不动** |

$k^{R}$ 保持不动的理由很直接：它被所有 head 共用，如果为了某一个 head 去缩放它，就会连带伤到别的 head。

原文特别强调了这一点不属于前向计算的一部分："this operation does **not** alter the forward/backward computation in the **current** step — we merely use the max logit as a **guiding signal**." 也就是说，这是一个作用在 optimizer 之后的处理钩子，不进入计算图。

效果也很明显：在中等规模（9B activated / 53B total）下用纯 Muon 训练，max logits 会**迅速超过 1000**；换成 MuonClip 并设 $\tau=100$ 之后，K2 在 **15.5T tokens 的训练里做到了 zero loss spike**，max logits 被压在 100 附近，直到约 30% 的训练步数之后才自然衰减下来。

> 这是一个「机制约束反过来决定训练技巧」的典型例子：正因为选择了 MLA，业界标准的 QK-Norm 就用不上了，只能自己设计一个作用在权重上的等效方案。

### 5.5 softmax 分母：off-by-one、learned sink、sigmoid attention

softmax 有一个结构性的问题：**概率必须和为 1，所以 attention 永远不能「什么都不看」**——哪怕所有 token 都不重要，也必须把概率分给某个地方。

**softmax-off-by-one / QuietAttention**（Evan Miller）给出了第一种解法：

$$
\mathrm{softmax}_1(x)_i = \frac{\exp(x_i)}{1 + \sum_j \exp(x_j)}
$$

它的关键性质是："This lets the vector as a whole **tend to zero if it wants**"，也就是说 $\lim_{x \to -\infty} \mathrm{softmax}_1(x)_i = 0$，整个输出向量可以趋于零。作者自己给出的等价实现是："if you **prefix every input context with a zero vector**, and ensure that your neural network doesn't add any bias … then the zero should pass through unaltered and will have the effect of **adding unity to every subsequent softmax denominator**." 这其实就是 StreamingLLM 里的 "Zero Sink"（[`03`](./03_sparse_static.md) §2）。

**GPT-OSS 的 learned sink logit**（[arXiv:2508.10925](https://arxiv.org/abs/2508.10925)）走的是同一条思路，但把偏置变成可学习的："**Each attention head has a learned bias in the denominator of the softmax**, similar to off-by-one attention and attention sinks, which **enables the attention mechanism to pay no attention to any tokens**."

```python
def attn_with_learned_sink(q, k, v, sink_logit, causal_mask):
    """sink_logit: [H] 可学习标量。等价于在 softmax 分母里加 exp(sink_logit)。"""
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    s = s.masked_fill(causal_mask, float('-inf'))                       # [B,H,T,T]
    sink = sink_logit.view(1, -1, 1, 1).expand(*s.shape[:-1], 1)        # [B,H,T,1]
    p = torch.cat([s, sink], dim=-1).softmax(-1)[..., :-1]              # 拼进去、softmax、丢掉
    return p @ v          # 注意 p 的每行和 < 1，缺的那部分就是「不看」

# 等价的、不物化拼接的写法：
#   p = s.softmax(-1) * (s.logsumexp(-1, True).exp() /
#                        (s.logsumexp(-1, True).exp() + sink_logit.exp()))
```

DeepSeek-V4 也采纳了同样的做法（Eq. 27，per-head 可学习的 $l'_h$）：

$$
a_{h,t,s} = \frac{\exp(l_{h,t,s})}{\sum_{s'} \exp(l_{h,t,s'}) + \exp(l'_h)}
$$

**Sigmoid attention**（[arXiv:2409.04431](https://arxiv.org/abs/2409.04431)，Apple）走得更激进，直接把归一化整个去掉：

$$
\mathrm{SigmoidAttn}(X) = \sigma(QK^{\top} / \sqrt{d_{qk}})\, V, \qquad \sigma(u) = \mathrm{sigmoid}(u + b)
$$

**关键超参**是附录 E 推出的 order-optimal 值 $b = -\log n$（$n$ 是序列长度），它"allows us to make sense of SigmoidAttn for **any sequence length**"。理论上它是一个 universal approximator，Jacobian 的谱范数上界只依赖 $\|x_i\|^2$ 的**平均值**，而 softmax 的经典结果依赖的是**最大值**，因此有"improved regularity"。工程上，**FlashSigmoid** 不再需要 row-max/row-sum 的累积与回写，实现更简单。不过原文也提醒："theoretically, SigmoidAttn is **slower** than SoftmaxAttn"，也就是说它的加速来自消除了硬件瓶颈，而不是省了 FLOPs。实践上真正关键的一点是："**stabilization of large initial attention norms during the early stages of training** as a crucial factor for the successful training."

> 这三种做法其实是同一条线索上的三个刻度：softmax₁ 相当于「加一个固定的 1」，learned sink 相当于「加一个可学习的 $\exp(l_h)$」，sigmoid attention 则是「彻底不做归一化」。它们回答的是同一个问题——**怎么让 attention 有权说「我不看」**。而 [`10`](./10_gating.md) 会给出第四个答案：**output gate**，思路是不改 softmax 本身，而是在输出侧提供一个乘性的零通道。

## 6. Differential Attention：基础机制层面的变体

DIFF Transformer（[arXiv:2410.05258](https://arxiv.org/abs/2410.05258)）值得单独介绍一下，因为它是少数不动 mask、不动递归、不动 KV 共享，而是直接**改 attention 本身的代数形式**的工作：

$$
\begin{aligned}
\mathrm{DiffAttn}(X) &= \big(\mathrm{softmax}(Q_1 K_1^{\top} / \sqrt{d}) - \lambda \cdot \mathrm{softmax}(Q_2 K_2^{\top} / \sqrt{d})\big)\, V \\
[Q_1; Q_2] &= X W^Q, \qquad [K_1; K_2] = X W^K, \qquad V = X W^V
\end{aligned}
$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{d_{\mathrm{model}} \times 2d}$，于是 $Q_1, Q_2, K_1, K_2 \in \mathbb{R}^{N \times d}$，而 $V \in \mathbb{R}^{N \times 2d}$。

**$\lambda$ 的重参数化（Eq. 2）** —— 保证 $\lambda$ 可以取任意实数且梯度良好：

$$
\begin{aligned}
\lambda &= \exp(\lambda_{q1} \cdot \lambda_{k1}) - \exp(\lambda_{q2} \cdot \lambda_{k2}) + \lambda_{\mathrm{init}} \\
\lambda_{\mathrm{init}} &= 0.8 - 0.6 \exp(-0.3\,(l-1))
\end{aligned}
$$

其中 $l$ 是层号。multi-head 版本（Eq. 3–4）有两个关键细节：

$$
\begin{aligned}
\overline{\mathrm{head}_i} &= (1 - \lambda_{\mathrm{init}}) \cdot \mathrm{LN}(\mathrm{head}_i) \\
\mathrm{MultiHead}(X) &= \mathrm{Concat}(\overline{\mathrm{head}_1}, \dots, \overline{\mathrm{head}_h})\, W^O, \qquad h = d_{\mathrm{model}} / (2d)
\end{aligned}
$$

第一行是逐 head RMSNorm 再乘一个固定常数；$h$ 减半是为了对齐参数量与 FLOPs。这里有两个值得留意的细节：$\lambda$ **在同一层的所有 head 间共享**；$\mathrm{LN}$ 是**逐 head 独立的 RMSNorm**（论文 Fig 2 里写成 GroupNorm 来强调这一点）。消融实验显示，去掉这个逐 head norm，loss 会从 3.086 恶化到 3.122，原文的解释是"degrades performance due to **training instability**… multiple heads tend to have **different statistics** in our method"；而如果给普通 Transformer 加上同样的 GroupNorm，"has **negligible** effect"，说明这个 norm 是 differential attention 特有的需求。另外，**固定乘子 $(1-\lambda_{\mathrm{init}})$** 的作用是"align the gradients with Transformer… enables us to **directly inherit similar hyperparameters**"，也就是让梯度尺度和普通 Transformer 对齐，方便直接复用超参数。

> ⚠️ **KV cache 并不会翻倍。** 因为 per head 需要缓存 $K_1, K_2$（合计 $2d$）加上 $V$（$2d$），一共 $4d$，而 head 数减半为 $d_{\mathrm{model}}/(2d)$，所以总量是 $h \cdot 4d = (d_{\mathrm{model}}/(2d)) \cdot 4d = 2d_{\mathrm{model}}$；等参数量的 MHA 是 $(d_{\mathrm{model}}/d) \cdot 2d = 2d_{\mathrm{model}}$。**两者完全相同。**

卖点里对 infra 最有用的一条是最后一个："long-context modeling, key information retrieval, hallucination mitigation, in-context learning, and **reduction of activation outliers**"。减少 activation outlier 这一点直接有利于低精度量化（可以对照 [低精度（Low Precision）：从 FP8 到 FP4 的训练与推理](../../low_precision/README.md)）。不过 DIFF 目前还没有 frontier 模型采用，属于值得了解但尚未进入生产实践的方法。

## 7. head 数与 head_dim 的权衡

这个话题严格说不属于位置编码，但和「机制选择」紧密相关，放在这里一并交代。以下均为 2025–2026 年的实测趋势。

**Kimi K2 把 head 数从 128 减到了 64**，理由是："To reduce computational overhead during inference, we **cut the number of attention heads to 64**, as opposed to 128 in DeepSeek-V3." 背后的精确论据是：

> "with a sequence length of **128k**, increasing the number of attention heads from 64 to 128, while keeping the total expert count fixed at 384, leads to an **83% increase in inference FLOPs**." 而收益侧却很有限："doubling the attention heads yields only **modest improvements in validation loss (ranging from 0.5% to 1.2%)**"。

⚠️ 这里需要注意，K2 的 KV cache 和 DeepSeek-V3 完全一样（都是 $d_c + d_h^{R} = 576$，与 $n_h$ 无关）。减少 head 数省下的是 **core-attention 的 FLOPs 以及 $W^{UQ}$/$W^{UV}$/$W^{O}$ 的参数与带宽**，而不是 KV cache。这正体现了 MLA 的一个有趣性质：**head 数和 cache 大小是解耦的**，改变一个不会影响另一个。

**GLM-4.5 走的是相反方向，反而增加了 head 数，而且效果有点反直觉**："we utilize **2.5 times more attention heads (96 heads for a 5120 hidden dimension)**. **Counterintuitively, while this increased head count does not improve training loss** compared to models with fewer heads, **it consistently improves performance on reasoning benchmarks such as MMLU and BBH**." 同时该团队还"reduce the width … and increase its height (number of layers), as we found that **deeper models exhibited better reasoning capacity**"，也就是把宽度让给深度。

把这些案例放在一起看，可以总结出一条共同规律：2025 到 2026 年的趋势是把 attention 的「总宽度」$n_h \cdot d_h$ 做到 hidden 维度的 **2–2.5 倍**，同时用 GQA 把 $n_{kv}$ 压到 2–8。至于怎么在 head 数和 head_dim 之间分配，各家的取向并不相同：

| Model | $n_h \times d_h$ | hidden | 倍数 | 取向 |
|---|---|---|---|---|
| Qwen3-Next-80B | 16 × **256** = 4096 | 2048 | 2.0× | 少头宽维 |
| GLM-4.6 | **96** × 128 = 12288 | 5120 | 2.4× | 多头标准维 |
| Command A | 96 × 128 = 12288 | 12288 | 1.0× | —— |
| MiniMax-M2 | 48 × 128 = 6144 | 3072 | 2.0× | 多头标准维 |
| gpt-oss-120b | 64 × **64** = 4096 | 2880 | 1.4× | **小 head_dim**（配 SWA-128，cache 极小） |

## 8. 小结

下表整理了本篇的结论会在后面哪些地方被继续用到：

| 本篇的东西 | 后面在哪里用到 |
|---|---|
| RoPE 不能与投影交换 | [`01`](./01_basics_head_sharing.md) §4.3 decoupled RoPE；MLA 的全部复杂性来源 |
| RoPE 自带弱 recency decay | [`09`](./09_linear_kda_kimi.md) §4：KDA 的乘性衰减是「数据相关的可学习位置编码」 |
| NoPE 在 decoder 里可行（causal mask 已含顺序） | [`11`](./11_hybrid.md) §4：Kimi Linear 全 MLA 层 NoPE → 退化成纯 MQA + 省掉 YaRN |
| partial RoPE 把「会崩的维度」限制在一小段 | Qwen3-Next / GLM / DeepSeek-V4 |
| softmax 分母恒正 ⇒ 必须把概率倒在某处 | [`03`](./03_sparse_static.md) §2：attention sink 的根因；GPT-OSS/V4 的 learned sink |
| logit 谱范数 → entropy collapse → 发散 | §5 四种应对方法；MLA 吸收模式下只能用 QK-Clip |
| 「让 attention 有权不看」 | [`10`](./10_gating.md)：output gate 是第四个答案，且它把 sink 从 46.7% 降到 4.8% |

---

下一篇：[03 · sparse 路线（一）：静态稀疏与推理期动态稀疏](./03_sparse_static.md) —— 进入 sparse 路线。先看最朴素的一类：mask 只由位置决定的静态稀疏，以及「推理期临时上稀疏」这条路的局限在哪里。
