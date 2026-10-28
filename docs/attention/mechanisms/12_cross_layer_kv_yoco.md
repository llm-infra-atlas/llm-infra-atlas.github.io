# 12 · 跨层 KV 共享与 YOCO

前面的 GQA/MLA 减少每层 KV 的宽度，sparse/linear attention 则改变历史 token 的访问方式。还有一个独立的维度：同一段历史为什么需要在每层各缓存一次？本篇从跨层共享讲起，推导 YOCO 的 decoder-decoder 结构，再讨论 DeepSeek-V4.1 的 causal encoder-decoder。YOCO 发表于 2024 年；它在这里作为理解后续架构的基础，而不是被标成最新发布的模型。

## 1. KV cache 的三个维度

对 K/V head dimension 相同的 GQA，设 batch 为 $B$、序列长度为 $N$、层数为 $L$、KV heads 为 $h_{kv}$、每头维度为 $d$、每元素字节数为 $b$。普通 decoder 的 cache 大小约为

$$
M_{\mathrm{KV}}=2BNLh_{kv}db.
$$

减少 $h_{kv}$ 是 head 共享，缩小 $d$ 或 latent width 是通道压缩，缩小有效历史长度是窗口/序列压缩。跨层共享则减少需要独立保存 KV 的层组数量 $G$：理想情况下，公式中的 $L$ 变成 $G$。

这几项可以组合，但减少存储不自动减少读取。即使十层共用同一块 cache，十层仍可能各自读取并计算一次 attention；节省的是缓存副本，吞吐收益还取决于 HBM 复用与算子调度。

## 2. Cross-Layer Attention

[Cross-Layer Attention（CLA）](https://arxiv.org/abs/2405.12981)把相邻若干层组成一组：组内某一层生成 K/V，其他层只生成自己的 Q 并复用这些 K/V。省略 norm、RoPE 与 head reshape，若 producer 是第 $g$ 层，consumer 是第 $\ell$ 层：

$$
K^{[g]}=H^{[g-1]}W_K^{[g]},\qquad
V^{[g]}=H^{[g-1]}W_V^{[g]},\qquad
Q^{[\ell]}=H^{[\ell-1]}W_Q^{[\ell]},
$$

$$
O^{[\ell]}=\operatorname{softmax}
\left(\frac{Q^{[\ell]}(K^{[g]})^\top}{\sqrt d}+M\right)V^{[g]}.
$$

consumer 仍能用不同 Q、output projection 和 FFN 形成不同表示；共享的是被查询的历史表示。训练也必须适应这一约束，不能一般性地把现有 checkpoint 的多层 KV 合并而声称输出不变。

还要区分 KV sharing、index sharing 和 value residual：只共享 top-k 索引没有合并 KV storage；只混合或复用 V 没有共享 K；复用相同的 attention mask 也不代表复用 attention 分数。

## 3. YOCO 的两段结构

[YOCO](https://arxiv.org/abs/2405.05254)把模型划成下半段 self-decoder 与上半段 cross-decoder。两段都使用 causal 依赖，整体仍是 autoregressive language model；它不是读取一个双向 source encoder 的传统翻译模型。

![YOCO 的 self-decoder 与 cross-decoder](assets/arxiv/2405.05254_yoco_arch.png){ width="560" style="background:white;padding:12px" }

图源：YOCO Fig. 2。下半段先编码 token，边界处生成全局 KV；上半段各层以自身 Q 查询这份共同历史。图中每段代表多层，而不是整模型只有两个 block。

### Self-decoder

设边界 hidden state 为 $H^{[L/2]}$。self-decoder 使用 cache 有界的 efficient self-attention。论文研究两种选择：gated retention 用固定大小矩阵状态；sliding window 用固定长度的局部 KV。前者的简化单头递推为

$$
S_t=\gamma_tS_{t-1}+k_tv_t^\top,\qquad
o_t=S_t^\top q_t.
$$

这里 $\gamma_t$ 是输入相关、head-wise 的标量衰减，不是 KDA 的 channel-wise gate。此层把先前上下文融入当前 token 表示，具体递推原理见 [linear attention](./06_linear_foundation.md)。

### Cross-decoder

在边界处一次生成共享 KV：

$$
\widehat K=\operatorname{Norm}(H^{[L/2]})W_K,\qquad
\widehat V=\operatorname{Norm}(H^{[L/2]})W_V.
$$

上半段第 $\ell$ 层的 attention 为

$$
O^{[\ell]}=
\operatorname{softmax}\left(
\frac{Q^{[\ell]}\widehat K^\top}{\sqrt d}+M_{\mathrm{causal}}
\right)\widehat V.
$$

Q 逐层变化，KV 不随 consumer 层重新生成。全局历史仍可逐 token 检索，因此 YOCO 的整体状态不是纯 RNN 的常数状态；它保留了一份随 $N$ 增长的全局 KV。

## 4. 为什么 prefill 可以提前结束

普通 decoder 的第 $\ell+1$ 层 KV 来自该层的输入 hidden states，所以要建立所有层的 prompt cache，必须让所有 prompt token 走过全模型。YOCO 上半段的未来查询只依赖边界共享 KV，不需要历史 token 在每个 cross-decoder 层的输出。

因此，若只需要下一 token 的分布，可以这样执行：

1. 所有 prompt token 经过 self-decoder，建立各层局部状态和边界共享 KV。
2. 只有最后一个 prompt 位置继续通过全部 cross-decoder 层，计算下一 token logits。
3. 每个新生成 token 再依次走完 self-decoder 和 cross-decoder，并向共享 KV 追加一项。

上半段对最后位置的查询仍会读取整个可见历史，但不必为所有 prompt 位置都执行上半段 FFN 和全局 attention。这里的提前结束依赖模型结构，不是省略对结果有影响的运算。

如果训练时每个位置都计算 loss，或服务要求返回所有 prompt token 的 log-prob，则这些位置仍须通过 cross-decoder，不能使用相同的提前结束。YOCO 的 prefill 优势也不能直接当成训练 FLOPs 同比例下降。

## 5. 显存与计算账本

只看单请求、各层相同 head shape。令 self-decoder 层数为 $L_s$，若使用固定窗口 $W$：

$$
M_{\mathrm{YOCO,window}}
\approx 2bh_{kv}d\left(N+L_s\min(N,W)\right).
$$

第一项是一份共享的全局 cache，第二项是各 self-decoder 层自己的窗口。若使用矩阵状态，则为

$$
M_{\mathrm{YOCO,recurrent}}
\approx 2b_{kv}Nh_{kv}d
+b_sL_sh_s d_kd_v.
$$

状态可能用 FP32，KV 用 BF16，所以分别写 $b_s,b_{kv}$。随着上下文增长，第一项通常成为主导；其系数没有再乘全部层数。权重、activation、临时 attention workspace 不包含在这两式内。

再看算术量：固定窗口/状态维度时，下半段 prefill 对 $N$ 为线性；若上半段只处理最后一个 query，其每层全局 attention 也是对 $N$ 线性。但逐位置训练时，上半段 full attention 仍是平方复杂度。decode 也仍要在每个 cross-decoder 层访问历史，不能把一次 cache 写入理解成一次 attention 读取。

## 6. DeepSeek-V4.1 的 CED

[DeepSeek-V4.1 技术报告](https://arxiv.org/abs/2609.19969)明确把 YOCO 作为 Causal Encoder-Decoder（CED）的启发来源。其 40 层 backbone 划成 20 层 causal encoder 与 20 层 decoder，但保留了更多局部层间依赖。

decoder 的全局压缩 KV 及 compression weights 来自 encoder 最后一层，而不是对应 decoder 层的 hidden state：

$$
C^{[\ell]}=H^{[L/2]}W_{KV}^{[\ell]},\qquad
Z^{[\ell]}=H^{[L/2]}W_Z^{[\ell]},\qquad \ell>L/2.
$$

这些是 layer-dependent projection，不代表上半段每层无条件共用唯一一套 KV。具体哪些层共用物理 cache，由 CSA2 的 Full/Reindex/Reuse 分组进一步决定。

| 设计 | YOCO | DeepSeek-V4.1 CED |
|---|---|---|
| 全局 KV 来源 | self-decoder 边界 | causal encoder 边界 |
| 上半段全局访问 | 共享 KV 的 cross-attention | CSA2 压缩、稀疏与跨层复用 |
| 上半段局部状态 | 原始 cross-decoder 不建立各层历史 self-attention KV | 保留 layer-local SWA KV |
| Prefill 提前结束 | 历史位置可跳过 cross-decoder | 还需考虑局部 SWA replay |
| 持久化状态 | 全局 KV + 下半段局部状态 | 全局压缩 KV、replay 所需状态等 |

SWA 带来一个重要边界：decoder 局部 KV 仍依赖该层真实输入。严格重建多层 SWA 的理论感受野会跨越多个窗口；V4.1 用 bounded replay 控制重建范围，并通过训练与实验支持这个取舍。不能把「只回放最近一个窗口」自动视为任意 SWA 网络上的数学恒等变换。

官方报告区分 552B backbone 与 196B Engram 参数，并分别报告 prefill 8B、decode 16B 激活。这里的 active 参数也随执行阶段变化了。890 bytes/token 指报告配置下的**全局 KV**，不包含整个推理实例的全部状态与临时 buffer。更多 CSA2 细节见 [稀疏架构](./05_sparse_dsa_frontier.md)。

## 7. 对 serving 和训练的影响

共享 cache 需要有明确的 producer 和 consumer 生命周期。prefix cache 的一份共享页不能在一个 consumer 用完后提前释放；P/D transfer 应按独立 cache group 搬运，避免按层重复发送相同 storage。CED 还需要描述哪些状态必须传输、哪些局部状态可以 replay。

训练时，同一 KV 可能被多个上层查询使用，其梯度是各 consumer 的贡献之和。若采用 PP，边界 producer 的 activation 与多层 consumer 可能跨 stage，需联合设计传输、归约与重算；它不是仅修改 KV cache 分配器即可完成的变化。

最后，这些结构仍需比较质量与系统成本。较少的独立 KV 限制了各层历史表示的自由度；较大的共享 KV 或更多 projection 可以补充容量，却也会增加 cache/compute。模型应在训练时形成适合共享的表示，系统再利用其中真实存在的依赖关系。

## 参考

- [YOCO](https://arxiv.org/abs/2405.05254)、[官方实现](https://github.com/microsoft/unilm/tree/master/YOCO)：decoder-decoder 与 prefill 提前结束。
- [Cross-Layer Attention](https://arxiv.org/abs/2405.12981)：相邻层共享 K/V。
- [DeepSeek-V4.1](https://arxiv.org/abs/2609.19969)：CED、CSA2 与 SWA bounded replay。

返回 [hybrid 架构](./11_hybrid.md) 比较层间布局，或看[推理服务](../../serving/README.md)如何管理这些不同的 cache。
