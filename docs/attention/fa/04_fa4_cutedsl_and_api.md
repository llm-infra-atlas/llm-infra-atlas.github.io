# 04 · FA4（CuTeDSL）与工程接口

> 前三篇讲的是算法和单 kernel 的硬件映射，本篇讲 infra 现实，共四件事：FA4 为什么把 kernel 从 C++ 模板搬到 Python（CuTeDSL）；work 如何调度到 SM（tile scheduler）；Python API 与 autograd 如何接线；以及训练和推理绕不开的四项功能——varlen（`cu_seqlens`）、GQA packing、paged KV、kvcache split-KV——外加 softcap / sliding window / ALiBi / dropout 这些 feature 在代码里的位置。本篇对照 [[flash-attention:flash_attn/cute/]] 和 [[flash-attention:flash_attn/flash_attn_interface.py]]。

---

## 1. FA4：用 Python 重写 FA3 的调度

FA3 的 C++ 模板有一个工程上的灾难：每个 `(head_dim, dtype, causal, local, softcap, varlen, paged, …)` 组合都要在编译期实例化一个 kernel，[[flash-attention:hopper/instantiations/]] 下有成百上千个 `.cu` 文件，编译需要几十分钟、二进制体积巨大、改一行代码就要全量重编。

FA4 改用 **CuTeDSL**（NVIDIA CUTLASS DSL）：kernel 直接用 Python 编写，运行时 JIT 编译成 PTX/CUBIN。CUTLASS 的核心抽象（`cute.Tensor`、layout、TiledMMA、TMA atom、pipeline）都有 Python 绑定，被 `@cute.jit` 标注的函数会被编译。好处是：

- **编译期特化用 Python 值完成**：被 `cutlass.Constexpr[...]` 标注的参数（如 `is_causal`、`head_dim`、`score_mod`）在 JIT 时被特化进 kernel，运行时零分支——和 C++ 模板特化等价，但用 Python 表达。看 `FlashAttentionForwardSm90.__init__`（[[flash-attention:flash_attn/cute/flash_fwd.py#L42-L60]]）：`tile_m=128, tile_n=128, num_stages, score_mod, mask_mod` 全是 constexpr 配置。
- **`score_mod` / `mask_mod` 是用户传入的 `@cute.jit` callable**，在编译时注入 kernel（FlexAttention 式的可编程 attention）。`apply_score_mod` 在 [[flash-attention:flash_attn/cute/flash_fwd.py#L1147-L1159]] 处插到 $QK^{\top}$ 之后、softmax 之前。
- **JIT cache**：编译产物按「dtype/head_dim/causal/mask hash/arch/block size」做 key 缓存（内存 LRU 加可选磁盘缓存），命中就免编译。

算法本身不变——`flash_fwd.py` 的内循环仍然是 01 那套流程：`gemm`($QK^{\top}$)、`mask_fn`、`softmax.online_softmax` 返回 `row_scale`、`softmax.rescale_O`、`gemm_rs`($PV$)（[[flash-attention:flash_attn/cute/flash_fwd.py#L1172-L1189]]）。SM90 走 FA3 的 warp-spec，SM100（Blackwell）另有 `flash_fwd_sm100.py`（UMMA、2CTA、原生 SplitKV/paged）。

## 2. Tile scheduler

「一个 CTA 算哪个 `(batch, head, m_block)`」由 tile scheduler 决定。FA4 把它抽象成可替换的策略（[[flash-attention:flash_attn/cute/tile_scheduler.py]]）：

| 策略 | 行为 | 适用 |
|---|---|---|
| `SingleTileScheduler`（[[flash-attention:flash_attn/cute/tile_scheduler.py#L169]]） | grid 直接铺满所有 tile，一个 CTA 算一个 tile 后退出（就是 FA2 的 `grid(num_m_block,b,h)`） | 静态、均匀负载 |
| `StaticPersistentTileScheduler`（[[flash-attention:flash_attn/cute/tile_scheduler.py#L287]]） | **persistent kernel**：只起 `#SM` 个 CTA，每个 CTA 循环领多个 tile，`get_next_work` 取下一个 | 减少 CTA 启动/尾声开销，tile 远多于 SM |
| `ClcDynamicPersistentTileScheduler`（[[flash-attention:flash_attn/cute/tile_scheduler.py#L56]]） | 用 Hopper 的 **CLC（cluster launch control）** 硬件做动态负载均衡，CTA 完成一个 tile 后向硬件请求下一个 | varlen / causal 等负载不均场景 |

**persistent kernel** 是关键 infra 概念：传统做法是每个 tile 起一个 CTA，而 CTA 的启动和退出（尤其是加载常量、建立 TMA descriptor）有固定开销；persistent 让少量长命 CTA 把工作「拉」过来做，摊薄这些开销，并让 producer 的预取能跨 tile 边界连续进行。这也是 FA3/FA4 的 producer-consumer 流水能持续填满的前提。在 [[flash-attention:hopper/flash_fwd_kernel_sm90.h#L328-L330]] 可以看到 producer 用 `scheduler.get_initial_work / get_next_work` 循环领取 tile。

## 3. Python API 与 autograd

FA2 的对外接口在 [[flash-attention:flash_attn/flash_attn_interface.py]]，公开函数如下（[[flash-attention:flash_attn/flash_attn_interface.py#L1019-L1545]]）：

```
flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, deterministic, ...)
flash_attn_varlen_func(..., cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, ...)   # 变长
flash_attn_qkvpacked_func / kvpacked_func                                                  # QKV 打包布局
flash_attn_with_kvcache(q, k_cache, v_cache, ..., cache_seqlens, block_table, ...)         # 推理 decode
```

每个 user-facing 函数背后是一个 `torch.autograd.Function`。看 `FlashAttnFunc`（[[flash-attention:flash_attn/flash_attn_interface.py#L828]]）：

- `forward`：`softmax_scale` 默认取 $d^{-0.5}$；head_dim 不是 8 的倍数时 pad 到 8 的倍数（[[flash-attention:flash_attn/flash_attn_interface.py#L852-L856]]，这是 TMA 的对齐要求）；调用 `_wrapped_flash_attn_forward`（自定义算子，支持 `torch.compile`，[[flash-attention:flash_attn/flash_attn_interface.py#L84]]）；`save_for_backward(q,k,v,out_padded,softmax_lse,rng_state)`（[[flash-attention:flash_attn/flash_attn_interface.py#L873]]）——只保存 $O(N)$ 大小的 out 和 LSE，不保存 $P$（01 第 5 节）。
- `backward`：取出 saved tensors，调用 `_wrapped_flash_attn_backward` 计算 `dq,dk,dv`，按需切掉 pad，返回的梯度元组与 forward 入参对齐（多余项填 `None`）。

底层 `_flash_attn_forward`（[[flash-attention:flash_attn/flash_attn_interface.py#L84-L114]]）就是把张量整理后调用 C++ 扩展 `flash_attn_gpu.fwd(...)`，把 `causal/window_size/softcap/alibi/dropout` 透传给 kernel。FA4 的对外接口在 [[flash-attention:flash_attn/cute/interface.py]]（`_flash_attn_fwd`，[[flash-attention:flash_attn/cute/interface.py#L294]]），多了 `score_mod/mask_mod/block_sparse_tensors/num_splits/pack_gqa/m_block_size` 等可编程参数。

## 4. varlen 与 `cu_seqlens`

训练时一个 batch 里各样本长度不同，padding 到 max 会浪费算力（attention 的复杂度是 $N^2$）。varlen 的做法是：把 batch 里所有样本的 token 首尾相接拼成一条长序列 `[total_tokens, heads, d]`，用 `cu_seqlens`（cumulative sequence lengths，前缀和）标记每个样本的边界。

```
样本长度 [3, 5, 2]  →  cu_seqlens_q = [0, 3, 8, 10]   (total=10)
样本 i 的 token 区间 = [cu_seqlens[i], cu_seqlens[i+1])
```

kernel 据此为每个 m_block 定位它属于哪个样本，attention 只在样本内部进行（不跨样本）。FA4 的 `SeqlenInfo`（[[flash-attention:flash_attn/cute/seqlen_info.py#L18-L45]]）：

```python
class SeqlenInfo:
    offset = 0 if cu_seqlens is None else cu_seqlens[batch_idx]               # :32  本样本起点
    seqlen = ... cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]            # :42  本样本长度
```

`offset_batch`（[[flash-attention:flash_attn/cute/seqlen_info.py#L47]]）把全局张量按 `offset` 切到本样本的视图。好处是零 padding 浪费：长短样本混在一起，也只各算各的 $N^2$。`max_seqlen` 仍需要传入（用来确定 grid 的 m_block 上界）。这套做法和 CP 的 `PackedSeqParams`、Megatron 的 packed sequence 是同一种思想。

## 5. GQA / MQA：`pack_gqa`

GQA 里 KV head 数 `nheads_kv` 远小于 Q head 数 `nheads_q`（如 32 个 Q head 共享 4 个 KV head，`qhead_per_kvhead=8`）。朴素实现要么把 KV broadcast 成 32 份（浪费带宽和显存），要么给每个 KV head 起独立 kernel。

FA 的 `pack_gqa` 把共享同一个 KV head 的多个 Q head 打包进一次 KV 读取。`pack_gqa_layout`（[[flash-attention:flash_attn/cute/pack_gqa.py#L15-L40]]）把 `(qhead_per_kvhead, seqlen)` 折进 m 维，于是一个 CTA 载入一次 $K_j/V_j$ 就能服务 `qhead_per_kvhead` 个 Q head 的 $QK^{\top}$——**KV 只从 HBM 读一次**，K/V 带宽降为原来的 `1/qhead_per_kvhead`。这对 decode 尤其关键，因为 decode 的瓶颈正是 KV 带宽。`make_packgqa_tiled_tma_atom`（[[flash-attention:flash_attn/cute/pack_gqa.py#L43]]）保持打包后的 TMA 维度不变（仍是 4D），不增加 descriptor 的复杂度。

## 6. paged KV 与 kvcache

推理 serving 的 attention 和训练长得很不一样，下面分两部分讲。

### 6.1 paged KV

KV cache 不连续存放，而是切成固定大小的 block（page），用 `block_table` 记录每个样本的逻辑 block 到物理 block 的映射（类似虚拟内存页表，即 PagedAttention 的做法）。好处是不同长度的序列能共享、复用显存页，没有外部碎片。FA 的 `flash_attn_with_kvcache(..., block_table=...)`（[[flash-attention:flash_attn/flash_attn_interface.py#L1485]]）和 FA3 的 `PagedKVNonTMA` 路径（[[flash-attention:hopper/paged_kv.h]]、[[flash-attention:flash_attn/cute/paged_kv.py]] 的 `PagedKVManager`）处理「按 block_table 间接寻址 KV」这件事。因为 page 不连续，K/V 可能要用 `cp.async` 而非 TMA（[[flash-attention:hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L211]]）。

### 6.2 kvcache 与 split-KV

decode 时 `seqlen_q=1`（一次生成一个 token），但 `seqlen_k` 可能有几万。`flash_attn_with_kvcache` 还能就地把新的 k,v 写进 cache（incremental decoding，[[flash-attention:flash_attn/flash_attn_interface.py#L1519]]），并顺带完成 RoPE。

`seqlen_q=1` 时沿 Q 并行退化（02 第 2 节），于是走 split-KV：把长 KV 切成 `num_splits` 段，每段一个 CTA 算出 partial 的 $(O, \mathrm{LSE})$，再用 combine kernel 按 LSE 归并（`num_splits_heuristic`，[[flash-attention:flash_attn/cute/interface.py#L257]]）。这是 decode 阶段把 SM 喂饱的唯一办法。

> 一个容易踩的语义坑：`flash_attn_with_kvcache` 的 causal mask 对齐到右下角（bottom-right），不是左上。`seqlen_q=2, seqlen_k=5` 时 query 能看到前 4/5 个 key（[[flash-attention:flash_attn/flash_attn_interface.py#L1530]]）。因为 decode 的 query 是序列末尾的 token，它应当能 attend 到几乎全部历史 KV。训练用的 `flash_attn_func` 则是左上对齐。

## 7. 其余 feature 在代码里的落点

| feature | 含义 | 落点 |
|---|---|---|
| **sliding window / local** | query $i$ 只 attend $[i - w_{\mathrm{left}}, i + w_{\mathrm{right}}]$ | `window_size=(left,right)`；kernel 用它裁 `n_block_min/max`（[[flash-attention:csrc/flash_attn/src/flash_fwd_kernel.h#L83-L87]]），`mask.py` 的 local 分支 |
| **softcap** | $s' = c \cdot \tanh(s / c)$，防 logit 爆 | 预乘进 scale（03 第 5 节，[[flash-attention:hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L543-L562|mainloop...:543-562]]） |
| **ALiBi** | 给 logit 加位置线性偏置 | `alibi_slopes`；`Mask` 类承载（[[flash-attention:csrc/flash_attn/src/flash_fwd_kernel.h#L288]]） |
| **dropout** | attention 概率上做 dropout | `dropout_p` + `rng_state`（backward 重放同一随机数，exact grad），[[flash-attention:flash_attn/flash_attn_interface.py#L873|interface.py:873]] |
| **deterministic** | $dQ$ 跨块累加可复现 | `ctx.deterministic`（02 第 5 节） |
| **score_mod / mask_mod** | 用户自定义 logit 变换 / 掩码（FlexAttention 式） | FA4 `@cute.jit` callable，编译期注入（[[flash-attention:flash_attn/cute/flash_fwd.py#L1147]]），`mask.py` 的 `AttentionMask` |

## 8. 小结

FA4 证明了「高性能 kernel 不必是不可维护的 C++ 模板堆砌」——CuTeDSL 让同一套 FA3 调度用 Python 表达、JIT 特化、跨 Hopper/Blackwell 复用。而 varlen / GQA pack / paged KV / split-KV 这四件事，是 FA 从「论文里的单 head attention」变成「撑起整个训练与 serving 栈的计算底座」所必需的工程接口：

- **训练**：varlen 拼接、零 padding 浪费，FA 是 Megatron/TE 的默认 attention kernel。
- **推理 prefill**：长 `seqlen_q`，沿 Q 并行，paged KV 管理显存。
- **推理 decode**：`seqlen_q=1`，split-KV 喂满 SM，pack_gqa 节省 KV 带宽，kvcache 就地更新。

回到 [`README`](./README.md) 第 5 节的系统位置图：CP 把 FA 的 KV 循环扩展到跨卡，serving 把 FA 包进 paged/split-KV，但最里层那个不变量永远是 01 的 online softmax 加 tiling。

---

下一篇：[05 · Flash Sparse Attention](./05_flash_sparse_attention.md) —— 同一套 tiling 加 online softmax，换成不规则 gather：NSA / MoBA / DSA。

做 lab：[[atlas:docs/attention/fa/fa_lab.ipynb]] —— 在 Mac CPU 上用纯 torch 手写 forward（online softmax 逐块累加 + LSE）和 recomputation backward，逐元素对齐 PyTorch SDPA，并量化「不物化 $[N, N]$」省下的显存。
