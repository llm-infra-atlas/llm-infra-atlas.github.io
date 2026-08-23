# 02 · 计算 op：matmul / einsum / reduction / gather-scatter / SDPA

> 上一篇把 Tensor 的内存布局讲清楚了，这一篇往上走一层，看真正用来做计算的那些 op：写 transformer、MoE、采样、归一化时会反复用到的高频算子。重点会放在每个 op 的语义边界、广播规则、index 类操作的方向该怎么判断，以及它们在 LLM 里的典型用法上。

---

## 1. matmul 家族与广播规则

LLM 训练和推理里，绝大部分（大约 90%）的 FLOPs 都花在下面这几个 op 上，它们之间的区别，说到底全在于各自怎么处理 batch 维。

| op | 语义 | batch 维 | 典型用途 |
|---|---|---|---|
| `torch.mm(A, B)` | 严格 2D × 2D | 无 | 纯矩阵乘，shape 校验最严 |
| `torch.bmm(A, B)` | 3D × 3D，逐 batch | 必须显式且相等 | `[b,n,m]×[b,m,p]` attention score |
| `torch.matmul(A, B)` / `@` | 任意维，自动广播 | **广播** | 通用，最常用 |
| `F.linear(x, W, b)` | `x @ W.T + b` | 广播 | nn.Linear 的底层，**注意是 W 的转置** |
| `torch.addmm(bias, A, B)` | `bias + A@B` | 无（2D） | GEMM+bias 融合，比分开写快 |
| `torch.baddbmm(bias, A, B)` | batched 版 addmm | 3D | attention 里 `bias + Q@K^T` |
| `torch.einsum(eq, ...)` | 任意收缩 | 由公式定 | 不规则乘法、可读性 |

`matmul` 的广播规则值得牢牢记住：它只对最后两维做矩阵乘，前面的维度全部按广播规则对齐。

```python
A = torch.randn(8, 4, 16, 64)    # [b, heads, s, d]
B = torch.randn(8, 4, 64, 16)    # [b, heads, d, s]
(A @ B).shape                    # [8, 4, 16, 16]，前两维对齐，后两维矩阵乘

# 向量与矩阵：matmul 会自动处理（mm 不行）
v = torch.randn(64)
M = torch.randn(64, 16)
(v @ M).shape                    # [16]
```

`F.linear` 有一个容易踩的转置陷阱：`nn.Linear(in, out)` 的权重 shape 是 `[out, in]`，`F.linear` 内部实际做的是 `x @ W.T`。手写 TP 切权重的时候，column-parallel 切的是 `out` 维，也就是 W 的第 0 维；row-parallel 切的是 `in` 维，也就是 W 的第 1 维——不要和裸 matmul 的语义搞混了。

```python
x = torch.randn(2, 16, 512)      # [b, s, in]
W = torch.randn(2048, 512)       # [out, in]
F.linear(x, W).shape             # [2, 16, 2048]，等价 x @ W.T
```

精度方面也有一点要留意：bf16 矩阵乘的累加默认发生在 fp32（这是硬件 tensor core 的行为），所以单个 GEMM 的数值通常还算稳；但如果是手动循环累加，比如梯度累加或者某些 reduction，就需要自己管理累加精度，不能指望硬件替你兜底。`torch.backends.cuda.matmul.allow_tf32` 这个开关控制的是要不要用 TF32 来加速 fp32 matmul。

---

## 2. einsum：不规则乘法与可读性

`torch.einsum` 用爱因斯坦记号来表达任意的张量收缩，写 attention、MoE、RoPE 这类逻辑时能大幅提升代码的可读性。规则很简单：重复出现的下标会被求和（也就是被收缩掉），只有出现在输出里的下标才会保留。

```python
# attention score: [b, h, sq, d] × [b, h, sk, d] -> [b, h, sq, sk]
scores = torch.einsum("bhqd,bhkd->bhqk", q, k)

# 加权求和 value: [b, h, sq, sk] × [b, h, sk, d] -> [b, h, sq, d]
out = torch.einsum("bhqk,bhkd->bhqd", attn, v)

# MoE: 每个 token 用自己的 expert 权重（grouped）
#   tokens [n, d_in], expert_w [e, d_in, d_out], 每 token 分到 expert e_i
#   先按 expert 分组后用 bmm 更高效，einsum 适合表达逻辑
y = torch.einsum("nd,ndo->no", x, w_per_token)   # 概念示意

# RMSNorm 里的均方：沿最后一维
ms = torch.einsum("...d,...d->...", x, x) / x.shape[-1]
```

用 einsum 时有几点需要记住：

- einsum 写起来清晰，但性能不一定总是最优——它内部会自行选择收缩顺序，遇到复杂表达式时可能不如手写 `bmm` 加 reshape 快。在性能敏感的路径上，最好实际 benchmark 一下再决定用不用。
- `...` 表示"剩余的任意多个维度"，用它来做 batch 广播非常方便。
- 只要 einsum 不需要做隐式的转置拷贝，它的执行就是高效的；一旦表达式要求物理上重排数据，就会触发一次拷贝。

---

## 3. reduction：sum / mean / max / norm 与数值稳定

```python
x.sum(dim=-1, keepdim=True)      # keepdim 保留维度，便于广播回去
x.mean(dim=(1, 2))               # 多维同时 reduce
x.amax(dim=-1)                   # 只要值（max 还返回 indices，amax 不返回）
x.norm(dim=-1)                   # L2 norm，等价 sqrt(sum(x^2))
x.var(dim=-1, unbiased=False)    # LayerNorm 里用有偏方差
```

| 易错点 | 说明 |
|---|---|
| `keepdim=True` | reduce 后想广播回原 shape（如归一化）几乎总要它 |
| `max` vs `amax` | `max(dim)` 返回 `(values, indices)` namedtuple；只要值用 `amax` |
| 累加精度 | bf16 张量 `.sum()` 默认在 bf16 累加，长序列会掉精度。`x.sum(dtype=torch.float32)` 强制 fp32 累加 |
| 空张量 | `sum` 给 0，`max` 报错 |

softmax 和 loss 计算里离不开下面这三件保证数值稳定的工具：

```python
# softmax: 减最大值防 exp 溢出（F.softmax 内部已做）
F.softmax(logits, dim=-1)
F.log_softmax(logits, dim=-1)            # 比 log(softmax) 稳定，CE loss 用

# logsumexp: log(sum(exp(x)))，online softmax / flash attention 的核心
torch.logsumexp(scores, dim=-1)          # 内部减 max，不会溢出

# CE loss 直接用 logits，别自己 softmax 再 log
F.cross_entropy(logits.view(-1, V), target.view(-1))   # 内部 = log_softmax + nll
```

> flash attention 的 online softmax（见 [FlashAttention](../attention/fa/README.md)）本质上就是把 `logsumexp` 改造成一个可以增量更新的 running max 加 running sum。

---

## 4. 广播（broadcasting）：规则与显式控制

广播的规则是从最后一维向前对齐：每个维度要么相等，要么其中一个是 1（会被拉伸），要么其中一个维度根本不存在（视为 1）。

```python
x = torch.randn(8, 1, 512)       # [b, 1, h]
bias = torch.randn(512)          # [h]
(x + bias).shape                 # [8, 1, 512]

scores = torch.randn(2, 8, 128, 128)   # [b, h, sq, sk]
mask = torch.randn(128, 128)           # [sq, sk]
(scores + mask).shape                  # [2, 8, 128, 128]，mask 广播到每个 b,h
```

除了依赖隐式广播，还有几个显式的 API 可以用：

- `x.expand(shape)`：零拷贝广播（stride=0），只读用（见 [01](./01_tensor_memory_layout.md#2-view-vs-copy)）。
- `torch.broadcast_tensors(a, b)`：返回广播后的 view。
- `x.unsqueeze(dim)` / `x[:, None]`：手动插入维度来对齐，比隐式广播更清晰、更不容易出错。

框架里的经验是：attention mask、RoPE、bias 的对齐几乎都要靠广播完成。写的时候宁可多写几个 `unsqueeze` 或 `[None]`，把维度对齐的意图显式表达出来，也不要依赖"恰好能广播"这种侥幸——后一种写法在 batch=1 的时候会悄悄掩盖掉维度上的 bug，等 batch 变大才暴露出来，排查起来很麻烦。

---

## 5. indexing：gather / scatter / index_*

这一组 op 是最容易搞错方向的，但 MoE routing、KV cache、token 重排都离不开它们，值得仔细弄清楚每个 op 到底是"按 index 收集"还是"按 index 写回"。

### 5.1 `gather`：沿某维按 index 收集

以 dim=1 为例，语义是 `out[i][j][k] = input[i][index[i][j][k]][k]`。`index` 必须和 `input` 的维度数相同，并且 index 的 dtype 必须是 int64。

```python
# top-k routing: 从 logits 里取每个 token 选中的 expert 的分数
logits = torch.randn(n_tokens, n_experts)
topk_idx = logits.topk(2, dim=-1).indices          # [n, 2]
topk_val = logits.gather(dim=-1, index=topk_idx)   # [n, 2]，每 token 选中 expert 的 logit

# 取每个序列位置预测的 token 概率（用于 RL / ppl）
logprobs = F.log_softmax(logits, -1)               # [n, V]
chosen = logprobs.gather(-1, target[:, None]).squeeze(-1)  # [n]
```

### 5.2 `scatter_` / `scatter_add_`：按 index 写回或累加

这是 `gather` 的逆操作（下划线后缀，in-place），以 dim=1 为例，语义是 `self[i][index[i][j]][k] = src[i][j][k]`。`scatter_add_` 和 `scatter_` 的区别在于，前者是把值累加到目标位置，而不是直接覆盖，所以同一个目标位置被写多次时会求和。

```python
# 构造 one-hot / dispatch mask
mask = torch.zeros(n_tokens, n_experts)
mask.scatter_(1, topk_idx, 1.0)                    # 选中的 expert 位置置 1

# MoE combine: 把 expert 输出按 token 原位置散射累加回去
output = torch.zeros(n_tokens, d_model)
output.scatter_add_(0, token_idx[:, None].expand(-1, d_model), expert_out)
```

### 5.3 `index_select` / `index_add_` / `index_copy_`：按一维 index 整行操作

这一组比 gather 简单：index 是一维的，操作的对象是沿某个 dim 选出或写入的整行（也就是一整个 slice），而不是逐元素。

```python
# MoE dispatch: 把 token 按 expert 分组（gather 选中的行）
sorted_tokens = x.index_select(0, sort_idx)        # 等价 x[sort_idx]，但更明确

# scatter_add 的整行版：把分组结果累加回原位置
out = torch.zeros_like(x)
out.index_add_(0, sort_idx, grouped_result)        # out[sort_idx[i]] += grouped[i]

# index_copy_: 覆盖而非累加（KV cache 写入）
kv_cache.index_copy_(0, positions, new_kv)
```

### 5.4 advanced indexing：`x[idx]` / `x[mask]`

```python
x[idx]              # idx 是 int tensor，等价 index_select（但更灵活，多维）；拷贝
x[mask]             # mask 是 bool tensor，选出 True 的元素，拉平成 1D；拷贝
x[mask] = 0         # bool 索引赋值，in-place 写选中位置
x[i, j]             # 多个 index tensor，按位置配对（不是笛卡尔积！）
```

把这几个操作的方向记清楚，可以归纳成这样几条：

- `gather` 的输出形状等于 index 的形状，沿指定 dim"按 index 挑"。
- `scatter_` 的输入 src 形状等于 index 的形状，沿指定 dim"按 index 放"。
- index 张量必须是 int64 类型。
- `scatter_add_` / `index_add_` 是可微的，MoE 反向里经常用到；普通的 `scatter_`（覆盖式写入）的可微性则要格外小心。

> MoE 的完整 dispatch/combine 数据流见 [EP](../parallel/05_ep/README.md)，那里把 `gather/scatter_add/index_select` 串成了 router → permute → grouped GEMM → unpermute 的完整链路。

---

## 6. 拼接 / 拆分 / 重排

| op | 语义 | 拷贝 | LLM 用途 |
|---|---|---|---|
| `torch.cat([...], dim)` | 沿已有维拼接 | 是 | KV cache 增长、拼 QKV |
| `torch.stack([...], dim)` | 新建一维堆叠 | 是 | 堆 layer 输出、batch 化 |
| `torch.split(x, size, dim)` | 按大小切成多块 | 否（view） | 拆 QKV、拆 gate/up |
| `torch.chunk(x, n, dim)` | 切成 n 块 | 否（view） | SwiGLU 拆 gate/up |
| `torch.unbind(x, dim)` | 沿某维拆成 tuple | 否（view） | 遍历 batch / head |
| `torch.repeat_interleave` | 逐元素重复 | 是 | GQA：KV head 重复到 Q head 数 |
| `tensor.repeat(...)` | 整体平铺重复 | 是 | 慎用，和 expand 区分 |
| `torch.roll` | 循环移位 | 是 | RoPE 的旋转、shift |
| `torch.flip` | 翻转 | 是 | reverse 序列 |

```python
# 拆 QKV（一次 GEMM 出 q,k,v）
qkv = F.linear(x, w_qkv)              # [b, s, 3*d]
q, k, v = qkv.chunk(3, dim=-1)        # 三个 view，零拷贝

# SwiGLU
gate_up = F.linear(x, w_gate_up)      # [b, s, 2*ffn]
gate, up = gate_up.chunk(2, dim=-1)
y = F.silu(gate) * up

# GQA: KV head 数 < Q head 数，把 KV 重复
# k: [b, n_kv, s, d] -> [b, n_q, s, d]
k = k.repeat_interleave(n_q // n_kv, dim=1)
```

---

## 7. top-k / sort / argmax / cumsum

```python
# top-k expert routing
vals, idx = logits.topk(k=2, dim=-1)          # 值 + 索引
# argmax: greedy decoding
next_token = logits.argmax(dim=-1)
# argsort: 按 expert id 排序 token（MoE permute 的核心）
sort_idx = expert_id.argsort()
# cumsum: 计算每个 expert 的 token offset（grouped GEMM 的分组边界）
counts = torch.bincount(expert_id, minlength=n_experts)
offsets = counts.cumsum(0)                    # 每组结束位置
```

| op | 返回 | 用途 |
|---|---|---|
| `topk(k, dim)` | `(values, indices)` | 路由、beam、采样候选 |
| `argmax/argmin(dim)` | indices | greedy decode |
| `argsort(dim)` | 排序索引 | MoE token permute |
| `sort(dim)` | `(values, indices)` | 需要排序值时 |
| `bincount` | 每个整数值的计数 | 统计每 expert token 数 |
| `cumsum/cumprod(dim)` | 前缀和/积 | 分组 offset、序列打包边界 |
| `unique` / `unique_consecutive` | 去重 | 统计 |
| `nonzero` | 非零位置坐标 | 从 mask 取 index（注意会同步！） |
| `where(cond, a, b)` | 逐元素选择 | masked 替换、mask fill |

这里有一个坑需要特别提醒：`nonzero`、`unique`、`item()`、`tolist()`、`bool(tensor)` 这类 op 的输出形状是由数据的实际取值决定的，因此都会触发一次 device 到 host 的同步（CPU 必须等 GPU 算完才能知道返回结果有多大）。在热路径上，这样的同步是实实在在的性能杀手，而且还会打断 CUDA Graph 的 capture 过程。能避开就尽量避开。

---

## 8. 采样与随机：温度 / top-p / multinomial

decode 阶段常用的采样原语大致是这样的：

```python
logits = logits / temperature                  # 温度缩放
probs = F.softmax(logits, dim=-1)
# top-k 截断
v, _ = probs.topk(k, dim=-1)
probs[probs < v[..., -1:]] = 0
# 采样
next_token = torch.multinomial(probs, num_samples=1)   # 按概率抽

# top-p (nucleus): 排序后累积到 p 截断
sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
cum = sorted_probs.cumsum(-1)
sorted_probs[cum - sorted_probs > p] = 0       # 累积超过 p 的丢弃
```

随机性的控制主要靠 `torch.manual_seed`、`torch.Generator`（一个独立的 RNG 流，避免污染全局状态）和 `torch.cuda.manual_seed_all`。有一点在 TP 场景下容易被忽略：dropout 需要保证各卡要么完全一致、要么完全独立，Megatron 为此专门实现了一套 RNG tracker（见 [TP/SP](../parallel/02_tp_sp/README.md)）。

---

## 9. `F.*` 中的高频融合 op

写这类逻辑时，应该优先用 torch 自带的融合实现，而不是手写一串小 op 拼起来——融合实现意味着更少的 kernel、更少的中间张量，数值表现通常也更稳定。

| op | 用途 | 备注 |
|---|---|---|
| `F.scaled_dot_product_attention` | **整个 attention**（见下节） | 2.x 起，自动选 flash/mem-efficient 后端 |
| `F.layer_norm` / `F.rms_norm` | 归一化 | `rms_norm` 是 2.4+ 新增 |
| `F.silu` / `F.gelu` / `F.relu` | 激活 | gelu 有 `approximate='tanh'` |
| `F.cross_entropy` | 分类 loss | = log_softmax + nll，数值稳定 |
| `F.embedding` | 查表 | 等价 `weight[idx]`，但支持 padding_idx |
| `F.dropout` | dropout | 训练/推理由 `training` 控制 |
| `F.normalize` | L2 归一化 | cosine 相似度 |

### 9.1 `scaled_dot_product_attention`（SDPA）

2.x 版本的 SDPA 把整个 $\mathrm{softmax}(QK^{\top}/\sqrt{d} + \text{mask})\,V$ 融合成了一个 op，会自动派发到 FlashAttention、memory-efficient 或者 math 后端。写推理框架时，这基本就是默认的入口。

```python
# q,k,v: [b, n_head, s, head_dim]
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,        # 加性 mask（float）或 bool mask
    dropout_p=0.0,
    is_causal=True,        # 因果 mask，比传 mask 更快（后端走优化路径）
    scale=None,            # 默认 1/sqrt(head_dim)
)
```

用它时有几点要留意：

- 优先用 `is_causal=True`，而不是手动传一个下三角 mask，因为后端为因果场景准备了专门的 causal kernel，速度更快也更省显存。
- 可以显式控制走哪个后端：`with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION]):`。flash 后端对 dtype（fp16/bf16）、head_dim、对齐方式都有要求，一旦不满足就会静默回退到 math 后端，速度慢而且更费显存，调试性能时一定要确认自己确实走的是 flash 路径。
- GQA 场景下，2.5+ 的 SDPA 支持 `enable_gqa=True`，允许 KV head 数少于 Q head 数，省去了手动 `repeat_interleave` 这一步。
- 变长或者 packed 序列的情况，可以用 `attn_mask`，也可以用更专门的 FlashAttention varlen 接口（第三方库里的 `flash_attn_varlen_func`）；纯 SDPA 处理变长序列的效率不如专用 kernel。

> SDPA 内部实现的正是 [FlashAttention](../attention/fa/README.md) 里讲的 IO-aware online softmax。理解了那一篇，就能明白 SDPA 为什么不需要实例化一个 `[s,s]` 的中间 score 矩阵。

这些前向 op 讲完之后，自然要问的下一个问题是：它们的反向是怎么来的？如果要自己写一个带自定义反向的算子，又该怎么下手？这正是 [03 · autograd：引擎、自定义 Function、hooks、checkpoint](./03_autograd.md) 要讲的内容。
