# 03 · autograd：引擎、自定义 Function、hooks、checkpoint

> 写 LLM 框架绕不开 autograd：自定义算子要自己写反向，并行通信要包成可微算子，省显存要做 gradient checkpointing，调试的时候还得挂 hook 去看梯度。这一篇讲清楚 autograd 的工作机制，以及你会真正动手用到的那些接口。

---

## 1. autograd 引擎的工作机制

每个 `requires_grad=True` 的张量参与运算时，autograd 都会在背后建一张 DAG（有向无环图）：图上的节点是 `Function`，对应每个 op 的反向逻辑；边则是张量本身。调用 `.backward()` 时，autograd 会从 loss 出发反向遍历这张图，对每个 op 调用它的 `backward`，用链式法则把梯度一路累积到叶子张量的 `.grad` 上。

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()
x.grad            # dy/dx = 2x
x.grad_fn         # None（x 是叶子）
(x**2).grad_fn    # <PowBackward0>，记录了反向怎么算
```

这里面涉及几个关键概念：

| 概念 | 含义 |
|---|---|
| **leaf tensor** | 用户直接创建、`requires_grad=True` 的张量（参数）。梯度累积到它的 `.grad` |
| **`grad_fn`** | 非叶子张量记录"我是哪个 op 产生的"，反向时调用它 |
| **`requires_grad`** | 张量是否需要梯度。任一输入 require，输出就 require |
| **`is_leaf`** | 是不是叶子 |
| **saved tensors** | forward 时被保存、backward 要用的中间张量（如 `x**2` 的反向要 `x`） |
| **version counter** | 检测张量被 in-place 改动，防止反向用到被污染的值 |
| **retain_graph** | `backward` 后默认释放图；多次反向同一图要 `retain_graph=True` |

有一点需要特别记住：梯度是累加的。`.backward()` 每次都会把新算出的梯度加到 `.grad` 上，而不是覆盖掉旧值，所以每个 iteration 开始前都要调用一次 `optimizer.zero_grad()`（用 `set_to_none=True` 会更省显存）。grad accumulation，也就是攒够好几个 micro-batch 的梯度再统一 step 的做法，正是利用了这个累加的性质。

---

## 2. backward vs grad：两种触发反向的方式

```python
# 方式 1: .backward()，梯度写进各叶子的 .grad（训练主流程）
loss.backward()                  # 标量直接调
loss.backward(retain_graph=True) # 保留图，可再次 backward

# 非标量要传 grad_outputs（上游梯度）
y.backward(gradient=torch.ones_like(y))

# 方式 2: torch.autograd.grad()，返回梯度而不写 .grad
grads = torch.autograd.grad(
    outputs=loss, inputs=[w1, w2],
    create_graph=True,           # 要二阶导（如 MAML、某些正则）时
    retain_graph=True,
)
```

| 选 backward | 选 grad |
|---|---|
| 标准训练，梯度进 `.grad`，optimizer 读 | 要梯度但不污染 `.grad`（如 meta-learning、影响函数、手动梯度操作） |
| 简单 | 要对部分输入求导、要高阶导、要函数式风格 |

`create_graph=True` 会让反向计算本身也变得可微，这是计算二阶导所必须的，但代价是计算图变大、速度变慢，显存开销也更高。

---

## 3. 自定义 `autograd.Function`

如果需要包一个第三方 CUDA kernel、把分布式通信变成可微操作、实现一个 torch 里没有的反向逻辑，或者在反向里做一些手动优化，这些场景都要靠自定义的 `autograd.Function` 来完成。

### 3.1 现代写法（split forward / setup_context）

PyTorch 2.x 推荐把 `ctx` 的设置从 `forward` 里拆出来单独处理，这样做是为了兼容 `torch.func` 和 vmap：

```python
class MyGELU(torch.autograd.Function):
    @staticmethod
    def forward(x):                          # 纯计算，不碰 ctx
        return x * 0.5 * (1 + torch.erf(x / 2**0.5))

    @staticmethod
    def setup_context(ctx, inputs, output):  # 决定保存什么
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_out):             # 接收上游梯度，返回对每个 input 的梯度
        (x,) = ctx.saved_tensors
        cdf = 0.5 * (1 + torch.erf(x / 2**0.5))
        pdf = torch.exp(-x**2 / 2) / (2 * torch.pi)**0.5
        return grad_out * (cdf + x * pdf)

y = MyGELU.apply(x)        # 用 .apply 调，不是直接调 forward
```

### 3.2 经典写法（forward 带 ctx）

老代码里更常见的写法是让 `forward(ctx, ...)` 一步到位：

```python
class ScaleBy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x)
        ctx.scale = scale                    # 非张量存成属性
        return x * scale

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # 返回值个数 = forward 的输入个数；不需要梯度的输入返回 None
        return grad_out * ctx.scale, None
```

写自定义 `Function` 时，有几条规则必须遵守：

1. `backward` 返回值的个数和顺序必须和 `forward` 的输入一一对应，不需要梯度的输入（比如标量、flag）要返回 `None`。
2. 要保存供反向使用的张量，必须用 `ctx.save_for_backward(...)`，而不是直接把它挂在 `ctx` 上当作普通属性——`save_for_backward` 会配合 version counter 一起检测 in-place 污染。只有非张量的标量或 flag 才适合直接存成属性。
3. 如果 `forward` 里做了 in-place 操作，要调用 `ctx.mark_dirty(...)`；返回的张量如果和输入共享内存，还要用 `mark_non_differentiable` 等机制显式声明。多数情况下，最好的做法是干脆不在 `Function` 里做 in-place。
4. 写完之后，用 `torch.autograd.gradcheck(fn, inputs)` 通过数值梯度来验证反向写对了没有（记得用 `double` 精度）。

```python
from torch.autograd import gradcheck
x = torch.randn(4, dtype=torch.double, requires_grad=True)
gradcheck(MyGELU.apply, (x,))    # True 表示反向正确
```

### 3.3 LLM 框架里的典型用途

- 把通信包成可微算子：Megatron 的 `f`/`g`（copy / all-reduce）本质上就是一对 `autograd.Function`，forward 时做的通信和 backward 时做的通信正好互为对偶（见 [TP/SP](../parallel/02_tp_sp/README.md)）。

```python
class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x)       # forward all-reduce
        return x
    @staticmethod
    def backward(ctx, g):
        return g                 # backward identity（reduce 的反向是 copy）

class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x        # forward identity
    @staticmethod
    def backward(ctx, g):
        dist.all_reduce(g); return g     # backward all-reduce（对偶）
```

- 融合 kernel 的反向：把 FlashAttention 或者自定义 RMSNorm 的 CUDA kernel 包进来，forward 调用 kernel 本身，backward 调用与之对应的 grad kernel。
- 用算力换显存的重算：在 backward 里重新计算被丢弃的中间量，gradient checkpointing 的底层实现，正是一个这样的 `Function`（见第 6 节）。

---

## 4. 关闭梯度：no_grad / inference_mode / detach

| 方式 | 作用 | 何时用 |
|---|---|---|
| `with torch.no_grad():` | 块内不建图、不追踪 | optimizer step、eval、不需要反向的前向 |
| `with torch.inference_mode():` | 比 no_grad **更彻底**，连 version counter 都不建，张量不能再进 autograd | 纯推理（vLLM/SGLang 风格），最快 |
| `x.detach()` | 返回共享数据但脱离图的新张量 | 截断梯度、存日志、EMA、stop-gradient |
| `x.requires_grad_(False)` | 原地改 flag | 冻结参数 |
| `@torch.no_grad()` | 装饰器形式 | 整个函数不建图 |

```python
# 推理框架：全程 inference_mode
with torch.inference_mode():
    logits = model(input_ids)

# EMA / target network: detach 防止梯度流过去
target = teacher(x).detach()

# 冻结 embedding
model.embed.weight.requires_grad_(False)
```

有一点容易踩坑：在 `inference_mode` 下创建的张量会带一个特殊标记，之后不能再拿到普通的 autograd 计算里复用，会直接报错。如果一个张量需要先在推理模式下生成、之后又要参与训练（比如先生成数据再算 loss），应该用 `no_grad` 而不是 `inference_mode`，或者对结果做一次 `.clone()`。

---

## 5. hooks：在反向时观察和修改梯度

hook 是调试梯度问题（比如 NaN、梯度爆炸）以及实现梯度裁剪、通信 overlap 的重要工具。

| hook | 挂在哪 | 触发时机 | 用途 |
|---|---|---|---|
| `tensor.register_hook(fn)` | 张量 | 该张量梯度算出时 | 看/改单个张量的梯度 |
| `module.register_full_backward_hook(fn)` | nn.Module | 模块反向完成 | 看模块输入/输出梯度 |
| `module.register_forward_hook(fn)` | nn.Module | 前向完成 | 抓激活、改输出 |
| `param.register_post_accumulate_grad_hook` | 参数 | 梯度累积到 `.grad` 后 | **DDP/FSDP 的梯度通信 overlap 就靠它** |

```python
# 调试：发现哪一层梯度出 NaN
def check_nan(grad):
    if torch.isnan(grad).any():
        print("NaN gradient!")
    return grad                    # 返回值会替换原梯度（返回 None 则不改）
h = some_tensor.register_hook(check_nan)
# ... backward ...
h.remove()                         # 用完移除

# DDP 风格：参数梯度一算好就异步 all-reduce（overlap 反向与通信）
def allreduce_hook(param):
    dist.all_reduce(param.grad, async_op=True)
param.register_post_accumulate_grad_hook(allreduce_hook)
```

> DDP 的 bucket 机制和反向通信 overlap、FSDP 的 reduce-scatter，底层其实都是靠 grad hook 在反向过程中触发通信，从而让通信和计算重叠起来。原理见 [DP](../parallel/01_dp/README.md)。

---

## 6. gradient checkpointing：用算力换显存

大模型训练时显存的大头是激活值——每一层 forward 产生的中间张量都要留到 backward 才会用到。checkpointing 的思路是反过来：forward 时不保存某一段的中间激活，等到 backward 真正需要的时候，再重新跑一遍 forward 把它算出来。

```python
from torch.utils.checkpoint import checkpoint

# 把一个 transformer layer 包起来：forward 只存输入，反向时重算
def forward(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x, use_reentrant=False)   # 关键参数见下
    return x
```

用它时要注意几点：

- `use_reentrant=False` 是应该采用的新版（non-reentrant）实现。旧的 reentrant 版本对 `requires_grad`、多输出、RNG 有不少限制，新代码一律建议用 `False`。
- 代价是要多做一次 forward，大致多花 33% 左右的算力，换来的是激活显存的大幅节省。通常做法是把每个 transformer block 整体包一层 checkpoint。
- RNG 的一致性也需要保证：重算的时候，dropout 这类随机 op 必须复现和原始 forward 完全一样的随机数。checkpoint 默认开启 `preserve_rng_state=True`，会自动保存和恢复 RNG 状态，确保重算结果和原来一致。
- 2.x 还提供了选择性 checkpoint：`torch.utils.checkpoint` 配合 SAC（selective activation checkpointing），可以只重算便宜的 op（比如 norm、激活函数），而保留那些计算代价高的结果（比如 matmul 的输出），从而更精细地权衡算力和显存。

> Megatron 和 FSDP 都内置了对 activation checkpointing 的封装，但底层用的就是这个 `checkpoint`。想把它和 PP 的显存权衡放在一起看，可以参考 [03 · 显存、通信 overlap 与并行协同](../parallel/03_pp/03_overlap_and_memory.md)。

---

## 7. 梯度裁剪与常见训练操作

```python
# 全局梯度范数裁剪（防爆炸，几乎所有 LLM 训练都用）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 清梯度（set_to_none 更省显存且更快，2.x 默认 True）
optimizer.zero_grad(set_to_none=True)

# 手动 SGD step（在 no_grad 下，in-place 更新参数）
with torch.no_grad():
    for p in model.parameters():
        if p.grad is not None:
            p.add_(p.grad, alpha=-lr)
```

`clip_grad_norm_` 在分布式场景下有一点要小心：每张卡通常只持有部分梯度（TP/PP/FSDP 都是这样），要算出全局的梯度范数，需要先跨卡把各自梯度的平方和做 all-reduce，再统一开方——主流框架（Megatron、FSDP）会自动处理这一步，但如果自己手写并行逻辑，一定不要漏掉这个环节。

---

## 8. `torch.func`：函数式变换

PyTorch 2.x 把 functorch 并入了 `torch.func`，提供了一套可组合的函数式 autograd 变换。在 LLM 场景里，它主要用于 per-sample gradient（比如差分隐私、影响函数的计算）、高效计算 Jacobian/Hessian，以及 ensemble 相关的场景。

| 变换 | 作用 |
|---|---|
| `torch.func.grad(f)` | 返回"计算 f 梯度的函数"（函数式，不用 `.backward`） |
| `torch.func.vmap(f)` | 自动向量化：把对单样本的函数批量化，无显式 batch 循环 |
| `torch.func.jacrev / jacfwd` | 反向/前向模式 Jacobian |
| `torch.func.functional_call` | 用外部传入的参数 dict 跑 module（无状态化） |
| `torch.func.grad_and_value` | 同时返回梯度和函数值 |

```python
from torch.func import grad, vmap, functional_call

# per-sample gradient: 对 batch 里每个样本单独求梯度
def compute_loss(params, x, y):
    pred = functional_call(model, params, (x,))
    return F.cross_entropy(pred, y)

params = dict(model.named_parameters())
per_sample_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(params, xs, ys)
```

需要注意的是，`vmap`/`grad` 对自定义 `autograd.Function` 有一定要求，这也正是前面 3.1 节现代写法要把 `setup_context` 拆出来的原因——只有这样 vmap 才能顺利往里注入 batch 维。如果一个函数完全由纯 torch op 组成，则不需要任何额外改动，开箱即用。

---

## 9. 调试 autograd 的实用开关

```python
# 反向出错时定位是哪个 forward op 引发的（生产环境别开，很慢）
torch.autograd.set_detect_anomaly(True)

# 检查反向是否正确（数值梯度对比）
torch.autograd.gradcheck(fn, inputs)        # 一阶
torch.autograd.gradgradcheck(fn, inputs)    # 二阶

# 看一个张量是不是 view（in-place 排查）
x._base is not None

# 临时关掉/打开 grad（嵌套）
with torch.set_grad_enabled(flag):
    ...
```

`set_detect_anomaly` 会让 forward 记录下每个 op 的调用栈，一旦反向报错，就能直接告诉你是哪一行 forward 代码引发的问题。排查"哪个 in-place 操作破坏了反向""哪里出现了 NaN"这类问题时非常好用，但开销不小，只建议在调试时打开。

把单卡上的 autograd 机制讲清楚之后，下一个自然的问题是：这些前反向要怎么扩展到多卡上？这就要用到 collective 通信原语、它们各自的可微对偶，以及组织多维并行拓扑的 DeviceMesh，也就是 [04 · torch.distributed：通信原语、process group、DeviceMesh](./04_distributed.md) 的内容。
