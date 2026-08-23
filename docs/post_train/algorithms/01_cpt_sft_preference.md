# CPT、SFT 与 preference learning

## 前置知识

本篇讨论在线 RL 之前的三个阶段：CPT、SFT 与 preference learning。三者的核心算子与 pretraining 相同，差异完全来自数据分布与 loss mask。阅读前建议：

- 已读[算法 overview](README.md)，知道 `tokens/logits/loss_mask` 的 shape。
- 熟悉 next-token prediction 与 cross entropy；可回看 [torch compute ops](../../torch/02_compute_ops.md)。

## 1. 同一个 NLL，三种数据语义

先设一个 token sequence $z=(z_1,\dots,z_T)$，模型在位置 $t-1$ 产生 `logits[t-1]`（shape 为 $[V]$），用来预测下一个 token $z_t$。这样一来，masked negative log-likelihood 就可以写成：

$$
\mathcal L_{\mathrm{NLL}}
=-\frac{\sum_{t=2}^{T}m_t\log\pi_\theta(z_t\mid z_{<t})}
{\max(1,\sum_{t=2}^{T}m_t)}.
$$

这里的 $m_t$ 本身不参与求导，它只决定哪些 token 会被计入梯度；softmax 和 gather 操作则都沿着 vocabulary 维 $V$ 进行。CPT 和 SFT 用的其实是完全相同的核心算子，两者的区别只来自训练数据本身和 mask 的取法：

| 阶段 | sequence | 常见 $m_t$ | 模型在学什么 |
| --- | --- | --- | --- |
| pretraining | raw corpus | 几乎所有非 padding token 为 1 | 通用语言/世界分布 |
| CPT | domain/time/language corpus | 同 pretraining | 目标分布的知识、风格与 token statistics |
| SFT | instruction + demonstration | 通常仅 assistant response 为 1 | 给定 instruction/history 时应采取的行为 |

### 1.1 CPT：continued pre-training

CPT，也常被写作 continual pretraining 或 domain-adaptive pretraining，指的是从已有的 base checkpoint 出发继续做 causal LM 训练。它比较适合下面这几种情况：

- 新领域 raw corpus 很多，但高质量 instruction/answer 很少；
- 需要补充新语言、代码栈、时间窗口或私有知识；
- 希望先改变 representation，再用少量 SFT 恢复可交互行为。

这样做的风险是 catastrophic forgetting 和 behavior drift，也就是模型在学习新领域知识的同时，可能会遗忘或偏离原有的通用能力。工程上常见的应对办法是混入一部分 replay 或者 general corpus，同时调低 learning rate，并持续监控通用 benchmark 和 domain perplexity 这两个指标，CPT 结束之后通常还要重新做一遍 SFT 或 preference alignment。这里有一个容易踩的坑：如果只是简单地把 instruction 数据去掉 prompt mask 拿来做 CPT，并不会自动得到高质量的结果，因为 chat template、重复出现的 assistant 语气风格，以及数据 packing 的方式，都会悄悄改变数据本身的分布。

### 1.2 SFT：response-masked behavior cloning

一个多轮样本可写成：

```text
[system][user_1][assistant_1][tool][user_2][assistant_2]
    0        0         1        0      0          1      <- loss mask
```

是否要对历史轮次里的 assistant token 计算 loss，属于 recipe 层面的选择，不同团队做法不同；但 tool 或 environment 返回的 observation 通常必须把 mask 设为 0，不能让模型学着去模仿这部分内容。SFT 本质上是一种 teacher forcing：训练时，`assistant_2` 这一轮的前缀直接来自 ground truth，而部署推理时前缀却来自模型自己上一轮生成的内容（teacher forcing 的完整定义与它为什么能让训练一次 forward 算完全部位置的 loss，见 [训练全景：从数据到权重更新](../../train/00_overview.md) §2.5）。这个差异带来的后果是，轨迹越长，累积的 exposure bias 就越明显。

在 slime 的实现里，SFT 先从 response spans 里取出 target log-prob，再套用负号做 sample 或 token 层面的 reducer，具体在 [[slime:slime/backends/megatron_utils/loss.py#L1233-L1280]]；同一文件的 `:1283-1342` 是统一的 dispatch 逻辑，负责在 `sft_loss`、`policy_loss` 和 `value_loss` 之间做选择。这说明 SFT 完全可以复用 RL 用到的同一套 Megatron data/parallel stack，只是它不需要 rollout reward、advantage、old policy 或者 critic 这些额外的东西。

### 1.3 Offline distillation

如果 teacher 先离线生成 hard response，再拿这些 response 做 SFT，那么优化的其实是 teacher trajectory 上的 NLL。如果能保存 teacher 完整的输出分布，还可以进一步换成 forward KL：

$$
D_{\mathrm{KL}}(\pi_T\|\pi_\theta)
=\sum_{v=1}^{V}\pi_T(v\mid h)\log\frac{\pi_T(v\mid h)}{\pi_\theta(v\mid h)}.
$$

无论用哪种形式，这里的期望都是在 teacher 或者数据的前缀分布上取的，而不是在 student 当前的 policy 上取的。一旦 student 在推理时偏离了这些离线前缀，就再也得不到任何 supervision 信号了——这正是 [OPD](04_opd.md) 要改变的地方。

## 2. 从 demonstration 到 preference

demonstration 要求标注者亲自写出一个完整的好答案，成本高，而且写出来的往往只代表一种风格；相比之下，preference 只需要标注者比较两个候选答案，判断 $y^+ \succ y^-$，这种“哪个更好”的判断要容易表达得多。经典路径是先训练一个 reward model，再用 PPO 去优化；direct preference methods 则跳过 reward model 这一步，直接更新 policy。

### 2.1 Reward model

reward model 的作用是给完整的 $(x,y)$ 输出一个标量 $r_\phi(x,y)$。按照 Bradley–Terry 模型的假设，chosen 答案胜出的概率可以写成：

$$
p_\phi(y^+\succ y^-\mid x)
=\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)),
$$

对应的 loss 则是：

$$
\mathcal L_{\mathrm{RM}}
=-\log\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)).
$$

注意这个 loss 只约束了 reward 的差值，并没有校准 reward 的绝对大小。当 RM 被用在在线 RL 里时，policy 会主动去寻找 RM 判断上的漏洞并加以利用，这就是常说的 reward hacking，因此实践中通常还需要 held-out preference 集合、reward ensemble 或者不确定性估计、rule-based 的 verifier，以及 adversarial audit 这几道防线。

### 2.2 DPO：把隐式 reward 代回 preference loss

DPO 的做法是使用 policy 相对于 reference 的 log-ratio：

$$
s_\theta(x,y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x),
$$

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\left(\beta[s_\theta(x,y^+)-s_\theta(x,y^-)]\right).
$$

这里的 $\log\pi_\theta(y\mid x)=\sum_t \log\pi_\theta(y_t\mid x,y_{<t})$，所以 response 的长度、以及是用 sum 还是 mean 来聚合，仍然会影响最终的优化结果。DPO 不需要在线 rollout、不需要单独 serving 一个 RM，也不需要 critic，因此它的 infra 需求非常接近 SFT；代价是它只能学习到固定 preference dataset 里已经覆盖到的那些 pair，没办法像在线 RL 那样在 environment 里主动搜索新的策略。

## 3. 数据工程中的常见风险

很多时候，真正让训练目标偏离预期的不是 loss 公式本身，而是数据工程里的一些细节。下面几类问题尤其容易被忽略。

### 3.1 Template 与 token identity

tokenizer、chat template、special token 和 tool schema 这几样东西必须固定下来，不能中途改变。一个容易被忽视的坑是：如果把已经采样出来的文本先 decode 成字符串，再重新 tokenize 一遍，token 的边界可能会发生变化。在普通 SFT 里，这种变化有时只会影响数据质量；但在 RL 或者 OPD 里，它还会导致之前保存下来的 log-prob 和实际的 action token 对不上号。

### 3.2 Packing 与 denominator

把多个样本 pack 成一条 $[1,T]$ 的序列可以提高 GPU 利用率，但这样做的前提是 attention boundary、position id 和 loss mask 都必须严格阻止信息跨样本泄漏。写 loss 的时候，应该先想清楚要定义的到底是哪一种目标，再去实现，常见的选择有：

- global token mean：对所有有效 token 求和再除总 token；
- sample mean：每样本先平均，再对样本平均；
- source-balanced mean：各 source 先归约，再加权。

在分布式训练里，这里的 denominator 必须跨 DP/CP 正确地做 all-reduce，才能得到全局一致的结果。slime 在 [[slime:slime/backends/megatron_utils/loss.py#L1317-L1325]] 构建了这个 reducer，并且在 `:1344-1350` 通过引入一个零值依赖，确保即使某个 CP rank 算出来的 loss 是空的，它也依然会参与到 backward 的 collective 通信里，不会因为某些 rank 缺席而导致集合通信出问题。

### 3.3 多轮 mask

多轮样本中各字段是否参与 loss，需要显式记录：

| 字段 | 是否给 loss | 原因 |
| --- | --- | --- |
| system/user | 通常否 | condition，不是 action |
| model assistant output | 是 | policy action |
| tool call JSON | 是，若模型生成 | action |
| tool result/environment observation | 否 | 外部状态，不应让模型模仿 |
| rejected response | DPO loss 中参与；SFT 通常否 | negative comparison |
| padding/截断后无效 token | 否 | 非真实 token |

## 4. 何时不应急于使用 RL

把上面几节的内容串起来，可以得到几条粗略的判断标准，帮助决定现在是不是应该直接上在线 RL：

- base model 在目标任务成功率近零：先 CPT/SFT/curriculum，让 rollout 能探索到成功；
- 有稳定的高质量 demonstrations，目标主要是格式/知识注入：SFT 更直接；
- 只有离线 preference pair，在线 reward/environment 不可靠：先 DPO family；
- 目标是从强 teacher 迁移 dense token signal：比较 offline distillation 与 [OPD](04_opd.md)；
- 只有可验证 outcome 且需要模型自行发现策略：进入 GRPO/PPO 等在线 RL。

## 5. 实践检查单

最后给一份实践中可以直接对照的检查清单：

1. 抽样 decode `tokens` 并着色显示 `loss_mask`，确认 tool/user token 为 0；
2. 对单样本手算 NLL，与 distributed loss 对齐；
3. 分别记录 token mean 与 sample mean，避免 denominator 静默改变；
4. 记录 source、length、language、template 版本；
5. CPT 同时看 domain gain 与 general retention；SFT 同时看 teacher-forced loss 与 free-running evaluation；
6. preference 数据检查 position bias、length bias、annotator agreement 与 tie handling。

## 参考

- Gururangan et al., [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964), 2020.
- Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155), 2022.
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023.

---

**下一篇**：讲完这些在线 RL 之前发生的事情，下一篇进入 [PPO](02_ppo.md)，看它怎样把一条 sequence reward 拆解成逐 token 的 advantage，以及 actor、critic、reference、reward 这四类模型各自承担什么角色，数据又是怎样在它们之间流动的。
