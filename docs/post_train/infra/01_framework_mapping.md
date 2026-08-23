# 框架映射

## 前置知识

本篇把 slime 的组件逐个定位到代码：一个 rollout sample 从 SGLang 生成，到变成 Megatron 的 loss，中间经过哪些对象、函数与张量边界。阅读前建议：

- 已读 [RL infra 总览](README.md)；知道 rollout/train 是两个可能独立的 process group。
- 已读 [GRPO 及其变种：baseline、ratio、clip 与 reducer](../algorithms/03_grpo_family.md)，知道 reward 与 advantage 的 shape。

## 1. 组件边界

先把 slime 里各个组件的职责边界摊开来看：

| 层 | slime 组件 | 责任 | 不应承担的事 |
| --- | --- | --- | --- |
| orchestration | Ray `train.py` / `train_async.py` | placement、生命周期、step 顺序 | 不在 driver 内重算 loss |
| rollout control | `RolloutManager` | server、router、data source、sample conversion | 不把文本当唯一训练事实 |
| inference | SGLang engine/router | prefill/decode、KV cache、token/log-prob、route metadata | 不假设 Megatron kernel 数值相同 |
| buffer | `DataSource`/Ray object store/NIXL | prompt 与 trajectory 暂存、重试、partial recycle | 不丢 policy version/mask |
| training actor | `TrainRayActor` + Megatron actor | DP/TP/PP/CP forward/backward、optimizer | 不把 rollout rank 当 training rank |
| loss | [[slime:slime/backends/megatron_utils/loss.py]]、[[slime:slime/utils/ppo_utils.py]] | KL、advantage、policy/value/SFT loss | 不省略 denominator/empty rank 处理 |
| sync | `update_weight/*` | gather/convert/transport/reload | 不在 generation 中途 swap |

## 2. driver loop

slime 提供了两种 driver 循环，对应不同程度的同步性。

### synchronous `train.py`

```text
create placement groups
create rollout manager + SGLang engines
create actor/critic training groups
initial actor.update_weights()
for rollout_id:
    data = rollout_manager.generate()
    optional offload rollout
    critic.train(data) → values
    actor.train(data, values)
    save/eval
    optional offload train
    actor.update_weights()
```

对应源码见 [[slime:train.py#L9-L95]]。这里有一个关键的边界需要留意：`rollout_manager.generate` 返回的是已经转成 CPU tensor 的 data ref，而不是 GPU 上的 activation；重建 log-prob、advantage 和 loss 这些工作，都是训练 actor 自己负责完成的。

### pipelined `train_async.py`

这个版本的做法是，在当前 batch 训练的同时，提前把下一批 generation 提交出去，实现见 [[slime:train_async.py#L31-L40]]。但更新权重之前，必须先把正在进行的 generation future 全部 drain 掉，防止权重更新恰好切入某个请求的中间过程，对应的处理在 `:66-70`。严格来说，这属于“有限 staleness、周期性设置 barrier”这一类 async，而不是真正的 fully async。

## 3. 从 Sample 到 train_data

`Sample` 这个数据结构，是用户自定义的 generate/reward 逻辑和框架内部 data path 之间的契约。一个可以拿去训练的 sample，至少需要包含下面这些字段：

```python
Sample(
    index, group_index, rollout_id,
    prompt, tokens,
    response_length,
    loss_mask,
    rollout_log_probs,
    reward, status,
    metadata={"policy_version": ...},
)
```

### 3.1 采样阶段

SGLang response 里的 `output_token_logprobs` 会被直接追加到 sample 上，而不是从 response 的文本反过来做 tokenize，实现见 [[slime:slime/rollout/sglang_rollout.py#L152-L221]]。`partial_rollout` 要求 sample 的状态可以增量更新；[[slime:slime/rollout/sglang_streaming_rollout.py#L93-L167]] 给出了一个 streaming 的示例，每收到一个 SSE chunk 就写入一次 tokens 和 log-probs。

### 3.2 reward 与 group normalize

`RolloutManager._post_process_rewards` 把原始的 reward reshape 成 `[num_groups,G]` 的形状，再做 mean/std normalization，实现见 [[slime:slime/ray/rollout.py#L722-L747]]。这里需要注意，动态 filter 可能导致实际的 group 数和固定的 batch 大小不一致，代码里应该以实际的 group layout 为准，而不能想当然地用 `B*G` 去猜测 shape。

### 3.3 conversion

`_convert_samples_to_train_data` 会构建出下面这些字段：

```text
tokens              list[[T_i]]
response_lengths    list[int]
loss_masks          list[[L_i]]
rewards             list[float]
rollout_ids         list[int]
rollout_mask_sums    list[float]  # per-rollout denominator
rollout_log_probs   optional list[[L_i]]
top-p metadata      optional
rollout_routed_experts optional [T_i-1, layers, topk]
teacher_log_probs   optional [L_i]
```

对应源码见 [[slime:slime/ray/rollout.py#L749-L866]]。这里 `rollout_mask_sums` 是一个非常容易被忽略的工程细节：如果同一个 rollout 被拆分到多个 micro-batch 里，每个 micro-batch 仍然要用这整条 rollout 的 mask 总数作为 denominator，这样才能避免 first-fit packing 的方式意外改变了优化目标。

### 3.4 DP/CP schedule

`_split_train_data_by_dp` 会根据 `rollout_ids` 计算出 DP 和 microbatch 的 schedule，实现见 [[slime:slime/ray/rollout.py#L871-L920]]。之后训练侧的 data iterator，再按照 Megatron 的 TP/PP/CP layout 把对应的 logits 提供出来。这里有一个容易踩的坑：不能直接按 Python 里 list 的 index 去假设同一个 sample 永远落在同一个 rank 上。

## 4. loss dispatch 与反向

```mermaid
flowchart LR
  X[packed tokens + masks] --> F[policy forward logits [T,V]]
  F --> LP[TP-aware target logp / entropy]
  LP --> ADV[reward + ref/teacher logp → Â/returns]
  ADV --> PL[policy / value / SFT loss]
  PL --> RED[mask-aware reducer + DP/CP normalization]
  RED --> BWD[Megatron backward + TP/PP/CP collectives]
```

### 4.1 advantage stage

`compute_advantages_and_returns` 运行在最后一个 PP stage 上，见 [[slime:slime/backends/megatron_utils/loss.py#L704-L741]]。整个流程是：先用 KL 对 reward 做 shaping，再根据配置选择 GRPO、GSPO、CISPO、PPO 或者 REINFORCE++ 这些 estimator 之一，最后再把 OPD 叠加上去，对应 `:743-816`。

### 4.2 policy stage

`policy_loss_function` 负责计算当前的 log-prob、sampled ratio、GSPO 的 sequence KL、CISPO/PPO 的 policy loss，以及 entropy 和各项监控指标，源码见 [[slime:slime/backends/megatron_utils/loss.py#L934-L1150]]。`ppo_utils.py` 里专门写的 custom autograd，明确处理了 TP 下 vocab parallel 的情况：softmax 的 max 和 sum 需要做 all-reduce，而 target 那一行的值只在拥有它的那个 TP rank 上才有。

### 4.3 loss reducer

`loss_function` 会在 `policy_loss`、`value_loss`、`sft_loss`、`custom_loss` 之间做选择，并按照 microbatch、global batch、DP×CP world size 做 rescale，见 [[slime:slime/backends/megatron_utils/loss.py#L1283-L1360]]。如果某个 CP rank 上没有有效 token，代码不能直接 early return，而是要加上一项 `0*logits.sum()`，保证 backward graph 在各个 rank 上保持对称，避免因为部分 rank 缺席而卡在 collective 通信上。

## 5. routing replay 的数据边界

启用 `--use-rollout-routing-replay` 之后，SGLang 的请求 payload 会带上 `return_routed_experts=True`，对应的 sample 里会保存一份 `[response positions, layers, topk]` 形状的 route 信息。conversion 阶段在 [[slime:slime/ray/rollout.py#L848-L852]] 会做 shape 和非零 MoE layer 的校验；actor 侧则在 [[slime:slime/backends/megatron_utils/actor.py#L322-L351]] 把这份 route 信息对齐到训练用的 token layout 上，再写进每一层的 `RoutingReplay`。

删除 `rollout_routed_experts` 之前，必须确认它已经 materialize 到了 CPU 或者 pinned memory 上，因为它并不是一份普通的 reward metadata，而是直接参与训练 forward/backward 的控制数据。

## 6. 框架扩展点

如果需要在这套框架里插入自定义逻辑，下面这张表给出了各个扩展点及其约束。

| 需求 | 推荐 hook | 约束 |
| --- | --- | --- |
| 自定义 tool/environment | `--custom-generate-function-path` | 返回 `Sample` 或同一 `rollout_id` 的 `list[Sample]` |
| 自定义 reward | `--custom-rm-path` | 明确 scalar/token/group reward 与 async contract |
| dynamic DAPO filter | `--dynamic-sampling-filter-path` | filter 原因必须可观测，不能静默丢 group |
| loss | `--custom-loss-function-path` | 保持 logits shape、mask、Megatron normalizer contract |
| TIS/IS | `--custom-tis-function-path` | 保存 behavior log-prob 与 support |
| sample→data | `--custom-convert-samples-to-train-data-path` | 仍需提供 schedule 所需 `rollout_ids/masks` |
| post-process | `--rollout-data-postprocess-path` | 不破坏 token/logp 对齐 |

接口完整表见 [[slime:docs/zh/get_started/customization.md#L5-L47]]。

## 7. Debug 顺序

遇到问题时，建议按下面的顺序逐步排查，而不是一次性打开所有新特性：

1. `debug_rollout_only`：验证 prompt、tokens、finish、reward；
2. dump rollout train data：核对 `loss_mask`, `rollout_log_probs`, `rollout_id`；
3. `debug_train_only`：固定 dump 重算 loss，和 CPU/reference 对齐；
4. `check_weight_update_equal`：确认 actor 到 engine 的权重一致；
5. 最后才开 async/partial/R3/FP8，否则多个变量同时变化无法定位。

---

**下一篇**：接下来[rollout–train 架构](02_rollout_train_architecture.md)会比较同步、有限 async、fully async 这几种执行方式，以及 colocated 和 disaggregated 这两种资源拓扑之间的取舍。
