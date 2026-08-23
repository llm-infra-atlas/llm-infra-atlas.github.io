# 07 · Serving 中的 speculative decoding

单请求 latency 上的「6×」和线上「同吞吐下用户 +70%」并不是同一个指标。本篇把 speculative decoding 放进真实 serving 的约束中讨论：continuous batching 已经用掉了那部分「闲置算力」、verify 和 draft 都要进 CUDA Graph、被拒绝后要回滚 KV、P-D 分离还要跨节点传递 hidden。最后用 SGLang 的算法枚举对齐实现名词，并给出一条选型决策。

> 代码（[[sglang:python/sglang/srt/speculative/]]）：
>
> - `spec_info.py` —— `SpeculativeAlgorithm`：`DFLASH` / `DSPARK` / `EAGLE` / `EAGLE3` / `FROZEN_KV_MTP` / `STANDALONE` / `NGRAM`
> - `eagle_worker_v2.py` / `multi_layer_eagle_worker_v2.py` / `dflash_worker_v2.py` / `dspark_components/`
> - `eagle_utils.py`、`ragged_verify.py`、`spec_registry.py`
>
> CUDA Graph：[07 · CUDA Graph](../torch/07_cuda_graph.md) §3.5。

---

## 1. 高并发下加速公式的失效

[`00`](./00_decode_bottleneck.md) 的加速建立在「verify ≈ 一次 decode」上。continuous batching 把几十、上百个请求的 decode 拼成大 GEMM 之后：

- 工作点已经右移，甚至进入 compute-bound 区域
- 再给每个请求加 $\gamma$ 个 verify token，$T_{\mathrm{verify}}$ 真实上涨
- 被拒绝的后缀占用的是其他用户的 batch 位

因此存在一条经验上的分界线：batch 小时 spec 收益明显；batch 大时 spec 不赚甚至为负。EAGLE-3 论文报告 SGLang bs=64 时仍有 1.38× 吞吐——这是「尚未越过分界线」的数据点，而不是「任意并发都有 6×」。

DSpark 的 scheduler 需要的 `SPS(B)` 曲线，就是引擎自身的「batch → 步频」关系。没有这条曲线，$\gamma$ 只能凭经验设定。

---

## 2. 引擎中的一轮 speculative decode

```
scheduler 选出一批 running requests
    │
    ├─ draft 阶段     （小模型 / MTP / EAGLE / DFlash / n-gram）
    │     写出每条请求的候选（链 / 树 / 块）
    │     可能自己有一份 draft KV、自己一张 CUDA Graph
    │
    ├─ 可选: 按 confidence / 负载裁 verify 长度     （DSpark ragged verify）
    │
    ├─ target verify  （ForwardMode.TARGET_VERIFY）
    │     token 数 = Σ_req N_verify(r)   ≠ batch size
    │     tree / block mask，一次 forward
    │
    ├─ accept / reject
    │     提交最长前缀 + bonus/correction
    │     回滚未提交的 KV / hidden
    │
    └─ 重叠: 下一轮 draft 可与本轮的 GPU 收尾 overlap（spec v2）
```

与普通 decode 的差别都源于一点：**shape 不再是「每请求 1 token」**：

| 阶段 | 每请求 token 数 | Graph 怎么分桶 |
|---|---|---|
| 普通 decode | 1 | 按 batch size |
| EAGLE draft | `topk`（一层的分支） | `eagle_draft_cuda_graph_runner`，桶更细 |
| EAGLE verify | `num_draft_tokens`（整棵树节点） | 与 decode 共用 runner，`TARGET_VERIFY` |
| DFlash draft | ≈ $\gamma$（一块） | dflash 自己的 graph |
| DSpark verify | **每请求不同** $\ell_r$ | `ragged_verify`，按 token-bucket |

`spec_info.py` 把「这一次 forward 每请求多少 token」收在 `SpecInput.num_tokens_per_req`，DP attention 的 `global_num_tokens` 要乘上这个因子（`spec_scale_global_num_tokens`），漏乘就会在 TP/DP 同步时对不齐。

---

## 3. CUDA Graph

decode 和 verify 都是「短序列、反复同一 shape」的负载，如果 CPU launch 跟不上，spec 省下的 GPU 时间会被 launch 开销吃掉。SGLang 的策略（见 [07 · CUDA Graph](../torch/07_cuda_graph.md)）：

- `ForwardMode.TARGET_VERIFY` 和 `DECODE` 一样走 `is_cuda_graph()`
- verify 的 `num_tokens_per_bs` = 每请求 verify 宽度
- draft 另有 runner，EAGLE 的 `num_tokens_per_bs = topk`
- spec 用**更细的桶列表**，因为 `(batch, tree_width)` 组合比普通 decode 多

DSpark 的 ragged verify 使每请求的宽度不同，固定 shape 的 graph 无法覆盖，因此单独开了以 token-bucket 为键的 verify graph。`supports_ragged_verify()` 只对 DSpark 为真。

Overlap 方面：spec v2 把 worker 放到与普通 decode 同一套 overlap schedule 上；`CustomSpecAlgo.supports_overlap=False` 已被标记为 deprecated，V1 同步路径已删除，插件也需要迁移到 V2。NGRAM 没有 GPU draft 可以掩盖 CPU 开销，grammar overlap 默认关闭。

---

## 4. KV、hidden 与 P-D 分离

**Target KV**：verify 写入的 draft 位置，只有被接受的前缀能留下。树的兄弟节点、拒绝点之后的位置必须回收，否则 cache 泄漏、后续 RoPE 位置出错。

**Draft KV**：EAGLE / DFlash 自己也有一份 cache（DFlash 还缓存注入的 $H_{\mathrm{ctx}}$）。NGRAM 没有（`has_draft_kv()==False`），少一轮 page 对齐。

**Hidden 传递**：EAGLE 家族下一轮 draft 需要使用上一轮 target 的 feature，SGLang 为此在 worker 里维护 `req_to_hidden_states_pool`，被拒绝后要 revert。`carries_draft_hidden_states()` 只对 EAGLE 为真——P-D 分离时，prefill 节点向 decode 节点传输必须带上这些 hidden，否则 decode 端的第一轮 EAGLE 没有输入可用。DFlash/DSpark 走 `dflash_disaggregation.py` / `dspark_disaggregation.py`；EAGLE 走 `eagle_disaggregation.py`。

**MTP 与共享 embed/head**：`multi_layer_eagle_worker_v2.py` 从 target `get_embed_and_head()` 再 `set_embed_and_head` 到每个 MTP step——和 DeepSeek 报告里「物理共享」一致。PP 下要保证这份共享落在同一 rank（V3 DualPipe 的动机之一）。

---

## 5. SGLang 算法枚举

`SpeculativeAlgorithm.from_string` 认识的名字（`spec_info.py`）：

| 枚举 | 谁来 draft | Worker | 备注 |
|---|---|---|---|
| `EAGLE` | EAGLE-1/2 式 feature AR | `EAGLEWorkerV2` | `NEXTN` 别名也落到这一家 |
| `EAGLE3` | EAGLE-3 | 同上 | `is_eagle3()` 单独打开融合路径 |
| `FROZEN_KV_MTP` | 冻 KV 的 MTP | `FrozenKVMTPWorkerV2` | 调度器暂且 `is_eagle()` |
| `DFLASH` | 并行扩散块 | `DFlashWorkerV2` | `is_dflash_family()` |
| `DSPARK` | DFlash + 半 AR + 调度 | `DSparkWorkerV2` | `supports_ragged_verify` |
| `STANDALONE` | 独立小模型 | `StandaloneWorkerV2` | 经典 Leviathan |
| `NGRAM` | 检索 | `NGRAMWorker` | 无 draft KV |
| `NONE` | — | — | 普通 decode |

`enable_multi_layer_eagle` 时 EAGLE 改走 `MultiLayerEagleWorkerV2`：逐步跑多个 MTP runner，chain 模式把 hidden 传给下一步（Step3.5），非 chain 每步都使用 target hidden。

插件用 `@SpeculativeAlgorithm.register("MY_SPEC")` 挂自己的 worker，但必须实现全部 `is_*` / `supports_*`，否则调度器里的 `if spec_algorithm.is_eagle()` 会直接报错（`spec_registry.py` 的 duck-type 断言）。

常用 CLI 参数（名称随版本变化，以当时的 `server_args` 为准）：

```
--speculative-algorithm {EAGLE,EAGLE3,NEXTN,DFLASH,DSPARK,NGRAM,...}
--speculative-draft-model-path ...
--speculative-num-steps            # EAGLE 树深
--speculative-eagle-topk           # 每层分支
--speculative-num-draft-tokens     # N_tree 或块长
```

DeepSeek-V3/R1 的最小可用路径：模型自带 MTP 权重 + `NEXTN`，不必另训 EAGLE。

---

## 6. 选型决策

先问负载，再问有没有现成的 drafter：

```
请求是否高度重复（代码补全 / 文档改写 / 模板）?
  是 → NGRAM / PLD，零训练，先吃免费加速
  否 ↓

target 是否自带 MTP（DeepSeek-V3/R1/V4，部分 Qwen / Hunyuan）?
  是、只要 ~1.5–2×、要稳 → MTP-1 / NEXTN
  是、还要更长 τ、有训 drafter 的预算 → DSpark（生产）或 EAGLE-3 / DFlash（开源）
  否 ↓

单用户 / 小 batch / 要无损 SOTA（开源）?
  → EAGLE-3（生态最全）或 DFlash（长 CoT、想压 T_draft）

高并发、有 SLA、能 profile SPS(B)?
  → DSpark（第三根杠杆）；至少给 EAGLE-3 加上「大 batch 关 spec / 减 γ」

只有一个同族小模型、不想训?
  → STANDALONE；先量 c，c 接近 α 就别开

完全不想维护第二份权重、接受 1.5–2×?
  → Lookahead / Medusa-1；新项目不优先
```

三条实践经验：

1. **先量接受率再看 speedup。** $\tau$ 上不去时，调 CUDA Graph 没有意义；$\tau$ 很高但线上吞吐下降，说明 $\gamma$ 或树太大，应走调度或减宽。
2. **同一套 drafter，代码 ≫ 数学 ≫ 开放闲聊。** 报告里的 6.5× 往往是 HumanEval。用自己的流量测 $\alpha(c)$。
3. **无损是默认，有损是产品决策。** typical acceptance、投机式跳步会改变分布；对外 API 要先声明。

---

## 7. 与 serving 章节的分工

[推理服务：从单请求推理到 SLO-aware 集群](../serving/README.md) 规划 continuous batching、PagedAttention、P-D 分离、调度。本章只补充 spec 特有的部分：

- verify 把「每请求 1 token」变成「每请求 $N_{\mathrm{verify}}$」
- 三张 graph（decode / draft / verify），DSpark 还要 ragged
- hidden / KV 的跨轮、跨节点契约
- 吞吐–延迟 Pareto 上，$\gamma$ 是负载的函数

读完本篇，本章的算法部分就闭环了。回到 [`README`](./README.md) 的对照表，三根轴上每个方法的位置应当已经可以自行填写。

---

下一篇：回到 [`README`](./README.md) 对照表，或进 [推理服务：从单请求推理到 SLO-aware 集群](../serving/README.md) 看引擎其余部分。若要从物理层再走一遍，[`00`](./00_decode_bottleneck.md) 的 roofline 是同一把尺。
