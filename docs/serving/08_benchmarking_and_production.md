# 08｜Benchmark、容量规划与生产化

前面七篇分别讨论了 serving 的各个环节，这一篇解决最后一个问题：怎样测量、规划并运营一个生产系统。这一篇需要用到[《01｜Prefill、Decode 与 Serving 指标》](./01_inference_and_metrics.md)里定义的各项指标公式；`02` 到 `07` 则可以按你实际要测的系统形态挑着读。这一篇不追求给出一个“标准分数”，而是给出一套真正能回答生产环境里 SLO 和容量问题的实验方法与运维流程。

一次可信的 serving benchmark，应该能够回答这样一个问题：

> 在明确的 model/hardware/workload/arrival/cache 状态下，系统最多能承受多少 offered load，同时还能达到给定的 TTFT/TPOT/E2E SLO、error/reject 目标和成本约束？

如果只是跑一次 `request_rate=inf` 得到一个 tok/s 数字，那测出来的是 offline saturation，并不是 online capacity，这两者不能混为一谈。

---

## 1. 实验 manifest

### 1.1 Model / software

```text
model + exact weight revision/checksum
tokenizer/chat template/processor revision
weight dtype/quant, KV dtype/scales, max_model_len
serving engine commit + attention/MoE/backend versions
prefix cache/chunk/P-D/LMCache/Mooncake/compile/CUDA Graph config
container/driver/CUDA/ROCm/NCCL/firmware
```

### 1.2 Hardware / topology

```text
GPU type/count/HBM/power cap/clocks
CPU sockets/cores/NUMA/RAM, pinned memory limits
NVLink/NVSwitch/PCIe topology
NIC type/count/speed/GDR, switch oversubscription
local SSD/GDS/distributed store
TP/PP/DP/EP/CP and P/D/E pool placement
```

同样都叫“8×H100”，PCIe 版和 SXM 版、机内 TP 和跨机 TP、NIC/GPU 是否在同一个 NUMA 域还是跨了 socket，测出来的结果都可能完全不同。所以必须完整记录拓扑信息，而不只是记一个 GPU 型号。

### 1.3 Workload

至少应该为每一条请求保存：arrival timestamp、input/output length、session/turn 信息、prefix group 或者 block hash（可以脱敏）、priority、model/LoRA、media 相关的 tokens/pixels/frames/duration，以及成功/失败状态。如果出于隐私考虑不能保存原始文本，至少要保存长度和重用结构的匿名 hash，但一定要保持它们之间原有的相关性：真实的 agent workload 里，长 input 往往对应短 output，多轮对话的到达间隔也不是彼此独立的 Poisson 过程，随意打乱这些相关性会让测出来的结果失真。

---

## 2. Open-loop、Closed-loop 与 coordinated omission

### 2.1 Closed-loop 的降压效应

closed-loop 的 client 会等上一个请求完成之后才发下一个：

```text
server变慢 -> client发得更慢 -> queue不增长 -> 看起来tail还好
```

它适合模拟固定用户数、带有“think time”的场景，但没办法测出给定 arrival rate 下系统真正的容量，因为它会自动跟着系统变慢而降低发送速度，把本该暴露的问题掩盖掉。

### 2.2 Open-loop 与过载暴露

open-loop 按照预先定好的 wall-clock timestamp 发请求，不会因为 server 变慢就自动降低 offered load。vLLM 的 `bench serve` 在发送之前会预先采样出符合 gamma 分布的 inter-arrival 时间；`burstiness=1` 对应 exponential/Poisson 分布，小于 1 更突发，大于 1 则更均匀，并且会把累计时间归一化到目标 rate，源码见 [[vllm:vllm/benchmarks/serve.py#L391-L499]]。

如果 client 自身的 CPU 或者 socket 资源不够，导致错过了计划好的发送时间，就要专门记录**scheduled vs actual send lag** 这个偏差；否则负载生成器本身反而变成了瓶颈，测出来的结果就不再反映被测系统的真实能力了。多个 client 分布在不同机器上时，应该同步好时钟或者统一用相对的单调时钟来记录。

### 2.3 Coordinated omission

如果压测工具在 server 卡住的时候不继续记录本该到达的那些请求，就会漏掉卡顿期间本该产生的等待样本，导致算出来的 p99 被系统性低估。解决办法包括：

- 用独立的 arrival generator 走 open-loop；
- client 的 concurrency 或者连接池要足够大，能积累住这些请求，不会因为连接数不够而被迫延后发送；
- 记录 client 侧观测到的 E2E，而不只是读 server 内部的 histogram；
- queue 满了之后返回的 429/timeout 也要计入结果，不能悄悄丢弃；
- 分别报告 offered、accepted、completed、good 这几种不同口径的 rate。

---

## 3. 四层实验

把所有测试压缩成一次“大压测”很容易得到片面的结论。更合理的做法是分成四个层次分别去测：

### 3.1 L0：kernel/model microbenchmark

这一层的目标是建立一个 cost model，不涉及 HTTP 或者 queue 这些外层机制：

- prefill latency/tok/s：扫 `S`、batch tokens、prefix length；
- decode step latency/output tok/s：扫 batch size 与 context 长度；
- paged attention：扫 page size、KV dtype、context/head layout；
- TP/EP collective：扫 message 大小/token 数、rank 数、node 数；
- encoder/DiT/VAE：扫 pixels/frames/steps/shape/SP；
- KV transfer：扫 bytes、fragment 数、concurrency、NIC/NUMA、P/D 的 TP mapping。

这一层的输出应该是一整条 latency curve，而不是单个数据点，这样才能供 scheduler/router 做预测，以及支撑 dynamic chunk 之类的机制。

### 3.2 L1：offline saturation

所有请求同时提交，或者用一个固定的大 batch 去测，得到最大的 prefill/decode/total throughput，以及对应的 HBM 占用和利用率。这一层回答的是“硬件上限在哪里、有没有性能回归”，并不回答 SLO 相关的问题，所以报告的时候必须单独标注为 offline 结果，不能和在线容量混着看。

### 3.3 L2：online capacity sweep

固定 workload，从低到高扫 offered RPS，每一档都要留足够的 warmup 时间，等系统进入 steady-state 再采样：

```text
RPS: 1, 2, 4, 6, 8, ... 在knee附近加密
每点: p50/p90/p99 TTFT, TPOT, ITL, E2E
      accepted/completed/rejected/error/timeout
      req/s, input/output tok/s, request goodput
      queue/running/KV/preemption/cache/transfer/GPU/network
```

真正的容量点，是同时满足 SLO attainment 和 error budget 的最高稳定 RPS。如果测试结束的时候 queue 仍然在持续增长，说明这个点并不稳定，即便测试期间的 completed throughput 看起来很高，也不能算作容量点。

### 3.4 L3：长稳、故障与变更

用数小时甚至数天级别的混合 trace 长时间运行，并主动注入各种异常：worker/NIC/store 故障、client cancel、burst、长 prompt、cache churn、rolling upgrade、autoscale/drain。很多 buffer、引用计数或者 lease 方面的泄漏问题，只有在处理了数十万条请求之后才会暴露出来，短时间的压测根本发现不了。

---

## 4. 一条可复现的 vLLM open-loop 示例

服务启动方式、模型和硬件参数应该另外单独保存记录；这里给出 client 侧的示意命令：

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --input-len 4096 \
  --output-len 256 \
  --num-prompts 2000 \
  --request-rate 8 \
  --burstiness 1 \
  --num-warmups 20 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --goodput ttft:2000 tpot:100 e2el:30000 \
  --save-result --save-detailed \
  --metadata workload=4k_256 rps=8
```

这里 `--goodput` 的数值单位是毫秒，只有逐个请求同时满足所给出的全部阈值，才会被计入 good request，具体的 CLI 定义在 [[vllm:vllm/benchmarks/serve.py#L1668-L1696]]，对应的公式推导见 [`01`](./01_inference_and_metrics.md) 第 5 节。

需要注意的是：固定的随机长度分布适合用来做可控的回归测试，但并不代表真实线上流量的样子；生产环境相关的结论应该换用自定义或者按时间轴回放的真实 trace，并且要用 tokenizer 处理之后核对实际长度是否符合预期。`ignore_eos` 这个选项可以固定住 output 的工作量，但这样一来会改变真实的 EOS 分布，使用时应该明确标注出来，不能悄悄用它替代真实设置。

---

## 5. Workload suite 的覆盖范围

| suite | 目的 | 必须变化的轴 |
|---|---|---|
| short chat | decode/launch 与低 TTFT | 128–1K input，短/中 output |
| RAG/long context | prefill、KV、cache | 8K–128K input、document prefix reuse |
| code/agent | 长 output、多轮 burst | input/output 相关性、tool think time |
| shared system prompt | APC/cache-aware routing | prefix 长度、working set、tenant 隔离 |
| adversarial mix | HOL/preemption/fairness | 少量 128K 与大量短请求、priority |
| multimodal understanding | E/P/D | resolution、images、frames/audio duration |
| image/video generation | denoiser | canvas、frames、steps、CFG、quality |
| overload burst | admission/recovery | burst width/height、queue bound、429 |

至少应该覆盖三种不同的到达模式：近似平稳的 Poisson 到达、真实的按时间轴回放的 trace，以及人工合成的突发流量。即便平均 RPS 相同，突发流量下测出来的 p99 和 KV 峰值也可能完全不同，只测平稳到达会低估真实风险。

### 5.1 Multi-turn replay

不能把一段多轮对话拆成一堆互不相关的随机 prompt 来测，那样会丢掉 session 之间的间隔、历史长度的增长过程，以及 sticky/cache 路由带来的影响。应该分别测试以下几种情况：

- same-replica 下已经积累了完整历史的 warm 场景；
- 使用分布式外部 cache 的场景；
- cache 被禁用、完全冷启动的 baseline 场景；
- working set 超过 L1/L2 容量之后，系统进入稳定命中率的场景；
- session 迁移或者 replica failure 之后系统恢复的过程。

### 5.2 Prefix cache 实验

应该报告：

```text
unique/shared prefix distribution
reuse distance / working-set bytes
L1/L2/L3 capacity and warmup policy
lookup hit / load success / effective hit tokens
saved prefill vs load latency
eviction/write amplification
```

只是把同一个 prompt 发送两次、测出第二次 100% warm hit，这只是一个功能性的 microbenchmark；真正能支撑容量结论的测试，需要用足够长的 trace，让 cache 真正进入稳定状态之后再采样。

### 5.3 P/D transfer 实验

需要把整个传输过程拆开来分别测：

- control/handshake/reservation 各自的耗时；
- 从 paged source 做 gather 的耗时；
- 实际线路传输的 useful GB/s；
- 到达 destination 之后 scatter 的耗时；
- 做了 layer/chunk overlap 之后，用户实际能感知到的 latency；
- 多对 P×D 并发时彼此之间的公平性；
- 同构 TP 和异构 TP 两种情况；
- 部分失败之后重算的行为。

只用一个连续的 40GB buffer 去测 RDMA 的峰值带宽，测出来的数字不能代表真实场景下分散的 page list 传输情况。Mooncake 的上游仓库提供了 TE/store 相关的 benchmark，SGLang 文档也给出了 PD transfer 的 profiling 入口；这些测试都应该在相同的拓扑下去验证，才有比较的意义。

---

## 6. Cold、Warm 与 Steady-state 分开

至少存在四种不同意义上的“warm”：

1. **model warm**：权重已经加载完毕，NCCL/process group 已经就位；
2. **kernel warm**：JIT、torch.compile、autotune、CUDA Graph 的 capture 都已经完成；
3. **allocator warm**：KV pool 和 workspace 已经建立好，lazy page 或者 register 操作都已完成；
4. **cache warm**：prefix、encoder、L2、L3 里已经有了目标数据。

应该分别报告：启动耗时、第一个请求的 latency，以及真正 warm 起来之后的在线容量。不要把用了 warm cache 测出来的结果和 cold baseline 直接对比，然后把差异都归因到 kernel 层面；也不要在每个参数扫描点重启系统的时候，只 warm 了其中一部分而忽略了别的。

滚动扩容出来的新 replica，即便健康检查已经通过，kernel 和 cache 层面也可能仍然是 cold 的；router 应该给它一个小流量的 ramp-up 过程，或者提前预热，避免把整批请求一下子打到一个 cold worker 上造成 p99 尖峰。

---

## 7. 生产 observability：从用户到 GPU

### 7.1 四个黄金信号在 LLM 中的展开

| 信号 | LLM serving 指标 |
|---|---|
| latency | client/server TTFT、TPOT、ITL、E2E；media/E/P/xfer/D breakdown |
| traffic | offered/accepted/completed RPS；input/output/media tokens/s |
| errors | HTTP/gRPC status、timeout、cancel、OOM、NaN、transfer/cache corruption |
| saturation | queue work、running/live KV、HBM、SM/HBM/NIC/SSD、CPU event loop lag |

### 7.2 vLLM 已有的关键 metrics

[[vllm:vllm/v1/metrics/loggers.py]] 里定义了这些指标：

```text
vllm:num_requests_running / waiting                   :457-476
vllm:kv_cache_usage_perc                              :524
vllm:prefix_cache_queries / hits                      :548-568
vllm:prompt_tokens(_by_source/_cached)                 :634-667
vllm:generation_tokens                                :668
vllm:time_to_first_token_seconds                      :759
vllm:inter_token_latency_seconds                      :792
vllm:request_time_per_output_token_seconds            :822
vllm:e2e_request_latency_seconds                      :876
```

但仅有这些还不够，还需要从 gateway、GPU exporter、NCCL/NIC、LMCache/Mooncake/HiCache 各处拼接出完整的视图。可以用 trace 或者 request id 把 R→E→P→transfer→D 这条链路串联起来，但 Prometheus 的 label 不要直接放原始的 request、user 或者 prompt hash，否则会带来高基数问题，也有隐私泄漏的风险。

### 7.3 Histogram 与告警

Prometheus histogram 的 bucket 划分需要覆盖实际的 SLO 阈值，并且在阈值附近划分得足够密；平均值永远不能替代尾部分布。建议组合使用下面这些告警：

```text
SLO burn rate (短+长窗口)
queue tokens/work持续上升
accepted-completed gap / reject / timeout
KV usage高 + preemption/recompute升
cache metadata hit高但load success/effective hit低
P/D transfer queue/latency/failed block
GPU util低但queue高（CPU/collective/IO stall）
GPU util高且goodput下降（overload）
```

要避免只在 `GPU util > 90%` 的时候才触发扩容：decode 阶段受 memory bandwidth 限制，SM 利用率可能不高，但系统实际上已经饱和了；而 prefill 阶段 SM 利用率很高的时候，queue 却可能仍然处于可控范围内。这两种情况如果只看 GPU util 一个信号，很容易做出错误的扩缩容决策。

---

## 8. 容量规划

### 8.1 单 replica 的 HBM 账本

沿用 [`03`](./03_paged_attention_and_kv_cache.md) 里的推导：

$$
C_{KV,tokens}\approx
\frac{M_{HBM}-M_{weights}-M_{runtime}-M_{transfer/workspace}-M_{headroom}}
{M_{KV/token}}.
$$

再用真实 trace 去估算 live KV 的规模，而不是假设每条请求都占用最大 context 长度：

$$
E[N_{live\ tokens}]\approx\lambda\cdot E[\text{token-residency area per request}].
$$

这里的“token-residency area”指的是一条请求在整个生命周期内，context 长度对时间的积分。长 output 的请求会一边持续增长 KV、一边长期占着这份 KV 不释放，比总 token 数相同但很快结束的短请求要昂贵得多，这一点在估算容量时容易被低估。

### 8.2 Queue/service 容量

用 L0 测出来的曲线去模拟或者回放每一轮 batch 的耗时，比简单地用“总 tokens 除以单 token 峰值吞吐”要准确得多。可以先用下面这个式子做个初步的合理性检查：

$$
N=\lambda W
$$

（也就是 Little's Law），再用在线的 sweep 测试去确定 queue 的 knee point 和尾部行为。容量规划时必须留出应对 burst、故障、autoscale 预热延迟，以及长度预测误差的余量；100% 的稳态利用率不是一个可以长期运营的目标，留一些余量才是正常的做法。

### 8.3 P/D/E 与 diffusion 池

每个池都应该按它自己的工作单位来估算容量：

```text
E: pixels/patches/audio-seconds/video-frames per second
P: residual prefill tokens + attention context cost
D: active requests/live KV + output tokens per second
DiT: latent tokens × denoising steps × CFG branches
VAE: pixels/frames per second
```

先分别用每个池各自的 service curve 算出需要多少 replica，再从端到端的角度调整，让各池的 queue rate 互相匹配。只要有任何一个 stage 长期处于 `arrival > service` 的状态，整条链路最终都会变得不稳定，哪怕别的 stage 都还有余量。

---

## 9. Autoscaling

### 9.1 Scale signal

推荐组合使用下面这几种信号：

- 已排队的预计工作量除以单 replica 的 service rate，也就是预计的清空时间；
- SLO deadline 的剩余余量，或者 SLO 的 burn rate；
- running/live KV 的量与当前还能准入多少新请求的容量；
- 各 stage 各自的利用率和 step latency；
- 趋势和突发检测器。

应该对 P/D/E 分别做 scale；如果只看整体的 GPU util 平均值，很容易掩盖掉“一个池已经饱和、另一个池却很空闲”这种情况。做 scale-up 的时候要把模型加载、NCCL 初始化、compile/graph 编译以及 cache 预热所需要的提前量都考虑进去，不能等到 p99 已经违约了才开始扩容，那样已经来不及了。

### 9.2 Scale down / rolling upgrade

```text
mark draining -> stop new admission
finish or migrate running sessions/KV
flush/commit required external cache/metrics
wait async transfer and unregister memory
remove from router -> terminate
```

直接 kill 掉一个正在服务 decode 的 replica，会中断正在进行的 stream，也会丢掉它本地的 KV；如果指望靠 client 重试来兜底，那这个重试过程必须是幂等的，并且能够从 prefix cache 或者外部 checkpoint 里恢复出等价的状态。在 P/D 架构下，下线之前要先解除 P 和 D 之间的配对关系及 lease，避免 P 还在继续向一个已经退出的 D 发送 RDMA 写入请求。

cache 的 namespace 应该包含软件和模型 layout 的版本号；滚动升级期间，新旧两套实例不应该互相读取到彼此不兼容的 KV。共享的 Store 可以先做双读或者新写、旧读的过渡策略，或者直接做版本隔离，确认新版本的命中率和质量都没问题之后，再清理旧的 namespace。

---

## 10. Admission、Load Shedding 与公平性

### 10.1 Bounded queue

不设上限的 queue，只会把 OOM 问题延迟变成分钟级的 TTFT 问题，本质上没有解决瓶颈，只是把它藏起来了。应该按 model/tenant/priority 分别设置：

- 最大排队的请求数**和**最大排队的 token/work 量；
- 最大的 input/output/media 限制；
- 支持 deadline-aware 的拒绝策略；
- 按租户设置 token bucket 或者并发数限制；
- 对总的 live KV、encoder、transfer 资源做预留限制。

应该返回明确的 429/503 状态码，并带上 `Retry-After` 头，让上游服务能够正确地做 backoff；不应该先接受了请求，然后又悄悄让它 timeout，那样对上游是没法区分处理的。

### 10.2 优先 shed 的请求

可以按优先级、deadline、预计工作量、是否已经投入了昂贵的 prefill 计算、cache locality，以及 tenant quota 来综合决策该丢弃谁。目标并不是只保护短请求：还需要 aging 机制或者预留一部分服务份额，防止长请求被系统性地饿死。一般来说，已经开始 decode 的交互式请求，通常比还没开始 prefill 的低优先级请求更值得保护，因为前者已经产生了用户可见的 stream，也已经在 KV 上投入了实际的计算成本，丢弃它的代价更大。

### 10.3 Degrade options

只有在 API 语义允许的情况下才能做降级处理：

- 降低最大 output 长度、分辨率、frame 数或者 denoising step 数；
- 把请求路由到更小的模型或者量化后的模型；
- 关闭价值较低的 remote cache store，或者关闭比较昂贵的 logprobs 计算；
- 更早地返回 partial response。

不能在用户不知情的情况下悄悄改变采样策略或者输出质量；返回结果应该明确标注自己处于 degraded mode，并且纳入一套独立的 SLO 体系去衡量，不能和正常模式的 SLO 混在一起统计。

---

## 11. Reliability 与正确性演练

### 11.1 Request lifecycle

每一条请求都应该带一个全局的 generation id，所有 R/E/P/D/cache/transfer 之间传递的消息都应该是幂等的。client 重试不能导致重复计费或者重复的 stream 输出；迟到的 packet 也不能被写入一个已经被复用过的 buffer。请求被取消的时候，需要传播到并清理下面这些地方：

```text
waiting/running queue entry
KV/encoder/latent physical pages and ref locks
P/D/EPD destination reservation and transfer task
LMCache/Mooncake lease/pin/staging buffer
HTTP stream/detokenize state
```

### 11.2 Fault matrix

| 注入 | 期望 |
|---|---|
| API/router 重启 | in-flight ownership 明确，retry 幂等 |
| engine worker crash | router 快速摘除；stream fail/recover 策略明确 |
| P/D 中任一侧 crash | lease 回收、旧 RDMA completion 不 commit |
| NIC rail down | failover 或 bounded fail；带宽/尾延迟告警 |
| external cache miss-after-hit | invalid block→recompute/fail，不读垃圾 |
| Store/master 故障 | local serving 可降级；metadata HA/timeout 有效 |
| OOM/allocator pressure | admission/preemption 可观测，无 deadlock/leak |
| corrupt/partial KV | checksum/version 或 logit 对齐测试捕获 |
| rolling model update | cache/encoder cache 原子失效/namespace 切换 |
| slow/disconnected client | backpressure/cancel 释放 GPU 状态 |

### 11.3 Accuracy parity

所有 memory、transfer、cache 相关的优化，都应该先做 deterministic 的 logits/token 逐步对齐测试：

- baseline 和 paged/quantized KV 之间的对比；
- 冷 prefill、本地 prefix 命中、L2/L3 load 这三种路径之间的对比；
- unified 部署和 P/D 同构/异构 TP 部署之间的对比；
- 多模态原始 encode 和使用缓存/precomputed embedding 之间的对比；
- transfer 重试或者部分失败之后 fallback 路径的行为。

有损的 KV eviction、CacheBlend、diffusion cache 这类方法本身就不追求 bitwise 一致，不能强行要求它们逐位对齐；应该针对具体任务定义出可接受的质量容差，并做分布层面的回归测试，而不是简单地判定通过或不通过。

---

## 12. 一页报告模板

```markdown
## Goal / SLO
TTFT p99 < ...; TPOT p99 < ...; E2E...; attainment ...%; error < ...

## Environment
engine/model commits; dtype/KV; GPU/CPU/NIC/topology; TP/PP/DP/EP/CP; pools

## Workload
arrival/burst; N; input/output distributions; prefix/session/media distribution

## Results by offered RPS
accepted/completed/good/rejected; req/input/output throughput
TTFT/TPOT/ITL/E2E p50/p90/p99; queue stability

## Resource / internals
KV/cache/preemption; batch/chunk; GPU/HBM/NIC/CPU; P-D transfer; E/DiT/VAE

## Capacity conclusion
highest stable SLO-compliant RPS; req/s/GPU and cost; headroom

## Correctness / failure
parity tests; injected faults; leaks; known limitations
```

---

## 13. 最终上线门槛

把这一整章的内容落到实处，上线前建议逐项核对下面这份清单：

```text
□ 指标口径client/server一致，TPOT与ITL不混
□ open-loop真实trace容量曲线，不只offline tok/s
□ warm/cold/steady cache与compile状态分别报告
□ queue以work/tokens有界，429/timeout进入SLO分母
□ HBM/KV/transfer/encoder/latent完整容量账本和headroom
□ P/D/E/DiT各stage可独立观测、backpressure和autoscale
□ cache key/version/tenant隔离，weight update安全失效
□ cancel/retry/drain/rolling upgrade幂等
□ worker/NIC/store/partial-transfer故障演练
□ 精确路径logit对齐；有损路径质量门槛
□ 长稳无page/ref/lease/buffer/metric-cardinality泄漏
```

至此，serving 这一章从单条请求的张量 shape 与指标定义出发，依次覆盖了 batch 调度、PagedAttention/KV 内存系统、prefix cache 与分层 cache、P/D 分离与 Mooncake/LMCache、集群并行与 overlap，以及理解型/生成型多模态，最后落到了可复现的 benchmark 方法与生产环境的运维闭环上。

返回总览：[推理服务：从单请求推理到 SLO-aware 集群](./README.md)。

---

## 参考

- vLLM benchmark：[[vllm:vllm/benchmarks/serve.py]]、[[vllm:vllm/benchmarks/lib/endpoint_request_func.py]]。
- vLLM production metrics：[[vllm:vllm/v1/metrics/]]、[[vllm:docs/design/metrics.md]]。
- SGLang serving benchmark/observability：[[sglang:docs/developer_guide/bench_serving.md]]、[[sglang:docs/advanced_features/observability.md]]。
- DistServe goodput 定义（[arXiv:2401.09670](https://arxiv.org/abs/2401.09670)）。
- Mooncake trace、benchmark 与故障设计：[[mooncake:FAST25-release/]]、[[mooncake:docs/source/performance/]]、[[mooncake:docs/source/design/]]。
- LMCache observability/cache simulator：[[lmcache:docs/source/production/observability/]]、[[lmcache:lmcache/tools/cache_simulator/README.md]]。
