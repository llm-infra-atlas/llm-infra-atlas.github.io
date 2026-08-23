# 04 · Sandbox：process confine 与 microVM

> 本章是 agent 方向中与 infra 关系最密切的一篇：讨论动作在什么样的隔离环境中执行。对照代码：[[agentenv:]]。
>
> AgentENV 的 README 写它支撑了 Kimi K3 的 agentic RL，但那是它的**客户**，不是它的定义。本篇只讲 runtime 本身：为什么需要隔离、隔离到哪一层、snapshot / fork 为什么是 agent 的一等原语。训练回路不在这里展开。

获取代码：

```bash
git clone https://github.com/kvcache-ai/AgentENV.git
# 或独立 clone
git clone https://github.com/kvcache-ai/AgentENV.git
```

---

## 1. Agent 对 sandbox 的需求

chat completion 的副作用停在 token。agent 的 action 会：

- 执行模型写出来的代码或 shell（CodeAct、coding agent）；
- 读工作区以外的文件、打内部网、装包；
- 把状态留到下一步（cwd、虚拟环境、浏览器登录态）；
- 一次任务里 **fork** 出多个候选世界（并行试两条修复、给 subagent 各一台机器）。

威胁模型和普通「用户点运行」不同：攻击者是**模型的输出**，不是坐在键盘前的人。prompt injection 一旦让模型生成 `curl … | sh` 或 `cat ~/.ssh/id_rsa`，harness 若在自己的 UID 下 exec，就等同于把 shell 交给了攻击者。

因此 sandbox 需要同时满足四个条件，缺少任何一条都会在规模化时出问题：

| 需求 | 含义 |
|---|---|
| **隔离** | guest 崩了、被 rm -rf 了、扫了内存，host 和邻居 sandbox 还在 |
| **密度** | 一台宿主机要跑几百、几千个闲置或半闲置的环境 |
| **快启动 / 快恢复** | agent 步间延迟是秒级，VM 冷启动分钟级不可接受 |
| **可复制状态** | pause / resume / snapshot / fork，而不是每次从 Dockerfile 重建 |

后两条是 agent 相对经典 serverless 多出来的。Lambda 式「每次调用一个干净文件系统」对「先 pytest 再根据失败改文件」不够；agent 要的是**一台活着的、能被拍快照的计算机**。

---

## 2. 隔离谱：四个层次

从弱到强（实现可以叠加，但信任边界一次只选一层当主边界）：

```
同进程解释器          共享宿主内核、共享地址空间     只适合纯函数
    │
本机进程 + OS sandbox  共享内核；用 Landlock /
（dsh ctx.sandbox）    bwrap / Seatbelt 限制*本进程*能写的路径
    │
容器 (runc / gVisor)   多数仍共享宿主内核；gVisor 用
                       用户态 syscall 拦截换隔离
    │
microVM (Firecracker)  独立 guest kernel + 极小设备模型
                       用 KVM；失败域是一台虚拟机
    │
全功能 VM / 独立机器   隔离最强，密度和启动最差
```

Firecracker（Agache et al., NSDI 2020）就是为「函数要硬隔离、又要秒级启动、又要高密度」造的 VMM：KVM、极简设备和设备模型、为 serverless 优化过的启动路径。AgentENV 把它从「跑一个无状态函数」扩成「跑一台对 agent 有状态的 Linux」。

和 [`03` §7](./03_deepseek_harness.md) 对上：

| | dsh `ctx.sandbox` | 容器 | AgentENV |
|---|---|---|---|
| 内核 | 宿主 | 通常宿主（gVisor 除外） | **guest** |
| 文件系统 | 宿主路径的 allow-list | overlay + mount | overlaybd 层叠块设备 |
| 网络 | **不管**（注释写明） | namespace + 策略 | 每槽 netns + iptables |
| 状态原语 | 无（一次 spawn） | commit 镜像慢 | pause / resume / snapshot / **fork** |
| 换它要动谁 | wrap argv 的 provider | runtime 配置 | `ctx.fs` + `ctx.subprocess` 整条 seam |

dsh 写得很清楚：容器和 microVM 是「整条执行世界 seam」的兄弟，不是 `SandboxProvider` 的一种 backend。把 Firecracker 塞进 `confine(argv)` 会把「这次 bash 能不能写 `/tmp`」和「这台机器还在不在」混在一个接口里。

---

## 3. AgentENV 的定位

官方定义（文档 *Getting Started*）：

> 自托管的 sandbox runtime。跑隔离的 Firecracker microVM，暴露 **E2B 兼容** HTTP API——已有 E2B SDK 不用改代码。

它不是 harness：没有 `ReactLoopAgent`，不拼 prompt，也不登记 `bash` tool。它是 [`00`](./00_definitions.md) 中定义的 **environment**，只是被做成了集群服务。

生产里它被推到的量级（README，Kimi K3 技术报告）：

- 按需加载 OCI 兼容镜像，overlaybd；本地盘只是有界缓存，**1.5 million images** 不必预热到每台机器；
- snapshot 启动 / 恢复 &lt; 50 ms，pause &lt; 100 ms；
- 增量 snapshot 在磁盘被重写时仍 &lt; 100 ms；运行中的环境可以 **fork** 成多个独立 sandbox；
- ublk I/O + 页缓存共享 + memory ballooning，生产过 **9.6×** 内存超卖。

这些数字会随版本变化，但设计意图不会：agent 的环境既要像一台真正的计算机，又要像一个可以被调度器自由迁移的对象。

E2B 兼容写在 API 标题上（[[agentenv:src/api/openapi.yml#L4]]，title `E2B API`），代理认 `e2b-sandbox-id` / `e2b-sandbox-port`（[[agentenv:src/api/proxy.rs#L86-L91]]）。dsh 的 [[deepseek-harness:packages/e2b/]] 只依赖官方 E2B SDK——把 `E2B_API_URL` 指到 AgentENV，就是两个仓库的接缝，无需新协议。

---

## 4. 单节点数据通路

文档 *How AgentENV Works* 的结构，对应到代码：

```
Client  ──HTTP──►  API (Axum, src/api/)
                      │
                      ▼
                 Orchestrator          生命周期状态机
                 (src/orchestrator/)
                      │
                      ▼
                 SandboxBackend        trait: start/pause/resume/snapshot/fork/stop
                 (src/sandbox/backend.rs:182)
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
   Firecracker microVM      块设备层
   guest kernel             overlaybd 层叠 + ublk (/dev/ublkbN)
          │
          ▼
   envd（guest 内 daemon）  跑命令、管文件、报健康
```

一次 `POST /sandboxes`：

1. API 校验、鉴权，交给 `Orchestrator::create_sandbox`（[[agentenv:src/orchestrator/service.rs#L330]]）。
2. `create_sandbox_inner` 按 `SandboxLaunchSource` 分支：从 **Snapshot** 恢复，或从 **Image**（OCI → overlaybd）冷启（`types.rs:13-25`，`service.rs:371`）。
3. backend `start`：建 Firecracker、挂 rootfs / extra drive、配 netns。
4. guest 里 envd 起来之后，client 经 reverse proxy 打进去跑命令、开 PTY、暴露端口。

生命周期状态（[[agentenv:src/orchestrator/types.rs#L58-L67]]）值得和 dsh 的 turn/step 对照着看——那是对话时钟，这是**机器时钟**：

```
Creating ──► Running ──┬──► Snapshotting ──► Running
                       ├──► Forking     ──► Running  (+ children Running)
                       ├──► Pausing     ──► Paused  ──► Resuming ──► Running
                       └──► Killing
```

`SandboxBackend` 把这组状态收拢成 trait（[[agentenv:src/sandbox/backend.rs#L208-L246]]）：

| 方法 | 语义 | 失败域 |
|---|---|---|
| `start` / `wait_for_ready` | 机器起来，envd 可接活 | 创建失败则没有 sandbox |
| `pause` | 拍下可恢复状态，之后 `stop` 放资源 | `Terminal`：现场已被改坏，不能再当 Running |
| `resume` | 从 paused 拉回；**幂等** | — |
| `snapshot` | 从 **Running** 拍持久快照，拍完继续跑 | 同上 Terminal / Recoverable |
| `fork` | 一台 Running 拆出多台独立 child，源恢复后并行起 child | 源 Terminal 则源不能留；child 失败互不影响 |
| `stop` | 放净；**幂等** | — |

`Recoverable` 必须保证返回时源已经回到 Running；`Terminal` 表示现场过了可安全 resume 的点，orchestrator 只能将其销毁（trait 注释，`backend.rs:202-207`）。对应到 agent 侧的产品语义：一次失败的「另存为模板」不应该毁掉正在修 bug 的环境；如果真的毁掉了，就不要假装还能继续 `steer`。

---

## 5. 存储：毫秒级 snapshot 与 fork

AgentENV 的核心不是「再包一层 Firecracker API」，而是把磁盘和内存都做成**可层叠、可共享的块设备**。文档 *Architecture* 画了两条正交路径：

**块设备（rootfs / extra drive）**

```
OCI 镜像 ──overlaybd──► LSMT 层叠
                         只读下层 + 一个可写 upper
                              │
                              ▼
                         ublk 用户态块设备  /dev/ublkbN
                              │
                              ▼
                         Firecracker virtio-blk
```

overlaybd（[[agentenv:storage/overlaybd/]]）是 LSMT：读从上往下找第一个命中的 `DiskSegmentMapping`，写只追加 upper。多个 sandbox 共享同一组只读层，互不拷贝。pause 的主路径是 `close_seal + restack`：把 live upper 封成最新的一层只读，再开一个空 upper（`CLAUDE.md` Storage 节；[[agentenv:src/sandbox/firecracker/overlaybd_snapshot.rs#L1-L6]]）。

ublk（[[agentenv:storage/ublk/]] + [[agentenv:storage/ublk-daemon/]]）用 Linux ublk + io_uring 把这幅层叠图暴露成真正的块设备。daemon 是独立进程，节点经 Unix socket 调 `CreateOverlaybd` / `RestackSnapshot` / 热池 acquire。kernel 6.8+ 是硬前置（README）。

**内存快照**

pause 时不走「Firecracker 先写完整 `mem.bin` 再包装」的慢路径（当前默认）：查 dirty / present 页，`process_vm_readv` 读出来，直接打成 overlaybd 内存层。resume 时这块层叠变成只读 ublk，交给 Firecracker 当 file-backed memory；guest 写时 COW 进匿名页，底层设备不被改。

**同一份 snapshot 起多台 sandbox，只引用一个内存 ublk**（refcount）。Linux page cache 被共享——这是密度数字的来源之一，不是「每台 VM 一份完整内存镜像」。

层数会涨。`overlaybd_snapshot.rs:35-37` 在远低于硬顶 255 的 32 层触发 compaction，避免每次 I/O 沿栈走太深。

把这套机制对 agent 的意义概括为三点：

1. **冷的是镜像层，热的是 upper 和 dirty 页。** 所以「一百万镜像」和「一秒内起一个 Ubuntu」可以同时成立。
2. **fork 不是 `docker commit` + 再 `run`。** 它复用只读层和内存层，只把分叉后的写入打到新的 upper。
3. **idle 可以 pause。** CPU / 匿名页还给 host，再来请求时 resume，而不是杀了重建。这是「闲置环境必须便宜」——agent 会并行开很多、又经常在等模型。

---

## 6. fork 作为 agent 原语

harness 侧已经有「一条 session 叉开」（dsh `ctx.sessions.fork`）。那份 fork 复制的是 log 前缀，而环境侧的 fork 复制的是**机器**：磁盘 upper、内存、网卡都是新的失败域。

典型用法（不必是 RL）：

- 同一个失败测试，并行试两种补丁（两个 child，源环境留着对照）；
- subagent 要「大胆重构」，父 agent 留在干净树上；
- 评测时从同一 snapshot 拉起 N 个独立 rollout，避免「第一个跑过的测试污染第二个」。

`Orchestrator::fork_sandbox`（`service.rs:513-526`）在同一节点上把一台 Running 拆成 `count` 个 child。backend 合同（`backend.rs:227-241`）：源恢复之后必须尝试**每一个** spec；某个 child 启动失败，成功的兄弟继续跑；只有源被改坏才报 `Terminal`。

和 session fork 对照来看：

```
dsh session.fork     =  对话历史分叉（模型接下来看见的 prefix）
AENV sandbox.fork    =  世界分叉（下一条 bash 改的那棵磁盘）
```

两个都做了，才是「这条分支上的 agent 和那条互不污染」。只 fork log、共用一台 VM，会把文件系统变成隐式共享内存。

---

## 7. 多节点：gateway + scheduler

单机够本地开发。集群时（文档 *Multi-Node*）：

```
Client ──HTTP──► Gateway :8080 ──gRPC──► Scheduler :9090
                      │                      │
                      │     选节点 / 查绑定
                      ▼                      ▼
                 Node A :8000           Node B :8000
```

- 新 sandbox：scheduler `Schedule()`（默认 round_robin / random）。
- 已有 sandbox：`LookupNode()`，gateway 按 sandbox id 反代，不靠客户端记住节点。
- 绑定默认在内存；`scheduler.redis_addr` 可让 query-only 副本在主调度重启时仍能把数据面打到对的节点。

P2P（[[agentenv:src/p2p/]]，可选 iroh）用来在节点之间搬 overlaybd layer / snapshot artifact，避免每个节点都回源对象存储。这是「百万镜像、本地盘只是缓存」在分布式下的补全，不是 agent loop 的一部分。

---

## 8. 两份代码的对接

```
用户
  │
  ▼
DeepSeek Harness                    上游 DeepSeek Harness
  ReactLoopAgent / session log
  ctx.tools  (bash, edit, …)
  ctx.fs + ctx.subprocess
        │
        │  本机：sandbox-local confine(argv)
        │  远程：dsh-e2b 持有一个 E2B Sandbox 句柄
        ▼
E2B 形状的 HTTP API
        │
        ▼
AgentENV                            上游 AgentENV
  Orchestrator + Firecracker
  overlaybd / ublk / envd
```

读代码时各看各的不变量：

| 你想搞清 | 去哪 |
|---|---|
| 模型这一步为什么会再被调用 | dsh `agent.ts` 的 `turn` / `step` |
| 这条 bash 为什么被拒写 | dsh `SandboxProvider.confine` + denial dialect |
| 这台机器从哪张快照来、能不能再 fork | AENV `SandboxLaunchSource` + `fork_sandbox` |
| 为什么起得这么快 | AENV overlaybd 层共享 + 内存 ublk refcount |
| 训练时一万条轨迹怎么挂环境 | **不在本章**；AENV 只保证「环境是可调度对象」 |

---

## 9. 选型建议

| 场景 | 倾向 | 原因 |
|---|---|---|
| 本机 coding agent，workspace 可信 | dsh process sandbox，`workspace-write` | 够用，无 KVM 依赖 |
| 模型会跑任意代码 / 装包 | 至少容器；要硬隔离就 microVM | 共享内核挡不住内核漏洞和部分旁路 |
| 需要 pause 几百个闲置环境 | AgentENV 一类 snapshot runtime | 容器 commit 太重，杀了重建丢失 cwd |
| 需要从同一状态并行试 N 条分支 | **必须有**环境级 fork | 只 fork session 会共享磁盘 |
| 评测模型本身、最小工具集 | dsh Minimal mode + 固定模板 snapshot | 减少 tool 堆对分数的干扰 |
| 把已有 E2B 客户端迁到自托管 | AgentENV | API 兼容是它的显式目标 |

没有「一种 sandbox 统治所有 agent」。harness 该做的是把「执行世界」留成 seam，让本机 Landlock 和远程 Firecracker 可以换，而不重写 `ReactLoopAgent`。

---

本章完。若要接着看模型怎么在 serving 里被连续调度，见 [推理服务：从单请求推理到 SLO-aware 集群](../serving/README.md)；若要把这些轨迹收成可训练样本，见 post-training / RL 章。回到 [Agent 系统](./README.md) 可以按轴把五篇串一遍。
