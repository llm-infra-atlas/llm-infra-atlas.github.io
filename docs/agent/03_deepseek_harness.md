# 03 · DeepSeek Harness：插件化 loop

> 对照代码：[[deepseek-harness:]]（`dsh-0.1.0-rc.7`）。这是一个 developer preview，API 仍可能变动；本篇把它当作 **harness 的标本**来分析，不写安装教程，也不评价产品。
>
> 阅读本篇只需要 [`00`](./00_definitions.md) 的五个概念和 [`01`](./01_loop_and_tool_use.md) 的闭环。Cordis 的细节以仓库 [[deepseek-harness:docs/architecture.md]] 为准。

获取代码：

```bash
git clone --branch dsh-0.1.0-rc.7 https://github.com/deepseek-ai/deepseek-harness.git
# 或独立 clone
git clone https://github.com/deepseek-ai/deepseek-harness.git
```

---

## 1. 它在设计空间里的位置

DeepSeek 官方把产品收成一句话：**Agent = Model + Harness**，再加一条架构原则：everything is a plugin。落到 [`02`](./02_classic_works.md) 的缺口上：

| 缺口 | dsh 怎么填 |
|---|---|
| A 想和做 | 默认 driver 就叫 `ReactLoopAgent` |
| B 何时调用 | 交给模型；harness 只提供 schema 和执行 |
| C ACI | 具体 tool（`str_replace_editor`、文件查看、bash）是插件，不是写死在 loop 里 |
| D 代码即动作 | Code mode：tools 经 SDK 暴露，模型写 TypeScript 组合多步 |
| E 世界是一台计算机 | `ctx.fs` / `ctx.subprocess` 可指向本机，也可指向 E2B sandbox |
| F 能跑过夜 | append-only session log + resume/fork + compaction + 审批 |

它既不是新模型，也不是新的 sandbox hypervisor。模型走 `ctx.llm` 的 adapter；隔离走 `ctx.sandbox`（同主机）或 [[deepseek-harness:packages/e2b/]]（远程世界）。loop 对两者都不绑定。

---

## 2. Cordis：插件机制

底层是 [Cordis](https://github.com/cordiverse/cordis)（设计陈述：[A Programming Paradigm for Spatiotemporal Composability](https://github.com/cordiverse/paper)）。要读懂 [[deepseek-harness:packages/]]，只需记住三条规则：

1. **插件往一个共享 `ctx` 上挂服务和事件。** `ctx.sessions`、`ctx.tools`、`ctx.llm`、`ctx.sandbox` 都是挂上去的，不是 import 死的单例。
2. **登记是 effect。** `register()` 返回 disposer；插件卸载时，它贡献的 tool / listener 一起撤。没有「核心里留一个 if 给扩展」。
3. **waterfall 必须 `next()`。** `agent/pre-step`、`tools/pre-execute` 这类闸门，listener 不调用 `next()` 就是短路。这是权限、审批、sandbox 能插在 loop **外面**的原因。

跑起来的 `dsh` 是一棵按层叠出来的插件树，不是一个 `main.py`：

```
profile (web / headless / …)
  └── 按顺序叠 bundle
        └── dsh-base：模型、tools、持久化、sandbox、审批、设置
        └── dsh-web-app 或 dsh-headless
        └── 用户 cordis.patch.yml
        └── --patch 覆盖
```

`dsh --profile web --dump-config` 打印你机器上真正启动的那棵树。任何一行都可以被更高层的 patch 整段替换。这就是「不改 harness 源码换能力」的实现方式。

官方还按「给模型多少动作」出厂了几种 mode，本质是**不同的 bundle 组合**，不是 fork 出来的产品：

| mode | 模型看见什么 |
|---|---|
| Standard | 完整编码工具：编辑、shell、搜索、skills、plan、goals、subagent、workflow |
| Code / PTC | Standard 的能力，但经 Code Mode SDK 暴露，一次 generate 可以写一段程序 |
| Minimal | 只有持久 bash 和 `str_replace_editor`，给模型做最小环境基准 |
| Creator | Standard + 运行时自省，用来在内存里试插件、组新 preset |

---

## 3. 六个核心 package

[[deepseek-harness:docs/architecture.md#L44-L51]] 的表是地图。只读这一条链就能把一次用户输入跟到 observation：

| Package | `ctx` key | 干什么 |
|---|---|---|
| `core/session` | `ctx.sessions` | append-only `SessionEvent` log；`deriveMessages()` |
| `core/system-prompt` | `ctx.systemPrompt` | 拼 prompt section 和 tool schema |
| `core/tools` | `ctx.tools` | 登记 + 带闸门的执行管线 |
| `core/agent` | `ctx.agents` | `Agent` 接口、inbox、`agent/*` 事件 |
| `core/agent-loop` | `ctx.agentLoop` | **唯一**的默认 driver：`ReactLoopAgent` |
| `llm/llm` | `ctx.llm` | message / stream 词汇和模型 adapter |

扩展插件依赖 `dsh-agent` 的事件和接口，**不**依赖 `dsh-agent-loop`。loop 因此可换——这是「driver 也是插件」这句话的代码含义（[[deepseek-harness:docs/architecture.md#L48-L49]]）。

---

## 4. 两级时钟：turn 和 step

dsh 把「一次模型调用」和「一次被唤醒的工作」拆开，避免把「用户又说了一句」和「工具还欠一次 request」混在同一个 while 里。[[deepseek-harness:docs/architecture.md#L65-L82]]：

- **step** = 一次 model request + 它点名的 tools。
- **turn** = 零个或多个 step。打开于认领第一次输入之前，关闭于不再欠工作。

```text
turn/start
  claim inbox + 拼 prompt / tool schemas
  agent/pre-step          → reject | enter(messages)
     step/start
     把 entered messages 写成 user/message
     deriveMessages() → agent/request → llm/stream
         assistant/chunk* → assistant/message
     tool/call* → tools/pre-execute → execute → post-execute → tool/result*
     step/end
     若还欠 request 或 next-step 有输入 → 下一个 step
  agent/turn-stopping
turn/end
```

实现就在 `ReactLoopAgent`：

- `kick()` 是 `while (await this.turn()) {}`（`agent.ts:210-212`）。
- `turn()` 先 `session.append('turn/start', { turn })`（`agent.ts:255`），再在内层 while 里 `preStep` → 可能 `step/start`（`agent.ts:279`）。
- 第一次 claim 被 reject、或被改写成空，turn 仍然闭合，但**不花** step——log 记录「试过了」（`agent.ts:267-276`）。
- `step()` 从 `this.session.deriveMessages()` 重建发给模型的 history（`agent.ts:341`），stream 的每个 chunk 都先入 log（`agent.ts:349`），再拼成 `assistant/message`。
- 没有 tool call 则 turn 完成；有则 `executeToolCalls(...)`（`agent.ts:393-399`）。

inbox 有两个队列，对应两种「人/系统插话」（[[deepseek-harness:packages/core/agent/src/types.ts#L10]]）：

| API | 队列 | 会不会立刻叫醒 driver |
|---|---|---|
| `followup(msg)` | `next-turn` | 会。当前 turn 结束后开新 turn |
| `steer(msg)` | `next-step` | 会。当前 step 的 tools 结束后立刻进下一 step |
| `inject(msg)` | `next-step` | **不会**。等到下一次有人叫醒，才进模型视野 |

`inject` 是 skill 正文、文件变更通知、子目录 `AGENTS.md` 这类「模型该看见、但不是一句新的用户命令」的入口。`agent.ts:122-132`。

---

## 5. Session log：model-visible ⟺ logged

`SessionEventMap`（[[deepseek-harness:packages/core/session/src/types.ts#L236-L333]]）是这条不变量的类型化清单。和模型对话有关的核心事件：

| 事件 | 进不进派生 history | 作用 |
|---|---|---|
| `turn/start` · `turn/end` | 否 | 回合边界；`turn/end` 带 `TurnEndReason` |
| `step/start` · `step/end` | 否 | 一次 generate + tools 的括号 |
| `user/message` | **是** | 用户句、`inject`、goal 续跑；`source` 区分来源 |
| `assistant/chunk` | 否 | token 级 replay / UI |
| `assistant/message` | **是** | 拼好的一步输出，可带 `usage` |
| `tool/call` | 否（call 本身） | 模型点名：`name` + **未 parse** 的 `arguments` 字符串 |
| `tool/result` | **是** | 模型看见的结果；可选 `error` / `meta` |
| `request/header` | 否 | 下一次请求的完整 header 快照 |
| `session/end-seed` | 否 | 区分「种子历史」和「本生命周期新写的」 |

派生规则在 `deriveMessages()`（[[deepseek-harness:packages/core/session/src/index.ts#L726-L747]]）：只投影 surface 上的节点——`user/message` / `assistant/message` / `tool/result`（`types.ts:343-346`）。compaction 对 surface 做 `replace` 时，被挡住的节点从派生里消失，但 raw log 仍在。

由此可以得到几条工程推论，都是实现 agent 时容易遇到的问题：

1. **想让模型看见 X，就给 X 一个 session event。** 偷偷改 messages 数组再 generate，会让 resume / 审计 / 「Trajectory 视图」对不上。dsh 用运行时 invariant 管这件（`AGENTS.md` 的 *Model-visible ⟺ logged*）。
2. **UI 和模型可以看不同的投影。** chunk 留给 UI；history 只用 assembled message。
3. **fork / resume / replay 是同一份流上的操作。** `ctx.sessions.fork(source, boundary?)` 拷的是 log 前缀，不是「再 new 一个 chatbot 对象」。

---

## 6. Tool 管线：三道 waterfall

`ctx.tools` 的事件（[[deepseek-harness:packages/core/tools/src/index.ts#L142-L175]]）：

```
tools/pre-execute     允许 / 拒绝 / 问人     （waterfall，默认 next() = allow）
tools/execute         包一层 timeout / 重试 / 指标
tools/post-execute    接受 / 改写 / 挡住结果
tools/result          只观察，已经冻结
```

权限、plan mode、sandbox 的「这次能不能跑」，都挂在 `pre-execute`，**不改** `ReactLoopAgent.step`。这就是 SWE-agent 之后平台化的那一步：ACI 和 policy 是插件，loop 只认「登记过的 tool + 三道闸门」。

first-party tool 用 `defineTool`（[[deepseek-harness:packages/core/tools/src/schema.ts#L545]]）拿到类型化 arguments；MCP 来的 schema 走同一条 `register()`。模型两侧无差别。

并行调度见 [`01` §3](./01_loop_and_tool_use.md)：`executeToolCalls`（[[deepseek-harness:packages/core/agent-loop/src/tool-calls.ts#L59]]）按模型序提交，执行可重叠，result 按模型序 commit。

---

## 7. Capability seam 与两类隔离

dsh 把「一种能力」拆成三个角色（[[deepseek-harness:docs/architecture.md#L98-L102]]）：

| 角色 | 干什么 | 例子 |
|---|---|---|
| Service Definition | 声明接口 | `ctx.fs`、`ctx.subprocess`、`ctx.sandbox` |
| Service Provider | 真正实现 | 本机 fs；E2B fs；Landlock sandbox |
| Consumer | 模型面对的 tool | `dsh-tool-bash`、`dsh-tool-fs` |

**单独一个角色不构成 seam。** 换 provider 而不换 consumer，是「同一条 `bash` tool，第一次在本机跑、第二次在 microVM 里跑」能成立的原因。

这里有两条必须区分开的隔离缝，对应 [`00` §6](./00_definitions.md)：

**同主机 process sandbox**（[[deepseek-harness:packages/sandbox/]]）：

- `SandboxMode = 'read-only' | 'workspace-write' | 'danger-full-access'`（[[deepseek-harness:packages/sandbox/sandbox/src/index.ts#L29]]）。
- `danger-full-access` **不**调用 `ctx.sandbox`，原样 spawn。
- `confine(argv, policy)` 返回要替换的 argv，或 fail-closed 抛 `SandboxUnavailableError`（`index.ts:153-176`）。禁止静默裸奔。
- 本机 provider（[[deepseek-harness:packages/sandbox/sandbox-local/README.md]]）：Linux 先试 bwrap 再 Landlock，macOS Seatbelt，Windows ACL。`enforcement: 'full' | 'partial'` 必须上报，不能假装老 Landlock ABI 能管所有文件效果。
- 这层**不管**网络、不管「看见别的进程」——注释中写得很明确（`index.ts:24-27`）。

**远程执行世界**（[[deepseek-harness:packages/e2b/]]）：

- `dsh-e2b` 持有一个 E2B sandbox 句柄（[[deepseek-harness:packages/e2b/e2b/src/index.ts#L69-L74]]）。
- `dsh-fs-e2b` / `dsh-subprocess-e2b` 实现 `ctx.fs` / `ctx.subprocess`。
- 已有的 `dsh-tool-bash`、terminal、LSP **不用**为 E2B 再写一份。它们只依赖这两条 seam（[[deepseek-harness:packages/e2b/README.md#L13]]）。
- harness 进程、Cordis 对象、模型调用、session log **不**搬进 sandbox。搬的是「可变的那台 Linux」。

把 E2B_API_URL 指到 AgentENV，就是 `04` 的对接方式。dsh 自己不实现 Firecracker。

---

## 8. 读代码的最短路径

按一次真实 turn 跟行号，比按目录逛快：

```
用户输入
  ReactLoopAgent.send / followup          agent.ts:113
  kick → turn                             agent.ts:210, 246
  session.append('turn/start')            agent.ts:255
  preStep:
      inbox.claim                         agent.ts:229
      systemPrompt.assemble               agent.ts:230
      agent/pre-step waterfall            agent.ts:234
  session.append('step/start')            agent.ts:279
  session.append('user/message')          agent.ts:282-284
  step:
      deriveMessages()                    agent.ts:341 · session/src/index.ts:726
      llm.stream → assistant/chunk        agent.ts:345-349
      assistant/message                   agent.ts:381
      executeToolCalls                    tool-calls.ts:59
          tools/pre-execute → execute     tools/src/index.ts:152-163
          （可选）ctx.sandbox.confine     sandbox/src/index.ts:175
          tool/result                     types.ts:291
  step/end · turn/end                     agent.ts:292, 319
```

然后按需再展开：compaction 家族（[[deepseek-harness:packages/compaction/]]）改 surface 投影；skill 家族（[[deepseek-harness:packages/skill/]]）往 `inject` 或 loader tool 送说明书；subagent 家族是另一套 `Agent`，共用同一套事件词汇。

---

## 9. 与自写 loop 的差别

用 80 行 Python 也能写出 ReAct。dsh 多出来的、对后面读 AgentENV 有用的，是这些不变量：

1. **log 是唯一的模型上下文来源**，不是「messages 变量 + 偶尔写盘」。
2. **loop 可替换，扩展走事件**，所以 sandbox / 审批 / MCP 不会逼你改 `step()`。
3. **同主机 confine 和远程世界是两条 seam**，不要把 Landlock 和 Firecracker 都塞进一个 `if sandbox:`。
4. **失败分类**：命令失败、权限拒绝、runner 没起来、用户取消，进 log 的方式不同（[`01` §3](./01_loop_and_tool_use.md) 的表在代码里都有对应）。

这四条是「harness」和「demo 脚本」的界限。下一篇看动作越过这条界限之后，落在什么样的机器上。

---

下一篇：[04 · Sandbox：process confine 与 microVM](./04_sandbox_and_agentenv.md) —— 从 process sandbox 走到 Firecracker，对照 AgentENV 的 snapshot / fork。
