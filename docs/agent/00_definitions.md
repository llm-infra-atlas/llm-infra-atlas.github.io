# 00 · 五个词：agent / tool / environment / harness / sandbox

> 本篇只讲清楚词汇本身。读完之后应该能说出：一次普通的 chat completion 缺了哪几块才会变成 agent；harness 和 model 之间的边界画在哪里；sandbox 为什么不能算作「再写一个 tool」。循环的协议细节、经典论文、两份开源代码分别放在 `01`–`04` 里展开。

---

## 1. 从一次 chat 说起

一次普通的 chat completion 是这样的：

```
messages  ──►  model.generate()  ──►  assistant text  ──►  给用户
```

模型的输出**就是**任务的结束。这里没有外部世界，也没有第二回合，失败了也只能重新生成一次。

Agent 多出来的，是一个闭环：模型的某些输出不再被当成最终答案，而是被解释成对外部世界的一次请求；世界执行完之后，把结果写成一条新的 message，再交给模型去看。

```
messages + tool schemas
        │
        ▼
  model.generate()
        │
        ├─ 最终文本 ──────────────────────────────► 结束
        │
        └─ tool calls ──► environment.execute()
                                │
                                ▼
                          observation ──► 写回 messages ──► 再 generate
```

「闭环」这两个字已经足够当作定义来用了。后面所有的术语，其实都是在给这个闭环的各个环节命名。

---

## 2. Agent：一个会行动的 policy

把这个交互写成和强化学习教材同构、但不依赖 RL 的最小形式（这是 ReAct 论文 §2 用的记号）：

$$
o_t \in \mathcal{O},\quad
a_t \in \mathcal{A},\quad
c_t = (o_1, a_1, \ldots, o_{t-1}, a_{t-1}, o_t),\quad
\pi(a_t \mid c_t)
$$

其中，**observation** $o_t$ 是环境给出的反馈，Wikipedia 段落、bash 的 stdout、编译器报错、页面 DOM，都算是 $o_t$。**action** $a_t$ 是会改变环境的操作，比如 `Search[Apple Remote]`、`bash -c 'pytest'`、`str_replace`，都算是 $a_t$。**context** $c_t$ 是到目前为止积累的轨迹，在 LLM 里它就是 messages 数组，外加 system prompt 和 tool schema。**policy** $\pi$ 负责在给定 $c_t$ 的情况下选出下一个 $a_t$，LLM agent 里的 $\pi$ 其实就是一次 generate。

ReAct 做的那一步推广，是把动作空间扩展成

$$
\hat{\mathcal{A}} \;=\; \mathcal{A} \;\cup\; \mathcal{L}
$$

其中 $\mathcal{L}$ 是语言。落在 $\mathcal{L}$ 里的 $\hat{a}_t$ 叫作 **thought**（或者 reasoning trace）：它**不会**改变环境，因此也不会产生新的 $o_{t+1}$；它只是被追加进 $c_{t+1}$，给后续的决策当脚手架用。thought 通常用来拆解目标、记录进度、处理例外情况、从 observation 里抽取要点——这些事情，只会 Act 不会 Thought 的 agent 做不好，而纯 CoT 又因为没有环境可以对照而无从验证。

综合起来可以这样定义：

> **LLM agent = 一个把 $\pi(\cdot \mid c_t)$ 实现成「语言模型 + 可选工具调用」的 policy，在某个 environment 上跑多步，直到产出最终答案或者耗尽预算为止。**

它**不是**一种新的模型架构，也**不是**一种训练算法。同一个基座模型，套上不同的 harness 或 environment，就会变成不同的 agent。

它和几个近亲的差别可以对照着看：

| | 决策 | 是否改外部世界 | 是否多步 |
|---|---|---|---|
| chat completion | 一次 generate | 否 | 否 |
| RAG / tool 一次调用 | 一次 generate，可能带一次检索 | 通常只读 | 否（单跳） |
| **agent** | 多次 generate | 是，且结果依赖上一步 | 是，步数不事先固定 |
| workflow / DAG | 人写死的图 | 是 | 是，但分支是编好的 |

workflow 里「下一步调谁」是写死在代码里的，agent 里「下一步调谁」则是模型当场采样出来的。前者可预测、便于审计；后者能处理那些事先没有枚举过的中间状态。生产系统里经常两者叠加使用：外层用 workflow 规定阶段，内层某个节点本身是一个 agent。

---

## 3. Tool：模型可调用的外部能力

**tool** 是 environment 暴露给模型的一等公民接口。一条 tool 至少要包含三样东西：一个**名字**，供模型点名使用（比如 `bash`、`read_file`、`search`）；一份 **schema**，用来描述参数，今天几乎总是 JSON Schema，模型会按 schema 去填 arguments；以及一个**执行体**，也就是 harness 在模型点名之后真正会去执行的函数、RPC 或子进程。

模型看得见的只是名字和 schema 这两样，执行体对它是不可见的。这其实是一条安全边界：模型不能凭空「发明」一个没有登记过的 tool，也拿不到执行体本身的指针。

有几个容易混淆的词值得放在一起区分：

| 词 | 指什么 |
|---|---|
| **function calling** | 模型原生的结构化输出通道：不再用 `Action: foo[bar]` 这种文本约定，而是在 API 里返回 `tool_calls: [{name, arguments}]` |
| **tool** | 一条登记过的能力。function calling 是「怎么说出口」，tool 是「说的是哪一条」 |
| **skill** | 一段可发现、可加载的**说明书**（典型是带 YAML front matter 的 `SKILL.md`），不是可执行函数。模型先读 skill，再决定调哪些 tool |
| **MCP tool** | 把 tool 的登记和调用做成跨进程协议（Model Context Protocol）之后的同一物件 |

tool 和 skill 的分工可以这样理解：tool 是动词（「跑这条 bash」），skill 是剧本（「修 CI 失败时先看 workflow 文件，再跑最小复现」）。DeepSeek Harness 把 skill 做成了独立的 capability 家族（[[deepseek-harness:packages/skill/]]，`ctx.skills`），而模型面对的仍然只是一条 `skill` loader tool。

---

## 4. Environment：动作真正改写的世界

**environment** 是 $T(o_{t+1} \mid c_t, a_t)$ 的实现：接收 action，返回 observation，并且自己可以持有隐藏状态。

同一套 loop 可以接入很不一样的环境：

| 环境 | observation 长什么样 | 状态存在哪 |
|---|---|---|
| Wikipedia API（ReAct 原论文） | 检索到的段落 | 无（只读） |
| ALFWorld / 文本游戏 | 「你看到一个抽屉」 | 模拟器内部 |
| 本地 git 仓库 | `cat` / `grep` / 测试输出 | 工作树、进程 cwd |
| 浏览器 / 桌面 | 截图、A11y tree、DOM | 页面 / OS |
| 远程 Linux sandbox | stdout、文件、HTTP 端口 | 一台隔离机器的磁盘和内存 |

环境有两个工程性质，`04` 会反复用到。第一个是**状态是否持久**：一次 `bash` 的 cwd、装过的包、写过的文件，下一步还在不在？CodeAct 和 coding agent 都要求「在」，所以环境往往是一台活的机器，而不是一个无状态函数。第二个是**失败是不是 observation**：命令返回非 0、文件不存在、网络超时，这些都必须回到模型面前变成 $o_t$，而不是把 harness 自己搞崩。ReAct 之所以能纠错，靠的正是失败对模型可见。

---

## 5. Harness：闭环的运行时

模型自己不会记 session、不会自己执行 JSON、不会在超长轨迹里做 compaction，也不会在用户说「停」的时候取消正在跑的 bash。这些全都是 **harness** 的工作。

DeepSeek 给出的定位是：

> 模型是 agent 的灵魂。Harness 给予 agent 理解环境、使用工具，并在真实场景中持续工作的能力。

把这句话操作化成一份清单，缺任何一项，其实只是在写 demo，还谈不上是在写 harness：

| 职责 | 具体做什么 |
|---|---|
| **loop** | 决定何时再调一次模型、何时该停、并行 tool 怎么排 |
| **session / trajectory** | 把模型看见过的一切记成可恢复的 log；resume / fork / replay 都从这份 log 来 |
| **prompt assembly** | 拼 system prompt、tool schema、派生 history |
| **tool registry + pipeline** | 登记、校验参数、执行前拦截（权限 / 审批）、执行、执行后改写结果 |
| **policy** | 哪些路径能写、要不要问人、何时升级权限 |
| **context 管理** | 超窗时 compact / 摘要 / 剪 tool result，且改动必须反映到 log |
| **surface** | CLI / Web UI / IDE / ACP，让人能看轨迹、注入、取消 |

harness **不是**模型，也**不是** environment。把 bash 的实现换到远程 VM 上，loop 和 log 完全可以不动，这正是 `03` 里 dsh 那道 capability seam 的意义；把模型从 DeepSeek 换成别的 adapter，loop 同样可以不动。

SWE-agent 强调的 **ACI** 是 harness 的一个切面：问题不在于「有没有 bash」，而在于「命令和反馈是不是按模型的注意力和错误模式设计出来的」。给模型一个 100 行的文件窗口，配合失败时拒绝语法错误的 edit，效果比把整个 Linux shell 原样塞进 prompt 要好得多。可以说 ACI 是接口设计，harness 是整台机器。

---

## 6. Sandbox：environment 的隔离边界

**sandbox** 回答的问题是：action 究竟发生在谁的内核里、谁的文件系统里、谁的网络里。

它不是又一条 tool。`bash` 是 tool；「这条 bash 是被 Landlock 包住的，还是进了一台 Firecracker」才是 sandbox 要回答的问题。同一条 `bash` tool，可以换成三种不同的落地方式：

```
harness ── ctx.shell / ctx.subprocess ──► 本机进程（danger-full-access）
                                      ──► 本机进程 + bwrap/Landlock（process sandbox）
                                      ──► 远程 microVM 里的进程（E2B / AgentENV）
```

有两层常见的隔离，职责并不相同，不应该混成一个词：

| | **process sandbox**（dsh `ctx.sandbox`） | **environment sandbox**（AgentENV / E2B） |
|---|---|---|
| 和 harness 的关系 | **同一台机器、同一个内核** | **另一台 guest**，独立 kernel |
| 管什么 | 这次 spawn 的 argv 能写哪些宿主机路径 | 整台机器的磁盘、内存、网卡、进程表 |
| 典型实现 | bubblewrap、Landlock、Seatbelt、Windows ACL | Firecracker microVM、gVisor、Kata |
| 换它影响谁 | 只影响「怎么 wrap argv」 | fs、shell、PTY、LSP 一起搬走 |
| dsh 里的位置 | [[deepseek-harness:packages/sandbox/]] | [[deepseek-harness:packages/e2b/]]（把 `ctx.fs` / `ctx.subprocess` 指到远程） |

dsh 自己在 [[deepseek-harness:packages/sandbox/sandbox/src/index.ts#L1-L5]] 里写得很清楚：process-sandbox 这道 seam 包的是同一个世界（same-world）里的 argv；容器、microVM、远程执行是「整条 capability seam」的兄弟实现，**不是** `ctx.sandbox` 的一种 provider。

`04` 会把这张谱系从「为什么要隔离」讲到 AgentENV 的 snapshot / fork。这里只需要记住一点：sandbox 是 environment 的实现约束，不是 loop 本身的一部分。

---

## 7. 一条最小轨迹示例

把上面五个词串成一次具体的运行来看（数字是示意）：

```
user: "这个测试为什么挂了？"

harness:
  append  user/message
  assemble system + tools=[bash, read_file, str_replace]
  model.generate()
      → tool_call bash("pytest tests/test_foo.py -q")
  sandbox: 在 workspace-write 下 spawn 被 confine 的 argv
  environment: 退出码 1，stderr 有 AssertionError
  append  tool/result  (observation)
  model.generate()
      → tool_call read_file("src/foo.py")
  ...
  model.generate()
      → assistant text "根因是 … 我改了第 42 行"
  turn/end
```

这里，**agent** 是整段交互的 policy（模型加上这套工具）；**tool** 是 `bash` 和 `read_file`；**environment** 是那棵 git 工作树和它上面跑着的进程；**harness** 是负责 append、assemble、generate、confine、再 append 的那台机器；**sandbox** 则是那道让 `pytest` 没能以 harness 的 UID 随便写 `/` 的边界。

DeepSeek Harness 把「模型看见过的一切」收进一份 append-only 的 `SessionEvent` log（[[deepseek-harness:packages/core/session/src/types.ts#L236]]）。下一次 generate 用到的 history，其实就是这份 log 的投影，而不是另外单独维护的一份 messages 数组。这就是 `03` 的入口不变量：**model-visible ⟺ logged**。

---

下一篇：[01 · 循环与协议：ReAct、function calling、MCP 与 CodeAct](./01_loop_and_tool_use.md) —— 把这条闭环的协议从 ReAct 文本，讲到 function calling、MCP 和 CodeAct。
