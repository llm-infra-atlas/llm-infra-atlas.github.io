# AGENTS.md

面向任何 AI coding agent 的工作约定。本仓库是一份 **LLM infra 方向的学习资料库**，核心产物是 `docs/` 下的笔记。

> 交互语言：默认用**中文**沟通与写作。除非用户要求，否则不得修改本文件。
> 维护本文件时：新约定优先并入既有条目、务必保持简洁。

## 写作风格约定（`docs/` 通用）

- **行文语气：半书面化的技术讲义**。句子完整、过渡自然，娓娓道来。把信息按逻辑顺序展开，而不是把高浓度术语压缩进短句。具体要求：
  - 不过度用比喻与黑话，直接陈述事实。
  - 加粗节制，只给真正关键的结论。
  - 标题（H1 与各级小节）用最短的短语说清内容，少用括号注释、箭头链、设问与比喻。
  - 篇首引言写成自然的承前启后段落。
- **教科书式讲解优先，代码对账殿后**：本质是教程而不是代码文档，禁止把文章写成以函数调用链为主线的逐行 code walkthrough。
- **调研顺序（写新内容前必做）**：先检索代表性论文 / 官方博客 / 技术报告（arXiv、NVIDIA/Google/Meta 工程博客、原作者 talk），弄清问题动机、符号约定与经典示意图；
- **中文行文，核心概念用英文**，不做无意义的中文转译：stride / view / contiguous / autograd / collective / dispatch / combine / grouped GEMM / all-to-all / attention 共轭算子 等一律保留英文。attention 不使用中文译名。计算 kernel 的中文一律写「算子」（或保留英文 kernel），禁用“内核”。
- **数学公式规范**：
  - inline 使用 `$...$`，display 使用各自独占一行的 `$$` 开闭；禁止 `\(...\)` / `\[...\]`、同一行 `$$...$$` 和未闭合 delimiter。标题、导航文字、链接文字与图片 alt 不写 TeX，改用 Unicode / plain text（如 `α`、`H×W`、`O(Ld)`），避免 TOC/alt 暴露原始公式。
  - 用 LaTeX 还是普通符号取决于：是否为数学主导章节，或参与本文整体性公式推导。用普通符号：带单位的工程估算（`≈ 586 MB`、`int32 × (1 × 146.5M)`）、倍数（`~2×`）、正文孤立的比较符（`≥`、`≠`、`≫`）、不进任何公式的孤立记号（`第 i 条`、`N-1 个 rank`、shape `[2, 4096]`）。没有公式的章节（如 dataloader、工程配置类）整体不用 LaTeX。
- **主动检索并嵌入经典第三方配图（提升直观性）**：讲某个机制时，优先使用出处论文 / 官方文档 / 知名开源仓库里已被广泛引用的示意图。

## 网站（GitHub Pages / MkDocs Material）

仓库以 GitHub Actions 把 `docs/` 直接发布到 <https://llm-infra-atlas.github.io>。没有中间文档副本、front matter 注入或 Jekyll 脚手架；`site/` 只是被忽略的构建产物。

- `mkdocs.yml` 是站点唯一配置：Material 主题、**完整显式导航**、Markdown 扩展、源码仓库 URL/pin 都在这里。
- 每个导航 section 的第一项必须是对应 `README.md`，配合 `navigation.indexes`。新增、移动或重命名页面时必须同步维护 `nav`，并按知识分类放入正确层级。
- 根 `README.md` 是 GitHub 与网站首页的唯一内容源，不得再维护第二份首页正文。README 中的章节链接写仓库根相对路径（如 `docs/parallel/README.md`）。
- 首次使用：`python3 -m pip install -r requirements.txt`。本地预览用 `mkdocs serve`。交付前校验：`mkdocs build --strict`；它会检查配置、内部文档链接、导航和源码 shortcode。

## 外部代码参考和引用规范

**代码引用规则：** `references/` 下的文件只用于本地阅读，不在这里写入改动，也不参与网站构建。正文统一使用 `[[project:path#Lx-Ly]]` 短语法。

常用写法：

```markdown
[[megatron-lm:megatron/core/transformer/moe/moe_utils.py#L579-L590]]
[[fla:fla/ops/kda/]]
[[atlas:docs/parallel/01_dp/dp_lab.ipynb|运行 DP lab]]
```

页面默认只显示简洁的 `文件名:Lx–Ly`；多段行号会生成多个锚点；`|...` 可覆盖可见文字。项目名、URL 与 commit 必须先登记到 `mkdocs.yml`。

## 提交约定

- 主要内容写在 `docs/`；站点静态资源放根 `assets/`；站点行为修改 `mkdocs.yml` 或 `hooks/`；不动 `references/` 镜像。
- commit message 简洁描述新增/修改的文档或 lab。除非用户要求，不主动 commit / push。