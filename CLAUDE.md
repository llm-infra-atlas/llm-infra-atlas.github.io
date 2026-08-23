# CLAUDE.md

本仓库的通用工作约定见 **[AGENTS.md](./AGENTS.md)** —— 仓库结构、`references/` 只读铁律、前置知识与写作深度、`docs/` 风格、调研顺序、配图稳定性、`docs/parallel/<dim>/` 目录形状、lab(notebook) 的生成方式与 gloo 坑，都在那里，**动手前先读它**。本文件只补充 Claude Code 专属事项。

## 一句话定位

LLM infra 学习资料库。**只在 `docs/` 创作**（深度笔记 + 可在 Mac CPU 上跑通的 lab）；`references/` 是只读 submodule，逐行对齐源码时引用其 `path:line`，**永不修改**。交互与写作默认用中文，核心概念保留英文。

## 最容易做错的三件事（速记）

1. **改了 `.ipynb` 而没改 `_build_nb.py`** —— 反了。notebook 由脚本生成：改 `docs/parallel/<dim>/_build_nb.py` 后 `python _build_nb.py <dim>_lab.ipynb` 重新生成，勿手编 ipynb。
2. **前置知识没写够 / 省略了定义** —— 不要写「读者画像」，也不要用「跳过 what-is」当借口。篇首给出前置清单；精巧设计依赖的定义式、算子作用维度、梯度怎么流，一律显式交代。AGENTS.md §2/§3 是硬约束：只讲小巧思、不铺基础逻辑＝写错了。
3. **引用源码不带行号 / 凭记忆杜撰 API** —— 必须去 `references/` 核实并写 `path:line`（Megatron pin 在 commit `e03878b5f`）。

## Claude Code 专属

- **写 lab 后务必本地跑通验证**：`python _build_nb.py ...` 生成 → 用 notebook 执行或 `jupyter nbconvert --execute` 确认所有 `allclose` 断言为 ✅，再交付。涉及 `mp.spawn` 的 lab 必须用 `%%writefile` 落盘 worker 模块。
- **触达 Claude / Anthropic API、模型选型、token/caching 等问题**：先用 `claude-api` skill，不要凭记忆作答。
- **写新文档前先调研再落笔**（见 AGENTS.md §3）：① 检索代表性论文/官方博客拿到定义与经典图；② 再浏览 `references/` / 上游代码对齐 `path:line`；③ 优先嵌入知名第三方经典示意图。
- **配图稳定性**（见 AGENTS.md §3）：**不要用 ar5iv `assets/xN.png` 直链**（常返回「no image available」占位图，且 HTTP 仍是 200）。优先从 arXiv e-print 取源图转到**引用该图的文档旁** `assets/arxiv/`，或用官方文档 / `raw.githubusercontent.com`。**铁律：核验时看尺寸/字节，不能只看 content-type**；每图配 `> 图：…（作者 年份, Fig N；arXiv）` 出处说明。
- **持久记忆**：`docs/parallel` 的风格/结构/lab 约定已记录在记忆 `parallel-docs-conventions.md`（含本机 gloo 已验证的 collective 与坑）。发现新的、跨 session 有用的约定时更新该记忆，而不是塞进对话。
- 探索代码用 Explore/Grep 等只读工具优先；`references/` 体量大，定位实现按 README 的代码映射表导航更快。
