# LLM Infra Atlas

<p align="center">
  <img src="./assets/images/favicon-512.webp" width="120" alt="LLM Infra Atlas">
</p>

<p align="center">
  <b>持续更新的 LLM Infra 全景笔记</b><br>
  <a href="https://llm-infra-atlas.github.io">🌐 llm-infra-atlas.github.io</a>
</p>

---

## 为什么 LLM 时代需要这个教程？

先说一个事实：**这个领域知识生产的速度，已经远远甩开了知识传播的速度。**

一本书从动笔到上架要一两年，一门课从备课到开讲要一学期，一篇论文从投稿到接收也异常漫长——而 LLM infra 的「最佳实践」是以天为单位迭代的：顶尖公司和实验室的开源项目在 day 0 就能成为行业标配。等你看到系统化的整理，它描述的往往已经是上一个时代。

另一个问题是**碎片化且不够深入**。传统知识检索方式无法看清全貌。论文/技术报告零碎地告诉你 what 和 why，代码告诉你 how，但中间隔着一层：论文里的示意图怎么对应到 vLLM / Megatron-LM 里哪个文件的哪一行？「通信-计算 overlap」落到 CUDA stream 上到底长什么样？

所以这份教程的生产方式本身就是对这个时代的回应：**用 agent 高效整合前沿知识**——论文、技术报告、第三方博客、开源实现，逐一对齐；再由人把关每一处定义、每一个行号、每一张图。Agent 负责跟上速度，人负责保证质量。

## 三个特质

**1. 够新，也够全。** 覆盖训练到推理的完整链路。写的不是教科书或论文里的「经典」，而是当下业界真正在用的东西。

**2. 面向生产，源码对照。** 每个机制都对齐到 SOTA 开源实现，引用精确到 `path:line`。

**3. 算法知识充分铺垫。** 先讲明白模型架构和算法相关的必要背景——infra 从何而来，要服务什么。

## 章节地图

章节按三个层次组织：**模型与算子**讲模型架构与关键算子，**训推系统**讲训练与推理的全过程系统实现，**底层技术**讲硬件、网络与开发技术栈。

### 模型与算子

| 章节 | 你会读到什么 |
| --- | --- |
| [前沿开源模型架构速览](docs/frontier_open_models.md) | 一张总表纵览前沿开源模型的架构选择：attention 形态、MoE 配置、上下文长度、多模态方案，也是后面各专题的索引 |
| [Attention](docs/attention/README.md) | 两条线索：FlashAttention 系列的 kernel 实现（IO-aware tiling、FA2/FA3/FA4、稀疏与线性变体）；attention 机制的演进（MHA→MLA、稀疏、线性、混合） |
| [MoE](docs/moe/README.md) | 架构视角：MoE 定义、细粒度专家、LatentMoE、三条负载均衡路线；算子视角：grouped GEMM、DeepEP 通信 kernel、MegaMoE 融合 kernel |
| [多模态](docs/multimodal/README.md) | 视觉/音频如何变成 token 进入 LLM：对比预训练、encoder、经典 VLM、融合与 connector、生成器；以及多模态带来的异构、变长、冗余三类系统问题 |
| [Residual](docs/residual/README.md) | 残差连接与 Norm 放置的经典结论，以及 Hyper-Connections、attention residual 等近期工作 |
| [Speculative Decoding](docs/speculative_decoding/README.md) | 用 draft-then-verify 无损加速 decode：token tree、draft 方法谱系、MTP、EAGLE，以及 serving 中的落地 |

### 训推系统

| 章节 | 你会读到什么 |
| --- | --- |
| [并行策略](docs/parallel/README.md) | DP/ZeRO/FSDP、TP/SP、PP、CP、EP 的全流程逻辑，每个维度都配一个可运行的 lab |
| [训练系统](docs/train/README.md) | 一个训练 iteration 的完整生命周期：train loop、optimizer、checkpoint、dataloader、buffer、activation recompute/offload、显存模型 |
| [推理服务](docs/serving/README.md) | prefill/decode 与 SLO、continuous batching、PagedAttention、分层 KV cache、P/D 分离、调度和 overlap、多模态 serving |
| [后训练](docs/post_train/README.md) | CPT/SFT 与 PPO/GRPO 家族等算法，以及 rollout–train 系统设计、训推一致性、agentic RL |
| [低精度](docs/low_precision/README.md) | 从 FP8 到 FP4：数值格式与 scaling 粒度、DeepSeek 在 Hopper 上的 FP8 训练方案、Blackwell 原生 FP4/FP8 支持 |
| [Agent](docs/agent/README.md) | Agent 的定义与循环、经典工作、DeepSeek Harness、process sandbox 与 microVM |

### 底层技术

| 章节 | 你会读到什么 |
| --- | --- |
| [HPC · 集群与网络](docs/hpc/README.md) | GPU 硬件参数、roofline 模型、scale-up/scale-out 网络、集合通信、RDMA/IB verbs、大规模可靠性 |
| [PyTorch](docs/torch/README.md) | 框架开发常用的底层 API：内存布局、autograd、distributed、CUDA stream/graph、compile、caching allocator |
| [Profiling](docs/profiling/README.md) | 性能与显存观测：torch.profiler、显存 snapshot/memory_viz、Nsight Systems / Compute |
| [CUDA & DSL](docs/cuda_dsl/README.md) | CUDA 编程基础与 Triton / CuTile / CuteDSL；GEMM+collective 通算融合的 tile/chunk 流水与 persistent kernel |

**不知道从哪开始？** 三条路线供参考：

- **想搭训练 infra**：先读[并行策略](docs/parallel/README.md)和[训练系统](docs/train/README.md)，配合 [HPC · 集群与网络](docs/hpc/README.md)理解硬件和环境，需要时查 [PyTorch](docs/torch/README.md) 和 [Profiling](docs/profiling/README.md)。
- **想做推理 / serving**：先读[推理服务](docs/serving/README.md)，再补 [Attention](docs/attention/README.md)、[低精度](docs/low_precision/README.md)与[投机解码](docs/speculative_decoding/README.md)，以及[并行策略](docs/parallel/README.md)中的 [TP](docs/parallel/02_tp_sp/README.md)/[EP](docs/parallel/05_ep/README.md)/[CP](docs/parallel/04_cp/README.md)。
- **想搞清楚模型架构和 infra 的 co-design**：从[前沿开源模型架构速览](docs/frontier_open_models.md)入手，按兴趣深入 [Attention](docs/attention/README.md)、[MoE](docs/moe/README.md)、[多模态](docs/multimodal/README.md)等，再接算子和训推的相关章节。

## 代码对照

| 项目 | 上游仓库 | 对照什么 |
| --- | --- | --- |
| Megatron-LM | [NVIDIA/Megatron-LM @ e03878b5](https://github.com/NVIDIA/Megatron-LM/tree/e03878b5ffc2698ed1d63fc2ec434cb2b5f122e2) | 训练框架：TP/PP/SP/EP、DistOpt、checkpoint |
| SGLang / vLLM | [sglang @ 1a5775a9](https://github.com/sgl-project/sglang/tree/1a5775a9df3069574109a22d00d167269fe9c0ff) / [vLLM @ 156b1266](https://github.com/vllm-project/vllm/tree/156b12667cf5fbb93914f3646acc25dca378b420) | 推理服务：调度、KV cache、P/D 分离 |
| Mooncake / LMCache | [Mooncake @ f90ae691](https://github.com/kvcache-ai/Mooncake/tree/f90ae691f109e49a60920e0c8abbf7e572826d8c) / [LMCache @ 09bc14c0](https://github.com/LMCache/LMCache/tree/09bc14c0a5fd6eec9afc3634f8f726f5249febf7) | KVCache-centric serving / 分层 KV cache |
| DeepEP / DeepGEMM | [DeepEP @ d4f41e4e](https://github.com/deepseek-ai/DeepEP/tree/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb) / [DeepGEMM @ 88965b07](https://github.com/deepseek-ai/DeepGEMM/tree/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1) | MoE all-to-all / FP8 grouped GEMM |
| FlashAttention / FLA | [flash-attention @ fb02fc8b](https://github.com/Dao-AILab/flash-attention/tree/fb02fc8b56413e647b7060418b537858d6175d89) / [fla @ 81091cc6](https://github.com/fla-org/flash-linear-attention/tree/81091cc6d71695e5a23e739f03bbd2fc627b72d4) | attention kernel / 线性·稀疏 attention |
| slime / checkpoint-engine | [slime @ 41014d1f](https://github.com/THUDM/slime/tree/41014d1f29e201137fdffce737bb8bac65bc5219) / [checkpoint-engine @ d1de07b3](https://github.com/MoonshotAI/checkpoint-engine/tree/d1de07b3aacff34050d09c3efa093f9a2fcdcf73) | RL 后训练 / serving 侧在线权重更新 |
| DeepSeek Harness / AgentENV | [deepseek-harness @ 99f6f02f](https://github.com/deepseek-ai/deepseek-harness/tree/99f6f02fecdb7dff40c3fbc9470f5907c29f74ca) / [AgentENV @ 547c1a8a](https://github.com/kvcache-ai/AgentENV/tree/547c1a8a515382cfc2ba02cb8aeb1bd134b7327f) | Agent loop / Firecracker sandbox |

## 如何 contribute

这个知识库由「人 + coding agent」共同维护：agent 负责检索、对齐、起草、验证，人负责定结构、把关正确性、砍掉正确的废话。写作的全部约定都在 [AGENTS.md](https://github.com/llm-infra-atlas/llm-infra-atlas.github.io/blob/main/AGENTS.md)。

发现错漏、过时的地方，或者有想看的主题，欢迎来 [GitHub](https://github.com/llm-infra-atlas/llm-infra-atlas.github.io) 开 issue，提 PR。这个领域唯一的常量就是变化，这个仓库也会持续更新。
