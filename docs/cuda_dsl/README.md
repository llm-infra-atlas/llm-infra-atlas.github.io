# CUDA & DSL

> 本章仍在持续维护中。考虑到这块的知识量与优质参考资料都很多，本页倾向于以「整理优质现有资源」的方式呈现，而不是从头重写一套教材。面向 LLM 算子的 GPU 编程，推荐参考 CMU Machine Learning Systems 课程系列的 [Modern GPU Programming for ML Systems](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/index.html)：以 GEMM 和 FlashAttention 4 为贯穿示例，从 GPU 执行模型、数据布局、TMA 异步数据搬运、tensor core 一路讲到 warp specialization 与完整的 attention kernel，与本章的主题高度重合。本页保持这一基调；[01 · 通算融合](./01_fused_collective_gemm.md) 则是一次专题研讨，把视线从单卡 kernel 延伸到多卡场景，讨论 GEMM 与 collective 如何按 tile/chunk 接成流水。

## 阅读顺序

| 文件 | 内容 | 关键问题 |
| --- | --- | --- |
| `README.md`（本文） | GPU 编程主题与 DSL 层次 | 不同抽象层分别控制什么 |
| [01 · 通算融合：GEMM 与 collective 的 tile 流水](./01_fused_collective_gemm.md) | producer–consumer、tile/chunk、GEMM+RS / GEMM+AR / AG+GEMM、persistent kernel、warp specialization、in-flight 与 buffer ownership | 如何把 tensor-level 依赖降成 tile/chunk 级流水 |

## 基础主题

- **执行模型**：grid/block/warp、SM 占用率（occupancy）、warp divergence。
- **存储层级**：global / shared / register、bank conflict、coalesced access、async copy (`cp.async`)。
- **同步与原语**：`__syncthreads`、warp shuffle、cooperative groups、barrier。
- **tensor core**：MMA、WGMMA（Hopper）、TMA，对接本文 §2 的 DSL 分层与 [FlashAttention —— Infra 视角深入](../attention/fa/README.md)。

## DSL 层次

这几种 DSL 的共同目标，是把 kernel 的 tile、layout 和 memory hierarchy 描述交给编译器去做 lowering；越往下走，用户就需要越接近线程映射、shared memory 和硬件 atom 这些底层细节：

```
抽象更高，编译器决策更多
│
├── Triton          tile 级：`tl.load` / `tl.store` / `tl.dot`
│                   SMEM、layout、pipeline 基本交给编译器
│
├── CuTile          NVIDIA CUDA Tile Python DSL / Tile IR
│                   和 Triton 一样写 tile，不直接写线程
│
├── CuteDSL         显式描述 layout、memory hierarchy、pipeline 与 warp policy
│
└── CUDA C++ / CUTLASS C++   thread / warp / hardware atom 级控制
抽象更低，硬件控制更多
```

| 层次 | 代表 DSL | 编程粒度 | 用户主要写什么 | 编译器主要代管什么 |
| --- | --- | --- | --- | --- |
| tile + 编译器 | Triton、CuTile | block / tile | tile 上的 load / store / MMA | SMEM、layout、多数 pipeline |
| tile + 显式存储 | CuteDSL | tile（block / warp） | 数据放置、pipeline、warp 切分 | layout 推断、后端 lowering |
| thread / SIMT | CUDA / CUTLASS C++ | thread / warp | 线程映射、memory hierarchy、同步 | 较少，换来更细的硬件控制 |

FA4 的 CuTeDSL 路径可以对照 [FA4 / CuTe DSL](../attention/fa/04_fa4_cutedsl_and_api.md) 来看；多卡算子如何把 GEMM epilogue 与 collective 接成 persistent producer–consumer pipeline，则见[通算融合](./01_fused_collective_gemm.md)。Triton、CuTile、CuteDSL 各自的实现细节，后续会继续在本页展开。
