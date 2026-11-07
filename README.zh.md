<div align="center">

<img width="50%" alt="Flash Linear Attention" src="images/logo.png">
<br>

[![hf_model](https://img.shields.io/badge/-Models-gray.svg?logo=huggingface&style=flat-square)](https://huggingface.co/fla-hub) [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/vDaJTmKNcS)

</div>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p>
  💥 Flash Linear Attention（FLA）汇聚了针对现代序列模型的硬件高效基础构件、开箱即用的训练层和核心组件，涵盖线性注意力（Linear Attention）、稀疏注意力（Sparse Attention）、状态空间模型（State Space Models, SSM）以及混合大语言模型架构（Hybrid LLM Architectures）。所有实现均具备跨平台通用性，并在 NVIDIA、AMD 和 Intel 硬件上经过了充分验证。热烈欢迎社区提交 Pull Request！
</p>

--------

* [最新动态](#最新动态)
* [支持模型](#支持模型)
* [安装指南](#安装指南)
* [使用方法](#使用方法)
  * [词元混合层 (Token Mixing)](#词元混合层-token-mixing)
  * [融合模块 (Fused Modules)](#融合模块-fused-modules)
  * [文本生成 (Generation)](#文本生成-generation)
  * [混合模型 (Hybrid Models)](#混合模型-hybrid-models)
* [模型训练](#模型训练)
* [模型评测](#模型评测)
* [性能基准 (Benchmarks)](#性能基准-benchmarks)
* [学术引用](#学术引用)
* [致谢](#致谢)

## 最新动态

- [2026-10] 🌊 在 `fla` 中新增 [CyFA](https://arxiv.org/abs/2609.36259)（Cyclic Flow Attention，循环流注意力）实现 —— 一种通过学习型循环传输按相对时间对记忆进行分区的线性 RNN。
- [2026-10] 🎉 在 `fla` 中新增折棍注意力（Stick-Breaking Attention）算子核心与模型实现（[论文](https://arxiv.org/abs/2410.17980)）。
- [2026-07] 🐈 在 `fla` 中新增 CAT（Compress and Attend Transformer，压缩与注意力 Transformer）实现（[论文](https://arxiv.org/abs/2511.05313)）—— 一种支持在测试时动态控制推理开销的*元序列混合器*（meta-sequence mixer）。
- [2026-07] 🧱 为 [AttnRes](fla/ops/attnres) 新增 [Gluon](https://triton-lang.org/main/getting-started/tutorials/gluon/) 后端。
- [2026-07] 🚀 为 [门控 DeltaNet](fla/ops/gated_delta_rule) 新增 [FlashQLA](https://github.com/QwenLM/FlashQLA) 后端支持。
- [2026-06] 🔭 在 `fla` 中新增 Parallax（视差注意力）实现（[论文](https://arxiv.org/abs/2605.29157)）。
- [2026-06] 🧮 在 `fla` 中新增预条件门控 DeltaNet（PGDN）与预条件 KDA（PKDA）（[论文](https://arxiv.org/abs/2604.21100)）—— 基于 ATK 预条件子实现感知曲率的线性循环预条件化。
- [2026-06] 🧱 在 `fla` 中新增 Wall attention（墙式注意力）实现（[技术博客](https://blog.tilderesearch.com/blog/wall-attn)）。
- [2026-05] 🚪 在 `fla` 中新增 Gated DeltaNet 2 (GDN-2) 实现（[论文](https://arxiv.org/abs/2605.22791)）。
- [2026-05] 🦅 在 `fla` 中新增 Raven（渡鸦模型）实现（[开源仓库](https://github.com/goombalab/raven)）。
- [2026-05] 🚀 在 `fla` 中新增 [YOCO](https://arxiv.org/abs/2405.05254)（You Only Cache Once，只需缓存一次）实现。
- [2026-05] ⚡ 在 `fla` 中新增融合版 [AttnRes](fla/ops/attnres) 支持（[论文](https://arxiv.org/abs/2603.15031)）。
- [2026-04] 🐍 在 `fla` 中新增 Mamba3 实现（[论文](https://arxiv.org/abs/2603.15569)）。
- [2026-04] 🧱 在 `fla` 中新增 [MoBA](https://arxiv.org/abs/2502.13189)（Mixture of Block Attention，分块混合注意力）实现，并支持 [FlashMoBA](https://github.com/mit-han-lab/flash-moba) 后端。
- [2026-04] 🧱 为部分核心算子新增 [TileLang](https://github.com/tile-ai/tilelang) 后端支持。
- [2026-04] 🎯 为 `fla` 的注意力算子新增类似 [GPT-OSS](https://openai.com/index/introducing-gpt-oss/) 风格的注意力池（Attention Sink）支持。
- [2026-03] 🚀 为 KDA 与 GDN 新增[上下文并行 (Context Parallel)](fla/ops/cp/README.md) 支持，实现沿序列维度的高效分布式训练。

<details>
<summary>点击展开历史动态</summary>

- [2025-10] 🌘 在 `fla` 中新增 Kimi Delta Attention (KDA) 实现（[论文](https://arxiv.org/abs/2510.26692)）。
- [2025-09] 🌲 在 `fla` 中新增 DeltaFormer 实现（[论文](https://arxiv.org/abs/2505.19488v1)）。
- [2025-09] 🐻 激动地宣布：[GDN](fla/ops/gated_delta_rule) 已正式集成到 Qwen3-Next 中！详情请参阅其[官方博客](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)。
- [2025-08] 🌲 在 `fla` 中新增 Log-Linear Attention（对数线性注意力）实现（[论文](https://arxiv.org/abs/2506.04761)）。
- [2025-08] 🎓 在 `fla` 中新增 MoM（Mixture-of-Memories）实现（[论文](https://arxiv.org/abs/2502.13685)）。
- [2025-07] 🐳 在 `fla` 中新增 MLA（Multi-head Latent Attention）实现（[论文](https://arxiv.org/abs/2405.04434)）。
- [2025-07] 🛣️ 在 `fla` 中新增 PaTH Attention 实现（[论文](https://arxiv.org/abs/2505.16381)）。
- [2025-06] 🎉 在 `fla` 中新增 MesaNet 实现（[论文](https://arxiv.org/abs/2506.05233)）。
- [2025-06] 🐍 在 `fla` 中新增 Comba 实现（[论文](https://arxiv.org/abs/2506.02475)）。
- [2025-05] 🎉 在 `fla` 中新增 Rodimus* 实现（[论文](https://arxiv.org/abs/2410.06577)）。
- [2025-04] 🎉 在 `fla` 中新增 DeltaProduct 实现（[论文](https://arxiv.org/abs/2502.10297)）。
- [2025-04] 🎉 在 `fla` 中新增 FoX 实现（[论文](https://arxiv.org/abs/2503.02130)）。
- [2025-03] ~~我们将默认的 `initializer_range` 修改为魔数 🐳 0.006~~ 现已将 `initializer_range` 回滚至默认值 0.02。在实际训练中，我们建议两者都进行尝试对比。
- [2025-02] 🐳 在 `fla` 中新增 NSA（Native Sparse Attention）实现。算子详情请见[此处](fla/ops/nsa)。
- [2025-01] 🔥 我们正在全面迁移至基于 `torchtitan` 的训练框架。详情请见 [flame](https://github.com/fla-org/flame) 仓库。
- [2025-01] 🦅 在 `fla` 中新增 RWKV7 实现（包含算子核心与模型）。
- [2024-12] 在 `fla-org` 组织下新增 `flash-bidirectional-attention`（[开源仓库](https://github.com/fla-org/flash-bidirectional-linear-attention)）。
- [2024-12] 🎉 在 `fla` 中新增 Gated DeltaNet 实现（[论文](https://arxiv.org/abs/2412.06464)）。
- [2024-12] 🚀 `fla` 现已正式支持处理变长输入序列（Variable-Length Inputs）的算子。
- [2024-11] 输入格式已从多头优先（head-first）切换为序列优先（seq-first）格式。
- [2024-11] 💥 `fla` 现提供了一种灵活训练混合模型（Hybrid Models）的机制。
- [2024-10] 🔥 隆重推出 `flame` —— 用于训练 `fla` 模型的极简且可扩展的训练框架。详情请见[此处](https://github.com/fla-org/flame)。
- [2024-09] `fla` 现已包含融合线性层与交叉熵层，大幅降低训练阶段的显存占用。
- [2024-09] 🎉 在 `fla` 中新增 GSA 实现（[论文](https://arxiv.org/abs/2409.07146)）。
- [2024-05] 🎉 在 `fla` 中新增 DeltaNet 实现（[论文](https://arxiv.org/abs/2102.11174)）。
- [2024-05] 💥 `fla` v0.1 发布：集成了多种次二次（subquadratic）复杂度算子、层与模型（RetNet/GLA/Mamba/HGRN/HGRN2/RWKV6 等，详见[支持模型](#支持模型)）。
- [2023-12] 💥 正式推出 `fla`，提供前沿线性注意力模型的高性能开源实现集合。

</details>

## 支持模型

| 年份 | 模型 | 论文 | 代码实现 |
| :--: | :---------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| 2022 | ABC | [ABC: Attention with Bounded-memory Control](https://arxiv.org/abs/2110.02488) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/abc.py) |
| 2023 | RetNet | [Retentive network: a successor to transformer for large language models](https://arxiv.org/abs/2307.08621) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py) |
| 2023 | HGRN | [Hierarchically Gated Recurrent Neural Network for Sequence Modeling](https://openreview.net/forum?id=P1TCHxJwLB) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn.py) |
| 2024 | GLA | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gla.py) |
| 2024 | Based | [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/based.py) |
| 2024 | Rebased | [Linear Transformers with Learnable Kernel Functions are Better In-Context Models](https://arxiv.org/abs/2402.10644) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rebased.py) |
| 2024 | DeltaNet | [Parallelizing Linear Transformers with Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py) |
| 2024 | HGRN2 | [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/abs/2404.07904) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn2.py) |
| 2024 | RWKV6 | [Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence](https://arxiv.org/abs/2404.05892) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rwkv6.py) |
| 2024 | LightNet | [You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet](https://arxiv.org/abs/2405.21022) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/lightnet.py) |
| 2024 | YOCO | [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/yoco) |
| 2024 | Mamba2 | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/mamba2) |
| 2024 | GSA | [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa) |
| 2024 | MLA | [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mla.py) |
| 2024 | Stick-Breaking Attention | [Scaling Stick-Breaking Attention: An Efficient Implementation and In-depth Study](https://arxiv.org/abs/2410.17980) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/stickbreaking_attn.py) |
| 2025 | Samba | [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/samba) |
| 2025 | Gated DeltaNet | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule) |
| 2025 | RWKV7 | [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/rwkv7) |
| 2025 | NSA | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/nsa) |
| 2025 | FoX | [Forgetting Transformer: Softmax Attention with a Forget Gate](https://arxiv.org/abs/2503.02130) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/forgetting_attn) |
| 2025 | DeltaProduct | [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltaproduct.py) |
| 2025 | Rodimus&ast; | [Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions](https://arxiv.org/abs/2410.06577) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rodimus.py) |
| 2025 | MesaNet | [MesaNet: Sequence Modeling by Locally Optimal Test-Time Training](https://arxiv.org/abs/2506.05233) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mesa_net.py) |
| 2025 | Comba | [Comba: Improving Bilinear RNNs with Closed-loop Control](https://arxiv.org/abs/2506.02475) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/comba.py) |
| 2025 | PaTH | [PaTH Attention: Position Encoding via Accumulating Householder Transformations](https://arxiv.org/abs/2505.16381) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/path_attn.py) |
| 2025 | MoM | [MoM: Linear Sequence Modeling with Mixture-of-Memories](https://arxiv.org/abs/2502.13685) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mom.py) |
| 2025 | Log-Linear Attention | [Log-Linear Attention](https://arxiv.org/abs/2506.04761) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/log_linear_attn) |
| 2025 | DeltaFormer | [Understanding Transformer from the Perspective of Associative Memory](https://arxiv.org/abs/2505.19488v1) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/deltaformer.py) |
| 2025 | KDA | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda) |
| 2025 | MoBA | [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/moba.py) |
| 2026 | Mamba3 | [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569) | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/mamba3) |
| 2026 | Raven | [Raven: High-Recall Sequence Modeling with Sparse Memory Routing](https://github.com/goombalab/raven/blob/main/raven.pdf) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/raven) |
| 2026 | Gated DeltaNet 2 | [Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/abs/2605.22791) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gdn2) |
| 2026 | Wall | [Wall Attention: Length Generalization With Diagonal Gates](https://blog.tilderesearch.com/blog/wall-attn) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/wall_attn) |
| 2026 | Parallax | [Parallax: Parameterized Local Linear Attention for Language Modeling](https://arxiv.org/abs/2605.29157) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/parallax) |
| 2026 | Preconditioned Gated DeltaNet | [Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences](https://arxiv.org/abs/2604.21100) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/precond_gated_delta_rule) |
| 2026 | Preconditioned KDA | [Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences](https://arxiv.org/abs/2604.21100) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/precond_kda) |
| 2026 | CAT | [Controllably Efficient Language Models](https://arxiv.org/abs/2511.05313) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/cat) |
| 2026 | CyFA | [CyFA: Linear Sequence Modeling with Relative-Time-Partitioned Memory](https://arxiv.org/abs/2609.36259) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/cyfa) |
| 2026 | MemoryAttention | [Memory Attention](https://arxiv.org/abs/2609.28399) | [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/memory_attn) |

## 安装指南

[![nvidia-h100-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-h100.yml/badge.svg?branch=main&event=push)](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-h100.yml)

`torch` 维护在独立的后端额外依赖中（`[cuda]` / `[rocm]` / `[xpu]` / `[npu]` / `[cpu]`）。CUDA 支持一条命令直接完成安装；其他后端通过两步命令进行安装，以确保 `torch`（以及 `torch` 传递依赖的具体 `triton` 变种版本）直接从 PyTorch wheel 官方源而非 PyPI 获取：

```sh
# CUDA
pip install flash-linear-attention[cuda]

# ROCm
pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch
pip install flash-linear-attention[rocm]
```

请参阅 [INSTALL.md](INSTALL.md) 获取完整的后端支持表、XPU / NPU（昇腾 Ascend）/ CPU 安装流程、源码编译安装说明，以及适配 `torch` 预发布版本 / `triton-nightly` 的 `--no-deps` 安装方案。

> [!NOTE]
> 请务必选择对应的后端额外依赖（extras）以安装适配的 `torch` 与 `triton` 依赖项。仅运行裸命令 `pip install flash-linear-attention` 将不会安装这些底层依赖。

## 使用方法

### 词元混合层 (Token Mixing)

我们在 `fla.layers` 中提供了多种“词元混合（token mixing）”线性注意力层供您直接调用。
您可以方便地将现有模型中的标准多头注意力层（Multi-Head Attention）替换为这些线性注意力层。
使用示例代码如下：

```py
>>> import torch
>>> from fla.layers import MultiScaleRetention
>>> batch_size, num_heads, seq_len, hidden_size = 32, 4, 2048, 1024
>>> device, dtype = 'cuda:0', torch.bfloat16
>>> retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
>>> x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
>>> y, *_ = retnet(x)
>>> y.shape
torch.Size([32, 2048, 1024])
```

我们还提供了与 🤗 Transformers 库全面兼容的模型实现。
以下示例展示了如何从 `fla` 默认配置初始化一个完整的 GLA 模型：

```py
>>> from fla.models import GLAConfig
>>> from transformers import AutoModelForCausalLM
>>> config = GLAConfig()
>>> model = AutoModelForCausalLM.from_config(config)
```

<details>
<summary>点击展开配置参数与模型结构详情</summary>

```py
>>> config
GLAConfig {
  "attn": null,
  "attn_mode": "chunk",
  "bos_token_id": 1,
  "clamp_min": null,
  "conv_size": 4,
  "elementwise_affine": true,
  "eos_token_id": 2,
  "expand_k": 0.5,
  "expand_v": 1,
  "feature_map": null,
  "fuse_cross_entropy": true,
  "fuse_norm": true,
  "fuse_swiglu": true,
  "hidden_act": "swish",
  "hidden_ratio": 4,
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": null,
  "max_position_embeddings": 2048,
  "model_type": "gla",
  "norm_eps": 1e-06,
  "num_heads": 4,
  "num_hidden_layers": 24,
  "num_kv_heads": null,
  "tie_word_embeddings": false,
  "transformers_version": "4.50.1",
  "use_cache": true,
  "use_gk": true,
  "use_gv": false,
  "use_output_gate": true,
  "use_short_conv": false,
  "vocab_size": 32000
}

>>> model
GLAForCausalLM(
  (model): GLAModel(
    (embeddings): Embedding(32000, 2048)
    (layers): ModuleList(
      (0-23): 24 x GLABlock(
        (attn_norm): RMSNorm(2048, eps=1e-06)
        (attn): GatedLinearAttention(
          (q_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (g_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (gk_proj): Sequential(
            (0): Linear(in_features=2048, out_features=16, bias=False)
            (1): Linear(in_features=16, out_features=1024, bias=True)
          )
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (g_norm_swish_gate): FusedRMSNormGated(512, eps=1e-06, activation=swish)
        )
        (mlp_norm): RMSNorm(2048, eps=1e-06)
        (mlp): GatedMLP(
          (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
          (swiglu_linear): SwiGLULinear()
        )
      )
    )
    (norm): RMSNorm(2048, eps=1e-06)
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
```

</details>

### 融合模块 (Fused Modules)

我们在 `fla.modules` 中提供了一系列融合模块，用以大幅提升训练吞吐：

* [`旋转位置编码 (Rotary Embedding)`](fla/modules/rotary)：Llama 架构（亦即 Transformer++）广泛采用的旋转位置编码。
* [`归一化层 (Norm Layers)`](fla/modules/norm/layernorm.py)：
  * `RMSNorm`、`LayerNorm` 与 `GroupNorm`。
  * `RMSNormLinear`、`LayerNormLinear` 与 `GroupNormLinear`：通过融合算子减少中间张量的显存实例化，显著提升显存利用率。
* [`带门控的归一化层 (Norm Layers with Gating)`](fla/modules/norm/fused_norm_gate.py)：将归一化层与逐元素的 sigmoid 或 swish 门控相融合，广泛应用于 RetNet 与 GLA。
* [`交叉熵损失 (Cross Entropy)`](fla/modules/fused_cross_entropy)：更高速度的 Triton 版交叉熵损失函数实现。
* [`融合线性交叉熵 (Linear Cross Entropy)`](fla/modules/fused_linear_cross_entropy)：融合线性投影层与交叉熵损失，避免显存实例化庞大的 logits 张量。亦可参考 [mgmalek](https://github.com/mgmalek/efficient_cross_entropy) 与 [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py) 的实现。
* [`融合线性 KL 散度 (Linear KL Divergence)`](fla/modules/fused_kl_div)：采用类似思路实现的融合线性层与 KL 散度损失。

> [!IMPORTANT]
> 您可以在模型配置中设置 `fuse_linear_cross_entropy` 来启用或禁用融合线性交叉熵损失。
>
> 该融合实现具有极高的显存效率，但在某些极端场景下可能会引入轻微的数值精度波动。由于这一权衡考量，该项默认处于禁用状态。
> 如果启用该特性并在训练过程中遇到不稳定性（如损失发散），建议先将其关闭以排查确认。

### 文本生成 (Generation)

预训练完成后，模型可直接借助 🤗 文本生成 API 执行高效推理。
以下给出文本生成的使用示例：

```py
>>> import fla
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> name = 'fla-hub/gla-1.3B-100B'
>>> tokenizer = AutoTokenizer.from_pretrained(name)
>>> model = AutoModelForCausalLM.from_pretrained(name).cuda()
>>> input_prompt = "Power goes with permanence. Impermanence is impotence. And rotation is castration."
>>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
>>> outputs = model.generate(input_ids, max_length=64)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

我们还在[此处](benchmarks/benchmark_generation.py)提供了一个简洁的基准测试脚本，用于测试模型的文本生成速度。
只需运行：

```sh
$ python -m benchmarks.benchmark_generation \
  --path 'fla-hub/gla-1.3B-100B' \
  --repetition_penalty 2. \
  --prompt="Hello everyone, I'm Songlin Yang"

Prompt:
Hello everyone, I'm Songlin Yang
Generated:
Hello everyone, I'm Songlin Yang.
I am a 20 year old girl from China who is currently studying in the United States of America for my Master degree and also working as an English teacher at school here on campus since last summer (1st semester). My main goal to be able do well with this course so that we can have

Prompt length: 10, generation length: 64
Total prompt processing + decoding time: 4593ms
```

目前所有可用的预训练模型检查点均可在 [`fla-hub`](https://huggingface.co/fla-hub) 中获取。

```py
>>> from huggingface_hub import list_models
>>> for model in list_models(author='fla-hub'): print(model.id)
```

### 混合模型 (Hybrid Models)

`fla` 提供了一种极具灵活性的方法，可将标准注意力层与现有的线性注意力模型融合构建混合架构。
只需在模型配置中通过 `attn` 参数进行指定即可轻松实现。
原始的字典传参形式会将统一的注意力配置应用到列表中声明的每一层。

例如，创建一个包含 2 层的 Samba 混合模型（第 0 层为 Mamba，第 1 层为滑动窗口大小为 2048 的局部注意力）：

```py
>>> from fla.models import SambaConfig
>>> from transformers import AutoModelForCausalLM
>>> config = SambaConfig(num_hidden_layers=2)
>>> config.attn = {
  'layers': [1],
  'num_heads': 18,
  'num_kv_heads': 18,
  'qkv_bias': False,
  'rope_theta': 10000.,
  'window_size': 2048
}
>>> model = AutoModelForCausalLM.from_config(config)
```

<details>
<summary>点击展开配置参数与模型结构详情</summary>

```py
>>> config
SambaConfig {
  "attn": {
    "layers": [
      1
    ],
    "num_heads": 18,
    "num_kv_heads": 18,
    "qkv_bias": false,
    "rope_theta": 10000.0,
    "window_size": 2048
  },
  "bos_token_id": 1,
  "conv_kernel": 4,
  "eos_token_id": 2,
  "expand": 2,
  "fuse_cross_entropy": true,
  "fuse_norm": true,
  "fuse_swiglu": true,
  "hidden_act": "swish",
  "hidden_ratio": 4,
  "hidden_size": 2304,
  "initializer_range": 0.02,
  "intermediate_size": 4608,
  "max_position_embeddings": 2048,
  "model_type": "samba",
  "norm_eps": 1e-05,
  "num_hidden_layers": 2,
  "pad_token_id": 0,
  "rescale_prenorm_residual": false,
  "residual_in_fp32": false,
  "state_size": 16,
  "tie_word_embeddings": false,
  "time_step_floor": 0.0001,
  "time_step_init_scheme": "random",
  "time_step_max": 0.1,
  "time_step_min": 0.001,
  "time_step_rank": 144,
  "time_step_scale": 1.0,
  "transformers_version": "4.50.1",
  "use_bias": false,
  "use_cache": true,
  "use_conv_bias": true,
  "vocab_size": 32000
}

>>> model
SambaForCausalLM(
  (backbone): SambaModel(
    (embeddings): Embedding(32000, 2304)
    (layers): ModuleList(
      (0): SambaBlock(
        (mixer_norm): RMSNorm(2304, eps=1e-05)
        (mixer): Mamba(
          (conv1d): Conv1d(4608, 4608, kernel_size=(4,), stride=(1,), padding=(3,), groups=4608)
          (in_proj): Linear(in_features=2304, out_features=9216, bias=False)
          (x_proj): Linear(in_features=4608, out_features=176, bias=False)
          (dt_proj): Linear(in_features=144, out_features=4608, bias=True)
          (out_proj): Linear(in_features=4608, out_features=2304, bias=False)
        )
        (mlp_norm): RMSNorm(2304, eps=1e-05)
        (mlp): GatedMLP(
          (gate_proj): Linear(in_features=2304, out_features=6144, bias=False)
          (up_proj): Linear(in_features=2304, out_features=6144, bias=False)
          (down_proj): Linear(in_features=6144, out_features=2304, bias=False)
          (swiglu_linear): SwiGLULinear()
        )
      )
      (1): SambaBlock(
        (mixer_norm): RMSNorm(2304, eps=1e-05)
        (mixer): Attention(
          (q_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (rotary): RotaryEmbedding(dim=128, base=10000.0, interleaved=False, pos_idx_in_fp32=True)
        )
        (mlp_norm): RMSNorm(2304, eps=1e-05)
        (mlp): GatedMLP(
          (gate_proj): Linear(in_features=2304, out_features=6144, bias=False)
          (up_proj): Linear(in_features=2304, out_features=6144, bias=False)
          (down_proj): Linear(in_features=6144, out_features=2304, bias=False)
          (swiglu_linear): SwiGLULinear()
        )
      )
    )
    (norm_f): RMSNorm(2304, eps=1e-05)
  )
  (lm_head): Linear(in_features=2304, out_features=32000, bias=False)
)
```

</details>

若要在不同网络深度应用差异化的注意力配置，只需传入配置列表即可。例如，以下这个 6 层 Samba 模型在第 1 层和第 3 层使用局部注意力，在第 5 层使用全注意力（Full Attention），而在第 0、2、4 层保留原生的 Mamba 混合器：

```py
>>> config = SambaConfig(
...   num_hidden_layers=6,
...   attn=[
...     {
...       'layers': [1, 3],
...       'num_heads': 18,
...       'num_kv_heads': 18,
...       'qkv_bias': False,
...       'rope_theta': 10000.,
...       'window_size': 2048,
...     },
...     {
...       'layers': [5],
...       'num_heads': 18,
...       'num_kv_heads': 18,
...       'qkv_bias': False,
...       'rope_theta': 10000.,
...       'window_size': None,
...     },
...   ],
... )
>>> model = AutoModelForCausalLM.from_config(config)
```

列表中的每一项配置均会独立标准化。在配置方案中未被声明的层，将继续保留模型原生的线性注意力、循环或状态空间混合器。

在模型推理与文本生成阶段，您**完全无需**进行任何额外调整！
模型能够直接进行自回归输出，无需任何额外的配置适配或代码修改。

## 模型训练

我们提供了构建在 `torchtitan` 之上的极简训练框架 [🔥 `flame`](https://github.com/fla-org/flame)，用于高效训练各类 `fla` 模型。

更多详细信息请参阅 [GLA 训练示例](https://github.com/fla-org/flash-linear-attention/blob/main/examples/training.md)。

## 模型评测

我们支持使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 通过命令 `python -m evals.harness` 运行基准评测，并通过 `python -m evals.ppl` 运行困惑度（Perplexity）评测。

请参阅[评测指南文档](evals/README.md)获取环境配置、单卡与多卡运行命令、RULER 长上下文评测基准以及困惑度评测的具体参数设置。

## 性能基准 (Benchmarks)

我们在多种张量形状配置下，将基于 Triton 的算子实现（`chunk_retention`、`chunk_gla`、`chunk_gdn`）与基于 CUDA 的 FlashAttention2 进行了详尽的速度对比测试。
测试均在单张 NVIDIA GB200 GPU 上完成（CUDA 12.9，PyTorch 2.9.0）。

```sh
# 运行前可能需要先通过 `pip install -e .` 安装 `fla` 以支持模块导入
$ python -m benchmarks.ops.run --op chunk_retention chunk_gla chunk_gdn flash_attn
=================================================================================
  Machine: NVIDIA GB200 | CUDA 12.9 | PyTorch 2.9.0+cu129.msh
=================================================================================
  fwd        B      T    H    D  op                            main[0a484709](ms)
          -----------------------------------------------------------------------
             1   8192   96  128  chunk_retention                            0.787
                                 chunk_gla                                  1.765
                                 chunk_gdn                                  1.265
                                 flash_attn                                 3.753
          -----------------------------------------------------------------------
             2  16384   16  128  chunk_retention                            0.792
                                 chunk_gla                                  1.445
                                 chunk_gdn                                  1.029
                                 flash_attn                                 5.035
          -----------------------------------------------------------------------
             4   2048   16  128  chunk_retention                            0.559
                                 chunk_gla                                  0.514
                                 chunk_gdn                                  0.753
                                 flash_attn                                 0.346
          -----------------------------------------------------------------------
             4   4096   64  128  chunk_retention                            0.997
                                 chunk_gla                                  2.251
                                 chunk_gdn                                  1.581
                                 flash_attn                                 2.560
          -----------------------------------------------------------------------
             8   1024    8   64  chunk_retention                            0.425
                                 chunk_gla                                  0.358
                                 chunk_gdn                                  0.631
                                 flash_attn                                 0.157
          -----------------------------------------------------------------------
             8   2048   32  256  chunk_retention                            1.174
                                 chunk_gla                                  2.897
                                 chunk_gdn                                  1.831
                                 flash_attn                                 1.408
=================================================================================
  fwdbwd     B      T    H    D  op                            main[0a484709](ms)
          -----------------------------------------------------------------------
             1   8192   96  128  chunk_retention                            2.618
                                 chunk_gla                                  7.670
                                 chunk_gdn                                  4.738
                                 flash_attn                                15.371
          -----------------------------------------------------------------------
             2  16384   16  128  chunk_retention                            2.122
                                 chunk_gla                                  5.984
                                 chunk_gdn                                  3.616
                                 flash_attn                                19.960
          -----------------------------------------------------------------------
             4   2048   16  128  chunk_retention                            1.047
                                 chunk_gla                                  1.434
                                 chunk_gdn                                  2.085
                                 flash_attn                                 0.902
          -----------------------------------------------------------------------
             4   4096   64  128  chunk_retention                            3.459
                                 chunk_gla                                 10.216
                                 chunk_gdn                                  5.964
                                 flash_attn                                10.815
          -----------------------------------------------------------------------
             8   1024    8   64  chunk_retention                            0.898
                                 chunk_gla                                  1.707
                                 chunk_gdn                                  1.974
                                 flash_attn                                 0.477
          -----------------------------------------------------------------------
             8   2048   32  256  chunk_retention                           51.103
                                 chunk_gla                                 13.797
                                 chunk_gdn                                  8.644
                                 flash_attn                                 6.748
=================================================================================
```

## 学术引用

如果您发现本项目对您的研究或开发有所助益，欢迎引用我们的工作：

```bib
@software{yang2024fla,
  title  = {FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism},
  author = {Yang, Songlin and Zhang, Yu},
  url    = {https://github.com/fla-org/flash-linear-attention},
  month  = jan,
  year   = {2024}
}

@misc{chen2026attnres,
  title         = {Attention Residuals},
  author        = {Chen, Guangyu  and Zhang, Yu  and Su, Jianlin  and Xu, Weixin  and Pan, Siyuan  and Wang, Yaoyu  and Wang, Yucheng  and Chen, Guanduo  and Yin, Bohong  and Chen, Yutian  and Yan, Junjie  and Wei, Ming  and Zhang, Y.  and Meng, Fanqing  and Hong, Chao  and Xie, Xiaotong  and Liu, Shaowei  and Lu, Enzhe  and Tai, Yunpeng  and Chen, Yanru  and Men, Xin  and Guo, Haiqing  and Charles, Y.  and Lu, Haoyu  and Sui, Lin  and Zhu, Jinguo  and Zhou, Zaida  and He, Weiran  and Huang, Weixiao  and Xu, Xinran  and Wang, Yuzhi  and Lai, Guokun  and Du, Yulun  and Wu, Yuxin  and Yang, Zhilin  and Zhou, Xinyu},
  year          = {2026},
  eprint        = {2603.15031},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CL}
}

@misc{zhang2025kda,
  title         = {Kimi Linear: An Expressive, Efficient Attention Architecture},
  author        = {Zhang, Yu  and Lin, Zongyu  and Yao, Xingcheng  and Hu, Jiaxi  and Meng, Fanqing  and Liu, Chengyin  and Men, Xin  and Yang, Songlin  and Li, Zhiyuan  and Li, Wentao  and Lu, Enzhe  and Liu, Weizhou  and Chen, Yanru  and Xu, Weixin  and Yu, Longhui  and Wang, Yejie  and Fan, Yu  and Zhong, Longguang  and Yuan, Enming  and Zhang, Dehao  and Zhang, Yizhi  and T. Liu, Y.  and Wang, Haiming  and Fang, Shengjun  and He, Weiran  and Liu, Shaowei  and Li, Yiwei  and Su, Jianlin  and Qiu, Jiezhong  and Pang, Bo  and Yan, Junjie  and Jiang, Zhejun  and Huang, Weixiao  and Yin, Bohong  and You, Jiacheng  and Wei, Chu  and Wang, Zhengtao  and Hong, Chao  and Chen, Yutian  and Chen, Guanduo  and Wang, Yucheng  and Zheng, Huabin  and Wang, Feng  and Liu, Yibo  and Dong, Mengnan  and Zhang, Zheng  and Pan, Siyuan  and Wu, Wenhao  and Wu, Yuhao  and Guan, Longyu  and Tao, Jiawen  and Fu, Guohong  and Xu, Xinran  and Wang, Yuzhi  and Lai, Guokun  and Wu, Yuxin  and Zhou, Xinyu  and Yang, Zhilin  and Du, Yulun},
  year          = {2025},
  eprint        = {2510.26692},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@inproceedings{yang2025path,
  title     = {PaTH Attention: Position Encoding via Accumulating Householder Transformations},
  author    = {Yang, Songlin  and Shen, Yikang and Wen, Kaiyue and Tan, Shawn  and Mishra, Mayank  and Ren, Liliang  and Panda, Rameswar  and Kim, Yoon},
  booktitle = {Proceedings of NeurIPS},
  year      = {2025}
}

@inproceedings{yang2024gdn,
  title     = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author    = {Yang, Songlin  and Kautz, Jan  and Hatamizadeh, Ali},
  booktitle = {Proceedings of ICLR},
  year      = {2025}
}

@inproceedings{yang2024deltanet,
  title     = {Parallelizing Linear Transformers with the Delta Rule over Sequence Length},
  author    = {Yang, Songlin and Wang, Bailin and Zhang, Yu and Shen, Yikang and Kim, Yoon},
  booktitle = {Proceedings of NeurIPS},
  year      = {2024}
}

@inproceedings{zhang2024gsa,
  title     = {Gated Slot Attention for Efficient Linear-Time Sequence Modeling},
  author    = {Zhang, Yu and Yang, Songlin and Zhu, Ruijie and Zhang, Yue and Cui, Leyang and Wang, Yiqiao and Wang, Bolun and Shi, Freda and Wang, Bailin and Bi, Wei and Zhou, Peng and Fu, Guohong},
  booktitle = {Proceedings of NeurIPS},
  year      = {2024}
}

@inproceedings{qin2024hgrn2,
  title     = {HGRN2: Gated Linear RNNs with State Expansion},
  author    = {Qin, Zhen and Yang, Songlin and Sun, Weixuan and Shen, Xuyang and Li, Dong and Sun, Weigao and Zhong, Yiran},
  booktitle = {Proceedings of COLM},
  year      = {2024}
}

@inproceedings{yang2024gla,
  title     = {Gated Linear Attention Transformers with Hardware-Efficient Training},
  author    = {Yang, Songlin and Wang, Bailin and Shen, Yikang and Panda, Rameswar and Kim, Yoon},
  booktitle = {Proceedings of ICML},
  year      = {2024}
}
```

## 致谢

衷心感谢 [Bitdeer（比特小鹿）](https://www.bitdeer.com/) 与 [Moonshot AI（月之暗面）](https://www.moonshot.ai/) 对本项目基础设施维护与算力支持的鼎力赞助。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月30日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
