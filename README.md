<div align="center">

# 💥 Flash Linear Attention

[![hf_model](https://img.shields.io/badge/-Models-gray.svg?logo=huggingface&style=flat-square)](https://huggingface.co/fla-hub)  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/vDaJTmKNcS)

</div>

This repo aims at providing a collection of efficient Triton-based implementations for state-of-the-art linear attention models. **Any pull requests are welcome!**

<div align="center">
  <img width="400" alt="image" src="https://github.com/fla-org/flash-linear-attention/assets/18402347/02ff2e26-1495-4088-b701-e72cd65ac6cf">
</div>

* [News](#news)
* [Models](#models)
* [Installation](#installation)
  * [ARM (aarch64) Support for Triton](#arm-aarch64-support-for-triton)
* [Usage](#usage)
  * [Token Mixing](#token-mixing)
  * [Fused Modules](#fused-modules)
  * [Generation](#generation)
  * [Hybrid Models](#hybrid-models)
* [Training](#training)
* [Evaluation](#evaluation)
* [Benchmarks](#benchmarks)
* [Citation](#citation)
* [Star History](#star-history)
* [Acknowledgments](#acknowledgments)

## News

- **$\texttt{[2025-07]}$:** 🐳 Add MLA implementation to `fla` ([paper](https://arxiv.org/abs/2405.04434)).
- **$\texttt{[2025-07]}$:** 🛣️ Added PaTH Attention to fla ([paper](https://arxiv.org/abs/2505.16381)).
- **$\texttt{[2025-06]}$:** 🎉 Added MesaNet to fla ([paper](https://arxiv.org/abs/2506.05233)).
- **$\texttt{[2025-06]}$:** 🐍 Add Comba implementation to `fla` ([paper](https://arxiv.org/abs/2506.02475)).
- **$\texttt{[2025-05]}$:** 🎉 Add Rodimus&ast; implementation to `fla` ([paper](https://arxiv.org/abs/2410.06577)).
- **$\texttt{[2025-04]}$:** 🎉 Add DeltaProduct implementation to `fla` ([paper](https://arxiv.org/abs/2502.10297)).
- **$\texttt{[2025-04]}$:** 🎉 Add FoX implementation to `fla` ([paper](https://arxiv.org/abs/2503.02130)).
- **$\texttt{[2025-03]}$:** ~~We have changed the default `initializer_range` to the magic 🐳 0.006~~ The `initializer_range` was rolled back to the default value of 0.02. For actual training, we recommend trying both.
- **$\texttt{[2025-02]}$:** 🐳 Add NSA implementations to `fla`. See kernels [here](fla/ops/nsa).
- **$\texttt{[2025-01]}$:** 🔥 We are migrating to `torchtitan`-based training framework. Check out the [flame](https://github.com/fla-org/flame) repo for more details.
- **$\texttt{[2025-01]}$:** 🎉 Add RWKV7 implementations (both kernels and models) to `fla`.
- **$\texttt{[2024-12]}$:** Integrated `flash-bidirectional-attention` to `fla-org` ([repo](https://github.com/fla-org/flash-bidirectional-linear-attention))
- **$\texttt{[2024-12]}$:** 🎉 Add Gated DeltaNet implementation to `fla` ([paper](https://arxiv.org/abs/2412.06464)).
- **$\texttt{[2024-12]}$:** 🚀 `fla` now officially supports kernels with variable-length inputs.
- **$\texttt{[2024-11]}$:** The inputs are now switched from head-first to seq-first format.
- **$\texttt{[2024-11]}$:** 💥 `fla` now provides a flexible way for training hybrid models.
- **$\texttt{[2024-10]}$:** 🔥 Announcing `flame`, a minimal and scalable framework for training `fla` models. Check out the details [here](training/README.md).
- **$\texttt{[2024-09]}$:** `fla` now includes a fused linear and cross-entropy layer, significantly reducing memory usage during training.
- **$\texttt{[2024-09]}$:** 🎉 Add GSA implementation to `fla` ([paper](https://arxiv.org/abs/2409.07146)).
- **$\texttt{[2024-05]}$:** 🎉 Add DeltaNet implementation to `fla` ([paper](https://arxiv.org/abs/2102.11174)).
- **$\texttt{[2024-05]}$:** 💥 `fla` v0.1: a variety of subquadratic kernels/layers/models integrated (RetNet/GLA/Mamba/HGRN/HGRN2/RWKV6, etc., see [Models](#models)).
- **$\texttt{[2023-12]}$:** 💥 Launched `fla`, offering a collection of implementations for state-of-the-art linear attention models.

## Models

Roughly sorted according to the timeline supported in `fla`. The recommended training mode is `chunk` when available.

| Year | Venue   | Model          | Paper                                                                                                                                         | Code                                                                                            |                                                                                                       |
| :--- | :------ | :------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------: |
| 2023 |         | RetNet         | [Retentive network: a successor to transformer for large language models](https://arxiv.org/abs/2307.08621)                                   | [official](https://github.com/microsoft/torchscale/tree/main)                                   | [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py) |
| 2024 | ICML    | GLA            | [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)                                      | [official](https://github.com/berlino/gated_linear_attention)                                   |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gla.py)          |
| 2024 | ICML    | Based          | [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668)                            | [official](https://github.com/HazyResearch/based)                                               |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/based.py)         |
| 2024 | ACL     | Rebased        | [Linear Transformers with Learnable Kernel Functions are Better In-Context Models](https://arxiv.org/abs/2402.10644)                          | [official](https://github.com/corl-team/rebased/)                                               |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rebased.py)        |
| 2024 | NeurIPS | DeltaNet       | [Parallelizing Linear Transformers with Delta Rule  over Sequence Length](https://arxiv.org/abs/2406.06484)                                   | [official](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py) |      [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py)       |
| 2022 | ACL     | ABC            | [ABC: Attention with Bounded-memory Control](https://arxiv.org/abs/2110.02488)                                                                |                                                                                                 |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/abc.py)          |
| 2023 | NeurIPS | HGRN           | [Hierarchically Gated Recurrent Neural Network for Sequence Modeling](https://openreview.net/forum?id=P1TCHxJwLB)                             | [official](https://github.com/OpenNLPLab/HGRN)                                                  |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn.py)         |
| 2024 | COLM    | HGRN2          | [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/abs/2404.07904)                                                             | [official](https://github.com/OpenNLPLab/HGRN2)                                                 |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn2.py)         |
| 2024 | COLM    | RWKV6          | [Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence](https://arxiv.org/abs/2404.05892)                                    | [official](https://github.com/RWKV/RWKV-LM)                                                     |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rwkv6.py)         |
| 2024 |         | LightNet       | [You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet](https://arxiv.org/abs/2405.21022)                           | [official](https://github.com/OpenNLPLab/LightNet)                                              |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/lightnet.py)       |
| 2025 | ICLR    | Samba          | [Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling](https://arxiv.org/abs/2406.07522)                 | [official](https://github.com/microsoft/Samba)                                                  |          [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/samba)          |
| 2024 | ICML    | Mamba2         | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) | [official](https://github.com/state-spaces/mamba)                                               |         [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/mamba2)          |
| 2024 | NeurIPS | GSA            | [Gated Slot Attention for Efficient Linear-Time Sequence Modeling](https://arxiv.org/abs/2409.07146)                                          | [official](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)          |           [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)           |
| 2025 | ICLR    | Gated DeltaNet | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)                                                    | [official](https://github.com/NVlabs/GatedDeltaNet)                                             |      [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule)      |
| 2025 |         | RWKV7          | [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456)                                                    | [official](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)                                |           [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/rwkv7)            |
| 2025 |         | NSA            | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)                         |                                                                                                 |            [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/nsa)             |
| 2025 | ICLR    | FoX            | [Forgetting Transformer: Softmax Attention with a Forget Gate](https://arxiv.org/abs/2503.02130)                                              | [official](https://github.com/zhixuan-lin/forgetting-transformer)                               |      [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/forgetting_attn)       |
| 2025 |         | DeltaProduct   | [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297)                            |                                                                                                 |  [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/layers/gated_deltaproduct.py)  |
| 2025 | ICLR    | Rodimus&ast;   | [Rodimus*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions](https://arxiv.org/abs/2410.06577)                            | [official](https://github.com/codefuse-ai/rodimus)                                              |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rodimus.py)        |
| 2025 |         | MesaNet        | [MesaNet: Sequence Modeling by Locally Optimal Test-Time Training](https://arxiv.org/abs/2506.05233)                                          |                                                                                                 |       [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mesa_net.py)       |
| 2025 |         | Comba          | [Comba: Improving Bilinear RNNs with Closed-loop Control](https://arxiv.org/abs/2506.02475)                                                   | [official](https://github.com/AwesomeSeq/Comba-triton)                                          |        [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/comba.py)         |
| 2025 |         | PaTH           | [PaTH Attention: Position Encoding via Accumulating Householder Transformations](https://arxiv.org/abs/2505.16381)                            |                                                                                                 |      [fla](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/path_attn.py)       |

## Installation

[![nvidia-4090-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-4090.yml/badge.svg?branch=main&event=push)](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-4090.yml) [![nvidia-a100-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-a100.yml/badge.svg?branch=main)](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-a100.yml) [![nvidia-h100-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-h100.yml/badge.svg?branch=main&event=push)](https://github.com/fla-org/flash-linear-attention/actions/workflows/nvidia-h100.yml) [![intel-a770-ci](https://github.com/fla-org/flash-linear-attention/actions/workflows/intel-a770.yml/badge.svg?event=push)](https://github.com/fla-org/flash-linear-attention/actions/workflows/intel-a770.yml)

The following requirements should be satisfied
- [PyTorch](https://pytorch.org/) >= 2.5
- [Triton](https://github.com/openai/triton) >=3.0 (or nightly version, see [FAQs](FAQs.md))
- [einops](https://einops.rocks/)
- [transformers](https://github.com/huggingface/transformers) >=4.45.0
- [datasets](https://github.com/huggingface/datasets) >=3.3.0
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) >=1.4.0

You can install `fla` with pip:
```sh
pip install flash-linear-attention
```
As `fla` is actively developed now, for the latest features and updates, an alternative way is to install the package from source
```sh
# uninstall `fla` first to ensure a successful upgrade
pip uninstall flash-linear-attention && pip install -U git+https://github.com/fla-org/flash-linear-attention
```
or manage `fla` with submodules
```sh
git submodule add https://github.com/fla-org/flash-linear-attention.git 3rdparty/flash-linear-attention
ln -s 3rdparty/flash-linear-attention/fla fla
```

If you have installed `triton-nightly` and `torch` pre version, please use the following command:
```sh
pip install einops ninja datasets transformers numpy
pip uninstall flash-linear-attention && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention --no-deps
```

### ARM (aarch64) Support for Triton

You need to choose a specific version to install, see [FAQs](FAQs.md)

## Usage

### Token Mixing

We provide ``token mixing'' linear attention layers in `fla.layers` for you to use.
You can replace the standard multihead attention layer in your model with other linear attention layers.
Example usage is as follows:
```py
>>> import torch
>>> from fla.layers import MultiScaleRetention
>>> batch_size, num_heads, seq_len, hidden_size = 32, 4, 2048, 1024
>>> device, dtype = 'cuda:0', torch.bfloat16
>>> retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)
>>> retnet
MultiScaleRetention(
  (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
  (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
  (v_proj): Linear(in_features=1024, out_features=2048, bias=False)
  (g_proj): Linear(in_features=1024, out_features=2048, bias=False)
  (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
  (g_norm_swish_gate): FusedRMSNormGated(512, eps=1e-05, activation=swish)
  (rotary): RotaryEmbedding(dim=256, base=10000.0, interleaved=False, pos_idx_in_fp32=True)
)
>>> x = torch.randn(batch_size, seq_len, hidden_size).to(device=device, dtype=dtype)
>>> y, *_ = retnet(x)
>>> y.shape
torch.Size([32, 2048, 1024])
```

We provide the implementations of models that are compatible with 🤗 Transformers library.
Here's an example of how to initialize a GLA model from the default configs in `fla`:

```py
>>> from fla.models import GLAConfig
>>> from transformers import AutoModelForCausalLM
>>> config = GLAConfig()
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
  "initializer_range": 0.006,
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

>>> AutoModelForCausalLM.from_config(config)
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

### Fused Modules

We offer a collection of fused modules in `fla.modules` to facilitate faster training:

* [`Rotary Embedding`](fla/modules/rotary.py): rotary positional embeddings as adopted by the Llama architecture, a.k.a., Transformer++.
* [`Norm Layers`](fla/modules/layernorm.py):
  * `RMSNorm`, `LayerNorm` and `GroupNorm`
  * `RMSNormLinear`, `LayerNormLinear` and `GroupNormLinear` to reduce memory usage of intermediate tensors for improved memory efficiency.
* [`Norm Layers with Gating`](fla/modules/fused_norm_gate.py): combine norm layers with element-wise sigmoid or swish gating, as used by RetNet/GLA.
* [`Cross Entropy`](fla/modules/fused_cross_entropy.py): faster Triton implementation of cross entropy loss.
* [`Linear Cross Entropy`](fla/modules/fused_linear_cross_entropy.py): fused linear layer and cross entropy loss to avoid the materialization of large logits tensors. Also refer to implementations by [mgmalek](https://github.com/mgmalek/efficient_cross_entropy) and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py).
* [`Linear KL Divergence`](fla/modules/fused_kl_div.py): fused linear layer and KL divergence loss in a similar vein as CE loss.

### Generation

Upon successfully pretraining a model, it becomes accessible for generating text using the 🤗 text generation APIs.
In the following, we give a generation example:
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

We also provide a simple script [here](benchmarks/benchmark_generation.py) for benchmarking the generation speed.
Simply run it by:
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

All of the pretrained models currently available can be found in [`fla-hub`](https://huggingface.co/fla-hub).
```py
>>> from huggingface_hub import list_models
>>> for model in list_models(author='fla-hub'): print(model.id)
```

### Hybrid Models

`fla` provides a flexible method to incorporate standard attention layers into existing linear attention models.
This is easily achieved by specifying the `attn` argument in the model configuration.

For example, to create a 2-layer Samba model with interleaved Mamba and local attention layers, using a sliding window size of 2048:

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

>>> AutoModelForCausalLM.from_config(config)
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

During inference, you **DO NOT** need to revise anything for generation!
The model will produce output as-is, without any need for additional configurations or modifications.

## Training

We provide a minimal framework called [🔥 `flame`](https://github.com/fla-org/flame) built on top of `torchtitan`, for efficient training of `fla` models.

Checkout [the GLA example](https://github.com/fla-org/flash-linear-attention/blob/main/examples/training.md) for more details.

## Evaluation

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library allows you to easily perform (zero-shot) model evaluations.
Follow the steps below to use this library:

1. Install `lm_eval` following [their instructions](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md).

2. Run evaluation with:
```sh
$ MODEL='fla-hub/gla-1.3B-100B'
$ python -m evals.harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
```

We've made `fla` compatible with hf-style evaluations, you can call [evals.harness](evals/harness.py) to finish the evaluations.
Running the command above will provide the task results reported in the GLA paper.

3. Multi-GPU Evaluation with Hugging Face accelerate 🚀

To perform data-parallel evaluation (where each GPU loads a separate full copy of the model), we leverage the accelerate launcher as follows:
```sh
$ MODEL='fla-hub/gla-1.3B-100B'
$ accelerate launch -m evals.harness --model hf  \
    --model_args pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True  \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64  \
    --num_fewshot 0  \
    --device cuda  \
    --show_config  \
    --trust_remote_code
```

4. 📏 RULER Benchmark suite

The RULER benchmarks are commonly used for evaluating model performance on long-context tasks.
You can evaluate `fla` models on RULER directly using `lm-evaluation-harness`. RULER is only available in a relatively recent version of `lm-evaluation-harness`, so make sure you have the latest version installed.

```
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```


Then, install the necessary dependencies for RULER:
```sh
pip install lm_eval["ruler"]
```
and run evaluation by (e.g., 32k contexts):
```sh
$ accelerate launch -m evals.harness \
    --output_path $OUTPUT \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_hotpot,ruler_qa_squad \
    --model_args pretrained=$MODEL,dtype=bfloat16,max_length=32768,trust_remote_code=True \
    --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
    --batch_size 2 \
    --show_config  \
    --trust_remote_code
```

If a GPU can't load a full copy of the model, please refer to [this link](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate) for FSDP settings.

> [!Tip]
> If you are using `lm-evaluation-harness` as an external library and can't find (almost) any tasks available, before calling `lm_eval.evaluate()` or `lm_eval.simple_evaluate()`, simply run the following to load the library's stock tasks!
```py
>>> from lm_eval.tasks import TaskManager; TaskManager().initialize_tasks()
```

## Benchmarks

We compared our Triton-based RetNet implementation with CUDA-based FlashAttention2, using a batch size of 8, 32 heads, and a head dimension of 128, across different sequence lengths.
These tests were conducted on a single H100 80GB GPU, as illustrated in the following graph
```py
# you might have to first install `fla` to enable its import via `pip install -e .`
$ python benchmark_retention.py
Performance:
         T  chunk_fwd  parallel_fwd  flash_fwd  chunk_fwdbwd  parallel_fwdbwd  flash_fwdbwd
0    128.0   0.264032      0.243536   0.083488      1.301856         1.166784      0.320704
1    256.0   0.273472      0.252848   0.094304      1.345872         1.300608      0.807936
2    512.0   0.303600      0.278896   0.098112      1.503168         1.433184      0.857216
3   1024.0   0.357248      0.367360   0.156528      1.773552         2.303424      1.160864
4   2048.0   0.454624      0.605616   0.340928      2.283728         4.483360      1.955936
5   4096.0   0.638960      1.378016   1.004992      3.374720        12.271215      4.813776
6   8192.0   1.012352      4.201344   3.625008      5.581808        40.833618     15.023697
7  16384.0   1.748512     14.489664  13.710080     10.191552       153.093765     54.336864
```

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/c2607015-63af-43d1-90d1-ad5fe1670a03">
</div>


## Citation
If you find this repository helpful, please cite our work:
```bib
@software{yang2024fla,
  title  = {FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism},
  author = {Yang, Songlin and Zhang, Yu},
  url    = {https://github.com/fla-org/flash-linear-attention},
  month  = jan,
  year   = {2024}
}

@inproceedings{yang2024gdn,
  title     = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
  author    = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
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

## Star History

[![Stargazers repo roster for @fla-org/flash-linear-attention](https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=fla-org&repo=flash-linear-attention)](https://github.com/fla-org/flash-linear-attention/stargazers)


[![Star History Chart](https://api.star-history.com/svg?repos=fla-org/flash-linear-attention&type=Date)](https://star-history.com/#fla-org/flash-linear-attention&Date)

## Acknowledgments

We extend our gratitude to [Intel Corporation](https://www.intel.com/) and [Bitdeer](https://www.bitdeer.com/) for providing CI server resources that power our infrastructure.
