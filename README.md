<div align="center">

# :boom: Flash Linear Attention

[![hf_model](https://img.shields.io/badge/-Models-gray.svg?logo=huggingface&style=flat-square)](https://huggingface.co/fla-hub)  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/vDaJTmKNcS)

</div>

This repo aims at providing a collection of efficient Triton-based implementations for state-of-the-art linear attention models. **Any pull requests are welcome!** 

<div align="center">
  <img width="400" alt="image" src="https://github.com/fla-org/flash-linear-attention/assets/18402347/02ff2e26-1495-4088-b701-e72cd65ac6cf">
</div>

* [News](#news)
* [Models](#models)
* [Installation](#installation)
* [Usage](#usage)
  * [Token Mixing](#token-mixing)
  * [Fused Modules](#fused-modules)
  * [Generation](#generation)
  * [Hybrid Models](#hybrid-models)
* [Evaluations](#evaluations)
* [Benchmarks](#benchmarks)
* [Citation](#citation)
* [Star History](#star-history)

## News

- **$\texttt{[2025-01]}$:** :fire: We are migrating to `torchtitan`-based training framework. Check out the [flame](https://github.com/fla-org/flame) repo for more details.
- **$\texttt{[2025-01]}$:** :tada: Add RWKV7 implementations (both kernels and models) to `fla`.
- **$\texttt{[2024-12]}$:** Integrated `flash-bidirectional-attention` to `fla-org` ([repo](https://github.com/fla-org/flash-bidirectional-linear-attention)) 
- **$\texttt{[2024-12]}$:** :tada: Add Gated DeltaNet implementation to `fla` ([paper](https://arxiv.org/abs/2412.06464)).
- **$\texttt{[2024-12]}$:** :rocket: `fla` now officially supports kernels with variable-length inputs.
- **$\texttt{[2024-11]}$:** The inputs are now switched from head-first to seq-first format.
- **$\texttt{[2024-11]}$:** :boom: `fla` now provides a flexible way for training hybrid models.
- **$\texttt{[2024-10]}$:** :fire: Announcing `flame`, a minimal and scalable framework for training `fla` models. Check out the details [here](training/README.md).
- **$\texttt{[2024-09]}$:** `fla` now includes a fused linear and cross-entropy layer, significantly reducing memory usage during training.
- **$\texttt{[2024-09]}$:** :tada: Add GSA implementation to `fla` ([paper](https://arxiv.org/abs/2409.07146)).
- **$\texttt{[2024-05]}$:** :tada: Add DeltaNet implementation to `fla` ([paper](https://arxiv.org/abs/2102.11174)).
- **$\texttt{[2024-05]}$:** :boom: `fla` v0.1: a variety of subquadratic kernels/layers/models integrated (RetNet/GLA/Mamba/HGRN/HGRN2/RWKV6, etc., see [Models](#models)).
- **$\texttt{[2023-12]}$:** :boom: Launched `fla`, offering a collection of implementations for state-of-the-art linear attention models.

## Models

Roughly sorted according to the timeline supported in `fla`. The recommended training mode is `chunk` when available.

| Year | Venue   | Model          | Title                                                                                                     |                       Paper                        |                                              Code                                               |                                               `fla` impl                                               |
| :--- | :------ | :------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------------------------------: | :---------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| 2023 |         | RetNet         | Retentive network: a successor to transformer for large language models                                   |      [link](https://arxiv.org/abs/2307.08621)      |                  [official](https://github.com/microsoft/torchscale/tree/main)                  | [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/multiscale_retention.py) |
| 2024 | ICML    | GLA            | Gated Linear Attention Transformers with Hardware-Efficient Training                                      |      [link](https://arxiv.org/abs/2312.06635)      |                  [official](https://github.com/berlino/gated_linear_attention)                  |         [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gla.py)          |
| 2024 | ICML    | Based          | Simple linear attention language models balance the recall-throughput tradeoff                            |      [link](https://arxiv.org/abs/2402.18668)      |                        [official](https://github.com/HazyResearch/based)                        |        [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/based.py)         |
| 2024 | ACL     | Rebased        | Linear Transformers with Learnable Kernel Functions are Better In-Context Models                          |      [link](https://arxiv.org/abs/2402.10644)      |                        [official](https://github.com/corl-team/rebased/)                        |       [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rebased.py)        |
| 2024 | NeurIPS | DeltaNet       | Parallelizing Linear Transformers with Delta Rule  over Sequence Length                                   |      [link](https://arxiv.org/abs/2406.06484)      | [official](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py) |      [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py)       |
| 2022 | ACL     | ABC            | Attention with Bounded-memory Control                                                                     |      [link](https://arxiv.org/abs/2110.02488)      |                                                                                                 |         [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/abc.py)          |
| 2023 | NeurIPS | HGRN           | Hierarchically Gated Recurrent Neural Network for Sequence Modeling                                       | [link](https://openreview.net/forum?id=P1TCHxJwLB) |                         [official](https://github.com/OpenNLPLab/HGRN)                          |         [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn.py)         |
| 2024 | COLM    | HGRN2          | HGRN2: Gated Linear RNNs with State Expansion                                                             |      [link](https://arxiv.org/abs/2404.07904)      |                         [official](https://github.com/OpenNLPLab/HGRN2)                         |        [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/hgrn2.py)         |
| 2024 | COLM    | RWKV6          | Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence                                    |      [link](https://arxiv.org/abs/2404.05892)      |                           [official](https://github.com/RWKV/RWKV-LM)                           |        [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rwkv6.py)         |
| 2024 |         | LightNet       | You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightNet                           |      [link](https://arxiv.org/abs/2405.21022)      |                       [official](https://github.com/OpenNLPLab/LightNet)                        |       [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/lightnet.py)       |
| 2025 | ICLR    | Samba          | Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling                 |      [link](https://arxiv.org/abs/2406.07522)      |                         [official](https://github.com/microsoft/Samba)                          |          [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/samba)          |
| 2024 | ICML    | Mamba2         | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality |      [link](https://arxiv.org/abs/2405.21060)      |                        [official](https://github.com/state-spaces/mamba)                        |         [code](https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/mamba2)          |
| 2024 | NeurIPS | GSA            | Gated Slot Attention for Efficient Linear-Time Sequence Modeling                                          |      [link](https://arxiv.org/abs/2409.07146)      |     [official](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)      |           [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gsa)           |
| 2025 | ICLR    | Gated DeltaNet | Gated Delta Networks: Improving Mamba2 with Delta Rule                                                    |      [link](https://arxiv.org/abs/2412.06464)      |                       [official](https://github.com/NVlabs/GatedDeltaNet)                       |     [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gated_deltanet)      |
| 2025 |         | RWKV7          |                                                                                                           |                                                    |                [official](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)                 |          [code](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/rwkv7)          |
 
## Installation

The following requirements should be satisfied 
- [PyTorch](https://pytorch.org/) >= 2.3
- [Triton](https://github.com/openai/triton) >=3.0 (or nightly version, see [FAQs](FAQs.md))
- [einops](https://einops.rocks/)
- [transformers](https://github.com/huggingface/transformers) >=4.45.0
- [datasets](https://github.com/huggingface/datasets) >=2.11.0
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) >=1.4.0

As `fla` is actively developed now, no released packages are provided at this time.
If you do need to use `fla` ops/modules and contemplate further explorations, an alternative way is to install the package from source
```sh
# uninstall `fla` first to ensure a successful upgrade
pip uninstall fla && pip install -U git+https://github.com/fla-org/flash-linear-attention
```
or manage `fla` with submodules
```sh
git submodule add https://github.com/fla-org/flash-linear-attention.git 3rdparty/flash-linear-attention
ln -s 3rdparty/flash-linear-attention/fla fla
```

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
  (g_norm_swish_gate): FusedRMSNormSwishGate(512, eps=1e-05)
  (rotary): RotaryEmbedding()
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
  "transformers_version": "4.45.0",
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
          (g_norm_swish_gate): FusedRMSNormSwishGate(512, eps=1e-06)
        )
        (mlp_norm): RMSNorm(2048, eps=1e-06)
        (mlp): GLAMLP(
          (gate_proj): Linear(in_features=2048, out_features=11264, bias=False)
          (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
          (act_fn): SiLU()
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
* [`Norm Layers with Gating`](fla/modules/fused_norm_gate.py): combine norm layers with element-wise gating, as used by RetNet/GLA.
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
    "window_size": 2048
  },
  "bos_token_id": 1,
  "conv_kernel": 4,
  "eos_token_id": 2,
  "expand": 2,
  "fuse_cross_entropy": true,
  "fuse_norm": true,
  "hidden_act": "silu",
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
  "transformers_version": "4.45.0",
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
        (mixer): MambaMixer(
          (conv1d): Conv1d(4608, 4608, kernel_size=(4,), stride=(1,), padding=(3,), groups=4608)
          (act): SiLU()
          (in_proj): Linear(in_features=2304, out_features=9216, bias=False)
          (x_proj): Linear(in_features=4608, out_features=176, bias=False)
          (dt_proj): Linear(in_features=144, out_features=4608, bias=True)
          (out_proj): Linear(in_features=4608, out_features=2304, bias=False)
        )
        (mlp_norm): RMSNorm(2304, eps=1e-05)
        (mlp): SambaMLP(
          (gate_proj): Linear(in_features=2304, out_features=12288, bias=False)
          (down_proj): Linear(in_features=6144, out_features=2304, bias=False)
          (act_fn): SiLU()
        )
      )
      (1): SambaBlock(
        (mixer_norm): RMSNorm(2304, eps=1e-05)
        (mixer): Attention(
          (q_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2304, bias=False)
          (rotary): RotaryEmbedding()
        )
        (mlp_norm): RMSNorm(2304, eps=1e-05)
        (mlp): SambaMLP(
          (gate_proj): Linear(in_features=2304, out_features=12288, bias=False)
          (down_proj): Linear(in_features=6144, out_features=2304, bias=False)
          (act_fn): SiLU()
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

## Evaluations

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library allows you to easily perform (zero-shot) model evaluations. 
Follow the steps below to use this library:

1. Install `lm_eval` following [their instructions](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md). 

2. Run evaluation with:
```sh
$ PATH='fla-hub/gla-1.3B-100B'
$ python -m evals.harness --model hf \
    --model_args pretrained=$PATH,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config                  
```

We've made `fla` compatible with hf-style evaluations, you can call [evals.harness](evals/harness.py) to finish the evaluations.
Running the command above will provide the task results reported in the GLA paper.

> [!Tip]
> If you are using `lm-evaluation-harness` as an external library and can't find (almost) any tasks available, before calling `lm_eval.evaluate()` or `lm_eval.simple_evaluate()`, simply run the following to load the library's stock tasks!
```py
>>> from lm_eval.tasks import TaskManager; TaskManager().initialize_tasks()
```

## Benchmarks

We compared our Triton-based RetNet implementation with CUDA-based FlashAttention2, using a batch size of 8, 32 heads, and a head dimension of 128, across different sequence lengths. 
These tests were conducted on a single A100 80GB GPU, as illustrated in the following graph
```py
# you might have to first install `fla` to enable its import via `pip install -e .`
$ python benchmark_retention.py
Performance:
   seq_len  fused_chunk_fwd  chunk_fwd  parallel_fwd  fused_chunk_fwdbwd  chunk_fwdbwd  parallel_fwdbwd  flash_fwd  flash_fwdbwd
0    128.0         0.093184   0.185344      0.067584            1.009664      1.591296         1.044480   0.041984      0.282624
1    256.0         0.165888   0.219136      0.126976            1.024000      1.596928         1.073152   0.074752      0.413696
2    512.0         0.308224   0.397312      0.265216            1.550336      1.603584         1.301504   0.156672      0.883712
3   1024.0         0.603136   0.747520      0.706560            3.044864      3.089408         3.529728   0.467968      2.342912
4   2048.0         1.191424   1.403904      2.141184            6.010880      6.059008        11.009024   1.612800      7.135232
5   4096.0         2.377728   2.755072      7.392256           11.932672     11.938816        37.792770   5.997568     24.435200
6   8192.0         4.750336   5.491712     26.402817           23.759359     23.952385       141.014023  22.682114     90.619904
7  16384.0         9.591296  10.870784    101.262337           47.666176     48.745472       539.853821  91.346947    346.318848
```

![Performance](https://github.com/fla-org/flash-linear-attention/assets/30831390/36961182-da39-48ba-96a6-84c572ce51d7)


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

@misc{yang2024gated,
    title         = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
    author        = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
    year          = {2024},
    eprint        = {2412.06464},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL}
}

@inproceedings{yang2024parallelizing,
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

