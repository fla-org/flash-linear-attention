# PHBA operator

Implementation of [PHBA: Prefix-State Hybrid Block Attention](https://arxiv.org/abs/2610.08527), by Ruijie Li, Jiaxi Hu, Shiyu Wang and Yuxuan Liang (2026).

PHBA combines causal local token attention, selected historical token blocks, and gated prefix-state slots in one joint softmax. Routing is discrete and is treated as constant in backward.

For gate logits `a_t` and positive normalizer `tau`, define `g_t = log_sigmoid(a_t) / tau` and `s_t = -expm1(g_t)`. Prefix states follow the recurrence

```text
S_{t+1} = diag(exp(g_t)) S_t + s_t [k_t, v_t]^T.
```

A block's prefix is the state before its first token. Token routes select up to `topk - 1` strictly earlier blocks from the same sequence. State routes contain the current block and token-route blocks, excluding the first prefix of each sequence. Causal local tokens, historical tokens and valid prefix slots share the final normalization.

## Storage and gradients

The CUDA implementation stores activations, insertion weights and prefix matrices in FP16/BF16 and uses FP32 accumulation. Prefix processing preserves its chunk-local cumulative-sum and dot-product order. The block-major backward assigns producers and owners deterministically, so route scatter order does not change floating-point accumulation order.

Both reference families live in `naive.py`. `naive.naive_phba` is the differentiable FP64 recurrent formula with frozen routes. `naive.reference_a_chunkwise` follows the staged storage conversions and supplies the storage-aware GPU comparison. Tests compare BF16/FP16 outputs and gradients against the storage-aware reference, and use the FP64 formula for gradcheck.

The operator returns `(output, final_state)`. When requested, final state is a pair of FP32 key/value states with no backward contract. Training supports first-order gradients for all six floating inputs. The internal attention backward is once-differentiable.

## Layouts and bounds

Inputs use `[B, T, H, D]` and gate logits use `[B, T, H, M]`. Packed inputs have `B == 1` and cumulative int32 lengths. A `SequenceLengthPlan` owns immutable geometry tensors and can be shared between layers; it never caches query, key, value or gradient data.

Global address offsets are widened to int64 before stride multiplication. Stored metadata remains int32; bounded local loop and tile indices retain int32 arithmetic, with automatic 64-bit fallback for unbounded workspace indices. Canonical sorting uses int32 keys only when the expert/query key range is safe, with int64 fallback otherwise. Invalid routes and metadata tails are preserved. Forward processing uses bounded query windows; deterministic backward uses the geometry plan saved by forward, with bounded grouped workspaces and a fixed-order fallback. Token metadata always supplies derived state metadata when historical routes exist; the no-history case retains direct state counting.

## Implementation layout

The public entry is `chunk.py`, with sequence geometry and workspace limits in `plans.py`. `parallel.py` owns joint attention forward and autograd dispatch; `prefix.py` owns gate algebra and the recurrent prefix computation. Internal responsibilities are grouped as follows:

| Module         | Responsibility                                                             |
| -------------- | -------------------------------------------------------------------------- |
| `routing.py`   | Block layout, centroids and discrete top-k routes                          |
| `metadata.py`  | Canonical token metadata, derived state metadata and no-history counting   |
| `chunk_bwd.py` | Attention backward, grouped tape kernels and deterministic query ownership |
| `utils.py`     | Input contracts, structured errors and gradient alignment                  |
| `naive.py`     | FP64 and storage-aware references, including routing and metadata oracles  |

`flash_attention.py` adapts the local-attention interface. `backends/cublaslt.py` owns projection autograd wrappers and optional acceleration, with native fallbacks. `backends/resources.py` carries the H20 C++ sources and screened algorithms in Python wheels. Internal module paths are implementation details; use the exports from `fla.ops.phba` for the public operator and sequence plan.

Production FlashAttention initialization checks dependencies, device support and the low-level parameter schema. Numerical ABI checks and layout/route assertions live in `tests/ops/test_phba.py`.

## Installation and model registration

Install FLA with CUDA support, then install `flash-attn>=2.8.3.post1,<2.9` for the joint local-attention backward interface.

```bash
pip install -e '.[cuda]'
pip install --no-build-isolation 'flash-attn>=2.8.3.post1,<2.9'
```

PHBA is registered by the shared `fla.models` exports. Import the configuration directly or use Transformers auto classes:

```python
import torch
from transformers import AutoModelForCausalLM
from fla.models import PHBAConfig

config = PHBAConfig(
    hidden_size=2048,
    num_hidden_layers=24,
    num_heads=16,
    num_kv_heads=16,
    intermediate_size=4272,
    max_position_embeddings=8192,
    moba_chunk_size=128,
    moba_topk=8,
    gsa_num_slots=128,
    qk_norm=False,
    use_output_gate=False,
)
model = AutoModelForCausalLM.from_config(config).cuda().train()
tokens = torch.randint(config.vocab_size, (1, 8192), device='cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = model(input_ids=tokens, labels=tokens).loss
loss.backward()
```

The public model classes are `PHBAConfig`, `PHBAModel`, and `PHBAForCausalLM` in `fla.models.phba`. The layer is `fla.layers.phba.PHBA`; the operator is `fla.ops.phba.chunk_phba`. `fla.models` exports all three classes; importing `fla` also registers PHBA with Transformers. Checkpoints retain the PHBA projection, normalization, MLP, embedding and language-head parameter names.

## Supported execution

The operator accepts CUDA FP16 or BF16 activation tensors and accumulates prefix and attention intermediates in FP32. Models can retain FP32 parameters under autocast. Query, key and value head counts must match; head dimensions are multiples of 8 from 16 through 128; prefix slot counts are 1–128. Supported chunk sizes are 32, 64, 128, 256, 512 and topk is 1–16, including the current block.

Dense inputs, packed variable-length inputs, nonempty padded sequences, first-order gradients, and non-reentrant gradient checkpointing are supported. `elementwise_affine=False` removes learnable scale parameters from all PHBA normalization layers. Packed operator inputs use `[1, total_tokens, heads, width]` and int32 cumulative sequence lengths. `SequenceLengthPlan.from_host` can supply known sequence lengths without a device-to-host length read. Dense `[B,T,...]` inputs with `B>1` require exactly `B` sequences of length `T`; packed inputs require `B=1`. When both a plan and `cu_seqlens` are supplied, their boundaries must agree. Passing `plan.cu_seqlens` itself, or a fresh, unmodified source used by `from_device`, avoids a device comparison. Other separate tensors require a value comparison and can synchronize. Model calls may pass `phba_length_plan` alone for packed sequences, and rotary positions use its boundaries. Padding masks and explicit packed `cu_seqlens` are mutually exclusive.

For language-model loss, mask padding and the first label of each packed or padded sequence with the criterion's ignore index (default `-100`) so shifted targets do not cross sequence boundaries. The model tests apply this caller-side label policy.

The model supports cache-free training, prefill and greedy generation with `use_cache=False`. `output_attentions=True`, persistent KV/prefix caching, an additional sliding window, and unequal query/key/value head counts are not implemented by this deterministic path. Final prefix states returned by the operator are FP32 and nondifferentiable. Context parallelism, tensor parallelism and full-graph compilation are outside the current validation scope.

## Optional H20 acceleration

PHBA projections retain the native linear path unless the exact-screened H20 geometry and runtime match. The accelerated path is restricted to PyTorch 2.9.0, CUDA 12.8, the validated cuBLASLt library, BF16 autocast and its three weight-gradient geometries. Actual GEMM operands must have 16-byte-aligned addresses; a contiguous view alone is insufficient. Unsupported alignment uses native weight-gradient operations without adding an alignment copy. The fused SwiGLU path checks its recomputed activation and incoming gradient and retains the native einsum fallback. The C++ helpers build from the Python-packaged source into the Torch extension cache; no prebuilt machine-specific binary is included. CUDA development headers and a C++ compiler are needed for this optional path. Other supported cases retain native PyTorch operations.

Training uses the optimizer selected by the training framework.

## Tests and benchmarks

```bash
FLA_CI_ENV=0 pytest tests/ops/test_phba.py tests/models/test_modeling_phba.py \
  tests/layers/test_phba_linear.py

python -m benchmarks.ops.run --op chunk_phba --no-base
```

The standard operator benchmark includes dense and packed shapes. Complete-step throughput and peak-memory results are recorded in the PR description.
