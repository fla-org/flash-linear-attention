# Gluon attention

Set `FLA_ATTN_GLUON=1` to enable Gluon implementations of `parallel_attn` and `attn_decoding_one_step`, or `FLA_GLUON=1` to enable all available Gluon backends. The global switch overrides `FLA_ATTN_GLUON=0`; set both to `0` to disable Gluon attention. Both switches default to `0`. Set `FLA_USE_TMA=1` to allow the TMA paths. Benchmark both settings for the target workload: descriptor setup can outweigh the GPU savings on short or windowed calls. This backend requires Triton >= 3.5.1, NVIDIA compute capability 9.x or 10.x, and matching fp16 or bf16 Q/K/V. Other inputs use the existing dispatch fallback. These switches select supported calls without predicting performance; some feature-enabled and short-sequence workloads regress compared with Triton. The assembler must support the target architecture; use `TRITON_PTXAS_PATH` to select a compatible CUDA toolkit assembler when needed.

Both query/key and value dimensions can independently range from 1 through 512. Parallel attention supports causal attention, GQA/MQA, dense and packed variable-length input, optional chunk indices, sliding windows, forgetting gates, attention sinks, and their gradients. The public wrappers still handle contiguity, gate preprocessing, and argument validation. Single-step decoding is an inference-only API without backward support; it preserves empty-sequence and zero-window behavior.

```python
import os

os.environ['FLA_ATTN_GLUON'] = '1'
os.environ['FLA_USE_TMA'] = '1'

import torch

from fla.ops.attn import parallel_attn

q = torch.randn(1, 1024, 8, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, 1024, 2, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, 1024, 2, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
o = parallel_attn(q=q, k=k, v=v)
o.float().square().mean().backward()
```

The kernels explicitly manage shared-memory layouts, TMA buffering, and asynchronous matrix operations. Compute capability 9.x uses WGMMA and register accumulators; its small-dimension forward path keeps probabilities in registers and uses 64-row query tiles. On compute capability 10.x, aligned dimensions up to 128 use separate TMA, matrix, and softmax partitions. The next QK tile overlaps normalization of the current tile, and consumed score TMEM holds packed probabilities until PV completes. Both architecture paths skip elementwise causal masks on fully interior key tiles, retaining masks at the diagonal and sequence boundary and for sliding windows. Backward also omits masks for complete interior tiles while preserving partial-row masks and the deterministic gradient passes. Larger dimensions use smaller tiles and split forward value tiles; wide TMEM output tiles are rescaled in smaller register slices. For wide backward tiles on compute capability 10.x, score and probability-gradient intermediates temporarily reuse a gradient accumulator slice, with its fp32 contents saved in registers and restored before accumulation. This keeps dK and dV in one pass through the query sequence even at dimension 512. Compute capability 9.x retains separate dK/dV passes when the combined accumulators exceed its register budget. Unaligned dimensions or storage, and runs with `FLA_USE_TMA=0`, use masked loads. Immutable descriptor layouts are cached by tile geometry and dtype; tensor addresses and descriptors are rebuilt for each call. Decoding splits long KV sequences across CTAs, distributes warps across wide value dimensions to reduce shared-memory reduction traffic, overlaps copies with computation, and merges partial softmax statistics in fp32.

Tensor-core operands retain the input dtype and accumulate in fp32. Probability and score-gradient operands retain the same casts as the original attention kernel. The opt-in sink gradient uses the fp32 backward probability expectation instead of the rounded forward output; the motivation and numerical impact are described in [the design discussion](https://github.com/fla-org/flash-linear-attention/issues/1322#issuecomment-5971334761).

The implementation follows the synchronization and layout APIs in the public Triton [asynchronous copy](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html), [TMA](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html), [WGMMA](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html), and [tcgen05](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html) tutorials, and the [Gluon attention example](https://github.com/triton-lang/triton/blob/v3.5.1/python/examples/gluon/01-attention-forward.py).

Run correctness checks before collecting timings:

```bash
FLA_CI_ENV=0 FLA_TILELANG=0 FLA_ATTN_GLUON=1 FLA_USE_TMA=1 python -m pytest tests/ops/test_attn.py tests/ops/test_attn_gluon.py
FLA_CI_ENV=0 FLA_TILELANG=0 FLA_ATTN_GLUON=1 FLA_USE_TMA=1 python -m benchmarks.ops.verify --op parallel_attn
```

Use the unified runner to measure attention forward and forward plus backward. Run both backends on the same device and software, with backend dispatch enabled:

```bash
FLA_DISABLE_BACKEND_DISPATCH=0 FLA_TILELANG=0 FLA_USE_TMA=1 python -m benchmarks.ops.run \
    --op parallel_attn --backend triton --no-base --json attn-triton.json
FLA_DISABLE_BACKEND_DISPATCH=0 FLA_TILELANG=0 FLA_USE_TMA=1 python -m benchmarks.ops.run \
    --op parallel_attn --backend gluon --no-base --json attn-gluon.json
```

Use `--custom-shapes '{"large": {"B": 1, "T": 4096, "H": 8, "D": 512}}'` to measure a large head dimension. The original parallel-attention implementation does not support K > 256; unsupported or resource-limited baseline cases have no speedup ratio. Performance depends on the workload and architecture: long-sequence dense and packed attention improve in the measured configurations, while some short or feature-enabled cases still regress. Operator forward-plus-backward timings do not establish model training throughput.

For model training throughput, use `benchmarks/benchmark_training_throughput.py` with `--name forgetting_transformer`; its attention wrapper calls `parallel_attn`. Compare `FLA_ATTN_GLUON=0` and `1` with `FLA_GLUON=0` and `FLA_TILELANG=0`, keeping the model configuration, input seed, TMA setting, warmup, and measurement steps identical. For example, use `--batch_size 1 --seq_len 8192 --num_heads 16 --head_dim 128 --num_hidden_layers 12 --warmup_steps 16 --steps 64`, then repeat with `--varlen`, optionally adding `--context_len 64` for short packed sequences. The script includes forward, backward, optimizer steps, and input preparation, and reports tokens/s, ms/step, and peak memory. Packed boundaries vary between steps; the kernels keep the chunk count as a runtime argument so these batches reuse compiled code.

The same registry accepts `HQ`, `V`, `input_dtype`, `cu_seqlens`, `use_gate`, `use_sink`, and `window_size`. For example, append this shape configuration to either command above to compare packed fp16 GQA with unequal query/key and value dimensions:

```bash
--custom-shapes '{"packed": {"B": 1, "T": 4096, "H": 2, "HQ": 8, "D": 128, "V": 64, "input_dtype": "float16", "cu_seqlens": [0, 1000, 2048, 4096], "use_gate": true, "use_sink": true, "window_size": 257}}'
```
