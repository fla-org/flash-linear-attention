# Gluon attention

Set `FLA_ATTN_GLUON=1` and `FLA_USE_TMA=1` to opt in to the Gluon implementations of `parallel_attn` and `attn_decoding_one_step`. The default backend is unchanged. This backend requires Triton >= 3.5.1, NVIDIA compute capability 9.x or 10.x, and matching fp16 or bf16 Q/K/V. Other inputs use the existing dispatch fallback. The assembler must support the target architecture; use `TRITON_PTXAS_PATH` to select a compatible CUDA toolkit assembler when needed.

Both query/key and value dimensions can independently range from 1 through 512. The backend supports causal attention, GQA/MQA, dense and packed variable-length input, optional chunk indices, sliding windows, forgetting gates, attention sinks, and their gradients. The public wrappers still handle contiguity, gate preprocessing, and argument validation. Decoding preserves empty-sequence and zero-window behavior.

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

The kernels explicitly manage shared-memory layouts, TMA buffering, and asynchronous matrix operations. Compute capability 9.x uses WGMMA and register accumulators; 10.x uses tcgen05 and TMEM accumulators. Large dimensions use smaller tiles, split forward value tiles, and separate dK/dV passes when necessary to keep live accumulators within the memory budget. Unaligned dimensions or storage, and runs with `FLA_USE_TMA=0`, use masked loads. Forward tiling accounts for available CTA parallelism; wide TMEM output tiles are rescaled in smaller register slices. Decoding splits long KV sequences across CTAs, distributes warps across wide value dimensions to reduce shared-memory reduction traffic, overlaps copies with computation, and merges partial softmax statistics in fp32.

Tensor-core operands retain the input dtype and accumulate in fp32. Probability and score-gradient operands retain the same casts as the original attention kernel. The opt-in sink gradient uses the fp32 backward probability expectation instead of the rounded forward output; the motivation and numerical impact are described in [the design discussion](https://github.com/fla-org/flash-linear-attention/issues/1322#issuecomment-5971334761).

The implementation follows the synchronization and layout APIs in the public Triton [asynchronous copy](https://triton-lang.org/main/getting-started/tutorials/gluon/03-async-copy.html), [TMA](https://triton-lang.org/main/getting-started/tutorials/gluon/04-tma.html), [WGMMA](https://triton-lang.org/main/getting-started/tutorials/gluon/05-wgmma.html), and [tcgen05](https://triton-lang.org/main/getting-started/tutorials/gluon/06-tcgen05.html) tutorials, and the [Gluon attention example](https://github.com/triton-lang/triton/blob/v3.5.1/python/examples/gluon/01-attention-forward.py).

Run correctness checks before collecting timings:

```bash
FLA_TILELANG=0 FLA_ATTN_GLUON=1 FLA_USE_TMA=1 python -m pytest tests/ops/test_attn.py tests/ops/test_attn_gluon.py
FLA_TILELANG=0 FLA_ATTN_GLUON=1 FLA_USE_TMA=1 python -m benchmarks.ops.verify --op parallel_attn
```

Use the unified runner to measure dense attention forward and forward plus backward. Run both backends on the same device and software, with backend dispatch enabled:

```bash
FLA_DISABLE_BACKEND_DISPATCH=0 FLA_TILELANG=0 FLA_USE_TMA=1 python -m benchmarks.ops.run \
    --op parallel_attn --backend triton --no-base --json attn-triton.json
FLA_DISABLE_BACKEND_DISPATCH=0 FLA_TILELANG=0 FLA_USE_TMA=1 python -m benchmarks.ops.run \
    --op parallel_attn --backend gluon --no-base --json attn-gluon.json
```

Use `--custom-shapes '{"large": {"B": 1, "T": 4096, "H": 8, "D": 512}}'` to measure a large head dimension. The original parallel-attention implementation does not support K > 256; unsupported or resource-limited baseline cases have no speedup ratio. The backend remains experimental: current measurements show forward-only regressions, and long-sequence dense training gains are limited.
