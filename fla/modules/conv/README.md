# Gluon causal convolution

Set `FLA_CONV_GLUON=1` to select the optional Gluon implementation through the existing module backend dispatcher. The public `causal_conv1d` and `ShortConvolution` interfaces stay unchanged; use their default `backend='triton'` entry point. Gluon is disabled by default.

The kernel assigns adjacent channels to adjacent lanes and independent time groups to warps. Each thread retains a sliding window of input values, sharing them across eight forward outputs instead of reloading every convolution tap. Backward retains sixteen time steps per thread, reuses shifted output gradients, and fuses the convolution recomputation needed by SiLU. It keeps FP32 arithmetic and preserves the existing backward's intermediate rounding to the input dtype. Weight and bias gradients use FP32 partial reductions.

The optimized path supports NVIDIA GPUs with Gluon available, channel-contiguous `[B, T, D]` inputs (including QKV views), convolution widths 2/3/4, FP32/FP16/BF16 inputs, and dense or packed sequences. Forward supports initial state, final state, bias, residual, and SiLU. The final-state copy, backward calls requiring state gradients, decode updates, distributed processes, and other unsupported layouts or widths retain the existing implementation. Distributed processes use the existing backend uniformly across ranks.

Small inputs split each 64-token logical chunk into two 32-token blocks, preserving the caller's chunk indices. Larger inputs use 64-token blocks. This increases parallelism for small workloads without changing sequence boundaries.

```bash
FLA_CONV_GLUON=1 python -m pytest tests/modules/test_conv.py tests/modules/test_conv_gluon.py -q
python -m benchmarks.modules.benchmark_conv_gluon --no-bias --weight-dtype bfloat16 --json conv-gluon.json
```

The benchmark checks output and gradient parity before timing, alternates the two backends on identical inputs, and records repeated CUDA graph timings. It includes dense and packed sequences, the small-input routing boundary, and larger training shapes. `--quick` selects three representative shapes; `--dtype`, `--weight-dtype`, and `--bias` control the numerical and fusion settings.
