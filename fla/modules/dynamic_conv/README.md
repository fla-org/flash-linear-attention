# Low-rank dynamic short convolution

`DynamicShortConvolution` implements the low-rank parameterization in [Dynamic Short Convolutions Improve Transformers](https://arxiv.org/abs/2606.03825).
For each token, the kernel is a learned static depthwise filter plus a linear combination of learned filter bases:

\[
w_{t,j,d} = b_{j,d} + \sum_r z_{t,r} U_{r,j,d}, \qquad
 y_{t,d} = \sum_{j=0}^{W-1} w_{t,j,d} x_{t-j,d}.
\]

Tap zero uses the current token. Missing history is zero. An optional channel bias is added before the output activation; an optional residual is added after it, matching `ShortConvolution`.
The Triton implementation expands latent coefficients inside the convolution kernels, without materializing `[B, T, W, D]` weights.

```python
import torch

from fla.modules import DynamicShortConvolution

conv = DynamicShortConvolution(
    hidden_size=value_dim,
    kernel_size=4,
    activation=None,
    device="cuda",
    dtype=torch.bfloat16,
    rank=16,
    generator_input_size=hidden_size,
    generator_activation=None,
)
v = v_proj(x)
v, state = conv(x=v, residual=v, generator_input=x, output_final_state=True)
v_next = v_proj(x_next)
v_next, state = conv(x=v_next, residual=v_next, generator_input=x_next, cache=state, output_final_state=True)
```

The constructor and forward arguments shared with `ShortConvolution` use the same names, order, and defaults.
`activation="silu"` is the default output activation, `bias=False` disables the channel bias, and `residual` is an optional forward tensor.
Only the Triton backend is supported. The generator adds keyword-only `rank`, `generator_input_size`, and `generator_activation` options, plus a keyword-only `generator_input` forward argument that defaults to `x`.
`generator_activation=None` gives a linear bottleneck; `"swish"` enables the nonlinear generator. Both activation arguments accept `"silu"` and `"swish"`.

The first generator projection has normal initialization, the kernel basis is zero-initialized, and the static filter is uniform in `[-1/sqrt(W), 1/sqrt(W)]`.
Thus initialization matches a static depthwise convolution with the configured channel bias and activation.
The module is independent of `ShortConvolution` and does not change existing model defaults or checkpoints.

## Functional contract

`dynamic_conv1d` accepts `x: [B,T,D]`, latent coefficients `z: [B,T,R]`, kernel bases `weight: [R,W,D]`, optional static kernel weights `kernel_bias: [W,D]`, optional channel `bias: [D]`, and an optional `residual: [B,T,D]`.
The functional API defaults to no output activation. Ranks from 1 through 128 and widths from 1 through 8 are supported, including non-power-of-two ranks and channel dimensions.
Floating tensors must share a CUDA/HIP device and use FP32, FP16, or BF16. Generator tensors may use different supported dtypes, including FP32 parameters under autocast; history must match the input dtype. Noncontiguous inputs are supported through contiguous copies at the autograd boundary.
Weight generation, convolution arithmetic, and backward reductions use FP32; outputs and gradients return in the corresponding input dtype.
With an output activation, backward retains the FP32 pre-activation and reuses the shared swish derivative.

Packed sequences use `B=1` and an int32/int64 `cu_seqlens` on the same device. Boundaries must start at zero, end at `T`, and be nondecreasing; empty segments are allowed.
Optional `chunk_indices` must be generated with `prepare_chunk_indices(cu_seqlens, 16)`.
A cache contains raw convolution inputs `[N,D,W]`, ordered oldest to newest. The functional API never modifies initial caches and returns newly allocated states.
The module follows `ShortConvolution` in updating a provided cache in place for single-token inference without autograd, including one-token-per-sequence packed inputs. Other calls preserve the initial cache.
An optional module `mask: [B,T]` zeros convolution and generator inputs; it cannot be combined with packed sequences.
Context parallelism and head-wise weight sharing are outside this implementation.

## Validation

```bash
FLA_CI_ENV=0 pytest tests/modules/test_dynamic_conv.py -q
python -m benchmarks.modules.benchmark_dynamic_conv
```

The independent materialized PyTorch reference computes in FP32 and covers outputs, all floating-input gradients, state gradients, empty/short segments, long sequence grids, noncontiguous inputs, mixed precision, module initialization, masks, and cached continuation.
