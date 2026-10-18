# HC operators

Stateless primitives for iHC and mHC. Routing projections, sigmoid gates, normalization weights, and module state belong to the caller. These operators can underpin the residual modules proposed in [#1422](https://github.com/fla-org/flash-linear-attention/pull/1422), following [#1332](https://github.com/fla-org/flash-linear-attention/issues/1332).

## Contract

For streams `x[..., N, D]`, routing weights `pre[..., N]` and `post[..., N]`, branch output `branch[..., D]`, and optional mixing `mixing[..., N, N]`:

- `hc_pre(x, pre)` returns `sum_i pre_i * x_i` with shape `[..., D]`.
- `hc_post(x, branch, post, mixing=None)` returns `x_i + post_i * branch` for iHC. With mixing, it returns `sum_j mixing_ij * x_j + post_i * branch` for mHC.
- `sinkhorn(logits, num_iters=20)` alternates column and row log-sum-exp normalization, then exponentiates. The iteration count is positive and finite; exact double stochasticity is not promised after a fixed number of iterations.

Leading dimensions must match exactly; there is no broadcasting. Tokens are independent, so both dense batches and flattened packed sequences use the same entry points without sequence metadata. Empty leading dimensions are supported. Optimized paths initially target NVIDIA CUDA, 1–8 streams, and FP32/FP16/BF16 inputs; streams and routing weights may have different dtypes. Hidden dimensions need not be powers of two. Noncontiguous inputs are normalized by the entry point. The output of pre/post has the streams' dtype; Sinkhorn returns FP32. All tensor inputs support first-order gradients; optimized double backward is outside this contract.

The naive PyTorch implementation additionally supports CPU and FP64 for gradient checks. Products, reductions, and Sinkhorn normalization use FP32, or FP64 in the double-precision reference. The optimized implementation does not use TF32. Triton is the default backend; Gluon and TileLang are optional, explicitly enabled alternatives. Enable optional backends with `FLA_HC_GLUON=1` or `FLA_HC_TILELANG=1` (or the corresponding global `FLA_GLUON` / `FLA_TILELANG` switch). Gluon takes precedence when both are enabled. Missing optional dependencies fall back to Triton. Backward retains the backend selected by forward.

## Usage

```python
from fla.ops.hc import hc_pre, hc_post, sinkhorn

hidden = hc_pre(streams, pre)
branch = layer(hidden)
streams = hc_post(streams, branch, post)
```

For mHC, pass `mixing=sinkhorn(mixing_logits)` to the write operation. `pre` and `post` are the activated routing weights, not their logits. The caller may repeat these operations across layers; gradients propagate through the streams and routing weights.

The `naive_hc_pre`, `naive_hc_post`, and `naive_sinkhorn` exports provide eager PyTorch references and can be wrapped with `torch.compile(..., fullgraph=True)`.

## Validation and performance

The reference and numerical budgets are fixed before tuning: relative RMS error at most `5e-5` for FP32, `2e-3` for FP16, and `6e-3` for BF16 pre/post outputs and gradients, and `1e-4` for FP32 Sinkhorn outputs and gradients. Tests cover all inputs' gradients, mixed coefficient dtypes, noncontiguous inputs, empty and singleton shapes, irregular tails, extreme Sinkhorn logits, autocast, dispatch, and chained iHC/mHC updates. Strict local checks use `FLA_CI_ENV=0` with allocation poisoning enabled.

Benchmarks compare eager naive PyTorch, `torch.compile`, Triton, Gluon, and TileLang on the same GPU after correctness passes, reporting forward and forward-plus-backward latency. Sinkhorn backward stores the probabilities of each normalization step, costing `2 * num_iters * tokens * N * N * 4` bytes; inference does not retain this history.

## References

- [mHC paper](https://arxiv.org/abs/2512.24880), equations 3 and 7–9.
- [iHC author's description](https://x.com/unakar666/status/2027566637319393412): identity residual mixing with learned pre/post routing.
