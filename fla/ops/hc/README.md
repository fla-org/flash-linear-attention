# HC

For each token, let $X \in \mathbb{R}^{N \times D}$ contain $N$ residual streams and let $p, q \in \mathbb{R}^N$ be the pre/post routing weights. A layer reads the streams with `hc_pre` and writes its output back with `hc_post`:

```math
h = \sum_{i=1}^{N} p_i X_i, \qquad
b = f(h), \qquad
Y_i = \sum_{j=1}^{N} M_{ij} X_j + q_i b.
```

iHC uses identity mixing, $M=I$. mHC obtains $M$ by applying Sinkhorn normalization to a learned $N \times N$ matrix of logits. Routing projections and activation functions are computed by the caller.

`sinkhorn` alternates column and row normalization in log space. Starting from the input logits $A^{(0)}$, each iteration computes:

```math
B_{ij}^{(k)} = A_{ij}^{(k)} - \log\sum_r \exp A_{rj}^{(k)}, \qquad
A_{ij}^{(k+1)} = B_{ij}^{(k)} - \log\sum_s \exp B_{is}^{(k)}.
```

It returns $M=\exp A^{(K)}$ with `num_iters=20` by default. Finite iteration counts approximate a doubly stochastic matrix.

## Usage

```python
from fla.ops.hc import hc_post, hc_pre, sinkhorn

branch = layer(hc_pre(x=x, pre=pre))
y_ihc = hc_post(x=x, branch=branch, post=post)
y_mhc = hc_post(x=x, branch=branch, post=post, mixing=sinkhorn(logits=logits))
```

`x` has shape `[..., N, D]`, `pre` and `post` have shape `[..., N]`, and `logits` has shape `[..., N, N]`. Leading dimensions must match. The CUDA kernels support 1–8 streams, FP32/FP16/BF16 inputs, and first-order gradients. Pre/post accumulate in FP32 and return the dtype of `x`; Sinkhorn returns FP32.

## References

- [mHC](https://arxiv.org/abs/2512.24880), equations 3 and 7–9.
- [iHC author's description](https://x.com/unakar666/status/2027566637319393412).
