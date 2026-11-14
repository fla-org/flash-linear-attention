# Gated Delta Rule

## ReWY backward

The opt-in ReWY backend implements the residual WY pullback and Gram-free gate reduction from [ReWY: Faster Gradient Computation for Delta-Rule Attention Family](https://arxiv.org/abs/2611.01940). It replaces the input-gradient assembly after the reverse state scan; forward computation, saved tensors, state recomputation, and the state scan use the existing implementation.

Set `FLA_REWY=1` before running training to enable the backend. It accepts matching FP16 or BF16 Q/K/V inputs on NVIDIA GPUs, equal key and value head counts, value dimensions up to 128, and chunk sizes 16, 32, or 64. Dense and packed sequences and both state layouts share the same kernel. Context parallelism and other unsupported inputs use the existing backward path. The backend is disabled by default. A backend that replaces the entire public operator, such as FlashQLA, takes precedence; set `FLA_FLASH_QLA=0` to exercise this implementation.

For one chunk, let `g` be the base-2 cumulative gate used inside the kernels, `H` the incoming state, and `D_beta` and `D_e` diagonal matrices with entries `beta_i` and `2**g_i`. Define

```text
M_ij = beta_i * (k_i @ k_j.T) * 2**(g_i - g_j),  i > j;  0 otherwise
A = (I + M)^-1
R = A @ D_beta @ (V - D_e @ K @ H)
P = A.T @ dR
dM = -P @ R.T
```

The strict lower triangle of `dM` gives the residual pullback without differentiating the inverse explicitly. The beta gradient uses `P` and the residual expression directly, so it does not divide by beta and supports zero beta. Gate derivatives reuse the row and column contractions already needed for the key gradients instead of constructing another key Gram matrix.

The kernel accumulates dot products and reductions in FP32. The local FP32 `P` supplies the value gradient and its direct beta-gradient contribution before conversion for subsequent dot products. Existing state storage and gradient-output dtypes are preserved, including the two low-precision key-gradient stores before their final addition. The new residual coefficient uses the input dtype as a tensor-core operand. This algebraic reordering requires reference comparisons at the existing tolerances; it is not a claim of bitwise equality with the default backward.

The kernel keeps the full value tile live and loops over key tiles of width 32. The value-dimension limit bounds that live tile; dispatch does not select implementations by benchmark case or sequence length.
