# Gated Oja Rule 2

Oja2 maintains a slot memory \(S_t \in \mathbb{R}^{K \times M}\), with an erase gate \(b_t \in \mathbb{R}^M\) and a write gate \(c_t \in \mathbb{R}^K\):

\[
S_t = S_{t-1}\operatorname{Diag}(\exp(g_t))
\left(I - (b_t \odot v_t)v_t^\top\right)
+ (c_t \odot k_t)v_t^\top,\qquad
o_t = \mathrm{scale}\,S_t^\top q_t.
\]

Here \(v_t\) is the slot code. The decay acts before the erase and write updates. Setting both gates to the same scalar recovers the gated Oja rule.

## Interfaces

`chunk_gated_oja_rule2` supports training with gradients for queries, keys, slot codes, decay, both gates, and the initial state. Queries and keys have shape `[B, T, H, K]`; slot codes, decay and erase gates have shape `[B, T, H, M]`; write gates have shape `[B, T, H, K]`. The slot dimension is at most 256. The chunk size can be 16, 32 or 64.

`fused_recurrent_gated_oja_rule2` supports forward-only inference with at most 128 slots. Use the chunk entry point when gradients are needed or the slot dimension exceeds this limit.

Both entry points return an output in the query dtype and an optional fp32 final state of shape `[N, H, K, M]`. Dense inputs have `N = B`. Packed inputs use `B = 1` and cumulative sequence lengths `cu_seqlens` of shape `[N + 1]`. Each packed sequence has its own initial and final state.

GSA2 supplies fp16 or bf16 queries, keys and gates alongside fp32 normalized slot codes and fp32 log-decay. The slot normalization is \(v / (\lVert v\rVert_2 + \epsilon)\); query/key normalization uses the existing FLA L2-normalization kernels. These normalization rules and the mixed operand dtypes are part of the numerical contract.

## Chunk computation

For chunk-local cumulative decay \(G_i\), form the strictly lower-triangular matrix

\[
L_{ij} = \mathbf{1}_{i>j}
\sum_m b_{i,m}v_{i,m}v_{j,m}\exp(G_{i,m}-G_{j,m}),
\qquad A=(I+L)^{-1}.
\]

The WY factors are \(W=A((b\odot v)\odot\exp(G))\) and \(U=A(c\odot k)\). The state and output stages reuse the gated Oja kernels; the score matrix and WY factors carry the channel-wise gates.

The score matrix, cumulative decay, and state accumulation use fp32. The solved inverse is stored in the key dtype, and each matrix product retains its existing operand conversions. The backward pass recomputes WY factors and chunk states. Masked rows must contribute zero before decay factors can introduce non-finite products.

## GSA2 integration

The GSA2 layer feeds the activated Oja2 readout and the slot code into a GDN-2 value memory. `level2_axis="head"` projects them to `head_dim` when needed; `"slot"` retains the slot dimension. Grouped value heads repeat the second-level queries and keys. Cache entries contain both recurrent states and the optional q/k/v/slot convolution states.

Correctness coverage is in `tests/ops/test_gated_oja_rule2.py`, `tests/layers/test_gsa2.py`, and `tests/models/test_modeling_gsa2.py`. The operator tests compare outputs, final states, and supported gradients with the differentiable PyTorch reference.
