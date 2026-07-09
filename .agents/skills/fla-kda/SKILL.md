---
name: fla-kda
description: >
  FLA KDA kernel workflow and public technical notes. Use when modifying
  or reviewing fla/ops/kda/**, KDA gate modes, chunk intra/inter kernels,
  safe_gate behavior, KDA backends, or KDA-specific tests and benchmarks.
---

# FLA KDA Skill

Use this skill for KDA-specific work under `fla/ops/kda/**` and tests that
exercise KDA behavior.

## Public code map

- Public API: `fla.ops.kda.chunk_kda`, `fla.ops.kda.fused_recurrent_kda`.
- Gate helpers: `naive_kda_gate`, `naive_kda_lowerbound_gate`,
  `kda_gate_fwd`, `kda_gate_bwd`, `fused_kda_gate`,
  `kda_gate_chunk_cumsum` in `fla/ops/kda/gate.py`.
- Chunk forward: `chunk_kda_fwd` in `chunk_fwd.py`.
- Intra/inter forward: `chunk_kda_fwd_intra`,
  `chunk_kda_fwd_kernel_intra_sub_chunk`,
  `chunk_kda_fwd_kernel_inter_solve_fused` in `chunk_intra.py`.
- Token-parallel non-safe path: `chunk_kda_fwd_intra_token_parallel` in
  `chunk_intra_token_parallel.py`.
- WY recompute: `recompute_w_u_fwd` and `recompute_w_u_fwd_kda_kernel` in
  `wy_fast.py`.
- Backward: `chunk_kda_bwd`, `chunk_kda_bwd_intra`,
  `chunk_kda_bwd_wy_dqkg_fused`.
- Backends: `FlashKDABackend`, `KDATileLangBackend`.

## Gate modes

`chunk_kda` has two gate input contracts:

1. Pre-gated mode: `use_gate_in_kernel=False`.
   - `g` is already the log-space decay tensor.
   - `A_log`, `dt_bias`, and `lower_bound` are not part of the gate activation.
2. In-kernel mode: `use_gate_in_kernel=True`.
   - `g` is raw gate input.
   - `A_log` is required and `dt_bias` is optional.
   - Without `safe_gate`, activation is `-exp(A_log) * softplus(g + dt_bias)`.
   - With `safe_gate`, activation is
     `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`.

`safe_gate=True` requires `use_gate_in_kernel=True`, `lower_bound is not None`,
and `-5 <= lower_bound < 0`.

## Safe gate numerical note

With `lower_bound=-5`, every per-token gate value is in `[-5, 0)` before the
`RCP_LN2` conversion used by `chunk_kda_fwd`. A 16-token sub-chunk can therefore
accumulate `-80` in natural-log units. Directly feeding the full span to `exp2`
would be larger in base-2 units, so the safe intra path relies on offsetting.

`chunk_kda_fwd_kernel_intra_sub_chunk` uses a midpoint offset before
exponentiation:

- `b_gm = b_g - b_gn`;
- `exp2(b_gm)` and `exp2(-b_gm)`.

With the midpoint offset, each exponent operand covers at most about half of the
16-token sub-chunk. Under `lower_bound=-5`, this is about `40 / ln(2)`, which is
below the kernel's `exp2` safety comment threshold. The important invariant is
not the raw cumulative value alone; it is that each exponentiation uses a local
offset rather than the full chunk cumsum directly.

For inter-subchunk work, `chunk_kda_fwd_kernel_inter_solve_fused` computes decay
ratios with paired offsets such as:

- `exp2(b_g1 - b_gn1)` and `exp2(b_gn1 - b_g0)`;
- `exp2(b_g2 - b_gn2)` and `exp2(b_gn2 - b_g1)`.

Both terms are non-positive under monotonic accumulated decay, so the off-diagonal
inter path avoids positive exponent growth. The triangular solve operates on
masked lower-triangular blocks, so it does not introduce an unbounded exponent
path.

## Safe vs non-safe intra path

- Safe path: `chunk_kda_fwd_intra(..., safe_gate=True)` calls
  `chunk_kda_fwd_kernel_intra_sub_chunk` for 16-token diagonal blocks, then
  calls `chunk_kda_fwd_kernel_inter_solve_fused` with `USE_SAFE_GATE=True`.
- Non-safe path: `safe_gate=False` calls `chunk_kda_fwd_intra_token_parallel`
  for diagonal blocks, then calls the same inter/solve kernel with
  `USE_SAFE_GATE=False`.
- Do not change one path without checking the other path unless the contract is
  explicitly safe-only or non-safe-only.

## Correctness checklist

Before finishing a KDA behavior change, use `fla-correctness-coverage` and cover
only axes affected by the change:

- dense and varlen sequence layout;
- forward and backward if training path is touched;
- pre-gated, non-safe in-kernel, and safe in-kernel gate modes where supported;
- raw beta logits and post-sigmoid beta where supported;
- `use_qk_l2norm_in_kernel=True/False` where relevant;
- MHA and GVA (`HV > H`);
- `D != Dv` when value dimension is involved;
- initial/final state, `return_intermediate_states`, and CP paths when touched;
- backend verifier behavior for FlashKDA / TileLang changes.
- gate numerical extremes when gate math or intra/inter decay is touched:
  `lower_bound=-5`, a lower bound close to `0`, large positive and negative
  `g + dt_bias`, extreme `A_log`, long-sequence cumulative decay, chunk
  boundaries, and ragged varlen boundaries.

## Style constraints

- Use platform helpers from `fla.utils` (`device`, `device_platform`, `IS_NVIDIA`,
  `IS_NVIDIA_HOPPER`, `IS_NVIDIA_BLACKWELL`, `IS_AMD`, `IS_INTEL`) instead of
  adding new direct `torch.cuda` platform checks in tests or public code. If no
  helper covers the condition, add one in `fla.utils` first.
- Keep math derivations in operator docs or PR text; in Triton kernels, prefer
  compact shape comments and one-line rationale comments.
- Do not include internal-only paths, private model names, local machine paths,
  or private workload identifiers in public tests or skills.
