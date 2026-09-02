# GDN Varlen CUDA Graph Walkthrough

## Scope and baseline

- Baseline commit: `35dceaee5408e69a555fec34cb215c93c375dabe`
- Working branch: `feat/gdn-varlen-cudagraph`
- In scope: `chunk_gated_delta_rule`, varlen layout, native Triton/CUDA, forward, backward, capture, and replay.
- Out of scope: KDA, causal convolution, fused recurrent decode, graph bucketing, scheduling frameworks, NPU, and Context Parallel.
- Mathematical contract: preserve the eager GDN operation, precision staging, accumulation order, output structure, and existing tolerances. Only scheduling metadata and boundary guards may change.

## Contract cells

| Source | Request | Chunk geometry | Gate/state axes | Disposition | Oracle and acceptance |
| ------ | ------- | -------------- | --------------- | ----------- | --------------------- |
| Existing public API and GDN tests | `use_graph=False`, dense or varlen, existing backends | `BT in {16, 32, 64}` | All previously supported combinations | Existing fallback | Existing tests and naive recurrent oracle; no route, output, or gradient change |
| New graph mode | `use_graph=True`, varlen, native Triton on NVIDIA CUDA | `BT in {16, 32, 64}`, fixed `T_max` and `N_max` | Raw or precomputed gate/beta, optional L2 norm, GVA, initial/final state | Optimized path | Same-call eager GDN on the real token prefix; existing per-output and per-gradient tolerances |
| New graph mode boundary | `use_graph=True`, dense input | Any | Any | Explicit unsupported error | Deterministic validation test before kernel execution |
| New graph mode boundary | `use_graph=True`, Context Parallel | Any | Any | Explicit unsupported error | Deterministic validation test before kernel execution |
| New graph mode boundary | `use_graph=True`, non-NVIDIA backend | Any | Any | Explicit unsupported error | Deterministic validation test before kernel execution |
| FlashQLA dispatch | `use_graph=True`, otherwise FlashQLA-compatible | `BT=64` | FlashQLA-supported subset | Existing fallback | FlashQLA verifier rejects graph mode; native Triton graph path supplies public semantics |

The graph capacity is defined by the physical input shape and the padded cumulative-length buffer:

```text
T_max = q.shape[1]
N_max = cu_seqlens.shape[0] - 1
NT_max = ceil_div(T_max, BT) + N_max - 1
```

The caller pads unused sequence entries by repeating `actual_t`. For example, two sequences with lengths `[100, 70]` at `N_max=4` use `cu_seqlens=[0, 100, 170, 170, 170]`.

## Numerical budget

No leaf stage may change its load, operand, accumulation, or store dtype. The graph path runs the same kernels with a fixed launch capacity; valid programs execute the existing instructions, while sentinel programs return before all tensor accesses. Forward and backward are accepted against `use_graph=False` with the tolerances already committed in `tests/ops/test_gdn.py`: `o/ht=0.005`, `dq/dv/dh0=0.007`, `dk=0.008`, and `dbeta/dg=0.015` for the default varlen path; the fused gate branch retains its existing `0.02` parameter-gradient limits.

## Before: dynamic data flow

```text
cu_seqlens
  -> prepare_lens
  -> ceil(length / BT) per sequence
  -> repeat_interleave in _segmented_arange
  -> chunk_indices[actual_NT, 2]
  -> Python len(chunk_indices)
  -> actual_NT kernel grids and [B, actual_NT, ...] state tensors
```

`prepare_chunk_indices` is cached by argument identity. Mutating the contents of the same `cu_seqlens` tensor can therefore reuse old metadata rather than recomputing it.

## Forward call chain before modification

```text
chunk_gated_delta_rule
  -> ChunkGatedDeltaRuleFunction.forward
  -> chunk_gated_delta_rule_fwd
     -> gdn_gate_chunk_cumsum OR chunk_local_cumsum
     -> chunk_gated_delta_rule_fwd_intra
        -> chunk_gated_delta_rule_fwd_kkt_solve_kernel                 (BT=64)
        -> chunk_scaled_dot_kkt_fwd -> solve_tril                     (BT=16/32)
        -> recompute_w_u_fwd
     -> chunk_gated_delta_rule_fwd_h
     -> chunk_fwd_o
```

| Forward stage | Grid/shape dependency before modification | Builds metadata | Host-visible dynamic value | Static padding behavior | Required graph change |
| ------------- | ----------------------------------------- | --------------- | -------------------------- | ----------------------- | --------------------- |
| `prepare_chunk_indices` | Output shape is `actual_NT` | Yes | `repeat_interleave` determines a data-dependent output size | None | Fixed `[NT_max, 2]` device output plus sentinel rows |
| Gate or local cumsum | Grid first axis is `len(chunk_indices)` | If not supplied | Python reads tensor shape | Writes only real chunk tokens | Fixed grid and sentinel early exit |
| Fused KKT plus solve (`BT=64`) | Grid first axis is `actual_NT`; `A` uses physical `T` | If not supplied | Python reads tensor shape | Invalid rows in `A` remain zero because `A` is zero-initialized | Fixed grid and sentinel early exit |
| Unfused KKT and `solve_tril` (`BT=16/32`) | Both grids use `actual_NT`; `A/Ai` use physical `T` | If not supplied | Python reads tensor shape | `Ai` is zero-initialized | Fixed grids and sentinel early exits |
| `recompute_w_u_fwd` | Grid first axis is `actual_NT`; `w/u` use physical `T` | If not supplied | Python reads tensor shape | Uncovered rows are undefined | Fixed grid and sentinel early exit |
| `chunk_gated_delta_rule_fwd_h` | Grid uses fixed `N`; `h` shape uses `actual_NT`; chunk offsets are identity-cached | May build indices and offsets | Python reads metadata shapes | Zero-length sequences retain their initial state | Fixed `h[NT_max]` and live device-side chunk offsets |
| `chunk_fwd_o` | Grid second axis is `actual_NT`; output uses physical `T` | If not supplied | Python reads tensor shape | Uncovered output rows are undefined | Fixed grid and sentinel early exit |

## Backward call chain before modification

```text
ChunkGatedDeltaRuleFunction.backward
  -> chunk_gated_delta_rule_bwd
     -> recompute_w_u_fwd
     -> chunk_gated_delta_rule_fwd_h
     -> chunk_bwd_dv_local
     -> chunk_gated_delta_rule_bwd_dhu
     -> chunk_bwd_dqkwg
     -> prepare_wy_repr_bwd
     -> chunk_local_cumsum(reverse=True)
     -> gdn_gate_bwd                                      (fused gate only)
```

| Backward stage | Grid/shape dependency before modification | Padding/reduction risk | Required graph change |
| -------------- | ----------------------------------------- | ---------------------- | --------------------- |
| Forward recomputation | Same dynamic grids and shapes as forward | Undefined inactive rows | Reuse static metadata and fixed grids |
| `chunk_bwd_dv_local` | Grid first axis is `actual_NT` | Uncovered `dv` rows undefined | Fixed grid and sentinel early exit |
| `chunk_gated_delta_rule_bwd_dhu` | Grid uses fixed `N`; `dh` shape uses `actual_NT`; offsets are cached | Unused `dh` rows stale | Fixed `dh[NT_max]` and live offsets |
| `chunk_bwd_dqkwg` | Grid second axis is `actual_NT`; `dg` has an extra `NK` reduction axis | Padding rows can remain undefined before reduction | Fixed grid, sentinel early exit, and no invalid-row contribution |
| `prepare_wy_repr_bwd` | Grid first axis is `actual_NT` | Gradient padding rows undefined | Fixed grid and sentinel early exit |
| Reverse local cumsum | Grid first axis is `actual_NT` | Its output feeds full-token gate reductions | Fixed grid, sentinel early exit, and zero inactive rows |
| `gdn_gate_bwd` | Grid covers physical `T_max`; `dA` and `dbias` reduce across all physical tokens | Padding can pollute parameter gradients | Device-side `actual_t` mask and zero padding gradients |

## Reproduction protocol

`scripts/repro_gdn_varlen_cudagraph.py` uses fixed-address tensors with `T_max=1024`, `N_max=4`, and two cumulative-length layouts:

```text
A = [0, 256, 512, 768, 1024]   actual_NT=16
B = [0, 100, 400, 1024, 1024]  actual_NT=17
```

The uncached stage exposes capture safety of metadata construction. The cached stage tests whether mutating the same `cu_seqlens` object rebuilds metadata or silently replays A's chunk schedule for B. Actual command output is recorded after running the script.

Command:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/repro_gdn_varlen_cudagraph.py --stage all
```

Observed on the baseline commit with PyTorch `2.11.0+cu128`, Triton `3.6.0`, and an RTX 4090:

- Uncached metadata: capture failed in `_segmented_arange -> torch.repeat_interleave` with `cudaErrorStreamCaptureUnsupported`; the outer context then reported `cudaErrorStreamCaptureInvalidated`.
- Identity-cached metadata: capture succeeded, but replay after changing the same cumulative-length buffer from A to B did not match eager B. Output `max_abs=0.3562774658203125`; final-state `max_abs=0.9203461408615112`.

This proves both failure modes in the original dependency chain: rebuilding actual-size metadata is not capture-safe, while skipping the rebuild through identity caching silently retains stale scheduling metadata.

## After: static data flow

Phase 2 implements the metadata portion of the target flow:

```text
fixed cu_seqlens[N_max + 1]
  -> direct on-device adjacent difference (no identity-cached helper)
  -> chunk_counts[N_max]
  -> chunk_offsets[N_max + 1]
  -> fixed slots[NT_max]
  -> searchsorted(chunk_offsets, slots)
  -> chunk_indices[NT_max, 2]
  -> invalid slots replaced with [-1, 0]
```

The implementation deliberately does not call `prepare_lens`, because that helper is identity-cached and would preserve old lengths when the same cumulative-length buffer is updated before replay. `torch.searchsorted` has a fixed output shape determined by `slots`, so no data-dependent allocation or host synchronization is required.

## File-level changes

| File | Function | Before | After | Reason |
| ---- | -------- | ------ | ----- | ------ |
| `fla/ops/utils/index.py` | `prepare_chunk_indices_static` | No fixed-capacity metadata builder | Returns fixed chunk indices and offsets with sentinel rows, entirely from device ops | Remove actual-`NT` shape and host-sync dependencies |
| `fla/ops/utils/__init__.py` | utility export | Static helper unavailable through `fla.ops.utils` | Exports the shared helper | Reuse the same implementation across graph-capable operators |
| `tests/ops/utils/test_index.py` | static metadata tests | Dynamic helper only | Checks `BT=16/32/64`, int32/int64, zero-length sequences, partial capacity, fixed shape, sentinels, and capture/replay | Prove semantic parity and graph-safe regeneration |
| `scripts/repro_gdn_varlen_cudagraph.py` | baseline reproducer | No focused GDN reproducer | Separates uncached capture failure from stale-cache replay | Preserve the original failure as executable evidence |
| `GDN_GRAPH_CAPTURE_WALKTHROUGH.md` | learning record | Absent | Records contract, call chains, evidence, and staged changes | Make the scheduling change reviewable without changing operator math |

## Key before/after code

Pending implementation and verification.

## What each test proves

- `test_prepare_chunk_indices_static`: the valid prefix equals dynamic `prepare_chunk_indices`; offsets have the same values; output shapes and dtypes remain fixed; all unused rows are `[-1, 0]`.
- `test_prepare_chunk_indices_static_graph_replay`: one captured metadata graph regenerates indices and offsets after in-place changes to the same cumulative-length buffer.
- Full `tests/ops/utils/test_index.py`: all 93 pre-existing tests plus 31 new static cases passed, for `124 passed` total.

## Benchmark results

Pending implementation and verification.

## Current limitations

- The planned graph path has fixed `T_max` and `N_max` capacities; callers must use eager execution above capacity.
- KDA, convolution, fused recurrent decode, Context Parallel, NPU, graph bucketing, and framework-level scheduling are outside this slice.

## Recommended reading order

Pending until the final file set is known.
