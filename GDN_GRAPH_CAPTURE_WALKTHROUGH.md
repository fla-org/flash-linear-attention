# GDN Varlen CUDA Graph Walkthrough

## Scope and baseline

- Baseline commit: `35dceaee5408e69a555fec34cb215c93c375dabe`
- Stage-one branch: `feat/cuda-graph-stage1`
- In scope: the varlen GDN chunk operator, the real `GatedDeltaNet` layer prefill/chunk path, its Triton short convolution, KDA chunk/layer paths, native Triton/CUDA forward, backward, capture, and replay.
- Out of scope: fused recurrent decode, cache-update decode kernels, automatic bucket management, scheduling frameworks, NPU graph execution, and new multi-GPU or Context Parallel support. Existing KDA Context Parallel behavior must be preserved.
- Mathematical contract: preserve the eager GDN operation, precision staging, accumulation order, output structure, and existing tolerances. Only scheduling metadata and boundary guards may change.

## Contract cells

| Source | Request | Chunk geometry | Gate/state axes | Disposition | Oracle and acceptance |
| ------ | ------- | -------------- | --------------- | ----------- | --------------------- |
| Existing public API and GDN tests | `use_graph=False`, dense or varlen, existing backends | `BT in {16, 32, 64}` | All previously supported combinations | Existing fallback | Existing tests and naive recurrent oracle; no route, output, or gradient change |
| New graph mode | `use_graph=True`, varlen, native Triton on NVIDIA CUDA | `BT in {16, 32, 64}`, fixed `T_max` and `N_max` | Precomputed or fused raw gate, post-sigmoid beta, initial/final state | Optimized path | Same-call eager GDN on the real token prefix; existing per-output and per-gradient tolerances |
| Dense graph mode | `use_graph=True` or explicit graph mode, dense B=1 | Fixed token shape | No layer cache or mask-derived packing | Optimized path | Layer forward/backward replay parity without external varlen metadata |
| New graph mode boundary | Forced graph, dense B>1 | Any | Any | Explicit unsupported error | Auto retains the eager path |
| GDN graph mode boundary | `use_graph=True`, Context Parallel | Any | Any | Explicit unsupported error | Public validation raises before kernel execution; existing KDA CP support is separate |
| New graph mode boundary | `use_graph=True`, non-NVIDIA backend | Any | Any | Explicit unsupported error | Public validation raises before kernel execution |
| FlashQLA dispatch | `use_graph=True`, otherwise FlashQLA-compatible | `BT=64` | FlashQLA-supported subset | Existing fallback | FlashQLA verifier rejects graph mode; native Triton graph path supplies public semantics |
| GatedDeltaNet layer | graph-compatible varlen prefill/chunk | `BT in {16, 32, 64}` | projection, short convolution, GDN chunk, output projection | Optimized path when layer constraints hold | Layer capture/replay tests compare valid output and gradients with eager |
| KDA chunk/layer | graph-compatible varlen chunk path | `BT in {32, 64}` | KDA gate/state and optional short convolution | Optimized path when layer constraints hold | KDA operator and layer capture/replay tests compare valid output and gradients with eager |

The graph capacity is defined by the physical input shape and the padded cumulative-length buffer:

```text
T_max = q.shape[1]
N_max = cu_seqlens.shape[0] - 1
NT_max = ceil_div(T_max, BT) + N_max - 1
```

The caller pads unused sequence entries by repeating `actual_t`. For example, two sequences with lengths `[100, 70]` at `N_max=4` use `cu_seqlens=[0, 100, 170, 170, 170]`.

## Numerical budget

No leaf stage may change its load, operand, accumulation, or store dtype. The graph path runs the same kernels with a fixed launch capacity; valid programs execute the existing instructions, while sentinel programs return before sequence or data tensor accesses. Forward and backward are accepted against `use_graph=False` with the tolerances already committed in `tests/ops/test_gdn.py`: `o/ht=0.005`, `dq/dv/dh0=0.007`, `dk=0.008`, and `dbeta/dg=0.015` for the default varlen path; the fused gate branch retains its existing `0.02` parameter-gradient limits.

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

Graph mode replaces the actual-size metadata dependency with a fixed-capacity path:

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

The public operator derives the capacity entirely from static tensor shapes:

```text
q.shape[1]                    -> T_max
cu_seqlens.shape[0] - 1       -> N_max
ceil_div(T_max, BT)+N_max-1   -> NT_max
```

Every graph-capable chunk kernel launches against `NT_max`. A valid row in `chunk_indices` maps the program to one sequence and one local chunk. A sentinel row contains `[-1, 0]`; the program tests the sequence id and returns before reading `cu_seqlens`, inputs, states, or outputs. State kernels remain launched over fixed `N_max`, read live `chunk_offsets`, and allocate their chunk-state axis at `NT_max`.

Token-shaped outputs and backward intermediates retain the physical `T_max` shape. They are zero-initialized only in graph mode so that tokens after `actual_t = cu_seqlens[-1]` cannot expose stale memory. The fused gate backward kernel additionally reads `actual_t` on-device and excludes padding from the `dA_log` and `ddt_bias` reductions.

```text
fixed-address input and cu_seqlens buffers
  -> device-side fixed-shape metadata[NT_max]
  -> fixed kernel launches
       -> valid row: execute the original eager math
       -> sentinel row: return before data access
  -> zero physical padding outside actual_t
  -> replay after caller updates the same buffers with copy_
```

## File-level changes

| File | Function | Before | After | Reason |
| ---- | -------- | ------ | ----- | ------ |
| `fla/ops/utils/index.py` | `prepare_chunk_indices_static` | No fixed-capacity metadata builder | Returns fixed chunk indices and offsets with sentinel rows, entirely from device ops | Remove actual-`NT` shape and host-sync dependencies |
| `fla/ops/utils/__init__.py` | utility export | Static helper unavailable through `fla.ops.utils` | Exports the shared helper | Reuse the same implementation across graph-capable operators |
| `tests/ops/utils/test_index.py` | static metadata tests | Dynamic helper only | Checks `BT=16/32/64`, int32/int64, zero-length sequences, partial capacity, fixed shape, sentinels, and capture/replay | Prove semantic parity and graph-safe regeneration |
| `scripts/repro_gdn_varlen_cudagraph.py` | baseline reproducer | No focused GDN reproducer | Separates uncached capture failure from stale-cache replay | Preserve the original failure as executable evidence |
| `fla/ops/gated_delta_rule/chunk.py` | public op, autograd forward/backward, helpers | Eager-only dynamic metadata | Adds opt-in `use_graph`, `graph_mode`, derives `NT_max`, accepts or builds static indices and offsets, and threads graph mode through forward/backward | Give capture an explicit contract while leaving `use_graph=False` unchanged |
| `fla/ops/gated_delta_rule/chunk_fwd.py` | fused KKT plus triangular solve | Every launched row was assumed valid | Adds a constexpr graph flag and sentinel exit before sequence/data access | Make the `BT=64` intra-chunk path safe at the fixed grid size |
| `fla/ops/gated_delta_rule/gate.py` | fused gate cumsum and gate backward | Dynamic chunk grid; full physical-token parameter reductions | Adds sentinel handling, graph-only zero initialization, and an on-device `actual_t` mask | Exclude inactive chunks and padding from gate outputs and parameter gradients |
| `fla/ops/gated_delta_rule/wy_fast.py` | `recompute_w_u_fwd`, `prepare_wy_repr_bwd` | Dynamic chunk grids and partially written `empty` outputs | Adds sentinel exits and graph-only zero initialization | Keep forward recomputation and WY gradients valid under `NT_max` launches |
| `fla/ops/utils/cumsum.py` | local chunk cumsum forward/backward | Dynamic chunk grids and `empty` outputs | Adds sentinel exits, fixed-grid propagation, and graph-only zero initialization | Keep gate scans capture-safe in both directions |
| `fla/ops/utils/solve_tril.py` | triangular solve variants | Every chunk row was assumed valid | Adds sentinel exits before sequence/data access and propagates graph mode | Make the unfused `BT=16/32` path safe at `NT_max` |
| `fla/ops/common/chunk_scaled_dot_kkt.py` | unfused KKT | Dynamic chunk grid and partially written `empty` output | Adds sentinel exit and graph-only zero initialization | Complete graph support for the non-64 chunk sizes |
| `fla/ops/common/chunk_delta_h.py` | chunk-state forward and backward | Rebuilt identity-cached offsets and sized state intermediates by actual `NT` | Accepts live static offsets, sizes intermediates by `NT_max`, and zero-initializes inactive rows | Preserve inter-chunk state flow without stale offsets or inactive-state data |
| `fla/ops/common/chunk_o.py` | output, local `dv`, and `dq/dk/dw/dg` kernels | Dynamic chunk grids and partially written token/reduction buffers | Adds sentinel exits and graph-only zero initialization | Prevent inactive chunks and physical padding from contaminating outputs or gradients |
| `fla/ops/common/backends/intracard.py` | backend verifier | Graph mode could reach an unsupported implementation | Rejects `use_graph=True` | Route the public contract only to the validated native Triton path |
| `fla/ops/common/backends/tilelang/__init__.py` | backend verifier | Graph mode could reach an unsupported implementation | Rejects `use_graph=True` | Avoid silent fallback to an unvalidated graph path |
| `fla/ops/gated_delta_rule/backends/flash_qla.py` | FlashQLA verifier | Graph mode was unknown to the verifier | Rejects `use_graph=True` | Keep graph semantics on the native Triton implementation |
| `fla/layers/gated_deltanet.py` | GatedDeltaNet prefill/chunk layer | Layer did not expose the graph contract | Routes graph parameters through projection, short convolution, GDN chunk, and output projection; unsupported cache/decode cases fall back or error | Validate the real layer path rather than an isolated operator |
| `fla/modules/conv/causal_conv1d.py`, `fla/modules/conv/short_conv.py` | Triton short convolution | Dynamic chunk/state allocation | Adds fixed graph-capacity metadata, grids, and padding-safe forward/backward buffers; decode cache paths remain unchanged | Keep the GDN prefill convolution capture-safe |
| `fla/ops/kda/*.py`, `fla/layers/kda.py` | KDA chunk and layer paths | Eager-only dynamic chunk metadata | Threads the same static metadata, sentinel, fixed-grid, and route contract through KDA forward/backward and layer prefill | Extend the validated slice without changing KDA decode |
| `tests/ops/test_gdn_graph.py` | GDN graph tests | No operator-level capture/replay coverage | Captures once, mutates fixed buffers, replays multiple layouts, and compares forward/state/all gradients with eager | Prove the complete varlen forward/backward vertical slice |
| `tests/layers/test_gated_deltanet_graph.py`, `tests/modules/test_conv_graph.py`, `tests/ops/test_kda_graph.py`, `tests/layers/test_kda_layer_graph.py` | Layer, convolution, and KDA graph tests | No integrated graph coverage | Capture/replay real layer and component paths, update metadata, and check padding and gradients | Prove the extended scope separately before using it in a model |
| `benchmarks/ops/benchmark_gdn_graph.py` | standalone latency benchmark | No eager-versus-replay benchmark | Warms both paths, captures outside timing, and measures steady eager/replay latency with CUDA Events | Record the cost and launch-overhead benefit without counting JIT/autotune/capture |
| `benchmarks/ops/benchmark_gdn_graph_load.py` | high-load operator/component benchmark | No capacity/layout/fallback matrix | Measures eager-live, eager-fixed, graph-replay, graph-update, copy-only, capture, p95, memory, and break-even; supports GDN, layer, convolution, and KDA components | Expose sparse-bucket regressions instead of reporting only ideal replay |
| `GDN_GRAPH_CAPTURE_WALKTHROUGH.md` | learning record | Absent | Records contract, call chains, evidence, and staged changes | Make the scheduling change reviewable without changing operator math |

## Key before/after code

### Dynamic metadata to fixed metadata

Before, the number of output rows was determined by the live sequence lengths:

```python
chunk_counts = ceil_div(lengths, chunk_size)
seg_id, local_chunk_id = _segmented_arange(chunk_counts)
chunk_indices = torch.stack([seg_id, local_chunk_id], 1)
```

`_segmented_arange` uses `repeat_interleave`, so `chunk_indices.shape[0]` is data-dependent. Graph mode instead enumerates a fixed number of slots and marks its invalid suffix:

```python
slots = torch.arange(nt_max, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
sequence_ids = torch.searchsorted(chunk_offsets, slots, right=True) - 1
local_chunk_ids = slots - chunk_offsets[sequence_ids]
valid = slots < chunk_offsets[-1]
sequence_ids = torch.where(valid, sequence_ids, torch.full_like(sequence_ids, -1))
local_chunk_ids = torch.where(valid, local_chunk_ids, torch.zeros_like(local_chunk_ids))
```

### Actual `NT` to `NT_max`

Before:

```python
chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
NT = len(chunk_indices)
```

Graph mode:

```python
nt_max = ceil_div(q.shape[1], chunk_size) + cu_seqlens.shape[0] - 2
chunk_indices, chunk_offsets = prepare_chunk_indices_static(cu_seqlens, chunk_size, nt_max)
NT = len(chunk_indices)  # always NT_max
```

### Ordinary chunk program to sentinel early exit

Before:

```python
i_n, i_t = load_chunk_index(i_t)
bos, eos = load_sequence_bounds(cu_seqlens, i_n)
```

Graph mode adds the guard before using `i_n` as an address:

```python
i_n, i_t = load_chunk_index(i_t)
if USE_GRAPH and i_n < 0:
    return
bos, eos = load_sequence_bounds(cu_seqlens, i_n)
```

### Physical `T_max` grid to live `actual_t` mask

The fused gate backward reduction covers the fixed physical token dimension. In graph mode it obtains the live prefix length without a host read:

```python
actual_t = tl.load(cu_seqlens + N).to(tl.int64) if USE_GRAPH else T
m_t = o_t < actual_t
```

The same graph can therefore reduce a different valid token prefix after `cu_seqlens.copy_(...)` without allowing padding to contribute to `dg`, `dA_log`, or `ddt_bias`.

### Eager-only operator to explicit graph mode

```python
def chunk_gated_delta_rule(..., use_graph: bool = False):
    if use_graph:
        chunk_indices, chunk_offsets = prepare_chunk_indices_static(...)
    elif chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(...)
```

The default remains the original path. Dense B=1 inputs can use graph mode without varlen metadata, including at the GDN/KDA layer entry. Forced GDN graph mode rejects dense B>1, Ascend, and Context Parallel; auto uses eager for these cases. This does not remove KDA's existing Context Parallel graph support. Device-resident `cu_seqlens` is required for captured packed-varlen execution; a CPU `cu_seqlens_cpu` may be supplied for host-side routing statistics. Caller-provided `chunk_indices` and `chunk_offsets` are accepted when they have the fixed graph shape, dtype, device, and contiguous layout. Layer inputs requiring mask-based unpadding must be prepacked before capture. Ascend graph execution remains incomplete; host-side backend compatibility tests are not NPU runtime validation.

When operator auto routing chooses eager, it clears graph metadata and re-enters backend dispatch once with explicit eager mode. Eligible FlashQLA/FlashKDA implementations retain their normal priority; otherwise the native eager implementation runs. This adds host dispatch work, not a second kernel execution. Forced Graph and graph-eligible auto calls do not take this re-entry.

## What each test proves

- `test_prepare_chunk_indices_static`: the valid prefix equals dynamic `prepare_chunk_indices`; offsets have the same values; output shapes and dtypes remain fixed; all unused rows are `[-1, 0]`.
- `test_prepare_chunk_indices_static_graph_replay`: one captured metadata graph regenerates indices and offsets after in-place changes to the same cumulative-length buffer.
- `test_gdn_varlen_graph_forward_replay`: one captured forward graph handles three in-place `cu_seqlens` layouts, different actual `NT`, `BT=16/32/64`, precomputed and fused gates, initial/final state, partial physical capacity, and deliberately nonzero padding. It compares the valid output and final state with a true-length eager call and requires graph padding output to be zero.
- `test_gdn_varlen_graph_backward_replay`: captures forward plus backward, then updates the same input, output-gradient, and cumulative-length buffers. It checks `dq`, `dk`, `dv`, `dg`, `dbeta`, `dinitial_state`, and, for fused gates, `dA_log` and `ddt_bias`; all token-gradient padding must be zero.
- `tests/ops/test_gdn.py`: protects the existing dense/varlen eager API and numerical behavior when `use_graph=False`.
- `tests/ops/test_gdn_kernels.py`, `tests/ops/utils/test_cumsum.py`, and `tests/ops/test_solve_tril.py`: protect the shared kernels whose signatures or graph-only branches changed.
- `tests/layers/test_gated_deltanet_graph.py`: captures the real projection → short convolution → GDN chunk → output projection layer path.
- `tests/modules/test_conv_graph.py`: checks short-convolution forward/backward replay and state/padding behavior.
- `tests/ops/test_kda_graph.py`, `tests/layers/test_kda_layer_graph.py`: check KDA chunk and layer forward/backward replay, metadata reuse, and eager fallback.

These tests describe coverage, not a claim that every dependent test has run on the current branch.

## Stage-one reference check

This branch is a CUDA-focused stage related to #1155, not completion of the full issue. Ascend changes are limited to eager-call signature compatibility and explicit rejection of newly introduced unsupported graph requests; no unfinished Ascend graph kernels are included. Backend compatibility tests use mocked launches and do not establish NPU numerical correctness.

Integration checks on RTX 4090:

- Layer, short-convolution, routing, graph utility, and backend compatibility checks: 90 passed.
- GDN replay checks: 13 passed. One older metadata test still expected dense B=1 without `cu_seqlens` to fail; it was updated to check invalid CPU metadata and missing external metadata in explicit forced mode, then passed in the next run. Numerical assertions and tolerances were unchanged.
- Corrected metadata test, KDA graph tests, index tests, and part of the cumulative-sum suite: 174 completed passes before intentionally interrupting compilation of unchanged global cumulative-sum cases. This is a partial run, not a full-suite pass. The KDA single-rank CP graph case passed; multi-rank CP was not run.
- Focused local cumulative-sum and triangular-solve checks: 11 passed, 1 skipped because the large-offset case requires Blackwell hardware.
- Commit hooks and repository copyright-header checks passed.

The following small packed-varlen check ran on the integrated implementation at `d3fa0761`, using an RTX 4090, PyTorch `2.11.0+cu128`, and Triton `3.6.0`:

```bash
python benchmarks/ops/benchmark_gdn_graph.py --extended \
  --components operator kda --t-max 512 --n-max 4 --actual-ratio 0.75 \
  --layouts balanced --heads 4 --value-heads 4 --key-dim 64 --value-dim 64 \
  --modes fwd fwdbwd --warmup 3 --iterations 10 --repeats 3
```

Both operators used bf16, `BT=64`, 384 live tokens, and 8 live chunks in an 11-chunk capacity. All four benchmark correctness checks and metadata-address checks passed. Values below are median synchronized wall times in milliseconds; compilation, capture, and warmup are excluded.

| Operator | Mode             | Eager, fixed capacity | Graph replay | Input update + replay |
| -------- | ---------------- | --------------------: | -----------: | --------------------: |
| GDN      | Forward          |                 0.843 |        0.036 |                 0.072 |
| GDN      | Forward+backward |                11.463 |        0.116 |                 0.144 |
| KDA      | Forward          |                 1.008 |        0.052 |                 0.074 |
| KDA      | Forward+backward |                 8.526 |        0.162 |                 0.320 |

The update measurement includes device-to-device input and metadata copies, plus output-gradient copies for forward+backward. These are execution-mode comparisons on the same implementation, not upstream-versus-candidate kernel regression measurements or isolated backward timings. The shared host was running other workloads, so these small launch-bound cases are reference checks, not production speedup claims. The benchmark's external auto policy selects eager at `8/11` chunk utilization with its default `0.75` threshold; the table separately measures forced replay and does not claim auto selects the fastest path.

The follow-up below adds selected two-rank CP coverage and consumer-GPU before/after measurements. Full dependent regression, target datacenter GPU measurements, and profiling remain incomplete. No NPU or full-model serving performance claim is made.

## PR 1230 validation follow-up (2026-09-08)

This is partial validation, not a green merge gate. The unchanged upstream snapshot is `9d981ffe`; candidate operator code is `8a88693d`. This follow-up changes only the benchmark harness, a CP test's port configuration, and this document. Hardware is RTX 4090, PyTorch `2.11.0+cu128`, Triton `3.6.0`.

### Correctness and distributed checks

- Focused eager regression: GDN 45 passed and 9 skipped, shared GDN kernels 56 passed, and KDA 21 passed before stopping at the first failure. The run excluded fused recurrent and FlashQLA paths; it is not a full-suite pass.
- The failing KDA varlen case uses H4, D60, fp16, boundaries `[0, 31, 96, 160]`, `mask_p=0.1`, fused gate, `safe_gate=False`, `disable_recompute=True`, and chunk size 32. It raises CUDA `misaligned address` in `fla/ops/gla/chunk.py::chunk_gla_fwd_kernel_o`. The exact case also fails on the unchanged upstream snapshot on the same GPU with `CUDA_LAUNCH_BLOCKING=1`. Both runs additionally report a teardown error after the CUDA context is poisoned. This establishes a pre-existing failure on this configuration, not a newly introduced regression; it does not make the KDA gate green.
- A subsequent KDA tail run was intentionally interrupted after 272 seconds with zero completed tests. It contributes no passing evidence. KDA before/after performance collection was deferred while its correctness gate is red.
- Triton short convolution eager dense/varlen checks: 11 passed, 9 non-Triton cases deselected.
- On physical GPUs 2 and 4, KDA Graph CP replay, GDN eager CP, and KDA eager CP passed. Conv CP initially failed before kernel execution because TCP port 29500 was occupied. The test now honors an externally supplied `MASTER_PORT`, retaining 29500 as its default; retrying only that case with `MASTER_PORT=29753` passed. Four distinct selected two-rank CP tests ultimately passed, not an exhaustive distributed matrix.

The eager regression command was:

```bash
CUDA_VISIBLE_DEVICES=4 python -m pytest \
  tests/ops/test_gdn.py tests/ops/test_gdn_kernels.py \
  tests/ops/test_kda.py::test_chunk tests/ops/test_kda.py::test_chunk_varlen \
  tests/ops/test_kda.py::test_chunk_state_v_first \
  tests/ops/test_kda.py::test_chunk_use_beta_sigmoid_in_kernel \
  tests/ops/test_kda.py::test_chunk_return_intermediate_states \
  tests/modules/test_conv.py::test_conv tests/modules/test_conv.py::test_conv_varlen \
  -k 'not fused_recurrent and not flash_qla' -q -o log_cli=false --disable-warnings --maxfail=1
```

The standalone convolution run selects the last two nodes with `-k triton`. CP nodes are `tests/context_parallel/test_cp_kda_graph.py` and `test_cp2_sequence_cut` in each of `test_cp_gdn.py`, `test_cp_kda.py`, and `test_cp_conv.py` under the same directory.

### Same-GPU reference measurements

`benchmarks/ops/benchmark_graph_regression.py` compares outputs, final states, and every input/parameter gradient across checkouts before timing. All eight GDN/Conv cases passed bitwise eager parity; candidate forward and forward+backward Graph checks passed against eager with the repository comparison helper at tolerance 0.005 and explicit finite checks. These fixed-content checks supplement, not replace, the earlier in-place metadata-update tests.

Run the same script sequentially on the same physical GPU, with the baseline output supplied to the candidate:

```bash
CUDA_VISIBLE_DEVICES=4 FLA_DISABLE_BACKEND_DISPATCH=1 python \
  /data2/users/dulz/code/FLA-pr-cuda/benchmarks/ops/benchmark_graph_regression.py \
  --repo /data2/users/dulz/code/FLA-pr-baseline --ops gdn conv \
  --output /data2/users/dulz/code/FLA/profile/gdn-cudagraph/pr1230-baseline-20260908

CUDA_VISIBLE_DEVICES=4 FLA_DISABLE_BACKEND_DISPATCH=1 python \
  /data2/users/dulz/code/FLA-pr-cuda/benchmarks/ops/benchmark_graph_regression.py \
  --repo /data2/users/dulz/code/FLA-pr-cuda --ops gdn conv --graph \
  --reference /data2/users/dulz/code/FLA/profile/gdn-cudagraph/pr1230-baseline-20260908.pt \
  --output /data2/users/dulz/code/FLA/profile/gdn-cudagraph/pr1230-candidate-20260908
```

Workload: seed 42, bf16 activations, B1, H4, D64, T512/2048. GDN uses chunk size 64, fused gates/beta, QK normalization, and initial/final states. Conv uses 256 channels, width 4, bias, SiLU, and initial/final states. Varlen has four sequences with boundaries `[0, 63, T//3, T-17, T]`; all physical tokens are live. Each median uses five samples of 30 calls after three warmup calls. FP32 matmul precision is highest and TF32 is disabled. The script verifies the imported checkout and records software/device metadata.

All values below are synchronized wall milliseconds. F+B means forward plus backward, including gradient-buffer clearing; it is not backward-only. Graph timing is replay-only, excluding input updates, JIT, capture, and warmup. The JSON files also contain raw wall samples and CUDA-event batch times; those event intervals can include GPU idle gaps caused by host submission and are not isolated kernel times.

| Operator | Layout | Tokens | Mode | Baseline eager | Candidate eager | Candidate replay |
| -------- | ------ | -----: | ---- | -------------: | --------------: | ---------------: |
| GDN      | dense  |    512 | F    |         0.7289 |          0.7051 |           0.0411 |
| GDN      | dense  |    512 | F+B  |         7.9636 |          4.8054 |           0.1369 |
| GDN      | dense  |   2048 | F    |         0.7270 |          0.7246 |           0.0673 |
| GDN      | dense  |   2048 | F+B  |         6.6925 |         10.5790 |           0.2527 |
| GDN      | varlen |    512 | F    |         0.7499 |          0.7380 |           0.0627 |
| GDN      | varlen |    512 | F+B  |         6.9079 |          4.1434 |           0.1537 |
| GDN      | varlen |   2048 | F    |         0.7505 |          0.7405 |           0.0833 |
| GDN      | varlen |   2048 | F+B  |         9.0904 |         10.6984 |           0.2437 |
| Conv     | dense  |    512 | F    |         0.1929 |          0.2636 |           0.0069 |
| Conv     | dense  |    512 | F+B  |         1.4759 |          1.7801 |           0.0339 |
| Conv     | dense  |   2048 | F    |         0.1919 |          0.3084 |           0.0083 |
| Conv     | dense  |   2048 | F+B  |         1.4346 |          1.2444 |           0.0423 |
| Conv     | varlen |    512 | F    |         0.2019 |          0.3583 |           0.0286 |
| Conv     | varlen |    512 | F+B  |         1.4805 |          1.4302 |           0.0591 |
| Conv     | varlen |   2048 | F    |         0.3010 |          0.3088 |           0.0310 |
| Conv     | varlen |   2048 | F+B  |         1.2272 |          1.3024 |           0.0719 |

The baseline file contains 16 timing rows and the candidate file 32, covering all eight cases. Replay reduces measured submission overhead on these workloads, but this is not evidence that eager performance is preserved: Conv forward latency increased by 2.6%-77.4%, and GDN T2048 F+B increased by 17.7%-58.1%. The shared host and large timing variation limit interpretation. These increases remain unresolved; attributing them solely to environmental noise or declaring no regression would be unsupported. Controlled repeat measurements and diagnosis are still required before a performance conclusion.

### Remaining gates

- KDA has the baseline-reproduced failure above; broader dependent regression is incomplete. No tests or tolerances were weakened to bypass it.
- Nsight Compute 2024.3.0 was found at `/usr/local/cuda/bin/ncu`. A one-launch full collection targeting GDN `chunk_fwd_kernel_o` failed with `ERR_NVGPUCTRPERM`. No valid counter metrics were obtained; enabling access requires administrator coordination.
- Target datacenter GPU evidence is unavailable. The RTX 4090 measurements are consumer reference data, not H100/H20-or-newer validation.
- Eager latency increases need controlled investigation. Do not tick the test/performance checklist items based on this follow-up alone.

Local artifacts are under `/data2/users/dulz/code/FLA/profile/gdn-cudagraph/`, with prefixes `pr1230-{cp,cp-conv-retry,eager,upstream-kda-boundary,eager-tail,conv-eager,baseline,candidate}-20260908`. XML files preserve test outcomes; benchmark JSON/PT files preserve timing samples and numerical snapshots. Raw artifacts are not included in the PR. Earlier pass counts are separate historical evidence and are not added to this run's totals.

## Historical benchmark results

The small benchmark below predates stage-one integration. It is historical reference evidence, not certification of this branch. Raw local reports are not included in the repository.

```bash
python benchmarks/ops/benchmark_gdn_graph.py \
  --t-max 256 512 1024 2048 --n-max 4 --heads 16 --dim 128 --chunk-size 64 \
  --actual-ratio 0.75 --dtype bfloat16 --modes fwd fwdbwd --warmup 5 --iterations 100 --repeats 5
```

Environment: ref `06106199`, NVIDIA GeForce RTX 4090, CUDA 12.8, PyTorch `2.11.0+cu128`, Triton `3.6.0`. Each reported value is the median of five CUDA Event samples, with 100 calls per sample. Both paths are warmed before capture; capture, JIT compilation, autotuning, and warmup are excluded. The workload uses the same fixed-capacity inputs for timing; changing buffer contents across replay is covered by the correctness tests rather than timed here.

| mode | T_max | actual_t | N_max | actual_NT | NT_max | eager run 1 (ms) | graph run 1 (ms) | speedup run 1 | eager run 2 (ms) | graph run 2 (ms) | speedup run 2 |
| ---- | ----: | -------: | ----: | --------: | -----: | ---------------: | ---------------: | ------------: | ---------------: | ---------------: | ------------: |
| fwd | 256 | 192 | 4 | 5 | 7 | 0.812 | 0.071 | 11.49x | 0.775 | 0.071 | 10.96x |
| fwd | 512 | 384 | 4 | 8 | 11 | 0.780 | 0.079 | 9.88x | 0.773 | 0.079 | 9.78x |
| fwd | 1024 | 768 | 4 | 14 | 19 | 0.784 | 0.105 | 7.44x | 0.775 | 0.105 | 7.37x |
| fwd | 2048 | 1536 | 4 | 26 | 35 | 0.790 | 0.185 | 4.28x | 0.774 | 0.185 | 4.19x |
| fwdbwd | 256 | 192 | 4 | 5 | 7 | 9.573 | 0.191 | 50.25x | 7.902 | 0.190 | 41.60x |
| fwdbwd | 512 | 384 | 4 | 8 | 11 | 7.071 | 0.234 | 30.19x | 7.038 | 0.233 | 30.18x |
| fwdbwd | 1024 | 768 | 4 | 14 | 19 | 9.157 | 0.363 | 25.26x | 5.155 | 0.363 | 14.21x |
| fwdbwd | 2048 | 1536 | 4 | 26 | 35 | 9.398 | 0.636 | 14.77x | 9.111 | 0.643 | 14.16x |

Replay latency was repeatable across the two runs, while eager forward-backward latency showed substantial host-side variation, especially at `T_max=1024`. These measurements demonstrate that the captured slice removes launch/dispatcher overhead on this setup, but the exact speedup ratio is not stable enough for a production performance claim. An RTX 4090 is a consumer reference GPU under the repository performance policy; datacenter H100/H20 or newer measurements are still required for MR-level performance evidence.

## Historical verification

The original GDN-only checks below used PyTorch `2.11.0+cu128` and Triton `3.6.0`. They predate the expanded stage-one scope and must not be presented as its final regression results:

| Command | Device | Result |
| ------- | ------ | ------ |
| `python -m pytest -q tests/ops/test_gdn_graph.py` | physical GPU 3, RTX 4090 | `8 passed, 14 warnings in 1.62s`; every logged graph/eager output and gradient difference was `0.0` |
| `python -m pytest -q tests/ops/test_gdn.py tests/ops/test_gdn_kernels.py` | physical GPU 3, RTX 4090 | `130 passed, 26 skipped, 14 warnings in 305.10s` |
| `python -m pytest -q tests/ops/utils/test_index.py tests/ops/utils/test_cumsum.py tests/ops/test_solve_tril.py` | physical GPU 4, RTX 4090 | `166 passed, 1 skipped, 14 warnings in 9.79s` |
| `python -m pytest -q tests/ops/test_gdn_graph.py tests/ops/utils/test_index.py` | physical GPU 3, RTX 4090 | `132 passed, 14 warnings in 7.27s` after installing the lint dependencies |
| `python -m py_compile <all changed Python files>` | CPU | Passed |
| `python -m ruff check <all changed Python files>` | CPU | Passed with Ruff `0.14.10` after two whitespace-only import-block fixes |
| `pre-commit run --files <all task files>` | CPU | Passed on the second run; Ruff, autopep8, repository hygiene, and banned Triton API checks passed |
| `git diff --check` for the baseline and worktree diffs | CPU | Passed |
| Search changed files for `tl.make_block_ptr` or `tl.advance` | CPU | No matches |
| Added-Python-line length check | CPU | No line exceeds 127 characters |
| Changed-file copyright-header audit | CPU | Passed for every tracked Python file in the baseline diff |

Raw logs and local environment diagnostics are intentionally excluded from this branch.

## Current limitations

- Graph mode has fixed `T_max` and `N_max` capacities. The caller must select a larger graph bucket or use eager execution above capacity.
- Inputs, output gradients, and `cu_seqlens` must keep the captured shapes and addresses. New contents are copied into those buffers with in-place operations such as `copy_` before replay.
- Fewer than `N_max` live sequences must be encoded as zero-length tail entries by repeating `actual_t`; the operator does not discover `actual_n` with a host synchronization.
- The graph route covers varlen `chunk_gated_delta_rule`, GatedDeltaNet prefill/chunk, Triton short convolution, and KDA chunk/layer paths on the native NVIDIA Triton backend. Dense B=1 is supported; dense B>1 and GDN Context Parallel graph execution are excluded. Existing KDA CP graph support is preserved. `cu_seqlens_cpu` is routing metadata only, while captured execution uses device-resident `cu_seqlens`.
- Fused recurrent decode, cache-update decode kernels, full-model capture, graph bucketing, vLLM/SGLang scheduling, NPU, and general multi-GPU serving integration are outside this slice. Selected two-rank CP checks are documented above.
- TileLang, intra-card, and FlashQLA implementations do not claim this graph contract and explicitly decline graph dispatch.
- Fixed `NT_max` launches and graph-only zero initialization can do more work than eager execution when a bucket is sparsely occupied. Bucket policy belongs to the caller and was not designed here.
- The historical benchmark is a steady-state operator microbenchmark on an RTX 4090. It excludes input-copy latency, graph selection, model-level work, and capture cost, and it is not a datacenter-GPU performance conclusion. The extended benchmark separately reports replay-only and input-update-plus-replay timings.

## Recommended reading order

1. Start with this document through "Before: dynamic data flow" to understand the dependency that broke replay.
2. Run `scripts/repro_gdn_varlen_cudagraph.py` (the original baseline reproducer) and read `fla/ops/utils/index.py` plus `tests/ops/utils/test_index.py` to see the dynamic failure and fixed-capacity metadata contract in isolation.
3. Read the public API and autograd plumbing in `fla/ops/gated_delta_rule/chunk.py`.
4. Follow the forward path through `gate.py` or `cumsum.py`, `chunk_fwd.py`, `wy_fast.py`, `common/chunk_delta_h.py`, and `common/chunk_o.py`.
5. Follow backward from `chunk.py` through `common/chunk_o.py`, `common/chunk_delta_h.py`, `wy_fast.py`, reverse cumsum, and fused gate backward.
6. Read `tests/ops/test_gdn_graph.py` to see fixed-address capture, in-place buffer updates, replay, eager comparison, and padding assertions together.
7. Run `benchmarks/ops/benchmark_gdn_graph_load.py` last; its numbers are meaningful only after the correctness contract and timing exclusions are understood.
