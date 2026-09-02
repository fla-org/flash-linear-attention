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
| New graph mode | `use_graph=True`, varlen, native Triton on NVIDIA CUDA | `BT in {16, 32, 64}`, fixed `T_max` and `N_max` | Precomputed or fused raw gate, post-sigmoid beta, initial/final state | Optimized path | Same-call eager GDN on the real token prefix; existing per-output and per-gradient tolerances |
| New graph mode boundary | `use_graph=True`, dense input | Any | Any | Explicit unsupported error | Public validation raises before kernel execution |
| New graph mode boundary | `use_graph=True`, Context Parallel | Any | Any | Explicit unsupported error | Public validation raises before kernel execution |
| New graph mode boundary | `use_graph=True`, non-NVIDIA backend | Any | Any | Explicit unsupported error | Public validation raises before kernel execution |
| FlashQLA dispatch | `use_graph=True`, otherwise FlashQLA-compatible | `BT=64` | FlashQLA-supported subset | Existing fallback | FlashQLA verifier rejects graph mode; native Triton graph path supplies public semantics |

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
| `fla/ops/gated_delta_rule/chunk.py` | public op, autograd forward/backward, helpers | Eager-only dynamic metadata | Adds opt-in `use_graph`, derives `NT_max`, builds/saves static indices and offsets, and threads graph mode through forward/backward | Give capture an explicit contract while leaving `use_graph=False` unchanged |
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
| `tests/ops/test_gdn_graph.py` | GDN graph tests | No operator-level capture/replay coverage | Captures once, mutates fixed buffers, replays multiple layouts, and compares forward/state/all gradients with eager | Prove the complete varlen forward/backward vertical slice |
| `benchmarks/ops/benchmark_gdn_graph.py` | standalone latency benchmark | No eager-versus-replay benchmark | Warms both paths, captures outside timing, and measures steady eager/replay latency with CUDA Events | Record the cost and launch-overhead benefit without counting JIT/autotune/capture |
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

The default remains the original path. Graph mode is deliberately rejected for dense input, non-NVIDIA devices, Context Parallel, caller-provided chunk indices, and CPU cumulative lengths instead of silently changing backend or metadata semantics.

## What each test proves

- `test_prepare_chunk_indices_static`: the valid prefix equals dynamic `prepare_chunk_indices`; offsets have the same values; output shapes and dtypes remain fixed; all unused rows are `[-1, 0]`.
- `test_prepare_chunk_indices_static_graph_replay`: one captured metadata graph regenerates indices and offsets after in-place changes to the same cumulative-length buffer.
- `test_gdn_varlen_graph_forward_replay`: one captured forward graph handles three in-place `cu_seqlens` layouts, different actual `NT`, `BT=16/32/64`, precomputed and fused gates, initial/final state, partial physical capacity, and deliberately nonzero padding. It compares the valid output and final state with a true-length eager call and requires graph padding output to be zero.
- `test_gdn_varlen_graph_backward_replay`: captures forward plus backward, then updates the same input, output-gradient, and cumulative-length buffers. It checks `dq`, `dk`, `dv`, `dg`, `dbeta`, `dinitial_state`, and, for fused gates, `dA_log` and `ddt_bias`; all token-gradient padding must be zero.
- `tests/ops/test_gdn.py`: protects the existing dense/varlen eager API and numerical behavior when `use_graph=False`.
- `tests/ops/test_gdn_kernels.py`, `tests/ops/utils/test_cumsum.py`, and `tests/ops/test_solve_tril.py`: protect the shared kernels whose signatures or graph-only branches changed.

The final command results are recorded in the verification section after all source and documentation changes are complete.

## Benchmark results

Command, run twice on physical GPU 3:

```bash
CUDA_VISIBLE_DEVICES=3 /home/dulz/miniconda3/envs/lz/bin/python benchmarks/ops/benchmark_gdn_graph.py \
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

## Final verification

All runtime checks used `/home/dulz/miniconda3/envs/lz/bin/python` with PyTorch `2.11.0+cu128` and Triton `3.6.0`:

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

The 14 warnings in each pytest process are the environment's existing `torch.jit.script_method` deprecation warnings. Ruff `0.14.10`, pre-commit `4.6.2`, and autopep8 `2.3.2` were installed in the `lz` environment after dependency installation was explicitly approved. The first pre-commit run removed one redundant blank line from each of the reproducer and graph test; the second run passed without modifications. The environment-wide `pip check` still reports pre-existing NumPy constraints from `brevitas` and `tonic`; these packages are outside FLA's dependency set and were not changed. The repository-wide header checker also reports three pre-existing, git-ignored files under `profile/gdn-cudagraph/`; the tracked changed-file audit passes, and those local diagnostics were left untouched.

## Current limitations

- Graph mode has fixed `T_max` and `N_max` capacities. The caller must select a larger graph bucket or use eager execution above capacity.
- Inputs, output gradients, and `cu_seqlens` must keep the captured shapes and addresses. New contents are copied into those buffers with in-place operations such as `copy_` before replay.
- Fewer than `N_max` live sequences must be encoded as zero-length tail entries by repeating `actual_t`; the operator does not discover `actual_n` with a host synchronization.
- The validated graph route is `chunk_gated_delta_rule` with varlen input on the native NVIDIA Triton backend. Dense input, Context Parallel, `cu_seqlens_cpu`, and caller-provided `chunk_indices` are rejected in graph mode.
- KDA, causal convolution, fused recurrent decode, full `GatedDeltaNet` layer/model capture, graph bucketing, vLLM/SGLang scheduling, NPU, and multi-GPU execution are outside this slice.
- TileLang, intra-card, and FlashQLA implementations do not claim this graph contract and explicitly decline graph dispatch.
- Fixed `NT_max` launches and graph-only zero initialization can do more work than eager execution when a bucket is sparsely occupied. Bucket policy belongs to the caller and was not designed here.
- The benchmark is a steady-state operator microbenchmark on an RTX 4090. It excludes input-copy latency, graph selection, model-level work, and capture cost, and it is not a datacenter-GPU performance conclusion.

## Recommended reading order

1. Start with this document through "Before: dynamic data flow" to understand the dependency that broke replay.
2. Run `scripts/repro_gdn_varlen_cudagraph.py` and read `fla/ops/utils/index.py` plus `tests/ops/utils/test_index.py` to see the dynamic failure and fixed-capacity metadata contract in isolation.
3. Read the public API and autograd plumbing in `fla/ops/gated_delta_rule/chunk.py`.
4. Follow the forward path through `gate.py` or `cumsum.py`, `chunk_fwd.py`, `wy_fast.py`, `common/chunk_delta_h.py`, and `common/chunk_o.py`.
5. Follow backward from `chunk.py` through `common/chunk_o.py`, `common/chunk_delta_h.py`, `wy_fast.py`, reverse cumsum, and fused gate backward.
6. Read `tests/ops/test_gdn_graph.py` to see fixed-address capture, in-place buffer updates, replay, eager comparison, and padding assertions together.
7. Run `benchmarks/ops/benchmark_gdn_graph.py` last; its numbers are meaningful only after the correctness contract and timing exclusions are understood.
