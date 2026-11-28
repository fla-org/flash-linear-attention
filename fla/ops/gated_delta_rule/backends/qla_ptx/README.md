# Experimental B300 PTX forward runner

This explicitly imported runner contains the `sm_103a` PTX and host launch adapters for the **171 configurations** in `benchmark_shared.json`. It is not registered with FLA dispatch and does not change any default operator, training, layer, or model path. Design discussion: [RFC #1844](https://github.com/fla-org/flash-linear-attention/issues/1844).

```python
from fla.ops.gated_delta_rule.backends.qla_ptx import run

# Inputs must satisfy the fixed benchmark contract described below.
output, final_state = run(q, k, v, g, beta, scale, initial_state, cu_seqlens)
```

## Input contract

This is a low-level experimental runner for the checked-in benchmark inputs, not a general replacement for `chunk_gated_delta_rule`. Calls require an SM103 GPU; contiguous BF16 Q/K/V with batch size 1 and 128-dimensional keys/values; FP32 precomputed log decays and sigmoid beta; a non-null FP32 K-first initial state; and packed int32 sequence offsets matching a listed configuration. Inputs must be on the same device. The benchmark uses normalized Q/K, `logsigmoid(randn)/16` decay with 25% zero-decay heads, standard-normal values and initial states, and scale `128**-0.5`. See the pinned upstream input generator for the precise recipe. Other inputs are not validated or supported.

Output is BF16 and final state is FP32. Backward, fused preprocessing, other state layouts, arbitrary shapes, and multi-GPU context parallelism are unsupported. Use the established FLA operators for those calls. Call under inference mode.

The numerical policy includes BF16 staging, TF32-assisted inversion, and conditional small-transition truncation. Truncation is an approximation; no universal error guarantee is claimed for arbitrary input magnitudes. Official workload validation does not establish production model accuracy. The RFC must resolve this domain before any automatic backend integration.

## Runtime and packaging

Runtime requires PyTorch, compatible TVM FFI providing `cpp.load_inline(embed_cubin=...)`, a CUDA toolkit and driver supporting `sm_103a` PTX, a C++ compiler, and Ninja. The measured environment used PyTorch 2.14.1+cu130 and a CUDA 13.2 toolkit. Runtime compiles the host launch adapters, and the CUDA driver JIT-compiles the checked-in PTX to device code. No numerical CUDA source is built. Warm up all configurations before CUDA graph capture. Initialize and invoke this runner on one host thread and stream; shared PDL synchronization buffers are not designed for simultaneous streams or threads.

`manifest.json` checksums all 103 PTX modules and host adapters and records 123 launch bindings. Factory call ordinals are explicit, so failed initialization can be retried without advancing a global binding counter. Host adapter bytes are identical to the standalone export; PTX differs only in terminal whitespace normalization for contribution checks.

## Reproduce the official comparison

Check out [FlashQLA](https://github.com/QwenLM/FlashQLA) at `a14dbf3474514e1674526431e960325f72081e9f` and install its benchmark dependencies. The `--deps` directory contains that benchmark's TileLang and TVM FFI packages. Run from this FLA checkout on an allocated B300:

```bash
FLA_CI_ENV=0 FLA_DISABLE_BACKEND_DISPATCH=1 python -m benchmarks.ops.benchmark_qla_ptx \
  --baseline <flashqla-checkout> --deps <benchmark-dependencies> --output <results.json>
FLA_CI_ENV=0 FLA_DISABLE_BACKEND_DISPATCH=1 python -m pytest tests/ops/test_qla_ptx.py -q
```

The comparator verifies the pinned upstream script digest, calls its input preparation function, shares the exact input storage via CUDA IPC, validates both implementations against FLA's independent FP32 recurrent reference, and uses the same CUPTI concurrent-activity protocol on both implementations. The full GPU activity span of 16 complete calls is divided by 16; three samples are reduced by their median. Overlapping kernel times are not summed. Compilation, initialization, reference computation and input preparation are excluded. This is profiled steady-state time per call, not isolated latency; profiling can perturb execution. CUPTI GPU activities are required.

[RESULTS.md](RESULTS.md) contains all per-case timings and numerical errors. `benchmark_results.json` preserves the measured summary and all per-case sample spans/counts, with individual activity records omitted to keep the report compact. `provenance.json` identifies the original standalone source and timing protocol; its source hashes describe that export rather than the import-adapted FLA files.
