# Issue #872: Reusable `final_state` Buffer

## Summary

`fused_recurrent_gated_delta_rule` now accepts an optional preallocated `final_state` buffer. When provided, the recurrent kernel writes the final state in-place and returns the same Tensor. Existing calls without a buffer keep the original allocation behavior.

## Correctness

Hardware: NVIDIA B200

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/ops/test_gdn.py
```

Result:

```text
80 passed, 20 skipped
```

The added coverage includes dense inputs, V-first state layout, variable-length inputs, repeated buffer reuse, initial/final state aliasing, NaN write coverage, and invalid buffer validation.

Dependent layer/model test command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/ops/test_gdn.py \
  tests/layers/test_gated_deltanet.py \
  tests/layers/test_layer_cache_layer_idx.py \
  tests/models/test_modeling_gated_deltanet.py \
  tests/models/test_modeling_gated_deltaproduct.py \
  tests/models/test_modeling_mom.py \
  tests/models/test_modeling_yoco.py
```

Result:

```text
130 passed, 29 skipped, 22 warnings
```

The warnings are existing TorchScript deprecations and layer configuration warnings; they are unrelated to the reusable buffer change.

## Performance methodology

- GPU: NVIDIA B200
- dtype: `bfloat16`
- mode: fused recurrent GDN
- timing: CUDA events, median of repeated rounds
- baseline: allocate `final_state` inside each call
- buffered: reuse a preallocated `final_state`

The single-call benchmark is implemented in `benchmark_fused_recurrent_final_state.py`. The autoregressive benchmark feeds each step's final state into the next step and is implemented in `benchmark_fused_recurrent_final_state_autoregressive.py`.

## Single-call decode results

Workload: `T=1, H=32, HV=32, K=128, V=256, dtype=bfloat16`, NVIDIA B200.

| B | Baseline (ms) | Buffered (ms) | Speedup |
|---:|---:|---:|---:|
| 1 | 0.074139 | 0.071357 | 1.0390x |
| 2 | 0.073129 | 0.071429 | 1.0238x |
| 4 | 0.073106 | 0.071662 | 1.0201x |
| 8 | 0.073223 | 0.070268 | 1.0421x |
| 16 | 0.073739 | 0.071307 | 1.0341x |
| 32 | 0.073152 | 0.071405 | 1.0245x |

## Autoregressive results

Workload: `B=8, T=1, H=32, HV=32, K=128, V=256, dtype=bfloat16`, NVIDIA B200.

| Steps | Baseline total (ms) | Buffered total (ms) | Speedup |
|---:|---:|---:|---:|
| 1,000, run 1 | 77.253777 | 74.325905 | 1.0394x |
| 1,000, run 2 | 75.828991 | 73.272305 | 1.0349x |

## Final B-sweep

Command:

```bash
for B in 1 2 4 8 16 32; do
  CUDA_VISIBLE_DEVICES=0 \
  python benchmarks/ops/benchmark_fused_recurrent_final_state_autoregressive.py \
    --B $B --H 32 --HV 32 --K 128 --V 256 \
    --dtype bfloat16 --steps 4096 --warmup 20 --rounds 5
done
```

Result: 4096 autoregressive steps, five timing rounds per configuration.

| B | Baseline ms/step | Buffered ms/step | Speedup |
|---:|---:|---:|---:|
| 1 | 0.075271 | 0.072985 | 1.0313x |
| 2 | 0.073729 | 0.070771 | 1.0418x |
| 4 | 0.074615 | 0.072983 | 1.0224x |
| 8 | 0.073829 | 0.072487 | 1.0185x |
| 16 | 0.074288 | 0.072566 | 1.0237x |
| 32 | 0.079126 | 0.077348 | 1.0230x |

The speedup range is `1.0185x` to `1.0418x`, with an equal-weight geometric mean of approximately `1.0268x` across the six batch sizes.

## Conclusion

The correctness suite passes, and autoregressive decode shows a consistent approximately 2–4% latency improvement across batch sizes from 1 to 32 on NVIDIA B200. Long prefill workloads are expected to show little benefit because the recurrent computation dominates the one-time output-buffer allocation.

## PR summary

### Summary

Add an optional reusable `final_state` output buffer to fused recurrent GDN. The buffer is validated for shape, dtype, device, contiguity, and gradient safety, then written in-place by the existing recurrent kernel. Calls that do not provide a buffer remain backward compatible.

### Test plan

- `pytest -q tests/ops/test_gdn.py`: 80 passed, 20 skipped on NVIDIA B200.
- Added dense, V-first, varlen, repeated reuse, aliasing, NaN write, and validation coverage.
- Dependent layer/model tests should be run before opening the PR.

### Benchmark

- Hardware: NVIDIA B200.
- Workload: autoregressive `T=1`, `B=1/2/4/8/16/32`, `H=HV=32`, `K=128`, `V=256`, `bfloat16`.
- 4096-step autoregressive speedup: `1.0185x`–`1.0418x`, geometric mean approximately `1.0268x`.
- Conclusion: consistent 2–4% decode improvement; no material benefit is expected for long prefill.

No NCU profile was collected because the Triton kernel computation and launch configuration are unchanged; this change targets output-buffer allocation and reuse.
