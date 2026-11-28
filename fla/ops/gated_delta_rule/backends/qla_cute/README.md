# Experimental B300 CuTe DSL forward runner

This explicitly imported runner implements the 171 official FlashQLA configurations in `benchmark_shared.json`, using the public NVIDIA CuTe DSL 4.8.0 release. It is not registered with FLA dispatch and does not change existing operator, training, layer, or model paths.

```python
from fla.ops.gated_delta_rule.backends.qla_cute import run

output, final_state = run(q, k, v, g, beta, scale, initial_state, cu_seqlens)
```

## Requirements and supported inputs

Install the public release with `pip install 'nvidia-cutlass-dsl[cu13]==4.8.0'`. The runner uses the publicly distributed `cutlass.experimental.primitives.nvvm_wrapper` helpers; their experimental API is the reason for the version pin. It requires PyTorch, CUDA Python bindings, and a B300 with 148 SMs. Compilation and initialization must complete before graph capture. Warm up every configuration before timing.

The validated input domain is the checked-in official benchmark: batch size 1; contiguous BF16 Q/K/V; 128-dimensional keys and values; precomputed FP32 log decays and sigmoid beta; non-null FP32 K-first initial states; and int32 packed sequence offsets. All inputs are on the same GPU. The official recipe uses normalized Q/K, `logsigmoid(randn)/16` decay with 25% zero-decay heads, standard-normal values and initial states, and scale `128**-0.5`. Other inputs are not validated or supported. Run under inference mode on one host thread and stream; backward and multi-GPU context parallelism are unsupported.

The source includes BF16 intermediates, tensor-core recurrence, and conditional transition truncation at a decay threshold of approximately `2**-34` when its contraction predicate holds. Truncation is an approximation; passing these inputs is not a universal error guarantee for arbitrary input magnitudes. No existing production backend is replaced.

## Audit and measurements

Only the five source modules reachable from the selected forward entry are included. Development launch-skipping hooks and unused variants are omitted. `provenance.json` records source digests and the public distribution version; public-installation and source-equivalence audit results accompany the measured results.

Run from this FLA checkout on an allocated B300, with FlashQLA checked out at `a14dbf3474514e1674526431e960325f72081e9f` and its benchmark dependencies installed:

```bash
FLA_CI_ENV=0 FLA_DISABLE_BACKEND_DISPATCH=1 python -m pytest tests/ops/test_qla_cute.py -q
FLA_CI_ENV=0 FLA_DISABLE_BACKEND_DISPATCH=1 python -m benchmarks.ops.benchmark_qla_cute \
  --baseline <flashqla-checkout> --deps <benchmark-dependencies> --output <results.json>
```

The comparison calls the pinned upstream input generator and shares exact input storage through CUDA IPC. Both implementations are checked independently against FLA's FP32 recurrent reference at relative L2 <= 0.005 for outputs and final states. CUPTI concurrent GPU activity tracing measures the complete span of 16 calls per graph; the median of three samples is reported. Overlapping kernel durations are not summed. Compilation, initialization, input preparation and reference evaluation are excluded. This is profiled steady-state GPU time per call; CPU dispatch and model throughput are outside the measurement, and profiling may perturb execution.
