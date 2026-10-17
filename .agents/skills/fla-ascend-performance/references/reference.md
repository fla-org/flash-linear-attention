# Ascend profiling reference

Use this reference for collection options, CSV interpretation, and choosing the next optimization. Kernel correctness hazards are in [TRAPS.md](TRAPS.md); implementation examples are in [cases.md](cases.md).

## Collection and analysis

Run the scripts from the active NPU environment. `profile_npu.py` accepts a workload file or inline source defining `workload()`. It executes that callable repeatedly, so initialize reusable inputs outside the callable and make repeated forward/backward calls valid.

```bash
python .agents/skills/fla-ascend-performance/scripts/profile_npu.py \
  --name my_op --out-dir profile/my_op-npu \
  --metrics PipeUtilization --analyze --exec-file path/to/workload.py
```

For library use, add the scripts directory to `PYTHONPATH` and call `profile_callable`:

```bash
export PYTHONPATH="$PWD/.agents/skills/fla-ascend-performance/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

```python
from profile_npu import profile_callable

trace_dir = profile_callable(fn=workload, name="my_op", out_dir="profile/my_op-npu", aic_metrics="PipeUtilization")
```

The returned directory contains the profiler output:

```text
my_op_profiling_TIMESTAMP/
└── localhost..._ascend_pt/
    └── ASCEND_PROFILER_OUTPUT/
        ├── op_statistic.csv
        └── kernel_details.csv
```

Pass that exact run directory when analyzing an existing trace. The analyzer takes the first matching CSV under the supplied directory, so a parent containing several runs can select an older result.

```bash
python .agents/skills/fla-ascend-performance/scripts/analyze_profile.py \
  profile/my_op-npu/my_op_profiling_TIMESTAMP --kernel-filter my_kernel --top-k 20
```

`--kernel-filter` applies to kernel details. If no name matches, the analyzer displays the top kernels instead; confirm the printed names before attributing metrics to the target.

## Choosing metrics

Each profiling run collects one `aic_metrics` set. Use separate runs for complementary metrics and keep the workload unchanged.

| Metric set                                       | Use                                                                     |
| ------------------------------------------------ | ----------------------------------------------------------------------- |
| `PipeUtilization`                                | First pass: compare Cube, Vector, scalar, and memory-transfer activity. |
| `MemoryUB`                                       | Check UB read/write bandwidth when memory traffic may limit the kernel. |
| `Memory`, `MemoryAccess`, `MemoryL0`             | Inspect other memory paths.                                             |
| `L2Cache`                                        | Investigate cache reuse.                                                |
| `ArithmeticUtilization`, `ResourceConflictRatio` | Investigate arithmetic occupancy or resource conflicts.                 |

Available metric names depend on the installed `torch_npu`; the collector reports available names when an option is unknown.

## Diagnosis

Start with total duration in `op_statistic.csv`, then inspect the dominant kernel in `kernel_details.csv`. Pipe activity alone does not establish the bottleneck.

| Evidence                                               | Check next                                                                                                   |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Target kernel absent or many small fallback operations | Backend selection and fusion.                                                                                |
| High Vector or Cube activity                           | Tile shape, useful work per tile, and scalar overhead around compute.                                        |
| High MTE activity                                      | Reuse, intermediate writebacks, and strided loads; inspect [gate loading](g-contiguous-loading.md).          |
| High scalar activity                                   | Address generation, branches, and repeated per-tile setup.                                                   |
| High UB bandwidth with low compute activity            | Live buffers, layout conversions, and repeated UB reads/writes.                                              |
| Low activity despite UB overflow at larger tiles       | Mutually exclusive DMA paths may remain live; see [DMA splitting](TRAPS.md#dma-paths-and-optional-pointers). |
| Many host launches for grid chunks                     | Consider a 1D task loop with the appropriate Cube or Vector core count.                                      |

## UB planning

Estimate peak live storage from simultaneous buffers, their dtype, and any compiler-created copies. Use `fla.utils.ascend_ub_manager` for device capacity and tiling rather than hard-coding the UB size. The helper's safety margin reserves capacity; it is separate from the multiplier that estimates live buffers.

If more capacity is available, test a larger tile and verify compilation and numerics. If the live set already fills the budget, improve reuse or split stages. Model forward and backward separately; inline gate computation can need more UB than a precomputed-gate path.

Useful code locations:

- `fla/utils/ascend_ub_manager.py`: tiling and grid helpers.
- `fla/utils/hardware.py`: `get_multiprocessor_count`, including `use_aicore=True` for Cube work.
- `fla/modules/norm/triton_ascend/l2norm.py`: entry-point registration and the Ascend implementation.
- `tests/ops/test_gdn_kernels.py`, `tests/ops/test_solve_tril.py`, and `tests/modules/test_conv.py`: relevant kernel comparisons.
