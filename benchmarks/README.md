# Benchmarks

Benchmarks are grouped by the workload they measure. Local runs and CI share operator definitions, timing, revision isolation, and result reporting.

| Workload            | Location   | Measurement                                                                  | Entry point                                                               |
| ------------------- | ---------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Operators           | `ops/`     | Forward and forward/backward latency across registered shapes                | `python -m benchmarks.ops.run`                                            |
| Modules             | `modules/` | Provider comparisons for normalization, convolution, activations, and losses | `python -m benchmarks.modules.benchmark_<module>`                         |
| Models              | `models/`  | Training throughput and prompt processing plus generation time               | `python -m benchmarks.models.benchmark_training` / `benchmark_generation` |
| Context parallelism | `cp/`      | Distributed operator and layer experiments                                   | `torchrun` with the selected script                                       |

## PR performance comparisons

The NVIDIA CI benchmark job selects affected registered operators from the source diff, measures the head and base revisions, and uploads `benchmark_results.json` with run metadata. Shared kernel or benchmark execution changes select all registered operators. The separate `benchmark-comment.yml` workflow reads these artifacts and updates the PR comment. It does not execute PR code. Accelerator correctness checks are managed by the calling CI workflow; the comparison command itself only measures performance.

Reproduce a CI comparison locally:

```bash
python -m benchmarks.compare --base origin/main --head HEAD --output benchmark_results.json
python -m benchmarks.compare --base origin/main --benchmark-ops chunk_gla chunk_kda --output benchmark_results.json
```

The default regression threshold is 5%. `--no-fail-on-regression` keeps regressions in the report while allowing a successful exit. Missing results or a failed worker still fail the comparison. `--plot <directory>` generates plots from the saved comparison JSON.

Both revisions use the benchmark definitions from the current checkout. Each runs in its own subprocess and detached worktree; the working branch and installed package are left intact. Named branches, commit SHAs, and refs such as `origin/main` are supported. Dependencies must already be installed for both revisions.

The Ascend benchmark workflow runs its configured operator, module, and graph workloads and uploads their results. It does not use the NVIDIA comment workflow.

## Local measurements

```bash
python -m benchmarks.ops.run --list
python -m benchmarks.ops.run --op chunk_gla --base origin/main
python -m benchmarks.ops.run --op chunk_gla --no-base --modes fwd
FLA_CI_ENV=0 python -m benchmarks.ops.verify --op chunk_gla --base origin/main
python -m benchmarks.modules.benchmark_layernorm
```

`ops.run` measures latency with Triton `do_bench`, using BF16 inputs and forward plus forward/backward modes by default. It warms all supported shapes before timing them. Per-operator shapes in the registry take precedence over the common sweep. `--backend` selects an explicitly registered backend override. `ops.verify` runs the selected pytest gate first and can collect a `torch.profiler` trace with `--profile`; use `--gate-k` only for a subset check, not as evidence for the complete test file.

Ascend graph timing is a separate execution mode with its own supported workloads:

```bash
python -m benchmarks.ops.run --graph --list
python -m benchmarks.ops.run --graph --op chunk_kda --base origin/main
```

Graph results report eager and replay p50/p95 latency. Replay includes live-input updates, while capture cost is reported separately. The baseline runs eager execution. This mode uses the existing graph-specific shapes and FP16 default; it does not share the regular runner's timing method.

Model measurements require CUDA and the corresponding optional dependencies:

```bash
pip install -e '.[cuda,benchmark]' accelerate
python -m benchmarks.models.benchmark_training --name kda --batch_size 2 --seq_len 8192
python -m benchmarks.models.benchmark_training --name kda --batch_size 2 --seq_len 8192 --varlen
python -m benchmarks.models.benchmark_generation --path fla-hub/gla-1.3B-100B
```

Training measures the existing input preparation, forward/backward, and optimizer loop. Generation measures prompt processing and decoding together. For distributed experiments, use each script's documented `torchrun` command and rank requirements; these workloads do not belong to the single-device operator sweep.

## Code organization

| Module                                                                   | Responsibility                                                            |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `discovery.py`                                                           | Map changed source files to registered operator benchmarks                |
| `ops/registry.py`                                                        | Declare operator functions, arguments, shapes, and correctness test paths |
| `ops/inputs.py`                                                          | Construct tensor and structured operator inputs                           |
| `ops/runner.py`                                                          | Warm and time registered operator workloads                               |
| `ops/run.py`, `ops/verify.py`                                            | Parse local commands and coordinate their execution                       |
| `ops/graph_registry.py`, `ops/graph_runner.py`, `ops/graph_benchmark.py` | Define, time, and coordinate Ascend graph workloads                       |
| `comparison.py`                                                          | Isolate benchmark workers and source revisions                            |
| `compare.py`                                                             | Coordinate affected-workload comparisons for CI and local use             |
| `environment.py`                                                         | Device synchronization and machine metadata                               |
| `reporting.py`, `visualize.py`                                           | Format comparison tables and render saved results                         |

To add an operator benchmark, register its `OpConfig` in `ops/registry.py`, reuse input specifications from `ops/inputs.py`, and provide custom shapes or input construction only where required. Keep timing and subprocess logic in the shared runner. Add host-side coverage under `tests/benchmarks/` when changing discovery, execution, or reporting.

See [CONTRIBUTING.md](../CONTRIBUTING.md#benchmarking) for correctness gates and performance evidence requirements.
