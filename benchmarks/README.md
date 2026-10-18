# Benchmarks

Each workload directory contains executable `benchmark_*.py` modules with a `main()` entry point. Run them with `python -m benchmarks.<category>.benchmark_<workload>`; distributed workloads use `torchrun --module`. Local runs and CI share operator definitions, timing, revision isolation, and reporting.

```text
benchmarks/
├── compare.py                  # PR base/head comparison
├── verify.py                   # correctness gate and optional operator profile
├── visualize.py                # plots from saved results
├── ops/
│   ├── benchmark_ops.py         # registered operator latency sweep
│   ├── benchmark_graph.py      # Ascend eager and graph replay latency
│   └── utils/                  # operator definitions, inputs, discovery, and timing
├── modules/
│   └── benchmark_<module>.py   # convolution, norms, activations, and losses
├── models/
│   ├── benchmark_training.py
│   └── benchmark_generation.py
├── cp/
│   ├── benchmark_conv.py
│   ├── benchmark_gdn.py
│   └── benchmark_kda.py
└── utils/                      # shared revision workers, environment, and reporting
```

| Workload            | Location   | Measurement                                                                  | Entry point                                                               |
| ------------------- | ---------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Operators           | `ops/`     | Forward and forward/backward latency across registered shapes                | `python -m benchmarks.ops.benchmark_ops`                                  |
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
python -m benchmarks.ops.benchmark_ops --list
python -m benchmarks.ops.benchmark_ops --op chunk_gla --base origin/main
python -m benchmarks.ops.benchmark_ops --op chunk_gla --no-base --modes fwd
FLA_CI_ENV=0 python -m benchmarks.verify --op chunk_gla --base origin/main
python -m benchmarks.modules.benchmark_layernorm
```

`ops.benchmark_ops` measures latency with Triton `do_bench`, using BF16 inputs and forward plus forward/backward modes by default. It warms all supported shapes before timing them. Per-operator shapes in the registry take precedence over the common sweep. `--backend` selects an explicitly registered backend override. `verify` runs the selected pytest gate first and can collect a `torch.profiler` trace with `--profile`; use `--gate-k` only for a subset check, not as evidence for the complete test file.

Ascend graph timing has its own entry point and supported workloads:

```bash
python -m benchmarks.ops.benchmark_graph --list
python -m benchmarks.ops.benchmark_graph --op chunk_kda --base origin/main
```

Graph results report eager and replay p50/p95 latency. Replay includes live-input updates, while capture cost is reported separately. The baseline runs eager execution. This mode uses the existing graph-specific shapes and FP16 default; it does not share the regular runner's timing method.

Model measurements require CUDA and the corresponding optional dependencies:

```bash
pip install -e '.[cuda,benchmark]' accelerate
python -m benchmarks.models.benchmark_training --name kda --batch_size 2 --seq_len 8192
python -m benchmarks.models.benchmark_training --name kda --batch_size 2 --seq_len 8192 --varlen
python -m benchmarks.models.benchmark_generation --path fla-hub/gla-1.3B-100B
```

Training measures the existing input preparation, forward/backward, and optimizer loop. Generation measures prompt processing and decoding together. Distributed experiments use the same module naming convention, with the required ranks:

```bash
torchrun --nproc_per_node=4 --module benchmarks.cp.benchmark_conv --cp
torchrun --nproc_per_node=4 --module benchmarks.cp.benchmark_gdn --ops --backward --bench
torchrun --nproc_per_node=8 --module benchmarks.cp.benchmark_kda --config cp8 --seqlen 131072 --backward
```

These workloads measure communication as well as computation and run independently of the single-device operator sweep.

## Shared implementation

Keep a self-contained benchmark in one file. Helpers shared by operator commands belong in `ops/utils/`; helpers shared across workload categories belong in `utils/`. Add another utility package only when there is actual shared code.

| Module                                                 | Responsibility                                             |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| `ops/utils/registry.py`, `ops/utils/graph_registry.py` | Declare workloads, shapes, and supported modes             |
| `ops/utils/inputs.py`                                  | Construct tensor and structured operator inputs            |
| `ops/utils/runner.py`                                  | Warm and time registered operators                         |
| `ops/utils/discovery.py`                               | Map changed source files to registered operator benchmarks |
| `utils/revisions.py`                                   | Isolate benchmark workers and source revisions             |
| `utils/environment.py`                                 | Synchronize devices and collect machine metadata           |
| `utils/reporting.py`                                   | Format operator, graph, and CI comparison tables           |
| `utils/perf_report.py`                                 | Run Triton module reports and save plots                   |

To add an operator to the latency sweep, register its `OpConfig` in `ops/utils/registry.py`, reuse input specifications from `ops/utils/inputs.py`, and provide custom shapes or input construction only where required. Module, model, and distributed benchmarks remain self-contained when their measurements differ from the operator sweep. Add host-side coverage under `tests/benchmarks/` when changing discovery, execution, or reporting.

See [CONTRIBUTING.md](../CONTRIBUTING.md#benchmarking) for correctness gates and performance evidence requirements.
