# Benchmarks

All workloads use one execution interface. A workload defines `add_arguments(parser)` and `run(args)`, which returns `BenchmarkResult` records. The common runner parses the command, runs the measurement, records the environment, and writes results. Workload modules contain no standalone command entry points, Git operations, or result-file writers.

```text
benchmarks/
├── run.py             # execute a workload
├── compare.py         # compare revisions and produce the CI artifact
├── verify.py          # operator correctness gate and optional profiling
├── visualize.py       # plot saved results
├── ops/               # latency.py, graph.py
├── modules/           # conv.py, layernorm.py, activations.py, ...
├── models/            # training.py, generation.py
├── cp/                # conv.py, gdn.py, kda.py
└── utils/             # shared execution, results, and workload helpers
```

The four workload directories have the same role and interface. Timing stays with the measured workload: Triton latency, graph replay, complete model steps, and distributed communication need different measurement boundaries.

## Run a workload

```bash
python -m benchmarks.run --list
python -m benchmarks.run ops.latency --help
python -m benchmarks.run ops.latency --op chunk_gla --json latency.json
python -m benchmarks.run ops.graph --op chunk_kda --json graph.json
python -m benchmarks.run modules.conv --json conv.json --save-dir conv_benchmark --plot
python -m benchmarks.run models.training --name kda --batch_size 2 --seq_len 8192 --json training.json
python -m benchmarks.run models.generation --path fla-hub/gla-1.3B-100B --json generation.json
```

Operator latency uses Triton `do_bench`, BF16 inputs, and forward plus forward/backward modes by default. Per-operator shapes take precedence over the common sweep. Graph latency requires Ascend and records eager and replay p50/p95, with capture time separate from replay. Module sweeps retain their Triton provider definitions and optionally save CSV and plots. Model workloads require CUDA and the optional benchmark dependencies (`pip install -e '.[cuda,benchmark]' accelerate`). Training includes input preparation, forward/backward, and optimization; generation includes prompt processing and decoding.

Distributed workloads use the same runner under `torchrun`:

```bash
torchrun --nproc_per_node=4 --module benchmarks.run cp.conv --cp --json conv_cp.json
torchrun --nproc_per_node=4 --module benchmarks.run cp.gdn --ops --backward --json gdn_cp.json
torchrun --nproc_per_node=8 --module benchmarks.run cp.kda --config cp8 --seqlen 131072 --backward --json kda_cp.json
```

Rank 0 writes the requested filename; other ranks add `.rank<N>` before the extension. Each file records its rank and world size. CP workloads initialize and release their process groups through the shared distributed context.

## Compare revisions

Any workload can run through the revision comparison command. Workload arguments follow `--`:

```bash
python -m benchmarks.compare --benchmark ops.latency --base origin/main -- --op chunk_gla
python -m benchmarks.compare --benchmark modules.conv --base origin/main --output conv_comparison.json
python -m benchmarks.compare --benchmark models.training --base origin/main -- --name kda --steps 32
```

The default head is the `HEAD` commit. Use `run` to measure uncommitted local changes. Comparison workers run the current benchmark definitions against the selected source revisions in separate processes and detached worktrees, with a temporary Triton cache per worker. The installed FLA package and existing caches stay untouched. Named refs and commit SHAs are supported; dependencies must already be installed for both revisions. Distributed comparisons also run under `torchrun` and save results per rank.

The default regression threshold is 5%. Each result declares its primary metric and whether higher values are better, so latency and throughput use the correct direction. `--no-fail-on-regression` allows a successful exit after a measured regression; failed workers, missing cases, and invalid measurements still fail. Comparison plots currently support operator latency with `--output <file> --plot <directory>`.

## CI and correctness

Without `--benchmark`, the comparison command selects affected registered operators from the source diff:

```bash
python -m benchmarks.compare --base origin/main --head HEAD --output benchmark_results.json
python -m benchmarks.compare --base origin/main --benchmark-ops chunk_gla chunk_kda --output benchmark_results.json
FLA_CI_ENV=0 python -m benchmarks.verify --op chunk_gla --base origin/main
```

NVIDIA CI uses this operator selection and the common runner. Shared kernel or execution changes select all registered operators. The comparison command formats the operator artifact expected by the separate `benchmark-comment.yml` workflow, which updates the PR comment without executing PR code. Ascend CI runs its configured operator, module, and graph workloads through the same runner and uploads their results.

Benchmarking does not replace correctness tests. `verify` runs the selected operator's pytest gate before timing and can collect a trace with `--profile`. Use `--gate-k` only for a subset check, not as evidence for a complete test file. Accelerator correctness checks also run in the calling CI workflows.

## Add or change workloads

Add a module to the appropriate workload directory with `add_arguments(parser)` and `run(args)`. It becomes available through `run --list`; no command wrapper or registration in the runner is needed. Return one `BenchmarkResult` per measured case, with a name, parameter dictionary, metrics with units in their names, and an explicit primary metric. The common output envelope contains the workload name, machine information, and result records.

Reuse `utils/perf_report.py` for Triton provider sweeps, `utils/distributed.py` for process groups, and `utils/reporting.py` for the result contract. Operator registrations, input builders, timing, and affected-file discovery live in `utils/ops/` because they are shared by the latency workload, correctness driver, and CI selection. Keep other workload-specific code with its workload instead of adding unused abstraction layers.

Add host-side coverage under `tests/benchmarks/` when changing command dispatch, result handling, revision isolation, or discovery. See [CONTRIBUTING.md](../CONTRIBUTING.md#benchmarking) for correctness gates and performance evidence requirements.
