# Benchmarks

Choose a measurement task, run it, and summarize its JSON results. Task names describe the measurement directly; `--list` shows the available tasks and their execution requirements without importing accelerator or model dependencies.

```bash
python -m benchmarks.run --list
python -m benchmarks.run operator_latency --help
python -m benchmarks.run operator_latency --op chunk_gla chunk_kda --json results/operators.json
python -m benchmarks.summarize results/*.json --output summary.md
```

## Choose a task

| Task                    | Measurement                                       | Selection                         |
| ----------------------- | ------------------------------------------------- | --------------------------------- |
| `operator_latency`      | Operator forward and forward/backward latency     | `--op chunk_gla chunk_kda`        |
| `*_latency`             | Component provider sweeps                         | See each task's `--help`          |
| `training`              | Complete model training throughput                | `--name kda`                      |
| `generation`            | Prompt processing and generation throughput       | `--path <checkpoint>`             |
| `kda_graph`             | Eager execution versus Ascend graph replay        | `--engines`, `--custom-shapes`    |
| `operator_cp`           | GDN/KDA operator context-parallel latency         | `--op chunk_gdn` or `chunk_kda`   |
| `layer_cp`              | GDN/KDA layer context-parallel latency            | `--layer gated_deltanet` or `kda` |
| `conv_cp`               | Context-parallel convolution latency              | `--tokens`, `--channels`          |
| `kda_cp_tp`             | KDA context/tensor-parallel configurations        | `--config cp8` or `cp2tp`         |

`activations_latency`, `conv_latency`, `cross_entropy_latency`, `l2norm_latency`, `layernorm_latency`, and `tokenshift_latency` compare component providers using their existing shape sweeps. Operator timing uses BF16 inputs and forward plus forward/backward modes by default; operator-specific shapes take precedence over the common sweep. Graph timing requires Ascend and records eager/replay latency, with capture time separate from replay.

```bash
python -m benchmarks.run conv_latency --json results/conv.json --save-dir conv_plots --plot
python -m benchmarks.run kda_graph --json results/kda_graph.json
python -m benchmarks.run training --name kda --batch_size 2 --seq_len 8192 --json results/training.json
python -m benchmarks.run generation --path fla-hub/gla-1.3B-100B --json results/generation.json
```

Training and generation require CUDA and optional dependencies (`pip install -e '.[cuda,benchmark]' accelerate`). Training includes input preparation, forward/backward, and optimization; generation includes prompt processing and decoding. Add `--varlen` to training for packed input measurements.

Distributed tasks run under `torchrun`:

```bash
torchrun --nproc_per_node=4 --module benchmarks.run operator_cp --op chunk_kda --backward --json results/kda_cp.json
torchrun --nproc_per_node=4 --module benchmarks.run layer_cp --layer kda --backward --json results/kda_layer_cp.json
torchrun --nproc_per_node=4 --module benchmarks.run conv_cp --json results/conv_cp.json
torchrun --nproc_per_node=8 --module benchmarks.run kda_cp_tp --config cp8 --backward --json results/kda_cp_tp.json
```

Rank 0 writes the requested filename; other ranks add `.rank<N>` before the extension. Each file records its rank and world size. Process groups are released after measurement.

The runner accepts quoted task patterns. `python -m benchmarks.run '*_latency' --list` lists latency tasks; remove `--list` to run them. Arguments must apply to every selected task and are checked before any measurements begin. A task producing no measurements fails.

## Summarize results

```bash
python -m benchmarks.summarize results/*.json --output summary.md
python -m benchmarks.summarize results/*.json --output summary.csv
python -m benchmarks.summarize results/*.json --benchmark operator_latency --name '*kda' --all-metrics
```

Markdown groups results by measurement task so cases from several files appear in the same table. A source table identifies each input file, revision, hardware, and rank. The default view shows each case's primary metric; `--all-metrics` includes secondary measurements such as capture time and peak memory. CSV uses the same selection, with one row per case and metric. `--benchmark` and `--name` accept quoted patterns. Repeated input paths are deduplicated; an empty selection fails.

Revision-comparison artifacts appear with base/head values, percentage change, and threshold status. Positive change denotes a regression for both latency and throughput. Comparisons use the declared primary metric. Shapes, units, and ranks remain separate; summaries do not average unrelated measurements.

## Compare revisions

```bash
python -m benchmarks.compare --benchmark operator_latency --base origin/main --output results/comparison.json -- --op chunk_kda
python -m benchmarks.compare --benchmark conv_latency --base origin/main --output results/conv_comparison.json
python -m benchmarks.compare --benchmark training --base origin/main --output results/training_comparison.json -- --name kda --steps 32
```

Task arguments follow `--`. The default head is committed `HEAD`; `run` measures the working tree, including uncommitted edits. Comparison workers use the current benchmark definitions with isolated source revisions and a temporary Triton cache per worker. Named refs and commit SHAs are supported; dependencies must be installed for both revisions. Distributed comparisons also use `torchrun` and save one artifact per rank.

The default regression threshold is 5%. Each result declares its primary metric and comparison direction. `--no-fail-on-regression` permits a successful exit after a measured regression; failed workers, missing cases, and invalid measurements still fail.

## CI and correctness

Without `--benchmark`, comparison selects affected operators from the source diff:

```bash
python -m benchmarks.compare --base origin/main --head HEAD --output benchmark_results.json
python -m benchmarks.compare --base origin/main --benchmark-ops chunk_gla chunk_kda --output benchmark_results.json
FLA_CI_ENV=0 python -m benchmarks.verify --op chunk_gla --base origin/main --profile
```

Shared kernel, operator configuration, or execution changes select all registered operators. NVIDIA CI uses this selection and the common runner. The resulting artifact is consumed by `benchmark-comment.yml`, which updates the PR comment without executing PR code. Default CI comparisons also support `--output <file> --plot <directory>`. Ascend CI runs its configured operator, convolution, and graph tasks through the same runner. Both workflows publish a Markdown job summary and upload it alongside their JSON artifacts.

Benchmarking does not replace correctness tests. `verify` runs the selected operator's pytest gate before timing and can collect a profiler trace. Use `--gate-k` only for a subset check, not as evidence for a complete test file. Calling CI workflows also run accelerator correctness checks.

## Code organization

```text
benchmarks/
├── run.py             # execute measurement tasks
├── compare.py         # compare source revisions and emit CI results
├── summarize.py       # filter and tabulate saved measurements
├── verify.py          # correctness gate and optional profiling
├── visualize.py       # plot CI operator comparisons
├── workloads/         # one complete measurement task per file
│   ├── operator_latency.py
│   ├── conv_latency.py
│   ├── training.py
│   ├── generation.py
│   ├── kda_graph.py
│   ├── operator_cp.py
│   └── ...
├── configs/
│   └── operators.py   # operator specifications and shape presets
└── utils/             # input builders, timers, adapters, revision workers, reports
```

A workload defines `add_arguments(parser)` and `run(args)` and returns `BenchmarkResult` records. It owns input preparation, the measured computation, and timing boundaries. Add its description and execution requirements to `workloads/__init__.py`; the catalog is also the CLI task list. The runner owns dispatch, machine metadata, and result files.

New variants using the existing operator latency method need only an entry in `configs/operators.py`. Add a workload when the measurement itself differs. Shared tensor input construction lives in `utils/inputs.py`; report conversion, graph timing, distributed helpers, and revision isolation have separate modules. The layer adapters are shared by the GDN/KDA context-parallel measurements.

Keep host coverage for dispatch, reports, revision isolation, and affected-file discovery in `tests/benchmarks/`. Follow [CONTRIBUTING.md](../CONTRIBUTING.md#benchmarking) for accelerator correctness and performance evidence.
