# Benchmarks

Benchmarks are organized by the object being measured, matching `fla/`. Each object owns its input definitions and supported execution scenarios. Shared timing, revision isolation, and reporting live in `utils/`.

```text
benchmarks/
├── run.py             # measure selected objects
├── compare.py         # compare revisions and produce the CI artifact
├── summarize.py       # summarize saved JSON as Markdown or CSV
├── verify.py          # operator correctness gate and optional profiling
├── visualize.py       # plot saved operator comparisons
├── ops/               # kda.py, gla.py, gated_delta_rule.py, ...
├── modules/           # conv.py, layernorm.py, activations.py, ...
├── layers/            # kda.py, gated_deltanet.py
├── models/            # causal_lm.py
└── utils/             # shared measurement and reporting
```

## Measure

```bash
python -m benchmarks.run --list
python -m benchmarks.run ops.kda --help
python -m benchmarks.run ops.kda --json results/kda.json
python -m benchmarks.run 'ops.*' --op chunk_gla chunk_kda --json results/operators.json
python -m benchmarks.run modules.conv --json results/conv.json --save-dir conv_benchmark --plot
python -m benchmarks.run models.causal_lm --scenario training --name kda --batch_size 2 --seq_len 8192 --json results/training.json
python -m benchmarks.run models.causal_lm --scenario generation --path fla-hub/gla-1.3B-100B --json results/generation.json
```

The selector accepts quoted shell patterns. `--scenario` chooses how to measure the selected object; omitted, it uses the first scenario in the table below. Patterns skip objects that do not support the requested scenario. Use `--help` with an object and scenario to see its arguments. A selection that produces no measurements fails.

| Object | Scenarios |
| --- | --- |
| `ops.*` | `latency` |
| `ops.gated_delta_rule` | `latency`, `cp` |
| `ops.kda` | `latency`, `graph`, `cp`, `cp_tp` |
| `modules.*` | `latency` |
| `modules.conv` | `latency`, `cp` |
| `layers.gated_deltanet`, `layers.kda` | `cp` |
| `models.causal_lm` | `training`, `generation` |

Operator latency uses Triton `do_bench`, BF16 inputs, and forward plus forward/backward modes by default. Per-operator shapes take precedence over the common sweep. Module sweeps retain their Triton provider definitions and optionally save CSV and plots. Model measurements require CUDA and the optional benchmark dependencies (`pip install -e '.[cuda,benchmark]' accelerate`). Training includes input preparation, forward/backward, and optimization; generation includes prompt processing and decoding.

KDA graph measurements require Ascend and record eager and replay latency, with capture time separate from replay:

```bash
python -m benchmarks.run ops.kda --scenario graph --json results/kda_graph.json
```

Distributed scenarios use the same runner under `torchrun`. `cp` measures context parallelism; KDA's `cp_tp` compares context/tensor-parallel configurations:

```bash
torchrun --nproc_per_node=4 --module benchmarks.run modules.conv --scenario cp --json results/conv_cp.json
torchrun --nproc_per_node=4 --module benchmarks.run ops.gated_delta_rule --scenario cp --backward --json results/gdn_cp.json
torchrun --nproc_per_node=4 --module benchmarks.run layers.kda --scenario cp --backward --json results/kda_layer_cp.json
torchrun --nproc_per_node=8 --module benchmarks.run ops.kda --scenario cp_tp --config cp8 --seqlen 131072 --backward --json results/kda_cp_tp.json
```

Rank 0 writes the requested filename; other ranks add `.rank<N>` before the extension. Each file records its rank and world size. Distributed scenarios initialize and release their process groups through the shared distributed context.

## Summarize

```bash
python -m benchmarks.summarize results/*.json
python -m benchmarks.summarize results/*.json --output summary.md
python -m benchmarks.summarize results/*.json --output summary.csv
```

Summaries accept both measurement and revision-comparison JSON. Markdown groups tables by source file and shows revision, hardware, and rank. Measurement tables retain every metric and case parameter. Comparison tables show the primary metric, base/head values, percentage change, and threshold status; positive change denotes a regression for both latency and throughput. CSV uses one row per case and metric, with source and machine columns for filtering. Different shapes, units, and ranks are kept separate.

## Compare revisions

Object arguments follow `--`:

```bash
python -m benchmarks.compare --benchmark ops.kda --base origin/main --output results/kda_comparison.json
python -m benchmarks.compare --benchmark modules.conv --base origin/main --output results/conv_comparison.json
python -m benchmarks.compare --benchmark models.causal_lm --base origin/main --output results/training_comparison.json -- --scenario training --name kda --steps 32
```

The default head is the `HEAD` commit. Use `run` to measure uncommitted local changes. Comparison workers run the current benchmark definitions against the selected source revisions in separate processes and detached worktrees, with a temporary Triton cache per worker. The installed FLA package and existing caches stay untouched. Named refs and commit SHAs are supported; dependencies must already be installed for both revisions. Distributed comparisons also run under `torchrun` and save results per rank.

The default regression threshold is 5%. Each result declares its primary metric and whether higher values are better, so latency and throughput use the correct direction. `--no-fail-on-regression` allows a successful exit after a measured regression; failed workers, missing cases, and invalid measurements still fail. Default CI operator comparisons also support `--output <file> --plot <directory>`.

## CI and correctness

Without `--benchmark`, the comparison command selects affected registered operators from the source diff:

```bash
python -m benchmarks.compare --base origin/main --head HEAD --output benchmark_results.json
python -m benchmarks.compare --base origin/main --benchmark-ops chunk_gla chunk_kda --output benchmark_results.json
FLA_CI_ENV=0 python -m benchmarks.verify --op chunk_gla --base origin/main
```

NVIDIA CI uses this selection and the common runner. Editing an operator's benchmark definition selects that operator family; shared kernel or execution changes select all registered operators. The comparison command produces the artifact expected by the separate `benchmark-comment.yml` workflow, which updates the PR comment without executing PR code. Ascend CI runs its configured operator, module, and graph scenarios through the same runner and uploads their results.

Benchmarking does not replace correctness tests. `verify` runs the selected operator's pytest gate before timing and can collect a trace with `--profile`. Use `--gate-k` only for a subset check, not as evidence for a complete test file. Accelerator correctness checks also run in the calling CI workflows.

## Add a benchmark

Add the object to its corresponding directory. Define `SCENARIOS`, `add_arguments(parser, scenario)`, and `run(args)`. The runner discovers objects by filename. Return one `BenchmarkResult` per measured case, with a name, parameter dictionary, metrics with units in their names, and an explicit primary metric. The runner adds the object and scenario to each record and records the environment once per artifact.

Operator latency definitions belong in the object's `OPERATORS` tuple; `utils/ops/registry.py` discovers them. Reuse `utils/ops/runner.py` for operator timing, `utils/perf_report.py` for Triton module sweeps, `utils/graph.py` for graph measurement, and `utils/distributed.py` for process groups. Keep object-specific inputs and adapters with the object.

Add host-side coverage under `tests/benchmarks/` when changing dispatch, reporting, revision isolation, or discovery. See [CONTRIBUTING.md](../CONTRIBUTING.md#benchmarking) for correctness gates and performance evidence requirements.
