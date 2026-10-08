# Skills

Add reusable workflows as:

```text
.agents/skills/<skill-name>/SKILL.md
```

Each skill should be self-contained and task-specific.
Include YAML frontmatter with `name` and `description`.
If a skill needs reference files, place them in a `references/` subdirectory inside the skill folder.
Symlinks to public repo docs are allowed when the referenced file is already tracked in this repository.

Do not add a `README.md` inside individual skill directories; the canonical entry point is `SKILL.md`.

## Choosing skills

Read `CONTRIBUTING.md` for contribution policy and `AGENTS.md` for agent operations and Git safety. The PR template defines the submission fields. Skills provide task-specific procedures; they do not replace those sources or require loading every skill for every task.

For kernel work, choose the relevant steps in this order:

| Stage                       | When needed                                                | Skill and responsibility                                                                                  |
| --------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Contract                    | Designing a kernel, backend route, or numerical change     | `fla-design-coverage`: supported cells, numerical budgets, compatibility, and production workloads        |
| Coverage                    | Adding or changing kernels and their tests                 | `fla-correctness-coverage`: turn the contract into reachable test cases and select dependent tests        |
| Operator                    | Changing KDA gates, kernels, or tests                      | `fla-kda`: operator-specific contracts and code map                                                       |
| Dispatch                    | Changing backend registration, verifiers, or selection     | `fla-dispatch-backends`: routing mechanics, rejection, and fallback tests                                 |
| Hardware and implementation | Profiling or optimizing a target backend; porting to Gluon | `fla-nvidia-performance` or `fla-ascend-performance`; add `fla-triton-to-gluon` for a Gluon port          |
| Optimization loop           | Iterating on kernel performance                            | `fla-optimization-loop`: freeze the baseline and correctness gate, then measure and record each candidate |
| Submission                  | Preparing any PR, including documentation or CI changes    | `fla-pr-readiness`: assemble the applicable evidence and complete the repository template                 |

Establish missing regression coverage and validate the baseline before freezing an optimization gate. `benchmarks.ops.verify` runs the test file in the working tree; it does not snapshot tests or references. Keep the frozen oracle, cases, and tolerances unchanged during the loop, and require the full gate before promotion. Operator, dispatch, and hardware skills supplement this workflow only where the change touches their scope.

## Current skills

| Skill                                                         | Purpose                                                                                                                                                           |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [fla-optimization-loop](fla-optimization-loop/SKILL.md)       | Disciplined, reproducible kernel optimization loop (task contract, three phases, iteration protocol, trap catalog) anchored on the frozen pytest correctness gate |
| [fla-nvidia-performance](fla-nvidia-performance/SKILL.md)     | NVIDIA GPU kernel performance workflow for Triton, Gluon, TileLang, CUDA backends, hardware baselines, and PR-ready profiling evidence                            |
| [fla-ascend-performance](fla-ascend-performance/SKILL.md)     | Ascend NPU profiling, bottleneck diagnosis, and Triton-Ascend kernel optimization                                                                                 |
| [fla-kda](fla-kda/SKILL.md)                                   | KDA-specific gate, intra/inter, backend, and test workflow                                                                                                        |
| [fla-dispatch-backends](fla-dispatch-backends/SKILL.md)       | `@dispatch` decorator and backend registry workflow                                                                                                               |
| [fla-correctness-coverage](fla-correctness-coverage/SKILL.md) | Coverage matrix and test guidance for `fla/ops/**` kernels                                                                                                        |
| [fla-design-coverage](fla-design-coverage/SKILL.md)           | Contract-first design: contract cells, numerical budgets, dispatch semantics, layered coverage, and production-first benchmarking                                 |
| [fla-pr-readiness](fla-pr-readiness/SKILL.md)                 | PR preparation checklist, test plan, and PR body structure                                                                                                        |
| [fla-triton-to-gluon](fla-triton-to-gluon/SKILL.md)           | Incremental workflow for porting a Triton kernel to Gluon (layouts, cp.async/TMA, WGMMA/tcgen05, scheduling), with API mapping and pitfall checklist              |
