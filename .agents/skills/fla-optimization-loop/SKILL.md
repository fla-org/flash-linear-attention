---
name: fla-optimization-loop
description: Iterate on FLA kernel performance with a frozen correctness gate, a measured baseline, and reproducible candidate comparisons.
---

# FLA Optimization Loop

Use this skill for repeated kernel performance work. Define supported workloads and numerical assumptions with [fla-design-coverage](../fla-design-coverage/SKILL.md); use the [NVIDIA](../fla-nvidia-performance/SKILL.md) or [Ascend](../fla-ascend-performance/SKILL.md) skill for profiling on that platform.

## Freeze the gate and baseline

Add missing regression or new-path coverage before starting the loop. Then keep the relevant tests, reference implementation, input distributions, assertions, tolerances, and public signature fixed. Do not narrow parameter matrices, add skips, relax precision or thresholds, special-case test values, cache results across trials, or change timing behavior to make a candidate pass. Fully initialize exposed outputs. Use kernel implementations for the optimization; replacing the whole operator with a vendor call solely to win the benchmark does not meet this workflow.

A suspected oracle or tolerance defect needs a separately justified correction, not an adjustment that makes the candidate pass. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md#submit-pull-requests) for precision, numerical-algorithm, and tolerance changes, and its [compatibility policy](../../../CONTRIBUTING.md#public-api-compatibility) for public behavior.

Record the entry point, target workloads, allowed implementation languages, validation/benchmark commands, baseline SHA, and success criterion in the local optimization log. Include production inputs from callers as well as applicable benchmark-registry shapes. Confirm the baseline passes the full gate and measure it before changing kernels. For a new kernel, establish correctness before tuning.

```bash
FLA_BENCH_BASE=$(git rev-parse origin/main)
FLA_CI_ENV=0 python -m benchmarks.ops.verify --op chunk_gla --base "$FLA_BENCH_BASE"
```

Use a recorded SHA: the baseline worktree cannot reuse a checked-out branch, and slash-containing refs break its temporary directory naming. `--base` benchmarks that revision; it does not run its correctness tests. Establish baseline correctness separately if the current tree already differs.

The driver runs the current pytest file and stops before timing when pytest fails. It neither freezes files nor rejects an all-skipped run. Check that the expected cases and benchmark rows ran. `--test-file` overrides a mismatched derived test path; `--gate-k` selects a quick subset, while `--no-gate` produces unverified measurements. Neither subset nor skipped-gate results qualify for promotion.

## Iterate from evidence

1. Profile the current bottleneck and choose one reasoned change.
2. Run the frozen gate, then benchmark only a passing candidate. A subset can provide an early signal; the final gate covers the full affected suite.
3. Record the change, gate result, measurement, and keep/drop decision before the next experiment. Preserve a recoverable candidate under the repository's git rules and the user's authorization; failed and unchanged attempts need records, not empty commits.

Read [measurement and correctness traps](references/TRAPS.md) before interpreting results. Keep seeds, numeric flags, hardware, and workload fixed. Candidate runtime helps rank local iterations; measure the final candidate directly against the frozen baseline in one session rather than adding speedups from separate runs.

Add specialized paths only when profiling shows different bottlenecks across shape regimes and measurements justify the extra routing. Record each condition, chosen path, and before/after result, and validate the full shape set including fallback boundaries.

After repeated attempts without an improvement above noise, re-profile and review the failed directions before continuing. Respect the user's stopping condition. A no-win conclusion needs a measured baseline, a reasoned candidate attempt, gate status, benchmark evidence, and an identified bound or blocker. An unavailable profiler or one losing candidate alone does not establish a performance floor; report unresolved validation as a limitation.

## Keep and report the result

Keep iteration notes and raw traces under ignored `profile/<op>-opt/`; [opt-log-template.md](references/opt-log-template.md) provides a compact record format. This search history stays out of kernel comments and the final PR description.

Promote only with a full green gate, a repeatable win on the claimed workloads, and profiling or a quantified bottleneck analysis that explains it. Report per-shape results and regressions, commands, baseline/candidate revisions, environment, and measurement variability using the relevant hardware skill. An unexplained large speedup needs investigation. Use [fla-pr-readiness](../fla-pr-readiness/SKILL.md) to prepare the final diff and evidence.
