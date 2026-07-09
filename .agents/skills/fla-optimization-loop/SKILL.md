---
name: fla-optimization-loop
description: >
  Disciplined, reproducible loop for making an FLA kernel faster (Triton, Gluon, TileLang, CuTe)
  without ever breaking or gaming correctness.
  Synthesizes the task-contract / three-phase / iteration-protocol / silent-bug-catalog discipline
  of agent kernel-optimization frameworks (KDA, the MLSys FlashInfer contest workflow, AKO4ALL/AKO4X),
  and anchors all of it on FLA's frozen pytest (forward AND backward, under NaN poisoning) as the immutable correctness gate.
  Use when iterating on `fla/ops/**` performance over multiple rounds.
---

# FLA Optimization Loop Skill

Use this when you are making an existing `fla/ops/**` kernel faster — or bringing a new kernel from correct to fast —
across more than one iteration, in any FLA backend language (Triton, Gluon, TileLang, CuTe DSL).
This skill is the **search discipline** that ties the other skills together; it does not replace them:

- **`fla-nvidia-performance`** — how to profile (NCU), hardware baselines, MR-ready perf evidence.
- **`fla-correctness-coverage`** — how to design the test coverage matrix for an op.
- **`fla-mr-readiness`** — how to package the promoted change into a PR.

This skill covers the loop *around* those: what to lock, how to iterate, what to record, and when to stop.

## 0. The inviolable rule: the test file is a frozen contract

The op's `tests/ops/test_<op>.py` and its `fla/ops/<op>/naive.py` reference are **frozen** for the entire optimization loop.
The whole point of "faster" only means something if correctness — forward **and** backward,
under the conftest NaN-memory poisoning — is held fixed.
During a perf loop you may **not**:

- edit the test file, or its `naive.py` reference;
- loosen an `assert_close` tolerance, or widen `rms_eps` / dtype to make a diff pass;
- drop, narrow, or `skip` parametrized shapes;
- special-case the kernel on values it only sees in the test;
- cache module-level tensors so trials reuse warm/identical data.

Run the gate with the unmodified test, every iteration:

```bash
python -m benchmarks.ops.verify --op <op> [--gate-k <subset>]
```

`verify.py` runs the pytest file as a black box and **refuses to report a speedup on a red gate**.
`--gate-k` only *selects* a shape subset for a fast signal — it never edits the test;
promote only on a full (no `-k`) green gate.

**Banned vs. allowed implementation (anti-reward-hacking):**

| Banned | Allowed |
|--------|---------|
| Making the op a thin wrapper that delegates the whole compute to a vendor lib (a plain `torch.matmul` / `F.scaled_dot_product_attention` standing in *as* the operator) just to win latency | Hand-written Triton / Gluon / TileLang / CuTe kernels; `torch` ops used as *glue* around a kernel you wrote |
| Returning uninitialized / partially-written outputs that happen to pass | Fully initialized outputs (NaN poisoning will catch partial writes) |
| Stream tricks / monkey-patching the bench to dodge timing | Genuine latency reduction measured by `verify.py` / `run.py` |
| One-sided numeric relaxation the baseline doesn't get — flipping `allow_tf32` on, dropping the fp32 accumulator to bf16/tf32, a config that quietly changes the numeric path | Same accumulation precision and numeric flags on both sides; speed comes from the kernel, not from computing something less accurate |

If you genuinely believe a test is **wrong**, that is a **separate PR** with its own justification —
never bundled into a perf change. Stop and ask the user.

## 1. Write the task contract first (before any code)

Put a short `docs/draft.md` in your scratch workspace (see §6) stating:

- **Op + entry point** — e.g. `chunk_gla` in `fla.ops.gla`.
- **Target** — which shapes (from `benchmarks/ops/registry.py` `SHAPE_CONFIGS`), and a target speedup vs. `main`.
- **Allowed languages** — Triton / Gluon / TileLang / CuTe (state any constraint).
- **Validation command** — `python -m benchmarks.ops.verify --op <op>` (the frozen gate).
- **Benchmark command** — `python -m benchmarks.ops.verify --op <op> --base main`.
- **Promotion criteria** — full green gate, a measured repeatable win, and a profiler reading that explains it (§7).
- **Frozen scope** — the test file, `naive.py`, and the public op signature.

Do not start editing kernels until the draft exists. (Borrowed from KDA: plan, then execute.)

## 2. Three phases

Run these in order; repeat 2 and 3 with progressively higher targets.

- **Phase 1 — correct baseline.** Confirm the current kernel passes the full gate, and record baseline numbers:
  `python -m benchmarks.ops.verify --op <op> --base main`.
  For a brand-new kernel, get the gate green first; performance is secondary here.
- **Phase 2 — profile-guided optimization.** Use `fla-nvidia-performance` for NCU evidence.
  Enumerate candidate directions, rank them by expected benefit vs. implementation risk,
  and explore each for at most a few iterations. Keep, revise, or reject each with evidence — don't optimize blindly.
- **Phase 3 — shape specialization.** Only when profiling shows *different* bottlenecks across shape regimes
  (short vs. long `T`, small vs. large `D`), add dispatch / specialized paths.
  Justify the added complexity with the measured win; validate on the full shape set, not the one you tuned on.
  Record each bucket (condition / entry point / per-bucket latency + speedup / reason) in `dispatch.md`
  (template in `references/opt-log-template.md`) — a specialized path without that evidence is unjustified complexity.

## 3. Iteration protocol

Every iteration is exactly three steps, in order, with no telescoping into the next iteration between them:

1. Make **one** change to the kernel.
2. Run `verify.py` — gate must stay green; record the bench number.
3. Append one row to `OPT_LOG.md` (see `references/opt-log-template.md`) and `git commit`.

A failed or no-change iteration is still an iteration: log it and commit before debugging the next direction.
(Borrowed from AKO4ALL: bench → log → commit, the most-skipped step in practice.)

**Stall handling.** After 3 consecutive iterations with no improvement (≥ a few % over current best, above noise),
stop and re-assess: re-profile, re-read `OPT_LOG.md` for which axes you've already tried,
and search for known techniques for this op family before picking a new direction.

**When to stop.** A user-set iteration cap is reached;
or re-assessment produces hard evidence of a floor (bandwidth-bound at HBM limit, launch-overhead dominated,
timer-resolution limited — cite it in `OPT_LOG.md`);
or you've documented ≥3 distinct directions tried with evidence.
Don't stop silently because a tool (e.g. NCU) was unavailable — that's a re-assessment input.

**No-go bar.** If stopping means concluding there's no win to promote — a no-go — that verdict has its own bar:
a first candidate losing doesn't clear it. A no-go needs a recorded baseline number, at least one reasoned
candidate attempt (not a blind guess), the gate status, the bench evidence, and a *named* active bound or
blocker (name which roofline/launch/timer limit, with the number). Without those five it's an unfinished loop.

## 4. Reproducibility

- **Seed** — the tests already pin `torch.manual_seed(42)`; don't undermine it.
- **Environment** — `verify.py` prints GPU / CUDA / PyTorch / Triton / commit SHA;
  paste that line into `OPT_LOG.md` so a number is interpretable later.
- **Full-shape verdict before promotion** — a `--gate-k` subset is a signal only.
- **Rank by the solution's own runtime** for fast iteration signal;
  pay for the full `--base main` comparison only at the verdict.
  Clock noise on unlocked GPUs can swing absolute speedup — see `references/TRAPS.md`.

## 5. Silent-bug & measurement traps

Read `references/TRAPS.md` before trusting any number.
It catalogs FLA-specific traps (NaN poisoning on partial writes, TF32 inflating fp32 reference diffs,
`assert_close` relative tolerance, one-sided numeric-flag relaxation, autotune-cache staleness across edits,
`int64` address arithmetic, implausible speedups that signal a silently-skipped path).
When you hit a new one, add it there with **Fact / Why / How to apply** so the next session doesn't re-learn it.
(Borrowed from AKO4X's `TRAPS.md`.)

## 6. Evidence records (scratch workspace)

Keep your search artifacts in a git-ignored directory — `profile/<op>-opt/` is already ignored:

```text
profile/<op>-opt/
  docs/draft.md       # the task contract (§1)
  OPT_LOG.md          # one row per iteration (template in references/)
  dispatch.md         # one row per shape bucket — only if Phase 3 specializes (template in references/)
  TRAPS.md            # traps you hit this session (seed: references/TRAPS.md)
  trace/              # torch.profiler / NCU artifacts (kept out of git)
```

Each kept candidate's `kernel` gets a short header (Identity / Delta / Lessons / Dead-ends / Open-directions)
per `references/opt-log-template.md`, so a later session can see what was tried and why.

## 7. Promotion → MR

A candidate is promotable only on a **full green gate** plus a measured, repeatable win on the target shapes,
**and** a profiler/roofline reading that *explains* the win (or, for a no-go, the blocker) —
a speedup you can't account for is a silent-skip suspect, not a result (see §5). Then:

- Keep the diff minimal — change only what the win needs,
  plus light cleanups (CONTRIBUTING "Protect battle-tested paths; keep diffs minimal").
- Collect perf evidence per `fla-nvidia-performance` — before/after, NCU summary, dense + varlen coverage,
  and a full final-claim stats block (median/mean/std/min/p10/p90 per shape, equal-weight geomean speedup,
  exact commands, baseline commit + candidate SHA, GPU id/model with idle-clock evidence —
  the last guards the clock-drift trap in §5).
- Package the PR per `fla-mr-readiness`.

The scratch workspace under `profile/<op>-opt/` stays local; it is not part of the PR.
