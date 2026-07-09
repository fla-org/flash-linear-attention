# Optimization Log Templates

Copy these into your scratch workspace `profile/<op>-opt/`.
They keep an optimization loop auditable: a future reader (or session) should be able to reconstruct
what was tried, what passed the gate, what was measured, and why a candidate was kept or dropped.
Keep this workspace out of git (`profile/` is already ignored).

---

## `OPT_LOG.md` — one row per iteration

Append a row after every iteration (including failed / no-change ones).
Record the environment line `verify.py` prints once at the top so the numbers are interpretable later.

```markdown
# Optimization log — <op>

Env: <paste the "Machine: ... | Triton ... | <branch>[sha]" line from verify.py>
Baseline (main): <median ms on target shapes>
Target: <e.g. 1.2x on B4_T4096_H64_D128, full gate green>

## Summary

| Iter | Direction (one line) | Gate | Bench (median ms) | vs best | Status |
| ---- | -------------------- | ---- | ----------------- | ------- | ------ |
| 0    | baseline             | pass | 1.934             | —       | base   |
| 1    | fuse gate mult       | pass | 1.701             | +12%    | keep   |
| 2    | larger BT block      | pass | 1.770             | -4%     | revert |
| 3    | wider vectorized ld  | fail | —                 | —       | drop   |

Status values: keep / revert / drop / floor.
```

Rules:
- **Gate** is the frozen pytest result (`pass` / `fail`), not a hand check.
- A `fail` row is benchmarked nothing — correctness comes first.
- After 3 consecutive non-`keep` rows, stop and re-assess (re-profile, review this table for untried axes)
  before the next iteration.

---

## Kept-candidate header

When you keep a candidate worth returning to, put a short header at the top of its kernel file
(in your scratch copy, not the repo) so its lineage is legible:

```markdown
# Identity:  <op> / <which kernel> / iter-<N>, parent iter-<M>
# Delta:     <the one change vs. parent, in one sentence>
# Lessons:   <what this iteration taught — mechanism, not just the number>
# Dead-ends: <what you tried inside this direction that did not work, and why>
# Open:      <levers not yet attempted — forensic note for the next session>
```

Keep provenance (specific iter numbers, exact ms, dates) in `OPT_LOG.md` and these headers —
not in the promoted PR diff or the shared SKILL docs.
The PR carries only the final minimal change plus the perf evidence (`fla-nvidia-performance`)
and PR structure (`fla-mr-readiness`).

---

## `dispatch.md` — one row per shape bucket (only if Phase 3 specializes)

A shape-specialized / dispatched kernel is justified only by evidence that buckets need *different* code.
Record that evidence so the added complexity is auditable — one row per bucket:

```markdown
# Dispatch — <op>

| Bucket condition | Entry point chosen  | Baseline ms | Candidate ms | Speedup | Why this path                          |
| ---------------- | ------------------- | ----------- | ------------ | ------- | -------------------------------------- |
| T <= 512         | `chunk_fwd_small`   | 0.42        | 0.31         | 1.35x   | launch-bound; smaller BT cuts overhead |
| T > 512, D <= 64 | `chunk_fwd_default` | 1.93        | 1.70         | 1.14x   | compute-bound; default tiling wins     |
```

Rules:
- A bucket earns a row only if profiler/bench evidence shows it needs a *different* path —
  don't split a bucket the single kernel already serves best.
- Every bucket's speedup is vs. the same frozen `main` baseline on *that bucket's* shapes.
