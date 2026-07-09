# FLA Optimization Traps

A catalog of silent-bug and measurement traps specific to optimizing FLA kernels.
Each entry has a **Fact**, a **Why** (so a future session can judge whether its context flips the fact),
and a **How to apply**.
This is a *seed* — when you hit a new trap during an optimization loop,
copy it into your scratch `profile/<op>-opt/TRAPS.md` and, if it generalizes, propose adding it here.

---

## Correctness traps

### Partially-initialized outputs pass in eager but are real bugs

**Fact:** A kernel that writes only part of its output tensor (a masked tail, an unfilled boundary block)
can pass a casual eager run yet be wrong.

**Why:** `tests/conftest.py` replaces `torch.empty` / `empty_like` / `new_empty` with **NaN-filled** tensors
for `tests/ops/` and `tests/modules/`. Uninitialized reads then surface as NaN in `assert_close`.
Outside that guard the same memory is often coincidentally zero, hiding the bug.

**How to apply:** Fully initialize every output element.
If the gate fails with NaN but an ad-hoc script "passes", trust the gate — you have a partial-write bug.

### TF32 inflates the fp32 reference diff

**Fact:** An fp32 correctness case can fail by a margin that is a measurement artifact, not a kernel bug,
if TF32 matmuls are on in the reference path.

**Why:** cuBLAS TF32 has a 10-bit mantissa;
an einsum/matmul backward in the naive reference then differs from a true-fp32 kernel by more than the tolerance.
`tests/ops/test_attnres.py` explicitly sets `torch.backends.cuda.matmul.allow_tf32 = False` for exactly this reason.

**How to apply:** Don't "fix" an fp32 diff by loosening tolerance.
Check whether the *reference* is running TF32 first; the test, not your kernel, owns that choice — and it is frozen.

### `assert_close` is a *relative* tolerance — never loosen it to pass

**Fact:** The gate uses `fla.utils.assert_close` with a relative tolerance.
Widening it to make a candidate pass converts a correctness regression into a silent one.

**Why:** A "win" that only passes at a looser tolerance is usually a numerically degraded kernel
(lower-precision accumulation, dropped correction term), not a faster correct one.

**How to apply:** Tolerance is part of the frozen contract.
If you think it's wrong, that's a separate PR with justification — ask the user.

### Numeric flags must be symmetric across baseline and candidate

**Fact:** A "win" can be a one-sided numeric relaxation — the candidate runs at a lower precision
the baseline never gets — rather than a genuinely faster kernel.

**Why:** `allow_tf32 = True`, a bf16/tf32 accumulator where the reference keeps fp32, or any flag that
changes code generation makes the candidate compute *something cheaper but less accurate*.
The gate may still pass within tolerance, so the speedup looks real while the comparison is rigged.

**How to apply:** Keep accumulation precision and numeric flags identical on both sides;
`verify.py` compares against `main`, so the win has to come from the kernel, not from the flag.
If a precision change is genuinely the point, it is a separate, justified PR — not bundled into a perf number.

**Fact:** Program IDs and grid-derived offsets multiplied by sizes/strides can silently overflow at large `T`,
producing wrong results only on big shapes.

**Why:** Triton program IDs and non-first grid dims can be narrow integers;
the overflow is silent and shape-dependent (CONTRIBUTING → Triton Kernels).
A kernel can pass small-shape cases and corrupt `T=8000+` ones.

**How to apply:** Cast to `tl.int64` before address arithmetic; keep all base/stride/offset math in `int64`.
Always run the gate's large-`T` cases, not just the fast small ones.

---

## Measurement traps

### A speedup implausible for the change you made signals a silent skip

**Fact:** A large speedup that doesn't match the structural change (e.g. you touched one reduction and latency halved)
usually means a code path was skipped, a shape was filtered out of the run, or correctness silently degraded.

**Why:** Optimizers reward-hack by accident — an early return, a wrong dispatch,
a `dim_constraints` filter dropping the expensive shape.
The contest/AKO harnesses flag implausible speedups for this reason.

**How to apply:** Before celebrating, confirm the gate is green on the *same* shapes you benchmarked,
and that the shape was actually run (not skipped). If the number is too good, it is.

### Autotune cache can be stale across edits

**Fact:** Triton autotune results are cached;
after editing a kernel, the first timed run may reflect the *old* best config or an un-retuned path.

**Why:** Caching speeds iteration but pins config choices made before your edit.

**How to apply:** `run.py` / `verify.py` warm up all shapes before timing — keep that.
When a result looks off after a config-space change, clear/rewarm the autotune cache and re-measure.

### Clock-state drift moves absolute speedup

**Fact:** The same unchanged kernel can report different absolute speedups across runs when GPU clocks are unlocked;
cumulative AB deltas across separate runs do not compose.

**Why:** Reference and solution timed in different clock states carry per-run noise;
summing N independent deltas accumulates noise that can reverse sign.

**How to apply:** For iteration *signal*, rank by the solution's own runtime in one run.
For the *verdict*, run the full `--base main` compare in one session.
Don't claim a cumulative win as the sum of prior separate measurements — re-measure directly against the baseline.
