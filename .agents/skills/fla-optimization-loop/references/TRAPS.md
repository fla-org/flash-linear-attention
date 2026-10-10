# Correctness and Measurement Traps

## Partial writes

`tests/conftest.py` poisons eligible FLA `empty` allocations in operator/module tests with NaNs. It excludes some allocations and does not cover layer/model/CP tests. A poisoned-run failure that disappears in an ad hoc script can reveal uninitialized reads; inspect masks and ensure every exposed output element is written. NaNs can also have numerical causes, so diagnose the failure rather than assuming its source.

## Reference precision and tolerance

`fla.utils.assert_close` checks relative RMS error with an absolute-error shortcut, not elementwise `rtol`. `FLA_CI_ENV=1` and explicit `warning=True` can downgrade tolerance failures to warnings. Use strict local validation and inspect warnings; keep the committed thresholds fixed.

TF32 in a reference matmul can change an fp32 comparison. Inspect the reference's precision settings before blaming the kernel, but do not change the frozen oracle to pass. Baseline and candidate need identical precision and numeric flags. An intentional precision change follows the numerical-change process in [CONTRIBUTING.md](../../../../CONTRIBUTING.md#submit-pull-requests).

## Address overflow

Cast program IDs and offsets to `tl.int64` before multiplication; widening an already-overflowed product cannot repair it. Small shapes may hide the error. Run large-offset and affected varlen/grid cases under the [Triton addressing rules](../../../../CONTRIBUTING.md#triton-kernels).

## Missing work in a fast result

Check unexpectedly large speedups against the actual selected path, executed tests, and measured shapes. `verify.py` accepts pytest's exit status, including an all-skipped success. The benchmark runner filters constrained shapes and omits failed warmups or measurements. Compare expected and returned rows before making a claim; a passing process alone is insufficient.

## Autotune and clocks

`run.py`, also used by `verify.py`, warms all shapes before timing. Preserve that warmup. After changing the configuration space, check cached selections and rewarm or clear the relevant cache if results look stale.

Unlocked clocks and changing device load can shift latency between runs. Rank local candidates by their own runtime, then remeasure the final candidate against the recorded baseline in one session. Report variability and the device state; speedups from separate experiments do not add up to a final result.
