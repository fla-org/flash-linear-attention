# Optimization Record

Keep one local log under ignored `profile/<op>-opt/`. Record the entry point, target workloads, baseline SHA and correctness, environment, commands, numeric settings, and success criterion once. Preserve enough candidate identity to reproduce each measurement.

| Attempt  | Change or hypothesis | Gate | Latency by workload | Decision and reason |
| -------- | -------------------- | ---- | ------------------- | ------------------- |
| Baseline | Unchanged revision   | Pass | Measured baseline   | Comparison point    |

Append an entry after each attempt, including failures and unchanged results. Identify whether the gate was full or selected; record no performance result for a failed gate. Retained candidates need their revision or recoverable diff, the reason they helped, and any unresolved limitation. Keep failed directions here so later work does not repeat them.

When specializing, add each routing condition, entry point, baseline/candidate latency, and measured reason for a separate path. Compare every bucket against the same baseline revision on that bucket's workloads, including fallback boundaries. Keep these details alongside the iteration results.
