## Summary

<!-- What changed and why, at a high level. Link the issue this PR resolves, e.g. "Fixes #1234". -->

## Test plan

<!-- How you verified it: commands run, hardware used.
     Identify dependent tests first: `python scripts/find_dependent_tests.py <changed_file_or_dir>` -->

## Benchmark / NCU (kernel changes only)

<!-- Same-hardware before/after numbers, with workload shape (batch, seq_len, heads, dims, dtype).
     State "neutral" if the change is not performance-related. -->

## Breaking changes

<!-- None / list any API or behavior changes and the migration path. -->

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [ ] I have read [AGENTS.md](../AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](../.agents/skills).
- [ ] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [ ] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.

### If you ticked the "minor" box above

Standalone minor PRs are normally not accepted (see [No busywork PRs](../CONTRIBUTING.md#submit-pull-requests)).
Justify here why yours is worth a maintainer's review time — minor PRs without a justification may be closed without review:

<!-- justification, or delete this section if not applicable -->
