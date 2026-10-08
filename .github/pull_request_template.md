## Summary

<!-- Start with **TL;DR:** and 1–2 sentences describing one concrete outcome and who it affects. Link the issue or design discussion. -->
<!-- Add only rationale, trade-offs, or limitations needed for review; leave variable/file inventories and implementation narration to the diff. See CONTRIBUTING.md#pr-description. -->

## Test plan

<!-- Describe the regression/new behavior covered, commands run, and results. Identify failed, skipped, and unrun checks. -->
<!-- For GPU/NPU tests include hardware, relevant software versions, and backend flags. Kernel/module changes need added/updated tests and passing reference comparisons for outputs, states, and supported gradients. -->
<!-- Find dependent tests first: `python scripts/find_dependent_tests.py <changed_file.py> [more_files.py ...]`. -->

## Benchmark / NCU

<!-- Required for kernel or performance-relevant changes: baseline/candidate commits, same hardware/software, shapes/dtypes, command, and a small before/after table with units. -->
<!-- Include dense/varlen where applicable and disclose regressions. "Neutral" needs measurements; otherwise explain N/A. NCU is optional. Link/collapse full logs. -->

## Breaking changes

<!-- None / affected APIs, supported inputs or backends, defaults, checkpoint compatibility, and migration. -->

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [ ] I have read [AGENTS.md](../AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](../.agents/skills).
- [ ] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [ ] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.
- [ ] I understand this PR exceeds 500 changed lines and have explained its single purpose and why it should stay together below.

### Large PR justification

<!-- Required only above 500 additions + deletions in the full PR diff; otherwise delete this section and leave the large-PR box unchecked. -->
<!-- Explain the single outcome, why splitting would hinder review/correctness, and a useful review order. See CONTRIBUTING.md#pr-scope-and-size for examples. -->

### If you ticked the "minor" box above

<!-- Standalone cosmetic PRs are normally not accepted. Explain why this exception is worth reviewing; otherwise delete this section. See CONTRIBUTING.md#submit-pull-requests. -->
