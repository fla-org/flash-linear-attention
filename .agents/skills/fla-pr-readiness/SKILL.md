---
name: fla-pr-readiness
description: Prepare or update an FLA pull request with focused scope, verified evidence, and the current repository template.
---

# FLA PR Readiness

Read [CONTRIBUTING.md](../../../CONTRIBUTING.md) and [AGENTS.md](../../../AGENTS.md) first. Use the current [PR template](../../../.github/pull_request_template.md) for headings and checkboxes; contribution policy and template wording are not duplicated here.

## Review scope and evidence

Read the complete diff against the PR base and audit affected shared callers. State its concrete outcome in one sentence and separate independent work. Check existing issues/PRs and the applicable design discussion. Follow [scope and size](../../../CONTRIBUTING.md#pr-scope-and-size) for a cohesive change above the normal limit, and [public compatibility](../../../CONTRIBUTING.md#public-api-compatibility) when interfaces or defaults are affected.

For code changes, discover dependent tests and run the relevant targets. For example:

```bash
python scripts/find_dependent_tests.py fla/ops/kda/chunk.py
FLA_CI_ENV=0 pytest tests/ops/test_kda.py
pre-commit run --files fla/ops/kda/chunk.py tests/ops/test_kda.py
```

For documentation-only changes, validate links, examples, and formatting as described in [Evidence by Change Type](../../../CONTRIBUTING.md#evidence-by-change-type).

The dependency helper only lists tests. Check that regression coverage exercises the changed path and that required numerical comparisons passed, including supported gradients. Record commands, tested revision, environment, and results; distinguish failures, skips, and unrun checks. Reproduce a failure on the baseline before calling it pre-existing.

For performance-relevant changes, collect same-hardware before/after measurements under [Benchmarking](../../../CONTRIBUTING.md#benchmarking) and the relevant hardware skill. A neutral result needs numbers; explain inapplicable checks. Inspect the final diff for stale comments and unrelated changes, leaving scratch logs and raw profiles outside it.

## Prepare and publish

Write for a reviewer who has not seen the conversation. Follow [PR Description](../../../CONTRIBUTING.md#pr-description) and [Evidence by Change Type](../../../CONTRIBUTING.md#evidence-by-change-type): explain the problem, resulting behavior, affected users, and evidence. Include only useful rationale or limitations; omit file inventories and the history of attempts. Keep incomplete work in draft and leave claims and checklist items consistent with actual validation.

The `check-pr-title` workflow also checks the body and size exception on edits. Preserve template checklist wording and distinguish required items from conditional ones.

`gh pr edit` fails with this repository's classic-Projects integration. Write the complete body to a temporary file and update through REST:

```bash
gh api -X PATCH "repos/fla-org/flash-linear-attention/pulls/$pr_number" -F "body=@$pr_body_file"
```

Confirm the published description, head revision, and checks. After review changes, rerun affected validation and update evidence that no longer describes the current code. Publishing and git operations remain subject to the user's authorization and repository rules.
