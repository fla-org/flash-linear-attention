---
name: fla-pr-readiness
description: >
  Prepare or update an FLA pull request with one concrete purpose, a concise
  description, verified tests and benchmarks, and justified size exceptions.
---

# FLA PR Readiness Skill

Use this skill before opening a PR or updating its scope and evidence during review. Read [CONTRIBUTING.md](../../../CONTRIBUTING.md) first; it defines the contribution rules. [AGENTS.md](../../../AGENTS.md) defines operational constraints. The [PR template](../../../.github/pull_request_template.md) is the source for headings and checkboxes; do not maintain a second copy here.

## Establish scope

- Read the complete diff against the PR base, including tests, documentation, and all shared callers. Express its single outcome in one sentence; separate independent fixes or optimizations.
- Search open issues and PRs before starting, for example `gh pr list --repo fla-org/flash-linear-attention --state open --search "<keywords>"`. Check matching discussions before duplicating work.
- Follow the design/RFC requirements in `AGENTS.md` before implementation. Use the [skill index](../README.md) to choose contract, operator, backend, and performance workflows relevant to the change.
- Inspect the full diff size. Above 500 additions plus deletions, verify that it is one cohesive change before filling the template's large-PR acknowledgement and justification. Explain why it should stay together and suggest a review order; a size exception does not justify unrelated work.

## Collect evidence

1. Discover affected tests with `python scripts/find_dependent_tests.py <changed_file_or_dir>`. Add or update the matching test for kernel/module changes, and check that a regression test actually exercises the modified path.
2. Run applicable tests and record the command, tested commit, environment, and results. Use strict local numerical validation (`FLA_CI_ENV=0`), retaining reference outputs, supported gradients, and existing tolerances. Separate failed, skipped, and unrun checks from passes; investigate failures and reproduce them on the baseline before calling them pre-existing.
3. For kernel or performance-relevant changes, collect same-hardware before/after results following `CONTRIBUTING.md#benchmarking`. Select the relevant hardware skill for profiling and workload coverage. A measured neutral result still needs numbers; use `N/A` only with an applicable reason.
4. Run pre-commit on changed files and inspect the final diff. Audit affected comments, public callers, checkpoint compatibility, and the supported shapes, dtypes, and backends. Keep scratch logs and raw profiles outside the PR.

## Write and check the description

Follow [PR Description](../../../CONTRIBUTING.md#pr-description) and [Evidence by Change Type](../../../CONTRIBUTING.md#evidence-by-change-type):

- Open `Summary` with a one- or two-sentence TL;DR naming the problem or capability and the observable result. Include only the rationale, compatibility effects, or limitations needed to evaluate it.
- Remove variable inventories, file-by-file narration, repeated claims, and the history of attempted approaches. Keep reproducible commands and compact evidence tables; link or collapse long logs.
- Make every correctness or performance claim traceable to evidence for the relevant path. Describe the final implementation and re-run affected validation when review changes invalidate earlier results.
- Preserve the template checklist wording. Tick the first four items only when complete or inapplicable, recording the reason in the relevant section. The minor/cosmetic box and large-PR box are conditional; do not tick every box automatically.
- Keep incomplete work in draft and state missing validation. A failing metadata check on an unfinished draft is not a reason to claim tests passed.

## Publishing and follow-up

The `check-pr-title` workflow validates the title, checklist, and size exception, including on body edits. It checks the presence of an explanation; reviewers judge whether the stated purpose, evidence, and exception are sound. Use an outcome-based `[Tag]` title and the current template.

`gh pr edit` fails on this repository because of its classic-Projects integration. Write the complete body to a temporary file and update it with the REST API:

```bash
gh api -X PATCH repos/fla-org/flash-linear-attention/pulls/<N> -F body=@<body-file>
```

Confirm the published title/body, head commit, and check results. Follow repository git safety rules and the user's authorization for publishing; this skill does not grant permission to post unrelated comments or rewrite pushed commits.
