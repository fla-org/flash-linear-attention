---
name: fla-pr-stacks
description: Split oversized FLA pull requests into focused changes, reuse existing PRs, and create or maintain verified GitHub native PR stacks.
---

# FLA PR Stacks

Use this workflow when a change needs decomposition or dependent PRs need a GitHub native stack. Read [CONTRIBUTING.md](../../../CONTRIBUTING.md#pr-scope-and-size) and [AGENTS.md](../../../AGENTS.md) for scope, size, compatibility, and git permissions. Use [PR readiness](../fla-pr-readiness/SKILL.md) for each resulting PR.

## Choose the boundaries

Inspect the full diff, commit history, worktree status, and open related issues/PRs before creating branches. Record the original head and preserve uncommitted work in an isolated worktree or recoverable snapshot before reconstructing commits. Compare changes against both their immediate parent and the intended trunk so inherited work is distinguishable from new work.

For each proposed PR, identify one observable outcome, its prerequisite PRs, the existing PR to reuse if any, and its validation. Prefer boundaries such as a behavior-preserving extraction, a shared API change with its callers, and a feature that consumes that API. Independent changes should target the trunk separately; a dependency graph does not always belong in one linear stack.

Keep implementation, regression tests, and required documentation together. Each layer must work with the layers below it, without depending on a later fix. Do not split by arbitrary line counts, leave callers broken between layers, or move tests out merely to reduce size. Use the contribution guide's size exception when a cohesive change still needs it, including for a mechanical migration.

## Reuse existing PRs

Read existing diffs and review discussions, not just titles. Put overlapping changes into an existing PR when its purpose and ownership fit, preserving attribution and useful review history. Check its current remote head before updating it; do not overwrite work that appeared since the initial inspection. A split request does not authorize merging or closing PRs, or incorporating unrelated changes.

Inspect repository identity as well as branch names:

```bash
gh api "repos/fla-org/flash-linear-attention/pulls/$pr_number" \
  --jq '{number, head_repo: .head.repo.full_name, head: .head.ref, sha: .head.sha, base_repo: .base.repo.full_name, base: .base.ref, stack}'
```

GitHub requires stack branches to be in the same repository; [cross-fork stacks are unsupported](https://docs.github.com/en/pull-requests/reference/stacked-pull-requests). Keep incompatible existing PRs as separate dependencies instead of recreating them under new PR numbers just to form a stack. Explain which PRs can be stacked and which remain external prerequisites. Resolve those prerequisites before representing dependent layers as ready to merge.

## Build and validate the branches

Reconstruct the series bottom to top in isolated branches or worktrees. Reuse suitable commits; split mixed commits by coherent changes while preserving authorship. Retain the original branch until comparison establishes that the decomposition preserved the intended result.

For a series based on `main`, these checks distinguish each layer's contribution from the aggregate result:

```bash
git diff --stat main...refactor/foundation
git diff --stat refactor/foundation...refactor/consumer
git diff original-head refactor/consumer
```

Replace the example refs with the recorded trunk, original head, and actual layer names. Inspect the complete diffs, not only their statistics. If the trunk changed, account for those changes explicitly when comparing final trees. Every difference from the original proposal should be intentional; removed overlaps should point to the existing PR that owns them.

Run affected tests at each layer's head, then check the combined result. A passing top branch does not establish that an intermediate PR works. Record tests and remaining gaps per PR, with a short dependency explanation in its description.

## Create the native stack

Changing PR bases alone does not register a native stack. Use GitHub's stack UI, REST Stacks API, or the official [`gh stack` extension](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-stacked-pull-requests). Check installed command help before use; publishing remains subject to the existing task authorization.

For existing branches, adopt them in bottom-to-top order and submit:

```bash
gh stack init --base main refactor/foundation refactor/consumer
gh stack submit --auto --remote origin
```

Choose the actual remote containing the stack's branches. `submit` pushes branches, creates or updates PRs, and registers the stack. With `--auto`, new PRs default to draft. Replace generated titles and bodies with reviewed descriptions using the repository's PR readiness workflow.

Alternatively, link existing PRs without adopting local tracking:

```bash
gh stack link --base main --remote origin 123 124
```

Use real PR numbers in dependency order. `link` reuses existing PRs and creates the native association; when given branch names, it also pushes branches and creates missing PRs. Inspect the result after an error before retrying, since some pushes or PR creations may already have succeeded. Do not use `--open` until the affected PRs are ready for review.

## Verify and maintain

For every PR, query its [REST stack membership](https://docs.github.com/en/pull-requests/reference/stacked-pull-requests-apis-and-webhooks):

```bash
gh api "repos/fla-org/flash-linear-attention/pulls/$pr_number" \
  --jq '{number, head: .head.ref, base: .base.ref, stack}'
```

Check that the non-null `stack` objects describe the same stack, intended order, size, and trunk, and that each PR diff contains only its layer. For locally tracked stacks, `gh stack view --json` also shows local status. Report the verified PR links and dependency order; a local branch chain or successful push alone is insufficient evidence of a native stack.

After changing a lower layer, inspect every affected upper diff and rerun relevant validation. Follow the repository's rules for rewriting published commits: `gh stack rebase` rewrites history, and `gh stack sync` also pushes with force-with-lease. Use those operations only within the user's authorization, preserving concurrent remote work. Recheck remote membership and test evidence after updating the series.

If a merge is requested, confirm the intended prefix: merging an upper PR also merges the PRs below it. Use a stack-aware merge operation; the legacy synchronous merge API does not support stacks. Do not merge as a side effect of creating or updating a series.
