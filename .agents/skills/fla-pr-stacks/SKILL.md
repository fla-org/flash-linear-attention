---
name: fla-pr-stacks
description: Split oversized FLA pull requests into focused changes, reuse existing PRs, and create or maintain verified GitHub native PR stacks.
---

# FLA PR Stacks

Use for decomposing large changes or managing dependent PRs. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md#pr-scope-and-size) for scope and size, [AGENTS.md](../../../AGENTS.md) for git permissions, and [PR readiness](../fla-pr-readiness/SKILL.md) for each PR's evidence and description.

## 1. Choose the split

Inspect the complete diff, commit history, and related open PRs, including their review discussions. Map each proposed layer to one outcome, its prerequisites, validation, and an existing PR to reuse where appropriate.

| Relationship                            | Action                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------ |
| Overlaps an existing PR's purpose       | Update that PR when ownership permits; preserve attribution and reviews. |
| Independent of the other changes        | Submit separately against the trunk.                                     |
| Depends on another layer                | Stack above that layer; include its own tests and documentation.         |
| Cohesive but still above the size limit | Use the contribution guide's exception with a concrete justification.    |

Each layer presents an increment over its parent and must work with its prerequisites, without needing later layers to fix it. Separate review does not imply dependency-free code. Do not split by arbitrary line counts, leave callers broken, or defer tests to a later PR.

Check head repository identity before arranging existing PRs. [GitHub stacks require all branches in the same repository](https://docs.github.com/en/pull-requests/reference/stacked-pull-requests); cross-fork stacks are unsupported. Keep incompatible PRs as explicit external prerequisites instead of duplicating them under new PR numbers.

## 2. Build and validate

Preserve the original head and uncommitted work before reconstructing commits in isolated branches or worktrees. Reuse suitable commits and preserve authorship. Check current remote heads before updating existing PRs.

Review each layer against its parent, then compare the top branch with the preserved original result:

```bash
git diff main...refactor/foundation
git diff refactor/foundation...refactor/consumer
git diff original-head refactor/consumer
```

Substitute the recorded trunk and actual refs. Account for trunk changes and intentional omissions, including fixes retained in existing PRs. Run relevant tests at each layer's head and on the combined result; passing only the top branch does not validate intermediate PRs.

## 3. Register and verify the native stack

Changing PR bases alone does not create a native stack. Use the official [`gh stack` extension](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-stacked-pull-requests), GitHub UI, or REST Stacks API. Check installed command help and existing publishing authorization first.

For prepared branches, adopt them bottom to top and submit:

```bash
gh stack init --base main refactor/foundation refactor/consumer
gh stack submit --auto --remote origin
```

Choose the actual remote containing the branches. `submit` pushes, creates or updates PRs, and registers the stack. `--auto` creates new PRs as drafts; replace generated titles and bodies using PR readiness.

For existing PRs, use `gh stack link --base main --remote origin 123 124` with their real numbers in dependency order. This reuses PRs without requiring local stack tracking. Branch arguments also push and create missing PRs. After an error, inspect what succeeded before retrying.

For every PR, verify [remote stack membership](https://docs.github.com/en/pull-requests/reference/stacked-pull-requests-apis-and-webhooks):

```bash
gh api "repos/fla-org/flash-linear-attention/pulls/$pr_number" \
  --jq '{number, head_repo: .head.repo.full_name, head: .head.ref, base: .base.ref, stack}'
```

Require non-null `stack` objects with the same stack identity, correct order, size, and trunk. Check each PR's incremental diff and report the verified links and remaining external prerequisites. A local branch chain or successful push is insufficient.

## 4. Maintain the series

After changing a lower layer, review affected upper diffs and rerun relevant tests. `gh stack rebase` rewrites history; `gh stack sync` also force-pushes with leases. Apply the repository's rewrite permissions and preserve concurrent remote work, then recheck published membership and evidence.

Creating a series does not authorize merging or closing PRs. If merging is requested, establish the intended prefix: selecting an upper PR also merges those below it. Use a stack-aware merge operation; the legacy synchronous merge API does not support stacks.
