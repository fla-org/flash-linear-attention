---
name: fla-mr-readiness
description: >
  Checklist and workflow for preparing an MR/PR in the FLA repo. Covers
  CONTRIBUTING.md compliance, test plan, benchmark evidence, and PR body structure.
---

# FLA MR Readiness Skill

Use this skill before opening a pull request to make sure the change is
well-scoped, well-tested, and well-documented.

## Pre-flight checklist

1. **Read `CONTRIBUTING.md`**
   - Confirm code style, docstring format, commit message conventions.
   - Make sure your branch is up to date with `main` (or the target branch).

2. **Confirm change scope**
   - List the files you modified.
   - If the change spans multiple layers (kernel + model + benchmark script),
     note the dependency chain in the PR description.

3. **Check for duplicate work**
   - Search open issues and PRs:
     ```bash
     gh pr list --repo fla-org/flash-linear-attention --state open --search "<keywords>"
     ```
   - If a related PR exists, comment on it rather than opening a competing one.

4. **Run dependent tests**
   - Find tests affected by your change:
     ```bash
     python scripts/find_dependent_tests.py <changed_file_or_dir>
     ```
   - Run those tests locally and ensure they pass.
   - If a test is flaky, retry once; if it still fails, explain why in the PR.

5. **Performance evidence (if touching kernel code)**
   - See `fla-nvidia-performance` skill for the full evidence requirements.
   - At minimum: before/after benchmark on the same hardware, dense + varlen
     workloads if applicable, and a summary of any NCU profiling you did.

6. **Write PR summary**
   - Use the CI-enforced structure below. The source of truth is `.github/pull_request_template.md`; the `check-pr-title` workflow rejects bodies that drop the checklist, so never trim it to save space.

7. **Code style review**
   - Follow `CONTRIBUTING.md` for Python style, docstrings, comments, and commit prefixes.
   - In tests and public code, use device/platform wrappers from `fla.utils`
     (`device`, `device_platform`, `IS_NVIDIA`, `IS_NVIDIA_HOPPER`,
     `IS_NVIDIA_BLACKWELL`, `IS_AMD`, `IS_INTEL`) instead of new direct
     `torch.cuda` platform checks. Add a small `fla.utils` helper first when
     the existing wrappers are not enough.
   - Keep NVIDIA-only profiling commands in performance docs or scripts, not in
     generic correctness tests.

## PR body structure

Follow `.github/pull_request_template.md` exactly, checklist included:

```markdown
## Summary
One-paragraph description of what changed and why.

## Test plan
- Unit tests added/modified: `<list>`
- Dependent tests run: `<list>`
- Varlen / CP / model tests: `<yes/no + details>`

## Benchmark / NCU (kernel changes only)
- Hardware: `<e.g., H100>`
- Workload: `<batch, seq_len, dtype>`
- Before: `<throughput or latency>`
- After: `<throughput or latency>`
- Conclusion: `<improvement / neutral / trade-off>`
  (state "neutral" when the change is not performance-related)

## Breaking changes
- None / list any API or behavior changes.

## Checklist

- [x] I have read [CONTRIBUTING.md](../CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [x] I have read [AGENTS.md](../AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](../.agents/skills).
- [x] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [x] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.

### If you ticked the "minor" box above

<justification — required when the "minor" box is ticked; otherwise delete this section>
```

What `check-pr-title` (`.github/workflows/check-pr-title.yml`) enforces:

- The first four checklist boxes must always be ticked; each item carries its own N/A reading, so a tick means "considered — done or not applicable" (e.g. benchmark numbers on a docs-only PR).
- The "minor" box is the inverse: leave it unticked for normal PRs. Tick it only when the PR genuinely is a typo/formatting/style-only tweak — then a justification of at least a sentence (≥ 20 non-whitespace characters, HTML comments stripped) under `### If you ticked` is required.
- Editing the body re-triggers the check. To edit a PR title/body on this repo, use the REST API, not `gh pr edit` (see AGENTS.md "Opening PRs").

## Important reminders

- **Do not** put raw performance numbers without context. Always include:
  - workload shape (batch, seq_len, heads, dims, dtype)
  - hardware model
  - benchmark command used
  - before vs after
  - your conclusion

- **Do not** commit `.ncu-rep` files or raw profile dumps. Summarize results in
  the PR body and keep artifacts local.

- **No busywork PRs**: bundle trivial cleanups into a substantive change; do not
  open a PR for a single typo unless it is part of a larger fix.
