# FLA Guidelines

Guidance for AI coding agents (Claude Code, Codex, etc.) working in this repo.

**Read `CONTRIBUTING.md` first.** It is the authoritative source for all code style, docstring, comment, commit, PR, and testing conventions, and applies to humans and agents alike. This file only covers agent-specific operational behavior that doesn't belong in a human contributor doc.

## Git safety

- **Never discard uncommitted work** with `git checkout HEAD -- <file>`, `git reset --hard`, or `git restore` to "get a clean base". Unstaged changes are unrecoverable (no blob, no reflog). Edit in place or `git stash` instead, and confirm with the user when in doubt.
- **On `main`**: never commit or push without explicit per-action approval. Suggest a feature branch first.
- Don't rewrite or amend already-pushed commits unless the user asks.

## Opening PRs

- **Check for duplicates first**: search open issues/PRs before starting so you don't redo in-flight work — `gh pr list --repo fla-org/flash-linear-attention --state open --search "<keywords>"`.
- **No busywork PRs**: don't open a one-off PR for a single typo or isolated style tweak; bundle trivial cleanups into substantive work.
- **No tool attribution**: no `Co-Authored-By` trailers, "Generated with ..." lines, or AI-tool names in commit messages, PR titles/bodies, or review comments.
- Grep your diff for `tl.make_block_ptr` / `tl.advance` before opening a PR — banned from mainline Triton code (see CONTRIBUTING "Triton Kernels").
- `gh pr edit` fails on this repo (classic-Projects GraphQL error). Edit a PR title/body via the REST API instead: `gh api -X PATCH repos/fla-org/flash-linear-attention/pulls/<N> -f title='...' -F body=@file`.

## Scope and direction

- **Align before coding when a design decision is involved**: a new operator or model, a new public API, a kernel rewrite with a different algorithm, or a repo-wide policy change (e.g. adding a dtype everywhere). Open an issue or a draft PR first; if a matching issue/PR already exists, comment there instead of opening a duplicate. Judge design impact separately from the PR size check: tests, config boilerplate, and mechanical repetition of an existing pattern alone do not make a change a design decision.
- **RFC first for precision or algorithm changes**: changing the precision a validated kernel computes or accumulates in (e.g. fp32 → bf16, bf16 → fp16), relaxing a tolerance, or replacing a numerical algorithm (e.g. a different inversion or solve scheme) needs an RFC issue before a PR, even if it would stay within test tolerance. FLA's kernels are widely validated and downstream users rely on their numerics, so these are design decisions to agree on, not fixes to ship. Ordinary reorderings within the same precision — tiling, accumulation order, scheduling — do not need an RFC; they are normal optimization PRs with before/after benchmarks as usual.
- **Stop and ask before breaking changes**: renaming or removing public symbols or arguments in `fla/layers/` / `fla/models/` or documented config fields; changing checkpoint/state-dict compatibility; changing observable behavior, including numerics beyond the existing test tolerance. Bug fixes that restore intended behavior are not breaking. If the user doesn't answer — or you are running non-interactively — do the non-breaking parts, leave the breaking part out, and describe it in your final report. Silence is not consent.

## Comment discipline

The style rules for comments and docstrings live in `CONTRIBUTING.md` ("Docstrings and Comments"), including the banned anti-patterns. What follows is agent-specific behavior:

- **You own the comments your change invalidates.** When a change makes a comment or docstring in the code you touched factually wrong, fix it or delete it in the same commit — "the code you touched" means the same function, kernel, or class, not a repo-wide sweep for stale mentions. If you can't tell whether a comment is still true, keep it and flag it in the PR description; never delete a "why" comment just because you can't verify it.
- **Comments are not your scratchpad.** State design conclusions, not the journey: "not X, it does not compile" is a welcome why comment; "tried X first, then switched to Y" is chain-of-thought noise. This covers docstrings too.
- **Fix the words, not the style.** Correcting stale content is not a license to reformat the surrounding docstring or comment style.
- Before opening a PR, re-read your own diff once with fresh eyes: check every comment it adds against the anti-pattern list, and every comment it invalidates against the rule above.

## Pre-PR self-check

Before opening a PR, verify both of these and be ready to show the answer:

- **Minimal diff**: every changed line serves the PR's stated purpose. Drop drive-by reformats, unrelated cleanups, and speculative generality before pushing, not after a reviewer asks (see CONTRIBUTING "Core Principles" 5).
- **Blast radius**: for every shared symbol you touched, all callsites still hold (Core Principle 4 — audit them in one pass, don't assume). For any behavior or numerics change, the PR description must state who is affected — which shapes, dtypes, backends, checkpoints, or downstream users — or state explicitly why no one is. "Tests pass" is not an impact analysis.

## Review comments

Keep review/PR comments concise and natural — skip heavy `**1.** **2.**` scaffolding, write like a person.

## Repo-local Skills

Use the [skill index](.agents/skills/README.md) to select task-specific workflows and their execution order. Skills supplement `CONTRIBUTING.md` and this file; they do not define competing contribution policies.

Load a relevant skill when the task matches its scope. In particular, load [fla-pr-readiness](.agents/skills/fla-pr-readiness/SKILL.md) before opening any PR.
