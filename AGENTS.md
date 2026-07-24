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
- `gh pr edit` fails on this repo (classic-Projects GraphQL error). Edit a PR title/body via the REST API instead: `gh api -X PATCH repos/fla-org/flash-linear-attention/pulls/<N> -f title='...' -F body=@file`.

## Scope and direction

- **Align before coding when a design decision is involved**: a new operator or model, a new public API, a kernel rewrite with a different algorithm, or a repo-wide policy change (e.g. adding a dtype everywhere). Open an issue or a draft PR first; if a matching issue/PR already exists, comment there instead of opening a duplicate. Line count is only a heuristic — tests, config boilerplate, and mechanical repetition of an existing pattern don't count toward it.
- **Stop and ask before breaking changes**: renaming or removing public symbols or arguments in `fla/layers/` / `fla/models/` or documented config fields; changing checkpoint/state-dict compatibility; changing observable behavior, including numerics beyond the existing test tolerance. Bug fixes that restore intended behavior are not breaking. If the user doesn't answer — or you are running non-interactively — do the non-breaking parts, leave the breaking part out, and describe it in your final report. Silence is not consent.

## Comment discipline

The style rules for comments and docstrings live in `CONTRIBUTING.md` ("Docstrings and Comments"), including the banned anti-patterns. What follows is agent-specific behavior:

- **You own the comments your change invalidates.** When a change makes a comment or docstring in the code you touched factually wrong, fix it or delete it in the same commit — "the code you touched" means the same function, kernel, or class, not a repo-wide sweep for stale mentions. If you can't tell whether a comment is still true, keep it and flag it in the PR description; never delete a "why" comment just because you can't verify it.
- **Fix the words, not the style.** Correcting stale content is not a license to reformat the surrounding docstring or comment style.
- Before opening a PR, re-read your own diff once with fresh eyes: check every comment it adds against the anti-pattern list, and every comment it invalidates against the rule above.

## Review comments

Keep review/PR comments concise and natural — skip heavy `**1.** **2.**` scaffolding, write like a person.

## Repo-local Skills

This repo provides task-specific workflow skills under `.agents/skills/*/SKILL.md`:

- **`fla-optimization-loop`** — disciplined, reproducible kernel optimization loop with a frozen pytest correctness gate (`benchmarks/ops/verify.py`)
- **`fla-nvidia-performance`** — NVIDIA GPU kernel / Triton / Gluon / TileLang / CUDA backend performance work
- **`fla-kda`** — KDA-specific gate, intra/inter, backend, and test workflow
- **`fla-dispatch-backends`** — `@dispatch` decorator and backend registry workflow
- **`fla-correctness-coverage`** — Kernel correctness testing and coverage for `fla/ops/**`
- **`fla-mr-readiness`** — Preparing MR/PR, test plans, and contribution compliance

Load the relevant skill when your task matches its scope — in particular, load
**`fla-mr-readiness`** before opening any PR. See `.agents/skills/README.md`
for the directory convention.
