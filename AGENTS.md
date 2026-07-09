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
- `gh pr edit` fails on this repo (classic-Projects GraphQL error). Edit a PR title/body via the REST API instead: `gh api -X PATCH repos/fla-org/flash-linear-attention/pulls/<N> -f title='...' -F body=@file`.

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

Load the relevant skill when your task matches its scope. See `.agents/skills/README.md`
for the directory convention.
