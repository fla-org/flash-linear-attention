# Skills

Add reusable workflows as:

```text
.agents/skills/<skill-name>/SKILL.md
```

Each skill should be self-contained and task-specific.
Include YAML frontmatter with `name` and `description`.
If a skill needs reference files, place them in a `references/` subdirectory inside the skill folder.
Symlinks to public repo docs are allowed when the referenced file is already tracked in this repository.

Do not add a `README.md` inside individual skill directories; the canonical entry point is `SKILL.md`.

## Current skills

| Skill                      | Purpose                                                                                                                                                           |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fla-optimization-loop`    | Disciplined, reproducible kernel optimization loop (task contract, three phases, iteration protocol, trap catalog) anchored on the frozen pytest correctness gate |
| `fla-nvidia-performance`   | NVIDIA GPU kernel performance workflow for Triton, Gluon, TileLang, CUDA backends, hardware baselines, and MR-ready profiling evidence                            |
| `fla-kda`                  | KDA-specific gate, intra/inter, backend, and test workflow                                                                                                        |
| `fla-dispatch-backends`    | `@dispatch` decorator and backend registry workflow                                                                                                               |
| `fla-correctness-coverage` | Coverage matrix and test guidance for `fla/ops/**` kernels                                                                                                        |
| `fla-mr-readiness`         | MR/PR preparation checklist, test plan, and PR body structure                                                                                                     |
