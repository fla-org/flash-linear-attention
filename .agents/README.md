# Agent Workspace

Repository-local agent configuration lives here.

Keep shared rules in `AGENTS.md`. Put optional task-specific workflows under `skills/`.

## Repo-local skills

Skills are self-contained workflow guides stored as:

```text
.agents/skills/<skill-name>/SKILL.md
```

Current skills:

| Skill                      | Purpose                                                                         |
| -------------------------- | ------------------------------------------------------------------------------- |
| `fla-optimization-loop`    | Correctness-gated kernel optimization loop (contract, phases, iteration, traps) |
| `fla-nvidia-performance`   | NVIDIA GPU kernel / Triton / Gluon / TileLang / CUDA backend performance work   |
| `fla-kda`                  | KDA-specific gate, intra/inter, backend, and test workflow                      |
| `fla-dispatch-backends`    | `@dispatch` decorator and backend registry workflow                             |
| `fla-correctness-coverage` | Kernel correctness testing and coverage for `fla/ops/**`                        |
| `fla-mr-readiness`         | Preparing MR/PR, test plans, and contribution compliance                        |

To add a new skill, create a directory under `skills/` containing `SKILL.md` and optional `references/` files.
Do not add redundant `README.md` files inside skill directories.
