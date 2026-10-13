# Repository skills

[CONTRIBUTING.md](../../CONTRIBUTING.md) defines contribution rules; [AGENTS.md](../../AGENTS.md) defines agent operations. Skills add task-specific guidance. Choose those relevant to the change.

For kernel work, define the contract and establish baseline coverage before optimization. Use the operator, dispatch, and hardware guides as needed, then prepare the PR with the collected evidence.

| Skill                                                     | Use for                                                    |
| --------------------------------------------------------- | ---------------------------------------------------------- |
| [Design coverage](fla-design-coverage/SKILL.md)           | Supported inputs, numerical budgets, and routing contracts |
| [Correctness coverage](fla-correctness-coverage/SKILL.md) | References, regression cases, and affected test selection  |
| [KDA](fla-kda/SKILL.md)                                   | KDA gate modes and kernel invariants                       |
| [Backend dispatch](fla-dispatch-backends/SKILL.md)        | Registration, verifiers, and fallback behavior             |
| [Optimization loop](fla-optimization-loop/SKILL.md)       | A fixed correctness gate and measured candidate iterations |
| [NVIDIA performance](fla-nvidia-performance/SKILL.md)     | GPU profiling and bottleneck diagnosis                     |
| [Ascend performance](fla-ascend-performance/SKILL.md)     | NPU profiling and compiler constraints                     |
| [Triton to Gluon](fla-triton-to-gluon/SKILL.md)           | Layouts, synchronization, and incremental kernel ports     |
| [PR readiness](fla-pr-readiness/SKILL.md)                 | Scope, validation evidence, and submission                 |

## Maintaining skills

Put each workflow in `.agents/skills/<skill-name>/SKILL.md` with YAML `name` and `description` fields. Follow [documentation and skill style](../../CONTRIBUTING.md#prose-and-markdown) for scope, organization, wording, and shared policy.

Use an existing `references/` directory for substantial task-specific detail. Add supporting files only when they make the workflow easier to use; `SKILL.md` is the entry point, so individual skill directories do not need a README. Symlinks may point to public documentation tracked in this repository.
