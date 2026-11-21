# Forward-mode derivative of the chunked gated delta rule

`chunk_gated_delta_rule_jvp` returns the output $\mathbf{o}$ of `chunk_gated_delta_rule` together with its directional derivatives $\dot{\mathbf{o}}$ along $D$ tangent directions $(\dot{\mathbf{q}}, \dot{\mathbf{k}}, \dot{\mathbf{v}}, \dot{g}, \dot{\beta})$.
This note derives the tangent of the chunked form and maps it to the kernels in `chunk_jvp.py`.

## Notation

One sequence and one head; chunk $n$ holds positions $r, s$, and $L$ is its last row.
$G_r = \sum_{s \le r} g_s$ is the cumulative log decay within the chunk, and a dot marks a tangent, so $\dot G_r = \sum_{s \le r} \dot g_s$.
The kernels hold $G$ in log2 units (`chunk_local_cumsum(g, scale=RCP_LN2)`, so that `exp2(g_cum)` $= e^{G}$) and $\dot G$ in natural units, since $\mathrm{d}\, e^{G} = e^{G} \dot G$.

## Forward

The library computes, per chunk,

```math
\begin{aligned}
M_{rs} &= \beta_r (\mathbf{k}_r^\top \mathbf{k}_s)\, e^{G_r - G_s} \quad (r > s), \qquad A = (I + M)^{-1}, \\
\mathbf{u} &= A\, (\beta \mathbf{v}), \qquad \mathbf{w} = A\, (\beta e^{G} \mathbf{k}), \qquad \boldsymbol{\nu} = \mathbf{u} - \mathbf{w} \mathbf{h}_n, \\
\mathbf{h}_{n+1} &= e^{G_L} \mathbf{h}_n + \textstyle\sum_r e^{G_L - G_r}\, \mathbf{k}_r \boldsymbol{\nu}_r^\top, \\
\mathbf{o}_r &= \text{scale} \left[ e^{G_r}\, \mathbf{q}_r^\top \mathbf{h}_n + \textstyle\sum_{s \le r} (\mathbf{q}_r^\top \mathbf{k}_s)\, e^{G_r - G_s} \boldsymbol{\nu}_s \right] =: O(\mathbf{q}, \mathbf{k}, \boldsymbol{\nu}, \mathbf{h}_n)_r,
\end{aligned}
```

where $\boldsymbol{\nu}$ is the library's `v_new` and $\mathbf{h}_n$ is the state entering chunk $n$.

## Tangent

Differentiating $A = (I + M)^{-1}$ gives $\dot A = -A \dot M A$, so the WY tangents need no explicit $\dot A$:

```math
\begin{aligned}
\dot M_{rs} &= \left[ \dot\beta_r (\mathbf{k}_r^\top \mathbf{k}_s) + \beta_r (\dot{\mathbf{k}}_r^\top \mathbf{k}_s + \mathbf{k}_r^\top \dot{\mathbf{k}}_s) + \beta_r (\mathbf{k}_r^\top \mathbf{k}_s)(\dot G_r - \dot G_s) \right] e^{G_r - G_s} \quad (r > s), \\
\dot{\mathbf{u}} &= A \left[ \dot\beta \mathbf{v} + \beta \dot{\mathbf{v}} - \dot M \mathbf{u} \right], \qquad
\dot{\mathbf{w}} = A \left[ e^{G} \big( (\dot\beta + \beta \dot G)\, \mathbf{k} + \beta \dot{\mathbf{k}} \big) - \dot M \mathbf{w} \right].
\end{aligned}
```

Across chunks, with $\dot{\mathbf{h}}_0 = 0$,

```math
\begin{aligned}
\dot{\boldsymbol{\nu}} &= \dot{\mathbf{u}} - \dot{\mathbf{w}} \mathbf{h}_n - \mathbf{w} \dot{\mathbf{h}}_n, \\
\dot{\mathbf{h}}_{n+1} &= e^{G_L} (\dot{\mathbf{h}}_n + \dot G_L \mathbf{h}_n) + \textstyle\sum_r e^{G_L - G_r} \left[ \mathbf{k}_r \big( \dot{\boldsymbol{\nu}}_r + (\dot G_L - \dot G_r) \boldsymbol{\nu}_r \big)^\top + \dot{\mathbf{k}}_r \boldsymbol{\nu}_r^\top \right].
\end{aligned}
```

$O$ is linear in $\mathbf{q}$, in $\mathbf{k}$ and in $(\boldsymbol{\nu}, \mathbf{h})$ separately, and differentiating its decays gives $\dot G_r \mathbf{o}_r$ minus the $\dot G_s \boldsymbol{\nu}_s$ terms, so

```math
\dot{\mathbf{o}} = O(\dot{\mathbf{q}}, \mathbf{k}, \boldsymbol{\nu}, \mathbf{h}) + O(\mathbf{q}, \dot{\mathbf{k}}, \boldsymbol{\nu}, 0) + O(\mathbf{q}, \mathbf{k}, \dot{\boldsymbol{\nu}} - \dot G \boldsymbol{\nu}, \dot{\mathbf{h}}) + \dot G\, \mathbf{o}.
```

With $P_{rs} = (\mathbf{q}_r^\top \mathbf{k}_s)\, e^{G_r - G_s}$ for $s \le r$ and $\dot P$ its tangent in $\mathbf{q}$ and $\mathbf{k}$, this is

```math
\dot{\mathbf{o}}_r = \text{scale} \left[ e^{G_r} \big( \dot{\mathbf{q}}_r^\top \mathbf{h}_n + \mathbf{q}_r^\top \dot{\mathbf{h}}_n + \dot G_r\, \mathbf{q}_r^\top \mathbf{h}_n \big) + \textstyle\sum_s \big( \dot P_{rs} + P_{rs} (\dot G_r - \dot G_s) \big) \boldsymbol{\nu}_s + \textstyle\sum_s P_{rs} \dot{\boldsymbol{\nu}}_s \right].
```

## Kernels

The primal runs through the library's forward kernels, which store $A$, $\mathbf{u}$, $\mathbf{w}$, $\boldsymbol{\nu}$ and $\mathbf{h}_n$.
The tangent directions are a grid dimension of every JVP kernel, so the primal tensors are never copied per direction.

| Kernel                                        | Computes                                                                                                    | Parallel over                           |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `chunk_gated_delta_rule_jvp_fwd_kernel_intra` | $\dot{\mathbf{u}}$, $\dot{\mathbf{w}}$                                                                      | chunks, heads, directions               |
| `chunk_gated_delta_rule_jvp_fwd_kernel_h`     | $\dot{\boldsymbol{\nu}}$, $\dot{\mathbf{h}}$ sequentially over chunks                                       | value blocks, heads, directions         |
| `chunk_gated_delta_rule_jvp_fwd_kernel_o`     | $\dot{\mathbf{o}}$, or with `fuse_output` the terms without $\dot{\mathbf{h}}$ and $\dot{\boldsymbol{\nu}}$ | value blocks, chunks, heads, directions |
| `chunk_gated_delta_rule_jvp_fwd_kernel_qk`    | $P$ for `fuse_output`; it does not depend on the direction                                                  | chunks, heads                           |

With `fuse_output=True`, the recurrence kernel writes $\text{scale} [e^{G_r} \mathbf{q}_r^\top \dot{\mathbf{h}}_n + \sum_s P_{rs} \dot{\boldsymbol{\nu}}_s]$ and the output kernel adds the remaining terms, so $\dot{\mathbf{h}}$ (`[D, B, NT, H, K, V]`) and $\dot{\boldsymbol{\nu}}$ are never stored.

## Reverse mode through the JVP

`fla::chunk_gated_delta_rule_jvp` in `custom_ops.py` registers a backward.
$\dot{\mathbf{o}} = J(x)\, \dot x$ is linear in the tangents, so the gradient of $\langle \boldsymbol{\lambda}, \mathbf{o} \rangle + \langle \dot{\boldsymbol{\lambda}}, \dot{\mathbf{o}} \rangle$ with respect to $\dot x$ is $J(x)^\top \dot{\boldsymbol{\lambda}}$: the library's backward with the directions folded into the batch.
With respect to the primal $x$ it is $J(x)^\top \boldsymbol{\lambda}$, from the same backward call, plus the second-order term $\partial_x \langle \dot{\boldsymbol{\lambda}}, J(x)\, \dot x \rangle$, which is computed by autograd through the fp32 PyTorch reference `chunk_gated_delta_rule_jvp_ref`.
That term is exact up to fp32 rounding but slower and more memory hungry than the kernels.
