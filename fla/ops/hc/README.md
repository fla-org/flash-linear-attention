# HC, mHC, and iHC

We follow the notation of [Attention Residuals, §2 and §6.2](https://arxiv.org/pdf/2603.15031#page=15): $l$ indexes layers, $h_l\in\mathbb{R}^d$ is the input to layer $l$, and $f_l$ is its transformation. Each attention or MLP sublayer counts as one layer. Batch and token indices are omitted.

## General form

Let $H_l\in\mathbb{R}^{d\times m}$ contain $m$ residual streams as columns. The vectors $\alpha_l,\beta_l\in\mathbb{R}^m$ give the pre (read) and post (write) weights; $A_{l+1}\in\mathbb{R}^{m\times m}$ mixes the existing streams on the transition from $H_l$ to $H_{l+1}$:

```math
h_l=H_l\alpha_l,\qquad
v_l=f_l(h_l),\qquad
H_{l+1}=H_lA_{l+1}+v_l\beta_l^\top.
```

The layer reads a weighted combination of the streams and writes its output back to them, while the residual transition carries the previous state forward. The weights may depend on the current state. FLA stores streams as rows: `x` corresponds to $H_l^\top$, and `mixing` to $A_{l+1}^\top$.

## HC, mHC, and iHC as special cases

Their residual transitions differ in one condition: **HC learns $A_l$ without stochastic constraints, mHC constrains it to be doubly stochastic, and iHC fixes it to $I_m$.**

| Variant    | Condition on $A_l$                                                            | Effect of the residual transition                  |
| ---------- | ----------------------------------------------------------------------------- | -------------------------------------------------- |
| [HC][hc]   | Learned $A_l\in\mathbb{R}^{m\times m}$                                        | Freely mixes, amplifies, or attenuates streams     |
| [mHC][mhc] | $A_l\geq0$, $A_l\mathbf{1}=\mathbf{1}$, $\mathbf{1}^\top A_l=\mathbf{1}^\top$ | Convex mixing that preserves the stream mean       |
| [iHC][ihc] | $A_l=I_m$                                                                     | Carries each stream unchanged before the new write |

mHC uses Sinkhorn normalization to obtain $A_l$; finite iterations approximate the doubly stochastic constraint. It also constrains the pre/post weights, using $\alpha_l=\sigma(\tilde\alpha_l)$ and $\beta_l=2\sigma(\tilde\beta_l)$, where the tildes denote learned logits. iHC retains learned pre/post weights, so streams still interact through $f_l(H_l\alpha_l)$ even though the residual transition is the identity.

Setting $m=1$ and $A_l=\alpha_l=\beta_l=1$ recovers the standard residual update $h_{l+1}=h_l+f_l(h_l)$.

[hc]: https://arxiv.org/abs/2409.19606
[mhc]: https://arxiv.org/abs/2512.24880
[ihc]: https://x.com/unakar666/status/2027566637319393412
