# HC, mHC, and iHC

A standard residual layer updates a single stream as $x_{l+1}=x_l+F_l(x_l)$. HC expands that state into $N$ streams, $X_l\in\mathbb{R}^{N\times D}$, while the layer $F_l$ still operates on a $D$-dimensional input. HC, mHC, and iHC differ in how the residual streams are connected across layers.

## HC

[HC](https://arxiv.org/abs/2409.19606) introduces three mappings: pre weights $p_l\in\mathbb{R}^N$ read from the streams, post weights $q_l\in\mathbb{R}^N$ distribute the layer output, and a residual matrix $M_l\in\mathbb{R}^{N\times N}$ mixes the existing streams:

```math
h_l=p_l^\top X_l,\qquad
b_l=F_l(h_l),\qquad
X_{l+1}=M_lX_l+q_lb_l.
```

The mappings can combine learned static coefficients with input-dependent coefficients. Expanding the residual state increases the information carried between layers without widening $F_l$. The direct residual path across several layers contains the product $M_{L-1}\cdots M_l$; unconstrained matrices can amplify or attenuate that path as depth increases.

## mHC

[mHC](https://arxiv.org/abs/2512.24880) constrains $M_l$ to the Birkhoff polytope of doubly stochastic matrices:

```math
M_l\geq 0,\qquad
M_l\mathbf{1}=\mathbf{1},\qquad
\mathbf{1}^\top M_l=\mathbf{1}^\top.
```

Each output stream is a convex combination of the input streams, and the stream mean is preserved by the residual mixing. Products of doubly stochastic matrices remain doubly stochastic, with spectral norm at most one. These properties apply to the direct residual path; they do not bound the Jacobian of the full layer, which also contains $F_l$ and input-dependent routing.

mHC obtains the mixing matrix by alternating row and column normalization of a positive matrix with Sinkhorn iterations. A finite number of iterations only approximates the constraint. Its pre/post mappings are nonnegative, with sigmoid gates for the readout and scaled sigmoid gates for the writeback.

## iHC

[iHC](https://x.com/unakar666/status/2027566637319393412) fixes the residual matrix to the identity, $M_l=I_N$:

```math
X_{l+1}=X_l+q_lF_l(p_l^\top X_l).
```

The direct residual path is then exactly an identity mapping at every depth. Learned pre/post routing still lets the streams interact through the layer output. This removes the learned residual-mixing matrix and its normalization; the residual update needs no $N\times N$ matrix multiplication. With one stream and $p_l=q_l=1$, the update reduces to the standard residual connection.

| Variant | Residual matrix $M_l$    | Direct residual path across layers        |
| ------- | ----------------------- | ----------------------------------------- |
| HC      | Learned, unconstrained  | Product of learned mixing matrices        |
| mHC     | Doubly stochastic       | Doubly stochastic under exact constraints |
| iHC     | Identity                | Identity                                  |
