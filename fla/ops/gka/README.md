# Gated KalmaNet (GKA)

[Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression](https://arxiv.org/abs/2511.21016)

At every token $`t`$, GKA solves a ridge regression over all past key-value pairs, with exponentially decaying weights:

```math
\mathbf{S}_t = \arg\min_{\mathbf{S}}\ \lambda_t\lVert\mathbf{S}\rVert_F^2 + \sum_{i \le t}\eta_{t,i}\,\lVert\mathbf{S}\mathbf{k}_i - \mathbf{v}_i\rVert^2,
\qquad \eta_{t,i} = \prod_{j=i+1}^{t} \gamma_j .
```

Its solution, $`\mathbf{S}_t = \mathbf{U}_t(\mathbf{H}_t + \lambda_t\mathbf{I})^{-1}`$, depends only on two running states:

```math
\mathbf{H}_t = \gamma_t\,\mathbf{H}_{t-1} + \mathbf{k}_t\mathbf{k}_t^\top \in \mathbb{R}^{K\times K}, \qquad
\mathbf{U}_t = \gamma_t\,\mathbf{U}_{t-1} + \mathbf{v}_t\mathbf{k}_t^\top \in \mathbb{R}^{V\times K} .
```

The output mixes the ridge solution with the query through a gate $`\alpha_t`$:

```math
\mathbf{x}_t = (\mathbf{H}_t + \lambda_t\mathbf{I})^{-1}\mathbf{q}_t, \qquad
\tilde{\mathbf{q}}_t = \mathbf{q}_t + \alpha_t(\mathbf{x}_t - \mathbf{q}_t), \qquad
\mathbf{o}_t = \mathbf{U}_t\tilde{\mathbf{q}}_t .
```

The decay $`\gamma_t = e^{g_t} \in (0, 1]`$ and the gate $`\alpha_t \in [0, 1]`$ are scalars per head.
With $`\alpha_t = 1`$, the output is the ridge prediction $`\mathbf{S}_t\mathbf{q}_t`$. With $`\alpha_t = 0`$, GKA reduces to simple GLA, so the gate lets the model fall back on the GLA readout when the fixed number of Chebyshev steps has not converged.

The ridge strength $`\lambda_t = a\,\lVert\mathbf{H}_t\rVert_F`$ adapts to the state, for a constant $`a > 0`$.
This bounds the condition number of $`\mathbf{H}_t + \lambda_t\mathbf{I}`$ by $`(a + 1)/a`$.

## Chebyshev iteration

The matrix $`\mathbf{A}_t = \mathbf{H}_t + \lambda_t\mathbf{I}`$ is symmetric positive definite, with eigenvalues in $`[\mu, L] = [\lambda_t, \lambda_t + \lVert\mathbf{H}_t\rVert_F]`$.
The kernels solve $`\mathbf{A}_t\mathbf{x} = \mathbf{q}_t`$ with a fixed number of Chebyshev iteration steps.
Starting from $`\omega_0 = 2`$, $`\boldsymbol{\xi}_{-1} = \mathbf{0}`$ and $`\boldsymbol{\xi}_0 = 2\mathbf{q}_t/(L + \mu)`$, step $`n \ge 1`$ computes

```math
\omega_n = \frac{4}{4 - \rho^2\omega_{n-1}}, \qquad
\boldsymbol{\xi}_n = \boldsymbol{\xi}_{n-1} - \frac{2\omega_n}{L + \mu}\left(\mathbf{A}_t\boldsymbol{\xi}_{n-1} - \mathbf{q}_t\right) + (\omega_n - 1)\left(\boldsymbol{\xi}_{n-1} - \boldsymbol{\xi}_{n-2}\right),
```

where $`\rho = (L - \mu)/(L + \mu)`$.
The last iterate is used as $`\mathbf{x}_t`$.
Unlike conjugate gradients, Chebyshev iteration needs no inner products, and the paper shows that it is more stable in low precision.

## Chunk-wise computation

The chunked kernel computes $`\mathbf{H}_t\boldsymbol{\xi}_{n-1}`$ and $`\lVert\mathbf{H}_t\rVert_F`$ without materializing $`\mathbf{H}_t`$.
Let $`G_t`$ be the cumulative sum of $`g`$ within the chunk, and $`\mathbf{H}_{[c]}`$ the state at the start of the chunk. Then

```math
\mathbf{H}_t = e^{G_t}\mathbf{H}_{[c]} + \sum_{j \le t} e^{G_t - G_j}\,\mathbf{k}_j\mathbf{k}_j^\top, \qquad
\lVert\mathbf{H}_t\rVert_F^2 = \Big\lVert\sum_{j \le t} e^{G_t - G_j}\mathbf{k}_j\mathbf{k}_j^\top\Big\rVert_F^2
+ e^{2G_t}\lVert\mathbf{H}_{[c]}\rVert_F^2 + 2e^{G_t}\sum_{j \le t} e^{G_t - G_j}\,\mathbf{k}_j^\top\mathbf{H}_{[c]}\mathbf{k}_j ,
```

where the sums run over the tokens $`j`$ of the current chunk.

## Backward pass

The backward pass uses implicit differentiation.
It treats $`\mathbf{x}_t`$ as the exact solution of $`\mathbf{A}_t\mathbf{x} = \mathbf{q}_t`$, so the gradients are those of the exact ridge solve rather than of the Chebyshev iterations.
Because $`\mathbf{A}_t`$ is symmetric, $`\mathrm{d}\mathbf{q}_t = \mathbf{A}_t^{-1}\mathrm{d}\mathbf{x}_t`$ reuses the forward solve.
The gradients w.r.t. $`\mathbf{H}_t`$ and $`\lambda_t`$ are derived in the paper.
