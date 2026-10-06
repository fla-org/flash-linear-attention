# Stick-Breaking Attention

This note describes the formulation and parallel implementation of stick-breaking attention.[^1][^2]

## Formulation

Let $\mathcal{V}_i$ be the visible keys for query $i$: $j < i$ by default, or $j \leq i$ with `attend_current=True`. The attention weights are:

```math
z_{ij} = \mathrm{scale}\, \mathbf{q}_i^\top \mathbf{k}_j, \qquad
\beta_{ij} = \sigma(z_{ij}), \qquad
A_{ij} = \beta_{ij} \prod_{\substack{\ell \in \mathcal{V}_i \\ \ell > j}} (1 - \beta_{i\ell}), \qquad j \in \mathcal{V}_i.
```

The nearest key takes its share first. The output and remaining stick are:

```math
\mathbf{o}_i = \sum_{j \in \mathcal{V}_i} A_{ij}\mathbf{v}_j, \qquad
r_i = \prod_{j \in \mathcal{V}_i}(1 - \beta_{ij}) = 1 - \sum_{j \in \mathcal{V}_i} A_{ij}.
```

## Parallel Forward Pass

The forward kernel processes key blocks from nearest to farthest in base-2 log space. Define:

```math
s_{ij} = z_{ij}\log_2 e, \qquad
b_{ij} = -\log_2(1 + 2^{s_{ij}}) = \log_2(1 - \beta_{ij}).
```

For a key block $\mathcal{B}$, let $\alpha_i$ be the log of the stick left by nearer blocks, initialized to zero. The weights within the block are:

```math
\log_2 A_{ij} = s_{ij} + \sum_{\substack{\ell \in \mathcal{B} \cap \mathcal{V}_i \\ \ell \geq j}} b_{i\ell} + \alpha_i.
```

After accumulating the block's contribution to $\mathbf{o}_i$, update $\alpha_i$ by adding the sum of $b_{ij}$ over its visible keys. This computes the output and $r_i = 2^{\alpha_i}$ without materializing the full attention matrix.

## Backward Pass

Let $d\mathbf{o}_i = \partial\mathcal{L}/\partial\mathbf{o}_i$ and $dr_i = \partial\mathcal{L}/\partial r_i$. Define:

```math
\begin{aligned}
a_{ij} &= A_{ij}(d\mathbf{o}_i)^\top\mathbf{v}_j, \\
\delta_{ij} &= \sum_{\substack{\ell \in \mathcal{V}_i \\ \ell < j}} a_{i\ell} + dr_i r_i.
\end{aligned}
```

Here $a_{ij} = \partial\mathcal{L}/\partial\log\beta_{ij}$, and $\delta_{ij} = \partial\mathcal{L}/\partial\log(1 - \beta_{ij})$ because every farther weight and the remaining stick carry the factor $1 - \beta_{ij}$. The logit gradient is:

```math
dz_{ij} = \frac{\partial\mathcal{L}}{\partial z_{ij}} = (1 - \beta_{ij})\,a_{ij} - \beta_{ij} \delta_{ij}.
```

The kernels take $1 - \beta_{ij} = 2^{b_{ij}}$ and $\beta_{ij} = 2^{s_{ij} + b_{ij}}$, and accumulate $\delta_{ij}$ by additions from the farthest key. The equivalent form $a_{ij} - \beta_{ij}(a_{ij} + \delta_{ij})$, with $a_{ij} + \delta_{ij}$ taken as a row total minus the nearer terms, cancels as $\beta_{ij} \to 1$ and loses the $(1 - \beta_{ij})\,a_{ij}$ term entirely once $\beta_{ij}$ rounds to 1, which `softplus2` makes exact for $z_{ij} > 15\ln 2 \approx 10.4$.

The query gradient kernel walks the key blocks twice: nearest first to store the remaining stick log before each block, then farthest first to store $\delta_{ij}$ at the start of each block. These snapshots let the separate key/value gradient kernel reconstruct each tile. Each key block has one owner, so gradient accumulation uses a fixed order without atomics.

[^1]: https://arxiv.org/abs/2410.17980
[^2]: https://github.com/shawntan/stickbreaking-attention
