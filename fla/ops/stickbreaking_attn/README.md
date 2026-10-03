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

The kernels process key blocks from nearest to farthest in base-2 log space. Define:

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
\delta_i &= \sum_{j \in \mathcal{V}_i} a_{ij} + dr_i r_i, \\
n_{ij} &= \sum_{\substack{\ell \in \mathcal{V}_i \\ \ell > j}} a_{i\ell}.
\end{aligned}
```

The logit gradient is:

```math
dz_{ij} = \frac{\partial\mathcal{L}}{\partial z_{ij}} = a_{ij} - \beta_{ij}(\delta_i - n_{ij}).
```

The backward computes $\delta_i$ from fp32 sums because using the rounded output in $(d\mathbf{o}_i)^\top\mathbf{o}_i$ loses precision during subtraction. It stores the remaining stick log and the sum over nearer keys before each key block, allowing the separate key/value gradient kernel to reconstruct each tile. Each key block has one owner, so gradient accumulation uses a fixed order without atomics.

[^1]: https://arxiv.org/abs/2410.17980
[^2]: https://github.com/shawntan/stickbreaking-attention
