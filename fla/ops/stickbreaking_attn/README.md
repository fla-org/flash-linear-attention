# Stick-breaking attention

For each visible key $j$, query $i$ assigns the weight

$$
z_{ij} = \mathrm{scale}\, q_i^\top k_j, \qquad
\beta_{ij} = \sigma(z_{ij}), \qquad
A_{ij} = \beta_{ij} \prod_{\ell > j} (1 - \beta_{i\ell}).
$$

The product includes only visible keys: $j < i$ by default, or $j \leq i$ with `attend_current=True`. The nearest key takes its share first. The output is $o_i = \sum_j A_{ij}v_j$, and the remaining stick is $r_i = \prod_j(1 - \beta_{ij})$.

The kernels use base-2 log space. Within a key block, let $s_{ij} = z_{ij}\log_2 e$ and $b_{ij} = -\operatorname{softplus}_2(s_{ij})$. If `b_acc` is the log of the stick left by nearer blocks, then

$$
\log_2 A_{ij} = s_{ij} + \sum_{\ell \geq j} b_{i\ell} + \texttt{b\_acc}_i.
$$

For backward, define $a_{ij} = A_{ij}\langle do_i, v_j\rangle$, $c_i = \sum_j a_{ij} + dr_i r_i$, and $n_{ij} = \sum_{\ell > j}a_{i\ell}$. The logit gradient is

$$
dz_{ij} = a_{ij} - \beta_{ij}(c_i - n_{ij}).
$$

The backward computes $c_i$ from fp32 sums because using the rounded output in $\langle do_i, o_i\rangle$ loses precision during subtraction. It stores the remaining stick log and the sum over nearer keys before each key block, allowing the separate key/value gradient kernel to reconstruct each tile. Each key block has one owner, so gradient accumulation uses a fixed order without atomics.

See [Tan et al., 2024](https://arxiv.org/abs/2410.17980) and the [authors' implementation](https://github.com/shawntan/stickbreaking-attention).
