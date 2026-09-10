# Generalized Delta Rule

In delta rule we have the recurrence:

```math
\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I}-\beta_t \mathbf{k}_t\mathbf{k}_t^T) + \beta_t \mathbf{v}_t\mathbf{k}_t^T
```

This repository implements a delta rule variant where $\mathbf{I}$ is not necessarily an identity matrix; $\mathbf{k}_t$ in $\mathbf{I} - \beta_t \mathbf{k}_t\mathbf{k}_t^T$ might be different from input $\mathbf{k}_t$ in $\mathbf{v}_t\mathbf{k}_t^T$.

## IPLR (Identity Plus Low Rank)

The first variant is IPLR, where we have:

```math
\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I}+\mathbf{a}_t\mathbf{b}_t^T) + \mathbf{v}_t\mathbf{k}_t^T
```

When $\mathbf{a}_t = -\beta_t \mathbf{k}_t$, $\mathbf{b}_t = \mathbf{k}_t$, $\mathbf{v}_t= \beta_t \mathbf{v}_t$, we recover the original delta rule. Since here the transition matrix is identity-plus-low-rank, we refer to this variant as IPLR.

### Numerical Stability

$\mathbf{a}_t$ and $\mathbf{b}_t$ must be in opposite directions, that is, $\mathbf{b}_t = \lambda_t \mathbf{a}_t$ where $\lambda_t < 0$. For an understanding of why this is necessary, you can derive the eigenvalues of the transition matrix.

## DPLR (Diagonal Plus Low Rank)

The second variant is DPLR, where we have:

```math
\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{D}_t+\mathbf{a}_t\mathbf{b}_t^T) + \mathbf{v}_t\mathbf{k}_t^T
```

Here, $\mathbf{I}$ is replaced by a diagonal matrix $\mathbf{D}_t$. This transition matrix structure has been utilized in RWKV7.

## Efficient Chunkwise Implementation
The original [technical note](https://drive.google.com/file/d/1qqc6THTRc2bw-LtwsbGNxNDw00sNzi5M/view?usp=sharing) on chunking DPLR contains minor mathematical inconsistencies. Below, we re-do the computations.

If you have questions about or comments about the below derivations, feel free to [reach out](https://phnazari.github.io).

Our goal is to show how to efficiently compute the DPLR representation
$$
    \mathbf S_t = \mathbf S_{t-1} \left( \mathbf D_t + \mathbf a_t \mathbf b_t^\top \right) + \mathbf v_t \mathbf k_t^\top
$$
for vectors $\mathbf a_t, \mathbf b_t, \mathbf v_t, \mathbf k_t \in \mathbb R^d$ and matrices $\mathbf D_t \in \mathbb R^{d, d}$.

In particular, if the $\mathbf D_t$ are diagonal matrices, this identity provides the WY representation for products of DPLR matrices.

### $WY$ Representation for $P_t$
Let $\mathbf \Gamma_i^t \coloneqq \prod_{j=i}^t \mathbf D_j$. Then
```math
\begin{equation*}
    \mathbf P_t = \mathbf \Gamma_1^t + \left( \sum_{i=1}^t \mathbf w_i \mathbf b_i^\top \mathbf \Gamma_{i+1}^{t} \right)
\end{equation*}
```
with
```math
    \mathbf w_i = \begin{cases}
        \mathbf a_1, & i=1 \\
        \mathbf \Gamma_1^{i-1} \mathbf a_i + \sum_{j=1}^{i-1} \mathbf w_j \mathbf b_j^\top \mathbf \Gamma_{j+1}^{i-1} \mathbf a_i, & i \geq 2.
    \end{cases}
```
where we define $\mathbf \Gamma_m^{n} \coloneqq \mathbf I$ for $m > n$.

We proceed by induction. The base case is quickly established for $t=1$, considering that $\mathbf \Gamma_1^1 = D_1$ and $\mathbf \Gamma_2^1 = \mathbf I$.

For the induction step, note that
```math
\begin{align*}
    \mathbf P_{t+1} &= \mathbf P_t (\mathbf D_{t+1} + \mathbf a_{t+1} \mathbf b_{t+1}^\top) \\
    &= \left( \mathbf \Gamma_{1}^t + \sum_{i=1}^t\mathbf w_i \mathbf b_i^\top \mathbf \Gamma_{i+1}^{t}  \right) \mathbf D_{t+1} + \left( \mathbf \Gamma_1^t + \sum_{i=1}^t \mathbf w_i \mathbf b_i^\top \mathbf \Gamma_{i+1}^{t} \right)\mathbf a_{t+1} \mathbf b_{t+1}^\top\\
    &= \mathbf \Gamma_{1}^{t+1} + \sum_{i=1}^t \mathbf w_i \mathbf b_i^\top \mathbf \Gamma_{i+1}^{t+1} + \underbrace{\left(\mathbf \Gamma_1^{t} \mathbf a_{t+1} + \sum_{i=1}^t \mathbf w_i \mathbf b_i^\top \mathbf \Gamma_{i+1}^{t}\mathbf a_{t+1}\right)}_{\eqqcolon \mathbf w_{t+1}} \mathbf b_{t+1}^\top \\
    &= \mathbf \Gamma_1^{t+1} + \sum_{i=1}^{t+1} \mathbf w_i \mathbf b_i^\top \mathbf \Gamma_{i+1}^{t+1},
\end{align*}
```
where we used $\mathbf \Gamma_{t+2}^{t+1} = \mathbf I$ in the last step.

### $WY$ Representation for $S_t$
The $WY$ representation for $\mathbf S_t$ reads
```math
    \mathbf S_t = \sum_{i=1}^t (\mathbf v_i \mathbf k_i^\top + \mathbf u_i \mathbf b_i^\top) \mathbf \Gamma_{i+1}^{t}
```
where
```math
    \mathbf u_i = \begin{cases}
        0, & i=1 \\
        \sum_{j=1}^{i-1} \left( \mathbf v_j \mathbf k_j^\top + \mathbf u_j \mathbf b_j^\top \right) \mathbf \Gamma_{j+1}^{i-1} \mathbf a_i, & i \geq 2.
    \end{cases}
```
We again show this claim by induction. The base case $t=1$ is clear, once we realize that $\mathbf u_1 \coloneqq 0$ and $\mathbf \Gamma_2^1 \coloneqq \mathbf I$.

For the induction step, we compute
```math
\begin{align*}
    \mathbf S_{t+1} &= \mathbf S_t (\mathbf D_{t+1} + \mathbf a_{t+1} \mathbf b_{t+1}^\top) + \mathbf v_{t+1} \mathbf k_{t+1}^\top \\
    &= \left[\sum_{i=1}^t (\mathbf v_i \mathbf k_i^\top + \mathbf u_i \mathbf b_i^\top) \mathbf \Gamma_{i+1}^{t}\right] \left(\mathbf D_{t+1} + \mathbf a_{t+1}\mathbf b_{t+1}^\top\right) + \mathbf v_{t+1} \mathbf k_{t+1}^\top \\
    &= \sum_{i=1}^t  (\mathbf v_i \mathbf k_i^\top + \mathbf u_i \mathbf b_i^\top) \mathbf \Gamma_{i+1}^{t+1} + \underbrace{\left[ \sum_{i=1}^t  \left(\mathbf v_i\mathbf k_i^\top + \mathbf u_i\mathbf b_i^\top \right)\mathbf \Gamma_{i+1}^{t}\mathbf a_{t+1} \right]}_{\eqqcolon \mathbf u_{t+1}}\mathbf b_{t+1}^\top + \mathbf v_{t+1}\mathbf k_{t+1}^\top \\
    &= \sum_{i=1}^{t+1} (\mathbf v_i \mathbf k_i^\top + \mathbf u_i \mathbf b_i^\top ) \mathbf \Gamma_{i+1}^{t+1},
\end{align*}
```
where we again used $\mathbf \Gamma_{t+2}^{t+1} = \mathbf I$.
