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

For detailed information about efficient chunkwise implementation, please refer to our [technical note](https://drive.google.com/file/d/1rJbO3dU4fe7OKG3w7Yg058z_BNIuavNF/view?usp=sharing).
