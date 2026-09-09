# Kalman Delta Networks

Kalman Delta Networks (KDN) use a recurrent uncertainty estimate to control writes to an associative memory. This integration covers the isotropic and diagonal variants from [the KDN project](https://github.com/ngocbh/kalman-delta-networks).

## Recurrence

For each head, the memory is a matrix $S_t\in\mathbb{R}^{K\times V}$. Both variants apply channel-wise retention before writing the value prediction error:

$$
\widehat S_t=\operatorname{Diag}(\alpha_t)S_{t-1},\qquad
e_t=v_t-k_t^\top\widehat S_t,\qquad
S_t=\widehat S_t+\kappa_t e_t^\top,\qquad
o_t=\mathrm{scale}\;q_t^\top S_t.
$$

The uncertainty state is precision $c_t$, the reciprocal of covariance. Process noise $\omega_t$ is nonnegative and observation noise $r_t$ is positive. The information scale $s$ defaults to the key width $K$.

For isotropic KDN, precision and process noise are scalars per head:

$$
\widehat p_t=\frac{\operatorname{mean}_i\alpha_{ti}^2}{c_{t-1}}+\omega_t,\qquad
\beta_t=\frac{\widehat p_t}{r_t+\widehat p_t\lVert k_t\rVert^2},\qquad
\kappa_t=\beta_t k_t,\qquad
c_t=\widehat p_t^{-1}+\frac{s\lVert k_t\rVert^2}{Kr_t}.
$$

For diagonal KDN, precision and process noise have one value per key channel:

$$
\widehat p_{ti}=\frac{\alpha_{ti}^2}{c_{t-1,i}}+\omega_{ti},\qquad
\kappa_{ti}=\frac{\widehat p_{ti}k_{ti}}{r_t+\sum_j\widehat p_{tj}k_{tj}^2},\qquad
c_{ti}=\widehat p_{ti}^{-1}+\frac{s k_{ti}^2}{r_t}.
$$

The diagonal approximation concerns uncertainty; both variants retain the full memory matrix. Information scale changes the posterior uncertainty and therefore future gains, without directly rescaling the current write.

## Proposed FLA integration

The public operators will accept sequence-first tensors and return `(output, (memory, precision))` when final state is requested. Initial states will use one entry per sequence, including packed sequences described by `cu_seqlens`.

The isotropic memory update reuses KDA with its Kalman-derived scalar `beta`. The diagonal update reuses DPLR with `k_write=kappa`, `a=-k*alpha`, `b=kappa`, and `gk=log(alpha)`. This ordering is essential because the prediction error uses the retained memory.

Only the uncertainty scan needs new Triton kernels. Its covariance recurrence is a fractional linear transform, allowing chunk summaries and inter-chunk carries. Independent PyTorch recurrences will check both outputs and gradients, including gradients through the two state components.

The planned layer and Transformers model provide both variants, causal short convolutions, learned retention/noise projections, and cached generation. Existing FLA operators and models retain their current behavior. Context parallelism and conversion of LitGPT checkpoints are outside this initial contribution.
