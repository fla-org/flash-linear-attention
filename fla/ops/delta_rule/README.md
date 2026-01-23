# Chunkwise-form Parallelism of DeltaNet

This section expands on the formulation presented in Appendix B of the DeltaNet paper.[^1]

To reduce notational clutter, we focus on the first chunk, denoting $\mathbf{S}^r=\mathbf{S}_{[1]}^r$. By unrolling the recurrence $`S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top`$, we have:
```math
\begin{equation}
\begin{aligned}
\mathbf{S}^r &= \mathbf{S}^{0}\underbrace{\left(\prod_{i=1}^r (\mathbf{I} - \beta^i \bf{k}^i \bf{k}^{i\top}) \right)}_{:= \mathbf{P}^r} + \overbrace{\sum_{i=1}^{r} \beta^i \bf{v}^i\bf{k}^{i\top}\underbrace{\left(\prod_{j=i+1}^r (\mathbf{I} - \beta^j \bf{k}^j \bf{k}^{j\top}) \right)}_{:= \mathbf{P}_{i+1}^r}}^{:=\mathbf{H}^r} \\
&=\mathbf{S}^{0} \mathbf{P}^r + \mathbf{H}^r
\end{aligned}
\end{equation}
```

where $\mathbf{P}_i^r$ involves cumulative products of generalized Householder matrices.
We abbreviate $\mathbf{P}_1^r$ as $\mathbf{P}^r$.
This can be optimized using the classical WY representation:
```math
\begin{equation}
\mathbf{P}^{r} = \mathbf{I} - \sum_{i=1}^{r}\bf{w}^i\bf{k}^{i\top}   \;\;\in \mathbb{R}^{d_k \times d_k};\qquad
\bf{w}^r = \beta^r \left(\bf{k}^r -  \sum_{i=1}^{r-1} \left(\bf{k}^{i\top}\bf{k}^r \right)\bf{w}^i  \right) \;\;\in \mathbb{R}^{d_k}
\end{equation}
```

We prove this by induction:
```math
\begin{align*}
\mathbf{P}^{r} &= \prod_{i=1}^r (\mathbf{I} - \beta^i \bf{k}^i \bf{k}^{i\top}) \\
&= \mathbf{P}^{r-1}\left(\mathbf{I} - \beta^r \bf{k}^r \bf{k}^{r\top}\right) \\
&= \left(\mathbf{I} - \sum_{i=1}^{r-1}\bf{w}^i\bf{k}^{i\top}\right)\left(\mathbf{I} - \beta^r \bf{k}^r \bf{k}^{r\top}\right) \\
&= \mathbf{I} - \sum_{i=1}^{r-1}\bf{w}^i\bf{k}^{i\top} - \beta^r \bf{k}^r \bf{k}^{r\top} + \left(\sum_{i=1}^{r-1}\bf{w}^i\bf{k}^{i\top}\right) \beta^r \bf{k}^r \bf{k}^{r\top} \\
&= \mathbf{I} - \sum_{i=1}^{r-1}\bf{w}^i\bf{k}^{i\top} - \left(\beta^r \bf{k}^r - \beta^r\sum_{i=1}^{r-1}\bf{w}^i\left(\bf{k}^{i\top} \bf{k}^r\right) \right) \bf{k}^{r\top} \\
&= \mathbf{I} - \sum_{i=1}^{r}\bf{w}^i\bf{k}^{i\top}
\end{align*}
```

Similarly, $\mathbf{H}^r$ can be represented as:
```math
\begin{equation}
\mathbf{H}^{r} = \sum_{i=1}^{r} \bf{u}^i \bf{k}^{i\top}   \;\;\in \mathbb{R}^{d_v \times d_k};\qquad \bf{u}^r = \beta^r \left(\bf{v}^r -  \sum_{i=1}^{r-1} \left(\bf{k}^{i\top}\bf{k}^r\right) \bf{u}^i \right) \;\;\in \mathbb{R}^{d_v}
\end{equation}
```

This can also be proven by induction:
```math
\begin{align*}
\mathbf{H}^{r} &= \sum_{i=1}^{r} \beta^i \bf{v}^i \bf{k}^{i\top}\mathbf{P}_{i+1}^r\\
&= \mathbf{H}^{r-1} \left(\mathbf{I} - \beta^r \bf{k}^r \bf{k}^{r\top}\right) +  \beta^r \bf{v}^r \bf{k}^{r\top}\\
&= \left(\sum_{i=1}^{r-1}\bf{u}^i \bf{k}^{i\top}\right)\left(\mathbf{I} - \beta^r \bf{k}^r \bf{k}^{r\top}\right) +\beta^r \bf{v}^r \bf{k}^{r\top}\\
&= \sum_{i=1}^{r-1}\bf{u}^i \bf{k}^{i\top} - \left(\sum_{i=1}^{r-1}\bf{u}^i \bf{k}^{i\top}\right)\beta^r \bf{k}^r \bf{k}^{r\top} +\beta^r \bf{v}^r \bf{k}^{r\top} \\
&= \sum_{i=1}^{r-1}\bf{u}^i \bf{k}^{i\top} + \left(\beta^r \bf{v}^{r}-\beta^r \sum_{i=1}^{r-1}\bf{u}^{i}\left(\bf{k}^{i\top}\bf{k}^{r}\right)\right) \bf{k}^{r\top} \\
&=\sum_{i=1}^{r} \bf{u}^i \bf{k}^{i\top}
\end{align*}
```


Since $\mathbf{P}$ and $\mathbf{H}$ are sums of outer products, they can be expressed in matrix form:
```math
\begin{equation}
\mathbf{P}=\mathbf{I}-\mathbf{W}^\top\mathbf{K}  \;\;\in \mathbb{R}^{d_k \times d_k}, \qquad\mathbf{H}=\mathbf{U}^\top\mathbf{K} \;\;\in \mathbb{R}^{d_v\times d_k}
\end{equation}
```


As derived in Appendix B.2 of the paper, the matrices $\mathbf{W}$ and $\mathbf{U}$ can be solved efficiently by converting their recursive definitions into linear triangular systems.
```math
\begin{align*}
\mathbf{W} &= \left(\mathbf{I} + \mathrm{tril}(\mathrm{diag}(\beta) \mathbf{K}\mathbf{K}^\top, -1)\right)^{-1}\mathrm{diag}(\beta) \mathbf{K}\\
\mathbf{U} &= \left(\mathbf{I} + \mathrm{tril}(\mathrm{diag}(\beta) \mathbf{K}\mathbf{K}^\top, -1)\right)^{-1}\mathrm{diag}(\beta) \mathbf{V}
\end{align*}
```

This can be written more compactly using the UT transform matrix $\mathbf{T}$:
```math
\begin{align*}
\mathbf{T} &= \left(\mathbf{I} + \mathrm{tril}\left(\mathrm{diag}(\beta)\mathbf{K} \mathbf{K}^\top,-1\right)\right)^{-1}\mathrm{diag}\left(\beta\right) \;\;\in \mathbb{R}^{C \times C}\\
\mathbf{W} &= \mathbf{T} \mathbf{K} \;\;\in \mathbb{R}^{C \times d_k}\\
\mathbf{U} &= \mathbf{T}\mathbf{V} \;\;\in \mathbb{R}^{C \times d_v}\\
\mathbf{P} &= \mathbf{I} - \mathbf{K}^\top T^\top \mathbf{K} \;\;\in \mathbb{R}^{d_k \times d_k}\\
\mathbf{H} &= \mathbf{V}^\top T^\top \mathbf{K} \;\;\in \mathbb{R}^{d_v \times d_k}
\end{align*}
```

Substituting these compact forms back into the state update and output equations yields the hardware-efficient chunkwise algorithm. For a given chunk $[t]$ with initial state $`\mathbf{S}^0_{[t]}`$, the final state $`\mathbf{S}_{[t+1]}`$ and output $`\mathbf{O}_{[t]}`$ are:
```math
\begin{equation}
\begin{aligned}
\mathbf{S}_{[t+1]} &= \mathbf{S}^0_{[t]} \mathbf{P} + \mathbf{H} \\
&= \mathbf{S}^0_{[t]} + \left(\mathbf{U}^\top -\mathbf{S}^0_{[t]} \mathbf{W}^\top\right) \mathbf{K} \\
&= \mathbf{S}^0_{[t]} + \left(\mathbf{V}^\top - \mathbf{S}^0_{[t]} \mathbf{K}^\top\right) \mathbf{T}^\top \mathbf{K} \;\;\in\mathbb{R}^{d_v \times d_k} \\
\mathbf{O}_{[t]} &= \mathbf{Q}_{[t]}\mathbf{S}_{[t+1]}^\top \\ &= \mathbf{Q}_{[t]} \left(\mathbf{S}^0_{[t]}\right)^\top + \left(\mathbf{Q}_{[t]} \mathbf{K}^{\top} \odot \mathbf{M}\right) \left(\mathbf{U} - \mathbf{W} \left(\mathbf{S}^0_{[t]}\right)^\top \right) \\
&= \mathbf{Q}_{[t]} \left(\mathbf{S}^0_{[t]}\right)^\top + \left(\mathbf{Q}_{[t]} \mathbf{K}^{\top} \odot \mathbf{M}\right)\mathbf{T} \left(\mathbf{V} - \mathbf{K}(\mathbf{S}^0_{[t]})^\top\right) \;\;\in \mathbb{R}^{C \times d_v}
\end{aligned}
\end{equation}
```


In this final form, the intra-chunk recurrence has been transformed into a series of efficient matrix multiplications (e.g., computing $\mathbf{T}$, $\mathbf{W}$, $\mathbf{U}$, and the final output), which can be highly optimized on modern hardware like GPUs.

[^1]: https://arxiv.org/abs/2406.06484
