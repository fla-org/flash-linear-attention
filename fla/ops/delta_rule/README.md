# Chunkwise-form Parallelism of DeltaNet

This section expands on the formulation presented in Appendix B of the DeltaNet paper.[^1]

To reduce notational clutter, we focus on the first chunk, denoting $\mathbf{S}^r=\mathbf{S}_{[1]}^r$. By partially expanding the recurrence, we have:
```math
\begin{equation}
\begin{aligned}
\mathbf{S}^r &= \underbrace{\left(\prod_{i=1}^r \mathbf{I} - \beta^i \bf{k}^i \bf{k}^{i\top} \right)}_{:= \mathbf{P}^r} \cdot\mathbf{S}^{0} + \overbrace{\sum_{i=1}^{r} \underbrace{\left(\prod_{j=i+1}^r \mathbf{I} - \beta^j \bf{k}^j \bf{k}^{j\top} \right)}_{:= \mathbf{P}_{i+1}^r}\beta^i \bf{k}^i\bf{v}^{i\top}}^{:=\mathbf{H}^r} \\
&=\mathbf{P}^r \cdot \mathbf{S}^{0} + \mathbf{H}^r
\end{aligned}
\end{equation}
```

where $\mathbf{P}_i^r$ involves cumulative products of generalized Householder matrices.
We abbreviate $\mathbf{P}_1^r$ as $\mathbf{P}^r$.
This can be optimized using the classical WY representation:
```math
\begin{equation}
\mathbf{P}^{r} = \mathbf{I} - \sum_{i=1}^{r}\bf{k}^i\bf{w}^{i\top}  \in \mathbb{R}^{d_k \times d_k};\qquad
\bf{w}^r = \beta^r \left(\bf{k}^r -  \sum_{i=1}^{r-1} \left(\bf{k}^{r\top}\bf{k}^i \right)\bf{w}^i  \right) \in \mathbb{R}^{d_k}
\end{equation}
```

We prove this by induction:
```math
\begin{align*}
\mathbf{P}^{r} &= \prod_{i=1}^r \mathbf{I} - \beta^i \bf{k}^i \bf{k}^{i\top} \\
&= \left(\mathbf{I} - \beta^r \bf{k}^r \bf{k}^{r\top}\right)\mathbf{P}^{r-1} \\
&= \left(\mathbf{I} - \beta^r \bf{k}^r \bf{k}^{r\top}\right)\left(\mathbf{I} - \sum_{i=1}^{r-1}\bf{k}^i\bf{w}^{i\top}\right) \\
&= \mathbf{I} - \sum_{i=1}^{r-1}\bf{k}^i\bf{w}^{i\top} - \beta^r \bf{k}^r \bf{k}^{r\top} + \beta^r\bf{k}^r \bf{k}^{r\top} \left(\sum_{i=1}^{r-1}\bf{k}^i\bf{w}^{i\top}\right) \\
&= \mathbf{I} - \sum_{i=1}^{r-1}\bf{k}^i\bf{w}^{i\top} - \beta^r \bf{k}^r \left(\bf{k}^{r} - \left(\sum_{i=1}^{r-1}\left(\bf{k}^{r\top} \bf{k}^i\right)\bf{w}^{i}\right) \right)^\top \\
&= \mathbf{I} - \sum_{i=1}^{r}\bf{k}^i\bf{w}^{i\top}
\end{align*}
```

Similarly, $\mathbf{H}^r$ can be represented as:
```math
\begin{equation}
\mathbf{H}^{r} = \sum_{i=1}^{r} \bf{k}^i \bf{u}^{i\top}  \in \mathbb{R}^{d_k \times d_v};\qquad \bf{u}^r = \beta^r \left(\bf{v}^r -  \sum_{i=1}^{r-1} \left(\bf{k}^{r\top}\bf{k}^i\right) \bf{u}^i \right)\in \mathbb{R}^{d_v}
\end{equation}
```

This can also be proven by induction:
```math
\begin{align*}
\mathbf{H}^{r} &= \sum_{i=1}^{r} \mathbf{P}_{i+1}^r \beta^i \bf{k}^i \bf{v}^{i\top}\\
&= \left(\mathbf{I} - \beta^r \bf{k}^r \bf{k}^{r\top}\right) \mathbf{H}^{r-1} +  \beta^r \bf{k}^r \bf{v}^{r\top}\\
&= \sum_{i=1}^{r-1}\bf{k}^i \bf{u}^{i\top} - \beta^r \bf{k}^r \bf{k}^{r\top} \sum_{i=1}^{r-1}\bf{k}^i \bf{u}^{i\top} +\beta^r \bf{k}^r \bf{v}^{r\top}\\
&= \sum_{i=1}^{r-1}\bf{k}^i \bf{u}^{i\top} + \bf{k}^r \left(\beta^r \bf{v}^{r\top}-\beta^r \bf{k}^{r\top} \sum_{i=1}^{r-1}\bf{k}^i \bf{u}^{i\top}\right) \\
&= \sum_{i=1}^{r-1}\bf{k}^i \bf{u}^{i\top} + \bf{k}^r \beta^r\left(\bf{v}^{r}-\sum_{i=1}^{r-1}\left(\bf{k}^{r\top}\bf{k}^{i}\right)\bf{u}^{i} \right)^\top \\
&=\sum_{i=1}^{r} \bf{k}^i \bf{u}^{i\top}
\end{align*}
```

In matrix form, $\mathbf{P}$ and $\mathbf{H}$ can be written as:
```math
\begin{equation}
\mathbf{P}=\mathbf{I}-\mathbf{K}^\top\mathbf{W} \in \mathbb{R}^{d_k \times d_k}, \qquad\mathbf{H}=\mathbf{K}^\top\mathbf{U} \in \mathbb{R}^{d_k\times d_v}
\end{equation}
```

Now we can derive the matrix form of $\mathbf{W}$ and $\mathbf{U}$:
```math
\begin{align*}
\mathbf{W} &= \mathrm{diag}(\beta) \mathbf{K} - \mathrm{tril}(\mathrm{diag}(\beta) \mathbf{K}\mathbf{K}^\top, -1)\mathbf{W}\\
\left(\mathbf{I} + \mathrm{tril}(\mathrm{diag}(\beta) \mathbf{K}\mathbf{K}^\top, -1)\right) \mathbf{W} &= \mathrm{diag}(\beta) \mathbf{K}
\end{align*}
```
A similar process holds for $\mathbf{U}$. We can further write $\mathbf{W}$ and $\mathbf{U}$ in matrix form:
```math
\begin{align*}
\mathbf{T} &= \left(\mathbf{I} + \mathrm{tril}\left(\mathrm{diag}(\beta)\mathbf{K} \mathbf{K}^\top,-1\right)\right)^{-1}\mathrm{diag}\left(\beta\right)\in \mathbb{R}^{C \times C}\\
\mathbf{W} &= \mathbf{T} \mathbf{K}\in \mathbb{R}^{C \times d_k}\\
\mathbf{U} &= \mathbf{T}\mathbf{V}\in \mathbb{R}^{C \times d_v}
\end{align*}
```

Substituting these back into the original equations yields a hardware-efficient chunkwise algorithm for DeltaNet that leverages matrix multiplications, enabling tensor core based GPU optimization:
```math
\begin{equation}
\begin{aligned}
\mathbf{S} &= \mathbf{P}\cdot\mathbf{S}^0 + \mathbf{H} \\
&= \mathbf{S}^0 + \mathbf{K}^\top (\mathbf{U} -\mathbf{W} \mathbf{S}^0) \in \mathbb{R}^{d_k \times d_v}\\
\mathbf{O} &= \mathbf{Q} \mathbf{S}^0 + (\mathbf{Q} \mathbf{K}^{\top} \odot \mathbf{M}) \left(\mathbf{U} - \mathbf{W} \mathbf{S}^0\right) \in \mathbb{R}^{C \times d_v}
\end{aligned}
\end{equation}
```

[^1]: https://arxiv.org/abs/2406.06484
