# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Copyright (c) 2026, Pieter-Jan Hoedt

import torch


def naive_recurrent_mlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    scale: float | None = None,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    output_final_state: bool = True,
    max_normalisation: bool | None = True,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None]:
    dtype = v.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, T, H, K, V = *q.shape, v.shape[-1]
    h = torch.empty_like(v)

    if initial_state is None:
        c = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
        n = None
        m = torch.full((B, H), -float("inf"), device=q.device, dtype=torch.float32)
        if max_normalisation is not None:
            n = torch.zeros((B, H, K), device=q.device, dtype=torch.float32)
    else:
        c, n, m = initial_state

    for t in range(T):
        log_f = m + torch.nn.functional.logsigmoid(f[:, t])
        m = torch.maximum(log_f, i[:, t]).detach()
        i_gate = torch.exp(i[:, t] - m).unsqueeze(dim=-1)
        f_gate = torch.exp(log_f - m).unsqueeze(dim=-1)

        kt = scale * k[:, t]
        c = f_gate.unsqueeze(dim=-1) * c
        c = c + torch.unsqueeze(i_gate * kt, dim=-1) * v[:, t].unsqueeze(dim=-2)
        ht = torch.sum(c * q[:, t].unsqueeze(dim=-1), dim=-2)
        if max_normalisation is not None:
            n = f_gate * n + i_gate * kt
            z = torch.sum(n * q[:, t], dim=-1, keepdim=True)
            if max_normalisation:
                _z = torch.maximum(torch.abs(z), torch.exp(-m.unsqueeze(dim=-1)) + eps)
            else:
                _z = z + eps

            ht = ht / _z

        h[:, t] = ht

    return h.to(dtype), (c, n, m) if output_final_state else None


def naive_chunk_mlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    scale: float | None = None,
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None = None,
    output_final_state: bool = True,
    max_normalisation: bool | None = True,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor] | None]:
    q, k, v, i, f = (x.transpose(1, 2) for x in (q, k, v, i, f))
    dtype = v.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, H, T, K, V = *q.shape, v.shape[-1]
    h = torch.empty_like(v)

    if initial_state is None:
        c = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
        n = None
        m = torch.full((B, H), -float("inf"), device=q.device, dtype=torch.float32)
        if max_normalisation is not None:
            n = torch.zeros((B, H, K), device=q.device, dtype=torch.float32)
    else:
        c, n, m = initial_state

    m = m.unsqueeze(dim=-1)
    if n is not None:
        n = n.unsqueeze(dim=-1)

    log_f = torch.nn.functional.logsigmoid(f)
    for t, qc, kc, vc, ic, log_fc in zip(range(0, T, chunk_size), *(
        torch.split(x, chunk_size, dim=2) for x in (q, k, v, i, log_f)
    )):
        log_f_matrix = _build_forget_matrix(log_fc)
        log_d = log_f_matrix + ic.unsqueeze(-2)
        max_log_d = torch.amax(log_d, dim=-1, keepdim=True)

        # recurrent gating
        log_prod_f = log_fc[..., :1] + log_f_matrix[..., 0]  # torch.cumsum(log_fc, dim=-1)
        log_f_rec = torch.unsqueeze(m + log_prod_f, dim=-1)
        mc = torch.maximum(max_log_d, log_f_rec).detach()
        fgate = torch.exp(log_f_rec - mc)

        # parallel computation
        d_scaled = scale * torch.exp(log_d - mc)
        e_par = d_scaled * (qc @ kc.transpose(-1, -2))
        s_par = e_par.to(dtype=vc.dtype) @ vc

        # incorporate recurrent state
        q_fgate = fgate * qc
        s_rec = q_fgate @ c

        # recurrent state output
        kc_gated = kc.transpose(-1, -2) * d_scaled[..., -1:, :]
        c = fgate[..., -1:, :] * c + kc_gated.to(dtype=vc.dtype) @ vc
        m = mc[..., -1, :]

        # combine results
        hc_tmp = s_rec + s_par

        if max_normalisation is not None:
            z_par = torch.sum(e_par, dim=-1, keepdim=True)
            z_rec = q_fgate @ n
            n = fgate[..., -1:, :] * n + torch.sum(kc_gated, dim=-1, keepdim=True)
            z_both = z_par + z_rec
            if max_normalisation:
                _z = torch.maximum(torch.abs(z_both), torch.exp(-mc) + eps)
            else:
                _z = z_both + eps

            hc_tmp = hc_tmp / _z

        h[:, :, t:t + chunk_size] = hc_tmp

    m = m.squeeze(dim=-1)
    if n is not None:
        n = n.squeeze(dim=-1)
    return h.to(dtype).transpose(1, 2), (c, n, m) if output_final_state else None


def naive_parallel_mlstm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    scale: float | None = None,
    eps: float = 1e-6,
    max_normalisation: bool | None = True,
) -> torch.Tensor:
    q, k, v, i, f = (x.transpose(1, 2) for x in (q, k, v, i, f))
    if scale is None:
        scale = q.shape[-1] ** -0.5

    log_f = torch.nn.functional.logsigmoid(f)
    log_f_matrix = _build_forget_matrix(log_f)
    log_scale = log_f_matrix + i.unsqueeze(dim=-2)
    max_log_scale = torch.amax(log_scale, dim=-1, keepdim=True).detach()

    qk = scale * q @ k.transpose(-1, -2)
    e = torch.exp(log_scale - max_log_scale) * qk
    h = e @ v
    if max_normalisation is not None:
        z = torch.sum(e, dim=-1, keepdim=True)
        if max_normalisation:
            _z = torch.maximum(torch.abs(z), torch.exp(-max_log_scale) + eps)
        else:
            _z = z + eps

        h = h / _z

    return h.transpose(1, 2)


def _build_forget_matrix(log_f: torch.Tensor) -> torch.Tensor:
    """
    Build logarithm of cumulative forget gate matrix.

    The output matrix models the cumulative effects of forget gates over time.
    It is useful to compute the mLSTM gating in parallel.

    Parameters
    ----------
    log_f : (..., T, H) torch.Tensor
        Logarithm of forget gates.

    Returns
    -------
    log_f_mat : (..., T, T) torch.Tensor
        The logarithmic cumulative forget gate matrix.
    """
    idx = torch.arange(log_f.shape[-1], device=log_f.device)
    ltr_mask = idx.unsqueeze(-1) > idx.unsqueeze(-2)
    log_f_mat = torch.where(ltr_mask, log_f.unsqueeze(-1), 0.)
    log_prod_f_mat = torch.cumsum(log_f_mat, dim=-2)
    return torch.masked_fill(log_prod_f_mat, ltr_mask.T, -torch.inf)
