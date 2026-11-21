# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_recurrent_triadic_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    gk: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reference recurrence for Triadic Kimi Delta Attention in FP64.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        k2 (torch.Tensor):
            Second keys of shape `[B, T, H, E]`.
        q2 (torch.Tensor):
            Second queries of shape `[B, T, H, E]`.
        gk (torch.Tensor):
            First-axis log decays of shape `[B, T, H, K]`.
        g2 (torch.Tensor):
            Second-axis log decays of shape `[B, T, H, E]`.
        beta (torch.Tensor, Optional):
            Scalar write strengths of shape `[B, T, H]`. Default: `None` (ones).
        scale (float, Optional):
            Scale factor of the output. Default: `None` (`K ** -0.5`).
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[N, H, E, K, V]`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final state of shape `[N, H, E, K, V]`. Default: `False`.
        cu_seqlens (torch.Tensor, Optional):
            Cumulative sequence lengths of shape `[N + 1]` for variable-length inputs (`B == 1`). Default: `None`.
            Supports `torch.int32` and `torch.int64`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` in `v.dtype`.
        final_state (torch.Tensor):
            FP32 final state of shape `[N, H, E, K, V]` if `output_final_state=True`, else `None`.
    """
    B, T, H, K = q.shape
    if cu_seqlens is not None and B != 1:
        raise ValueError(f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`.")
    E, V = k2.shape[-1], v.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    orig_dtype = v.dtype
    q_d, k_d, v_d, k2_d, q2_d, gk_d, g2_d = [x.double() for x in (q, k, v, k2, q2, gk, g2)]
    beta_d = torch.ones(B, T, H, device=q.device, dtype=torch.float64) if beta is None else beta.double()

    boundaries = cu_seqlens.tolist() if cu_seqlens is not None else None
    spans = [(0, s, t) for s, t in zip(boundaries[:-1], boundaries[1:])] if boundaries else [(i, 0, T) for i in range(B)]
    outputs, states = [], []
    for n, (batch, start, end) in enumerate(spans):
        state = q_d.new_zeros(H, E * K, V) if initial_state is None else initial_state[n].double().reshape(H, E * K, V)
        if end == start:
            outputs.append(v_d.new_empty(0, H, V))
            states.append(state.reshape(H, E, K, V))
            continue
        values = []
        for t in range(start, end):
            joint_k = (k2_d[batch, t, :, :, None] * k_d[batch, t, :, None, :]).flatten(-2)
            joint_q = (q2_d[batch, t, :, :, None] * q_d[batch, t, :, None, :]).flatten(-2)
            decay = (g2_d[batch, t, :, :, None] + gk_d[batch, t, :, None, :]).flatten(-2).exp()
            prior = state * decay[..., None]
            residual = beta_d[batch, t, :, None] * (v_d[batch, t] - torch.einsum('hk,hkv->hv', joint_k, prior))
            state = prior + joint_k[..., None] * residual[:, None, :]
            values.append(scale * torch.einsum('hk,hkv->hv', joint_q, state))
        outputs.append(torch.stack(values))
        states.append(state.reshape(H, E, K, V))
    output = torch.cat(outputs, dim=0)[None] if boundaries else torch.stack(outputs)
    final_state = torch.stack(states).float() if output_final_state else None
    return output.to(orig_dtype), final_state


def naive_chunk_triadic_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    gk: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Factorized chunkwise reference for Triadic Kimi Delta Attention in FP64.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        k2 (torch.Tensor):
            Second keys of shape `[B, T, H, E]`.
        q2 (torch.Tensor):
            Second queries of shape `[B, T, H, E]`.
        gk (torch.Tensor):
            First-axis log decays of shape `[B, T, H, K]`.
        g2 (torch.Tensor):
            Second-axis log decays of shape `[B, T, H, E]`.
        beta (torch.Tensor, Optional):
            Scalar write strengths of shape `[B, T, H]`. Default: `None` (ones).
        scale (float, Optional):
            Scale factor of the output. Default: `None` (`K ** -0.5`).
        initial_state (torch.Tensor, Optional):
            Initial state of shape `[N, H, E, K, V]`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the final state of shape `[N, H, E, K, V]`. Default: `False`.
        cu_seqlens (torch.Tensor, Optional):
            Cumulative sequence lengths of shape `[N + 1]` for variable-length inputs (`B == 1`). Default: `None`.
            Supports `torch.int32` and `torch.int64`.
        chunk_size (int, Optional):
            Number of tokens per chunk. Default: 64.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` in `v.dtype`.
        final_state (torch.Tensor):
            FP32 final state of shape `[N, H, E, K, V]` if `output_final_state=True`, else `None`.
    """
    B, T, H, K = q.shape
    if cu_seqlens is not None and B != 1:
        raise ValueError(f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`.")
    E, V = k2.shape[-1], v.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    orig_dtype = v.dtype
    beta_d = torch.ones(B, T, H, device=q.device, dtype=torch.float64) if beta is None else beta.double()
    inputs = [x.double() for x in (q, k, v, k2, q2, gk, g2)] + [beta_d]

    boundaries = cu_seqlens.tolist() if cu_seqlens is not None else None
    spans = [(0, s, t) for s, t in zip(boundaries[:-1], boundaries[1:])] if boundaries else [(i, 0, T) for i in range(B)]
    outputs, states = [], []
    for n, (batch, start, end) in enumerate(spans):
        state = q.new_zeros(H, E, K, V, dtype=torch.float64) if initial_state is None else initial_state[n].double()
        if end == start:
            outputs.append(v.new_empty(0, H, V, dtype=torch.float64))
            states.append(state)
            continue
        chunks = []
        for pos in range(start, end, chunk_size):
            qi, ki, vi, k2i, q2i, gki, g2i, betai = [
                x[batch, pos:min(pos + chunk_size, end)].transpose(0, 1) for x in inputs
            ]
            gki, g2i = gki.cumsum(1), g2i.cumsum(1)
            C = qi.shape[1]
            mask = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
            dk = (gki[:, :, None] - gki[:, None]).masked_fill(~mask[..., None], -torch.inf).exp()
            d2 = (g2i[:, :, None] - g2i[:, None]).masked_fill(~mask[..., None], -torch.inf).exp()
            lk = (ki[:, :, None] * ki[:, None] * dk).sum(-1)
            l2 = ((betai[..., None] * k2i)[:, :, None] * k2i[:, None] * d2).sum(-1)
            mk = (qi[:, :, None] * ki[:, None] * dk).sum(-1)
            m2 = (q2i[:, :, None] * k2i[:, None] * d2).sum(-1)
            system = (lk * l2).tril(-1) + torch.eye(C, device=q.device, dtype=torch.float64)
            rhs = betai[..., None] * (vi - torch.einsum('htk,hte,hekv->htv', ki * gki.exp(), k2i * g2i.exp(), state))
            residual = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
            output = torch.einsum('htk,hte,hekv->htv', qi * gki.exp(), q2i * g2i.exp(), state)
            output = scale * (output + (mk * m2) @ residual)
            state = state * (g2i[:, -1, :, None] + gki[:, -1, None, :]).exp()[..., None] + torch.einsum(
                'htk,hte,htv->hekv', ki * (gki[:, -1:] - gki).exp(), k2i * (g2i[:, -1:] - g2i).exp(), residual,
            )
            chunks.append(output.transpose(0, 1))
        outputs.append(torch.cat(chunks))
        states.append(state)
    output = torch.cat(outputs, dim=0)[None] if boundaries else torch.stack(outputs)
    final_state = torch.stack(states).float() if output_final_state else None
    return output.to(orig_dtype), final_state
