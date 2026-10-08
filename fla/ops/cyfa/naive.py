# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


# Portions adapted from CyclicFlowAttention, Copyright (c) 2026 Yixiao Chen.
# https://github.com/Chyxx/CyclicFlowAttention

from __future__ import annotations

import torch

from .utils import CyFAState, build_readout_table, prepare_cu_seqlens, validate_cyfa_inputs, validate_initial_state


def _rotation_terms(
    lambdas: torch.Tensor,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.arange(num_slots, device=lambdas.device, dtype=torch.long)
    is_cos = idx.remainder(2) == 0
    mate = torch.where(is_cos, idx + 1, idx - 1).clamp(0, num_slots - 1)
    period = num_slots - 1
    pair = torch.arange(num_slots // 2, device=lambdas.device, dtype=torch.long)
    pair = pair.repeat_interleave(2)
    whole = torch.floor(lambdas)
    fraction = lambdas - whole
    coordinate = whole.to(torch.long).remainder(period).unsqueeze(-1)
    coordinate = (coordinate * pair).remainder(period).to(lambdas.dtype)
    coordinate = coordinate + fraction.unsqueeze(-1) * pair.to(lambdas.dtype)
    coordinate = torch.where(coordinate > period * 0.5, coordinate - period, coordinate)
    phase = coordinate * (2 * torch.pi / period)
    return is_cos, mate, torch.cos(phase), torch.sin(phase)


def _rotate_atom(
    lambdas: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    is_cos, _, c, s = _rotation_terms(lambdas, num_slots)
    pair = torch.arange(num_slots, device=lambdas.device) // 2
    scale = torch.where(pair == 0, 1.0, 2.0).div(num_slots - 1).sqrt()
    return scale * torch.where(is_cos.view(1, 1, -1), c, -s)


def _rotate_modal_pos(
    lambdas: torch.Tensor,
    x: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    is_cos, mate, c, s = _rotation_terms(lambdas, num_slots)
    xm = x[:, :, mate]
    y_cos = c * x - s * xm
    y_sin = s * xm + c * x
    y = torch.where(is_cos.view(1, 1, -1), y_cos, y_sin)
    return y


def _rotate_modal_neg(
    lambdas: torch.Tensor,
    x: torch.Tensor,
    num_slots: int,
) -> torch.Tensor:
    is_cos, mate, c, s = _rotation_terms(lambdas, num_slots)
    xm = x[:, :, mate]
    y_cos = c * x + s * xm
    y_sin = -s * xm + c * x
    y = torch.where(is_cos.view(1, 1, -1), y_cos, y_sin)
    return y


def _rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + float(eps))
    y = y * weight.float()
    return y.to(x.dtype)


def naive_recurrent_cyfa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    delta: torch.Tensor,
    beta: torch.Tensor,
    readout: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_eps: float,
    k_norm_eps: float,
    scale: float | None = None,
    initial_state: CyFAState | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, CyFAState | None]:
    """Token-by-token PyTorch reference for the canonical CyclicFlowAttention recurrence."""
    num_slots = validate_cyfa_inputs(q, k, v, g, delta, beta, readout)
    cu_seqlens = prepare_cu_seqlens(
        cu_seqlens,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        device=q.device,
    )
    if cu_seqlens is not None:
        boundaries = cu_seqlens.detach().cpu().tolist()
        n_seq = len(boundaries) - 1
        if initial_state is not None and initial_state[0].shape[0] != n_seq:
            raise ValueError("Initial state count must match the packed sequence count.")
        outputs = []
        final_parts = [[], [], []]
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            state_i = None if initial_state is None else tuple(x[i: i + 1] for x in initial_state)
            output_i, final_i = naive_recurrent_cyfa(
                q=q[:, start:end],
                k=k[:, start:end],
                v=v[:, start:end],
                g=g[:, start:end],
                delta=delta[:, start:end],
                beta=beta[:, start:end],
                readout=readout,
                scale=scale,
                initial_state=state_i,
                output_final_state=output_final_state,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                q_norm_eps=q_norm_eps,
                k_norm_eps=k_norm_eps,
            )
            outputs.append(output_i)
            if output_final_state:
                for j, value in enumerate(final_i):
                    final_parts[j].append(value)
        output = torch.cat(outputs, dim=1)
        final_state = (
            tuple(torch.cat(parts, dim=0) for parts in final_parts)
            if output_final_state
            else None
        )
        return output, final_state

    dtype = q.dtype
    q = _rmsnorm_forward(q, q_norm_weight, q_norm_eps)
    k = _rmsnorm_forward(k, k_norm_weight, k_norm_eps)

    q, k, v = map(lambda x: x.transpose(1, 2).contiguous().float(), (q, k, v))
    delta = delta.transpose(1, 2).contiguous().float()
    beta = beta.transpose(1, 2).contiguous().float()
    g = g.contiguous().float()
    g = g.transpose(1, 2).contiguous()
    bsz, n_heads, seqlen, d_k = q.shape
    d_v = v.shape[-1]
    if scale is None:
        scale = d_k ** -0.5

    state = validate_initial_state(
        initial_state,
        batch=bsz,
        n_heads=n_heads,
        num_slots=num_slots,
        d_k=d_k,
        d_v=d_v,
        device=q.device,
    )
    if state is None:
        hk = torch.zeros(bsz, n_heads, d_k, num_slots, device=q.device)
        hv = torch.zeros(bsz, n_heads, num_slots, d_v, device=q.device)
        lambda0 = torch.zeros(bsz, n_heads, device=q.device)
    else:
        hk, hv, lambda0 = state
    lambdas = delta.cumsum(dim=-1) + lambda0[:, :, None]
    readout_table = build_readout_table(readout)
    out = torch.zeros_like(v)
    for i in range(seqlen):
        decay = g[:, :, i].exp()
        hk = hk * decay[:, :, None, None]
        hv = hv * decay[:, :, None, None]
        write = beta[:, :, i, None] * _rotate_atom(
            lambdas[:, :, i],
            num_slots,
        )
        hk = hk + k[:, :, i, :, None] * write[:, :, None, :]
        hv = hv + write[:, :, :, None] * v[:, :, i, None, :]

        raw_logits = torch.einsum("bhkm,bhk->bhm", hk, q[:, :, i] * float(scale))
        rotated_logits = _rotate_modal_pos(lambdas[:, :, i], raw_logits, num_slots)
        logits = torch.einsum("hrm,bhm->bhr", readout_table, rotated_logits)
        numer = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
        denom = numer.sum(dim=-1, keepdim=True)
        probs = torch.where(denom > 0.0, numer / denom, torch.zeros_like(numer))
        coeff = torch.einsum("hrm,bhr->bhm", readout_table, probs)
        weights = _rotate_modal_neg(lambdas[:, :, i], coeff, num_slots)
        out[:, :, i] = torch.einsum("bhm,bhmv->bhv", weights, hv)

    out = out.transpose(1, 2).contiguous().to(dtype)
    if output_final_state:
        if cu_seqlens is None:
            final_lambda = lambdas[:, :, -1]
        else:
            ends = cu_seqlens[1:].to(device=lambdas.device, dtype=torch.long) - 1
            final_lambda = lambdas[0, :, ends].transpose(0, 1).contiguous()
        final_state = (hk, hv, final_lambda)
    else:
        final_state = None
    return out, final_state
