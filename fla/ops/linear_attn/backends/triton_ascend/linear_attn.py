# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# ruff: noqa: E501

"""Public entry points for un-gated linear attention on Ascend NPU.

M7-2 implemented the dense ``chunk_linear_attn`` forward path.  M7-6 adds the
un-gated dense backward family (``chunk_bwd_dh`` / ``chunk_bwd_dqkwg`` /
``chunk_bwd_dv_local`` + ``chunk_bwd_dv_state``) behind a custom autograd
``Function`` so the public entry participates in ``torch.autograd``.
"""

from __future__ import annotations

import torch

from .chunk_h import chunk_fwd_h
from .chunk_h_ascend import chunk_bwd_dh_npu
from .chunk_o import chunk_fwd_o
from .chunk_o_ascend_bwd import chunk_bwd_dqkwg_npu, chunk_bwd_dv_npu
from .global_cumsum import chunk_global_cumsum
from .utils import empty_attn_outputs, normalize_chunk_size


class StatefulNormalizeFunction(torch.autograd.Function):
    """Normalized linear-attention divide step with a hand-written backward.

    Forward computes the running key-cumsum ``z_t`` and divides the output by
    ``<q_t, z_t> * scale``.  Backward mirrors the FLA simple-GLA pattern.
    """

    @staticmethod
    def forward(ctx, o_buf, q, k, scale, z_init, reverse, cu_seqlens):
        if q.shape[1] == 0:
            num_seqs = 1 if cu_seqlens is None else len(cu_seqlens) - 1
            if z_init is None:
                z_state_out = q.new_zeros(num_seqs, 1, q.shape[2], q.shape[3])
            else:
                z_dtype = torch.promote_types(k.dtype, z_init.dtype)
                z_state_out = z_init.to(z_dtype).clone()
            ctx.save_for_backward(o_buf, q, k)
            ctx.is_empty = True
            ctx.has_z_init = z_init is not None
            ctx.z_init_shape = z_state_out.shape
            ctx.k_dtype = k.dtype
            return o_buf, z_state_out

        k_cum = chunk_global_cumsum(
            k, reverse=reverse, cu_seqlens=cu_seqlens, output_dtype=k.dtype
        )
        if z_init is not None:
            if cu_seqlens is not None:
                seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
                z_init_b = (
                    z_init.squeeze(1).repeat_interleave(seq_lens, dim=0).unsqueeze(0)
                )
            else:
                z_init_b = z_init
            k_cum = k_cum + z_init_b
        denom = (q * scale * k_cum).sum(-1, keepdim=True) + 1e-10
        o_out = o_buf / denom
        if cu_seqlens is not None:
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
            # An empty sequence owns no token, so there is no cumsum entry to
            # read its final state from: ``cu_seqlens[1:] - 1`` underflows to
            # ``-1`` there and Ascend faults on the out-of-range gather.  Clamp
            # the index and keep the incoming state (``z_init`` or zeros).
            valid = seq_lens > 0
            idx = (cu_seqlens[:-1] if reverse else cu_seqlens[1:] - 1).to(torch.long)
            idx = torch.where(valid, idx, torch.zeros_like(idx))
            z_state_out = k_cum[0, idx].unsqueeze(1)
            if not bool(valid.all()):
                fill = (
                    z_init.to(z_state_out.dtype)
                    if z_init is not None
                    else torch.zeros_like(z_state_out)
                )
                z_state_out = torch.where(
                    valid.view(-1, 1, 1, 1),
                    z_state_out,
                    fill.expand_as(z_state_out),
                )
            z_state_out = z_state_out.contiguous()
        else:
            z_state_out = (k_cum[:, :1] if reverse else k_cum[:, -1:]).contiguous()

        ctx.save_for_backward(o_buf, q, k_cum, denom)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        ctx.has_z_init = z_init is not None
        ctx.k_dtype = k.dtype
        ctx.is_empty = False
        return o_out, z_state_out

    @staticmethod
    def backward(ctx, do, dz_state):
        if ctx.is_empty:
            _, q, k = ctx.saved_tensors
            dz_init = None
            if ctx.has_z_init:
                dz_init = (
                    torch.zeros(
                        ctx.z_init_shape,
                        device=q.device,
                        dtype=ctx.k_dtype,
                    )
                    if dz_state is None
                    else dz_state.to(ctx.k_dtype)
                )
            return (
                do,
                torch.zeros_like(q),
                torch.zeros_like(k),
                None,
                dz_init,
                None,
                None,
            )

        o_buf, q, k_cum, denom = ctx.saved_tensors
        scale = ctx.scale
        reverse = ctx.reverse
        cu_seqlens = ctx.cu_seqlens

        d_o_in = do / denom
        ddenom = -(do * o_buf).sum(-1, keepdim=True) / (denom * denom)

        dq = ddenom * scale * k_cum
        dk_cum = ddenom * scale * q

        # Index of the last token owned by each sequence, plus a mask telling
        # empty sequences apart (see the matching comment in ``forward``).
        valid = None
        if cu_seqlens is not None:
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
            valid = seq_lens > 0

        if dz_state is not None:
            if cu_seqlens is not None:
                idx = (cu_seqlens[:-1] if reverse else cu_seqlens[1:] - 1).to(
                    torch.long
                )
                contrib = dz_state.squeeze(1).to(dk_cum.dtype)
                # Empty sequences scatter nothing into the token-level cumsum
                # gradient; their state gradient is routed to ``z_init`` below.
                dk_cum[0].index_add_(0, idx[valid], contrib[valid])
            elif reverse:
                dk_cum[:, :1].add_(dz_state.to(dk_cum.dtype))
            else:
                dk_cum[:, -1:].add_(dz_state.to(dk_cum.dtype))

        dz_init = None
        if ctx.has_z_init:
            if cu_seqlens is not None:
                seg_id = torch.repeat_interleave(
                    torch.arange(seq_lens.numel(), device=dk_cum.device), seq_lens
                )
                flat = dk_cum[0]
                dz_init = torch.zeros(
                    seq_lens.numel(),
                    flat.shape[1],
                    flat.shape[2],
                    device=flat.device,
                    dtype=flat.dtype,
                )
                dz_init.index_add_(0, seg_id, flat)
                if dz_state is not None and not bool(valid.all()):
                    # An empty sequence passes its whole state gradient on to
                    # ``z_init`` instead of accumulating token gradients.
                    dz_init[~valid] += dz_state.squeeze(1).to(dz_init.dtype)[~valid]
                dz_init = dz_init.unsqueeze(1)
            else:
                dz_init = dk_cum.sum(dim=1, keepdim=True)

        dk = chunk_global_cumsum(
            dk_cum, reverse=not reverse, cu_seqlens=cu_seqlens, output_dtype=ctx.k_dtype
        )
        grads = (d_o_in, dq, dk, None, dz_init, None, None)
        return grads


def normalize_with_z_state(
    o_buf: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    z_init: torch.Tensor | None,
    reverse: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply running normalization ``z_t = sum_{s<=t} k_s`` (or reverse)."""
    return StatefulNormalizeFunction.apply(
        o_buf, q, k, scale, z_init, reverse, cu_seqlens
    )


def _chunk_linear_attn_forward(
    q, k, v, scale, initial_state, output_final_state, cu_seqlens, chunk_size
):
    h, final_state = chunk_fwd_h(
        k=k,
        v=v,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o_buf = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    ).to(q.dtype)
    return o_buf, final_state


class ChunkLinearAttnFunction(torch.autograd.Function):
    """Autograd wrapper for the un-gated chunk linear-attention family.

    The normalize=False path is fully differentiable in M7-6.  Normalize=True
    still uses the forward-only path until the denominator-chain backward is
    added in M7-8.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_size,
    ):
        o_buf, final_state = _chunk_linear_attn_forward(
            q,
            k,
            v,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            chunk_size,
        )
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return o_buf, final_state

    @staticmethod
    def backward(ctx, do, dht):
        do = do.contiguous()
        if dht is not None:
            dht = dht.contiguous()
        q, k, v = ctx.saved_tensors
        initial_state = ctx.initial_state
        cu_seqlens = ctx.cu_seqlens
        chunk_size = ctx.chunk_size

        # Recompute h in fp32 for the backward pass, matching the upstream
        # simple-GLA backward recipe.
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            states_in_fp32=True,
        )
        dh, dh0 = chunk_bwd_dh_npu(
            q=q,
            k=k,
            v=v,
            do=do,
            h0=initial_state,
            dht=dht,
            scale=ctx.scale,
            g=None,
            g_gamma=None,
            gk=None,
            gv=None,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            states_in_fp32=True,
        )
        dq, dk, _, _ = chunk_bwd_dqkwg_npu(
            q=q,
            k=k,
            v=v,
            do=do,
            h=h,
            dh=dh,
            w=None,
            g=None,
            g_gamma=None,
            dv=None,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        dv = chunk_bwd_dv_npu(
            q=q,
            k=k,
            do=do,
            dh=dh,
            g=None,
            g_gamma=None,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        grads = (dq, dk, dv, None, dh0, None, None, None)
        return grads


@torch.compiler.disable
def chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | tuple | None = None,
    output_final_state: bool = False,
    normalize: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | tuple | None]:
    """Chunked un-gated linear attention forward.

    Args:
        q, k, v: tensors of shape ``[B, T, H, K]`` / ``[B, T, H, V]``.
        scale: query scale; defaults to ``K**-0.5``.
        initial_state: state tensor ``[N, H, K, V]`` when ``normalize=False``;
            tuple ``(kv_state, z_state)`` when ``normalize=True``.
        output_final_state: return the final recurrent state.
        normalize: apply the running-denominator normalization.
        cu_seqlens: cumulative sequence lengths for packed varlen input.
        chunk_size: optional power-of-two chunk length; default 64.
    """
    if normalize and isinstance(initial_state, tuple):
        kv_init, z_init = initial_state
    else:
        kv_init, z_init = initial_state, None

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if isinstance(kv_init, torch.Tensor):
        kv_init = kv_init.contiguous()
    if z_init is not None:
        z_init = z_init.contiguous()

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                "The batch size is expected to be 1 rather than "
                f"{q.shape[0]} when using `cu_seqlens`."
            )
        if kv_init is not None and kv_init.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of initial states is expected to be equal to the "
                f"number of input sequences, i.e., {len(cu_seqlens) - 1} rather "
                f"than {kv_init.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5
    if chunk_size is None:
        chunk_size = 64
    # Ascend 950 has 216 KiB UB.  BT=128 together with the fp32 state/output
    # tiles exceeds it and also selects a numerically unstable fp16 dot path.
    # Chunk boundaries are an implementation detail, so use a finer internal
    # grid while preserving the public contract.
    chunk_size = normalize_chunk_size(chunk_size)

    if q.shape[1] == 0:
        num_seqs = 1 if cu_seqlens is None else len(cu_seqlens) - 1
        o, final_state, z_state = empty_attn_outputs(
            q, k, v, kv_init, z_init, output_final_state, num_seqs
        )
        return (o, (final_state, z_state)) if normalize else (o, final_state)

    if not normalize:
        return ChunkLinearAttnFunction.apply(
            q,
            k,
            v,
            scale,
            kv_init,
            output_final_state,
            cu_seqlens,
            chunk_size,
        )

    o_buf, final_state = ChunkLinearAttnFunction.apply(
        q,
        k,
        v,
        scale,
        kv_init,
        output_final_state,
        cu_seqlens,
        chunk_size,
    )
    o_buf, z_state = normalize_with_z_state(
        o_buf, q, k, scale, z_init, reverse=False, cu_seqlens=cu_seqlens
    )
    return o_buf, (final_state, z_state)
