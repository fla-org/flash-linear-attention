# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.common.backends.tilelang import TileLangBackend
from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv, chunk_fwd_o
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, input_guard


def _can_use_shadow_state_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None,
) -> bool:
    _, _, H, K = k.shape
    HQ = q.shape[2]
    HV, V = v.shape[2], v.shape[-1]
    return (
        TileLangBackend.is_available()
        and TileLangBackend.is_enabled()
        and g is not None
        and g_gamma is None
        and initial_state is None
        and dht is None
        and cu_seqlens is None
        and chunk_indices is None
        and not state_v_first
        and chunk_size == 64
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and do.dtype == q.dtype
        and HQ == H
        and H == HV
        and K == V
        and K >= 128
        and V >= 128
        and K % 64 == 0
        and V % 64 == 0
    )


def _can_use_direct_mixed_state_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None,
) -> bool:
    return _can_use_shadow_state_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )


def _can_use_v_first_direct_state_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None,
) -> bool:
    K = k.shape[-1]
    V = v.shape[-1]
    return (
        _can_use_shadow_state_dqkwg(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            initial_state=initial_state,
            do=do,
            dht=dht,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        and K in (128, 256)
        and V in (128, 256)
    )


def _can_use_v_first_d256_matured_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None,
) -> bool:
    K = k.shape[-1]
    V = v.shape[-1]
    if not q.is_cuda:
        return False
    device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    return (
        check_shared_mem('hopper', device_index)
        and _can_use_shadow_state_dqkwg(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            initial_state=initial_state,
            do=do,
            dht=dht,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        and K == 256
        and V == 256
    )


def _can_use_dh_shadow_state_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None,
) -> bool:
    return _can_use_shadow_state_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )


def _can_use_dh_shadow_terminal_dot_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    chunk_indices: torch.LongTensor | None,
) -> bool:
    return _can_use_shadow_state_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )


def chunk_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=False,
        state_v_first=state_v_first,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
    )
    return o, ht


def chunk_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_gamma: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (SY 09/22) states_in_fp32 seems not affecting the error of dg but for safety, set to True
    use_v_first_d256_matured = _can_use_v_first_d256_matured_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    use_v_first_direct_state = _can_use_v_first_direct_state_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    ) and not use_v_first_d256_matured
    use_dh_shadow_terminal_dot = _can_use_dh_shadow_terminal_dot_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    ) and not use_v_first_d256_matured and not use_v_first_direct_state
    use_dh_shadow_state = _can_use_dh_shadow_state_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    ) and not use_v_first_d256_matured and not use_v_first_direct_state and not use_dh_shadow_terminal_dot
    use_direct_mixed_state = _can_use_direct_mixed_state_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    ) and (
        not use_v_first_d256_matured
        and not use_v_first_direct_state
        and not use_dh_shadow_terminal_dot
        and not use_dh_shadow_state
    )
    use_shadow_state = _can_use_shadow_state_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        initial_state=initial_state,
        do=do,
        dht=dht,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    ) and (
        not use_v_first_d256_matured
        and not use_v_first_direct_state
        and not use_dh_shadow_terminal_dot
        and not use_dh_shadow_state
        and not use_direct_mixed_state
    )
    internal_state_v_first = state_v_first or use_v_first_d256_matured or use_v_first_direct_state
    h_result = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=True,
        state_v_first=internal_state_v_first,
        output_mma_state=use_shadow_state,
    )
    if use_shadow_state:
        h, _, h_mma = h_result
    else:
        h, _ = h_result

    dh_result = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=True,
        state_v_first=internal_state_v_first,
        output_mma_state=use_shadow_state or use_dh_shadow_state or use_dh_shadow_terminal_dot,
        h_for_hdh=h if (use_shadow_state or use_dh_shadow_state) else None,
    )
    if use_shadow_state or use_dh_shadow_state:
        dh, dh0, dh_mma, hdh_last = dh_result
    elif use_dh_shadow_terminal_dot:
        dh, dh0, dh_mma = dh_result
    else:
        dh, dh0 = dh_result

    if use_v_first_d256_matured:
        from fla.ops.common.backends.tilelang.chunk_bwd import (
            chunk_bwd_dqkwg_tilelang_k_inner_v_first_d256,
        )
        dq, dk, _, dg = chunk_bwd_dqkwg_tilelang_k_inner_v_first_d256(
            q=q,
            k=k,
            v=v,
            do=do,
            h=h,
            dh=dh,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )
    elif use_v_first_direct_state:
        from fla.ops.common.backends.tilelang.chunk_bwd import (
            chunk_bwd_dqkwg_tilelang_k_inner_v_first,
        )
        dq, dk, _, dg = chunk_bwd_dqkwg_tilelang_k_inner_v_first(
            q=q,
            k=k,
            v=v,
            do=do,
            h=h,
            dh=dh,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )
    elif use_dh_shadow_terminal_dot:
        from fla.ops.common.backends.tilelang.chunk_bwd import (
            chunk_bwd_dqkwg_tilelang_k_inner_dh_shadow_terminal_dot,
        )
        dq, dk, _, dg = chunk_bwd_dqkwg_tilelang_k_inner_dh_shadow_terminal_dot(
            q=q,
            k=k,
            v=v,
            do=do,
            h=h,
            dh=dh,
            dh_mma=dh_mma,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )
    elif use_dh_shadow_state:
        from fla.ops.common.backends.tilelang.chunk_bwd import (
            chunk_bwd_dqkwg_tilelang_k_inner_dh_shadow,
        )
        dq, dk, _, dg = chunk_bwd_dqkwg_tilelang_k_inner_dh_shadow(
            q=q,
            k=k,
            v=v,
            do=do,
            h=h,
            dh_mma=dh_mma,
            hdh_last=hdh_last,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )
    elif use_shadow_state:
        from fla.ops.common.backends.tilelang.chunk_bwd import (
            chunk_bwd_dqkwg_tilelang_k_inner_shadow_state,
        )
        dq, dk, _, dg = chunk_bwd_dqkwg_tilelang_k_inner_shadow_state(
            q=q,
            k=k,
            v=v,
            do=do,
            h_mma=h_mma,
            dh_mma=dh_mma,
            hdh_last=hdh_last,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )
    elif use_direct_mixed_state:
        from fla.ops.common.backends.tilelang.chunk_bwd import (
            chunk_bwd_dqkwg_tilelang_k_inner,
        )
        dq, dk, _, dg = chunk_bwd_dqkwg_tilelang_k_inner(
            q=q,
            k=k,
            v=v,
            do=do,
            h=h,
            dh=dh,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )
    else:
        dq, dk, _, dg = chunk_bwd_dqkwg(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            h=h,
            do=do,
            dh=dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )
    if use_v_first_d256_matured:
        from fla.ops.common.backends.tilelang.chunk_bwd import (
            chunk_bwd_dv_tilelang_v_first_d256,
        )
        dv = chunk_bwd_dv_tilelang_v_first_d256(
            q=q,
            k=k,
            g=g,
            do=do,
            dh=dh,
            scale=scale,
            chunk_size=chunk_size,
        )
    else:
        dv = chunk_bwd_dv(
            q=q,
            k=k,
            g=g,
            g_gamma=g_gamma,
            do=do,
            dh=dh,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=internal_state_v_first,
        )
    return dq, dk, dv, dg, dh0


class ChunkSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_size: int | None = None,
    ):
        if chunk_size is None:
            chunk_size = 64

        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens,
                chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        else:
            chunk_indices = None

        # Pre-scale by RCP_LN2 so downstream kernels can use exp2 directly.
        if g is not None:
            g = chunk_local_cumsum(
                g,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        if g_gamma is not None:
            g_gamma = g_gamma * RCP_LN2
        o, ht = chunk_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )
        ctx.save_for_backward(q, k, v, g, g_gamma, initial_state, chunk_indices)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.state_v_first = state_v_first
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
        q, k, v, g, g_gamma, initial_state, chunk_indices = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            initial_state=initial_state,
            do=do,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=ctx.state_v_first,
        )
        if g is not None:
            dg = chunk_local_cumsum(
                dg,
                chunk_size=chunk_size,
                reverse=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            ).to(g)
        else:
            dg = None
        return dq.to(q), dk.to(k), dv.to(v), dg, None, None, dh0, None, None, None, None, None


@torch.compiler.disable
def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (Optional[torch.Tensor]):
            Forget gates of shape `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
            Default: `None`.
        g_gamma (Optional[torch.Tensor]):
            Log decay of shape `[H]`.
            Head-wise data-independent decay is used if `g_gamma` is provided.
            Only one of `g` or `g_gamma` should be provided. Default: `None`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` (or `[N, H, V, K]` if `state_v_first=True`)
            for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`
            (or `[N, H, V, K]` if `state_v_first=True`). Default: `False`.
        state_v_first (Optional[bool]):
            Store the recurrent state in V-first `[V, K]` layout instead of the default `[K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_cpu (torch.LongTensor):
            CPU copy of `cu_seqlens` to avoid unnecessary device synchronization. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` (or `[N, H, V, K]` if `state_v_first=True`)
            if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.simple_gla import chunk_simple_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, device='cuda'))
        >>> o, ht = chunk_simple_gla(
            q, k, v, g,
            initial_state=None,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_simple_gla(
            q, k, v, g,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    chunk_size = kwargs.pop('chunk_size', None)
    if chunk_size is not None and chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"`chunk_size` must be a power of 2, got {chunk_size}.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkSimpleGLAFunction.apply(
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_size,
    )
    return o, final_state
