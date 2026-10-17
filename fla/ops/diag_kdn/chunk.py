# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.modules.l2norm import l2norm
from fla.ops.diag_kdn.chunk_h import chunk_diag_kdn_bwd_dhu, chunk_diag_kdn_fwd_h
from fla.ops.diag_kdn.chunk_intra import chunk_diag_kdn_bwd_intra, chunk_diag_kdn_fwd_intra
from fla.ops.diag_kdn.chunk_o import chunk_diag_kdn_bwd_dAv, chunk_diag_kdn_fwd_o
from fla.ops.diag_kdn.gain import diag_kdn_gain
from fla.ops.diag_kdn.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.kda.gate import fused_kda_gate
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

# the memory floors the per-step decay at 1e-6, the gain sees the unclamped decay
MIN_LOG_DECAY = math.log(1e-6)


def chunk_diag_kdn_memory_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    kappa: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 32,
):
    w, u, kg, Aqk, Akk = chunk_diag_kdn_fwd_intra(
        q=q,
        k=k,
        kappa=kappa,
        v=v,
        gk=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_diag_kdn_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    o = chunk_diag_kdn_fwd_o(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    return o, Aqk, Akk, final_state


def chunk_diag_kdn_memory_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    kappa: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 32,
):
    w, u, kg, qg = recompute_w_u_fwd(
        k=k,
        kappa=kappa,
        v=v,
        A=Akk,
        gk=g,
        q=q,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, _ = chunk_diag_kdn_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dAqk, dv = chunk_diag_kdn_bwd_dAv(
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    dh, dh0, du = chunk_diag_kdn_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        do=do,
        dv=dv,
        h0=initial_state,
        dht=dht,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dq, dk, dkappa, dv, dg, dAkk = prepare_wy_repr_bwd(
        q=q,
        k=k,
        kappa=kappa,
        v=v,
        v_new=v_new,
        g=g,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        du=du,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dq, dk, dkappa, dg = chunk_diag_kdn_bwd_intra(
        q=q,
        k=k,
        kappa=kappa,
        g=g,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        dkappa=dkappa,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    # `dg` is w.r.t. the chunk-local cumulative gates, the reverse cumsum maps it back to the raw gates
    dg = chunk_local_cumsum(
        g=dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return dq, dk, dkappa, dv, dg, dh0


class ChunkDiagKDNMemoryFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        kappa: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
    ):
        chunk_size = 32
        chunk_indices = None
        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        g = chunk_local_cumsum(
            g=g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        o, Aqk, Akk, final_state = chunk_diag_kdn_memory_fwd(
            q=q,
            k=k,
            kappa=kappa,
            v=v,
            g=g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, k, kappa, v, g, Aqk, Akk, initial_state, cu_seqlens, chunk_indices)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        q, k, kappa, v, g, Aqk, Akk, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors
        dq, dk, dkappa, dv, dg, dh0 = chunk_diag_kdn_memory_bwd(
            q=q,
            k=k,
            kappa=kappa,
            v=v,
            g=g,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
        )
        return dq.to(q), dk.to(k), dkappa.to(kappa), dv.to(v), dg, None, dh0, None, None, None


@torch.compiler.disable
def chunk_diag_kdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    omega: torch.Tensor,
    r: torch.Tensor,
    scale: float | None = None,
    initial_state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    output_final_state: bool = False,
    info_scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    r"""
    Diagonal Kalman Delta Network (DiagKDN).
    The memory decays with the per-channel forget gates and is corrected towards ``v`` along the write key ``kappa``,

    .. math::
        S_t = \text{Diag}(\exp(g_t)) S_{t-1}, \quad
        S_t \leftarrow S_t + \kappa_t (v_t - S_t^\top k_t)^\top,

    where ``kappa`` is the diagonal Kalman gain, see :func:`fla.ops.diag_kdn.gain.diag_kdn_gain`.
    The memory update runs on chunks of 32 steps with float32 states and write keys,
    and floors the per-step decay ``exp(g_t)`` at ``1e-6``.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, H, V]``.
        g (torch.Tensor):
            Forget gates (in log space) of shape ``[B, T, H, K]``,
            or the raw gate input if ``use_gate_in_kernel=True``.
        omega (torch.Tensor):
            Positive process noise of shape ``[B, T, H, K]``.
        r (torch.Tensor):
            Positive observation noise of shape ``[B, T, H]``.
        scale (float, Optional):
            Scale factor of the attention scores. Default: `None`, i.e., ``1 / sqrt(K)``.
        initial_state (tuple[torch.Tensor, torch.Tensor], Optional):
            Initial memory of shape ``[N, H, K, V]`` and initial precision of shape ``[N, H, K]``,
            both in ``float32``, for ``N`` input sequences.
            Either entry may be `None`, in which case the memory starts at zero and the precision at one.
            Default: `None`.
        output_final_state (bool, Optional):
            Whether to output the final memory and precision. Default: `False`.
        info_scale (float, Optional):
            Information contributed by one unit-norm key. Default: `None`, i.e., ``K``.
        use_qk_l2norm_in_kernel (bool, Optional):
            Whether to L2-normalize queries and keys before the gain and the memory update. Default: `False`.
        use_gate_in_kernel (bool, Optional):
            Whether to compute the log-space decay ``-exp(A_log) * softplus(g + dt_bias)`` internally.
            Default: `False`.
        A_log (torch.Tensor, Optional):
            Log decay rates of shape ``[H]``, required if ``use_gate_in_kernel=True``. Default: `None`.
        dt_bias (torch.Tensor, Optional):
            Bias of shape ``[H * K]`` added to ``g`` before the activation if ``use_gate_in_kernel=True``.
            Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape ``[N+1]`` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.
        cu_seqlens_cpu (torch.LongTensor, Optional):
            CPU copy of ``cu_seqlens`` that avoids a device synchronization. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape ``[B, T, H, V]``.
        final_state (tuple[torch.Tensor, torch.Tensor]):
            Final memory of shape ``[N, H, K, V]`` and precision of shape ``[N, H, K]``
            if ``output_final_state=True`` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.diag_kdn import chunk_diag_kdn
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 128, 128
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float, device='cuda'))
        >>> omega = F.softplus(torch.randn(B, T, H, K, dtype=torch.float, device='cuda'))
        >>> r = F.softplus(torch.randn(B, T, H, dtype=torch.float, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.float, device='cuda')
        >>> c0 = torch.ones(B, H, K, dtype=torch.float, device='cuda')
        >>> o, (ht, ct) = chunk_diag_kdn(
            q, k, v, g, omega, r,
            initial_state=(h0, c0),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, omega, r = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, omega, r))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, (ht, ct) = chunk_diag_kdn(
            q, k, v, g, omega, r,
            initial_state=(h0, c0),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    """
    if kwargs.pop('cp_context', None) is not None:
        raise NotImplementedError("Context parallelism is not supported for DiagKDN yet.")
    if kwargs:
        raise TypeError(f"Unexpected arguments for DiagKDN: {', '.join(kwargs)}.")
    if use_gate_in_kernel and A_log is None:
        raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
    assert v.shape[2] == q.shape[2], "DiagKDN does not support grouped value attention."
    if scale is None:
        scale = k.shape[-1] ** -0.5
    h0, c0 = initial_state if initial_state is not None else (None, None)
    if c0 is not None:
        assert c0.dtype == torch.float32, "The initial precision must be in float32."
    if use_qk_l2norm_in_kernel:
        # the gain is sensitive to the key norm, so the normalized keys are kept in float32
        q, k = l2norm(q), l2norm(k, output_dtype=torch.float32)
    if use_gate_in_kernel:
        g = fused_kda_gate(g, A_log, dt_bias)
    kappa, ct = diag_kdn_gain(
        k=k,
        g=g,
        omega=omega,
        r=r,
        initial_state=c0,
        output_final_state=output_final_state,
        info_scale=info_scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    o, ht = ChunkDiagKDNMemoryFunction.apply(
        q,
        k.to(q.dtype),
        kappa,
        v,
        g.clamp_min(MIN_LOG_DECAY),
        scale,
        h0,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
    )
    return o, ((ht, ct) if output_final_state else None)
