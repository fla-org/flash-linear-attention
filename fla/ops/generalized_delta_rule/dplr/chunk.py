# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings

import torch

from fla.ops.generalized_delta_rule.dplr.chunk_A_bwd import chunk_dplr_bwd_dqk_intra
from fla.ops.generalized_delta_rule.dplr.chunk_A_fwd import chunk_dplr_fwd_intra
from fla.ops.generalized_delta_rule.dplr.chunk_h_bwd import chunk_dplr_bwd_dhu
from fla.ops.generalized_delta_rule.dplr.chunk_h_fwd import chunk_dplr_fwd_h
from fla.ops.generalized_delta_rule.dplr.chunk_o_bwd import chunk_dplr_bwd_dAu, chunk_dplr_bwd_dv, chunk_dplr_bwd_o
from fla.ops.generalized_delta_rule.dplr.chunk_o_fwd import chunk_dplr_fwd_o
from fla.ops.generalized_delta_rule.dplr.wy_fast_bwd import chunk_dplr_bwd_wy
from fla.ops.generalized_delta_rule.dplr.wy_fast_fwd import prepare_wy_repr_fwd
from fla.ops.rwkv6.chunk import chunk_rwkv6_fwd_cumsum
from fla.ops.utils import prepare_chunk_indices
from fla.utils import TRITON_ABOVE_3_4_0, autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_dplr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    chunk_indices: torch.LongTensor | None = None,
    disable_recompute: bool = False,
):
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)

    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    # A_ab, A_ak, gi, ge torch.float32
    # A_qk, A_qb, qg, kg, ag, bg, dtype=q.dtype, eg: bf16
    w, u, A_ab_inv = prepare_wy_repr_fwd(
        ag=ag,
        A_ab=A_ab,
        A_ak=A_ak,
        v=v,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    o = chunk_dplr_fwd_o(
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    if disable_recompute:
        return o, final_state, (gi, ge, A_qk, A_qb, A_ak, qg, kg, ag, bg, w, h, v_new, A_ab_inv)
    else:
        return o, final_state, None


class ChunkDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
    ):
        # Due to gate numerical stability consideration, we only support chunk_size=16 when safe_gate=True
        # And in practice, chunk_size=16 is sufficient for no safe gate situations.
        # It's different from the other chunk implementations.
        if chunk_size is None:
            chunk_size = 16
        elif TRITON_ABOVE_3_4_0:
            chunk_size = chunk_size
        else:
            # Avoid Triton Compiler error
            warnings.warn(
                "Set chunk_size to 16, to avoid triton compiler erorr. "
                f"original chunk_size {chunk_size}",
                category=RuntimeWarning,
                stacklevel=2,
            )
            chunk_size = 16
        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None

        o, final_state, cache = chunk_dplr_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            chunk_indices=chunk_indices,
            disable_recompute=disable_recompute,
        )

        if disable_recompute:
            gi, ge, A_qk, A_qb, A_ak, qg, kg, ag, bg, w, h, v_new, A_ab_inv = cache
            ctx.save_for_backward(q, k, v, a, b, gk, initial_state, gi, ge, A_qk,
                                  A_qb, A_ak, qg, kg, ag, bg, w, h, v_new, A_ab_inv)
        else:
            ctx.save_for_backward(q, k, v, a, b, gk, initial_state)
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.chunk_indices = chunk_indices
        ctx.safe_gate = safe_gate
        ctx.disable_recompute = disable_recompute
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        if ctx.disable_recompute:
            (
                q, k, v, a, b, gk, initial_state,
                gi, ge, A_qk, A_qb, A_ak, qg, kg, ag, bg, w, h, v_new, A_ab_inv,
            ) = ctx.saved_tensors
        else:
            q, k, v, a, b, gk, initial_state = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        scale = ctx.scale

        if not ctx.disable_recompute:
            # ******* start recomputing everything, otherwise i believe the gpu memory will be exhausted *******
            gi, ge = chunk_rwkv6_fwd_cumsum(gk, chunk_size, cu_seqlens=cu_seqlens, chunk_indices=ctx.chunk_indices)

            A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
                q=q,
                k=k,
                a=a,
                b=b,
                gi=gi,
                ge=ge,
                scale=scale,
                cu_seqlens=cu_seqlens,
                safe_gate=ctx.safe_gate,
                chunk_size=chunk_size,
                chunk_indices=ctx.chunk_indices,
            )
            w, u, A_ab_inv = prepare_wy_repr_fwd(
                ag=ag,
                A_ab=A_ab,
                A_ak=A_ak,
                v=v,
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size,
                chunk_indices=ctx.chunk_indices,
            )
            del A_ab
            h, v_new, _ = chunk_dplr_fwd_h(
                kg=kg,
                bg=bg,
                v=v,
                w=w,
                u=u,
                gk=gi,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size,
                chunk_indices=ctx.chunk_indices,
            )
            del u
            # ******* end of recomputation *******
        # A_ak, A_ab_inv, gi, ge torch.float32
        # A_qk, A_qb, qg, kg, ag, bg, v_new dtype=q.dtype, eg: bf16

        dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(
            v=v,
            v_new=v_new,
            do=do,
            A_qb=A_qb,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=ctx.chunk_indices,
        )

        dh, dh0, dv_new = chunk_dplr_bwd_dhu(
            qg=qg,
            bg=bg,
            w=w,
            gk=gi,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv_new_intra,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=ctx.chunk_indices,
        )

        dv = chunk_dplr_bwd_dv(
            A_qk=A_qk,
            kg=kg,
            do=do,
            dh=dh,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=ctx.chunk_indices,
        )
        del A_qk

        dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(
            k=kg,
            b=bg,
            v=v,
            v_new=v_new,
            do=do,
            h=h,
            dh=dh,
            dv=dv_new,
            w=w,
            gk=gi,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            scale=scale,
            chunk_indices=ctx.chunk_indices,
        )
        del v_new

        dA_ab, dA_ak, dv, dag = chunk_dplr_bwd_wy(
            A_ab_inv=A_ab_inv,
            A_ak=A_ak,
            v=v,
            ag=ag,
            dw=dw,
            du=dv_new,
            dv0=dv,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=ctx.chunk_indices,
        )
        del A_ak

        dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(
            q=q,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            dAqk=dA_qk,
            dAqb=dA_qb,
            dAak=dA_ak,
            dAab=dA_ab,
            dgk_last=dgk_last,
            dqg=dqg,
            dkg=dkg,
            dag=dag,
            dbg=dbg,
            chunk_size=chunk_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            safe_gate=ctx.safe_gate,
            chunk_indices=ctx.chunk_indices,
        )

        return dq.to(q), dk.to(k), dv.to(v), da.to(a), db.to(b), dgk.to(gk), None, dh0, None, None, None, None, None, None


@torch.compiler.disable
def chunk_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    head_first: bool = False,
    safe_gate: bool = False,
    chunk_size: int | None = None,
    disable_recompute: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        a (torch.Tensor):
            activations of shape `[B, T, H, K]`.
        b (torch.Tensor):
            betas of shape `[B, T, H, K]`.
        gk (torch.Tensor):
            gk of shape `[B, T, H, K]`. decay term in log space!
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_cpu (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        safe_gate (bool):
            Whether the kernel can assume the input gate values `g` are in a safe range.
            When `True`, the kernel can use M=16 TensorCore acceleration.
            The safe range is approximately [-5, 0). Default: `False`.
        chunk_size (Optional[int]):
            Chunk size for the chunked computation. Default: `None`, which means 16.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
        )
    if q.dtype == torch.float32:
        warnings.warn(
            """ChunkDeltaRuleFunction does not support float32 on some platforms. Please use bfloat16/float16.
            If you want to use float32, please solve the issue by yourself.""",
            category=RuntimeWarning,
            stacklevel=2,
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        gk,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        chunk_size,
        disable_recompute,
    )
    return o, final_state
