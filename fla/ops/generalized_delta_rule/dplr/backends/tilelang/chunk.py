# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Low-retention DPLR path with FLA-compatible active-varlen semantics."""

from contextvars import ContextVar

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu_pre_process,
    chunk_gated_delta_rule_fwd_h_pre_process,
)
from fla.ops.generalized_delta_rule.dplr.chunk_o_bwd import chunk_dplr_bwd_dAu
from fla.ops.utils.constant import RCP_LN2

from .chunk_A_bwd import (
    chunk_dplr_bwd_dqk_intra_fused_qside_into,
)
from .chunk_A_fwd import (
    chunk_dplr_fwd_intra,
    chunk_dplr_fwd_intra_from_gk,
)
from .chunk_h_fwd import chunk_dplr_fwd_h
from .chunk_ho_fwd import (
    chunk_dplr_fwd_ho,
    chunk_dplr_fwd_ho_context_elided,
    chunk_dplr_fwd_ho_with_context,
)
from .chunk_stream_bwd import chunk_dplr_bwd_stream_dhu_o_into
from .cumsum import chunk_local_cumsum
from .utils import ChunkLayout, build_varlen_chunk_layout
from .wy_fast_bwd import chunk_dplr_bwd_wy_into
from .wy_fast_fwd import prepare_wy_repr_fwd

_CHECKPOINT_PHASE_NORMAL = "normal"
_CHECKPOINT_PHASE_FORWARD = "forward"
_CHECKPOINT_PHASE_RECOMPUTE = "recompute"
_DPLR_CHECKPOINT_PHASE: ContextVar[str] = ContextVar(
    "rwkv7_dplr_checkpoint_phase",
    default=_CHECKPOINT_PHASE_NORMAL,
)

# Active FLACPContext for the duration of a CP call. Custom ops cannot take
# the context object, so it is threaded through a ContextVar instead.
_DPLR_CP_CONTEXT: ContextVar = ContextVar("fla_dplr_cp_context", default=None)


class _DPLRCheckpointPhaseMode(TorchDispatchMode):
    """Checkpoint phase marker that does not enter the dispatch stack."""

    def __init__(self, phase: str):
        super().__init__()
        self.phase = phase
        self._token = None

    def __enter__(self):
        token = _DPLR_CHECKPOINT_PHASE.set(self.phase)
        self._token = token
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        token = self._token
        self._token = None
        if token is not None:
            _DPLR_CHECKPOINT_PHASE.reset(token)
        return False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def dplr_checkpoint_context_fn() -> tuple[TorchDispatchMode, TorchDispatchMode]:
    """Mark eager non-reentrant checkpoint forward and recompute phases."""
    return (
        _DPLRCheckpointPhaseMode(_CHECKPOINT_PHASE_FORWARD),
        _DPLRCheckpointPhaseMode(_CHECKPOINT_PHASE_RECOMPUTE),
    )


def _checkpoint_phase_for_dispatch(is_compiling: bool) -> str:
    if is_compiling:
        return _CHECKPOINT_PHASE_NORMAL
    return _DPLR_CHECKPOINT_PHASE.get()


def _prepare_cuda_cu_seqlens(
    cu_seqlens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if cu_seqlens.device.type != "cuda":
        raise ValueError("DPLR varlen layout requires CUDA cu_seqlens")
    if cu_seqlens.device != device:
        raise ValueError(
            f"cu_seqlens device {cu_seqlens.device} must match tensor device {device}"
        )
    return cu_seqlens.to(dtype=torch.int32).contiguous()


def _layout_from_saved(
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    chunk_offsets: Tensor,
    is_varlen: bool,
) -> ChunkLayout | None:
    if not is_varlen:
        return None
    return ChunkLayout(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )


def _empty_layout_metadata(q: Tensor) -> tuple[Tensor, Tensor]:
    return (
        q.new_empty((0, 2), dtype=torch.int32),
        q.new_empty((0,), dtype=torch.int32),
    )


def _chunk_dplr_delta_rule_bwd_core(
    do: Tensor,
    dht: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    gk: Tensor,
    h0: Tensor,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    chunk_offsets: Tensor,
    scale: float,
    has_initial_state: bool,
    is_varlen: bool,
    chunk_size: int,
    saved_h: Tensor | None = None,
    saved_v_new: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if saved_h is not None and saved_h.numel() == 0:
        saved_h = None
    if saved_v_new is not None and saved_v_new.numel() == 0:
        saved_v_new = None

    cu = cu_seqlens if is_varlen else None
    layout = _layout_from_saved(cu_seqlens, chunk_indices, chunk_offsets, is_varlen)
    initial_state = h0 if has_initial_state else None
    dht_arg = dht if dht.numel() > 0 else None

    use_from_gk_bwd = q.shape[-1] == 64
    if use_from_gk_bwd:
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg, gi = chunk_dplr_fwd_intra_from_gk(
            q=q,
            k=k,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            chunk_size=chunk_size,
            cu_seqlens=cu,
            chunk_layout=layout,
        )
        ge = None
    else:
        gi, ge = chunk_local_cumsum(
            gk,
            chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu,
            chunk_layout=layout,
        )
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
            q=q,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            scale=scale,
            chunk_size=chunk_size,
            cu_seqlens=cu,
            chunk_layout=layout,
        )
    w, u, A_ab_inv = prepare_wy_repr_fwd(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab=A_ab,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        chunk_layout=layout,
    )
    del A_ab
    cp_context = _DPLR_CP_CONTEXT.get()
    if cp_context is not None:
        # CP: rebuild the corrected initial state (boundary exchange across
        # ranks) before recomputing the chunk states
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=kg,
            w=w,
            u=u,
            gk=gi,
            bg=bg,
            v=v,
            cu_seqlens=cu,
            initial_state=None,
            context=cp_context,
            chunk_size=chunk_size,
        )
    if saved_h is not None and saved_v_new is not None:
        h = saved_h
        v_new = saved_v_new
    else:
        h, v_new, _ = chunk_dplr_fwd_h(
            kg=kg,
            v=v,
            w=w,
            u=u,
            bg=bg,
            gk=gi,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu,
            chunk_size=chunk_size,
            chunk_layout=layout,
        )

    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    n_chunks = (
        chunk_indices.shape[0]
        if is_varlen
        else batch * ((tokens + chunk_size - 1) // chunk_size)
    )
    n_seqs = cu_seqlens.shape[0] - 1 if is_varlen else batch
    n_dh0 = n_seqs if has_initial_state else 1

    dh0_workspace = torch.empty(
        (n_dh0, heads, key_dim, value_dim),
        dtype=torch.float32,
        device=q.device,
    )
    dgk_last = torch.empty(
        (n_chunks, heads, key_dim),
        dtype=torch.float32,
        device=q.device,
    )
    dv_full_workspace = torch.empty_like(v_new)
    if cp_context is not None:
        # CP: compute the local dh boundary contribution and fold the
        # following ranks' dh into this rank's terminal dh
        dv_new_intra, _, _ = chunk_dplr_bwd_dAu(
            v=v,
            v_new=v_new,
            do=do,
            A_qb=A_qb,
            scale=scale,
            cu_seqlens=cu,
            chunk_size=chunk_size,
            chunk_indices=layout.chunk_indices if is_varlen else None,
        )
        dht_arg, _ = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=qg,
            k=kg,
            w=w,
            do=do,
            dv=dv_new_intra,
            gk=gi,
            bg=bg,
            scale=1.0,
            cu_seqlens=cu,
            dht=None,
            initial_state=None,
            context=cp_context,
            chunk_size=chunk_size,
        )
    dqg, dkg, dw, dbg, dgk_last, dv2, dv_full, dh0 = (
        chunk_dplr_bwd_stream_dhu_o_into(
            qg=qg,
            bg=bg,
            w=w,
            kg=kg,
            v=v,
            v_new=v_new,
            gk=gi,
            h=h,
            h0=initial_state,
            dht=dht_arg,
            do=do,
            A_qb_for_dv=A_qb,
            A_qk=A_qk,
            dq_out=qg,
            dk_out=kg,
            dw_out=w,
            db_out=bg,
            dgk_last_out=dgk_last,
            dv2_out=u,
            dv_full_out=dv_full_workspace,
            dh0_out=dh0_workspace,
            cu_seqlens=cu,
            chunk_size=chunk_size,
            scale=scale,
            chunk_layout=layout,
        )
    )
    del A_qb
    del A_qk

    dA_ab_workspace = torch.empty(A_ak.shape, dtype=q.dtype, device=A_ak.device)
    dA_ak_workspace = torch.empty(A_ak.shape, dtype=q.dtype, device=A_ak.device)
    dA_ab, dA_ak, dv_out, dag = chunk_dplr_bwd_wy_into(
        A_ab_inv=A_ab_inv,
        A_ak=A_ak,
        v=v,
        ag=ag,
        dw=dw,
        du=dv2,
        dv0=dv_full,
        dA_ab_out=dA_ab_workspace,
        dA_ak_out=dA_ak_workspace,
        dv_out=dv_full,
        dag_out=ag,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        chunk_layout=layout,
    )
    del h
    del dv2

    # dgk is returned in gk's dtype; recycle the w buffer only when they agree
    dgk_out = w if w.dtype == gk.dtype else torch.empty_like(gk)
    dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra_fused_qside_into(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        gk=gk if use_from_gk_bwd else None,
        do=do,
        v=v,
        v_new=v_new,
        dAak=dA_ak,
        dAab=dA_ab,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        dgk_last=dgk_last,
        dq_out=qg,
        dk_out=kg,
        da_out=ag,
        db_out=bg,
        dgk_out=dgk_out,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        scale=scale,
        chunk_layout=layout,
        dgk_dtype=gk.dtype,
    )

    dh0_out = dh0 if (cp_context is None and has_initial_state and dh0 is not None) else h0.new_empty((0,))
    return dq, dk, dv_out, da, db, dgk, dh0_out


@torch.library.custom_op(
    "fla::chunk_dplr_delta_rule_bwd",
    mutates_args=(),
)
def _chunk_dplr_delta_rule_bwd_op(
    do: Tensor,
    dht: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    gk: Tensor,
    h0: Tensor,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    chunk_offsets: Tensor,
    scale: float,
    has_initial_state: bool,
    is_varlen: bool,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _chunk_dplr_delta_rule_bwd_core(
        do,
        dht,
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        scale,
        has_initial_state,
        is_varlen,
        chunk_size,
    )


@_chunk_dplr_delta_rule_bwd_op.register_fake
def _chunk_dplr_delta_rule_bwd_fake(
    do,
    dht,
    q,
    k,
    v,
    a,
    b,
    gk,
    h0,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale: float,
    has_initial_state: bool,
    is_varlen: bool,
    chunk_size: int,
):
    dh0 = h0.new_empty(h0.shape) if has_initial_state else h0.new_empty((0,))
    return (
        q.new_empty(q.shape),
        k.new_empty(k.shape),
        v.new_empty(v.shape),
        a.new_empty(a.shape),
        b.new_empty(b.shape),
        gk.new_empty(gk.shape),
        dh0,
    )


@torch.library.custom_op(
    "fla::chunk_dplr_delta_rule_bwd_ctx",
    mutates_args=(),
)
def _chunk_dplr_delta_rule_bwd_ctx_op(
    do: Tensor,
    dht: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    gk: Tensor,
    h0: Tensor,
    cu_seqlens: Tensor,
    chunk_indices: Tensor,
    chunk_offsets: Tensor,
    h_ctx: Tensor,
    v_new_ctx: Tensor,
    scale: float,
    has_initial_state: bool,
    is_varlen: bool,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _chunk_dplr_delta_rule_bwd_core(
        do,
        dht,
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        scale,
        has_initial_state,
        is_varlen,
        chunk_size,
        saved_h=h_ctx,
        saved_v_new=v_new_ctx,
    )


@_chunk_dplr_delta_rule_bwd_ctx_op.register_fake
def _chunk_dplr_delta_rule_bwd_ctx_fake(
    do,
    dht,
    q,
    k,
    v,
    a,
    b,
    gk,
    h0,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    h_ctx,
    v_new_ctx,
    scale: float,
    has_initial_state: bool,
    is_varlen: bool,
    chunk_size: int,
):
    del h_ctx, v_new_ctx
    return _chunk_dplr_delta_rule_bwd_fake(
        do,
        dht,
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        scale,
        has_initial_state,
        is_varlen,
        chunk_size,
    )


@torch.library.custom_op(
    "fla::chunk_dplr_delta_rule_fwd",
    mutates_args=(),
)
def _chunk_dplr_delta_rule_fwd_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    gk: Tensor,
    h0: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    has_initial_state: bool,
    output_final_state: bool,
    is_varlen: bool,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    cu = cu_seqlens if is_varlen else None
    layout = (
        build_varlen_chunk_layout(
            cu_seqlens,
            chunk_size,
            q.shape[0] * q.shape[1],
        )
        if is_varlen
        else None
    )
    initial_state = h0 if has_initial_state else None

    # from_gk (in-CTA chunk-local cumsum from bf16 gk) is the default A-stage
    # for head_dim 64, in training and eval, rectangular and varlen.  It
    # removes the standalone fp32 gi/ge cumsum kernels and the fp32 ge tensor.
    use_from_gk_a = q.shape[-1] == 64
    if use_from_gk_a:
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg, gi = chunk_dplr_fwd_intra_from_gk(
            q=q,
            k=k,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            chunk_size=chunk_size,
            cu_seqlens=cu,
            chunk_layout=layout,
        )
    else:
        gi, ge = chunk_local_cumsum(
            gk,
            chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu,
            chunk_layout=layout,
        )
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
            q=q,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            scale=scale,
            chunk_size=chunk_size,
            cu_seqlens=cu,
            chunk_layout=layout,
        )
    w, u, _ = prepare_wy_repr_fwd(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab=A_ab,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        chunk_layout=layout,
    )
    cp_context = _DPLR_CP_CONTEXT.get()
    if cp_context is not None:
        # CP: exchange the compact local boundary state (h = M @ h_in + c)
        # across ranks and fold it into the corrected initial state
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=kg,
            w=w,
            u=u,
            gk=gi,
            bg=bg,
            v=v,
            cu_seqlens=cu,
            initial_state=None,
            context=cp_context,
            chunk_size=chunk_size,
        )
    o, final_state = chunk_dplr_fwd_ho(
        qg=qg,
        kg=kg,
        v=v,
        w=w,
        u=u,
        bg=bg,
        gk=gi,
        A_qk=A_qk,
        A_qb=A_qb,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        chunk_layout=layout,
    )

    if final_state is None:
        final_state = q.new_empty((0,), dtype=torch.float32)
    if layout is None:
        chunk_indices, chunk_offsets = _empty_layout_metadata(q)
    else:
        chunk_indices = layout.chunk_indices
        chunk_offsets = layout.chunk_offsets
    return o, final_state, chunk_indices, chunk_offsets


@_chunk_dplr_delta_rule_fwd_op.register_fake
def _chunk_dplr_delta_rule_fwd_fake(
    q,
    k,
    v,
    a,
    b,
    gk,
    h0,
    cu_seqlens,
    scale: float,
    has_initial_state: bool,
    output_final_state: bool,
    is_varlen: bool,
    chunk_size: int,
):
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if is_varlen:
        n_seqs = cu_seqlens.shape[0] - 1
        n_chunks = (
            (batch * tokens + chunk_size - 1) // chunk_size
            + cu_seqlens.shape[0]
            - 2
        )
        indices_shape = (n_chunks, 2)
        offsets_shape = (cu_seqlens.shape[0],)
    else:
        n_seqs = batch
        indices_shape = (0, 2)
        offsets_shape = (0,)
    final_shape = (
        (n_seqs, heads, key_dim, value_dim)
        if output_final_state
        else (0,)
    )
    return (
        v.new_empty(v.shape),
        q.new_empty(final_shape, dtype=torch.float32),
        q.new_empty(indices_shape, dtype=torch.int32),
        q.new_empty(offsets_shape, dtype=torch.int32),
    )


def _chunk_dplr_delta_rule_fwd_ctx_core(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    gk: Tensor,
    h0: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    has_initial_state: bool,
    output_final_state: bool,
    is_varlen: bool,
    chunk_size: int,
    store_context: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    cu = cu_seqlens if is_varlen else None
    layout = (
        build_varlen_chunk_layout(
            cu_seqlens,
            chunk_size,
            q.shape[0] * q.shape[1],
        )
        if is_varlen
        else None
    )
    initial_state = h0 if has_initial_state else None

    gi, ge = chunk_local_cumsum(
        gk,
        chunk_size,
        scale=RCP_LN2,
        cu_seqlens=cu,
        chunk_layout=layout,
    )
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu,
        chunk_layout=layout,
    )
    w, u, _ = prepare_wy_repr_fwd(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab=A_ab,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        chunk_layout=layout,
    )
    cp_context = _DPLR_CP_CONTEXT.get()
    if cp_context is not None:
        # CP: exchange the compact local boundary state (h = M @ h_in + c)
        # across ranks and fold it into the corrected initial state
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=kg,
            w=w,
            u=u,
            gk=gi,
            bg=bg,
            v=v,
            cu_seqlens=cu,
            initial_state=None,
            context=cp_context,
            chunk_size=chunk_size,
        )
    if not store_context:
        o, final_state, h_ctx, v_new_ctx = chunk_dplr_fwd_ho_context_elided(
            qg=qg,
            kg=kg,
            v=v,
            w=w,
            u=u,
            bg=bg,
            gk=gi,
            A_qk=A_qk,
            A_qb=A_qb,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu,
            chunk_size=chunk_size,
            chunk_layout=layout,
        )
    else:
        o, final_state, h_ctx, v_new_ctx = chunk_dplr_fwd_ho_with_context(
            qg=qg,
            kg=kg,
            v=v,
            w=w,
            u=u,
            bg=bg,
            gk=gi,
            A_qk=A_qk,
            A_qb=A_qb,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu,
            chunk_size=chunk_size,
            chunk_layout=layout,
        )

    if final_state is None:
        final_state = q.new_empty((0,), dtype=torch.float32)
    if layout is None:
        chunk_indices, chunk_offsets = _empty_layout_metadata(q)
    else:
        chunk_indices = layout.chunk_indices
        chunk_offsets = layout.chunk_offsets
    return o, final_state, chunk_indices, chunk_offsets, h_ctx, v_new_ctx


@torch.library.custom_op(
    "fla::chunk_dplr_delta_rule_fwd_ctx",
    mutates_args=(),
)
def _chunk_dplr_delta_rule_fwd_ctx_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    gk: Tensor,
    h0: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    has_initial_state: bool,
    output_final_state: bool,
    is_varlen: bool,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _chunk_dplr_delta_rule_fwd_ctx_core(
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        scale,
        has_initial_state,
        output_final_state,
        is_varlen,
        chunk_size,
        store_context=True,
    )


@_chunk_dplr_delta_rule_fwd_ctx_op.register_fake
def _chunk_dplr_delta_rule_fwd_ctx_fake(
    q,
    k,
    v,
    a,
    b,
    gk,
    h0,
    cu_seqlens,
    scale: float,
    has_initial_state: bool,
    output_final_state: bool,
    is_varlen: bool,
    chunk_size: int,
):
    del k, a, b, gk, scale, has_initial_state
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if is_varlen:
        n_seqs = cu_seqlens.shape[0] - 1
        n_chunks = (
            (batch * tokens + chunk_size - 1) // chunk_size
            + cu_seqlens.shape[0]
            - 2
        )
        indices_shape = (n_chunks, 2)
        offsets_shape = (cu_seqlens.shape[0],)
    else:
        n_seqs = batch
        n_chunks = batch * ((tokens + chunk_size - 1) // chunk_size)
        indices_shape = (0, 2)
        offsets_shape = (0,)
    final_shape = (
        (n_seqs, heads, key_dim, value_dim)
        if output_final_state
        else (0,)
    )
    return (
        v.new_empty(v.shape),
        q.new_empty(final_shape, dtype=torch.float32),
        q.new_empty(indices_shape, dtype=torch.int32),
        q.new_empty(offsets_shape, dtype=torch.int32),
        q.new_empty((n_chunks, heads, key_dim, value_dim)),
        v.new_empty(v.shape),
    )


@torch.library.custom_op(
    "fla::chunk_dplr_delta_rule_fwd_ctx_elided",
    mutates_args=(),
)
def _chunk_dplr_delta_rule_fwd_ctx_elided_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    gk: Tensor,
    h0: Tensor,
    cu_seqlens: Tensor,
    scale: float,
    has_initial_state: bool,
    output_final_state: bool,
    is_varlen: bool,
    chunk_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _chunk_dplr_delta_rule_fwd_ctx_core(
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        scale,
        has_initial_state,
        output_final_state,
        is_varlen,
        chunk_size,
        store_context=False,
    )


@_chunk_dplr_delta_rule_fwd_ctx_elided_op.register_fake
def _chunk_dplr_delta_rule_fwd_ctx_elided_fake(
    q,
    k,
    v,
    a,
    b,
    gk,
    h0,
    cu_seqlens,
    scale: float,
    has_initial_state: bool,
    output_final_state: bool,
    is_varlen: bool,
    chunk_size: int,
):
    outputs = _chunk_dplr_delta_rule_fwd_ctx_fake(
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        scale,
        has_initial_state,
        output_final_state,
        is_varlen,
        chunk_size,
    )
    return (
        *outputs[:4],
        q.new_empty((1,)).expand(outputs[4].shape),
        v.new_empty((1,)).expand(outputs[5].shape),
    )


def _chunk_dplr_setup_context(ctx, inputs, output):
    (
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        scale,
        has_initial_state,
        output_final_state,
        is_varlen,
        chunk_size,
    ) = inputs
    _, final_state, chunk_indices, chunk_offsets = output
    if not output_final_state:
        ctx.mark_non_differentiable(final_state)
    ctx.mark_non_differentiable(chunk_indices, chunk_offsets)
    ctx.save_for_backward(
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
    )
    ctx.scale = float(scale)
    ctx.has_initial_state = bool(has_initial_state)
    ctx.is_varlen = bool(is_varlen)
    ctx.chunk_size = int(chunk_size)
    ctx.cp_context = _DPLR_CP_CONTEXT.get()


def _chunk_dplr_backward(
    ctx,
    do,
    dht,
    _dchunk_indices,
    _dchunk_offsets,
):
    (
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
    ) = ctx.saved_tensors
    dht_arg = dht if dht is not None else q.new_empty((0,), dtype=torch.float32)
    token = _DPLR_CP_CONTEXT.set(getattr(ctx, "cp_context", None))
    try:
        dq, dk, dv, da, db, dgk, dh0 = _chunk_dplr_delta_rule_bwd_op(
            do,
            dht_arg,
            q,
            k,
            v,
            a,
            b,
            gk,
            h0,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            ctx.scale,
            ctx.has_initial_state,
            ctx.is_varlen,
            ctx.chunk_size,
        )
    finally:
        _DPLR_CP_CONTEXT.reset(token)
    return (
        dq,
        dk,
        dv,
        da,
        db,
        dgk,
        dh0 if ctx.has_initial_state else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_chunk_dplr_delta_rule_fwd_op.register_autograd(
    _chunk_dplr_backward,
    setup_context=_chunk_dplr_setup_context,
)


def _chunk_dplr_ctx_setup_context(ctx, inputs, output):
    (
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        scale,
        has_initial_state,
        output_final_state,
        is_varlen,
        chunk_size,
    ) = inputs
    _, final_state, chunk_indices, chunk_offsets, h_ctx, v_new_ctx = output
    if not output_final_state:
        ctx.mark_non_differentiable(final_state)
    ctx.mark_non_differentiable(chunk_indices, chunk_offsets)
    ctx.save_for_backward(
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        h_ctx,
        v_new_ctx,
    )
    ctx.scale = float(scale)
    ctx.has_initial_state = bool(has_initial_state)
    ctx.is_varlen = bool(is_varlen)
    ctx.chunk_size = int(chunk_size)
    ctx.cp_context = _DPLR_CP_CONTEXT.get()


def _chunk_dplr_ctx_backward_from_saved(
    ctx,
    saved_tensors,
    do,
    dht,
):
    (
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        h_ctx,
        v_new_ctx,
    ) = saved_tensors
    dht_arg = dht if dht is not None else q.new_empty((0,), dtype=torch.float32)
    token = _DPLR_CP_CONTEXT.set(getattr(ctx, "cp_context", None))
    try:
        dq, dk, dv, da, db, dgk, dh0 = _chunk_dplr_delta_rule_bwd_ctx_op(
            do,
            dht_arg,
            q,
            k,
            v,
            a,
            b,
            gk,
            h0,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            h_ctx,
            v_new_ctx,
            ctx.scale,
            ctx.has_initial_state,
            ctx.is_varlen,
            ctx.chunk_size,
        )
    finally:
        _DPLR_CP_CONTEXT.reset(token)
    return (
        dq,
        dk,
        dv,
        da,
        db,
        dgk,
        dh0 if ctx.has_initial_state else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _chunk_dplr_ctx_backward(
    ctx,
    do,
    dht,
    _dchunk_indices,
    _dchunk_offsets,
    _dh_ctx,
    _dv_new_ctx,
):
    return _chunk_dplr_ctx_backward_from_saved(
        ctx,
        ctx.saved_tensors,
        do,
        dht,
    )


def _checkpoint_context_is_materialized(tensor: Tensor) -> bool:
    required_bytes = tensor.numel() * tensor.element_size()
    return tensor.untyped_storage().nbytes() >= required_bytes


def _chunk_dplr_ctx_elided_backward(
    ctx,
    do,
    dht,
    _dchunk_indices,
    _dchunk_offsets,
    _dh_ctx,
    _dv_new_ctx,
):
    saved_tensors = ctx.saved_tensors
    h_ctx, v_new_ctx = saved_tensors[-2:]
    if not (
        _checkpoint_context_is_materialized(h_ctx)
        and _checkpoint_context_is_materialized(v_new_ctx)
    ):
        raise RuntimeError(
            "DPLR checkpoint context-write elision requires non-reentrant "
            "checkpoint recomputation before backward"
        )
    return _chunk_dplr_ctx_backward_from_saved(
        ctx,
        saved_tensors,
        do,
        dht,
    )


_chunk_dplr_delta_rule_fwd_ctx_op.register_autograd(
    _chunk_dplr_ctx_backward,
    setup_context=_chunk_dplr_ctx_setup_context,
)

_chunk_dplr_delta_rule_fwd_ctx_elided_op.register_autograd(
    _chunk_dplr_ctx_elided_backward,
    setup_context=_chunk_dplr_ctx_setup_context,
)


def chunk_dplr_delta_rule_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    safe_gate: bool = False,
    chunk_size: int | None = None,
    disable_recompute: bool = False,
    cp_context=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del cu_seqlens_cpu, safe_gate
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed; inputs must use [B, T, H, ...]"
        )
    if kwargs:
        raise TypeError(f"unexpected DPLR kwargs: {', '.join(sorted(kwargs))}")
    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens
    chunk_size = 16 if chunk_size is None else int(chunk_size)
    scale_f = float(q.shape[-1] ** -0.5 if scale is None else scale)
    n_seqs = len(cu_seqlens) - 1 if cu_seqlens is not None else q.shape[0]
    if initial_state is not None:
        h0 = initial_state
    elif cp_context is not None:
        # the corrected initial state is produced inside the op by the CP
        # boundary exchange; this buffer only carries its shape/ABI
        h0 = q.new_empty((n_seqs, q.shape[2], q.shape[3], v.shape[3]), dtype=torch.float32)
    else:
        h0 = q.new_empty((0,), dtype=torch.float32)
    cu = (
        _prepare_cuda_cu_seqlens(cu_seqlens, q.device)
        if cu_seqlens is not None
        else q.new_empty((0,), dtype=torch.int32)
    )
    is_compiling = torch.compiler.is_compiling()
    checkpoint_phase = _checkpoint_phase_for_dispatch(is_compiling)
    op_args = (
        q,
        k,
        v,
        a,
        b,
        gk,
        h0,
        cu,
        scale_f,
        initial_state is not None or cp_context is not None,
        output_final_state,
        cu_seqlens is not None,
        chunk_size,
    )
    token = _DPLR_CP_CONTEXT.set(cp_context)
    try:
        if disable_recompute:
            if not is_compiling and checkpoint_phase == _CHECKPOINT_PHASE_FORWARD or (
                not is_compiling
                and checkpoint_phase == _CHECKPOINT_PHASE_NORMAL
                and not torch.is_grad_enabled()
            ):
                o, final_state, _, _, _, _ = (
                    _chunk_dplr_delta_rule_fwd_ctx_elided_op(*op_args)
                )
            else:
                o, final_state, _, _, _, _ = _chunk_dplr_delta_rule_fwd_ctx_op(
                    *op_args
                )
        else:
            o, final_state, _, _ = _chunk_dplr_delta_rule_fwd_op(*op_args)
    finally:
        _DPLR_CP_CONTEXT.reset(token)
    return o, final_state if output_final_state else None
