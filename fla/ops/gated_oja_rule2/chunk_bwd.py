# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.gated_oja_rule.chunk_h import chunk_oja_bwd_dhu, chunk_oja_bwd_dvwg_h, chunk_oja_fwd_h
from fla.ops.gated_oja_rule.chunk_o import chunk_oja_bwd_dA, chunk_oja_bwd_dqk, chunk_oja_bwd_dv_o
from fla.ops.gated_oja_rule2.chunk_kkt import chunk_oja2_kkt_bwd_gk
from fla.ops.gated_oja_rule2.wy_fast import prepare_wy_repr_bwd_oja2, recompute_w_u_fwd_oja2
from fla.ops.utils import chunk_local_cumsum


def chunk_oja2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    A: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    w, u, vg = recompute_w_u_fwd_oja2(
        k=k,
        v=v,
        b_gate=b,
        c_gate=c,
        A=A,
        gv=gv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, k_new, _ = chunk_oja_fwd_h(
        v=vg,
        w=w,
        u=u,
        gv=gv,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dAqk = chunk_oja_bwd_dA(
        v=v,
        gv=gv,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    Aqk, dq, dk_new = chunk_oja_bwd_dqk(
        q=q,
        k=k_new,
        h=h,
        gv=gv,
        dA=dAqk,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dh, dh0, dk_new = chunk_oja_bwd_dhu(
        q=q,
        vg=vg,
        w=w,
        gv=gv,
        h0=initial_state,
        dht=dht,
        do=do,
        dk=dk_new,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        states_in_fp32=False,
        chunk_size=chunk_size,
    )

    dv, dw, dgv_last = chunk_oja_bwd_dvwg_h(
        k=k_new,
        v=v,
        gv=gv,
        h=h,
        dh=dh,
        dk=dk_new,
        dgk=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dv, dgv1 = chunk_oja_bwd_dv_o(
        v=v,
        gv=gv,
        o=o,
        A=Aqk,
        dv=dv,
        do=do,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dk, dv1, db, dc, dgv2, dAvv = prepare_wy_repr_bwd_oja2(
        k=k,
        v=v,
        b_gate=b,
        c_gate=c,
        A=A,
        dw=dw,
        du=dk_new,
        gv=gv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    dv2, dgv3, db2 = chunk_oja2_kkt_bwd_gk(
        k=v,
        g=gv,
        b=b,
        dA=dAvv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    dv = dv.add_(dv1).add_(dv2)
    db = db.add_(db2)
    dgv = dgv_last.add_(chunk_local_cumsum(
        g=dgv1.add_(dgv2).add_(dgv3),
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    ))
    return dq, dk, dv, db, dc, dgv, dh0
