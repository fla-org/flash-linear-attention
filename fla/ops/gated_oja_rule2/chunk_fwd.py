# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.gated_oja_rule.chunk_h import chunk_oja_fwd_h
from fla.ops.gated_oja_rule.chunk_o import chunk_oja_fwd_o
from fla.ops.gated_oja_rule2.chunk_kkt import chunk_oja2_kkt_fwd
from fla.ops.gated_oja_rule2.wy_fast import recompute_w_u_fwd_oja2
from fla.ops.utils import chunk_local_cumsum, solve_tril


def chunk_oja2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    gv = chunk_local_cumsum(gv, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    A = chunk_oja2_kkt_fwd(
        k=v,
        gk=gv,
        b=b,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
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
    h, k_new, final_state = chunk_oja_fwd_h(
        v=vg,
        w=w,
        u=u,
        gv=gv,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    _, o = chunk_oja_fwd_o(
        q=q,
        k=k_new,
        v=v,
        h=h,
        gv=gv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return gv, o, A, final_state
