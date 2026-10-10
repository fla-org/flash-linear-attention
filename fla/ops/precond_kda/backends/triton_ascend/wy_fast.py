# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Precond-KDA WY-recompute kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

import torch
import triton

from fla.ops.precond_kda.wy_fast import recompute_w_u_fwd_kernel
from fla.ops.utils import prepare_chunk_indices

# triton-ascend accepts 'ieee'/'hf32' only; hf32 is the Ascend analogue of tf32.
_NPU_DOT_PRECISION = 'hf32'


def recompute_w_u_fwd_npu(
    k: torch.Tensor,
    k_precond: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    q: torch.Tensor | None = None,
    output_qg: bool = False,
    output_kg: bool = True,
):
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if output_qg and q is not None else None
    kg = torch.empty_like(k_precond) if output_kg else None

    grid = (NT, B * H)
    recompute_w_u_fwd_kernel[grid](
        q=q,
        k=k,
        k_precond=k_precond,
        qg=qg,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        DOT_PRECISION=_NPU_DOT_PRECISION,
    )
    return w, u, qg, kg
