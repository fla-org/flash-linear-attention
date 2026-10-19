# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.utils import input_guard


def chunk_kda_bwd_intra_verifier(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    use_graph: bool = False,
) -> tuple[bool, str | None]:
    if torch.is_grad_enabled():
        return False, 'AscendC intra backward does not support higher-order autograd'
    if q.device.type != 'npu' or 'Ascend950' not in torch.npu.get_device_name(q.device):
        return False, 'AscendC intra backward currently requires Ascend950'
    if use_graph or cu_seqlens is not None or chunk_indices is not None:
        return False, 'AscendC intra backward currently supports dense eager calls only'
    if not safe_gate or chunk_size != 64:
        return False, 'AscendC intra backward requires safe_gate=True and chunk_size=64'
    if q.ndim != 4 or min(q.shape) <= 0 or q.shape[-1] not in (64, 128, 256):
        return False, 'AscendC intra backward requires nonempty [B, T, H, K] with K=64/128/256'
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False, 'AscendC intra backward requires bfloat16 q/k'
    if any(x.shape != q.shape for x in (k, g, dq, dk, dg)):
        return False, 'AscendC intra backward requires matching vector shapes without GVA'
    if beta.shape != q.shape[:-1] or db.shape != beta.shape or beta.dtype not in (torch.bfloat16, torch.float32):
        return False, 'AscendC intra backward requires beta/db [B, T, H] and bfloat16 or float32 beta'
    if dAqk.shape != (*q.shape[:-1], chunk_size) or dAkk.shape != dAqk.shape:
        return False, 'AscendC intra backward requires dAqk/dAkk [B, T, H, chunk_size]'
    if any(x.dtype != torch.float32 for x in (g, dAqk, dAkk, dq, dk, db, dg)):
        return False, 'AscendC intra backward requires float32 cumulative gates and gradients'
    if any(x.device != q.device for x in (k, g, beta, dAqk, dAkk, dq, dk, db, dg)):
        return False, 'AscendC intra backward requires all tensors on the same NPU'
    return True, None


@input_guard
def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    use_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from fla_npu.ops.ascendc import chunk_kda_bwd_intra as npu_chunk_kda_bwd_intra

    return npu_chunk_kda_bwd_intra(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        layout='BSND',
    )
