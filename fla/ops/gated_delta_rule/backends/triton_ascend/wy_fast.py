from typing import Optional, Tuple

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

from fla.ops.utils.op import exp2
from .utils import prepare_chunk_indices


def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.heuristics({
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "USE_G": lambda args: args["g"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    B: tl.constexpr,
    task_num: tl.constexpr,
    num_core: tl.constexpr,
):
    T_max = T
    core_id = tl.program_id(0)

    for task_id in tl.range(core_id, task_num, num_core):
        i_t_o = task_id // (B * HV)
        i_bh = task_id % (B * HV)
        i_b, i_h = i_bh // HV, i_bh % HV
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
            bos_bh = bos
        else:
            i_t = i_t_o
            bos, eos = i_b * T, i_b * T + T
            bos_bh = i_b * HV * T_max

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T

        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = A + (bos * HV + i_h) * BT + offs_t_2d * (HV * BT) + offs_bt * 1
        mask_A = mask_t[:, None]
        b_A = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

        ptr_beta = beta + bos_bh + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = (mask_t[:, None]) & (offs_v < V)

            ptr_v = v + (bos * HV + i_h) * V + offs_t_2d * (HV * V) + offs_v * 1
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)

            b_vb = b_v * b_beta[:, None]
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)

            ptr_u = u + (bos * HV + i_h) * V + offs_t_2d * (HV * V) + offs_v * 1
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        if USE_G:
            ptr_g = g + bos_bh + i_h * T_max + global_offs_t
            b_g = exp2(tl.load(ptr_g, mask=mask_t, other=0.0)).to(tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = (mask_t[:, None]) & (offs_k < K)
            ptr_k = (
                k
                + (bos * H + i_h // (HV // H)) * K
                + offs_t_2d * (H * K)
                + offs_k * 1
            )
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)

            b_kb = b_k * b_beta[:, None]
            if USE_G:
                b_kb = b_kb * b_g[:, None]
            b_w = tl.dot(b_A, b_kb)

            ptr_w = w + (bos * HV + i_h) * K + offs_t_2d * (HV * K) + offs_k * 1
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


def recompute_w_u_fwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[-2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = 64
    BV = 64

    u = torch.empty_like(v)
    w = k.new_empty(B, T, HV, K)
    beta = beta.transpose(1, 2).contiguous()
    if g is not None:
        g = g.transpose(1, 2).contiguous()

    num_core = get_npu_properties()["num_aicore"]
    task_num = NT * B * HV
    recompute_w_u_fwd_kernel[(num_core,)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        B=B,
        task_num=task_num,
        num_core=num_core,
        num_warps=4,
        num_stages=3,
    )
    return w, u
