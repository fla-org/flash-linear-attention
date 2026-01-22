# # -*- coding: utf-8 -*-
# # Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from fla.ops.oja2.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.oja2.chunk_kkt import chunk_scaled_dot_kkt_fwd, chunk_scaled_dot_kkt_bwd_gk
from fla.ops.oja2.chunk_h import (
    chunk_oja2_fwd_h, 
    chunk_oja2_bwd_dhu, 
    chunk_oja2_bwd_dvwg_h)
from fla.ops.oja2.chunk_o import (
    chunk_oja2_fwd_o, 
    chunk_oja2_bwd_dA, 
    chunk_oja2_bwd_dqk, 
    chunk_oja2_bwd_dv_o, 
    )




def chunk_oja2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    g_cumsum: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None
):  
    if g_cumsum:
        gv = chunk_local_cumsum(gv, chunk_size=64, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=v,
        gk=gv,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    # w = Avg, u = Ak
    w, u, vg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gv=gv,
        cu_seqlens=cu_seqlens,
    )
    # grid in K
    h, k_new, final_state = chunk_oja2_fwd_h(
        v=vg,
        w=w,
        u=u,
        gv=gv,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    _, o = chunk_oja2_fwd_o(
        q=q,
        k=k_new,
        v=v,
        h=h,
        gv=gv,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return gv, o, A, final_state


def chunk_oja2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dgk: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):  
    w, u, vg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gv=gv,
        cu_seqlens=cu_seqlens,
    )
    # w = w.to(torch.float32)
    # u = u.to(torch.float32)
    # vg = vg.to(torch.float32)
    h, k_new, _ = chunk_oja2_fwd_h(
        v=vg,
        w=w,
        u=u,
        gv=gv,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    """
    对于S = g_last * S + Vg @ (U - WS)
    O = g_i * (QS + tri(Q @ (U - WS)) (V/g))
    1. 计算dA = do * g_i * v/g
    2. 计算dA里面的dk_new=dA * q, 顺便收集tri(A), 计算全部dq = do * g_i * S  ::  🚩所有dq完毕
    3. 计算dS, 进一步收集所有S里面的dk_new, 计算递归中的dS以及dk_new中的dS  ::  🚩所有dk_new(du), dS, dS0完毕
    4. 计算o递归里的dv = do * g_i * A(细粒度), 顺便收集dg
    5. 计算S中的dv以及dk_new里的dw  ::  🚩所有dw, dv完毕
    @ 至此dq, dk_new, dv, dw, du, dS, dS0完毕,还需要最后解开WY表征
    6. 先计算W = M * beta * AV以及U = M * beta * K外面的dbeta, dk, dv, dg, 存下来dM
    7. 通过存下来的dM计算内部的dv, dbeta, dg  ::  🚩所有dq, dk, dv, dw, du, dS, dS0, dbeta, dg完毕
    """ 
    # grid = (NV, NT * NC * NC, B * H)
    
    dAqk = chunk_oja2_bwd_dA(
        v=v,
        gv=gv,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    # (NK, NT, B * H)
    Aqk, dq, dk_new = chunk_oja2_bwd_dqk(
        q=q,
        k=k_new,
        h=h,
        gv=gv,
        dA=dAqk,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    # (NK, B*H)
    dh, dh0, dk_new = chunk_oja2_bwd_dhu(
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
        states_in_fp32=False,
    )

    # grid = (NV, NT, B * H)
    dv, dw, dgv_last = chunk_oja2_bwd_dvwg_h(
        k=k_new,
        v=v,
        gv=gv,
        h=h,
        dh=dh,
        dk=dk_new,
        dgk=dgk,
        cu_seqlens=cu_seqlens,
    )

    # (NV, NT * NC, B * H)
    dv, dgv1 = chunk_oja2_bwd_dv_o(
        v=v,
        gv=gv,
        o=o,
        A=Aqk,
        dv=dv,
        do=do,
        cu_seqlens=cu_seqlens,
    )

    # (NT, B * H)
    dk, dv1, db, dgv2, dAvv = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        gv=gv,
        A=A,
        dw=dw,
        du=dk_new,
        cu_seqlens=cu_seqlens,
    )

    # (NK, NT * NC, B * H)
    dv2, dgv3, db2 = chunk_scaled_dot_kkt_bwd_gk(
        k=v,
        g=gv,
        beta=beta,
        dA=dAvv,
        cu_seqlens=cu_seqlens,
    )
    
    dv = dv.add_(dv1).add_(dv2)
    db = db.add_(db2)
    dgv = dgv_last.add_(chunk_local_cumsum(dgv1.add_(dgv2).add_(dgv3), chunk_size=64, reverse=True, cu_seqlens=cu_seqlens))
    return dq, dk, dv, db, dgv, dh0


class ChunkOJA2Function(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gv: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_q_l2norm: bool = False,
        use_k_l2norm: bool = False,
    ):
        q_rstd, k_rstd = None, None
        if use_q_l2norm:
            q, q_rstd = l2norm_fwd(q)
        if use_k_l2norm:
            k, k_rstd = l2norm_fwd(k)

        gv, o, A, final_state = chunk_oja2_fwd(
            q=q,
            k=k,
            v=v,
            gv=gv,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, gv, beta, A, o, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_q_l2norm = use_q_l2norm
        ctx.use_k_l2norm = use_k_l2norm
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, gv, beta, A, o, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_oja2_bwd(
            q=q,
            k=k,
            v=v,
            gv=gv,
            beta=beta,
            A=A,
            o=o,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
        )
        # === 遍历检查所有梯度，定位具体是哪个 NaN ===
        # 将变量名和tensor对应起来
        # grad_tensors = {
        #     'dq': dq, 'dk': dk, 'dv': dv, 'db': db, 
        #     'dg': dg, 'dh0': dh0
        # }

        # for name, t in grad_tensors.items():
        #     if t is not None and torch.isnan(t).any():
        #         import os
        #         import torch.distributed as dist
                
        #         # 获取 Rank ID
        #         # try:
        #         #     rank = dist.get_rank() if dist.is_initialized() else 0
        #         # except:
        #         #     rank = 0
        #         rank = 0

        #         base_dir = "/mnt/moonfs/hujiaxi-m2/oja_nan_12"
        #         os.makedirs(base_dir, exist_ok=True)
                
        #         # 保存路径：nan_dump_rank{卡号}.pt
        #         save_path = os.path.join(base_dir, f"nan_dump_rank{rank}.pt")
                
        #         torch.save({
        #             "q": q,
        #             "k": k,
        #             "v": v,
        #             "beta": beta,
        #             "gv": gv,
        #             "do": do,
        #             "cu_seqlens": cu_seqlens,
        #             "error_source": name  # 顺便把出错的变量名也存进文件
        #         }, save_path)
                
        #         # 明确报错：指出是哪个变量出的问题
        #         raise RuntimeError(f"NaN detected in [{name}] on Rank {rank}! Context saved to: {save_path}")
        if ctx.use_q_l2norm:
            dq = l2norm_bwd(q, q_rstd, dq)
        if ctx.use_k_l2norm:
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(gv), db.to(beta), None, dh0, None, None, None, None


@torch.compiler.disable
def chunk_oja2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gv: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_q_l2norm: bool = False,
    use_k_l2norm: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if 'use_qk_l2norm_in_kernel' in kwargs and (not use_q_l2norm and not use_k_l2norm):
        use_q_l2norm = True
        use_k_l2norm = True

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkOJA2Function.apply(
        q,
        k,
        v,
        gv,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_q_l2norm,
        use_k_l2norm
    )
    return o, final_state
