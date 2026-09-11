# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""FlashKDA CUDA training backend for chunk_kda (forward + backward).

Provided by the `flash_kda` package built with training support
(https://github.com/MoonshotAI/FlashKDA, training fork). The CUDA kernels
replicate the Triton `chunk_kda_fwd`/`chunk_kda_bwd` pipeline stage by stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.backends import BaseBackend
from fla.ops.common.gate import fused_beta_sigmoid, fused_beta_sigmoid_bwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class ChunkKDACUDAFunction(torch.autograd.Function):
    """Mirrors ChunkKDAFunction, dispatching each stage to the CUDA kernels."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        chunk_size: int = 64,
    ):
        from flash_kda.train import chunk_kda_train_fwd

        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        beta_raw = beta
        if use_beta_sigmoid_in_kernel:
            beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)

        chunk_indices = None
        if cu_seqlens is not None:
            from flash_kda.train import prepare_chunk_indices
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        g_input = g

        (o, final_state, g_cumsum, Aqk, Akk) = chunk_kda_train_fwd(
            q=q,
            k=k,
            v=v,
            g=g_input,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
        )

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g_cumsum, g_input, beta_raw, beta, A_log, dt_bias, Aqk, Akk,
            initial_state, cu_seqlens, chunk_indices
        )
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.state_v_first = state_v_first
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        from flash_kda.train import chunk_kda_train_bwd

        (q, q_rstd, k, k_rstd, v, g_cumsum, g_input, beta_raw, beta, A_log, dt_bias, Aqk, Akk,
         initial_state, cu_seqlens, chunk_indices) = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_train_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            g=g_cumsum,
            g_org=g_input if ctx.use_gate_in_kernel else None,
            state_v_first=ctx.state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = fused_beta_sigmoid_bwd(beta_raw, db, scale=2.0 if ctx.allow_neg_eigval else 1.0)

        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g_input), db.to(beta_raw), dA, dbias, None, dh0,
                None, None, None, None, None, None, None, None, None, None, None)


class FlashKDATrainBackend(BaseBackend):
    """CUDA training path for chunk_kda from the flash_kda package.

    Lower priority than the inference-only FlashKDA backend and only accepts
    calls with grad enabled; everything unsupported falls back to Triton.
    """

    backend_type = "flash_kda_train"
    package_name = "flash_kda_train_C"
    env_var = "FLA_FLASH_KDA_TRAIN"
    default_enable = False
    priority = 4

    def chunk_kda_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if not torch.is_grad_enabled():
            return False, "FlashKDATrain only accepts training (grad-enabled) calls"
        if q.dtype != torch.bfloat16:
            return False, f"FlashKDATrain requires bfloat16, got {q.dtype}"
        if q.shape[-1] != 128:
            return False, f"FlashKDATrain requires K=128, got {q.shape[-1]}"
        if v.shape[-1] != 128:
            return False, f"FlashKDATrain requires V=128, got {v.shape[-1]}"
        if v.shape[2] != q.shape[2]:
            return False, f"FlashKDATrain does not support GVA (HV={v.shape[2]} != H={q.shape[2]})"
        if initial_state is not None and initial_state.dtype != torch.float32:
            return False, f"FlashKDATrain requires fp32 initial_state, got {initial_state.dtype}"
        if cp_context is not None:
            return False, "FlashKDATrain does not support context parallel"
        if return_intermediate_states:
            return False, "FlashKDATrain does not support return_intermediate_states"
        if kwargs.get("chunk_size", 64) != 64:
            return False, f"FlashKDATrain requires chunk_size=64, got {kwargs.get('chunk_size')}"
        return True, None

    def chunk_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ):
        A_log = kwargs.get("A_log")
        dt_bias = kwargs.get("dt_bias")
        chunk_size = kwargs.get("chunk_size", 64)
        if scale is None:
            scale = q.shape[-1] ** -0.5

        return ChunkKDACUDAFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            scale,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            use_gate_in_kernel,
            use_beta_sigmoid_in_kernel,
            allow_neg_eigval,
            state_v_first,
            cu_seqlens,
            cu_seqlens_cpu,
            safe_gate,
            lower_bound,
            chunk_size,
        )
