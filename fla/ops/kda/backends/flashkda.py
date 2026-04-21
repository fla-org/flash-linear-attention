# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Moonshot AI

"""FlashKDA CUTLASS forward backend for chunk_kda (inference only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class FlashKDABackend(BaseBackend):
    """Copyright (c) 2026 Moonshot AI

    Fused CUTLASS forward (replaces the multi-kernel Triton path).
    https://github.com/MoonshotAI/FlashKDA

    Enabled only under ``torch.inference_mode()``; disable with ``FLA_FLASH_KDA=0``.
    The kernel fuses q/k L2 norm, beta sigmoid, and the KDA gate, so callers must pass
    raw tensors and set all three ``*_in_kernel`` flags.
    """

    backend_type = "flashkda"
    package_name = "flash_kda"
    env_var = "FLA_FLASH_KDA"
    default_enable = True
    priority = 3

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
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if torch.is_grad_enabled():
            return False, "FlashKDA only supports inference mode"
        if q.dtype != torch.bfloat16:
            return False, f"FlashKDA requires bfloat16, got {q.dtype}"
        if q.shape[-1] != 128:
            return False, f"FlashKDA requires K=128, got {q.shape[-1]}"
        if v.shape[-1] != 128:
            return False, f"FlashKDA requires V=128, got {v.shape[-1]}"
        if v.shape[2] != q.shape[2]:
            return False, f"FlashKDA does not support GVA (HV={v.shape[2]} != H={q.shape[2]})"
        if not use_gate_in_kernel:
            return False, "FlashKDA requires use_gate_in_kernel=True"
        if not use_qk_l2norm_in_kernel:
            return False, "FlashKDA requires use_qk_l2norm_in_kernel=True"
        if not use_beta_sigmoid_in_kernel:
            return False, "FlashKDA requires use_beta_sigmoid_in_kernel=True"
        if not transpose_state_layout:
            return False, "FlashKDA requires transpose_state_layout=True"
        if cp_context is not None:
            return False, "FlashKDA does not support context parallel"
        if return_intermediate_states:
            return False, "FlashKDA does not support return_intermediate_states"
        if not safe_gate:
            return False, "FlashKDA requires safe_gate=True"
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
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        **kwargs,
    ):
        import flash_kda

        if scale is None:
            scale = q.shape[-1] ** -0.5

        K = q.shape[-1]
        HV, V = v.shape[2], v.shape[-1]
        N = len(cu_seqlens) - 1 if cu_seqlens is not None else q.shape[0]

        if dt_bias is not None and dt_bias.ndim == 1:
            dt_bias = dt_bias.view(HV, -1)

        out_buf = torch.empty_like(v)

        if initial_state is not None:
            initial_state = initial_state.contiguous()

        final_state = None
        if output_final_state:
            final_state = torch.empty(N, HV, K, V, dtype=torch.float32, device=q.device)

        if cu_seqlens is not None and cu_seqlens.dtype != torch.long:
            cu_seqlens = cu_seqlens.to(torch.long)

        flash_kda.fwd(
            q, k, v, g, beta,
            scale,
            out_buf,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )

        return out_buf, final_state
