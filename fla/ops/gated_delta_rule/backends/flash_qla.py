# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 Qwen Team, Alibaba Cloud

"""FlashQLA backend for chunk_gated_delta_rule."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_NVIDIA_HOPPER, IS_NVIDIA_SM100

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class FlashQLABackend(BaseBackend):
    """Copyright (c) 2026 Qwen Team, Alibaba Cloud

    Fused TileLang forward and backward with intra-card CP (replaces the multi-kernel Triton path).
    https://github.com/QwenLM/FlashQLA

    Disable with ``FLA_FLASH_QLA=0``.
    """

    backend_type = "flash_qla"
    package_name = "flash_qla"
    env_var = "FLA_FLASH_QLA"
    default_enable = True
    priority = 3

    def chunk_gated_delta_rule_verifier(
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
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if not (IS_NVIDIA_HOPPER or IS_NVIDIA_SM100):
            return False, "FlashQLA requires NVIDIA SM90 or SM100"
        if q.dtype != torch.float16 and q.dtype != torch.bfloat16:
            return False, f"FlashQLA requires dtype float16 or bfloat16, got {q.dtype}"
        if not (q.dtype == k.dtype == v.dtype):
            return False, f"FlashQLA requires q, k, v to have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
        if q.shape[-1] != 128:
            return False, f"FlashQLA requires K=128, got {q.shape[-1]}"
        if v.shape[-1] != 128:
            return False, f"FlashQLA requires V=128, got {v.shape[-1]}"
        if kwargs.get('use_gate_in_kernel'):
            return False, "FlashQLA does not support use_gate_in_kernel"
        if use_beta_sigmoid_in_kernel:
            return False, "FlashQLA does not support use_beta_sigmoid_in_kernel"
        if allow_neg_eigval:
            return False, "FlashQLA does not support allow_neg_eigval"
        if 'transpose_state_layout' in kwargs:
            return False, "FlashQLA does not support the deprecated transpose_state_layout"
        if cp_context is not None:
            return False, "FlashQLA does not support inter-card context parallel"
        return True, None

    def chunk_gated_delta_rule(
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
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ):
        import flash_qla

        return flash_qla.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            auto_cp=True,
        )
