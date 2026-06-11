# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Copyright (c) 2026 FlagOS Contributors

"""Fused Triton inference backend for KDA operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class KDAFusedInferBackend(BaseBackend):

    backend_type = "fused_infer"
    package_name = "triton"
    env_var = "FLA_FUSED_INFER"
    default_enable = True
    priority = 4

    @classmethod
    def is_available(cls) -> bool:
        try:
            import triton  # noqa: F401
            return True
        except ImportError:
            return False

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
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if torch.is_grad_enabled():
            return False, "Fused infer backend only supports inference mode"
        if return_intermediate_states:
            return False, "Fused infer backend does not support return_intermediate_states"
        chunk_size = kwargs.get("chunk_size", 16)
        if chunk_size != 16:
            return False, f"Fused infer backend requires chunk_size=16, got {chunk_size}"
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
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from fla.ops.kda.chunk_fwd_infer import chunk_kda_fwd_infer
        return chunk_kda_fwd_infer(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            cp_context=cp_context,
            **kwargs,
        )
