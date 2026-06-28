# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fused Triton inference backend for GDN-2."""

from __future__ import annotations

import torch

from fla.ops.backends import BaseBackend


class GDN2FusedInferBackend(BaseBackend):
    backend_type = "fused_infer"
    package_name = "triton"

    # 仿照 PR915
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

    def chunk_gdn2_verifier(
        self,
        q,
        k,
        v,
        g,
        b,
        w,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if torch.is_grad_enabled():
            return False, "Fused infer backend only supports inference mode"

        if return_intermediate_states:
            return False, "Fused infer backend does not support return_intermediate_states"

        # 用户没传 chunk_size 时：
        # fused backend 默认按 16 处理；
        # fallback 原函数仍会使用原本默认 64。
        chunk_size = kwargs.get("chunk_size", 16)
        if chunk_size != 16:
            return False, f"Fused infer backend requires chunk_size=16, got {chunk_size}"

        return True, None
    def chunk_gdn2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        **kwargs,
    ):
        print("[GDN2 fused] SELECTED")

        from fla.ops.gdn2.chunk_fwd_infer import chunk_gdn2_fwd_infer

        return chunk_gdn2_fwd_infer(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first,
            **kwargs,
        )