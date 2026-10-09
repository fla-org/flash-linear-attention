# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for simple GLA ops."""

from __future__ import annotations

import torch

from fla.backends import BaseBackend, register_backend


def _verify_npu(args, kwargs) -> tuple[bool, str | None]:
    from fla.utils import IS_NPU
    if not IS_NPU:
        return False, "not running on NPU"
    q = args[0] if args else kwargs.get("q")
    if q is None or q.device.type != "npu":
        return False, "input device is not NPU"
    return True, None


@register_backend('simple_gla')
class TritonAscendSimpleGLABackend(BaseBackend):
    """Ascend NPU backend for the simple GLA entry points.

    Retention's four entry points (``chunk``, ``fused_chunk``, ``parallel`` and
    ``fused_recurrent``) all wrap simple GLA, so this single backend covers them.
    """

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def fused_chunk_simple_gla_verifier(self, *args, **kwargs):
        return _verify_npu(args, kwargs)

    def fused_chunk_simple_gla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor = None,
        g_gamma: torch.Tensor = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        from fla.ops.simple_gla.backends.triton_ascend.fused_chunk import fused_chunk_simple_gla_npu
        return fused_chunk_simple_gla_npu(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    def parallel_simple_gla_verifier(self, *args, **kwargs):
        return _verify_npu(args, kwargs)

    def parallel_simple_gla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        scale: float | None = None,
        output_attentions: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
    ):
        from fla.ops.simple_gla.backends.triton_ascend.parallel import parallel_simple_gla_npu
        return parallel_simple_gla_npu(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    def fused_recurrent_simple_gla_verifier(self, *args, **kwargs):
        return _verify_npu(args, kwargs)

    def fused_recurrent_simple_gla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor = None,
        g_gamma: torch.Tensor = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        reverse: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ):
        from fla.ops.simple_gla.backends.triton_ascend.fused_recurrent import fused_recurrent_simple_gla_npu
        return fused_recurrent_simple_gla_npu(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
