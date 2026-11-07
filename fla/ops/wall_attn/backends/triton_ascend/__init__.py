# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Triton-Ascend Ascend NPU backend for Wall attention ops."""

from __future__ import annotations

import torch

from fla.backends import BaseBackend, register


@register('wall_attn')
class WallAttnTritonAscendBackend(BaseBackend):
    """Ascend NPU backend for the Wall attention training and decode entries."""

    backend_type = 'triton_ascend'
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    @staticmethod
    def _verify_head_dim(k: torch.Tensor) -> tuple[bool, str | None]:
        if k.shape[-1] > 256:
            return False, f'wall attention NPU kernels require K <= 256, got K={k.shape[-1]}'
        return True, None

    def parallel_wall_attn_fwd_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cumsum: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if q.device.type != 'npu':
            return False, 'parallel_wall_attn_fwd NPU override requires NPU tensors'
        return self._verify_head_dim(k)

    def parallel_wall_attn_fwd(self, *args, **kwargs):
        from fla.ops.wall_attn.backends.triton_ascend.parallel import parallel_wall_attn_fwd_npu
        return parallel_wall_attn_fwd_npu(*args, **kwargs)

    def parallel_wall_attn_bwd_verifier(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        g_cumsum: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if q.device.type != 'npu':
            return False, 'parallel_wall_attn_bwd NPU override requires NPU tensors'
        return self._verify_head_dim(k)

    def parallel_wall_attn_bwd(self, *args, **kwargs):
        from fla.ops.wall_attn.backends.triton_ascend.parallel import parallel_wall_attn_bwd_npu
        return parallel_wall_attn_bwd_npu(*args, **kwargs)

    def parallel_wall_attn_decode_verifier(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        p_curr: torch.Tensor,
        k_tilde: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[bool, str | None]:
        if q.device.type != 'npu':
            return False, 'parallel_wall_attn_decode NPU override requires NPU tensors'
        return self._verify_head_dim(k_tilde)

    def parallel_wall_attn_decode(self, *args, **kwargs):
        from fla.ops.wall_attn.backends.triton_ascend.decode import parallel_wall_attn_decode_npu
        return parallel_wall_attn_decode_npu(*args, **kwargs)


__all__ = ['WallAttnTritonAscendBackend']
