# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register_backend


@register_backend('modules.rotary')
class TritonAscendBackend(BaseBackend):
    """Ascend NPU backend for rotary."""

    backend_type = "triton_ascend"
    package_name = None
    env_var = None
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def rotary_embedding_fwdbwd(
        self,
        x,
        cos,
        sin,
        seqlen_offsets=0,
        cu_seqlens=None,
        interleaved=False,
        inplace=False,
        conjugate=False,
        chunk_indices=None,
    ):
        from fla.modules.rotary.backends.triton_ascend.ops import rotary_embedding_fwdbwd_npu
        return rotary_embedding_fwdbwd_npu(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=conjugate,
            chunk_indices=chunk_indices,
        )
