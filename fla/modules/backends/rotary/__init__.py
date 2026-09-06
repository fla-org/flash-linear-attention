# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backend registration for rotary."""

import torch

from fla.backends import BackendRegistry, BaseBackend


class TritonAscendRotaryBackend(BaseBackend):
    backend_type = "triton_ascend"
    priority = 0

    @classmethod
    def is_available(cls) -> bool:
        from fla.utils import IS_NPU
        return IS_NPU

    def rotary_embedding_fwdbwd(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlen_offsets: int | torch.Tensor = 0,
        cu_seqlens: torch.Tensor | None = None,
        interleaved: bool = False,
        inplace: bool = False,
        conjugate: bool = False,
        chunk_indices: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        from fla.modules.backends.rotary.triton_ascend import rotary_embedding_fwdbwd
        return rotary_embedding_fwdbwd(
            x=x,
            cos=cos,
            sin=sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=conjugate,
            chunk_indices=chunk_indices,
        )


rotary_registry = BackendRegistry("modules.rotary")
rotary_registry.register(TritonAscendRotaryBackend())
