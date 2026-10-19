# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.utils.constant import RCP_LN2
from fla.utils import input_guard


def kda_gate_chunk_cumsum_verifier(
    g: torch.Tensor,
    A_log: torch.Tensor | None,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    lower_bound: float | None = None,
    use_graph: bool = False,
    **kwargs,
) -> tuple[bool, str | None]:
    if torch.is_grad_enabled():
        return False, 'AscendC gate requires disabled gradient recording, including inside custom autograd'
    if g.device.type != 'npu' or 'Ascend950' not in torch.npu.get_device_name(g.device):
        return False, 'AscendC gate currently requires Ascend950'
    if use_graph or cu_seqlens is not None or chunk_indices is not None or kwargs:
        return False, 'AscendC gate currently supports dense eager calls only'
    if chunk_size not in (32, 64, 128) or scale != RCP_LN2:
        return False, 'AscendC gate requires chunk_size=32/64/128 and scale=RCP_LN2'
    if (output_dtype or g.dtype) != torch.float32:
        return False, 'AscendC gate produces float32 output'
    if g.ndim != 4 or min(g.shape) <= 0 or g.shape[-1] not in (64, 128, 256):
        return False, 'AscendC gate requires nonempty [B, T, H, K] with K=64/128/256'
    if g.dtype not in (torch.bfloat16, torch.float32):
        return False, 'AscendC gate requires bfloat16 or float32 input'
    if A_log is None or A_log.shape != (g.shape[2],) or A_log.dtype != torch.float32 or A_log.device != g.device:
        return False, 'AscendC gate requires float32 A_log [H] on the gate device'
    if dt_bias is not None:
        if dt_bias.shape != (g.shape[2] * g.shape[3],) or dt_bias.dtype != torch.float32 or dt_bias.device != g.device:
            return False, 'AscendC gate requires float32 dt_bias [H*K] on the gate device'
    if lower_bound is not None and not (-5 <= lower_bound < 0):
        return False, 'AscendC bounded gate requires lower_bound in [-5, 0)'
    return True, None


@input_guard
def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor | None,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    lower_bound: float | None = None,
    use_graph: bool = False,
    **kwargs,
) -> torch.Tensor:
    from fla_npu.ops.ascendc import kda_gate_cumsum

    # the wheel uses head-major tensors and includes the 1/ln(2) scale.
    result = kda_gate_cumsum(
        g=g.transpose(1, 2).contiguous(),
        chunk_size=chunk_size,
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
    )
    return result.transpose(1, 2).contiguous()
