# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import triton
import triton.language as tl

from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_TMA_SUPPORTED, assert_close, device

pytestmark = pytest.mark.skipif(
    not IS_TMA_SUPPORTED,
    reason='TMA is unsupported on this arch, or FLA_USE_TMA is not set to 1',
)


@triton.jit(do_not_specialize=['T'])
def tma_transposed_tile_kernel(
    V,
    O,
    T,
    Vd: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_v, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    # V is [T, H, Vd]. A descriptor cannot describe the [Vd, T] view directly, because that would
    # need an innermost stride of H*Vd; describe the natural layout, where Vd is the stride-1 axis.
    d_v = make_tensor_descriptor(V + i_h * Vd, [T, Vd], [H * Vd, 1], [BT, BV])
    # O is [H, Vd, T], already row-major, so the transpose lands in registers rather than in the
    # addressing. Both axes are ragged here: the descriptor zero-fills OOB loads and drops OOB
    # stores, which is what replaces the explicit masks of the pointer path.
    d_o = make_tensor_descriptor(O + i_h * Vd * T, [Vd, T], [T, 1], [BV, BT])
    # [BT, BV] -> [BV, BT]
    b_v = tl.trans(d_v.load([i_t * BT, i_v * BV]))
    d_o.store([i_v * BV, i_t * BT], (b_v * 2.0).to(d_o.dtype))


def tma_transposed_tile(v: torch.Tensor) -> torch.Tensor:
    T, H, Vd = v.shape
    o = torch.zeros((H, Vd, T), device=v.device, dtype=v.dtype)
    BT, BV = 64, min(64, Vd)
    grid = (triton.cdiv(T, BT), triton.cdiv(Vd, BV), H)
    tma_transposed_tile_kernel[grid](v, o, T, Vd=Vd, H=H, BT=BT, BV=BV)
    return o


def naive_transposed_tile(v: torch.Tensor) -> torch.Tensor:
    return v.permute(1, 2, 0).contiguous() * 2.0


@pytest.mark.parametrize(
    ('T', 'H', 'Vd', 'dtype'),
    [
        pytest.param(*test, id="T{}-H{}-Vd{}-{}".format(*test))
        for test in [
            # T is not a multiple of BT=64 in the first three, so the ragged tail is exercised.
            (520, 3, 64, torch.float16),
            # Vd=96 against BV=64 makes the value axis ragged as well.
            (520, 3, 96, torch.float16),
            (520, 2, 64, torch.float32),
            (1024, 4, 128, torch.bfloat16),
        ]
    ],
)
def test_tma_transposed_tile(T: int, H: int, Vd: int, dtype: torch.dtype):
    """A 2-D tile consumed transposed, as V is when it is the RHS of a matmul.

    Every shape here satisfies the descriptor alignment rules, which are not checked by the
    compiler and fault at launch when violated: both descriptor bases (`i_h * Vd` into V and
    `i_h * Vd * T` into O) and both leading strides (`H * Vd` and `T`) must be multiples of
    16 bytes, and the innermost axis must be stride-1.
    """
    torch.manual_seed(42)
    itemsize = torch.empty(0, dtype=dtype).element_size()
    assert (Vd * itemsize) % 16 == 0 and (H * Vd * itemsize) % 16 == 0 and (T * itemsize) % 16 == 0

    v = torch.randn(T, H, Vd, dtype=dtype, device=device)

    ref = naive_transposed_tile(v)
    tri = tma_transposed_tile(v)

    assert_close('o', ref, tri, 0)
