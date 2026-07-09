# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.utils.index import prepare_lens
from fla.ops.utils.pack import pack_sequence, unpack_sequence
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'padding_side', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-padding_side{}-{}".format(*test))
        for test in [
            (1, 63, 1, 30, 'left', torch.float),
            (2, 500, 4, 60, 'right', torch.float),
            (2, 1000, 5, 128, 'left', torch.float),
            (3, 1024, 6, 500, 'right', torch.float),
            (4, 2048, 8, 1024, 'left', torch.float),
        ]
    ],
)
def test_pack_sequence(
    B: int,
    T: int,
    H: int,
    D: int,
    padding_side: str,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    cu_seqlens = torch.cat(
        [torch.tensor([0])]+[torch.randint(0, T, (1,)).clamp(min=1) for _ in range(B)],
    ).cumsum(-1).to(device)
    lens = prepare_lens(cu_seqlens)

    if padding_side == 'left':
        ref = torch.cat([x[i, -length:] for i, length in enumerate(lens.tolist())], 0)
    else:
        ref = torch.cat([x[i, :length] for i, length in enumerate(lens.tolist())], 0)
    dy = torch.randn_like(ref)
    ref.backward(dy)
    ref_dx, x.grad = x.grad.clone(), None

    tri = pack_sequence(x, cu_seqlens, padding_side=padding_side)
    tri.backward(dy)
    tri_dx, x.grad = x.grad.clone(), None

    assert_close('y', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'padding_side', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-padding_side{}-{}".format(*test))
        for test in [
            (1, 63, 1, 30, 'left', torch.float),
            (2, 500, 4, 60, 'right', torch.float),
            (2, 1000, 5, 128, 'left', torch.float),
            (3, 1024, 6, 500, 'right', torch.float),
            (4, 2048, 8, 1024, 'left', torch.float),
        ]
    ],
)
def test_unpack_sequence(
    B: int,
    T: int,
    H: int,
    D: int,
    padding_side: str,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    cu_seqlens = torch.cat(
        [torch.tensor([0])]+[torch.randint(0, T, (1,)).clamp(min=1) for _ in range(B)],
    ).cumsum(-1).to(device)
    lens = prepare_lens(cu_seqlens)
    desired_shape = (B, lens.max().item() + torch.randint(0, 10, (1,)).item(), H, D)

    x = torch.randn(cu_seqlens[-1].item(), H, D, dtype=dtype).to(device).requires_grad_(True)
    ref = torch.zeros(desired_shape, device=device, dtype=dtype)
    dy = torch.randn_like(ref)
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)):
        length = eos - bos
        if padding_side == 'left':
            ref[i, -length:] = x[bos:eos]
        else:
            ref[i, :length] = x[bos:eos]
    ref.backward(dy)
    ref_dx, x.grad = x.grad.clone(), None

    tri = unpack_sequence(x, cu_seqlens, padding_side=padding_side, desired_shape=desired_shape)
    tri.backward(dy)
    tri_dx, x.grad = x.grad.clone(), None

    assert_close('y', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
