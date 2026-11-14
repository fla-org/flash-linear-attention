# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules.token_shift import token_shift, token_shift_ref
from fla.utils import assert_close, device

test_b_list = [4]
test_t_list = [512, 4100, 8192]
test_h_list = [2560, 4096]
test_cu_seqlens_list = [
    None,
    [0, 4, 7, 40, 128],
    [0, 10, 20, 64],
    [0, 32],
    [0, 1, 3, 4],
]
test_dtype_list = [torch.float]


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('cu_seqlens_val', test_cu_seqlens_list)
@pytest.mark.parametrize('dtype', test_dtype_list)
def test_token_shift(B, T, H, cu_seqlens_val, dtype):
    if cu_seqlens_val is not None:
        B = 1
        T = cu_seqlens_val[-1]
        cu_seqlens_tensor = torch.tensor(cu_seqlens_val, dtype=torch.int32, device=device)
    else:
        cu_seqlens_tensor = None

    torch.manual_seed(42)

    x = torch.randn(B, T, H, device=device).to(dtype).requires_grad_(True)
    dy = torch.randn_like(x)

    ref = token_shift_ref(x, cu_seqlens_tensor)
    tri = token_shift(x, cu_seqlens_tensor)

    ref.backward(dy)
    ref_dx, x.grad = x.grad, None

    tri.backward(dy)
    tri_dx, x.grad = x.grad, None

    assert_close(' x', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D'),
    [
        pytest.param(*test, id="B{}-T{}-D{}".format(*test))
        for test in [
            (2, 2, 64),
            (2, 512, 1024),
            (1, 8192, 1024),
        ]
    ],
)
def test_token_shift_cache(B, T, D):
    torch.manual_seed(42)

    x = torch.randn(B, T, D, device=device, dtype=torch.float, requires_grad=True)
    dy = torch.randn_like(x)

    ref = token_shift_ref(x=x)
    ref.backward(dy)
    ref_dx, x.grad = x.grad.clone(), None

    split = T // 2
    tri0, cache0 = token_shift(x=x[:, :split].contiguous(), output_cache=True)
    assert_close('cache0', x[:, split - 1], cache0, 1e-3)

    tri1, cache1 = token_shift(x=x[:, split:].contiguous(), cache=cache0, output_cache=True)
    assert_close('cache1', x[:, -1], cache1, 1e-3)

    tri = torch.cat((tri0, tri1), dim=1)
    tri.backward(dy)
    tri_dx, x.grad = x.grad.clone(), None

    assert_close('o', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)


@pytest.mark.parametrize(
    ('cu_seqlens', 'cache'),
    [
        pytest.param([0, 2, 3], [10., 20.], id='mixed-with-cache'),
        pytest.param([0, 1, 2, 3], None, id='singletons-without-cache'),
    ],
)
def test_token_shift_cache_varlen(cu_seqlens, cache):
    x = torch.tensor([[[1.], [2.], [3.]]], device=device)
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    if cache is not None:
        cache = torch.tensor(cache, device=device).unsqueeze(-1)

    ref = token_shift_ref(x=x, cu_seqlens=cu_seqlens)
    if cache is not None:
        ref[0, cu_seqlens[:-1]] += cache
    ref_cache = x[0, cu_seqlens[1:] - 1]

    tri, tri_cache = token_shift(x=x, cu_seqlens=cu_seqlens, cache=cache, output_cache=True)

    assert_close('o', ref, tri, 1e-3)
    assert_close('cache', ref_cache, tri_cache, 1e-3)


def _packed_cu_seqlens(total_tokens: int, doc_len: int) -> torch.Tensor:
    n_docs = (total_tokens + doc_len - 1) // doc_len
    lens = torch.full((n_docs,), doc_len, device=device, dtype=torch.long)
    lens[-1] -= n_docs * doc_len - total_tokens
    return F.pad(lens.cumsum(0), (1, 0))


def test_token_shift_varlen_uses_max_doc_len_for_short_kernel():
    total_tokens, hidden_size = 262_144, 8
    torch.manual_seed(42)

    cu_seqlens = _packed_cu_seqlens(total_tokens, doc_len=752)
    x = torch.randn(1, total_tokens, hidden_size, device=device, dtype=torch.bfloat16)

    ref = token_shift_ref(x, cu_seqlens)
    tri = token_shift(x, cu_seqlens)

    assert_close(" x", ref, tri, 1e-3)


def test_token_shift_varlen_long_kernel_matches_reference():
    cu_seqlens = torch.tensor([0, 4097, 8200, 12304], dtype=torch.long, device=device)
    torch.manual_seed(42)

    x = torch.randn(1, cu_seqlens[-1].item(), 32, device=device, dtype=torch.float, requires_grad=True)
    dy = torch.randn_like(x)

    ref = token_shift_ref(x, cu_seqlens)
    tri = token_shift(x, cu_seqlens)

    ref.backward(dy)
    ref_dx, x.grad = x.grad.clone(), None

    tri.backward(dy)
    tri_dx, x.grad = x.grad.clone(), None

    assert_close(" x", ref, tri, 1e-3)
    assert_close("dx", ref_dx, tri_dx, 1e-3)
