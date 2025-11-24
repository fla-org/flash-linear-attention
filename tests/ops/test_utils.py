
import os

import pytest
import torch

from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum, mean_pooling
from fla.ops.utils.index import prepare_lens
from fla.ops.utils.pack import pack_sequence, unpack_sequence
from fla.utils import assert_close, device


def reversed_cumsum(x, dim=-1):
    dtype = x.dtype
    x = x.float()
    c = x.cumsum(dim)
    y = x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c
    return y.to(dtype)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 30, torch.float),
            (2, 500, 4, 60, torch.float),
            (2, 1000, 5, 128, torch.float),
            (3, 1024, 6, 500, torch.float),
            (4, 2048, 8, 1024, torch.float),
        ]
    ],
)
def test_global_cumsum(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    s = torch.randn(B, T, H, dtype=dtype).to(device)
    ref = s.float().cumsum(1).to(dtype)
    tri = chunk_global_cumsum(s)
    assert_close('global_cumsum', ref, tri, 1e-3)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = s.float().cumsum(1).to(dtype)
    tri = chunk_global_cumsum(s)
    assert_close('global_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 60, [0, 15], torch.float),
            (3, 100, [0, 256, 500, 1000], torch.float),
            (4, 256, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 500, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (2, 1024, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_global_cumsum_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([
        s[:, start:end].float().cumsum(1)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
    ], 1).to(dtype)
    tri = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
    assert_close('global_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        s[:, start:end].float().cumsum(1)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
        ], 1).to(dtype)
    tri = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
    assert_close('global_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 30, torch.float),
            (2, 500, 4, 60, torch.float),
            (2, 1000, 5, 128, torch.float),
            (3, 1024, 6, 500, torch.float),
            (4, 2048, 8, 1024, torch.float),
        ]
    ],
)
def test_global_reversed_cumsum(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    s = torch.randn(B, T, H, dtype=dtype).to(device)
    ref = reversed_cumsum(s, dim=(1)).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True)
    assert_close('global_cumsum', ref, tri, 1e-3)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = reversed_cumsum(s, dim=(1)).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True)
    assert_close('global_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('H', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 60, [0, 15], torch.float),
            (3, 100, [0, 256, 500, 1000], torch.float),
            (4, 256, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 500, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (2, 1024, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_global_reversed_cumsum_varlen(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1)
                    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)], 1).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True, cu_seqlens=cu_seqlens)
    assert_close('global_reversed_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1)
                    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)], 1).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True, cu_seqlens=cu_seqlens)
    assert_close('global_reversed_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'C', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-C{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 16, 30, torch.float),
            (2, 500, 4, 32, 60, torch.float),
            (2, 1000, 5, 64, 128, torch.float),
            (3, 1024, 6, 64, 500, torch.float),
            (4, 2048, 8, 128, 1024, torch.float),
        ]
    ],
)
def test_local_cumsum(
    B: int,
    T: int,
    H: int,
    C: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    s = torch.randn(B, T, H, dtype=dtype).to(device)
    ref = torch.cat([s[:, i:i+C, :].float().cumsum(1) for i in range(0, T, C)], 1)
    tri = chunk_local_cumsum(s, chunk_size=C)
    assert_close('local_cumsum', ref, tri, 1e-3)

    s = torch.randn(B, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([s[:, i:i+C, :].float().cumsum(1) for i in range(0, T, C)], 1)
    tri = chunk_local_cumsum(s, chunk_size=C)
    assert_close('local_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('H', 'C', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-C{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 32, 60, [0, 15], torch.float),
            (3, 64, 100, [0, 256, 500, 1000], torch.float),
            (4, 64, 256, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 128, 500, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (2, 128, 1024, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_local_cumsum_varlen(
    H: int,
    C: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, cu_seqlens=cu_seqlens)
    assert_close('local_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, cu_seqlens=cu_seqlens)
    assert_close('local_cumsum', ref, tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'C', 'D', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-C{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 16, 30, torch.float),
            (2, 500, 4, 32, 60, torch.float),
            (2, 1000, 5, 64, 128, torch.float),
            (3, 1024, 6, 64, 500, torch.float),
            (4, 2048, 8, 128, 1024, torch.float),
        ]
    ],
)
def test_mean_pooling(
    B: int,
    T: int,
    H: int,
    C: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D, dtype=dtype).to(device)
    x.requires_grad = True
    ref = torch.cat([x[:, i:i+C, :].float().mean(1, True) for i in range(0, T, C)], 1).to(dtype)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = mean_pooling(x, chunk_size=C)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None

    assert_close('mean_pooling', ref, tri, 1e-3)
    assert_close('mean_pooling', ref_dx, tri_dx, 1e-3)


@pytest.mark.parametrize(
    ('H', 'C', 'D', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-C{}-D{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 32, 60, [0, 15], torch.float),
            (3, 64, 100, [0, 256, 500, 1000], torch.float),
            (4, 64, 256, [0, 15, 100, 300, 1200, 2000], torch.float),
            (4, 128, 500, [0, 1, 100, 300, 1200, 2048], torch.float16),
            (2, 128, 1024, [0, 200, 512, 1200, 2048], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_mean_pooling_varlen(
    H: int,
    C: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    x = torch.randn(1, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    ref = torch.cat([
        torch.cat([x[:, i:min(end, i+C), :].float().mean(1, True) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
    ], 1).to(dtype)
    do = torch.randn_like(ref)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None

    tri = mean_pooling(x, chunk_size=C, cu_seqlens=cu_seqlens)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None

    torch.testing.assert_close(ref, tri.to(ref.dtype), rtol=1.6e-2, atol=3e-5)
    torch.testing.assert_close(ref_dx, tri_dx.to(ref_dx.dtype), rtol=1.6e-2, atol=3e-5)


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
