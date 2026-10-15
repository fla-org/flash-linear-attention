# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
from importlib import import_module

import pytest
import torch

from fla.ops.utils import chunk_global_cumsum, chunk_local_cumsum
from fla.utils import IS_NPU, assert_close, device


@pytest.mark.skipif(not IS_NPU, reason="Ascend task-loop sentinel handling")
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_local_cumsum_graph_sentinels(monkeypatch, vector, reverse):
    """Sentinels between valid tasks must not read state or terminate the task loop."""
    module = import_module("fla.ops.utils.backends.triton_ascend.cumsum")
    monkeypatch.setattr(module, "get_multiprocessor_count", lambda: 1)
    torch.manual_seed(42)
    shape = (1, 128, 2, 32) if vector else (1, 128, 2)
    s = torch.randn(shape, device=device)
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)
    indices = torch.tensor([[-1, 0], [0, 0], [-1, 0], [1, 0], [-1, 0]], dtype=torch.int32, device=device)
    ref = torch.cat([
        reversed_cumsum(part, dim=1) if reverse else part.cumsum(1)
        for part in s.split(64, dim=1)
    ], dim=1)
    actual = chunk_local_cumsum(
        s,
        chunk_size=64,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        chunk_indices=indices,
        use_graph=True,
    )
    assert_close("local_cumsum_sentinels", ref, actual, 1e-3)


def reversed_cumsum(x, dim=-1):
    dtype = x.dtype
    x = x.float()
    return x.flip(dim).cumsum(dim).flip(dim).to(dtype)


@pytest.mark.parametrize(('B', 'H'), [(1, 1), (2050, 32)], ids=['B1-H1', 'B2050-H32'])
def test_scalar_reversed_cumsum_preserves_small_suffix(B: int, H: int):
    prefix = torch.full((32,), 1e8, dtype=torch.float, device=device)
    suffix = torch.ones(32, dtype=torch.float, device=device)
    s = torch.cat((prefix, suffix)).reshape(1, 64, 1).expand(B, 64, H).contiguous()
    expected_suffix = torch.arange(32, 0, -1, dtype=torch.float, device=device).view(1, 32, 1).expand(B, 32, H)

    local = chunk_local_cumsum(s, chunk_size=64, reverse=True)
    global_ = chunk_global_cumsum(s, reverse=True)

    torch.testing.assert_close(local[:, 32:], expected_suffix, rtol=0, atol=0)
    torch.testing.assert_close(global_[:, 32:], expected_suffix, rtol=0, atol=0)


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
            # exceed the y-axis grid limit with multiple feature tiles
            (32, 33, range(0, 32801, 16), torch.float),
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
    cu_seqlens: list[int] | range,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    T = cu_seqlens[-1]
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    s = torch.randn(1, T, H, dtype=dtype).to(device)
    ref = torch.cat([
        s[:, start:end].float().cumsum(1)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
    ], 1).to(dtype)
    tri = chunk_global_cumsum(s, cu_seqlens=cu_seqlens)
    assert_close('global_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        s[:, start:end].float().cumsum(1)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
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
                    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)], 1).to(dtype)
    tri = chunk_global_cumsum(s, reverse=True, cu_seqlens=cu_seqlens)
    assert_close('global_reversed_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([reversed_cumsum(s[:, start:end], 1)
                    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)], 1).to(dtype)
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
            (2, 16, 60, [0, 15], torch.float),
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
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, cu_seqlens=cu_seqlens)
    assert_close('local_cumsum', ref, tri, 1e-3)

    s = torch.randn(1, T, H, D, dtype=dtype).to(device)
    ref = torch.cat([
        torch.cat([s[:, i:min(end, i+C), :].float().cumsum(1) for i in range(start, end, C)], 1)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)
    ], 1)
    tri = chunk_local_cumsum(s, chunk_size=C, cu_seqlens=cu_seqlens)
    assert_close('local_cumsum', ref, tri, 1e-3)


@pytest.fixture
def ascendc_native(monkeypatch):
    if not IS_NPU or os.getenv('FLA_ASCENDC', '0') == '0':
        pytest.skip('Requires an Ascend950 native wheel and FLA_ASCENDC=1 at process startup')
    import fla.utils.ascendc as native

    assert native.is_available(), 'The requested AscendC test must use a native wheel'
    assert torch.npu.get_device_name().startswith('Ascend950')
    assert os.getenv('FLA_DISABLE_BACKEND_DISPATCH', '0') != '1'
    calls = []
    implementation = native.chunk_local_cumsum

    def record(*args, **kwargs):
        calls.append(kwargs['g'].shape)
        return implementation(*args, **kwargs)

    monkeypatch.setattr(native, 'chunk_local_cumsum', record)
    return calls


@pytest.mark.parametrize('shape', [(1, 1, 1), (2, 63, 3), (2, 64, 4), (2, 65, 3),
                                   (1, 127, 5), (1, 128, 5), (2, 129, 3), (1, 257, 7), (2, 1025, 17)])
@pytest.mark.parametrize('chunk_size', [64, 128])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('scale', [None, 0.0, -0.5, 1.3])
@pytest.mark.parametrize('strided', [False, True])
def test_ascendc_local_cumsum(ascendc_native, shape, chunk_size, dtype, scale, strided):
    torch.manual_seed(42)
    B, T, H = shape
    x = torch.randn(B, T, H * (2 if strided else 1), dtype=dtype, device=device)
    if strided:
        x = x[..., ::2]
    cpu = x.cpu().double()
    ref = torch.cat([part.cumsum(1) for part in cpu.split(chunk_size, dim=1)], dim=1)
    ref = (ref * (1.0 if scale is None else scale)).float()
    actual = chunk_local_cumsum(x, chunk_size=chunk_size, scale=scale)
    torch.npu.synchronize()
    assert ascendc_native == [torch.Size((B, H, T))], 'The public entry must execute the native adapter exactly once'
    assert actual.shape == x.shape and actual.dtype == torch.float32 and actual.device == x.device
    assert torch.isfinite(actual).all()
    actual = actual.cpu()
    assert_close('ascendc_local_cumsum', ref, actual, 1e-3)
    error_ratio = (ref - actual).square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-8)
    # enforce the existing budget even when FLA_CI_ENV enables warnings
    assert error_ratio < 1e-3


@pytest.mark.parametrize('case', ['vector', 'reverse', 'varlen', 'output_dtype', 'chunk32'])
def test_ascendc_local_cumsum_fallback(ascendc_native, case):
    torch.manual_seed(42)
    shape = (1, 129, 3, 5) if case == 'vector' else (1, 129, 3)
    x = torch.randn(shape, device=device)
    chunk_size = 32 if case == 'chunk32' else 64
    kwargs = {'reverse': case == 'reverse'}
    boundaries = [0, 33, 129] if case == 'varlen' else [0, 129]
    if case == 'varlen':
        kwargs['cu_seqlens'] = torch.tensor(boundaries, device=device, dtype=torch.int32)
    if case == 'output_dtype':
        kwargs['output_dtype'] = torch.float16
    cpu = x.cpu().double()
    parts = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for part in cpu[:, start:end].split(chunk_size, dim=1):
            parts.append(part.flip(1).cumsum(1).flip(1) if case == 'reverse' else part.cumsum(1))
    ref = torch.cat(parts, dim=1).to(kwargs.get('output_dtype', torch.float32))
    actual = chunk_local_cumsum(x, chunk_size=chunk_size, **kwargs)
    torch.npu.synchronize()
    assert ascendc_native == [], 'Unsupported calls must retain the existing backend route'
    assert torch.isfinite(actual).all()
    assert_close('ascendc_fallback', ref.float(), actual.cpu().float(), 1e-3)


def test_ascendc_local_cumsum_stream(ascendc_native):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        x = torch.arange(129, device=device, dtype=torch.float32).reshape(1, 129, 1)
        x = x * 0.5 + 1
        actual = chunk_local_cumsum(x, chunk_size=64)
        consumed = actual + 2
    stream.synchronize()
    ref = torch.cat([part.cumsum(1) for part in x.cpu().split(64, dim=1)], dim=1) + 2
    assert len(ascendc_native) == 1
    torch.testing.assert_close(consumed.cpu(), ref, rtol=0, atol=0)
