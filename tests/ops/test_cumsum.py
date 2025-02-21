# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.utils.cumsum import chunk_global_cumsum, chunk_local_cumsum


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


def rev_cumsum(s, dim=-1):
    return torch.flip(torch.cumsum(torch.flip(s, dims=[dim]), dim), dims=[dim])


def cumsum_local_reference(s, reverse=False, head_first=False, chunk_size=128):
    o = torch.zeros_like(s)
    T = s.size(2) if head_first else s.size(1)
    fn = torch.cumsum if not reverse else rev_cumsum
    for i in range(0, T, chunk_size):
        if head_first:
            s_chunk = s[:, :, i:i+chunk_size]
            o[:, :, i:i+chunk_size] = fn(s_chunk.float(), dim=2).to(o)
        else:
            s_chunk = s[:, i:i+chunk_size]
            o[:, i:i+chunk_size] = fn(s_chunk.float(), dim=1).to(o)

    return o


def cumsum_global_reference(s, reverse=False, head_first=False):
    fn = torch.cumsum if not reverse else rev_cumsum
    return fn(s.float(), dim=2).to(s) if head_first else fn(s.float(), dim=1).to(s)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [300])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [32, 64, 100])
@pytest.mark.parametrize("chunk_size", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsum_local_vector(B, T, H, D, dtype, head_first, reverse, chunk_size):
    if head_first:
        s = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    else:
        s = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_()
    ref = cumsum_local_reference(s, reverse=reverse, head_first=head_first, chunk_size=chunk_size)
    tri = chunk_local_cumsum(s, reverse=reverse, head_first=head_first, chunk_size=chunk_size)
    assert_close("local cumsum vector", ref, tri, 0.001 if dtype == torch.float else 0.003)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [300])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("chunk_size", [32, 64])
def test_cumsum_local_scalar(B, T, H, dtype, head_first, reverse, chunk_size):
    if head_first:
        s = torch.randn((B, H, T), dtype=dtype, device='cuda').requires_grad_()
    else:
        s = torch.randn((B, T, H), dtype=dtype, device='cuda').requires_grad_()
    ref = cumsum_local_reference(s, reverse=reverse, head_first=head_first, chunk_size=chunk_size)
    tri = chunk_local_cumsum(s, reverse=reverse, head_first=head_first, chunk_size=chunk_size)
    assert_close("local cumsum scalar", ref, tri, 0.001 if dtype == torch.float else 0.003)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [32, 64, 100])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("reverse", [True, False])
def test_cumsum_global_vector(B, T, H, D, dtype, head_first, reverse):
    if head_first:
        s = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    else:
        s = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_()
    ref = cumsum_global_reference(s, reverse=reverse, head_first=head_first)
    tri = chunk_global_cumsum(s, reverse=reverse, head_first=head_first)
    assert_close("global cumsum vector", ref, tri, 0.001 if dtype == torch.float else 0.003)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("reverse", [True, False])
def test_cumsum_global_scalar(B, T, H, dtype, head_first, reverse):
    if head_first:
        s = torch.randn((B, H, T), dtype=dtype, device='cuda').requires_grad_()
    else:
        s = torch.randn((B, T, H), dtype=dtype, device='cuda').requires_grad_()
    ref = cumsum_global_reference(s, reverse=reverse, head_first=head_first)
    tri = chunk_global_cumsum(s, reverse=reverse, head_first=head_first)
    assert_close("global cumsum scalar", ref, tri, 0.001 if dtype == torch.float else 0.003)
