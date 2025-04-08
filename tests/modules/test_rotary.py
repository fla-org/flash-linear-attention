# -*- coding: utf-8 -*-

import pytest
import torch

from fla.modules.rotary import RotaryEmbedding, rotary_embedding_ref
from fla.ops.utils.testing import assert_close
from fla.utils import device


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [2048, 4096])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_rotary(B: int, T: int, H: int, G: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D).to(device).to(dtype=dtype).requires_grad_()
    k = torch.randn(B, T, H//G, D).to(device).to(dtype=dtype).requires_grad_()
    rotary = RotaryEmbedding(D).to(device)

    tri_q, tri_k = rotary(q, k)
    tri_dq = torch.autograd.grad(tri_q.sum(), q, retain_graph=True)[0]
    tri_dk = torch.autograd.grad(tri_k.sum(), k, retain_graph=True)[0]

    ref_q = rotary_embedding_ref(q.float(), rotary._cos_cached, rotary._sin_cached).to(dtype=dtype)
    ref_k = rotary_embedding_ref(k.float(), rotary._cos_cached, rotary._sin_cached).to(dtype=dtype)
    ref_dq = torch.autograd.grad(ref_q.sum(), q, retain_graph=True)[0]
    ref_dk = torch.autograd.grad(ref_k.sum(), k, retain_graph=True)[0]

    assert_close(" q", ref_q, tri_q, ratio=1e-5)
    assert_close(" k", ref_k, tri_k, ratio=1e-5)
    assert_close("dq", ref_dq, tri_dq, ratio=1e-5)
    assert_close("dk", ref_dk, tri_dk, ratio=1e-5)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [2048, 4096])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_with_offsets(B: int, T: int, H: int, G: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D).to(device).to(dtype=dtype).requires_grad_()
    k = torch.randn(B, T, H//G, D).to(device).to(dtype=dtype).requires_grad_()
    seqlen_offset = torch.randint(0, T//2, (B,)).to(device)
    max_seqlen = T + seqlen_offset.max().item()
    rotary = RotaryEmbedding(D).to(device)

    tri_q, tri_k = rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)
    tri_dq = torch.autograd.grad(tri_q.sum(), q, retain_graph=True)[0]
    tri_dk = torch.autograd.grad(tri_k.sum(), k, retain_graph=True)[0]

    ref_q = torch.cat([
        rotary_embedding_ref(
            q[i:i+1].float(),
            rotary._cos_cached[offset:offset+T],
            rotary._sin_cached[offset:offset+T]
        )
        for i, offset in enumerate(seqlen_offset.tolist())
    ]).to(dtype=dtype)
    ref_k = torch.cat([
        rotary_embedding_ref(
            k[i:i+1].float(),
            rotary._cos_cached[offset:offset+T],
            rotary._sin_cached[offset:offset+T]
        )
        for i, offset in enumerate(seqlen_offset.tolist())
    ]).to(dtype=dtype)
    ref_dq = torch.autograd.grad(ref_q.sum(), q, retain_graph=True)[0]
    ref_dk = torch.autograd.grad(ref_k.sum(), k, retain_graph=True)[0]

    assert_close(" q", ref_q, tri_q, ratio=1e-5)
    assert_close(" k", ref_k, tri_k, ratio=1e-5)
    assert_close("dq", ref_dq, tri_dq, ratio=1e-5)
    assert_close("dk", ref_dk, tri_dk, ratio=1e-5)


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [2048, 4096])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("G", [1, 4])
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_varlen(N: int, T: int, H: int, G: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    q = torch.randn(1, T, H, D).to(device).to(dtype=dtype).requires_grad_()
    k = torch.randn(1, T, H//G, D).to(device).to(dtype=dtype).requires_grad_()
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(1, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    rotary = RotaryEmbedding(D).to(device)

    tri_q, tri_k = rotary(q, k, cu_seqlens=cu_seqlens)
    tri_dq = torch.autograd.grad(tri_q.sum(), q, retain_graph=True)[0]
    tri_dk = torch.autograd.grad(tri_k.sum(), k, retain_graph=True)[0]

    ref_q = torch.cat([
        rotary_embedding_ref(
            q[0, start:end].float(),
            rotary._cos_cached[:end-start],
            rotary._sin_cached[:end-start]
        )
        for start, end in zip(cu_seqlens.tolist(), cu_seqlens[1:].tolist())
    ]).to(dtype=dtype).unsqueeze(0)
    ref_k = torch.cat([
        rotary_embedding_ref(
            k[0, start:end].float(),
            rotary._cos_cached[:end-start],
            rotary._sin_cached[:end-start]
        )
        for start, end in zip(cu_seqlens.tolist(), cu_seqlens[1:].tolist())
    ]).to(dtype=dtype).unsqueeze(0)
    ref_dq = torch.autograd.grad(ref_q.sum(), q, retain_graph=True)[0]
    ref_dk = torch.autograd.grad(ref_k.sum(), k, retain_graph=True)[0]

    assert_close(" q", ref_q, tri_q, ratio=1e-5)
    assert_close(" k", ref_k, tri_k, ratio=1e-5)
    assert_close("dq", ref_dq, tri_dq, ratio=1e-5)
    assert_close("dk", ref_dk, tri_dk, ratio=1e-5)
