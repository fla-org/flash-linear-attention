# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.rotary import RotaryEmbedding, rotary_embedding_ref
from fla.utils import IS_NPU, assert_close, device


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
            rotary._sin_cached[offset:offset+T],
        )
        for i, offset in enumerate(seqlen_offset.tolist())
    ]).to(dtype=dtype)
    ref_k = torch.cat([
        rotary_embedding_ref(
            k[i:i+1].float(),
            rotary._cos_cached[offset:offset+T],
            rotary._sin_cached[offset:offset+T],
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
        torch.tensor([T], dtype=torch.long),
    ], 0).to(device).sort()[0]
    rotary = RotaryEmbedding(D).to(device)

    tri_q, tri_k = rotary(q, k, cu_seqlens=cu_seqlens)
    tri_dq = torch.autograd.grad(tri_q.sum(), q, retain_graph=True)[0]
    tri_dk = torch.autograd.grad(tri_k.sum(), k, retain_graph=True)[0]

    ref_q = torch.cat([
        rotary_embedding_ref(
            q[0, start:end].float(),
            rotary._cos_cached[:end-start],
            rotary._sin_cached[:end-start],
        )
        for start, end in zip(cu_seqlens.tolist(), cu_seqlens[1:].tolist(), strict=False)
    ]).to(dtype=dtype).unsqueeze(0)
    ref_k = torch.cat([
        rotary_embedding_ref(
            k[0, start:end].float(),
            rotary._cos_cached[:end-start],
            rotary._sin_cached[:end-start],
        )
        for start, end in zip(cu_seqlens.tolist(), cu_seqlens[1:].tolist(), strict=False)
    ]).to(dtype=dtype).unsqueeze(0)
    ref_dq = torch.autograd.grad(ref_q.sum(), q, retain_graph=True)[0]
    ref_dk = torch.autograd.grad(ref_k.sum(), k, retain_graph=True)[0]

    assert_close(" q", ref_q, tri_q, ratio=1e-5)
    assert_close(" k", ref_k, tri_k, ratio=1e-5)
    assert_close("dq", ref_dq, tri_dq, ratio=1e-5)
    assert_close("dk", ref_dk, tri_dk, ratio=1e-5)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [2048, 4096])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rotary_left_padding(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    # Left-padding gives a NEGATIVE per-sequence seqlen_offset (real_len - T), as the
    # attention layer builds it for a padded batch. Existing tests cover only positive
    # offsets; this checks the real (in-range) tokens are rotated correctly under it.
    torch.manual_seed(42)
    pads = torch.arange(B, device=device) * (T // (2 * B))   # 0, p, 2p, ... left-pad per sequence
    seqlen_offset = -pads                                     # negative offset == left padding
    q = torch.randn(B, T, H, D).to(device).to(dtype=dtype)
    k = torch.randn(B, T, H, D).to(device).to(dtype=dtype)
    rotary = RotaryEmbedding(D).to(device)

    tri_q, tri_k = rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=T)

    for i, pad in enumerate(pads.tolist()):
        ref_q = rotary_embedding_ref(q[i:i+1, pad:].float(), rotary._cos_cached[:T-pad],
                                     rotary._sin_cached[:T-pad]).to(dtype=dtype)
        ref_k = rotary_embedding_ref(k[i:i+1, pad:].float(), rotary._cos_cached[:T-pad],
                                     rotary._sin_cached[:T-pad]).to(dtype=dtype)
        assert_close(f" q[{i}]", ref_q, tri_q[i:i+1, pad:], ratio=1e-5)
        assert_close(f" k[{i}]", ref_k, tri_k[i:i+1, pad:], ratio=1e-5)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [256, 2048])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("rotary_dim", [64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rotary_partial(B: int, T: int, H: int, D: int, rotary_dim: int, dtype: torch.dtype):
    # Partial rotary (rotary_dim < head_dim): only [:rotary_dim] is rotated and the
    # non-rotated tail [rotary_dim:] is carried over from the input. Previously untested.
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D).to(device).to(dtype=dtype)
    k = torch.randn(B, T, H, D).to(device).to(dtype=dtype)
    rotary = RotaryEmbedding(rotary_dim).to(device)

    tri_q, tri_k = rotary(q, k)

    ref_q = rotary_embedding_ref(q.float(), rotary._cos_cached[:T], rotary._sin_cached[:T]).to(dtype=dtype)
    ref_k = rotary_embedding_ref(k.float(), rotary._cos_cached[:T], rotary._sin_cached[:T]).to(dtype=dtype)
    assert_close(" q", ref_q, tri_q, ratio=1e-5)
    assert_close(" k", ref_k, tri_k, ratio=1e-5)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [2048])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rotary_left_padding_no_uninit_leak(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    # Regression guard for the uninitialized-output bug. Under left-padding the kernel
    # skips out-of-range rows, so an uninitialized output buffer leaks whatever was in
    # the reused allocation. We force the worst case by seeding the caching allocator's
    # free list with NaN-filled, output-shaped blocks (after a warm-up so the next
    # same-size allocation is the output buffer), then require the result to be finite.
    # Fails on `torch.empty_like`; passes once the buffer is initialized.
    torch.manual_seed(0)
    pads = torch.arange(B, device=device) * (T // (2 * B))
    seqlen_offset = -pads                                     # negative offset == left padding
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    rotary = RotaryEmbedding(D).to(device)

    rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=T)   # warm up cos/sin cache + kernel
    if IS_NPU:
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()
    junk = [torch.full_like(q, float("nan")) for _ in range(32)]   # poison the free list
    del junk

    tri_q, tri_k = rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=T)
    assert torch.isfinite(tri_q).all(), "rotary leaked uninitialized memory into q under left-padding"
    assert torch.isfinite(tri_k).all(), "rotary leaked uninitialized memory into k under left-padding"
