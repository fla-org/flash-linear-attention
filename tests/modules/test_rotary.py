# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from itertools import product

import pytest
import torch

from fla.modules.rotary import RotaryEmbedding
from fla.modules.rotary.ops import rotary_embedding_ref
from fla.utils import assert_close, device, device_torch_lib


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


@pytest.mark.parametrize(
    ("B", "T", "H", "G", "D", "dtype", "max_seqlen_delta"),
    [
        (*case, 0)
        for case in product([2], [2048, 4096], [4], [1, 4], [128, 256], [torch.float32, torch.bfloat16])
    ] + [(2, 8, 1, 1, 8, torch.float32, -1), (2, 8, 1, 1, 8, torch.float32, None)],
)
def test_rotary_with_offsets(B: int, T: int, H: int, G: int, D: int, dtype: torch.dtype, max_seqlen_delta: int | None):
    torch.manual_seed(42)
    test_device = "cpu" if max_seqlen_delta != 0 else device
    q = torch.randn(B, T, H, D).to(test_device).to(dtype=dtype).requires_grad_()
    k = torch.randn(B, T, H//G, D).to(test_device).to(dtype=dtype).requires_grad_()
    seqlen_offset = torch.randint(0, T//2, (B,)).to(test_device)
    rotary = RotaryEmbedding(D).to(test_device)

    if max_seqlen_delta is None:
        with pytest.raises(AssertionError, match="Tensor offsets require an initialized cache"):
            rotary(q, k, seqlen_offset=seqlen_offset)
        return

    max_seqlen = T + seqlen_offset.max().item() + max_seqlen_delta
    if max_seqlen_delta < 0:
        with pytest.raises(RuntimeError, match="Rotary cache is too short"):
            rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)
        return

    tri_q, tri_k = rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)
    tri_dq = torch.autograd.grad(tri_q.sum(), q, retain_graph=True)[0]
    tri_dk = torch.autograd.grad(tri_k.sum(), k, retain_graph=True)[0]

    ref_q = torch.cat([
        rotary_embedding_ref(q[i:i+1].float(), rotary._cos_cached[offset:offset+T], rotary._sin_cached[offset:offset+T])
        for i, offset in enumerate(seqlen_offset.tolist())
    ]).to(dtype=dtype)
    ref_k = torch.cat([
        rotary_embedding_ref(k[i:i+1].float(), rotary._cos_cached[offset:offset+T], rotary._sin_cached[offset:offset+T])
        for i, offset in enumerate(seqlen_offset.tolist())
    ]).to(dtype=dtype)
    ref_dq = torch.autograd.grad(ref_q.sum(), q, retain_graph=True)[0]
    ref_dk = torch.autograd.grad(ref_k.sum(), k, retain_graph=True)[0]

    assert_close(" q", ref_q, tri_q, ratio=1e-5)
    assert_close(" k", ref_k, tri_k, ratio=1e-5)
    assert_close("dq", ref_dq, tri_dq, ratio=1e-5)
    assert_close("dk", ref_dk, tri_dk, ratio=1e-5)


@pytest.mark.parametrize(
    ("N", "T", "H", "G", "D", "dtype", "lengths", "seqlen_offset", "interleaved", "max_seqlen_delta"),
    [
        pytest.param(*case, None, 0, False, 0, id=f"N{case[0]}-T{case[1]}-G{case[3]}-D{case[4]}-{case[5]}")
        for case in product([4], [2048, 4096], [4], [1, 4], [128, 256], [torch.float32, torch.bfloat16])
    ] + [
        pytest.param(
            len(lengths), sum(lengths), 2, 2, 16, dtype, lengths, offset, interleaved, 0,
            id=f"lengths{lengths}-offset{offset}-interleaved{interleaved}-{dtype}",
        )
        for lengths, offset, interleaved, dtype in product(
            [(3, 5), (0, 3, 0, 5, 0)], [0, 2, -1], [False, True],
            [torch.float32, torch.float16, torch.bfloat16],
        )
    ] + [
        pytest.param(
            2, 8, 2, 2, D, dtype, (3, 5), 2, True, 0,
            id=f"small-head-D{D}-offset2-interleavedTrue-{dtype}",
        )
        for D, dtype in product([4, 8], [torch.float32, torch.float16, torch.bfloat16])
    ] + [
        pytest.param(2, 65536, 2, 2, 4, torch.float16, (32767, 32769), 0, True, 0, id="small-head-long-packed"),
    ] + [
        pytest.param(
            2, 8, 2, 2, 16, torch.float32, (3, 5), offset, interleaved, -1,
            id=f"short-cache-offset{offset}-interleaved{interleaved}",
        )
        for offset, interleaved in product([0, 2, -1], [False, True])
    ] + [
        pytest.param(
            2, 8, 2, 2, 16, torch.float32, (3, 5), 2**32, False, -(2**32),
            id="short-cache-large-scalar-offset",
        ),
    ],
)
def test_rotary_varlen(
    N: int,
    T: int,
    H: int,
    G: int,
    D: int,
    dtype: torch.dtype,
    lengths: tuple[int, ...] | None,
    seqlen_offset: int,
    interleaved: bool,
    max_seqlen_delta: int,
):
    torch.manual_seed(42)
    test_device = "cpu" if max_seqlen_delta < 0 else device
    q = torch.randn(1, T, H, D).to(test_device).to(dtype=dtype).requires_grad_()
    k = torch.randn(1, T, H//G, D).to(test_device).to(dtype=dtype).requires_grad_()
    if lengths is None:
        cu_seqlens = torch.cat([
            torch.tensor([0], dtype=torch.long),
            torch.arange(1, T)[torch.randperm(T - 1)[:N-1]],
            torch.tensor([T], dtype=torch.long),
        ]).to(test_device).sort()[0]
    else:
        cu_seqlens = torch.tensor([0, *lengths], dtype=torch.int32, device=test_device).cumsum(0, dtype=torch.int32)
    rotary_dim = D if lengths is None else D // 2
    rotary = RotaryEmbedding(rotary_dim, interleaved=interleaved).to(test_device)
    max_seqlen = None if lengths is None else max(lengths) + seqlen_offset + max_seqlen_delta

    if max_seqlen_delta < 0:
        error = AssertionError if seqlen_offset > max_seqlen else RuntimeError
        with pytest.raises(error, match="Rotary cache is too short"):
            rotary(q, k, seqlen_offset=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return

    tri_q, tri_k = rotary(q, k, seqlen_offset=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    dq = torch.ones_like(q) if lengths is None else torch.randn_like(q)
    dk = torch.ones_like(k) if lengths is None else torch.randn_like(k)
    tri_dq, tri_dk = torch.autograd.grad((tri_q, tri_k), (q, k), (dq, dk))

    def reference(x):
        outputs = []
        for start, end in zip(cu_seqlens.tolist(), cu_seqlens[1:].tolist(), strict=False):
            if start == end:
                continue
            pad = min(max(-seqlen_offset, 0), end - start)
            prefix = x[0, start:start+pad].float()
            outputs.append(torch.cat((torch.zeros_like(prefix[..., :rotary_dim]), prefix[..., rotary_dim:]), dim=-1))
            offset = max(seqlen_offset, 0)
            outputs.append(rotary_embedding_ref(
                x[0, start+pad:end].float(),
                rotary._cos_cached[offset:offset+end-start-pad],
                rotary._sin_cached[offset:offset+end-start-pad],
                interleaved=interleaved,
            ))
        return torch.cat(outputs).to(dtype=dtype).unsqueeze(0)

    ref_q, ref_k = reference(q), reference(k)
    ref_dq, ref_dk = torch.autograd.grad((ref_q, ref_k), (q, k), (dq, dk))
    expected = (ref_q, ref_k, ref_dq, ref_dk)
    for name, ref, tri in zip(("q", "k", "dq", "dk"), expected, (tri_q, tri_k, tri_dq, tri_dk), strict=True):
        assert torch.isfinite(tri).all()
        assert_close(name, ref, tri, ratio=1e-5)

    if lengths is not None:
        tensor_offsets = torch.full((N,), seqlen_offset, dtype=torch.long, device=test_device)
        tri_q, tri_k = rotary(q, k, seqlen_offset=tensor_offsets, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        tri_dq, tri_dk = torch.autograd.grad((tri_q, tri_k), (q, k), (dq, dk))
        for name, ref, tri in zip(("q", "k", "dq", "dk"), expected, (tri_q, tri_k, tri_dq, tri_dk), strict=True):
            assert torch.isfinite(tri).all()
            assert_close(name, ref, tri, ratio=1e-5)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [2048, 4096])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rotary_left_padding(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    # negative offsets align valid tokens after each sequence's left padding.
    torch.manual_seed(42)
    pads = torch.arange(B, device=device) * (T // (2 * B))
    # negative offsets represent left padding.
    seqlen_offset = -pads
    q = torch.randn(B, T, H, D).to(device).to(dtype=dtype)
    k = torch.randn(B, T, H, D).to(device).to(dtype=dtype)
    rotary = RotaryEmbedding(D).to(device)

    tri_q, tri_k = rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=T)

    for i, pad in enumerate(pads.tolist()):
        ref_q = rotary_embedding_ref(
            x=q[i:i+1, pad:].float(),
            cos=rotary._cos_cached[:T-pad],
            sin=rotary._sin_cached[:T-pad],
        ).to(dtype=dtype)
        ref_k = rotary_embedding_ref(
            x=k[i:i+1, pad:].float(),
            cos=rotary._cos_cached[:T-pad],
            sin=rotary._sin_cached[:T-pad],
        ).to(dtype=dtype)
        assert_close(f" q[{i}]", ref_q, tri_q[i:i+1, pad:], ratio=1e-5)
        assert_close(f" k[{i}]", ref_k, tri_k[i:i+1, pad:], ratio=1e-5)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "rotary_dim", "dtype"),
    [
        (B, T, H, D, 64, dtype)
        for B, T, H, D, dtype in product([2], [256, 2048], [4], [128, 256], [torch.float32, torch.float16])
    ] + [(2, 8, 1, 128, 63, torch.float32), (2, 8, 1, 32, 64, torch.float32)],
)
def test_rotary_partial(B: int, T: int, H: int, D: int, rotary_dim: int, dtype: torch.dtype):
    # partial rotary must preserve the unrotated tail of each head.
    if rotary_dim % 2:
        with pytest.raises(AssertionError, match="Rotary dimension must be even"):
            RotaryEmbedding(rotary_dim)
        return

    torch.manual_seed(42)
    test_device = "cpu" if rotary_dim > D else device
    q = torch.randn(B, T, H, D).to(test_device).to(dtype=dtype)
    k = torch.randn(B, T, H, D).to(test_device).to(dtype=dtype)
    rotary = RotaryEmbedding(rotary_dim).to(test_device)

    if rotary_dim > D:
        with pytest.raises(AssertionError, match="Rotary dimension must not exceed head dimension"):
            rotary(q, k)
        return

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
    # padded output rows must remain finite when the allocator reuses NaN-filled memory.
    torch.manual_seed(0)
    pads = torch.arange(B, device=device) * (T // (2 * B))
    # negative offsets represent left padding.
    seqlen_offset = -pads
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    rotary = RotaryEmbedding(D).to(device)

    # warm up cos/sin cache + kernel
    rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=T)
    device_torch_lib.synchronize()
    # poison the free list
    junk = [torch.full_like(q, float("nan")) for _ in range(32)]
    del junk

    tri_q, tri_k = rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=T)
    assert torch.isfinite(tri_q).all(), "rotary leaked uninitialized memory into q under left-padding"
    assert torch.isfinite(tri_k).all(), "rotary leaked uninitialized memory into k under left-padding"
