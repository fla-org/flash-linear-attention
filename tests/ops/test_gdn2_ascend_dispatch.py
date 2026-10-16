# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend dispatch and leaf-kernel coverage for GDN-2."""

from unittest.mock import Mock

import pytest
import torch

from fla.ops.gdn2.wy_fast import recompute_w_u_fwd_gdn2
from fla.utils import IS_NPU, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NPU, reason="Ascend NPU required")

_BT = 64


def _sequence_spans(B: int, T: int, lengths: tuple[int, ...] | None) -> list[tuple[int, int, int]]:
    if lengths is None:
        return [(i_b, 0, T) for i_b in range(B)]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return [(0, offsets[i], offsets[i + 1]) for i in range(len(lengths))]


def _make_wy_inputs(
    *,
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    spans: list[tuple[int, int, int]],
    has_q: bool,
) -> tuple[torch.Tensor | None, ...]:
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.25 if has_q else None
    k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.25
    v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.25
    b = torch.rand(B, T, H, K, dtype=dtype, device=device) * 0.5
    w_gate = torch.rand(B, T, H, V, dtype=dtype, device=device) * 0.5
    gk = torch.empty(B, T, H, K, dtype=torch.float32, device=device)
    A = torch.zeros(B, T, H, _BT, dtype=torch.float32, device=device)

    for i_b, seq_start, seq_end in spans:
        for chunk_start in range(seq_start, seq_end, _BT):
            chunk_end = min(chunk_start + _BT, seq_end)
            chunk_len = chunk_end - chunk_start
            increments = -(torch.rand(chunk_len, H, K, device=device) * 0.02 + 0.01)
            gk[i_b, chunk_start:chunk_end] = increments.cumsum(0)

            eye = torch.eye(_BT, dtype=torch.float32, device=device).expand(H, -1, -1)
            noise = torch.randn(H, _BT, _BT, dtype=torch.float32, device=device) * 0.01
            block = eye + torch.tril(noise, diagonal=-1)
            A[i_b, chunk_start:chunk_end] = block[:, :chunk_len].permute(1, 0, 2)

    return q, k, v, b, w_gate, A.to(dtype), gk.to(dtype)


def _wy_reference(
    *,
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    spans: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    kg = torch.empty_like(k)
    gk_exp = torch.exp2(gk.float())
    qg = (q.float() * gk_exp).to(q.dtype) if q is not None else None

    for i_b, seq_start, seq_end in spans:
        for chunk_start in range(seq_start, seq_end, _BT):
            chunk_end = min(chunk_start + _BT, seq_end)
            chunk_len = chunk_end - chunk_start
            A_chunk = A[i_b, chunk_start:chunk_end].permute(1, 0, 2).float()[..., :chunk_len]

            vb = (v[i_b, chunk_start:chunk_end].float() *
                  w_gate[i_b, chunk_start:chunk_end].float()).to(v.dtype)
            u_chunk = torch.matmul(A_chunk, vb.permute(1, 0, 2).float())
            u[i_b, chunk_start:chunk_end] = u_chunk.permute(1, 0, 2).to(v.dtype)

            kb = (k[i_b, chunk_start:chunk_end].float() *
                  b[i_b, chunk_start:chunk_end].float() *
                  gk_exp[i_b, chunk_start:chunk_end]).to(k.dtype)
            w_chunk = torch.matmul(A_chunk, kb.permute(1, 0, 2).float())
            w[i_b, chunk_start:chunk_end] = w_chunk.permute(1, 0, 2).to(k.dtype)

            g_last = gk[i_b, chunk_end - 1].float()
            decay = torch.exp2(g_last[None, :, :] - gk[i_b, chunk_start:chunk_end].float())
            kg[i_b, chunk_start:chunk_end] = (k[i_b, chunk_start:chunk_end].float() * decay).to(k.dtype)

    return w, u, qg, kg


@pytest.mark.parametrize(
    ("layout", "dtype", "K", "V", "has_q"),
    [
        pytest.param("dense", torch.float16, 48, 32, False, id="dense-fp16-k48-v32-no-q"),
        pytest.param("dense", torch.float32, 32, 48, True, id="dense-fp32-k32-v48-q"),
        pytest.param("dense", torch.bfloat16, 256, 256, True, id="dense-bf16-k256-v256-q"),
        pytest.param("varlen", torch.bfloat16, 48, 32, False, id="varlen-bf16-k48-v32-no-q"),
        pytest.param("varlen", torch.float16, 32, 48, True, id="varlen-fp16-k32-v48-q"),
    ],
)
def test_recompute_w_u_fwd_gdn2_npu_leaf(layout, dtype, K, V, has_q, monkeypatch):
    """The Ascend WY leaf must match its channel-wise Torch definition."""
    from fla.ops.gdn2.backends.triton_ascend import wy_fast

    spy = Mock(wraps=wy_fast.recompute_w_u_fwd_gdn2_npu)
    monkeypatch.setattr(wy_fast, "recompute_w_u_fwd_gdn2_npu", spy)

    lengths = None if layout == "dense" else (15, 65, 65)
    B = 2 if layout == "dense" and K < 256 else 1
    H = 2 if K < 256 else 1
    T = 65 if lengths is None else sum(lengths)
    spans = _sequence_spans(B, T, lengths)
    q, k, v, b, w_gate, A, gk = _make_wy_inputs(
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        dtype=dtype,
        spans=spans,
        has_q=has_q,
    )
    cu_seqlens = None
    if lengths is not None:
        cu_seqlens = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), device=device)

    expected = _wy_reference(q=q, k=k, v=v, b=b, w_gate=w_gate, A=A, gk=gk, spans=spans)
    actual = recompute_w_u_fwd_gdn2(
        k=k,
        v=v,
        b=b,
        w_gate=w_gate,
        A=A,
        q=q,
        gk=gk,
        cu_seqlens=cu_seqlens,
    )

    spy.assert_called_once()
    for name, ref, tri in zip(("w", "u", "qg", "kg"), expected, actual):
        if ref is None:
            assert tri is None
        else:
            assert tri is not None
            assert_close(name, ref, tri, 0.005)
