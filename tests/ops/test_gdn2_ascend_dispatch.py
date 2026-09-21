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

from fla.ops.gdn2 import chunk_gdn2
from fla.utils import IS_NPU, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NPU, reason="Ascend NPU required")

_BT = 64
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
def test_recompute_w_u_fwd_gdn2_npu_leaf(layout, dtype, K, V, has_q):
    """The Ascend WY leaf must match its channel-wise Torch definition."""
    from fla.ops.gdn2.backends.triton_ascend.wy_fast import recompute_w_u_fwd_gdn2_npu

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
    actual = recompute_w_u_fwd_gdn2_npu(
        k=k,
        v=v,
        b=b,
        w_gate=w_gate,
        A=A,
        q=q,
        gk=gk,
        cu_seqlens=cu_seqlens,
    )

    for name, ref, tri in zip(("w", "u", "qg", "kg"), expected, actual):
        if ref is None:
            assert tri is None
        else:
            assert tri is not None
            assert_close(name, ref, tri, 0.005)


def _route_inputs(dtype: torch.dtype, T: int = _BT) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    shape_k = (1, T, 1, 32)
    shape_v = (1, T, 1, 32)
    tensors = {
        "q": torch.randn(shape_k, dtype=dtype, device=device) * 0.1,
        "k": torch.randn(shape_k, dtype=dtype, device=device) * 0.1,
        "v": torch.randn(shape_v, dtype=dtype, device=device) * 0.1,
        "g": torch.empty(shape_k, dtype=dtype, device=device).uniform_(-0.05, -0.01),
        "b": torch.rand(shape_k, dtype=dtype, device=device) * 0.05,
        "w": torch.rand(shape_v, dtype=dtype, device=device) * 0.5,
    }
    for tensor in tensors.values():
        tensor.requires_grad_()
    return tensors


@pytest.mark.parametrize("layout", ("dense", "varlen"))
@pytest.mark.parametrize("disable_recompute", [False, True], ids=["recompute", "saved-forward"])
def test_gdn2_ascend_recompute_and_bwd_intra_routes(monkeypatch, layout, disable_recompute):
    """Training must route each reachable leaf exactly as its recompute policy requires."""
    from fla.ops.gdn2.backends.triton_ascend import chunk_intra as npu_chunk_intra
    from fla.ops.gdn2.backends.triton_ascend import wy_fast as npu_wy

    wy_spy = Mock(wraps=npu_wy.recompute_w_u_fwd_gdn2_npu)
    bwd_intra_spy = Mock(wraps=npu_chunk_intra.chunk_gdn2_bwd_intra_npu)
    monkeypatch.setattr(npu_wy, "recompute_w_u_fwd_gdn2_npu", wy_spy)
    monkeypatch.setattr(npu_chunk_intra, "chunk_gdn2_bwd_intra_npu", bwd_intra_spy)

    cu_seqlens = None
    cu_seqlens_cpu = None
    if layout == "varlen":
        cu_seqlens = torch.tensor([0, 15, 80], dtype=torch.int64, device=device)
        cu_seqlens_cpu = cu_seqlens.cpu()
    inputs = _route_inputs(torch.bfloat16, T=80 if layout == "varlen" else _BT)
    o, final_state = chunk_gdn2(
        **inputs,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        disable_recompute=disable_recompute,
        output_final_state=True,
    )
    (o.float().sum() + final_state.float().sum()).backward()

    expected_q_is_none = [False] if disable_recompute else [True, False]
    assert wy_spy.call_count == len(expected_q_is_none)
    assert [call.kwargs["q"] is None for call in wy_spy.call_args_list] == expected_q_is_none
    bwd_intra_spy.assert_called_once()
    bwd_kwargs = bwd_intra_spy.call_args.kwargs
    assert bwd_kwargs["chunk_size"] == _BT

    routed_calls = [*wy_spy.call_args_list, bwd_intra_spy.call_args]
    for call in routed_calls:
        if cu_seqlens is None:
            assert call.kwargs["cu_seqlens"] is None
            assert call.kwargs["chunk_indices"] is None
        else:
            assert torch.equal(call.kwargs["cu_seqlens"], cu_seqlens)
            assert call.kwargs["chunk_indices"].ndim == 2
            assert call.kwargs["chunk_indices"].shape[-1] == 2


def _wy_verifier_args(
    *,
    dtype: torch.dtype = torch.float16,
    B: int = 1,
    K: int = 32,
    V: int = 32,
    chunk_size: int = _BT,
) -> dict[str, torch.Tensor]:
    k = torch.zeros(B, 1, 1, K, dtype=dtype, device=device)
    v = torch.zeros(B, 1, 1, V, dtype=dtype, device=device)
    return {
        "k": k,
        "v": v,
        "b": torch.zeros_like(k),
        "w_gate": torch.zeros_like(v),
        "A": torch.zeros(B, 1, 1, chunk_size, dtype=dtype, device=device),
        "q": torch.zeros_like(k),
        "gk": torch.zeros_like(k),
    }


def _bwd_intra_verifier_args(
    *,
    dtype: torch.dtype = torch.float16,
    B: int = 1,
    K: int = 32,
    chunk_size: int = _BT,
) -> dict[str, torch.Tensor | int]:
    x = torch.zeros(B, 1, 1, K, dtype=dtype, device=device)
    dA = torch.zeros(B, 1, 1, chunk_size, dtype=dtype, device=device)
    return {
        "q": x.clone(),
        "k": x,
        "g": x.clone(),
        "b": x.clone(),
        "dAqk": dA,
        "dAkk": dA.clone(),
        "dq": x.clone(),
        "dk": x.clone(),
        "db": x.clone(),
        "dg": x.clone(),
        "chunk_size": chunk_size,
    }


@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES, ids=("fp16", "bf16", "fp32"))
@pytest.mark.parametrize("leaf", ("wy", "bwd_intra"))
def test_gdn2_ascend_leaf_verifiers_accept_supported_dtypes(leaf, dtype):
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    backend = TritonAscendGDN2Backend()
    if leaf == "wy":
        accepted, reason = backend.recompute_w_u_fwd_gdn2_verifier(**_wy_verifier_args(dtype=dtype))
    else:
        accepted, reason = backend.chunk_gdn2_bwd_intra_verifier(**_bwd_intra_verifier_args(dtype=dtype))
    assert accepted
    assert reason is None


@pytest.mark.parametrize("leaf", ("wy", "bwd_intra"))
def test_gdn2_ascend_leaf_verifiers_accept_ub_boundary(leaf):
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    backend = TritonAscendGDN2Backend()
    if leaf == "wy":
        accepted, reason = backend.recompute_w_u_fwd_gdn2_verifier(**_wy_verifier_args(K=256, V=256))
    else:
        accepted, reason = backend.chunk_gdn2_bwd_intra_verifier(**_bwd_intra_verifier_args(K=256))
    assert accepted
    assert reason is None


def test_recompute_w_u_fwd_gdn2_verifier_accepts_partitioned_v():
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    accepted, reason = TritonAscendGDN2Backend().recompute_w_u_fwd_gdn2_verifier(
        **_wy_verifier_args(V=257),
    )
    assert accepted
    assert reason is None


@pytest.mark.parametrize(
    ("case", "reason_fragment"),
    [
        pytest.param("chunk-size", "chunk_size=64", id="chunk-size"),
        pytest.param("k-zero", "1 <= K <= 256", id="k-zero"),
        pytest.param("k-oversized", "1 <= K <= 256", id="k-oversized"),
        pytest.param("v-zero", "V >= 1", id="v-zero"),
        pytest.param("packed-batch", "batch size 1", id="packed-batch"),
        pytest.param("gk-none", "requires gk", id="gk-none"),
        pytest.param("index-dtype", "int32 or int64", id="index-dtype"),
        pytest.param("cu-seqlens-shape", "cu_seqlens to be rank 1", id="cu-seqlens-shape"),
        pytest.param("chunk-indices-shape", "chunk_indices shape [NT, 2]", id="chunk-indices-shape"),
        pytest.param("shape", "shape", id="shape"),
        pytest.param("device", "same NPU device", id="device"),
        pytest.param("dtype", "unsupported dtype", id="dtype"),
    ],
)
def test_recompute_w_u_fwd_gdn2_verifier_rejects_invalid_contract(case, reason_fragment):
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    if case == "k-zero":
        kwargs = _wy_verifier_args(K=0)
    elif case == "k-oversized":
        kwargs = _wy_verifier_args(K=257)
    elif case == "v-zero":
        kwargs = _wy_verifier_args(V=0)
    elif case == "packed-batch":
        kwargs = _wy_verifier_args(B=2)
        kwargs["cu_seqlens"] = torch.tensor([0, 1], dtype=torch.int64, device=device)
    elif case == "gk-none":
        kwargs = _wy_verifier_args()
        kwargs["gk"] = None
    else:
        kwargs = _wy_verifier_args(chunk_size=32 if case == "chunk-size" else _BT)
        if case == "index-dtype":
            kwargs["cu_seqlens"] = torch.tensor([0, 1], dtype=torch.float32, device=device)
        elif case == "cu-seqlens-shape":
            kwargs["cu_seqlens"] = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        elif case == "chunk-indices-shape":
            kwargs["chunk_indices"] = torch.tensor([0, 0], dtype=torch.int64, device=device)
        elif case == "shape":
            kwargs["b"] = torch.zeros(1, 1, 1, 31, dtype=torch.float16, device=device)
        elif case == "device":
            kwargs["q"] = kwargs["q"].cpu()
        elif case == "dtype":
            kwargs["b"] = kwargs["b"].to(torch.int32)

    accepted, reason = TritonAscendGDN2Backend().recompute_w_u_fwd_gdn2_verifier(**kwargs)
    assert not accepted
    assert reason is not None
    assert reason_fragment in reason


@pytest.mark.parametrize(
    ("case", "reason_fragment"),
    [
        pytest.param("chunk-size", "chunk_size=64", id="chunk-size"),
        pytest.param("k-zero", "1 <= K <= 256", id="k-zero"),
        pytest.param("k-oversized", "1 <= K <= 256", id="k-oversized"),
        pytest.param("packed-batch", "batch size 1", id="packed-batch"),
        pytest.param("index-dtype", "int32 or int64", id="index-dtype"),
        pytest.param("cu-seqlens-shape", "rank-1 cu_seqlens", id="cu-seqlens-shape"),
        pytest.param("chunk-indices-shape", "chunk_indices shaped [NT, 2]", id="chunk-indices-shape"),
        pytest.param("shape", "shape", id="shape"),
        pytest.param("device", "same NPU device", id="device"),
        pytest.param("dtype", "unsupported dtype", id="dtype"),
    ],
)
def test_chunk_gdn2_bwd_intra_verifier_rejects_invalid_contract(case, reason_fragment):
    from fla.ops.gdn2.backends.triton_ascend import TritonAscendGDN2Backend

    if case == "k-zero":
        kwargs = _bwd_intra_verifier_args(K=0)
    elif case == "k-oversized":
        kwargs = _bwd_intra_verifier_args(K=257)
    elif case == "packed-batch":
        kwargs = _bwd_intra_verifier_args(B=2)
        kwargs["cu_seqlens"] = torch.tensor([0, 1], dtype=torch.int64, device=device)
    else:
        kwargs = _bwd_intra_verifier_args(chunk_size=32 if case == "chunk-size" else _BT)
        if case == "index-dtype":
            kwargs["chunk_indices"] = torch.tensor([[0, 0]], dtype=torch.float32, device=device)
        elif case == "cu-seqlens-shape":
            kwargs["cu_seqlens"] = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
        elif case == "chunk-indices-shape":
            kwargs["chunk_indices"] = torch.tensor([0, 0], dtype=torch.int64, device=device)
        elif case == "shape":
            kwargs["dAkk"] = torch.zeros(1, 1, 1, 32, dtype=torch.float16, device=device)
        elif case == "device":
            kwargs["q"] = kwargs["q"].cpu()
        elif case == "dtype":
            kwargs["q"] = kwargs["q"].to(torch.int32)

    accepted, reason = TritonAscendGDN2Backend().chunk_gdn2_bwd_intra_verifier(**kwargs)
    assert not accepted
    assert reason is not None
    assert reason_fragment in reason
