# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from types import SimpleNamespace

import pytest
import torch

from fla.ops.kda import chunk_kda
from fla.ops.kda.chunk_fwd import chunk_kda_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.utils import IS_NVIDIA, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason="Compact KDA states require NVIDIA")


def _assert_bitwise(actual, expected):
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def _make_inputs(total=130, key_dim=64, value_dim=64):
    torch.manual_seed(42)
    q, k = [torch.randn(1, total, 1, key_dim, device=device, dtype=torch.bfloat16) * 0.1 for _ in range(2)]
    v = torch.randn(1, total, 1, value_dim, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, total, 1, key_dim, device=device, dtype=torch.bfloat16)
    beta = torch.rand(1, total, 1, device=device, dtype=torch.bfloat16)
    return q, k, v, g, beta


@pytest.mark.parametrize("chunk_size", [32, 64])
@pytest.mark.parametrize("state_v_first", [False, True])
def test_compact_h_slots(chunk_size, state_v_first):
    q, k, v, g, beta = _make_inputs()
    for boundaries in ([0, 130], [0, 1, 70, 71, 130], [0, 32, 96, 130], [0, 1, 2, 3, 130]):
        cu_seqlens = torch.tensor(boundaries, dtype=torch.long, device=device)
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        outputs = []
        for compact_h in (False, True):
            result = chunk_kda_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=q.shape[-1] ** -0.5,
                initial_state=None,
                output_final_state=False,
                state_v_first=state_v_first,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
                disable_recompute=True,
                compact_h=compact_h,
            )
            h = result[10]
            assert h.shape[1] == (q.shape[1] // chunk_size if compact_h else len(chunk_indices))
            outputs.append(result)

        dense, compact = outputs
        _assert_bitwise(compact[0], dense[0])
        _assert_bitwise(compact[9], dense[9])
        for flat_chunk, (sequence, local_chunk) in enumerate(chunk_indices.cpu().tolist()):
            if local_chunk == 0:
                assert torch.count_nonzero(dense[10][0, flat_chunk]) == 0
            else:
                slot = boundaries[sequence] // chunk_size + local_chunk - 1
                _assert_bitwise(compact[10][0, slot], dense[10][0, flat_chunk])


@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("disable_recompute", [False, True])
@pytest.mark.parametrize("use_final_gradient", [False, True])
def test_compact_state_matches_dense_backward(state_v_first, disable_recompute, use_final_gradient, monkeypatch):
    q, k, v, g, beta = _make_inputs()
    boundaries = [0, 1, 70, 71, 130]
    cu_seqlens = torch.tensor(boundaries, dtype=torch.long, device=device)
    do = torch.randn_like(v)
    state_dims = (1, v.shape[-1], q.shape[-1]) if state_v_first else (1, q.shape[-1], v.shape[-1])
    dht = torch.randn(len(boundaries) - 1, *state_dims, device=device)
    results = []

    for compact_h in (False, True):
        inputs = tuple(x.detach().clone().requires_grad_() for x in (q, k, v, g, beta))
        monkeypatch.setenv("FLA_KDA_COMPACT_H", str(int(compact_h)))
        output, final_state = chunk_kda(
            *inputs,
            output_final_state=True,
            state_v_first=state_v_first,
            disable_recompute=disable_recompute,
            cu_seqlens=cu_seqlens,
        )
        assert output.grad_fn.compact_h == compact_h
        # Backward must use the layout selected during forward.
        monkeypatch.setenv("FLA_KDA_COMPACT_H", str(int(not compact_h)))
        gradients = torch.autograd.grad(
            (output, final_state) if use_final_gradient else output,
            inputs,
            (do, dht) if use_final_gradient else do,
        )
        results.append((output, final_state, *gradients))

    for compact, dense in zip(results[1], results[0], strict=True):
        _assert_bitwise(compact, dense)


def test_compact_state_preserves_public_layouts(monkeypatch):
    monkeypatch.setenv("FLA_KDA_COMPACT_H", "1")
    q, k, v, g, beta = _make_inputs()
    cu_seqlens = torch.tensor([0, 1, 70, 71, 130], dtype=torch.long, device=device)
    h0 = torch.zeros(4, 1, q.shape[-1], v.shape[-1], device=device)
    output, _ = chunk_kda(
        q.requires_grad_(),
        k,
        v,
        g,
        beta,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
    )
    assert not output.grad_fn.compact_h

    with torch.inference_mode():
        _, _, h = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens=cu_seqlens,
            return_intermediate_states=True,
        )
    assert h.shape[1] == len(prepare_chunk_indices(cu_seqlens, 64))


def test_compact_h_rejects_zigzag_cp():
    q, k, v, g, beta = _make_inputs()
    with pytest.raises(AssertionError, match="contiguous CP"):
        chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=q.shape[-1] ** -0.5,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=torch.tensor([0, 130], dtype=torch.long, device=device),
            cp_context=SimpleNamespace(layout="zigzag"),
            compact_h=True,
        )
