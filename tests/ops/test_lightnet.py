# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.gla.naive import naive_recurrent_gla
from fla.ops.lightnet import chunk_lightnet, fused_recurrent_lightnet
from fla.utils import assert_close, device


def naive_lightnet_gate(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    lengths: tuple[int, ...],
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = x.unbind(0) if layout == "dense" else x.squeeze(0).split(lengths)
    keys, gates, final_states = [], [], []
    for i, sequence in enumerate(sequences):
        state = initial_state[i] if initial_state is not None else None
        if sequence.shape[0] == 0:
            keys.append(sequence)
            gates.append(sequence)
            if state is None:
                state = x.new_full((1, *x.shape[2:]), float("-inf"), dtype=torch.float32)
            final_states.append(state)
            continue

        z = sequence.float().logcumsumexp(0)
        if state is not None:
            z = torch.logaddexp(state, z)
            previous = torch.cat([state, z[:-1]], dim=0)
        else:
            previous = torch.cat([z[:1], z[:-1]], dim=0)
        keys.append((sequence.float() - z).exp().to(x.dtype))
        gates.append(torch.nan_to_num(previous - z, nan=0.0, posinf=0.0, neginf=0.0).to(x.dtype))
        final_states.append(z[-1:])

    if layout == "dense":
        k, g = torch.stack(keys), torch.stack(gates)
    else:
        k, g = torch.cat(keys).unsqueeze(0), torch.cat(gates).unsqueeze(0)
    return k, g, torch.stack(final_states)


@pytest.mark.parametrize("impl", [chunk_lightnet, fused_recurrent_lightnet], ids=["chunk", "fused-recurrent"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    ("layout", "lengths", "use_initial_state", "state_v_first", "output_final_state"),
    [
        pytest.param("dense", (1, 1), False, False, False, id="dense-singleton-cold"),
        pytest.param("dense", (65, 65), True, True, True, id="dense-T65-warm"),
        pytest.param("packed", (0, 1, 33, 65, 0), False, False, True, id="varlen-cold"),
        pytest.param("packed", (0, 1, 65, 0), True, True, True, id="varlen-warm"),
    ],
)
def test_chunk(
    impl,
    dtype: torch.dtype,
    layout: str,
    lengths: tuple[int, ...],
    use_initial_state: bool,
    state_v_first: bool,
    output_final_state: bool,
):
    torch.manual_seed(42)
    N, H, K, V = len(lengths), 2, 32, 16
    B, T = (N, lengths[0]) if layout == "dense" else (1, sum(lengths))
    q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
    ref_q, ref_k, ref_v = (x.detach().clone().requires_grad_() for x in (q, k, v))
    h0, z0 = None, None
    if use_initial_state:
        state_shape = (N, H, V, K) if state_v_first else (N, H, K, V)
        h0 = torch.randn(state_shape, device=device, dtype=torch.float32, requires_grad=True)
        z0 = (torch.randn(N, 1, H, K, device=device, dtype=torch.float32) + 2).requires_grad_()
    ref_h0 = h0.detach().clone().requires_grad_() if h0 is not None else None
    ref_z0 = z0.detach().clone().requires_grad_() if z0 is not None else None
    cu_seqlens = None
    if layout == "packed":
        cu_seqlens = torch.tensor((0, *lengths), device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)

    ref_keys, ref_gates, ref_zt = naive_lightnet_gate(x=ref_k, initial_state=ref_z0, lengths=lengths, layout=layout)
    ref_outputs, ref_states = [], []
    start = 0
    for i, length in enumerate(lengths):
        index = (slice(i, i + 1), slice(None)) if layout == "dense" else (slice(None), slice(start, start + length))
        state = ref_h0[i:i + 1] if ref_h0 is not None else None
        if state_v_first:
            state = state.transpose(-1, -2)
        ref_o, ref_ht = naive_recurrent_gla(
            q=ref_q[index],
            k=ref_keys[index],
            v=ref_v[index],
            gk=ref_gates[index],
            initial_state=state,
            output_final_state=output_final_state,
        )
        ref_outputs.append(ref_o)
        if output_final_state:
            ref_states.append(ref_ht.transpose(-1, -2) if state_v_first else ref_ht)
        start += length
    ref_o = torch.cat(ref_outputs, dim=0 if layout == "dense" else 1)
    ref_ht = torch.cat(ref_states, dim=0) if output_final_state else None
    o, (ht, zt) = impl(
        q=q,
        k=k,
        v=v,
        initial_state=(h0, z0) if use_initial_state else None,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
    )

    tol = {torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
    assert_close("o", ref_o, o, tol)
    assert torch.equal(torch.isneginf(ref_zt), torch.isneginf(zt))
    finite = torch.isfinite(ref_zt)
    assert_close("zt", ref_zt[finite], zt[finite], tol)
    ref_outputs, outputs = [ref_o, ref_zt], [o, zt]
    if output_final_state:
        assert_close("ht", ref_ht, ht, tol)
        ref_outputs.append(ref_ht)
        outputs.append(ht)
    else:
        assert ht is None
    gradients = [torch.randn_like(output) for output in outputs]
    torch.autograd.backward(ref_outputs, gradients)
    torch.autograd.backward(outputs, gradients)
    for name, ref, actual in zip(("dq", "dk", "dv"), (ref_q, ref_k, ref_v), (q, k, v)):
        assert_close(name, ref.grad, actual.grad, tol)
    if use_initial_state:
        assert_close("dh0", ref_h0.grad, h0.grad, tol)
        assert_close("dz0", ref_z0.grad, z0.grad, tol)
