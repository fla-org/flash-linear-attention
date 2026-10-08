# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.utils.logcumsumexp import logcumsumexp_normalize
from fla.utils import assert_close, device


def naive_logcumsumexp_normalize(
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    ("layout", "lengths", "state_mode", "H", "K"),
    [
        pytest.param("dense", (1, 1), "cold", 2, 17, id="dense-singleton-cold"),
        pytest.param("dense", (33, 33), "cold", 1, 32, id="dense-T33-cold"),
        pytest.param("dense", (65, 65), "warm", 2, 17, id="dense-T65-warm"),
        pytest.param("dense", (1024, 1024), "warm", 3, 64, id="dense-T1024-warm"),
        pytest.param("dense", (0, 0, 0), "cold", 2, 17, id="dense-empty-cold"),
        pytest.param("dense", (0, 0, 0), "mixed", 2, 17, id="dense-empty-mixed-cache"),
        pytest.param("packed", (0, 1, 31, 32, 33, 0, 63, 64, 65, 0), "cold", 2, 17, id="varlen-boundaries-cold"),
        pytest.param("packed", (0, 1, 31, 32, 33, 0, 63, 64, 65, 0), "warm", 2, 17, id="varlen-boundaries-warm"),
        pytest.param("packed", (0, 1, 31, 32, 33, 0, 63, 64, 65, 0), "mixed", 2, 17, id="varlen-mixed-cache"),
        pytest.param("packed", (0, 0, 0), "cold", 2, 17, id="varlen-empty-cold"),
        pytest.param("packed", (0, 0, 0), "mixed", 2, 17, id="varlen-empty-mixed-cache"),
        pytest.param("packed", (1024, 1, 0, 65), "mixed", 2, 17, id="varlen-T1024-mixed-cache"),
    ],
)
def test_logcumsumexp_normalize(dtype: torch.dtype, layout: str, lengths: tuple[int, ...], state_mode: str, H: int, K: int):
    torch.manual_seed(42)
    N = len(lengths)
    B, T = (N, lengths[0]) if layout == "dense" else (1, sum(lengths))
    x = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    initial_state = None
    if state_mode != "cold":
        initial_state = torch.randn(N, 1, H, K, device=device, dtype=torch.float32) + 2
        if state_mode == "mixed":
            initial_state[::2] = float("-inf")
        initial_state.requires_grad_()
    reference_state = initial_state.detach().clone().requires_grad_() if initial_state is not None else None
    cu_seqlens = None
    if layout == "packed":
        cu_seqlens = torch.tensor((0, *lengths), device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)

    actual = logcumsumexp_normalize(x=x, initial_state=initial_state, cu_seqlens=cu_seqlens)
    expected = naive_logcumsumexp_normalize(x=reference_x, initial_state=reference_state, lengths=lengths, layout=layout)
    tol = {torch.float32: 1e-3, torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
    for name, reference, result in zip(("k", "g"), expected[:2], actual[:2]):
        assert result.shape == x.shape
        assert result.dtype == dtype
        assert torch.isfinite(result).all()
        if reference.numel():
            assert_close(name, reference, result, tol)

    assert actual[2].shape == (N, 1, H, K)
    assert actual[2].dtype == torch.float32
    assert torch.equal(torch.isneginf(actual[2]), torch.isneginf(expected[2]))
    finite = torch.isfinite(expected[2])
    if finite.any():
        assert_close("final_state", expected[2][finite], actual[2][finite], tol)

    gradients = [torch.randn_like(output) for output in expected]
    assert all(output.requires_grad for output in actual[:2])
    if x.numel() or initial_state is not None:
        assert actual[2].requires_grad
    # direct upstream gradients also exercise identity state propagation through empty sequences.
    for outputs in (actual, expected):
        differentiable = [(output, grad) for output, grad in zip(outputs, gradients) if output.requires_grad]
        torch.autograd.backward([output for output, _ in differentiable], [grad for _, grad in differentiable])
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    if x.numel():
        assert_close("dx", reference_x.grad, x.grad, tol)
    if initial_state is not None:
        assert torch.equal(initial_state, reference_state)
        assert initial_state.grad is not None
        assert torch.isfinite(initial_state.grad).all()
        assert_close("dinitial_state", reference_state.grad, initial_state.grad, tol)
