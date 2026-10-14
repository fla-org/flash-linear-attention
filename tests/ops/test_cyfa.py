# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.cyfa import chunk_cyfa, fused_recurrent_cyfa
from fla.ops.cyfa.naive import naive_recurrent_cyfa
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='CyFA kernels currently require NVIDIA GPUs')


def make_inputs(B, T, H, K, V, M, dtype, initial_state=False, cu_seqlens=None):
    inputs = {
        'q': torch.randn(B, T, H, K, device=device, dtype=dtype),
        'k': torch.randn(B, T, H, K, device=device, dtype=dtype),
        'v': torch.randn(B, T, H, V, device=device, dtype=dtype),
        'g': -0.1 * torch.rand(B, T, H, device=device),
        'delta': torch.randn(B, T, H, device=device).sigmoid(),
        'beta': torch.randn(B, T, H, device=device).sigmoid(),
        'readout': torch.eye(M - 1, device=device).expand(H, -1, -1).clone()
        + 0.05 * torch.randn(H, M - 1, M - 1, device=device),
        'q_norm_weight': 1 + 0.1 * torch.randn(K, device=device),
        'k_norm_weight': 1 + 0.1 * torch.randn(K, device=device),
    }
    inputs = {name: value.requires_grad_() for name, value in inputs.items()}
    if initial_state:
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        inputs['initial_state'] = (
            (0.1 * torch.randn(N, H, K, M, device=device)).requires_grad_(),
            (0.1 * torch.randn(N, H, M, V, device=device)).requires_grad_(),
            (17 + torch.rand(N, H, device=device)).requires_grad_(),
        )
    inputs.update(q_norm_eps=1e-6, k_norm_eps=1e-6, cu_seqlens=cu_seqlens, output_final_state=True)
    return inputs


def compare_chunk(inputs, checkpoint_level):
    ref, ref_state = naive_recurrent_cyfa(**inputs)
    ref_state = ref_state or ()
    do = torch.randn_like(ref)
    ds = tuple(torch.randn_like(s) for s in ref_state)
    named_inputs = [(name, x) for name, x in inputs.items() if isinstance(x, torch.Tensor) and x.requires_grad]
    named_inputs.extend(zip(('hk0', 'hv0', 'lambda0'), inputs.get('initial_state', ()), strict=False))
    tensors = [x for _, x in named_inputs]
    ref_loss = (ref * do).sum() + sum((s * d).sum() for s, d in zip(ref_state, ds, strict=True))
    ref_grads = torch.autograd.grad(ref_loss, tensors)

    actual, actual_state = chunk_cyfa(**inputs, checkpoint_level=checkpoint_level)
    actual_state = actual_state or ()
    actual_loss = (actual * do).sum() + sum((s * d).sum() for s, d in zip(actual_state, ds, strict=True))
    actual_grads = torch.autograd.grad(actual_loss, tensors)
    output_tol, grad_tol = (0.005, 0.01) if inputs['cu_seqlens'] is None else (0.006, 0.012)
    assert len(ref_state) == len(actual_state) == (3 if inputs['output_final_state'] else 0)
    names = ('hkt', 'hvt', 'lambda')[:len(ref_state)]
    for name, expected, result in [('o', ref, actual), *zip(names, ref_state, actual_state, strict=True)]:
        assert torch.isfinite(result).all(), name
        assert_close(name, expected.float(), result.float(), output_tol)
    for (name, _), expected, result in zip(named_inputs, ref_grads, actual_grads, strict=True):
        assert torch.isfinite(result).all(), name
        tol = 0.02 if name in ('g', 'delta', 'beta', 'lambda0') else grad_tol
        assert_close('d' + name, expected.float(), result.float(), tol)


@pytest.mark.parametrize('checkpoint_level', [0, 1])
@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'M', 'dtype', 'initial_state'),
    [
        pytest.param(1, 64, 2, 32, 32, 128, torch.float32, False, id='fp32', marks=pytest.mark.smoke),
        pytest.param(2, 256, 2, 64, 64, 128, torch.float32, True, id='fp32-state'),
        pytest.param(2, 100, 3, 64, 64, 128, torch.float16, True, id='fp16-ragged'),
        pytest.param(1, 64, 1, 128, 128, 128, torch.bfloat16, False, id='bf16-d128'),
        pytest.param(1, 64, 1, 256, 256, 128, torch.bfloat16, True, id='bf16-d256'),
        pytest.param(1, 65, 2, 48, 32, 128, torch.float16, False, id='fp16-k48-v32'),
        pytest.param(1, 65, 2, 32, 48, 128, torch.float16, True, id='fp16-k32-v48'),
    ],
)
def test_chunk(B, T, H, K, V, M, dtype, initial_state, checkpoint_level):
    torch.manual_seed(42)
    compare_chunk(make_inputs(B, T, H, K, V, M, dtype, initial_state), checkpoint_level)


@pytest.mark.parametrize('checkpoint_level', [0, 1])
@pytest.mark.parametrize('initial_state', [False, True])
@pytest.mark.parametrize(
    ('cu_seqlens', 'dtype'),
    [
        pytest.param([0, 64, 128], torch.float32, id='fp32'),
        pytest.param([0, 15, 100, 256], torch.float16, id='fp16-ragged'),
    ],
)
def test_chunk_varlen(cu_seqlens, dtype, initial_state, checkpoint_level):
    torch.manual_seed(42)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    compare_chunk(make_inputs(1, cu_seqlens[-1], 2, 64, 64, 128, dtype, initial_state, cu), checkpoint_level)


@pytest.mark.parametrize('checkpoint_level', [0, 1])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_chunk_without_final_state(dtype, checkpoint_level):
    torch.manual_seed(42)
    inputs = make_inputs(1, 65, 2, 64, 64, 128, dtype, initial_state=True)
    inputs['output_final_state'] = False
    compare_chunk(inputs, checkpoint_level)


@pytest.mark.parametrize('T', [1, 64])
@pytest.mark.parametrize('varlen', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@torch.no_grad()
def test_fused_recurrent(T, varlen, dtype):
    torch.manual_seed(42)
    B, tokens = (1, 2 * T + 3) if varlen else (2, T)
    cu = torch.tensor([0, T, tokens], dtype=torch.int32, device=device) if varlen else None
    inputs = make_inputs(B, tokens, 2, 64, 64, 128, dtype, True, cu)
    ref, ref_state = naive_recurrent_cyfa(**inputs)
    actual, state = fused_recurrent_cyfa(**inputs)
    assert_close('o', ref.float(), actual.float(), 0.005)
    for name, expected, result in zip(('hkt', 'hvt', 'lambda'), ref_state, state, strict=True):
        assert torch.isfinite(result).all(), name
        assert_close(name, expected, result, 0.005)


@pytest.mark.smoke
@pytest.mark.parametrize('T', [1, 64])
@pytest.mark.parametrize('name', ['k', 'q_norm_weight', 'k_norm_weight'])
@torch.no_grad()
def test_fused_recurrent_invalid_shape(T, name):
    torch.manual_seed(42)
    inputs = make_inputs(1, T, 1, 32, 32, 16, torch.float32)
    inputs[name] = inputs[name][..., :-1]
    with pytest.raises(ValueError, match='Q/K RMSNorm'):
        fused_recurrent_cyfa(**inputs)


@pytest.mark.smoke
@pytest.mark.parametrize('T', [1, 64])
@torch.no_grad()
def test_fused_recurrent_noncontiguous_weights(T):
    torch.manual_seed(42)
    inputs = make_inputs(1, T, 2, 64, 64, 128, torch.bfloat16, initial_state=True)
    for name in ('q_norm_weight', 'k_norm_weight'):
        inputs[name] = torch.stack((inputs[name], torch.zeros_like(inputs[name])), dim=-1)[:, 0]
        assert not inputs[name].is_contiguous()
    ref, ref_state = naive_recurrent_cyfa(**inputs)
    actual, state = fused_recurrent_cyfa(**inputs)
    assert_close('o', ref.float(), actual.float(), 0.005)
    for name, expected, result in zip(('hkt', 'hvt', 'lambda'), ref_state, state, strict=True):
        assert torch.isfinite(result).all(), name
        assert_close(name, expected, result, 0.005)


@pytest.mark.smoke
@torch.no_grad()
def test_fused_recurrent_prefill():
    torch.manual_seed(42)
    inputs = make_inputs(2, 69, 2, 64, 64, 128, torch.bfloat16)
    ref, ref_state = chunk_cyfa(**inputs)
    prefix = {name: x[:, :65] if name in ('q', 'k', 'v', 'g', 'delta', 'beta') else x for name, x in inputs.items()}
    output, state = chunk_cyfa(**prefix)
    outputs = [output]
    for t in range(65, 69):
        step = {name: x[:, t:t + 1] if name in ('q', 'k', 'v', 'g', 'delta', 'beta') else x for name, x in inputs.items()}
        output, state = fused_recurrent_cyfa(**step, initial_state=state)
        outputs.append(output)
    assert_close('o', ref.float(), torch.cat(outputs, dim=1).float(), 0.006)
    for name, expected, result in zip(('hkt', 'hvt', 'lambda'), ref_state, state, strict=True):
        assert_close(name, expected, result, 0.006)


@pytest.mark.parametrize('state_only', [False, True])
def test_fused_recurrent_requires_grad(state_only):
    torch.manual_seed(42)
    inputs = make_inputs(1, 1, 1, 32, 32, 16, torch.float32, initial_state=state_only)
    if state_only:
        inputs = {name: x.detach() if isinstance(x, torch.Tensor) else x for name, x in inputs.items()}
    with pytest.raises(NotImplementedError, match='inference-only'):
        fused_recurrent_cyfa(**inputs)
