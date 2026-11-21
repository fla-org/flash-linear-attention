# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import triton

from fla.ops.cyfa import chunk_cyfa, fused_recurrent_cyfa
from fla.ops.cyfa.chunk import _prune_table_grad_configs
from fla.ops.cyfa.fused_recurrent import _fused_recurrent_cyfa_fwd
from fla.ops.cyfa.naive import naive_recurrent_cyfa
from fla.ops.cyfa.qk_norm import qk_rmsnorm_fwd
from fla.ops.cyfa.utils import build_readout_table
from fla.utils import IS_NVIDIA, IS_NVIDIA_HOPPER, IS_NVIDIA_SM100, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='CyFA kernels currently require NVIDIA GPUs')


@pytest.mark.smoke
@pytest.mark.parametrize(
    ('dtype', 'as_kwargs'),
    [
        pytest.param(torch.float32, False, id='fp32-nargs'),
        pytest.param(torch.float32, True, id='fp32-kwargs'),
        pytest.param(torch.float16, False, id='fp16'),
        pytest.param(torch.bfloat16, False, id='bf16'),
        pytest.param(None, False, id='missing-right-a'),
    ],
)
def test_prune_table_grad_configs(dtype, as_kwargs):
    configs = [
        triton.Config({'BN': 64, 'BR': 32, 'BC': 64}, num_warps=4),
        triton.Config({'BN': 64, 'BR': 64, 'BC': 64}, num_warps=4),
        triton.Config({'BN': 64, 'BR': 64, 'BC': 64}, num_warps=8),
        triton.Config({'BN': 64, 'BR': 64, 'BC': 64}, num_warps=2),
        triton.Config({'BN': 64, 'BR': 32, 'BC': 64}, num_warps=8),
    ]
    args = {} if dtype is None else {'right_a': torch.empty(1, dtype=dtype, device=device)}

    actual = _prune_table_grad_configs(configs, None, **args) if as_kwargs else _prune_table_grad_configs(configs, args)

    expected = configs
    if dtype == torch.float32:
        if IS_NVIDIA_SM100:
            expected = [configs[0], configs[3], configs[4]]
        elif IS_NVIDIA_HOPPER:
            expected = [configs[0], *configs[2:]]
    assert actual == expected


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
        pytest.param(1, 65, 2, 64, 64, 64, torch.float32, False, id='fp32-slots64', marks=pytest.mark.smoke),
        pytest.param(1, 65, 2, 64, 64, 128, torch.float32, False, id='fp32-slots128-v64', marks=pytest.mark.smoke),
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


@pytest.mark.smoke
@pytest.mark.parametrize('checkpoint_level', [0, 1])
def test_chunk_split_dqk_dv(monkeypatch, checkpoint_level):
    # Pre-Hopper parts route K <= 128 to the separate dqk/dv kernels; force that
    # branch here so the Hopper CI part exercises it too.
    monkeypatch.setattr('fla.ops.cyfa.chunk.USE_FUSED_DQK_DV', False)
    torch.manual_seed(42)
    compare_chunk(make_inputs(1, 64, 1, 128, 128, 128, torch.bfloat16, True), checkpoint_level)


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
@pytest.mark.parametrize('empty_idx', [0, 1, 2], ids=['first', 'middle', 'last'])
def test_chunk_varlen_empty_gradient(checkpoint_level, empty_idx):
    torch.manual_seed(42)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
    cu = torch.cat((cu[:empty_idx + 1], cu[empty_idx:]))
    inputs = make_inputs(1, 6, 1, 32, 32, 128, torch.float16, True, cu)
    nonempty = [i for i in range(3) if i != empty_idx]
    ref, ref_state = naive_recurrent_cyfa(
        **{
            **inputs,
            'cu_seqlens': torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
            'initial_state': tuple(x.index_select(0, torch.tensor(nonempty, device=device)) for x in inputs['initial_state']),
        }
    )
    ref_state = tuple(
        torch.cat((state[:empty_idx], initial[empty_idx:empty_idx + 1], state[empty_idx:]))
        for state, initial in zip(ref_state, inputs['initial_state'], strict=True)
    )
    do = torch.randn_like(ref)
    ds = tuple(torch.randn_like(s) for s in ref_state)
    named_inputs = [(name, x) for name, x in inputs.items() if isinstance(x, torch.Tensor) and x.requires_grad]
    named_inputs.extend(zip(('hk0', 'hv0', 'lambda0'), inputs['initial_state'], strict=True))
    tensors = [x for _, x in named_inputs]
    ref_loss = (ref * do).sum() + sum((s * d).sum() for s, d in zip(ref_state, ds, strict=True))
    ref_grads = torch.autograd.grad(ref_loss, tensors)

    actual, actual_state = chunk_cyfa(**inputs, checkpoint_level=checkpoint_level)
    actual_loss = (actual * do).sum() + sum((s * d).sum() for s, d in zip(actual_state, ds, strict=True))
    actual_grads = torch.autograd.grad(actual_loss, tensors)
    for name, expected, result in [('o', ref, actual), *zip(('hkt', 'hvt', 'lambda'), ref_state, actual_state, strict=True)]:
        assert torch.isfinite(result).all(), name
        assert_close(name, expected.float(), result.float(), 0.006)
    for (name, _), expected, result in zip(named_inputs, ref_grads, actual_grads, strict=True):
        assert torch.isfinite(result).all(), name
        tol = 0.02 if name in ('g', 'delta', 'beta', 'lambda0') else 0.012
        assert_close('d' + name, expected.float(), result.float(), tol)


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


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'K', 'V', 'M', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-K{}-V{}-M{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (2, 1, 2, 64, 64, 128, None, torch.float32),
            (2, 300, 2, 64, 64, 128, None, torch.bfloat16),
            (1, 400, 2, 64, 64, 128, [0, 15, 250, 400], torch.float32),
        ]
    ],
)
@torch.no_grad()
def test_fused_recurrent_inplace_out(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    M: int,
    cu_seqlens: list[int] | None,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = -0.1 * torch.rand(B, T, H, device=device)
    delta = torch.randn(B, T, H, device=device).sigmoid()
    beta = torch.randn(B, T, H, device=device).sigmoid()
    readout = torch.eye(M - 1, device=device).expand(H, -1, -1).clone()
    readout = readout + 0.05 * torch.randn(H, M - 1, M - 1, device=device)
    q_norm_weight = 1 + 0.1 * torch.randn(K, device=device)
    k_norm_weight = 1 + 0.1 * torch.randn(K, device=device)
    if cu_seqlens is not None:
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    h0 = (
        0.1 * torch.randn(N, H, K, M, device=device),
        0.1 * torch.randn(N, H, M, V, device=device),
        17 + torch.rand(N, H, device=device),
    )
    is_decode = T == 1 and cu_seqlens is None
    if is_decode:
        q, k, q_rstd = q.contiguous(), k.contiguous(), None
    else:
        q, k, q_rstd, _ = qk_rmsnorm_fwd(q, k, q_norm_weight, k_norm_weight, 1e-6, 1e-6, store_q=False, store_rstd=True)
    readout_table = build_readout_table(readout).to(q.dtype)

    kwargs = dict(
        q=q, q_norm_weight=q_norm_weight, q_rstd=q_rstd, k=k, k_norm_weight=k_norm_weight,
        v=v, delta=delta, beta=beta, g=g, readout_table=readout_table,
        scale=K ** -0.5, q_norm_eps=1e-6, k_norm_eps=1e-6, cu_seqlens=cu_seqlens,
    )
    ref, ref_state = _fused_recurrent_cyfa_fwd(initial_state=tuple(s.clone() for s in h0), output_final_state=True, **kwargs)
    state = tuple(s.clone() for s in h0)
    out = torch.empty_like(v)
    tri, tri_state = _fused_recurrent_cyfa_fwd(
        initial_state=state, output_final_state=True, inplace_final_state=is_decode, out=out, **kwargs,
    )

    assert tri.data_ptr() == out.data_ptr()
    assert torch.equal(ref, tri)
    for expected, result in zip(ref_state, tri_state, strict=True):
        assert torch.equal(expected, result)
    if is_decode:
        assert tri_state[0].data_ptr() == state[0].data_ptr()
        assert tri_state[1].data_ptr() == state[1].data_ptr()
