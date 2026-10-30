# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""NPU GDN fixed-capacity forward/backward replay and eager isolation tests."""

from importlib import import_module

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import IS_NPU, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NPU, reason="GDN graph replay requires Ascend NPU")


def _inputs(dtype=torch.bfloat16, dim=64, capacity=193, heads=2, n=4, seed=0):
    gen = torch.Generator(device).manual_seed(seed)
    shape = (1, capacity, heads, dim)
    q, k = [F.normalize(torch.randn(shape, device=device, generator=gen), dim=-1).to(dtype) for _ in range(2)]
    v = torch.randn(shape, dtype=dtype, device=device, generator=gen)
    g = F.logsigmoid(torch.randn(shape[:3], device=device, generator=gen)) / 8
    beta = torch.rand(shape[:3], dtype=dtype, device=device, generator=gen)
    h0 = torch.randn(n, heads, dim, dim, device=device, generator=gen)
    return q, k, v, g, beta, h0


def _call(inputs, cu, use_graph=True, state=True, final=True, scale=0.25, **kwargs):
    q, k, v, g, beta, h0 = inputs
    return chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0 if state else None,
        output_final_state=final,
        cu_seqlens=cu,
        scale=scale,
        use_graph=use_graph,
        **kwargs,
    )


def _reference(inputs, offsets, state=True, final=True, scale=0.25):
    # run ordinary chunk eager per nonempty sequence; assemble empty states and inactive capacity explicitly.
    q, k, v, g, beta, h0 = inputs
    out = torch.zeros_like(v)
    ht = h0.clone() if state else torch.zeros_like(h0)
    for n, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start == end:
            continue
        args = [x[:, start:end].contiguous() for x in (q, k, v, g, beta)]
        o, h = chunk_gated_delta_rule(
            q=args[0],
            k=args[1],
            v=args[2],
            g=args[3],
            beta=args[4],
            scale=scale,
            initial_state=h0[n:n+1] if state else None,
            output_final_state=True,
            use_graph=False,
        )
        out[:, start:end] = o
        ht[n:n+1] = h
    return out, ht if final else None


def _check(actual, expected):
    for name, result, ref in zip(('o', 'ht'), actual, expected):
        if ref is None:
            assert result is None
        else:
            assert torch.isfinite(result).all(), name
            assert_close(name, ref, result, 3e-3)


@pytest.mark.parametrize(
    'dtype,dim,cu_dtype',
    [
        (torch.float16, 64, torch.int32), (torch.bfloat16, 64, torch.int64),
        (torch.float16, 128, torch.int64), (torch.bfloat16, 128, torch.int32),
    ],
    ids=[
        'dtype-float16-dim-64-cu_dtype-int32',
        'dtype-bfloat16-dim-64-cu_dtype-int64',
        'dtype-float16-dim-128-cu_dtype-int64',
        'dtype-bfloat16-dim-128-cu_dtype-int32',
    ],
)
@pytest.mark.parametrize(
    'state,final',
    [(False, False), (False, True), (True, False), (True, True)],
    ids=[
        'state-false-final-false',
        'state-false-final-true',
        'state-true-final-false',
        'state-true-final-true',
    ],
)
@pytest.mark.parametrize('capture', ['direct', 'callable'], ids=['capture-direct', 'capture-callable'])
@torch.no_grad()
def test_replay_changes_values_lengths_and_empty_sequences(dtype, dim, cu_dtype, state, final, capture):
    sample = _inputs(dtype, dim)
    cu = torch.tensor([0, 1, 64, 129, 193], device=device, dtype=cu_dtype)

    def step(*args):
        o, ht = _call(args[:-1], args[-1], state=state, final=final)
        return (o, ht) if final else o

    if capture == 'direct':
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                step(*sample, cu)
        torch.npu.current_stream().wait_stream(stream)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = step(*sample, cu)
    else:
        replay = torch.npu.make_graphed_callables(step, sample + (cu,), allow_unused_input=True)

    layouts = [
        [0, 1, 64, 129, 193], [0, 1, 2, 3, 193], [0, 0, 65, 65, 129], [0, 0, 0, 0, 0],
        [0, 1, 2, 3, 193],
        [0, 63, 64, 65, 193], [0, 1, 1, 1, 1], [0, 64, 128, 192, 193],
    ]
    for seed, offsets in enumerate(layouts, 1):
        inputs = _inputs(dtype, dim, seed=seed)
        # poison inactive capacity; the kernels must not use it in reductions.
        for tensor in inputs[:5]:
            tensor[:, offsets[-1]:] = float('nan')
        before = inputs[-1].clone()
        current_cu = torch.tensor(offsets, dtype=cu_dtype, device=device)
        uncaptured = _call(inputs, current_cu, state=state, final=final)
        if capture == 'direct':
            for dst, src in zip(sample, inputs):
                dst.copy_(src)
            cu.copy_(current_cu)
            graph.replay()
            result = captured
        else:
            result = replay(*inputs, current_cu)
        actual = result if final else (result, None)
        for result_tensor, expected in zip(actual, uncaptured):
            if expected is None:
                assert result_tensor is None
            else:
                torch.testing.assert_close(result_tensor, expected, rtol=0, atol=0)
        _check(actual, _reference(inputs, offsets, state, final))
        if offsets[-1] == inputs[0].shape[1] and all(a < b for a, b in zip(offsets, offsets[1:])):
            _check(actual, _call(inputs, current_cu, use_graph=False, state=state, final=final))
        assert torch.count_nonzero(actual[0][:, offsets[-1]:]) == 0
        torch.testing.assert_close(inputs[-1], before, rtol=0, atol=0)
        if capture == 'direct':
            torch.testing.assert_close(sample[-1], before, rtol=0, atol=0)
        if final:
            for n, (start, end) in enumerate(zip(offsets, offsets[1:])):
                if start == end:
                    expected = inputs[-1][n] if state else torch.zeros_like(inputs[-1][n])
                    torch.testing.assert_close(actual[1][n], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    'length',
    [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129],
    ids=[
        'length-1',
        'length-2',
        'length-15',
        'length-16',
        'length-17',
        'length-31',
        'length-32',
        'length-33',
        'length-63',
        'length-64',
        'length-65',
        'length-127',
        'length-128',
        'length-129',
    ],
)
@torch.no_grad()
def test_graph_chunk_boundaries(length):
    inputs = _inputs(capacity=length, heads=1)
    offsets = [0, 0, length, length, length]
    cu = torch.tensor(offsets, device=device)
    _check(_call(inputs, cu, scale=None), _reference(inputs, offsets, scale=None))


@pytest.mark.parametrize(
    'option,value',
    [
        ('use_qk_l2norm_in_kernel', True), ('use_beta_sigmoid_in_kernel', True),
        ('allow_neg_eigval', True), ('state_v_first', True), ('use_gate_in_kernel', True),
        ('chunk_size', 32), ('chunk_size', 16), ('head_first', False), ('unknown_option', True),
        ('cp_context', object()), ('chunk_indices', object()), ('cu_seqlens_cpu', object()),
        ('A_log', object()), ('dt_bias', object()),
    ],
    ids=[
        'option-use_qk_l2norm_in_kernel-value-true',
        'option-use_beta_sigmoid_in_kernel-value-true',
        'option-allow_neg_eigval-value-true',
        'option-state_v_first-value-true',
        'option-use_gate_in_kernel-value-true',
        'option-chunk_size-value-32',
        'option-chunk_size-value-16',
        'option-head_first-value-false',
        'option-unknown_option-value-true',
        'option-cp_context-value-object',
        'option-chunk_indices-value-object',
        'option-cu_seqlens_cpu-value-object',
        'option-a_log-value-object',
        'option-dt_bias-value-object',
    ],
)
@torch.no_grad()
def test_graph_rejects_unsupported_options(option, value):
    inputs = _inputs()
    cu = torch.tensor([0, 193, 193, 193, 193], device=device)
    with pytest.raises(ValueError, match=option):
        _call(inputs, cu, **{option: value})


@pytest.mark.parametrize(
    'case',
    [
        'dense', 'batch', 'gva', 'dim', 'dtype', 'g_dtype', 'beta_dtype', 'h_dtype', 'h_shape',
        'cu_dtype', 'cu_rank', 'cu_empty', 'strides', 'device', 'capacity', 'scale',
    ],
    ids=[
        'case-dense',
        'case-batch',
        'case-gva',
        'case-dim',
        'case-dtype',
        'case-g_dtype',
        'case-beta_dtype',
        'case-h_dtype',
        'case-h_shape',
        'case-cu_dtype',
        'case-cu_rank',
        'case-cu_empty',
        'case-strides',
        'case-device',
        'case-capacity',
        'case-scale',
    ],
)
@torch.no_grad()
def test_graph_rejects_unsupported_metadata(case):
    inputs = list(_inputs())
    cu = torch.tensor([0, 193, 193, 193, 193], device=device)
    scale = 0.25
    if case == 'dense':
        cu = None
    elif case == 'batch':
        inputs[0] = inputs[0].repeat(2, 1, 1, 1)
    elif case == 'gva':
        inputs[2] = inputs[2].repeat(1, 1, 2, 1)
    elif case == 'dim':
        inputs = list(_inputs(dim=32))
    elif case == 'dtype':
        inputs[1] = inputs[1].float()
    elif case == 'g_dtype':
        inputs[3] = inputs[3].half()
    elif case == 'beta_dtype':
        inputs[4] = inputs[4].float()
    elif case == 'h_dtype':
        inputs[5] = inputs[5].half()
    elif case == 'h_shape':
        inputs[5] = inputs[5][:1]
    elif case == 'cu_dtype':
        cu = cu.float()
    elif case == 'cu_rank':
        cu = cu[None]
    elif case == 'cu_empty':
        cu = cu[:1]
    elif case == 'strides':
        inputs[0] = inputs[0].transpose(-1, -2).contiguous().transpose(-1, -2)
    elif case == 'device':
        inputs[3] = inputs[3].cpu()
    elif case == 'capacity':
        inputs = list(_inputs(capacity=0))
    elif case == 'scale':
        scale = torch.tensor(1., device=device)
    with pytest.raises(ValueError):
        _call(inputs, cu, scale=scale)


@pytest.mark.parametrize(
    'override',
    [
        {'B': 2}, {'D': 256}, {'DV': 128}, {'HV': 4}, {'chunk_size': 32}, {'fused_options': True},
        {'disable_recompute': True}, {'safe_gate': True}, {'allow_neg_eigval': True}, {'dtype': 'float32'},
    ],
    ids=['batch', 'dim', 'value-dim', 'heads', 'chunk-size', 'fused',
         'recompute', 'safe-gate', 'negative-eigenvalue', 'dtype'],
)
def test_benchmark_rejects_unsupported_graph_config(override):
    from benchmarks.ops.graph_registry import get_graph_op

    with pytest.raises(ValueError):
        get_graph_op('chunk_gdn').build_case({'B': 1, 'T': 128, 'H': 2, 'D': 64, **override}, 'fwd', 'npu:0', 42)


def test_runtime_and_benchmark_share_graph_config_validation(monkeypatch):
    from benchmarks.ops.graph_registry import get_graph_op

    graph = import_module('fla.ops.gated_delta_rule.graph')
    calls = []

    def reject(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError('shared graph contract')

    monkeypatch.setattr(graph, '_validate_graph_config', reject)
    inputs = _inputs()
    cu = torch.tensor([0, 193, 193, 193, 193], device=device)
    with pytest.raises(ValueError, match='shared graph contract'):
        _call(inputs, cu)
    with pytest.raises(ValueError, match='shared graph contract'):
        get_graph_op('chunk_gdn').build_case({'B': 1, 'T': 128, 'H': 2, 'D': 64}, 'fwd', 'npu:0', 42)
    assert len(calls) == 2


def test_benchmark_input_geometry_matches_metadata(monkeypatch):
    from benchmarks.ops.graph_registry import get_graph_op

    graph = import_module('fla.ops.gated_delta_rule.graph')
    monkeypatch.setattr(graph, '_validate_graph_config', lambda **kwargs: None)
    case = get_graph_op('chunk_gdn').build_case(
        shape={'B': 2, 'T': 16, 'H': 2, 'D': 64, 'HV': 4, 'DV': 128, 'N': 4},
        mode='fwdbwd',
        device='npu:0',
        seed=42,
    )
    q, k, v, g, beta, h0, cu = case.graph_args
    assert q.shape == k.shape == (2, 16, 2, 64)
    assert v.shape == (2, 16, 4, 128)
    assert g.shape == beta.shape == (2, 16, 4)
    assert h0.shape == (4, 4, 64, 128)
    assert cu.shape == (5,)


@torch.no_grad()
def test_persistent_kernels_skip_holes_without_losing_later_tasks(monkeypatch):
    from fla.ops.utils.index import prepare_chunk_indices_static
    from fla.ops.utils.triton_ascend.cumsum import chunk_local_cumsum_npu
    from fla.ops.utils.triton_ascend.solve_tril import _solve_tril_npu

    kkt = import_module('fla.ops.common.backends.triton_ascend.chunk_scaled_dot_kkt')
    wy = import_module('fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast')
    cs = import_module('fla.ops.utils.triton_ascend.cumsum')
    for module in (kkt, wy):
        monkeypatch.setattr(module, 'get_npu_properties', lambda: {'num_aicore': 1})
    monkeypatch.setattr(cs, 'get_multiprocessor_count', lambda: 1)
    q, k, v, g, beta, _ = _inputs(capacity=65, heads=1)
    cu = torch.tensor([0, 65], device=device)
    indices, _ = prepare_chunk_indices_static(cu, 64, 2)
    holes = torch.tensor([[-1, 0], [0, 0], [-1, 0], [0, 1], [-1, 0]], device=device)

    def run(index):
        common = dict(cu_seqlens=cu, chunk_indices=index, use_graph=True)
        cg = chunk_local_cumsum_npu(g, chunk_size=64, **common)
        a = kkt.chunk_scaled_dot_kkt_fwd_npu(k, cg, beta, **common)
        inv = _solve_tril_npu(a, output_dtype=k.dtype, **common)
        w, u = wy.recompute_w_u_fwd_npu(k, v, beta, inv, cg, **common)
        return cg, a, inv, w, u

    for actual, expected in zip(run(holes), run(indices)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('dim', [64, 128], ids=['dim-64', 'dim-128'])
@pytest.mark.parametrize('packed', [False, True], ids=['packed-false', 'packed-true'])
def test_default_eager_matches_explicit_false(dim, packed):
    inputs = _inputs(dim=dim, capacity=65)
    outputs = []
    for explicit in (False, True):
        leaves = [x.detach().clone().requires_grad_() for x in inputs]
        q, k, v, g, beta, h0 = leaves
        kwargs = {'use_graph': False} if explicit else {}
        cu = torch.tensor([0, 1, 17, 64, 65], device=device) if packed else None
        o, ht = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=h0 if packed else h0[:1],
            output_final_state=True,
            cu_seqlens=cu,
            **kwargs,
        )
        (o.float().square().sum() + ht.square().sum()).backward()
        outputs.append((o.detach(), ht.detach(), *(x.grad for x in leaves)))
    for actual, expected in zip(*outputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_output_persistent_kernel_skips_holes(monkeypatch):
    torch.manual_seed(42)
    output = import_module('fla.ops.common.backends.triton_ascend.chunk_o')
    monkeypatch.setattr(output, 'get_npu_properties', lambda: {'num_aicore': 1})
    q, k, v, g, _, _ = _inputs(capacity=65, heads=1)
    cu = torch.tensor([0, 65], device=device)
    offsets = torch.tensor([0, 2], device=device)
    indices = torch.tensor([[0, 0], [0, 1]], device=device)
    holes = torch.tensor([[-1, 0], [0, 0], [-1, 0], [0, 1], [-1, 0]], device=device)
    h = torch.randn(1, 2, 1, 64, 64, dtype=q.dtype, device=device)
    padded_h = torch.full((1, 5, 1, 64, 64), float('nan'), dtype=q.dtype, device=device)
    padded_h[:, 1] = h[:, 0]
    padded_h[:, 3] = h[:, 1]
    common = dict(q=q, k=k, v=v, g=g, cu_seqlens=cu, chunk_offsets=offsets, use_graph=True)
    expected = output.chunk_fwd_o_npu(h=h, chunk_indices=indices, **common)
    actual = output.chunk_fwd_o_npu(h=padded_h, chunk_indices=holes, **common)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_graph_rejects_cpu_before_loading_backend():
    inputs = tuple(x.cpu() for x in _inputs())
    with pytest.raises(ValueError, match='only on NPU'):
        _call(inputs, torch.tensor([0, 193, 193, 193, 193]))


_INPUT_NAMES = ('q', 'k', 'v', 'g', 'beta', 'h0')
_TRAINING_TOLERANCES = {'o': 0.005, 'ht': 0.005, 'dq': 0.007, 'dk': 0.008,
                        'dv': 0.007, 'dg': 0.015, 'dbeta': 0.015, 'dh0': 0.007}


def _leaves(inputs, required=tuple(range(6))):
    return tuple(x.detach().clone().requires_grad_(i in required) for i, x in enumerate(inputs))


def _training_step(state=True, final=True, use_graph=True):
    def step(*args):
        o, ht = _call(args[:-1], args[-1], state=state, final=final, use_graph=use_graph)
        return (o, ht) if final else o
    return step


def _training_snapshot(step, inputs, cu, do, dht, final=True, contribution='both'):
    outputs = step(*inputs, cu)
    o, ht = outputs if final else (outputs, None)
    result = {'o': o.detach().clone()}
    if final:
        result['ht'] = ht.detach().clone()
    differentiated, upstream = [], []
    if contribution != 'ht':
        differentiated.append(o)
        upstream.append(do)
    if final and contribution != 'o':
        differentiated.append(ht)
        upstream.append(dht)
    active = [(name, x) for name, x in zip(_INPUT_NAMES, inputs) if x.requires_grad]
    grads = torch.autograd.grad(differentiated, [x for _, x in active], upstream)
    result.update((f'd{name}', grad.detach().clone()) for (name, _), grad in zip(active, grads))
    return result


def _training_reference(inputs, offsets, state=True, final=True):
    outputs = _reference(inputs, offsets, state=state, final=final)
    # empty slices establish zero derivatives without reading poisoned capacity.
    zero = sum(x[:, :0].sum() for x in inputs[:5]) + inputs[5][:0].sum()
    return tuple(x + zero if x is not None else None for x in outputs)


def _assert_training(actual, expected, exact=False):
    assert actual.keys() == expected.keys()
    for name, result in actual.items():
        assert torch.isfinite(result).all(), name
        assert torch.isfinite(expected[name]).all(), name
        if exact:
            torch.testing.assert_close(result, expected[name], rtol=0, atol=0)
        else:
            assert_close(name, expected[name], result, _TRAINING_TOLERANCES[name])


def _training_upstream(inputs, seed, active_tokens=None, strided=False):
    gen = torch.Generator(device).manual_seed(seed)
    do = torch.randn(inputs[2].shape, dtype=inputs[2].dtype, device=device, generator=gen)
    dht = torch.randn(inputs[5].shape, device=device, generator=gen)
    if active_tokens is not None:
        do[:, active_tokens:] = float('nan')
    if strided:
        do = do.transpose(-1, -2).contiguous().transpose(-1, -2)
        dht = dht.transpose(-1, -2)
    return do, dht


@pytest.mark.parametrize(
    'dtype,dim,cu_dtype,state,final,contribution',
    [
        (torch.float16, 64, torch.int32, True, True, 'both'),
        (torch.bfloat16, 64, torch.int64, False, True, 'both'),
        (torch.float16, 128, torch.int64, True, False, 'o'),
        (torch.bfloat16, 128, torch.int32, True, True, 'ht'),
        (torch.bfloat16, 128, torch.int64, True, True, 'both'),
        (torch.bfloat16, 64, torch.int32, True, True, 'o'),
        (torch.float16, 64, torch.int64, False, False, 'o'),
    ],
    ids=[
        'dtype-float16-dim-64-cu_dtype-int32-state-true-final-true-contribution-both',
        'dtype-bfloat16-dim-64-cu_dtype-int64-state-false-final-true-contribution-both',
        'dtype-float16-dim-128-cu_dtype-int64-state-true-final-false-contribution-o',
        'dtype-bfloat16-dim-128-cu_dtype-int32-state-true-final-true-contribution-ht',
        'dtype-bfloat16-dim-128-cu_dtype-int64-state-true-final-true-contribution-both',
        'dtype-bfloat16-dim-64-cu_dtype-int32-state-true-final-true-contribution-o',
        'dtype-float16-dim-64-cu_dtype-int64-state-false-final-false-contribution-o',
    ],
)
def test_training_replay_changes_layout_and_upstream(dtype, dim, cu_dtype, state, final, contribution):
    required = tuple(range(6 if state else 5))
    sample = _leaves(_inputs(dtype=dtype, dim=dim), required)
    cu = torch.tensor([0, 1, 64, 129, 193], dtype=cu_dtype, device=device)
    step = _training_step(state=state, final=final)
    replay = torch.npu.make_graphed_callables(step, sample + (cu,), allow_unused_input=True)
    layouts = [[0, 1, 64, 129, 193], [0, 1, 2, 3, 193], [0, 0, 65, 65, 129],
               [0, 0, 0, 0, 0], [0, 1, 2, 3, 193]]
    for seed, offsets in enumerate(layouts, 1):
        values = _inputs(dtype=dtype, dim=dim, seed=seed)
        for tensor in values[:5]:
            tensor[:, offsets[-1]:] = float('nan')
        inputs = _leaves(values, required)
        h0_before = inputs[5].detach().clone()
        current_cu = torch.tensor(offsets, dtype=cu_dtype, device=device)
        do, dht = _training_upstream(inputs, seed + 50, active_tokens=offsets[-1])
        actual = _training_snapshot(replay, inputs, current_cu, do, dht, final, contribution)
        uncaptured = _training_snapshot(step, _leaves(values, required), current_cu, do, dht, final, contribution)
        _assert_training(actual, uncaptured, exact=True)

        def reference(*args):
            result = _training_reference(args[:-1], offsets, state=state, final=final)
            return result if final else result[0]

        safe_do = do.clone()
        safe_do[:, offsets[-1]:] = 0
        expected = _training_snapshot(reference, _leaves(values, required), current_cu, safe_do, dht, final, contribution)
        _assert_training(actual, expected)
        if offsets[-1] == 193 and all(a < b for a, b in zip(offsets, offsets[1:])):
            eager = _training_step(state=state, final=final, use_graph=False)
            expected = _training_snapshot(eager, _leaves(values, required), current_cu, do, dht, final, contribution)
            _assert_training(actual, expected)
        for name in ('o', 'dq', 'dk', 'dv', 'dg', 'dbeta'):
            assert torch.count_nonzero(actual[name][:, offsets[-1]:]) == 0, name
        torch.testing.assert_close(inputs[5], h0_before, rtol=0, atol=0)
        if state:
            for n, (start, end) in enumerate(zip(offsets, offsets[1:])):
                if start == end:
                    expected_dh0 = dht[n] if final and contribution != 'o' else torch.zeros_like(dht[n])
                    torch.testing.assert_close(actual['dh0'][n], expected_dh0, rtol=0, atol=0)


@pytest.mark.parametrize(
    'required',
    [(0,), (1,), (2,), (3,), (4,), (5,), (0, 3, 5)],
    ids=[
        'required-0',
        'required-1',
        'required-2',
        'required-3',
        'required-4',
        'required-5',
        'required-0-3-5',
    ],
)
def test_training_requires_grad_subsets_and_strided_upstream(required):
    values = _inputs(capacity=17, heads=1)
    sample = _leaves(values, required)
    offsets = [0, 0, 1, 17, 17]
    cu = torch.tensor(offsets, device=device)
    step = _training_step()
    replay = torch.npu.make_graphed_callables(step, sample + (cu,), allow_unused_input=True)
    inputs = _leaves(_inputs(capacity=17, heads=1, seed=1), required)
    do, dht = _training_upstream(inputs, 8, strided=True)
    actual = _training_snapshot(replay, inputs, cu, do, dht)

    def reference(*args):
        return _training_reference(args[:-1], offsets)

    expected = _training_snapshot(reference, _leaves(inputs, required), cu, do, dht)
    _assert_training(actual, expected)
    assert all(x.grad is None for x in inputs)
    assert {name for name in actual if name.startswith('d')} == {f'd{_INPUT_NAMES[i]}' for i in required}


@pytest.mark.parametrize(
    'length',
    [1, 17, 33, 63, 64, 65, 129],
    ids=[
        'length-1',
        'length-17',
        'length-33',
        'length-63',
        'length-64',
        'length-65',
        'length-129',
    ],
)
def test_training_chunk_boundaries(length):
    inputs = _leaves(_inputs(capacity=length, heads=1))
    offsets = [0, 0, length, length, length]
    cu = torch.tensor(offsets, device=device)
    do, dht = _training_upstream(inputs, 4)
    actual = _training_snapshot(_training_step(), inputs, cu, do, dht)

    def reference(*args):
        return _training_reference(args[:-1], offsets)

    expected = _training_snapshot(reference, _leaves(inputs), cu, do, dht)
    _assert_training(actual, expected)


@pytest.mark.parametrize('tiled', [False, True], ids=['tiled-false', 'tiled-true'])
def test_training_long_bf16_matches_eager(monkeypatch, tiled):
    if tiled:
        output = import_module('fla.ops.common.backends.triton_ascend.chunk_o')
        monkeypatch.setattr(output, '_get_dv_full_tiles', lambda **kwargs: None)
        monkeypatch.setattr(output, '_get_dqkwg_full_tiles', lambda **kwargs: None)
    values = _inputs(dtype=torch.bfloat16, dim=128, capacity=1024, heads=8)
    sample = _leaves(values)
    cu = torch.tensor([0, 256, 512, 768, 1024], device=device)
    step = _training_step()
    replay = torch.npu.make_graphed_callables(step, sample + (cu,), allow_unused_input=True)
    for seed, offsets in enumerate(([0, 1, 65, 129, 1024], [0, 0, 0, 0, 0], [0, 1, 65, 129, 1024]), 1):
        inputs = _leaves(_inputs(dtype=torch.bfloat16, dim=128, capacity=1024, heads=8, seed=seed))
        live_cu = torch.tensor(offsets, device=device)
        do, dht = _training_upstream(inputs, seed + 30)
        actual = _training_snapshot(replay, inputs, live_cu, do, dht)
        expected = _training_snapshot(step, _leaves(inputs), live_cu, do, dht)
        _assert_training(actual, expected, exact=True)

        def reference(*args):
            return _training_reference(args[:-1], offsets)

        expected = _training_snapshot(reference, _leaves(inputs), live_cu, do, dht)
        _assert_training(actual, expected)


@pytest.mark.parametrize('tiled', [False, True], ids=['tiled-false', 'tiled-true'])
@pytest.mark.parametrize('dim', [64, 128], ids=['dim-64', 'dim-128'])
@torch.no_grad()
def test_backward_kernels_skip_sentinel_holes(monkeypatch, tiled, dim):
    torch.manual_seed(42)
    output = import_module('fla.ops.common.backends.triton_ascend.chunk_o')
    wy = import_module('fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast')

    def properties():
        return {'num_aicore': 1, 'num_vectorcore': 1}

    monkeypatch.setattr(output, 'get_npu_properties', properties)
    monkeypatch.setattr(wy, 'get_npu_properties', properties)
    if tiled:
        monkeypatch.setattr(output, '_get_dv_full_tiles', lambda **kwargs: None)
        monkeypatch.setattr(output, '_get_dqkwg_full_tiles', lambda **kwargs: None)
    q, k, v, g, beta, _ = _inputs(capacity=66, heads=1, dim=dim)
    do, dw, du = [torch.randn_like(v) for _ in range(3)]
    a = torch.randn(1, 66, 1, 64, dtype=q.dtype, device=device) * 0.01
    for tensor in (q, k, v, g, beta, do, dw, du, a):
        tensor[:, 65:] = float('nan')
    cu = torch.tensor([0, 65], device=device)
    indices = torch.tensor([[0, 0], [0, 1]], device=device)
    holes = torch.tensor([[-1, 0], [0, 0], [-1, 0], [0, 1], [-1, 0]], device=device)
    h = torch.randn(1, 2, 1, dim, dim, dtype=q.dtype, device=device)
    dh = torch.randn_like(h)

    def run(index, states, state_grads):
        common = dict(cu_seqlens=cu, chunk_indices=index, use_graph=True)
        dv_local = output._chunk_bwd_dv_local_npu(q=q, k=k, do=do, g=g, **common)
        grads = output._chunk_bwd_dqkwg_npu(q=q, k=k, v=v, do=do, h=states, dh=state_grads, w=k, g=g, dv=du, **common)
        wy_grads = wy._prepare_wy_repr_bwd_npu(k=k, v=v, beta=beta, A=a, dw=dw, du=du, g=g, **common)
        return (dv_local,) + grads + wy_grads

    padded = []
    for states in (h, dh):
        expanded = torch.full((1, 5, 1, dim, dim), float('nan'), dtype=q.dtype, device=device)
        expanded[:, 1] = states[:, 0]
        expanded[:, 3] = states[:, 1]
        padded.append(expanded)
    for actual, expected in zip(run(holes, *padded), run(indices, h, dh)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.count_nonzero(actual[:, 65:]) == 0


def test_training_uncaptured_contexts_and_leaf_gradient_accumulation():
    offsets_a, offsets_b = [0, 1, 17, 65, 65], [0, 0, 33, 33, 65]
    cu_a, cu_b = [torch.tensor(offsets, device=device) for offsets in (offsets_a, offsets_b)]
    values_a, values_b = [_inputs(capacity=65, heads=1, seed=seed) for seed in (1, 2)]
    inputs_a, inputs_b = _leaves(values_a), _leaves(values_b)
    do_a, dht_a = _training_upstream(inputs_a, 20)
    do_b, dht_b = _training_upstream(inputs_b, 21)
    step = _training_step()
    outputs_a, outputs_b = step(*inputs_a, cu_a), step(*inputs_b, cu_b)

    def saved_a(*args):
        return outputs_a

    def saved_b(*args):
        return outputs_b

    actual_b = _training_snapshot(saved_b, inputs_b, cu_b, do_b, dht_b)
    actual_a = _training_snapshot(saved_a, inputs_a, cu_a, do_a, dht_a)
    for actual, values, offsets, cu, do, dht in (
        (actual_a, values_a, offsets_a, cu_a, do_a, dht_a),
        (actual_b, values_b, offsets_b, cu_b, do_b, dht_b),
    ):
        def reference(*args):
            return _training_reference(args[:-1], offsets)
        expected = _training_snapshot(reference, _leaves(values), cu, do, dht)
        _assert_training(actual, expected)

    accumulated = _leaves(values_a)
    for iteration in range(2):
        torch.autograd.backward(step(*accumulated, cu_a), (do_a, dht_a))
        for name, leaf in zip(_INPUT_NAMES, accumulated):
            torch.testing.assert_close(leaf.grad, actual_a[f'd{name}'] * (iteration + 1), rtol=0, atol=0)
    for leaf in accumulated:
        leaf.grad = None
    torch.autograd.backward(step(*accumulated, cu_a), (do_a, dht_a))
    for name, leaf in zip(_INPUT_NAMES, accumulated):
        torch.testing.assert_close(leaf.grad, actual_a[f'd{name}'], rtol=0, atol=0)


def test_training_replay_accumulates_into_owned_leaf_gradients():
    inputs = _leaves(_inputs(capacity=65, heads=2, dim=128))
    cu = torch.tensor([0, 1, 17, 64, 65], device=device)
    step = _training_step()
    replay = torch.npu.make_graphed_callables(step, inputs + (cu,), allow_unused_input=True)
    for leaf in inputs:
        leaf.grad = torch.zeros_like(leaf)
    pointers = [leaf.grad.data_ptr() for leaf in inputs]
    expected_sum = [torch.zeros_like(leaf) for leaf in inputs]
    for seed in (7, 11, 19):
        values = _inputs(capacity=65, heads=2, dim=128, seed=seed)
        with torch.no_grad():
            for leaf, value in zip(inputs, values):
                leaf.copy_(value)
        do, dht = _training_upstream(inputs, seed + 80)
        expected = _training_snapshot(step, _leaves(values), cu, do, dht)
        # a new accumulation window keeps the caller-owned storage.
        if seed == 19:
            for leaf, total in zip(inputs, expected_sum):
                leaf.grad.zero_()
                total.zero_()
        torch.autograd.backward(replay(*inputs, cu), (do, dht))
        for name, leaf, total, pointer in zip(_INPUT_NAMES, inputs, expected_sum, pointers):
            total.add_(expected[f'd{name}'])
            assert leaf.grad.data_ptr() == pointer
            torch.testing.assert_close(leaf.grad, total, rtol=0, atol=0)


@pytest.mark.parametrize('tiled', [False, True], ids=['tiled-false', 'tiled-true'])
def test_training_external_preprocessing_gradients(monkeypatch, tiled):
    if tiled:
        output = import_module('fla.ops.common.backends.triton_ascend.chunk_o')
        monkeypatch.setattr(output, '_get_dv_full_tiles', lambda **kwargs: None)
        monkeypatch.setattr(output, '_get_dqkwg_full_tiles', lambda **kwargs: None)
    torch.manual_seed(42)
    values = list(_inputs(capacity=65, heads=2, dim=128))
    values[3] = torch.randn_like(values[3])
    values[4] = torch.randn_like(values[4])
    raw = _leaves(values)
    offsets = [0, 1, 17, 64, 65]
    cu = torch.tensor(offsets, device=device)

    def prepare(args):
        q, k, v, g, beta, h0 = args
        q, k = [F.normalize(x.float(), dim=-1).to(x.dtype) for x in (q, k)]
        return q, k, v, F.logsigmoid(g) / 8, beta.sigmoid(), h0

    sample = _leaves(prepare(raw))
    replay = torch.npu.make_graphed_callables(_training_step(), sample + (cu,), allow_unused_input=True)

    def graph_step(*args):
        return replay(*prepare(args[:-1]), args[-1])

    def reference(*args):
        return _training_reference(prepare(args[:-1]), offsets)

    do, dht = _training_upstream(raw, 57)
    actual = _training_snapshot(graph_step, raw, cu, do, dht)
    expected = _training_snapshot(reference, _leaves(values), cu, do, dht)
    _assert_training(actual, expected)


def test_training_independent_graphs_are_isolated():
    cu_a = torch.tensor([0, 1, 17, 65, 65], device=device)
    cu_b = torch.tensor([0, 0, 33, 33, 65], device=device)
    step = _training_step()
    inputs_a, inputs_b = [_leaves(_inputs(capacity=65, heads=1, seed=seed)) for seed in (1, 2)]
    graph_a = torch.npu.make_graphed_callables(step, inputs_a + (cu_a,), allow_unused_input=True)
    graph_b = torch.npu.make_graphed_callables(step, inputs_b + (cu_b,), allow_unused_input=True)
    saved = None
    for replay, inputs, cu, seed in ((graph_a, inputs_a, cu_a, 8), (graph_b, inputs_b, cu_b, 9),
                                     (graph_a, inputs_a, cu_a, 8)):
        do, dht = _training_upstream(inputs, seed)
        actual = _training_snapshot(replay, inputs, cu, do, dht)
        expected = _training_snapshot(step, _leaves(inputs), cu, do, dht)
        _assert_training(actual, expected, exact=True)
        if replay is graph_a:
            if saved is not None:
                _assert_training(actual, saved, exact=True)
            saved = actual


def test_training_direct_capture_forward_backward():
    inputs = _leaves(_inputs(capacity=65, heads=1))
    cu = torch.tensor([0, 1, 17, 64, 65], device=device)
    do, dht = _training_upstream(inputs, 10)
    step = _training_step()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            torch.autograd.backward(step(*inputs, cu), (do, dht))
    torch.npu.current_stream().wait_stream(stream)
    for leaf in inputs:
        leaf.grad.zero_()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for leaf in inputs:
            leaf.grad.zero_()
        o, ht = step(*inputs, cu)
        torch.autograd.backward((o, ht), (do, dht))
    for seed, offsets in ((1, [0, 0, 1, 65, 65]), (2, [0, 0, 0, 0, 0]), (3, [0, 1, 17, 64, 65])):
        values = _inputs(capacity=65, heads=1, seed=seed)
        live_do, live_dht = _training_upstream(values, seed + 20)
        with torch.no_grad():
            for dst, src in zip(inputs, values):
                dst.copy_(src)
            cu.copy_(torch.tensor(offsets, device=device))
            do.copy_(live_do)
            dht.copy_(live_dht)
        graph.replay()
        actual = {'o': o.detach().clone(), 'ht': ht.detach().clone()}
        actual.update((f'd{name}', leaf.grad.detach().clone()) for name, leaf in zip(_INPUT_NAMES, inputs))
        expected = _training_snapshot(step, _leaves(values), cu, live_do, live_dht)
        _assert_training(actual, expected, exact=True)


def test_graph_backward_rejects_higher_order_gradients():
    inputs = _leaves(_inputs(capacity=1, heads=1))
    cu = torch.tensor([0, 0, 1, 1, 1], device=device)
    o, _ = _call(inputs, cu)
    with pytest.raises((RuntimeError, NotImplementedError), match='higher|create_graph|first.order'):
        torch.autograd.grad(o, inputs, torch.ones_like(o), create_graph=True)
