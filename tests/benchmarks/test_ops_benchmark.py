# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import os

import pytest
import torch
import triton

from benchmarks.ops import run
from benchmarks.ops.registry import generate_inputs, get_op
from fla import utils


@pytest.mark.parametrize('custom_shapes', [False, True])
@pytest.mark.parametrize('compile', [False, True])
def test_benchmark(monkeypatch, custom_shapes, compile):
    shape = {'B': 1, 'T': 3, 'H': 1, 'D': 7, 'N': 2}
    defaults = {'default': {**shape, 'T': 5, 'N': 4}}
    monkeypatch.setattr(run.get_op('naive_hc_pre'), 'default_shapes', defaults)
    monkeypatch.setattr(utils, 'device_name', 'cpu')
    monkeypatch.setattr(run, '_device_synchronize', lambda device: None)
    graphs = []
    torch_compile = torch.compile

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def compile_fn(fn, fullgraph):
        return torch_compile(fn, fullgraph=fullgraph, backend=compiler)

    def do_bench(fn, **kwargs):
        fn()
        return [1.0, 0.8, 1.2]

    monkeypatch.setattr(torch, 'compile', compile_fn)
    monkeypatch.setattr(triton.testing, 'do_bench', do_bench)
    results = run.benchmark_op(
        op_name='naive_hc_pre',
        shapes={'custom': shape} if custom_shapes else None,
        compile=compile,
    )
    expected = shape if custom_shapes else defaults['default']
    assert len(results) == 2
    assert [row['mode'] for row in results] == ['fwd', 'fwdbwd']
    assert all({key: row[key] for key in expected} == expected for row in results)
    assert all(row['op'] == ('naive_hc_pre[compile]' if compile else 'naive_hc_pre') for row in results)
    assert bool(graphs) == compile
    assert all(row['median_ms'] == 1.0 for row in results)


@pytest.mark.parametrize('backend', ['triton', 'gluon', 'tilelang'])
def test_backend(monkeypatch, backend):
    for name in ('FLA_GLUON', 'FLA_TILELANG', 'FLA_HC_GLUON', 'FLA_HC_TILELANG'):
        monkeypatch.setenv(name, '1')
    run.benchmark_op(op_name='fused_hc_pre', shapes={}, backend=backend)
    assert os.environ['FLA_GLUON'] == '0'
    assert os.environ['FLA_TILELANG'] == '0'
    assert os.environ['FLA_HC_GLUON'] == str(int(backend == 'gluon'))
    assert os.environ['FLA_HC_TILELANG'] == str(int(backend == 'tilelang'))


@pytest.mark.parametrize('name,shape,dtype', [
    ('hc_pre', (2, 3, 65), torch.bfloat16),
    ('ihc_post', (2, 3, 4, 65), torch.bfloat16),
    ('mhc_post', (2, 3, 4, 65), torch.bfloat16),
    ('sinkhorn', (2, 3, 4, 4), torch.float32),
])
def test_hc_inputs(name, shape, dtype):
    config = get_op('naive_' + name)
    inputs = generate_inputs(config=config, B=2, T=3, H=1, D=65, N=4, dtype=torch.bfloat16, device='cpu')
    op = getattr(importlib.import_module(config.import_path), config.func_name or config.name)
    out = op(**inputs, **config.extra_kwargs)
    assert out.shape == shape
    assert out.dtype == dtype
    grads = torch.autograd.grad(outputs=out.sum(), inputs=tuple(inputs.values()))
    assert all(torch.isfinite(grad).all() for grad in grads)
    if name in ('ihc_post', 'mhc_post'):
        assert ('mixing' in inputs) == (name == 'mhc_post')
    for key in ('pre', 'post', 'mixing', 'logits'):
        if key in inputs:
            assert inputs[key].dtype == torch.float32


@pytest.mark.parametrize('axis', ['L', 'N'])
def test_results(capsys, axis):
    row = {'op': 'residual', 'mode': 'fwd', 'B': 1, 'T': 3, 'H': 1, 'D': 7, 'median_ms': 1.0}
    results = [{**row, axis: n} for n in (2, 4)]
    run.print_results_table(results=results, baseline=[{**r, 'median_ms': 2.0} for r in results])
    output = capsys.readouterr().out
    header = next(line for line in output.splitlines() if line.strip().startswith('fwd'))
    assert header.split()[:6] == ['fwd', axis, 'B', 'T', 'H', 'D']
    rows = [line.split() for line in output.splitlines() if 'residual' in line]
    assert [row[:5] for row in rows] == [['2', '1', '3', '1', '7'], ['4', '1', '3', '1', '7']]
    assert output.count('2.00x') == 2
