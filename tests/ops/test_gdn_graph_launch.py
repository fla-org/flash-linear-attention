# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Host-only checks of graph backward launch partitioning."""

from __future__ import annotations

import ast
import inspect
from collections import Counter
from itertools import product
from math import prod
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_launch_helpers(max_grid):
    root = Path(__file__).resolve().parents[2]
    namespace = {'ASCEND_MAX_GRID_DIM': max_grid, '_NUM_WARPS': 4}
    sources = {
        root / 'fla/utils/ascend_ub_manager.py': {'max_grid_axis_chunks'},
        root / 'fla/ops/common/backends/triton_ascend/chunk_o.py': {
            '_launch_bwd_2d_kernel', '_launch_bwd_3d_kernel',
        },
        root / 'fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py': {'_launch_wy_kernel'},
        root / 'fla/ops/utils/triton_ascend/solve_tril.py': {'_launch_solve_tril_kernel'},
    }
    # load the production host functions without importing accelerator dependencies.
    for path, names in sources.items():
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in functions} == names
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize(
    'launcher,nk,max_grid,nt,bh_total',
    [('bwd2d', 1, 4, 7, 7), ('bwd3d', 2, 4, 7, 7), ('bwd3d', 2, 5, 13, 2),
     ('wy', 1, 4, 7, 7), ('solve', 1, 5, 13, 2)],
    ids=['2d-head-splits', '3d-key-and-head-splits', '3d-chunk-tail', 'wy-head-splits', 'solve-chunk-tail'],
)
def test_graph_backward_partition_preserves_global_chunk_addresses(launcher, nk, max_grid, nt, bh_total):
    helpers = _load_launch_helpers(max_grid)
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                launches.append((grid, kwargs.copy()))
            return launch

    indices = object()
    cu = object()
    kwargs = {'cu_seqlens': cu, 'chunk_indices': indices, 'USE_GRAPH': True}
    common = dict(kernel=Kernel(), nt=nt, bh_total=bh_total, kernel_kwargs=kwargs)
    if launcher == 'bwd2d':
        helpers['_launch_bwd_2d_kernel'](**common)
    elif launcher == 'bwd3d':
        helpers['_launch_bwd_3d_kernel'](nk=nk, **common)
    else:
        name = '_launch_wy_kernel' if launcher == 'wy' else '_launch_solve_tril_kernel'
        common['NT'] = common.pop('nt')
        helpers[name](**common)

    tasks = Counter()
    chunk_spans = {}
    for grid, arguments in launches:
        assert prod(grid) <= max_grid
        assert arguments['chunk_indices'] is indices
        assert arguments['cu_seqlens'] is cu
        k_index = arguments.get('K_OFFSET', 0)
        nt_offset, bh_offset = arguments['NT_OFFSET'], arguments['BH_OFFSET']
        nt_length, bh_length = grid[-2:]
        chunk_spans[nt_offset] = nt_length
        tasks.update(product([k_index], range(nt_offset, nt_offset + nt_length), range(bh_offset, bh_offset + bh_length)))

    assert tasks == Counter(product(range(nk), range(nt), range(bh_total)))
    cursor = 0
    for offset, length in sorted(chunk_spans.items()):
        assert offset == cursor
        cursor += length
    assert cursor == nt


class _Tensor:
    def __init__(self, shape, *, dtype='float16', device='npu', zero=False):
        self.shape, self.dtype, self.device, self.zero = tuple(shape), dtype, device, zero

    def __len__(self):
        return self.shape[0]

    def new_empty(self, *shape, **kwargs):
        return _Tensor(shape, dtype=kwargs.get('dtype', self.dtype), device=self.device)

    def new_zeros(self, *shape, **kwargs):
        return self.new_empty(*shape, **kwargs).zero_()

    def zero_(self):
        self.zero = True
        return self

    def contiguous(self):
        return self

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return _Tensor(shape, dtype=self.dtype, device=self.device, zero=self.zero)

    def sum(self, dim):
        return _Tensor(self.shape[:dim] + self.shape[dim+1:], dtype=self.dtype, device=self.device, zero=self.zero)


def _load_graph_launchers(capturing, tiled):
    root = Path(__file__).resolve().parents[2]
    launches = []

    def forbid_host_metadata(*args, **kwargs):
        raise AssertionError('graph launches must not inspect or repartition live sequence lengths')

    def like(tensor, **kwargs):
        return _Tensor(tensor.shape, dtype=kwargs.get('dtype', tensor.dtype), device=tensor.device)

    namespace = _load_launch_helpers(max_grid=64)
    namespace.update(
        torch=SimpleNamespace(
            empty=lambda *shape, **kw: _Tensor(shape, **kw), zeros=lambda *shape, **kw: _Tensor(shape, **kw).zero_(),
            empty_like=like, zeros_like=lambda x, **kw: like(x, **kw).zero_(), float32='float32', float='float32',
            permute=lambda x, axes: _Tensor([x.shape[i] for i in axes], dtype=x.dtype),
            npu=SimpleNamespace(is_current_stream_capturing=lambda: capturing),
        ),
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b, next_power_of_2=lambda n: 1 << (n-1).bit_length()),
        _recurrence_cu_seqlens_cpu=forbid_host_metadata, time_axis_needs_mask=forbid_host_metadata,
        _varlen_chunk_passes=forbid_host_metadata, prepare_chunk_indices=forbid_host_metadata,
        prepare_chunk_offsets=forbid_host_metadata,
        get_npu_properties=lambda: {'num_aicore': 2, 'num_vectorcore': 2},
        ascend_compile_kwargs=lambda: {'multibuffer': False},
        _FWD_H_COMPILE={}, _DHU_COMPILE={},
        _select_fwd_h_tiles=lambda *args, **kw: (64, 32),
        _select_bwd_dhu_tiles=lambda **kw: (64, 32),
        _prepare_fwd_g_gates=lambda g, **kw: (g, None, None),
        _prepare_fwd_gk_last_exp=lambda gk, **kw: (gk, None),
        _prepare_dhu_gate_tensors=lambda g, **kw: (g, None, None, None, False),
        _get_bk=lambda **kw: 64, _get_bc=lambda **kw: 32, _get_bv=lambda **kw: 32,
        _get_fwd_bk=lambda **kw: 64,
        _get_dv_full_tiles=lambda **kw: None if tiled else (64, 64),
        _get_dqkwg_full_tiles=lambda **kw: None if tiled else (64, 64),
        _get_hdh_tiles=lambda **kw: (32, 32), _g_npu_arg=lambda g, **kw: (g, False),
        get_wy_block_size=lambda bt, dim, mult, cap: min(dim, cap, 64),
    )
    sources = {
        'fla/ops/common/backends/triton_ascend/chunk_delta_h.py': {
            '_launch_core_grid', 'need_dma_mask', 'need_kv_dma_mask',
            'chunk_gated_delta_rule_fwd_h_npu', 'chunk_gated_delta_rule_bwd_dhu_npu',
        },
        'fla/ops/common/backends/triton_ascend/chunk_o.py': {
            'chunk_fwd_o_npu', 'chunk_bwd_dv_local_npu', 'chunk_bwd_dqkwg_npu',
        },
        'fla/ops/common/backends/triton_ascend/chunk_scaled_dot_kkt.py': {'chunk_scaled_dot_kkt_fwd_npu'},
        'fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py': {
            'prepare_gate_input', 'prepare_beta_input', 'allocate_time_buffer',
            'get_fwd_block_sizes', '_candidate_fwd_tiles', 'launch_wy_core_grid', 'launch_wy_kernel',
            '_launch_wy_core_full_tail', '_launch_wy_kernel_full_tail',
            'recompute_w_u_fwd_npu', 'prepare_wy_repr_bwd_npu',
        },
        'fla/ops/utils/triton_ascend/solve_tril.py': {'solve_tril_npu'},
    }

    class Kernel:
        def __init__(self, node, heuristics):
            self.name, self.heuristics = node.name, heuristics
            defaults = [inspect.Parameter.empty] * (len(node.args.args) - len(node.args.defaults))
            defaults += [ast.literal_eval(value) for value in node.args.defaults]
            self.signature = inspect.Signature([
                inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
                for arg, default in zip(node.args.args, defaults)
            ])

        def __getitem__(self, grid):
            def launch(**kwargs):
                options = {'num_warps', 'multibuffer', 'enable_ubuf_saving', 'unit_flag'}
                arguments = {name: value for name, value in kwargs.items() if name not in options}
                self.signature.bind(**{**self.heuristics, **arguments})
                launches.append((self.name, grid, arguments))
            return launch

    for path, names in sources.items():
        tree = ast.parse((root / path).read_text(encoding='utf-8'))
        functions = []
        dictionaries = {node.targets[0].id: node.value for node in tree.body
                        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Dict)}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                namespace[node.targets[0].id] = node.value.value
            if not isinstance(node, ast.FunctionDef):
                continue
            if any('triton.jit' in ast.unparse(d) for d in node.decorator_list):
                heuristics = {}
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == 'triton.heuristics':
                        value = decorator.args[0]
                        value = dictionaries[value.id] if isinstance(value, ast.Name) else value
                        heuristics.update({key.value: None for key in value.keys})
                namespace[node.name] = Kernel(node, heuristics)
            elif node.name in names:
                node.decorator_list = []
                functions.append(node)
        assert {node.name for node in functions} == names
        exec(compile(ast.Module(body=functions, type_ignores=[]), path, 'exec'), namespace)
    return namespace, launches


@pytest.mark.parametrize(
    'capturing,tiled,dim',
    [
        pytest.param(False, False, 64, id='d64-full-uncaptured-warmup'),
        pytest.param(True, True, 64, id='d64-tiled-capture'),
        pytest.param(True, False, 128, id='d128-full-capture'),
        pytest.param(False, True, 128, id='d128-tiled-uncaptured-warmup'),
    ],
)
def test_graph_launches_keep_live_metadata_and_bind_current_kernel_interfaces(capturing, tiled, dim):
    helpers, launches = _load_graph_launchers(capturing=capturing, tiled=tiled)
    q = k = v = _Tensor((1, 128, 2, dim))
    g, beta = _Tensor((1, 128, 2), dtype='float32'), _Tensor((1, 128, 2))
    cu, indices, offsets = _Tensor((5,)), _Tensor((6, 2)), _Tensor((5,))
    h0 = _Tensor((4, 2, dim, dim), dtype='float32')
    common = dict(cu_seqlens=cu, chunk_indices=indices, use_graph=True)
    A = helpers['chunk_scaled_dot_kkt_fwd_npu'](k=k, g=g, beta=beta, **common)
    Ai = helpers['solve_tril_npu'](A=A, output_dtype=k.dtype, **common)
    w, u = helpers['recompute_w_u_fwd_npu'](k=k, v=v, g=g, beta=beta, A=Ai, **common)
    h, v_new, ht = helpers['chunk_gated_delta_rule_fwd_h_npu'](
        k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=True, chunk_offsets=offsets, **common,
    )
    o = helpers['chunk_fwd_o_npu'](q=q, k=k, v=v_new, h=h, g=g, scale=0.25, chunk_offsets=offsets, **common)
    dv = helpers['chunk_bwd_dv_local_npu'](q=q, k=k, do=v, g=g, **common)
    dh, dh0, dv2 = helpers['chunk_gated_delta_rule_bwd_dhu_npu'](
        q=q, k=k, w=w, do=v, dv=dv, g=g, h0=h0, dht=ht, scale=0.25, chunk_offsets=offsets, **common,
    )
    dq, dk, dw, dg = helpers['chunk_bwd_dqkwg_npu'](
        q=q, k=k, v=v_new, do=v, h=h, dh=dh, w=w, g=g, dv=dv2, **common,
    )
    dk2, dv3, db, dg2 = helpers['prepare_wy_repr_bwd_npu'](k=k, v=v, beta=beta, A=Ai, dw=dw, du=dv2, g=g, **common)
    assert all(x.zero for x in (w, u, v_new, ht, o, dv, dh, dh0, dv2, dq, dk, dw, dg, dk2, dv3, db, dg2))
    recurrence_tasks = 4 * 2 * (dim // 32)
    for name, _, arguments in launches:
        if 'chunk_indices' in arguments:
            assert arguments['chunk_indices'] is indices
        assert arguments['cu_seqlens'] is cu
        if 'MASK_DMA' in arguments:
            assert arguments['MASK_DMA'] is True
        if 'MASK_T' in arguments:
            assert arguments['MASK_T'] is True
        if name in ('chunk_gated_delta_rule_h_fwd_kernel', 'chunk_gated_delta_rule_h_bwd_kernel'):
            assert arguments['task_num'] == recurrence_tasks
            assert arguments['chunk_offsets'] is offsets
        else:
            assert arguments['USE_GRAPH'] is True
    kv_launch = next(args for name, _, args in launches if name == 'prepare_wy_repr_bwd_kv_npu')
    assert kv_launch['dA'].zero
