# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Host-only checks of graph backward launch partitioning."""

import ast
from collections import Counter
from itertools import product
from math import prod
from pathlib import Path

import pytest


def _load_launch_helpers(max_grid):
    root = Path(__file__).resolve().parents[2]
    namespace = {'ASCEND_MAX_GRID_DIM': max_grid}
    sources = {
        root / 'fla/utils/ascend_ub_manager.py': {'max_grid_axis_chunks'},
        root / 'fla/ops/common/backends/triton_ascend/chunk_o.py': {
            '_launch_bwd_2d_kernel', '_launch_bwd_3d_kernel',
        },
    }
    # load the production host functions without importing accelerator dependencies.
    for path, names in sources.items():
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in functions} == names
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('dimensions,nk', [(2, 1), (3, 2)], ids=['dimensions-2-nk-1', 'dimensions-3-nk-2'])
@pytest.mark.parametrize(
    'max_grid,nt,bh_total',
    [(4, 7, 7), (5, 13, 2)],
    ids=[
        'max_grid-4-nt-7-bh_total-7',
        'max_grid-5-nt-13-bh_total-2',
    ],
)
def test_graph_backward_partition_preserves_global_chunk_addresses(dimensions, nk, max_grid, nt, bh_total):
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
    if dimensions == 2:
        helpers['_launch_bwd_2d_kernel'](**common)
    else:
        helpers['_launch_bwd_3d_kernel'](nk=nk, **common)

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
