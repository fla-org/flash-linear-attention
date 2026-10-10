# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from pathlib import Path

import pytest

from scripts.find_dependent_tests import DependencyFinder, find_backend_op_files


@pytest.fixture
def backend_project(tmp_path):
    directory = tmp_path / 'fla'
    directory.mkdir()
    sources = {
        'kda': "from fla.backends import dispatch\n@dispatch('kda')\ndef compute(x): return x\n",
        'gdn2': "from fla.backends import dispatch as route\n@route('gdn2')\ndef compute(x): return x\n",
        'dplr': (
            "from fla.backends import dispatch\n"
            "@dispatch('generalized_delta_rule.dplr')\ndef compute(x): return x\n"
        ),
        'modules': "from fla.backends import dispatch\n@dispatch('modules.activations')\ndef compute(x): return x\n",
        'unrelated': (
            'from other.backends import dispatch\n'
            'from fla.ops.kda.backends import helper\n'
            "@dispatch('kda')\ndef compute(x): return x\n"
        ),
    }
    for name, source in sources.items():
        (directory / f'{name}.py').write_text(source)
    owner = directory / 'modules/activations'
    (owner / 'triton_ascend').mkdir(parents=True)
    (owner / 'backends.py').write_text('from fla.modules.activations.triton_ascend import TritonAscendBackend\n')
    return tmp_path


@pytest.mark.parametrize(
    ('changed', 'expected'),
    [
        ('fla/backends.py', {'kda', 'gdn2', 'dplr', 'modules'}),
        ('fla/ops/kda/backends/__init__.py', {'kda'}),
        ('fla/ops/kda/backends/triton_ascend/chunk.py', {'kda'}),
        ('fla/ops/gdn2/backends/triton_ascend/__init__.py', {'gdn2'}),
        ('fla/ops/generalized_delta_rule/dplr/backends/__init__.py', {'dplr'}),
        ('fla/modules/activations/triton_ascend/ops.py', {'modules'}),
        ('fla/modules/activations/triton_ascend/__init__.py', {'modules'}),
        ('fla/modules/activations/backends.py', {'modules'}),
        ('fla/utils/_compat.py', set()),
    ],
)
def test_backend_changes_follow_dispatch_ownership(backend_project, changed, expected):
    paths = find_backend_op_files([changed], backend_project)
    assert set(paths) == {f'fla/{name}.py' for name in expected}


@pytest.mark.parametrize(
    ('backend_file', 'expected'),
    [
        ('fla/ops/gdn2/backends/triton_ascend/chunk_bwd.py', {'fla/kda.py', 'fla/gdn2.py'}),
        ('fla/modules/activations/triton_ascend/ops.py', {'fla/kda.py', 'fla/modules.py'}),
    ],
)
def test_backend_changes_follow_shared_kernel_imports(backend_project, backend_file, expected):
    path = backend_project / backend_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('from fla.ops.kda.backends.triton_ascend.chunk_bwd import chunk_kda_bwd_kernel_wy_k_part_npu\n')

    paths = find_backend_op_files(['fla/ops/kda/backends/triton_ascend/chunk_bwd.py'], backend_project)
    assert set(paths) == expected


def test_dependencies_follow_reexports_and_aliases(tmp_path):
    sources = {
        'fla/modules/__init__.py': "_MODULE_ALIASES = {'l2norm': ('norm.ops',)}\n",
        'fla/modules/norm/ops.py': 'def l2norm(x): return x\nl2_norm = l2norm\n',
        'fla/modules/norm/unrelated.py': 'def unrelated(x): return x\n',
        'fla/modules/norm/__init__.py': 'from .ops import l2_norm as normalized\nfrom .unrelated import unrelated\n',
        'tests/test_alias.py': 'from fla.modules.l2norm import l2norm\n',
        'tests/test_norm.py': 'from fla.modules.l2norm import l2_norm as normalize\n',
        'tests/test_unrelated.py': 'from fla.modules.norm import unrelated\n',
    }
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    finder = DependencyFinder(search_dirs=[tmp_path / 'fla'], test_dir=tmp_path / 'tests', project_root=tmp_path)
    tests = finder.find_dependent_tests([tmp_path / 'fla/modules/norm/ops.py'])
    assert tests == {str(tmp_path / 'tests/test_norm.py'), str(tmp_path / 'tests/test_alias.py')}


@pytest.mark.parametrize('backend_file', ['triton_ascend.py', 'triton_ascend/ops.py', 'triton_ascend/__init__.py'])
def test_dependencies_follow_backend_registration(tmp_path, backend_file):
    backend_module = 'fla.modules.rotary.triton_ascend' + ('.ops' if '/' in backend_file else '')
    backend_package, backend_name = backend_module.rsplit('.', 1)
    sources = {
        'fla/modules/rotary/ops.py': (
            'from fla.backends import dispatch\n'
            '@dispatch\n'
            'def rotary(x): return x\n'
        ),
        'fla/modules/rotary/__init__.py': (
            'from fla.modules.rotary.ops import rotary\n'
            f'from {backend_package} import {backend_name}\n'
        ),
        backend_module.replace('.', '/') + '.py': (
            'from fla.backends import TritonAscendBackend, register\n'
            'from fla.modules.rotary.ops import rotary\n'
            '@register(rotary, backend=TritonAscendBackend)\n'
            'def rotary_npu(x): return x\n'
        ),
        'tests/test_rotary.py': 'from fla.modules.rotary.ops import rotary\n',
    }
    if '/' in backend_file:
        sources['fla/modules/rotary/triton_ascend/__init__.py'] = ''
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    changed = f'fla/modules/rotary/{backend_file}'
    consumers = find_backend_op_files([changed], tmp_path)
    assert consumers == ['fla/modules/rotary/ops.py']
    assert find_backend_op_files(['fla/backends.py'], tmp_path) == consumers
    finder = DependencyFinder(search_dirs=[tmp_path / 'fla'], test_dir=tmp_path / 'tests', project_root=tmp_path)
    tests = finder.find_dependent_tests([tmp_path / path for path in [changed, *consumers]])
    assert tests == {str(tmp_path / 'tests/test_rotary.py')}


@pytest.fixture(scope='module')
def repository_finder():
    root = Path(__file__).resolve().parents[1]
    return DependencyFinder(search_dirs=[root / 'fla'], test_dir=root / 'tests', project_root=root)


@pytest.mark.parametrize(
    ('changed', 'expected', 'unrelated'),
    [
        ('fla/ops/attn/backends/tilelang/parallel.py', 'fla/ops/attn/parallel.py', 'fla/ops/common/chunk_o.py'),
        ('fla/ops/common/backends/tilelang/chunk_bwd.py', 'fla/ops/common/chunk_o.py', 'fla/ops/attn/parallel.py'),
        ('fla/ops/kda/backends/triton_ascend/chunk_bwd.py', 'fla/ops/gdn2/chunk_bwd.py', 'fla/ops/attn/parallel.py'),
        ('fla/ops/kda/backends/flash_kda.py', 'fla/ops/kda/chunk.py', 'fla/ops/attn/parallel.py'),
    ],
)
def test_backend_changes_follow_repository_ownership(changed, expected, unrelated):
    root = Path(__file__).resolve().parents[1]
    paths = find_backend_op_files([changed], root)
    assert expected in paths
    assert unrelated not in paths


@pytest.mark.parametrize(
    ('changed', 'expected'),
    [
        ('fla/modules/norm/triton_ascend/layernorm.py', {'tests/modules/test_layernorm.py'}),
        ('fla/modules/norm/triton_ascend/__init__.py', {'tests/modules/test_l2norm.py'}),
        ('fla/modules/norm/triton_ascend/l2norm.py', {'tests/modules/test_l2norm.py'}),
        ('fla/modules/norm/triton_ascend/fused_norm_gate.py', {'tests/modules/test_layernorm_gated.py'}),
        ('fla/modules/causal_conv1d/backends/__init__.py', {'tests/modules/test_conv.py'}),
        ('fla/modules/causal_conv1d/backends/gluon.py', {'tests/modules/test_conv.py'}),
        pytest.param('fla/modules/causal_conv1d/backends/cuda.py', {'tests/modules/test_conv.py'}, id='cuda-module'),
        ('fla/modules/fused_cross_entropy/ops.py', {'tests/modules/test_cross_entropy.py', 'tests/modules/test_l2warp.py'}),
        ('fla/ops/attn/backends/tilelang/parallel.py', {'tests/ops/test_attn.py'}),
        ('fla/ops/kda/backends/triton_ascend/chunk_bwd.py', {'tests/ops/test_kda.py', 'tests/ops/test_gdn2.py'}),
        pytest.param('fla/modules/activations/triton_ascend.py', {'tests/modules/test_activation.py'}, id='ascend-module'),
        (
            'fla/ops/simple_gla/backends/triton_ascend/fused_chunk.py',
            {'tests/ops/test_simple_gla.py', 'tests/ops/test_retention.py'},
        ),
        (
            'fla/ops/simple_gla/backends/triton_ascend/parallel.py',
            {'tests/ops/test_simple_gla.py', 'tests/ops/test_retention.py'},
        ),
        ('fla/ops/simple_gla/backends/triton_ascend/fused_recurrent.py', {
            'tests/ops/test_simple_gla.py', 'tests/ops/test_retention.py',
        }),
        ('fla/ops/common/backends/triton_ascend/chunk_o.py', {'tests/ops/test_simple_gla.py', 'tests/ops/test_retention.py'}),
    ],
)
def test_backend_changes_select_original_test_files(repository_finder, changed, expected):
    root = repository_finder.project_root
    changed_files = [changed, *find_backend_op_files([changed], root)]
    tests = repository_finder.find_dependent_tests([root / path for path in changed_files])
    assert {str(root / path) for path in expected} <= tests
