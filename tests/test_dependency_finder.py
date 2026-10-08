# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from pathlib import Path

import pytest

from scripts.find_dependent_tests import DependencyFinder, find_backend_op_files


def _dispatch_source(owner, syntax):
    operation = owner.removesuffix('.backends').removeprefix('fla.ops.').removeprefix('fla.')
    source = owner
    symbol = 'dispatch'
    decorator = 'dispatch'
    if syntax in ('central', 'legacy-central', 'central-call'):
        source = 'fla.ops.backends' if syntax == 'legacy-central' else 'fla.backends'
        decorator = f'dispatch({operation!r})'
    elif syntax == 'named-registry':
        symbol = f'{operation.rsplit(".", maxsplit=1)[-1]}_registry as registry'
        decorator = 'registry.dispatch'
    if syntax == 'central-call':
        return f'from {source} import dispatch as route\ndef compute(x): return x\ncompute = route({operation!r})(compute)\n'
    return f'from {source} import {symbol}\n@{decorator}\ndef compute(x): return x\n'


@pytest.mark.parametrize('syntax', ['central', 'legacy-central', 'local-dispatch', 'named-registry', 'central-call'])
@pytest.mark.parametrize(
    ('changed', 'expected'),
    [
        ('fla/backends.py', {'kda', 'gdn2', 'dplr', 'modules'}),
        ('fla/ops/backends/__init__.py', {'kda', 'gdn2', 'dplr', 'modules'}),
        ('fla/ops/kda/backends/__init__.py', {'kda'}),
        ('fla/ops/kda/backends/triton_ascend/chunk.py', {'kda'}),
        ('fla/ops/gdn2/backends/triton_ascend/__init__.py', {'gdn2'}),
        ('fla/ops/generalized_delta_rule/dplr/backends/__init__.py', {'dplr'}),
        ('fla/modules/backends/triton_ascend/activations.py', {'modules'}),
        ('fla/utils/_compat.py', set()),
    ],
)
def test_backend_changes_follow_dispatch_ownership(tmp_path, syntax, changed, expected):
    directory = tmp_path / 'fla'
    directory.mkdir()
    owners = {
        'kda': 'fla.ops.kda.backends',
        'gdn2': 'fla.ops.gdn2.backends',
        'dplr': 'fla.ops.generalized_delta_rule.dplr.backends',
        'modules': 'fla.modules.backends',
    }
    for name, owner in owners.items():
        (directory / f'{name}.py').write_text(_dispatch_source(owner, syntax))
    (directory / 'unrelated.py').write_text(
        'from other.backends import dispatch, unrelated_registry\n'
        'from fla.utils import helper_registry\n'
        'from fla.ops.kda.backends import helper\n'
        "@dispatch('kda')\n"
        'def compute(x): return x\n'
    )
    paths = find_backend_op_files([changed], tmp_path)
    assert set(paths) == {f'fla/{name}.py' for name in expected}


@pytest.mark.parametrize('syntax', ['central', 'legacy-central', 'local-dispatch', 'named-registry', 'central-call'])
@pytest.mark.parametrize('changed', ['__init__.py', 'tilelang/__init__.py', 'tilelang/parallel.py'])
def test_backend_changes_follow_shared_adapter_imports(tmp_path, syntax, changed):
    sources = {
        'fla/ops/common/backends/__init__.py': 'from fla.ops.common.backends.tilelang import TileLangBackend\n',
        'fla/ops/common/backends/tilelang/__init__.py': (
            "from fla.backends import register_backend\n"
            "@register_backend('common', 'attn')\n"
            'class TileLangBackend: pass\n'
        ),
        'fla/ops/attn/backends/__init__.py': 'from fla.ops.common.backends.tilelang import TileLangBackend\n',
        'fla/ops/other/backends/__init__.py': 'from fla.ops.attn.backends import TileLangBackend\n',
    }
    for operation in ['common', 'attn', 'other', 'unrelated']:
        sources[f'fla/ops/{operation}/parallel.py'] = _dispatch_source(f'fla.ops.{operation}.backends', syntax)
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    paths = find_backend_op_files([f'fla/ops/common/backends/{changed}'], tmp_path)
    assert set(paths) == {f'fla/ops/{operation}/parallel.py' for operation in ['common', 'attn', 'other']}


def test_dependencies_follow_reexports_and_aliases(tmp_path):
    sources = {
        'fla/modules/norm/ops.py': 'def l2norm(x): return x\nl2_norm = l2norm\n',
        'fla/modules/norm/unrelated.py': 'def unrelated(x): return x\n',
        'fla/modules/norm/__init__.py': (
            'from .ops import l2_norm as normalized\nfrom .unrelated import unrelated\n'
        ),
        'fla/modules/l2norm.py': 'from fla.modules.norm import normalized as l2_norm\n',
        'tests/test_norm.py': 'from fla.modules.l2norm import l2_norm as normalize\n',
        'tests/test_unrelated.py': 'from fla.modules.norm import unrelated\n',
    }
    for name, source in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

    finder = DependencyFinder([tmp_path / 'fla'], tmp_path / 'tests', tmp_path)
    tests = finder.find_dependent_tests([tmp_path / 'fla/modules/norm/ops.py'])
    assert tests == {str(tmp_path / 'tests/test_norm.py')}


@pytest.fixture(scope='module')
def repository_finder():
    root = Path(__file__).resolve().parents[1]
    return DependencyFinder([root / 'fla'], root / 'tests', root)


@pytest.mark.parametrize(('changed', 'expected', 'unrelated'), [
    (
        'fla/ops/attn/backends/tilelang/parallel_attn_fwd.py',
        'fla/ops/attn/parallel.py',
        'fla/ops/common/chunk_o.py',
    ),
    (
        'fla/ops/common/backends/tilelang/chunk_bwd.py',
        'fla/ops/common/chunk_o.py',
        'fla/ops/attn/parallel.py',
    ),
])
def test_backend_changes_preserve_attention_and_common_ownership(changed, expected, unrelated):
    root = Path(__file__).resolve().parents[1]
    paths = find_backend_op_files([changed], root)
    assert expected in paths
    assert unrelated not in paths


@pytest.mark.parametrize(
    ('changed', 'expected'),
    [
        ('fla/ops/attn/backends/tilelang/parallel_attn_fwd.py', {'tests/ops/test_attn.py'}),
        ('fla/ops/attn/backends/tilelang/parallel_attn_bwd.py', {'tests/ops/test_attn.py'}),
    ],
)
def test_backend_changes_select_original_test_files(repository_finder, changed, expected):
    root = repository_finder.project_root
    changed_files = [changed, *find_backend_op_files([changed], root)]
    tests = repository_finder.find_dependent_tests([root / path for path in changed_files])
    assert {str(root / path) for path in expected} <= tests
