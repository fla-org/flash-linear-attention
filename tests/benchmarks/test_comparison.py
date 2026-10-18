# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
import subprocess
import sys

import pytest

from benchmarks import compare, comparison, discovery
from benchmarks.ops.registry import list_ops


@pytest.fixture
def benchmark_repository(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    root.mkdir()

    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=root, text=True).strip()

    git('init', '-b', 'feature/current')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.com')
    (root / '.gitignore').write_text('__pycache__/\n')
    (root / 'fla').mkdir()
    (root / 'fla/__init__.py').write_text('VALUE = "baseline"\n')
    (root / 'benchmarks').mkdir()
    (root / 'benchmarks/__init__.py').write_text('')
    (root / 'benchmarks/worker.py').write_text('raise RuntimeError("old benchmark definitions")\n')
    git('add', '.')
    git('commit', '-m', 'Baseline')
    git('branch', 'baseline')
    (root / 'fla/__init__.py').write_text('VALUE = "head"\n')
    git('add', '.')
    git('commit', '-m', 'Head')
    (root / 'fla/__init__.py').write_text('VALUE = "dirty"\n')

    harness = tmp_path / 'harness/benchmarks'
    harness.mkdir(parents=True)
    (harness / '__init__.py').write_text('')
    (harness / 'worker.py').write_text(
        'import json, sys\n'
        'from pathlib import Path\n'
        'import fla\n'
        'if "--fail" in sys.argv:\n'
        '    raise RuntimeError("worker failed")\n'
        'output = Path(sys.argv[sys.argv.index("--json") + 1])\n'
        'output.write_text(json.dumps({"results": [{"source": fla.VALUE, "harness": "current"}], "machine_info": {}}))\n'
    )
    monkeypatch.setattr(comparison, '__file__', str(harness / 'comparison.py'))
    monkeypatch.setattr(comparison, 'find_project_root', lambda: str(root))
    return git


@pytest.mark.parametrize('ref', ['baseline', 'feature/current', None], ids=['base', 'checked-out-branch', 'working-tree'])
@pytest.mark.parametrize('fail', [False, True], ids=['success', 'worker-failure'])
def test_revision_isolation(benchmark_repository, ref, fail):
    git = benchmark_repository
    before = git('status', '--porcelain')
    worktrees = git('worktree', 'list', '--porcelain')
    arguments = ['--fail'] if fail else []
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            comparison.run_benchmark(module='benchmarks.worker', arguments=arguments, ref=ref)
    else:
        results, info = comparison.run_benchmark(module='benchmarks.worker', arguments=arguments, ref=ref)
        expected = {'baseline': 'baseline', 'feature/current': 'head', None: 'dirty'}[ref]
        assert results == [{'source': expected, 'harness': 'current'}]
        if ref is not None:
            assert info['git_label'] == f'{ref}[{git("rev-parse", ref)[:7]}]'
    assert git('status', '--porcelain') == before
    assert git('worktree', 'list', '--porcelain') == worktrees
    assert git('branch', '--show-current') == 'feature/current'


@pytest.mark.parametrize('allow_regression', [False, True], ids=['fail-on-regression', 'report-only'])
def test_compare_artifact(tmp_path, monkeypatch, allow_regression):
    output = tmp_path / 'benchmark_results.json'
    argv = ['compare', '--base', 'base', '--head', 'head', '--benchmark-ops', 'chunk_gla', '--output', str(output)]
    if allow_regression:
        argv.append('--no-fail-on-regression')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(compare, 'get_commit_sha', lambda ref: ref)
    monkeypatch.setattr(compare, 'clear_triton_cache', lambda: None)
    monkeypatch.setenv('FLA_BENCH_COOLDOWN_SEC', '0')
    calls = []

    def run_benchmarks(op_names, shapes, modes, backend=None, ref=None):
        calls.append((op_names, shapes, modes, ref))
        results = [{
            'op': 'chunk_gla', 'mode': 'fwd', 'B': 1, 'T': 64, 'H': 2, 'D': 16,
            'median_ms': 2.0 if ref == 'head' else 1.0, 'p20_ms': 1.0, 'p80_ms': 2.0,
        }]
        return results, {'gpu_name': 'test', 'git_label': ref}

    monkeypatch.setattr(compare, 'run_benchmarks', run_benchmarks)
    assert compare.main() == (0 if allow_regression else 1)
    data = json.loads(output.read_text())
    assert data.keys() == {
        'base_sha', 'head_sha', 'machine_info', 'base_results', 'head_results',
        'regressions', 'speedups', 'has_regression', 'has_speedup',
    }
    assert data['base_sha'] == 'base'
    assert data['head_sha'] == 'head'
    assert data['has_regression'] is True
    assert data['has_speedup'] is False
    assert data['regressions'][0]['change_pct'] == 100.0
    assert calls[0][:-1] == calls[1][:-1]
    assert [call[-1] for call in calls] == ['head', 'base']


@pytest.mark.parametrize('failure', ['exception', 'missing-results'])
def test_compare_failure_has_no_success_artifact(tmp_path, monkeypatch, failure):
    output = tmp_path / 'benchmark_results.json'
    monkeypatch.setattr(sys, 'argv', [
        'compare', '--benchmark-ops', 'chunk_gla', '--output', str(output), '--no-fail-on-regression',
    ])
    monkeypatch.setattr(compare, 'get_commit_sha', lambda ref: ref)
    monkeypatch.setattr(compare, 'clear_triton_cache', lambda: None)
    monkeypatch.setenv('FLA_BENCH_COOLDOWN_SEC', '0')

    def run_benchmarks(**kwargs):
        if failure == 'exception':
            raise subprocess.CalledProcessError(1, ['worker'])
        return [], {}

    monkeypatch.setattr(compare, 'run_benchmarks', run_benchmarks)
    if failure == 'exception':
        with pytest.raises(subprocess.CalledProcessError):
            compare.main()
    else:
        assert compare.main() == 1
    assert not output.exists()


@pytest.mark.parametrize(
    ('path', 'expected'),
    [
        ('fla/ops/kda/chunk.py', ['chunk_kda']),
        ('fla/layers/example.py', ['chunk_kda']),
        ('fla/ops/common/chunk_h.py', list_ops()),
        ('fla/ops/utils/solve_tril.py', list_ops()),
        ('benchmarks/ops/inputs.py', list_ops()),
        ('benchmarks/comparison.py', list_ops()),
        ('README.md', []),
    ],
    ids=['operator', 'layer-import', 'common-kernel', 'kernel-utility', 'input-builder', 'revision-worker', 'documentation'],
)
def test_affected_ops(tmp_path, monkeypatch, path, expected):
    layer = tmp_path / 'fla/layers/example.py'
    layer.parent.mkdir(parents=True)
    layer.write_text('from fla.ops.kda import chunk_kda\n')
    monkeypatch.setattr(discovery, 'find_project_root', lambda: str(tmp_path))
    assert discovery.find_affected_op_names(changed_files=[path]) == expected
