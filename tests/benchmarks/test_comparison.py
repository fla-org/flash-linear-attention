# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmarks import compare, run
from benchmarks.utils import revisions
from benchmarks.utils.ops import discovery
from benchmarks.utils.ops.registry import list_ops
from benchmarks.utils.reporting import BenchmarkResult


@pytest.fixture
def benchmark_repository(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    root.mkdir()
    cache = tmp_path / 'parent-cache'
    cache.mkdir()
    (cache / 'sentinel').write_text('keep')
    monkeypatch.setenv('TRITON_CACHE_DIR', str(cache))

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
    (root / 'benchmarks/run.py').write_text('raise RuntimeError("old benchmark definitions")\n')
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
    (harness / 'run.py').write_text(Path(run.__file__).read_text())
    (harness / 'utils').mkdir()
    (harness / 'utils/__init__.py').write_text('')
    (harness / 'utils/reporting.py').write_text((Path(run.__file__).parent / 'utils/reporting.py').read_text())
    (harness / 'utils/environment.py').write_text('def get_machine_info(): return {}\n')
    (harness / 'ops').mkdir()
    (harness / 'ops/__init__.py').write_text('')
    (harness / 'ops/worker.py').write_text(
        'import os\n'
        'from pathlib import Path\n'
        'import fla\n'
        'from benchmarks.utils.reporting import BenchmarkResult\n'
        'def add_arguments(parser):\n'
        '    parser.add_argument("--fail", action="store_true")\n'
        'def run(args):\n'
        '    cache = Path(os.environ["TRITON_CACHE_DIR"])\n'
        '    assert cache.name == "triton" and not cache.exists()\n'
        '    cache.mkdir()\n'
        '    if args.fail:\n'
        '        raise RuntimeError("worker failed")\n'
        '    return [BenchmarkResult(name="source", parameters={"source": fla.VALUE}, metrics={"median_ms": 1.0})]\n'
    )
    monkeypatch.setattr(revisions, '__file__', str(harness / 'utils/revisions.py'))
    monkeypatch.setattr(revisions, 'find_project_root', lambda: str(root))
    yield git
    assert (cache / 'sentinel').read_text() == 'keep'


@pytest.mark.parametrize('ref', ['baseline', 'feature/current', None], ids=['base', 'checked-out-branch', 'working-tree'])
@pytest.mark.parametrize('fail', [False, True], ids=['success', 'worker-failure'])
@pytest.mark.parametrize('rank', [0, 1])
def test_revision_isolation(benchmark_repository, monkeypatch, ref, fail, rank):
    monkeypatch.setenv('RANK', str(rank))
    monkeypatch.setenv('WORLD_SIZE', '2')
    git = benchmark_repository
    before = git('status', '--porcelain')
    worktrees = git('worktree', 'list', '--porcelain')
    arguments = ['ops.worker', '--fail'] if fail else ['ops.worker']
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            revisions.run_benchmark(module='benchmarks.run', arguments=arguments, ref=ref)
    else:
        results, info = revisions.run_benchmark(module='benchmarks.run', arguments=arguments, ref=ref)
        expected = {'baseline': 'baseline', 'feature/current': 'head', None: 'dirty'}[ref]
        assert results == [asdict(BenchmarkResult(name='source', parameters={'source': expected}, metrics={'median_ms': 1.0}))]
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
    monkeypatch.setenv('FLA_BENCH_COOLDOWN_SEC', '0')
    calls = []

    def run_benchmarks(op_names, shapes, modes, backend=None, ref=None):
        calls.append((op_names, shapes, modes, ref))
        results = [asdict(BenchmarkResult(
            name='chunk_gla',
            parameters={'mode': 'fwd', 'B': 1, 'T': 64, 'H': 2, 'D': 16},
            metrics={'median_ms': 2.0 if ref == 'head' else 1.0, 'p20_ms': 1.0, 'p80_ms': 2.0},
        ))]
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
        ('benchmarks/utils/ops/inputs.py', list_ops()),
        ('benchmarks/utils/revisions.py', list_ops()),
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


def test_compare_workload_uses_common_runner(tmp_path, monkeypatch):
    output = tmp_path / 'training.json'
    monkeypatch.setattr(sys, 'argv', [
        'compare', '--benchmark', 'models.training', '--base', 'base', '--head', 'head',
        '--output', str(output), '--', '--name', 'kda',
    ])
    monkeypatch.setattr(compare, 'get_commit_sha', lambda ref: ref)
    monkeypatch.setenv('FLA_BENCH_COOLDOWN_SEC', '0')
    calls = []

    def run_benchmark(module, arguments, ref=None):
        calls.append((module, arguments, ref))
        result = BenchmarkResult(
            name='kda',
            parameters={'B': 1, 'T': 128},
            metrics={'tokens_per_second': 90.0 if ref == 'head' else 100.0},
            primary_metric='tokens_per_second',
            higher_is_better=True,
        )
        return [asdict(result)], {'git_label': ref}

    monkeypatch.setattr(compare, 'run_benchmark', run_benchmark)
    assert compare.main() == 1
    assert calls == [
        ('benchmarks.run', ['models.training', '--name', 'kda'], 'head'),
        ('benchmarks.run', ['models.training', '--name', 'kda'], 'base'),
    ]
    data = json.loads(output.read_text())
    assert data['benchmark'] == 'models.training'
    assert data['regressions'][0]['change_pct'] == pytest.approx(10.0)
