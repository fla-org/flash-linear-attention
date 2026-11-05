# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.ops.graph_registry import get_graph_op, normalize_gdn_shape


@pytest.mark.parametrize(
    'failure',
    ['none', 'ordinary-missing', 'ordinary-duplicate', 'ordinary-op', 'ordinary-nan',
     'graph-missing', 'graph-duplicate', 'graph-dtype', 'graph-validation', 'graph-timing', 'graph-memory'],
)
def test_gdn_cache_rejects_incomplete_or_invalid_results(tmp_path, monkeypatch, failure):
    root = Path(__file__).resolve().parents[2]
    workflow = (root / '.github/workflows/ascend-a2-benchmark-ci.yml').read_text()
    validation = workflow.split('      - name: Validate GDN benchmark cache inputs\n', 1)[1]
    script = validation.split("          python - <<'PY'\n", 1)[1].split('          PY\n', 1)[0]
    script = '\n'.join(line[10:] for line in script.splitlines())

    registry = ModuleType('benchmarks.ops.registry')
    registry.SHAPE_CONFIGS = {'smoke': dict(B=1, T=128, H=2, D=64)}
    registry.get_op = lambda _: SimpleNamespace(default_shapes=None)
    monkeypatch.setitem(sys.modules, 'benchmarks.ops.registry', registry)
    ordinary = [dict(op='chunk_gdn', mode=mode, **registry.SHAPE_CONFIGS['smoke'],
                     median_ms=1.0, p20_ms=0.9, p80_ms=1.1) for mode in ('fwd', 'fwdbwd')]
    config = get_graph_op('chunk_gdn')
    graph = []
    for name, shape in config.default_shapes.items():
        for mode in config.supported_modes:
            graph.append(dict(
                op='chunk_gdn', shape=name, mode=mode, **normalize_gdn_shape(shape),
                validation='passed', capture_ms=1.0,
                **{engine: dict(p50_ms=1.0, p95_ms=1.1, mean_ms=1.0, min_ms=0.9)
                   for engine in ('eager', 'graph', 'graph_safe')},
                memory=dict(baseline_allocated_bytes=100, peak_allocated_bytes=200, peak_reserved_bytes=300),
            ))
    if failure.endswith('-missing'):
        (ordinary if failure.startswith('ordinary') else graph).pop()
    elif failure.endswith('-duplicate'):
        rows = ordinary if failure.startswith('ordinary') else graph
        rows.append(copy.deepcopy(rows[0]))
    elif failure == 'ordinary-op':
        ordinary[0]['op'] = 'chunk_kda'
    elif failure == 'ordinary-nan':
        ordinary[0]['median_ms'] = float('nan')
    elif failure == 'graph-dtype':
        graph[0]['dtype'] = 'float32'
    elif failure == 'graph-validation':
        graph[0]['validation'] = 'not-run'
    elif failure == 'graph-timing':
        graph[0]['graph_safe']['p50_ms'] = 0
    elif failure == 'graph-memory':
        graph[0]['memory']['peak_allocated_bytes'] = 0
    monkeypatch.chdir(tmp_path)
    Path('benchmark_chunk_gdn.json').write_text(json.dumps(dict(results=ordinary)))
    Path('benchmark_gdn_graph.json').write_text(json.dumps(dict(current=dict(results=graph))))
    if failure == 'none':
        exec(compile(script, '<GDN cache validator>', 'exec'), {})
    else:
        with pytest.raises(AssertionError):
            exec(compile(script, '<GDN cache validator>', 'exec'), {})


@pytest.mark.parametrize('event', ['schedule', 'pull_request', 'push', 'workflow_dispatch'])
@pytest.mark.parametrize('kda_hit,gdn_hit', [(True, True), (True, False), (False, True), (False, False)])
def test_gdn_cache_prepares_environment_on_partial_hit(event, kda_hit, gdn_hit):
    root = Path(__file__).resolve().parents[2]
    workflow = (root / '.github/workflows/ascend-a2-benchmark-ci.yml').read_text()
    step = workflow.split('      - name: Install dependencies\n', 1)[1].split('\n        run:', 1)[0]
    condition = step.split('if: ', 1)[1].strip()
    values = {
        'github.event_name': event,
        'steps.triton-cache.outputs.cache-hit': str(kda_hit).lower(),
        'steps.gdn-cache.outputs.cache-hit': str(gdn_hit).lower(),
    }
    for name, value in values.items():
        condition = condition.replace(name, repr(value))
    condition = condition.replace('||', 'or')
    assert eval(condition, {'__builtins__': {}}) == (event != 'schedule' or not (kda_hit and gdn_hit))
