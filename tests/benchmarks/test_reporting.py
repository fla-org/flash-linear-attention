# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from dataclasses import asdict

import pytest

from benchmarks.utils.reporting import BenchmarkResult, compare_results


@pytest.mark.parametrize('higher_is_better', [False, True], ids=['latency', 'throughput'])
@pytest.mark.parametrize('regression', [False, True], ids=['speedup', 'regression'])
def test_compare_metric_direction(higher_is_better, regression):
    metric = 'tokens_per_second' if higher_is_better else 'median_ms'
    base = asdict(BenchmarkResult(
        name='case', parameters={'T': 128}, metrics={metric: 100.0},
        primary_metric=metric, higher_is_better=higher_is_better,
    ))
    head = {**base, 'metrics': {metric: 90.0 if regression == higher_is_better else 110.0}}
    changes = compare_results(base=[base], head=[head], threshold=5.0)
    assert changes[0]['regression'] is regression
    assert changes[0]['speedup'] is not regression
    assert changes[0]['change_pct'] == pytest.approx(10.0 if regression else -10.0)


@pytest.mark.parametrize('metric', [0.0, -1.0, float('nan'), float('inf')])
def test_compare_rejects_invalid_measurements(metric):
    result = asdict(BenchmarkResult(name='case', parameters={}, metrics={'median_ms': metric}))
    with pytest.raises(ValueError, match='must be positive'):
        compare_results(base=[result], head=[result], threshold=5.0)


def test_compare_rejects_missing_and_duplicate_cases():
    result = asdict(BenchmarkResult(name='case', parameters={'T': 128}, metrics={'median_ms': 1.0}))
    with pytest.raises(ValueError, match='same benchmark cases'):
        compare_results(base=[result], head=[], threshold=5.0)
    with pytest.raises(ValueError, match='Duplicate benchmark'):
        compare_results(base=[result, result], head=[result], threshold=5.0)
