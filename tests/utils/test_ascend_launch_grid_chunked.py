# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import gc
import weakref

from fla.utils.ascend_ub_manager import launch_grid_chunked


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def _launch(**kwargs):
            self.calls.append((grid, dict(kwargs)))

        return _launch


class _NoopKernel:
    def __getitem__(self, grid):
        def _launch(**kwargs):
            pass

        return _launch


class _Sentinel:
    pass


def test_launch_grid_chunked_2d_grid_offsets_and_compile_kwargs():
    kernel = _RecordingKernel()
    launch_grid_chunked(
        kernel,
        (5, 7),
        offset_keys=('offset_0', 'offset_1'),
        kernel_kwargs={'payload': 7},
        budget=4,
        compile_kwargs={'num_warps': 4},
    )

    expected = []
    for offset_0 in range(5):
        for offset_1, size_1 in ((0, 4), (4, 3)):
            expected.append((
                (1, size_1),
                {
                    'payload': 7,
                    'offset_0': offset_0,
                    'offset_1': offset_1,
                    'num_warps': 4,
                },
            ))
    assert kernel.calls == expected


def test_launch_grid_chunked_3d_grid_offsets_and_order():
    kernel = _RecordingKernel()
    launch_grid_chunked(
        kernel,
        (3, 4, 5),
        offset_keys=('offset_0', 'offset_1', 'offset_2'),
        kernel_kwargs={'payload': 'keep'},
        budget=4,
    )

    expected = []
    for offset_0 in range(3):
        for offset_1 in range(4):
            for offset_2, size_2 in ((0, 4), (4, 1)):
                expected.append((
                    (1, 1, size_2),
                    {
                        'payload': 'keep',
                        'offset_0': offset_0,
                        'offset_1': offset_1,
                        'offset_2': offset_2,
                    },
                ))
    assert kernel.calls == expected


def test_launch_grid_chunked_quanta_preserves_boundaries_and_tail_chunks():
    kernel = _RecordingKernel()
    launch_grid_chunked(
        kernel,
        (5, 10),
        offset_keys=('offset_0', 'offset_1'),
        kernel_kwargs={},
        quanta=(2, 4),
        budget=16,
    )

    assert kernel.calls == [
        ((2, 8), {'offset_0': 0, 'offset_1': 0}),
        ((2, 2), {'offset_0': 0, 'offset_1': 8}),
        ((2, 8), {'offset_0': 2, 'offset_1': 0}),
        ((2, 2), {'offset_0': 2, 'offset_1': 8}),
        ((1, 8), {'offset_0': 4, 'offset_1': 0}),
        ((1, 2), {'offset_0': 4, 'offset_1': 8}),
    ]


def _launch_with_weakref_sentinel():
    sentinel = _Sentinel()
    sentinel_ref = weakref.ref(sentinel)
    launch_grid_chunked(
        _NoopKernel(),
        (2, 2),
        offset_keys=('offset_0', 'offset_1'),
        kernel_kwargs={'sentinel': sentinel},
        budget=4,
    )
    return sentinel_ref


def test_launch_grid_chunked_does_not_retain_kernel_kwargs_reference_cycle():
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        sentinel_ref = _launch_with_weakref_sentinel()
        assert sentinel_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()
