# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import inspect

import pytest
import torch
import triton
import triton.language as tl

from fla.modules.causal_conv1d.ops import causal_conv1d_bwd_kernel, causal_conv1d_fwd_kernel
from fla.modules.token_shift import token_shift_bwd_kernel_long, token_shift_fwd_kernel_long
from fla.ops.gla.chunk import chunk_gla_bwd_kernel_inter
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import AutotuneKey, fla_cache_autotune
from fla.utils import device


# Multiple configs are required to actually exercise the autotune benchmark path, which is where Triton's default pre_hook
# clones each `restore_value` argument and previously crashed when the argument was None.
@fla_cache_autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
    restore_value=['x'],
)
@triton.jit
def _optional_restore_kernel(x, y, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    if x is not None:
        tl.store(y + idx, tl.load(x + idx, mask=mask), mask=mask)
    else:
        tl.store(y + idx, tl.zeros_like(idx), mask=mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fla_cache_autotune_handles_none_restore_value():
    """A None argument listed in restore_value must not crash autotuning.

    Triton's default pre_hook clones every ``restore_value`` argument during benchmarking;
    if the argument is None this raises ``AttributeError: 'NoneType' object has no attribute 'clone'``.
    CachedAutotuner installs None-safe pre/post hooks to avoid this;
    the test exercises the multi-config (benchmark) path so the pre_hook is actually invoked.
    """
    N = 1024
    y = torch.zeros(N, dtype=torch.int32, device=device)

    # Baseline: restore-value arg is a real tensor — should produce ones.
    x = torch.ones(N, dtype=torch.int32, device=device)
    _optional_restore_kernel[(triton.cdiv(N, 128),)](x, y, N)
    assert torch.equal(y, torch.ones(N, dtype=torch.int32, device=device))

    # Use a different key so the benchmark path runs again, this time with the restore_value arg set to None.
    # Pre-fix this raised AttributeError: 'NoneType' object has no attribute 'clone'.
    M = 2048
    y2 = torch.full((M,), 7, dtype=torch.int32, device=device)
    _optional_restore_kernel[(triton.cdiv(M, 128),)](None, y2, M)
    assert torch.equal(y2, torch.zeros(M, dtype=torch.int32, device=device))


@pytest.mark.parametrize("kernel", [
    causal_conv1d_fwd_kernel,
    causal_conv1d_bwd_kernel,
    token_shift_fwd_kernel_long,
    token_shift_bwd_kernel_long,
], ids=["causal_conv1d_fwd", "causal_conv1d_bwd", "token_shift_fwd_long", "token_shift_bwd_long"])
def test_kernels_do_not_take_unused_nb(kernel):
    """NB (a 1024-token bucket of the input size) is never read inside these kernel bodies, so none of them may take it:
    in the autotune key it forces a re-tune on every new bucket, and as a constexpr it still compiles a new kernel for each one.
    """
    autotuner = kernel.fn
    assert 'NB' not in autotuner.keys
    assert 'NB' not in autotuner.arg_names


@pytest.fixture
def conv_forward_launch(monkeypatch):
    from fla.modules.causal_conv1d import ops

    torch.manual_seed(42)
    autotuner = ops.causal_conv1d_fwd_kernel.fn
    calls = []

    class KernelSpy:
        def __getitem__(self, grid):
            def launch(**kwargs):
                key = AutotuneKey.build(
                    arg_names=autotuner.arg_names,
                    key_names=autotuner.keys,
                    positional_args=(),
                    runtime_kwargs=kwargs,
                )
                calls.append({'key': key.autotune_key, 'grid': grid({'BD': 16}), 'BT': kwargs['BT']})
            return launch

    monkeypatch.setattr(ops, 'causal_conv1d_fwd_kernel', KernelSpy())

    def launch(B, T, cu_seqlens=None):
        indices = prepare_chunk_indices(cu_seqlens, 64) if cu_seqlens is not None else None
        # unwrap dispatch/device guards while executing the real Python launcher on CPU tensors.
        inspect.unwrap(ops.causal_conv1d_fwd)(
            x=torch.zeros(B, T, 16, device='cpu'),
            weight=torch.ones(16, 4, device='cpu'),
            bias=None,
            residual=None,
            cu_seqlens=cu_seqlens,
            chunk_indices=indices,
        )
        return calls[-1]

    return launch


def test_conv_forward_cache_keys_for_dense_lengths(conv_forward_launch):
    lengths = range(1, 131)
    keys = {conv_forward_launch(B=1, T=length)['key'] for length in lengths}
    # changing every token count must not create a fresh tuning problem.
    assert len(keys) <= len(lengths).bit_length() + 1


def test_conv_forward_cache_keys_for_equal_workloads(conv_forward_launch):
    calls = [conv_forward_launch(B=B, T=T) for B, T in [(8, 65), (8, 127), (16, 33), (16, 64)]]
    assert len({call['BT'] for call in calls}) == 1
    assert [call['grid'] for call in calls] == [(1, 2, 8), (1, 2, 8), (1, 1, 16), (1, 1, 16)]
    assert len({call['key'] for call in calls}) == 1


def test_conv_forward_cache_keys_for_packed_lengths(conv_forward_launch):
    keys = set()
    for length, chunks in [(128, 3), (257, 5), (1024, 17), (2049, 33)]:
        cu_seqlens = torch.tensor([0, length // 3, length], dtype=torch.int64, device='cpu')
        call = conv_forward_launch(B=1, T=length, cu_seqlens=cu_seqlens)
        assert call['BT'] == 64
        assert call['grid'] == (1, chunks, 1)
        keys.add(call['key'])
    assert len(keys) == 1


def test_conv_forward_workload_keys_are_runtime_only():
    autotuner = causal_conv1d_fwd_kernel.fn
    kernel = autotuner.fn
    for name in set(autotuner.keys) - {'D', 'W', 'BT'}:
        parameter = kernel.params[autotuner.arg_names.index(name)]
        assert not parameter.is_constexpr
        assert parameter.do_not_specialize


@pytest.mark.parametrize(
    ('BK', 'num_warps', 'accepted'),
    [
        pytest.param(32, 4, False, id='unsafe-wgmma'),
        pytest.param(64, 2, False, id='oversized-key-tile'),
        pytest.param(32, 2, True, id='builtin-config'),
        pytest.param(16, 4, True, id='external-config'),
    ],
)
def test_gla_cached_config_respects_pruning(monkeypatch, BK, num_warps, accepted):
    autotuner = chunk_gla_bwd_kernel_inter.fn
    configs = [triton.Config({'BK': bk, 'BV': 64}, num_warps=2, num_stages=2) for bk in (32, 64)]
    cfg = triton.Config({'BK': BK, 'BV': 64}, num_warps=num_warps, num_stages=2)
    key = AutotuneKey(autotune_key=(64, True, 32, 64))
    monkeypatch.setattr('fla.ops.gla.chunk.IS_NVIDIA_HOPPER', True)
    monkeypatch.setattr('fla.ops.gla.chunk.TRITON_ABOVE_3_6_0', True)
    monkeypatch.setattr('fla.ops.gla.chunk.TRITON_ABOVE_3_8_0', False)
    monkeypatch.setattr(autotuner, 'configs', configs)
    monkeypatch.setattr(autotuner, 'cache', {key.autotune_key: cfg})
    monkeypatch.setattr('fla.ops.utils.cache.load_cached_config', lambda kernel_name, key: {
        'kwargs': cfg.kwargs,
        'num_warps': cfg.num_warps,
        'num_stages': cfg.num_stages,
        'num_ctas': 1,
    })

    autotuner.maybe_load_cached_config(key=key, nargs={}, runtime_kwargs={'K': 32, 'V': 64, 'STATE_V_FIRST': True})

    if accepted:
        assert autotuner.cache[key.autotune_key].all_kwargs() == cfg.all_kwargs()
    else:
        assert key.autotune_key not in autotuner.cache
