# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import triton
import triton.language as tl

from fla.modules.conv.triton.kernels import causal_conv1d_bwd_kernel, causal_conv1d_fwd_kernel
from fla.ops.gla.chunk import chunk_gla_bwd_kernel_inter
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


@pytest.mark.parametrize("kernel", [causal_conv1d_fwd_kernel, causal_conv1d_bwd_kernel])
def test_causal_conv1d_autotune_key_excludes_unused_nb(kernel):
    """NB (ceil(B*T / 1024)) is never read inside either kernel body, so it must not sit in the
    autotune key: leaving it in forces a redundant re-tune on every distinct B*T even though D
    and W, the values that actually pick the fastest config, are unchanged.
    """
    autotuner = kernel.fn
    assert 'NB' not in autotuner.keys

    arg_names = autotuner.arg_names

    def build_key(nb):
        values = {'D': 64, 'W': 4, 'NB': nb}
        args = tuple(values.get(name) for name in arg_names)
        return AutotuneKey.build(arg_names, autotuner.keys, args, {})

    assert build_key(nb=5).autotune_key == build_key(nb=7).autotune_key


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
