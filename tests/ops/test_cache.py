# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import json

import pytest
import torch
import triton
import triton.language as tl

from fla.ops.utils import cache as cache_mod
from fla.ops.utils.cache import fla_cache_autotune
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


def test_load_cached_config_default_mode(tmp_path, monkeypatch):
    """DEFAULT mode returns the top-level default_config of the kernel's json file."""
    monkeypatch.setattr(cache_mod, "FLA_CACHE_MODE", cache_mod.FlaCacheMode.DEFAULT)
    monkeypatch.setenv("FLA_CONFIG_DIR", str(tmp_path))

    kernel = "cache_test_kernel_default"
    default_config = {"kwargs": {"BLOCK": 64}, "num_warps": 4, "num_stages": 2, "num_ctas": 1}
    (tmp_path / f"{kernel}.json").write_text(json.dumps({"default_config": default_config}))

    assert cache_mod.load_cached_config(kernel) == default_config


def test_load_kernel_config_file_memoizes_missing_file(tmp_path, monkeypatch):
    """A missing config file is looked up on disk once, then served from memory.

    Repeated lookups for kernels without a config file previously performed one
    filesystem stat per kernel launch, which showed up as excessive small-file
    IO in multi-rank trainings.
    """
    monkeypatch.setattr(cache_mod, "FLA_CACHE_MODE", cache_mod.FlaCacheMode.FULL)
    monkeypatch.setenv("FLA_CONFIG_DIR", str(tmp_path))

    kernel = "cache_test_kernel_missing"
    assert cache_mod.load_cached_config(kernel) is None

    # outside ALWAYS mode the negative result is memoized, matching the
    # existing behavior of load_config_file for files created mid-run
    (tmp_path / f"{kernel}.json").write_text(json.dumps({"default_config": {"kwargs": {}}}))
    assert cache_mod.load_cached_config(kernel) is None


def test_load_cached_config_always_mode_rereads(tmp_path, monkeypatch):
    """ALWAYS mode picks up config file edits on the next lookup."""
    monkeypatch.setattr(cache_mod, "FLA_CACHE_MODE", cache_mod.FlaCacheMode.ALWAYS)
    monkeypatch.setenv("FLA_CONFIG_DIR", str(tmp_path))

    kernel = "cache_test_kernel_always"
    config_file = tmp_path / f"{kernel}.json"

    config_file.write_text(json.dumps({"default_config": {"kwargs": {"BLOCK": 64}}}))
    assert cache_mod.load_cached_config(kernel)["kwargs"]["BLOCK"] == 64

    config_file.write_text(json.dumps({"default_config": {"kwargs": {"BLOCK": 128}}}))
    assert cache_mod.load_cached_config(kernel)["kwargs"]["BLOCK"] == 128
