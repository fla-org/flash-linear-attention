# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import gc
from contextlib import contextmanager

import pytest
import torch
from triton.runtime.autotuner import Autotuner

from fla.models import GSA2Config, GSA2ForCausalLM
from fla.utils import IS_AMD, IS_NPU, IS_NVIDIA, assert_close, device

from .test_modeling_base import (
    run_test_generate_matches_forward,
    run_test_generation,
    run_test_model_forward_backward,
)

# GSA2 runs the GDN-2 kernels, which tests/ops/test_gdn2.py guards the same way
pytestmark = pytest.mark.skipif(not (IS_NVIDIA or IS_AMD or IS_NPU), reason="CUDA/ROCm or Ascend NPU required")

_INT_DTYPES = ('torch.int32', 'torch.int64')
# key entries that tell a dense call from the matching varlen call: the varlen flag itself, and the batch size,
# which is B for the dense batch and 1 once the same tokens are packed with cu_seqlens
_LAYOUT_KEYS = ('IS_VARLEN', 'B')


class _DenseVarlenSharedCache(dict):
    """Autotune cache that maps a dense call and the matching varlen call to the same entry."""

    def __init__(self, key_names):
        super().__init__()
        self.layout_at = sorted((i for i, name in enumerate(key_names) if name in _LAYOUT_KEYS), reverse=True)
        self.n_keys = len(key_names)

    def _canonical(self, key):
        if not isinstance(key, tuple):
            return key
        if len(key) >= self.n_keys:
            key = list(key)
            for i in self.layout_at:
                del key[i]
        return tuple(k for k in key if k not in _INT_DTYPES)

    def __contains__(self, key):
        return super().__contains__(self._canonical(key))

    def __getitem__(self, key):
        return super().__getitem__(self._canonical(key))

    def __setitem__(self, key, value):
        super().__setitem__(self._canonical(key), value)

    def get(self, key, default=None):
        return super().get(self._canonical(key), default)

    def pop(self, key, *default):
        return super().pop(self._canonical(key), *default)


def _all_autotuners():
    # not every Autotuner is reachable as a module attribute, so collect them from the heap. `type()` rather than
    # `isinstance()`: the latter reads `__class__`, which deprecated proxies such as `torch.distributed.reduce_op` warn on
    return [obj for obj in gc.get_objects() if issubclass(type(obj), Autotuner)]


@contextmanager
def _shared_dense_varlen_autotune():
    """Let dense and varlen calls of every fla kernel run with the same autotuned config, and restore the caches after."""
    tuners = _all_autotuners()
    saved = {id(t): t.cache for t in tuners}
    original_init = Autotuner.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.cache = _DenseVarlenSharedCache(self.keys)

    Autotuner.__init__ = init
    for tuner in tuners:
        tuner.cache = _DenseVarlenSharedCache(tuner.keys)
    try:
        yield
    finally:
        Autotuner.__init__ = original_init
        for tuner in tuners:
            tuner.cache = saved[id(tuner)]


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'num_slots', 'level2_axis', 'oja_out_act', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-M{}-{}-{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True,  128, 'head', 'silu',    None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 128, 'head', 'softmax', None, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 64,  'head', 'silu',    1,    torch.bfloat16),
            (4, 4, 1024, 4, 64, False, 256, 'slot', 'silu',    4,    torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    use_l2warp: bool,
    num_slots: int,
    level2_axis: str,
    oja_out_act: str,
    attnres_block_size: int | None,
    dtype: torch.dtype,
):
    # cu_seqlens changes the dtype part of Triton's autotune key, so the dense and the varlen pass autotune on their own and
    # may settle on different configs. In bf16 the reordered reductions alone move the output of the two stacked recurrences
    # past the 1e-3 this comparison allows (KDA shows the same on H20), so both passes share one autotune result here,
    # while tests/ops/test_gated_oja_rule2.py covers the varlen-specific configs against the naive reference.
    with _shared_dense_varlen_autotune():
        run_test_model_forward_backward(
            L,
            B,
            T,
            H,
            D,
            GSA2Config,
            use_l2warp=use_l2warp,
            num_slots=num_slots,
            level2_axis=level2_axis,
            oja_out_act=oja_out_act,
            attnres_block_size=attnres_block_size,
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    # chunked prefill and token-by-token decode round differently in fp16, and GSA2 stacks two recurrences per layer where
    # GDN and KDA have one. On H20 the single-recurrence models measure 1.6e-3 to 1.7e-3 against the default 2e-3 and GSA2
    # measures 2.9e-3, so the budget is twice the default. In fp32 the same comparison gives 2e-6.
    run_test_generation(L, B, T, H, D, GSA2Config, dtype, tol=4e-3)


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 2, 64, 8, 64, torch.float32),
        ]
    ],
)
def test_generate_prefill(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    run_test_generate_matches_forward(L, B, T, H, D, GSA2Config, dtype)


@torch.no_grad()
def test_logits_to_keep_with_labels():
    B, T, H, D, V = 2, 8, 2, 8, 32
    config = GSA2Config(
        hidden_size=H * D,
        num_hidden_layers=1,
        num_heads=H,
        head_dim=D,
        num_slots=64,
        level2_axis='slot',
        expand_v=1,
        vocab_size=V,
        fuse_cross_entropy=False,
    )
    model = GSA2ForCausalLM(config).eval().to(device)
    input_ids = torch.randint(V, (B, T), device=device)

    expected = model(input_ids, labels=input_ids)
    actual = model(input_ids, labels=input_ids, logits_to_keep=1)
    inference = model(input_ids, logits_to_keep=1)

    assert actual.logits.shape == expected.logits.shape == (B, T, V)
    assert inference.logits.shape == (B, 1, V)
    assert torch.isfinite(actual.loss)
    assert_close('loss', expected.loss, actual.loss, 1e-6)
