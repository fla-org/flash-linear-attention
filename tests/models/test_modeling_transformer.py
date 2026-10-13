# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from fla.models import TransformerConfig
from fla.utils import assert_close, device, find_spec_cached

from .test_modeling_base import run_test_generation, run_test_model_forward_backward
from .test_modeling_utils import create_model_and_config

# Mark (not importorskip): a fully skipped module reports "no tests collected" and exits
# with code 5, which CI per-file loops treat as a failure.
pytestmark = pytest.mark.skipif(find_spec_cached("flash_attn") is None,
                                reason="Attention requires flash-attn (`pip install flash-attn --no-build-isolation`).")


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'attnres_block_size', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-bs{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, None, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, None, torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 1,    torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 2,    torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, 4,    torch.bfloat16),
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
    attnres_block_size: int | None,
    dtype: torch.dtype,
):
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        TransformerConfig,
        use_l2warp=use_l2warp,
        attnres_block_size=attnres_block_size,
        dtype=dtype,
    )


# ===================================================================================
# Test for Generation
# ===================================================================================
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
    run_test_generation(L, B, T, H, D, TransformerConfig, dtype)


@pytest.mark.parametrize('backend', ['triton', 'gluon'])
def test_fused_cross_entropy(backend, monkeypatch):
    from fla.backends import _DISPATCH_DISABLED, GluonBackend
    from fla.modules.fused_cross_entropy import ops

    if backend == 'gluon' and (not GluonBackend.is_available() or _DISPATCH_DISABLED):
        pytest.skip("Gluon dispatch is unavailable")
    torch.manual_seed(42)
    model, config = create_model_and_config(
        config_class=TransformerConfig,
        L=2,
        H=4,
        D=64,
        dtype=torch.bfloat16,
        vocab_size=32003,
        intermediate_size=512,
        fuse_cross_entropy=True,
        fuse_linear_cross_entropy=False,
        use_l2warp=False,
    )
    model.eval()
    input_ids = torch.randint(config.vocab_size, (2, 127), device=device)
    labels = input_ids.clone()
    labels[:, ::7] = -100
    shifted = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim=1)
    logits = model(input_ids=input_ids, use_cache=False).logits
    ref = F.cross_entropy(logits.float().reshape(-1, config.vocab_size), shifted.reshape(-1))
    ref.backward()
    ref_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    model.zero_grad(set_to_none=True)

    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_CE_GLUON', '1' if backend == 'gluon' else '0')
    if backend == 'gluon':
        from fla.modules.fused_cross_entropy import gluon

        forward = Mock(wraps=gluon._cross_entropy_fwd_gluon)
        monkeypatch.setattr(gluon, '_cross_entropy_fwd_gluon', forward)
    else:
        forward = Mock(wraps=ops.cross_entropy_fwd)
        monkeypatch.setattr(ops, 'cross_entropy_fwd', forward)
    loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
    loss.backward()
    forward.assert_called_once()
    assert_close('loss', ref, loss, ratio=1e-2)
    grads = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    assert grads.keys() == ref_grads.keys()
    for name in ref_grads:
        assert_close(name, ref_grads[name], grads[name], ratio=1e-2)
