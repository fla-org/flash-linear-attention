# -*- coding: utf-8 -*-

import os

import pytest
import torch
import torch.nn.functional as F

from fla.modules import FusedKLDivLoss
from fla.utils import device

compiled_mode = os.getenv("COMPILER_MODE") == "1"
ci_env = os.getenv("CI_ENV") == "1"


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio, warning=False):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    error_rate = get_err_ratio(ref, tri)
    if warning or str(prefix).strip().lower() == "dh0" or compiled_mode or (ci_env and error_rate < 0.1):
        if error_rate > ratio:
            import warnings
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [16, 32])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("reduction", ["batchmean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused(B: int, T: int, D: int, V: int, reduction: str, dtype: torch.dtype):
    torch.manual_seed(42)
    x = torch.randn(B * T, D).to(device).to(dtype=dtype).requires_grad_()
    x_weight = torch.randn(V, D).to(device).to(dtype=dtype).requires_grad_()
    target_x = torch.randn(B * T, D).to(device).to(dtype=dtype)
    target_weight = torch.randn(V, D).to(device).to(dtype=dtype)

    ref = F.kl_div(
        F.linear(x, x_weight).log_softmax(-1),
        F.linear(target_x, target_weight).softmax(-1),
        reduction=reduction
    ).to(dtype)
    do = torch.randn_like(ref).to(device)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dw, x_weight.grad = x_weight.grad.clone(), None

    tri = FusedKLDivLoss(reduction)(x, target_x, x_weight, target_weight).to(dtype=dtype)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dw, x_weight.grad = x_weight.grad.clone(), None

    assert_close("  o", ref, tri, 1e-2)
    assert_close(" dx", ref_dx, tri_dx, 1e-2)
    assert_close(" dw", ref_dw, tri_dw, 1e-2)
