# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.modules.l2norm import l2_norm
from fla.utils import assert_close, device


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("D", [50, 64, 128, 1000, 2048])
def test_l2norm(B: int, H: int, T: int, D: int):
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref_y = F.normalize(x, dim=-1, p=2)
    tri_y = l2_norm(x)
    ref_dx = torch.autograd.grad(ref_y.sum(), x)[0]
    tri_dx = torch.autograd.grad(tri_y.sum(), x)[0]

    assert_close(' y', ref_y, tri_y, ratio=1e-3)
    assert_close('dx', ref_dx, tri_dx, ratio=1e-3)
