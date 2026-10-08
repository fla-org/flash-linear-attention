# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.feature_map import (
    DPFPFeatureMap,
    GELUFeatureMap,
    HadamardFeatureMap,
    HedgehogFeatureMap,
    LearnableOuterProductFeatureMap,
    LearnablePolySketchNonNegativeFeatureMap,
    RebasedFeatureMap,
    ReLUFeatureMap,
    SigmoidFeatureMap,
    SquaredReLUFeatureMap,
    SwishFeatureMap,
    T2RFeatureMap,
    TaylorFeatureMap,
    flatten_diag_outer_product,
    flatten_diag_outer_product_off1,
)
from fla.utils import device

# (constructor, kwargs, output last-dim given head_dim D) for every feature map
FEATURE_MAPS = [
    (HedgehogFeatureMap, {"head_dim": 16}, 32),
    (T2RFeatureMap, {"head_dim": 16}, 16),
    (T2RFeatureMap, {"head_dim": 16, "dot_dim": 8}, 8),
    (DPFPFeatureMap, {"head_dim": 16, "nu": 4}, 16 * 2 * 4),
    (HadamardFeatureMap, {"head_dim": 16}, 16),
    (LearnableOuterProductFeatureMap, {"head_dim": 16, "feature_dim": 8}, 8 * 9 // 2),
    (LearnablePolySketchNonNegativeFeatureMap, {"head_dim": 16}, 16 * 17 // 2),
    (TaylorFeatureMap, {"head_dim": 16}, 1 + 16 + 16 + 16 * 15 // 2),
    (RebasedFeatureMap, {"head_dim": 16}, 16 + 16 * 15 // 2),
    (ReLUFeatureMap, {}, 16),
    (SquaredReLUFeatureMap, {}, 16),
    (GELUFeatureMap, {}, 16),
    (SwishFeatureMap, {}, 16),
    (SigmoidFeatureMap, {}, 16),
]


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(("fmap", "kwargs", "expected_dim"), FEATURE_MAPS)
def test_feature_map_forward(fmap, kwargs: dict, expected_dim: int, dtype: torch.dtype):
    torch.manual_seed(42)

    x = torch.randn(2, 4, 8, 16, device=device, dtype=dtype)
    fm = fmap(**kwargs).to(device=device, dtype=dtype)
    out = fm(x)

    assert out.shape[:-1] == x.shape[:-1]
    assert out.shape[-1] == expected_dim
    assert torch.isfinite(out).all()


def test_hedgehog_identity_init():
    # the trainable map is initialized to identity with zero bias
    torch.manual_seed(42)
    fm = HedgehogFeatureMap(head_dim=16).to(device)

    assert torch.allclose(fm.layer.weight, torch.eye(16, device=device))
    assert fm.layer.bias.abs().sum() == 0


def test_flatten_diag_outer_product():
    # the helper gathers the upper triangle (incl. diagonal) of the outer product
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8, 6, device=device)
    y = torch.randn(2, 4, 8, 6, device=device)

    z = torch.einsum("...i,...j->...ij", x, y)
    idx = torch.triu_indices(6, 6, device=device)
    diag = torch.arange(6, device=device)

    flat = flatten_diag_outer_product(x, y)
    assert torch.equal(flat, z[..., idx[0], idx[1]])

    off1, on_diag = flatten_diag_outer_product_off1(x, y)
    idx1 = torch.triu_indices(6, 6, 1, device=device)
    assert torch.equal(off1, z[..., idx1[0], idx1[1]])
    assert torch.equal(on_diag, z[..., diag, diag])
