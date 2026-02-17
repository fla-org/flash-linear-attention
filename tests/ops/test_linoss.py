import os

import pytest
import torch

from fla.ops.linoss import fused_recurrent_linoss
from fla.ops.linoss.naive import naive_recurrent_linoss
from fla.utils import assert_close, device


def generate_linoss_inputs(B, T, H, P, dtype, with_initial_state=False):
    """Generate random inputs for LinOSS testing."""
    torch.manual_seed(42)

    x = torch.randn((B, T, H), dtype=dtype, device=device)
    B_re = torch.randn((P, H), dtype=dtype, device=device) / (H ** 0.5)
    B_im = torch.randn((P, H), dtype=dtype, device=device) / (H ** 0.5)
    C_re = torch.randn((H, P), dtype=dtype, device=device) / (P ** 0.5)
    C_im = torch.randn((H, P), dtype=dtype, device=device) / (P ** 0.5)
    a_diag = torch.rand((P,), dtype=dtype, device=device)
    dt = torch.randn((P,), dtype=dtype, device=device)
    d_skip = torch.randn((H,), dtype=dtype, device=device)

    h0 = None
    if with_initial_state:
        h0 = torch.randn((B, 2, P), dtype=dtype, device=device) * 0.1

    return x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip, h0


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'P', 'discretization', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-P{}-{}-{}".format(*test))
        for test in [
            (1, 16, 32, 16, 'IM', torch.float),
            (2, 64, 64, 32, 'IM', torch.float),
            (2, 128, 128, 64, 'IM', torch.float),
            (1, 16, 32, 16, 'IMEX', torch.float),
            (2, 64, 64, 32, 'IMEX', torch.float),
            (2, 128, 128, 64, 'IMEX', torch.float),
        ]
    ],
)
def test_fused_recurrent_fwd(
    B: int,
    T: int,
    H: int,
    P: int,
    discretization: str,
    dtype: torch.dtype,
):
    """Test forward pass: naive vs fused_recurrent."""
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip, _ = generate_linoss_inputs(B, T, H, P, dtype)

    ref, _ = naive_recurrent_linoss(
        x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip,
        discretization=discretization,
    )
    tri, _ = fused_recurrent_linoss(
        x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip,
        discretization=discretization,
    )

    assert_close(f'  fwd ({discretization})', ref, tri, 0.01)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'P', 'discretization', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-P{}-{}-{}".format(*test))
        for test in [
            (1, 16, 32, 16, 'IM', torch.float),
            (2, 64, 64, 32, 'IM', torch.float),
            (1, 16, 32, 16, 'IMEX', torch.float),
            (2, 64, 64, 32, 'IMEX', torch.float),
        ]
    ],
)
def test_fused_recurrent_bwd(
    B: int,
    T: int,
    H: int,
    P: int,
    discretization: str,
    dtype: torch.dtype,
):
    """Test backward pass: gradients from naive vs fused_recurrent."""
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip, _ = generate_linoss_inputs(B, T, H, P, dtype)

    inputs = [x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip]
    inputs = [i.detach().clone().requires_grad_(True) for i in inputs]
    x1, B_re1, B_im1, C_re1, C_im1, a_diag1, dt1, d_skip1 = inputs

    inputs2 = [x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip]
    inputs2 = [i.detach().clone().requires_grad_(True) for i in inputs2]
    x2, B_re2, B_im2, C_re2, C_im2, a_diag2, dt2, d_skip2 = inputs2

    do = torch.randn_like(x)

    ref, _ = naive_recurrent_linoss(
        x1, B_re1, B_im1, C_re1, C_im1, a_diag1, dt1, d_skip1,
        discretization=discretization,
    )
    (ref * do).sum().backward()

    tri, _ = fused_recurrent_linoss(
        x2, B_re2, B_im2, C_re2, C_im2, a_diag2, dt2, d_skip2,
        discretization=discretization,
    )
    (tri * do).sum().backward()

    assert_close(f'  dx ({discretization})', x1.grad, x2.grad, 0.01)
    assert_close(f'dB_re ({discretization})', B_re1.grad, B_re2.grad, 0.01)
    assert_close(f'dB_im ({discretization})', B_im1.grad, B_im2.grad, 0.01)
    assert_close(f'dC_re ({discretization})', C_re1.grad, C_re2.grad, 0.01)
    assert_close(f'dC_im ({discretization})', C_im1.grad, C_im2.grad, 0.01)
    assert_close(f'da ({discretization})', a_diag1.grad, a_diag2.grad, 0.01)
    assert_close(f'ddt ({discretization})', dt1.grad, dt2.grad, 0.01)
    assert_close(f'dD ({discretization})', d_skip1.grad, d_skip2.grad, 0.01)


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'P', 'discretization', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-P{}-{}-{}".format(*test))
        for test in [
            (1, 16, 32, 16, 'IM', torch.float),
            (2, 64, 64, 32, 'IMEX', torch.float),
        ]
    ],
)
def test_fused_recurrent_with_initial_state(
    B: int,
    T: int,
    H: int,
    P: int,
    discretization: str,
    dtype: torch.dtype,
):
    """Test forward with initial state."""
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip, h0 = generate_linoss_inputs(
        B, T, H, P, dtype, with_initial_state=True
    )

    ref, ref_ht = naive_recurrent_linoss(
        x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip,
        initial_state=h0,
        output_final_state=True,
        discretization=discretization,
    )
    tri, tri_ht = fused_recurrent_linoss(
        x, B_re, B_im, C_re, C_im, a_diag, dt, d_skip,
        initial_state=h0,
        output_final_state=True,
        discretization=discretization,
    )

    assert_close(f'  fwd+h0 ({discretization})', ref, tri, 0.01)
    if ref_ht is not None and tri_ht is not None:
        assert_close(f'  ht ({discretization})', ref_ht, tri_ht, 0.01)
