
import os

import pytest
import torch

from fla.models import MambaConfig
from fla.utils import device

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True, torch.bfloat16),
            (4, 4, 1024, 4, 64, False, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, torch.bfloat16),
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
    dtype: torch.dtype,
):
    run_test_model_forward_backward(L, B, T, H, D, MambaConfig, use_l2warp=use_l2warp, dtype=dtype)


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
    run_test_generation(L, B, T, H, D, MambaConfig, dtype)


# ===================================================================================
# Layer-level: Custom fla.layers.mamba.Mamba vs Official mamba_ssm.Mamba
# ===================================================================================

def _copy_params(src, dst):
    """Copy parameters with matching names and shapes from src to dst."""
    dst_dict = dict(dst.named_parameters())
    with torch.no_grad():
        for name, param in src.named_parameters():
            if name in dst_dict and dst_dict[name].shape == param.shape:
                dst_dict[name].data.copy_(param.data)


def _make_mamba_pair(d_model, d_state=16, d_conv=4, expand=2, dtype=torch.float32):
    """Create custom & official Mamba layers sharing identical weights.

    Returns (custom, official) with all parameters copied from custom → official.
    Skips the test if mamba_ssm or causal_conv1d are not installed.
    """
    pytest.importorskip("mamba_ssm")
    pytest.importorskip("causal_conv1d")
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba

    from fla.layers.mamba import Mamba as CustomMamba

    os.environ['FLA_CONV_BACKEND'] = 'cuda'

    torch.manual_seed(42)
    custom = CustomMamba(
        hidden_size=d_model,
        state_size=d_state,
        conv_kernel=d_conv,
        use_conv_bias=True,
        intermediate_size=expand * d_model,
        dt_rank="auto",
        use_bias=False,
        hidden_act="silu",
        layer_idx=0,
        backend="cuda",
    ).to(dtype=dtype, device=device)

    official = OfficialMamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        bias=False,
        conv_bias=True,
        use_fast_path=True,
        device=device,
        dtype=dtype,
    )

    _copy_params(custom, official)
    return custom, official


@pytest.mark.parametrize(
    ['B', 'T', 'd_model', 'd_state', 'expand', 'dtype', 'atol'],
    [
        pytest.param(*t, id="B{}-T{}-d{}-s{}-e{}-{}-atol{}".format(*t))
        for t in [
            (2, 64, 256, 16, 2, torch.float32, 1e-4),
            (2, 64, 128, 16, 2, torch.bfloat16, 1e-2),
            (2, 64, 128, 16, 2, torch.float16, 5e-3),
        ]
    ],
)
def test_mamba_layer_vs_official_inference(B, T, d_model, d_state, expand, dtype, atol):
    """
    Step-by-step inference comparison between custom Mamba and official mamba_ssm.Mamba.
    Starting with empty inference params and passing in identical sequences of input tokens step by step.
    """
    custom, official = _make_mamba_pair(d_model, d_state, expand=expand, dtype=dtype)
    custom.eval()
    official.eval()

    torch.manual_seed(7)
    x = torch.randn(B, T, d_model, dtype=dtype, device=device)

    # Official inference state
    d_inner = expand * d_model
    conv_state = torch.zeros(B, d_inner, custom.conv_kernel_size, device=device, dtype=dtype)
    ssm_state = torch.zeros(B, d_inner, d_state, device=device, dtype=dtype)

    # FLA inference state (using FLACache)
    from fla.models.utils import Cache
    cache = Cache()

    for i in range(T):
        token = x[:, i:i+1, :]

        # FLA step
        with torch.no_grad():
            fla_out, _, returned_cache = custom(token, past_key_values=cache, use_cache=True)
            assert returned_cache is not None
            cache = returned_cache

        # Official step
        # Official Mamba.step takes (hidden_states, conv_state, ssm_state)
        with torch.no_grad():
            official_out, conv_state, ssm_state = official.step(token, conv_state, ssm_state)

        assert fla_out.shape == official_out.shape
        diff = (fla_out - official_out).abs().max().item()
        assert torch.allclose(fla_out, official_out, atol=atol, rtol=0), \
            f"Output mismatch at step {i}: max diff = {diff}"


@pytest.mark.parametrize(
    ['B', 'T', 'd_model', 'd_state', 'expand', 'dtype', 'atol'],
    [
        pytest.param(*t, id="B{}-T{}-d{}-s{}-e{}-{}-atol{}".format(*t))
        for t in [
            (2, 128, 256, 16, 2, torch.float32, 1e-5),
            (2, 256, 128, 16, 2, torch.bfloat16, 1e-2),
            (2, 256, 128, 16, 2, torch.float16, 5e-3),
        ]
    ],
)
def test_mamba_layer_vs_official_train(B, T, d_model, d_state, expand, dtype, atol):
    """
    Training-mode output of custom Mamba must match official mamba_ssm.Mamba.

    Both should use mamba_inner_fn (the fused CUDA kernel) in this mode.
    """
    custom, official = _make_mamba_pair(d_model, d_state, expand=expand, dtype=dtype)
    custom.train()
    official.train()

    torch.manual_seed(7)
    x = torch.randn(B, T, d_model, dtype=dtype, device=device)

    custom_out = custom(x)[0]
    official_out = official(x)

    assert custom_out.shape == official_out.shape, \
        f"Shape mismatch: {custom_out.shape} vs {official_out.shape}"
    diff = (custom_out - official_out).abs().max().item()
    assert torch.allclose(custom_out, official_out, atol=atol, rtol=0), \
        f"Output mismatch (train): max diff = {diff}"


@pytest.mark.parametrize(
    ['B', 'T', 'd_model', 'd_state', 'expand', 'dtype', 'atol'],
    [
        pytest.param(*t, id="B{}-T{}-d{}-s{}-e{}-{}-atol{}".format(*t))
        for t in [
            (2, 128, 256, 16, 2, torch.float32, 1e-4),
            (2, 256, 128, 16, 2, torch.bfloat16, 1e-2),
            (2, 256, 128, 16, 2, torch.float16, 5e-3),
        ]
    ],
)
def test_mamba_layer_vs_official_eval(B, T, d_model, d_state, expand, dtype, atol):
    """
    Eval-mode output of custom Mamba must match official mamba_ssm.Mamba.

    Custom uses selective_scan_fn in eval; official uses mamba_inner_fn.
    Different code paths should still agree numerically.
    """
    custom, official = _make_mamba_pair(d_model, d_state, expand=expand, dtype=dtype)
    custom.eval()
    official.eval()

    torch.manual_seed(7)
    x = torch.randn(B, T, d_model, dtype=dtype, device=device)

    with torch.no_grad():
        custom_out = custom(x)[0]
        official_out = official(x)

    assert custom_out.shape == official_out.shape
    diff = (custom_out - official_out).abs().max().item()
    assert torch.allclose(custom_out, official_out, atol=atol, rtol=0), \
        f"Output mismatch (eval): max diff = {diff}"


@pytest.mark.parametrize(
    ['B', 'T', 'd_model', 'd_state', 'expand', 'dtype', 'atol'],
    [
        pytest.param(*t, id="B{}-T{}-d{}-s{}-e{}-{}-atol{}".format(*t))
        for t in [
            (2, 128, 256, 16, 2, torch.float32, 1e-4),
            (2, 128, 128, 16, 2, torch.bfloat16, 5e-2),
            (2, 128, 128, 16, 2, torch.float16, 5e-3),
        ]
    ],
)
def test_mamba_layer_gradient_vs_official(B, T, d_model, d_state, expand, dtype, atol):
    """
    Gradients of custom Mamba (train) must match official mamba_ssm.Mamba.

    Both use mamba_inner_fn in train mode, so gradients flow through the same kernel.
    We compare both input gradients and parameter gradients.
    """
    custom, official = _make_mamba_pair(d_model, d_state, expand=expand, dtype=dtype)
    custom.train()
    official.train()

    torch.manual_seed(7)
    base = torch.randn(B, T, d_model, dtype=dtype, device=device)
    x_c = base.clone().detach().requires_grad_(True)
    x_o = base.clone().detach().requires_grad_(True)

    custom(x_c)[0].sum().backward()
    official(x_o).sum().backward()

    # Cast to float32 before comparing: the official model keeps some params
    # (D, A_log) in float32 even when constructed with bfloat16, so gradient
    # dtypes can differ between the two models.
    diff = (x_c.grad.float() - x_o.grad.float()).abs().max().item()
    assert torch.allclose(x_c.grad.float(), x_o.grad.float(), atol=atol, rtol=0), \
        f"Input grad mismatch: max diff = {diff}"

    c_grads = {n: p.grad for n, p in custom.named_parameters() if p.grad is not None}
    o_grads = {n: p.grad for n, p in official.named_parameters() if p.grad is not None}
    for name in c_grads:
        if name in o_grads:
            cg, og = c_grads[name].float(), o_grads[name].float()
            diff = (cg - og).abs().max().item()
            assert torch.allclose(cg, og, atol=atol, rtol=0), \
                f"Param '{name}' grad mismatch: max diff = {diff}"
