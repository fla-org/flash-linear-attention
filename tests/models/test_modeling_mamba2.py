
import os

import pytest
import torch

from fla.models import Mamba2Config, Mamba2ForCausalLM
from fla.utils import device

from .test_modeling_base import run_test_generation


# ===================================================================================
# Test for Modeling (Forward/Backward Pass)
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'dtype', 'conv_backend'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-use_l2warp{}-{}-conv-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, True, torch.bfloat16, 'cuda'),
            (4, 4, 1024, 4, 64, False, torch.bfloat16, 'cuda'),
            (4, 4, 1024, 4, 128, False, torch.bfloat16, 'cuda'),
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
    conv_backend: str,
):
    """
    Test the forward and backward pass of the Mamba2 model by manually
    instantiating the configuration and the model.
    """
    os.environ['FLA_CONV_BACKEND'] = conv_backend

    # Manually create a consistent configuration
    # The key relationship is: num_heads = expand * hidden_size / head_dim
    # To ensure consistency, we derive hidden_size from other parameters.
    expand = 2
    hidden_size = H * D // expand

    config = Mamba2Config(
        num_hidden_layers=L,
        hidden_size=hidden_size,
        expand=expand,
        num_heads=H,
        head_dim=D,
        use_l2warp=use_l2warp,
        vocab_size=1000,  # dummy vocab size
    )

    model = Mamba2ForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()

    # Create random input tensor
    x = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Forward pass
    y = model(x)

    # Assert output shape is correct
    assert y.logits.shape == (B, T, config.vocab_size)

    # Backward pass
    y.logits.sum().backward()
    print(f"Test test_modeling passed with H={H}, D={D}, backend={conv_backend}.")


# ===================================================================================
# Test for Generation
# ===================================================================================
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype', 'conv_backend'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}-conv-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, torch.float16, 'cuda'),
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
    conv_backend: str,
):
    os.environ['FLA_CONV_BACKEND'] = conv_backend
    expand = 2
    hidden_size = H * D // expand

    config = Mamba2Config(
        num_hidden_layers=L,
        hidden_size=hidden_size,
        expand=expand,
        num_heads=H,
        head_dim=D,
        vocab_size=1000,
    )
    model = Mamba2ForCausalLM(config).to(device=device, dtype=dtype)
    run_test_generation(L, B, T, H, D, Mamba2Config, dtype, model=model, config=config)


# ===================================================================================
# Layer-level: Custom fla.layers.mamba2.Mamba2 vs Official mamba_ssm.Mamba2
# ===================================================================================

def _copy_params(src, dst):
    """Copy parameters with matching names and shapes from src to dst."""
    dst_dict = dict(dst.named_parameters())
    with torch.no_grad():
        for name, param in src.named_parameters():
            if name in dst_dict and dst_dict[name].shape == param.shape:
                dst_dict[name].data.copy_(param.data)


def _make_mamba2_pair(d_model, d_state=128, headdim=64, ngroups=1, expand=2,
                      dtype=torch.float32, official_use_mem_eff_path=True):
    """Create custom & official Mamba2 layers sharing identical weights.

    Args:
        official_use_mem_eff_path: Controls whether the official Mamba2 uses the
            fused kernel path (True) or the split conv+scan path (False).
            The custom layer uses the fused path in train and split path in eval.

    Returns (custom, official) with all parameters copied from custom → official.
    """
    pytest.importorskip("mamba_ssm")
    pytest.importorskip("causal_conv1d")
    from mamba_ssm.modules.mamba2 import Mamba2 as OfficialMamba2

    from fla.layers.mamba2 import Mamba2 as CustomMamba2

    os.environ['FLA_CONV_BACKEND'] = 'cuda'
    nheads = expand * d_model // headdim

    torch.manual_seed(42)
    custom = CustomMamba2(
        num_heads=nheads,
        head_dim=headdim,
        hidden_size=d_model,
        state_size=d_state,
        expand=expand,
        n_groups=ngroups,
        conv_kernel=4,
        use_conv_bias=True,
        hidden_act="silu",
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        use_bias=False,
        norm_eps=1e-5,
        chunk_size=256,
        layer_idx=0,
        backend="cuda",
    ).to(dtype=dtype, device=device)

    official = OfficialMamba2(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        expand=expand,
        headdim=headdim,
        ngroups=ngroups,
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=official_use_mem_eff_path,
        device=device,
        dtype=dtype,
    )

    _copy_params(custom, official)
    return custom, official


@pytest.mark.parametrize(
    ['B', 'T', 'd_model', 'd_state', 'headdim', 'expand', 'dtype', 'atol'],
    [
        pytest.param(*t, id="B{}-T{}-d{}-s{}-hd{}-e{}-{}-atol{}".format(*t))
        for t in [
            (2, 64, 256, 128, 64, 2, torch.float32, 1e-4),
            (2, 64, 256, 64, 64, 2, torch.bfloat16, 1e-2),
            (2, 64, 256, 64, 64, 2, torch.float16, 5e-3),
        ]
    ],
)
def test_mamba2_layer_vs_official_inference(B, T, d_model, d_state, headdim, expand, dtype, atol):
    """
    Step-by-step inference comparison between custom Mamba2 and official mamba_ssm.Mamba2.
    Starting with empty inference params and passing in identical sequences of input tokens step by step.
    """
    custom, official = _make_mamba2_pair(
        d_model, d_state, headdim, expand=expand, dtype=dtype,
        official_use_mem_eff_path=False,
    )
    custom.eval()
    official.eval()

    torch.manual_seed(7)
    x = torch.randn(B, T, d_model, dtype=dtype, device=device)

    # Official inference state
    # Mamba2 official doesn't have a simple 'allocate_inference_cache' like Mamba1 in some versions,
    # but we can use the same logic as in their generation.
    # For Mamba2, the state is (conv_state, ssm_state)
    d_inner = expand * d_model
    nheads = d_inner // headdim
    conv_state = torch.zeros(B, custom.conv_dim, custom.conv_kernel_size, device=device, dtype=dtype)
    ssm_state = torch.zeros(B, nheads, headdim, d_state, device=device, dtype=dtype)

    # FLA inference state (using Cache)
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
        # Official Mamba2.step takes (hidden_states, conv_state, ssm_state)
        with torch.no_grad():
            official_out, conv_state, ssm_state = official.step(token, conv_state, ssm_state)

        assert fla_out.shape == official_out.shape
        diff = (fla_out - official_out).abs().max().item()
        assert torch.allclose(fla_out, official_out, atol=atol, rtol=0), \
            f"Output mismatch at step {i}: max diff = {diff}"


@pytest.mark.parametrize(
    ['B', 'T', 'd_model', 'd_state', 'headdim', 'expand', 'dtype', 'atol'],
    [
        pytest.param(*t, id="B{}-T{}-d{}-s{}-hd{}-e{}-{}-atol{}".format(*t))
        for t in [
            (2, 128, 256, 128, 64, 2, torch.float32, 1e-5),
            (2, 256, 128, 64, 64, 2, torch.bfloat16, 5e-3),
            (2, 256, 128, 64, 64, 2, torch.float16, 5e-3),
        ]
    ],
)
def test_mamba2_layer_vs_official_train(B, T, d_model, d_state, headdim, expand, dtype, atol):
    """Training-mode output of custom Mamba2 must match official mamba_ssm.Mamba2.

    Both should use mamba_split_conv1d_scan_combined (the fused kernel) in this mode.
    """
    custom, official = _make_mamba2_pair(
        d_model, d_state, headdim, expand=expand, dtype=dtype,
        official_use_mem_eff_path=True,
    )
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
    ['B', 'T', 'd_model', 'd_state', 'headdim', 'expand', 'dtype', 'atol'],
    [
        pytest.param(*t, id="B{}-T{}-d{}-s{}-hd{}-e{}-{}-atol{}".format(*t))
        for t in [
            (2, 128, 256, 128, 64, 2, torch.float32, 1e-4),
            (2, 256, 256, 64, 64, 2, torch.bfloat16, 1e-2),
            (2, 256, 256, 64, 64, 2, torch.float16, 5e-3),
        ]
    ],
)
def test_mamba2_layer_vs_official_eval(B, T, d_model, d_state, headdim, expand, dtype, atol):
    """Eval-mode output of custom Mamba2 must match official mamba_ssm.Mamba2.

    Both use the non-fused path: causal_conv1d_fn + mamba_chunk_scan_combined.
    (Official is created with use_mem_eff_path=False to force the split path.)

    Note: bfloat16 tests need projection_size = 2*d_inner + 2*ngroups*d_state + nheads
    to be divisible by 8, because the official passes a non-contiguous transposed
    tensor to causal_conv1d_fn whose channel-last path requires 8-byte alignment.
    """
    custom, official = _make_mamba2_pair(
        d_model, d_state, headdim, expand=expand, dtype=dtype,
        official_use_mem_eff_path=False,
    )
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
    ['B', 'T', 'd_model', 'd_state', 'headdim', 'expand', 'dtype', 'atol'],
    [
        pytest.param(*t, id="B{}-T{}-d{}-s{}-hd{}-e{}-{}-atol{}".format(*t))
        for t in [
            (2, 128, 256, 128, 64, 2, torch.float32, 1e-4),
            (2, 128, 256, 64, 64, 2, torch.bfloat16, 5e-2),
            (2, 128, 256, 64, 64, 2, torch.float16, 5e-3),
        ]
    ],
)
def test_mamba2_layer_gradient_vs_official(B, T, d_model, d_state, headdim, expand, dtype, atol):
    """Gradients of custom Mamba2 (train) must match official mamba_ssm.Mamba2.

    Both use the fused mamba_split_conv1d_scan_combined kernel in train mode,
    so gradients should flow through the same computation.
    We compare both input gradients and parameter gradients.
    """
    custom, official = _make_mamba2_pair(
        d_model, d_state, headdim, expand=expand, dtype=dtype,
        official_use_mem_eff_path=True,
    )
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
