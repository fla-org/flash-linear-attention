# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib.util

import pytest
import torch

from fla.models import Mamba3Config, Mamba3ForCausalLM, Mamba3Model
from fla.utils import device


def _cuda_runtime_available() -> bool:
    """Real runtime check, not just `torch.cuda.is_available()`.

    The driver may be too old for the installed torch build, in which case
    `is_available()` is True but any allocation raises.
    """
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception:
        return False
    return True


def _triton_at_least(major: int, minor: int) -> bool:
    try:
        import triton
        parts = triton.__version__.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


def _mamba3_siso_kernel_available() -> bool:
    if not _cuda_runtime_available():
        return False
    # Upstream SISO kernel uses tl.make_tensor_descriptor (triton 3.5+).
    if not _triton_at_least(3, 5):
        return False
    if importlib.util.find_spec("mamba_ssm") is None:
        return False
    try:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined  # noqa: F401
    except Exception:
        return False
    return True


def _mamba3_mimo_kernel_available() -> bool:
    if not _cuda_runtime_available():
        return False
    if not _triton_at_least(3, 5):
        return False
    if importlib.util.find_spec("mamba_ssm") is None:
        return False
    try:
        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo  # noqa: F401
    except Exception:
        return False
    return True


requires_mamba3_kernels = pytest.mark.skipif(
    not _mamba3_siso_kernel_available(),
    reason="Mamba-3 SISO kernels or CUDA runtime not available in this env.",
)

requires_mamba3_mimo = pytest.mark.skipif(
    not _mamba3_mimo_kernel_available(),
    reason="Mamba-3 MIMO (TileLang) kernels or CUDA runtime not available in this env.",
)


# ---------------------------------------------------------------------------
# Construction / metadata
# ---------------------------------------------------------------------------
def test_mamba3_construction_cpu():
    """Construct the config, mixer, and ForCausalLM head without any CUDA path."""
    cfg = Mamba3Config(
        num_hidden_layers=2,
        hidden_size=128,
        state_size=16,
        head_dim=32,
        expand=2,
        n_groups=1,
        vocab_size=256,
        chunk_size=64,
    )
    # Config invariants.
    assert cfg.model_type == "mamba3"
    assert cfg.num_heads == cfg.expand * cfg.hidden_size // cfg.head_dim

    from fla.layers.mamba3 import Mamba3

    mixer = Mamba3(
        hidden_size=cfg.hidden_size,
        state_size=cfg.state_size,
        expand=cfg.expand,
        head_dim=cfg.head_dim,
        n_groups=cfg.n_groups,
        chunk_size=cfg.chunk_size,
        layer_idx=0,
    )
    # Projection width matches the split we plan to do inside forward().
    expected = (
        2 * mixer.intermediate_size
        + 2 * cfg.state_size * cfg.n_groups * mixer.mimo_rank
        + 3 * cfg.num_heads
        + mixer.num_rope_angles
    )
    assert mixer.in_proj.out_features == expected

    # Parameter init matches the Mamba-3 reference: softplus(dt_bias) in range,
    # D/B_bias/C_bias all ones.
    dt = torch.nn.functional.softplus(mixer.dt_bias)
    assert (dt >= cfg.dt_init_floor - 1e-6).all()
    assert (dt <= cfg.dt_max + 1e-4).all()
    assert torch.allclose(mixer.D, torch.ones_like(mixer.D))
    assert torch.allclose(mixer.B_bias, torch.ones_like(mixer.B_bias))
    assert torch.allclose(mixer.C_bias, torch.ones_like(mixer.C_bias))

    # CausalLM model builds and is on the expected device (CPU here).
    model = Mamba3ForCausalLM(cfg)
    assert isinstance(model.backbone, Mamba3Model)
    assert model.lm_head.out_features == cfg.vocab_size


def test_mamba3_mimo_init_shapes():
    """MIMO projections must have shape (H, R, D) with the documented init values."""
    from fla.layers.mamba3 import Mamba3

    mixer = Mamba3(
        hidden_size=128, state_size=16, expand=2, head_dim=32, n_groups=1,
        is_mimo=True, mimo_rank=2, layer_idx=0,
    )
    assert mixer.mimo_x.shape == (mixer.num_heads, 2, mixer.head_dim)
    # reference: mimo_x / mimo_o = 1/R, mimo_z = 1
    assert torch.allclose(mixer.mimo_x, torch.full_like(mixer.mimo_x, 0.5))
    assert torch.allclose(mixer.mimo_o, torch.full_like(mixer.mimo_o, 0.5))
    assert torch.allclose(mixer.mimo_z, torch.ones_like(mixer.mimo_z))


def test_mamba3_cpu_forward_raises():
    """Mamba-3 mixer must refuse to run on CPU rather than silently do the wrong thing.

    We call the mixer directly — the ForCausalLM path also trips over FLA's
    RMSNorm on CPU (its custom_device_ctx helper is CUDA-only), which would
    mask the mixer's own guard.
    """
    from fla.layers.mamba3 import Mamba3

    mixer = Mamba3(
        hidden_size=64, state_size=16, expand=2, head_dim=32,
        n_groups=1, chunk_size=32, layer_idx=0,
    )
    x = torch.randn(2, 8, 64)
    with pytest.raises(NotImplementedError, match="CUDA"):
        mixer(x)


# ---------------------------------------------------------------------------
# Kernel-gated smoke tests (skip if mamba_ssm mamba3 kernels are missing)
# ---------------------------------------------------------------------------
@requires_mamba3_kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mamba3_modeling_forward(dtype):
    """Build a tiny Mamba3ForCausalLM and run one forward + backward pass."""
    cfg = Mamba3Config(
        num_hidden_layers=2,
        hidden_size=128,
        state_size=16,
        head_dim=32,
        expand=2,
        n_groups=1,
        vocab_size=256,
        chunk_size=64,
    )
    model = Mamba3ForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()

    B, T = 2, 128
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = model(x)
    assert y.logits.shape == (B, T, cfg.vocab_size)
    y.logits.sum().backward()


@requires_mamba3_kernels
def test_mamba3_attention_mask_zeros_padding_effect():
    """Padded positions must not influence the SSM state.

    We build two inputs: one of length T, and one of length T padded to T+K with
    zeros + attention_mask. The unpadded prefix of the masked output should
    match the unpadded run up to numerical tolerance.
    """
    cfg = Mamba3Config(
        num_hidden_layers=1, hidden_size=128, state_size=16, head_dim=32,
        expand=2, n_groups=1, vocab_size=128, chunk_size=64,
    )
    model = Mamba3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    model.eval()

    torch.manual_seed(0)
    B, T, K = 2, 96, 32
    x = torch.randint(1, cfg.vocab_size, (B, T), device=device)
    pad = torch.zeros((B, K), dtype=x.dtype, device=device)
    x_padded = torch.cat([x, pad], dim=1)
    mask = torch.cat([torch.ones(B, T, dtype=torch.long, device=device),
                      torch.zeros(B, K, dtype=torch.long, device=device)], dim=1)

    with torch.no_grad():
        y_ref = model(x).logits
        y_masked = model(x_padded, attention_mask=mask).logits[:, :T]

    # bf16 slack; the real check is "padding did not corrupt the prefix".
    assert torch.allclose(y_ref, y_masked, atol=5e-2, rtol=0), \
        f"padding leaked into output: max diff = {(y_ref - y_masked).abs().max().item()}"


@requires_mamba3_kernels
def test_mamba3_cache_path_single_step():
    """Prefill + one-step decode with a Cache must not raise and must return logits."""
    cfg = Mamba3Config(
        num_hidden_layers=1, hidden_size=128, state_size=16, head_dim=32,
        expand=2, n_groups=1, vocab_size=128, chunk_size=64,
    )
    model = Mamba3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    model.eval()

    from fla.models.utils import Cache
    cache = Cache()
    B, T = 2, 32
    x = torch.randint(1, cfg.vocab_size, (B, T), device=device)
    with torch.no_grad():
        out = model(x, past_key_values=cache, use_cache=True)
        nxt = torch.randint(1, cfg.vocab_size, (B, 1), device=device)
        out_step = model(nxt, past_key_values=out.past_key_values, use_cache=True)
    assert out.logits.shape == (B, T, cfg.vocab_size)
    assert out_step.logits.shape == (B, 1, cfg.vocab_size)


@requires_mamba3_mimo
def test_mamba3_mimo_forward():
    """MIMO prefill path must actually dispatch to the TileLang MIMO kernel and return logits."""
    cfg = Mamba3Config(
        num_hidden_layers=1,
        hidden_size=128,
        state_size=16,
        head_dim=32,
        expand=2,
        n_groups=1,
        vocab_size=128,
        chunk_size=64,
        is_mimo=True,
        mimo_rank=2,
    )
    model = Mamba3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    model.eval()

    B, T = 2, 64
    x = torch.randint(1, cfg.vocab_size, (B, T), device=device)
    with torch.no_grad():
        y = model(x).logits
    assert y.shape == (B, T, cfg.vocab_size)
