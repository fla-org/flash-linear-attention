# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest import mock

import pytest
import torch

from fla.modules import ShortConvolution
from fla.modules.convolution import can_fuse_qkv_short_conv, fused_qkv_short_conv
from fla.utils import assert_close, device


def _as_input(x: torch.Tensor) -> torch.Tensor:
    # `ShortConvolution.forward` applies the mask in-place (`x.mul_`), so the
    # tensors we pass in must be non-leaf to avoid PyTorch's leaf-inplace error.
    x = x * 1.0
    x.requires_grad_(True)
    return x


def _make_convs(
    dq: int,
    dk: int,
    dv: int,
    kernel_size: int,
    bias: bool,
    activation: str | None,
    dtype: torch.dtype,
    backend: str = 'triton',
) -> tuple[ShortConvolution, ShortConvolution, ShortConvolution]:
    def _build(d: int) -> ShortConvolution:
        return ShortConvolution(
            hidden_size=d,
            kernel_size=kernel_size,
            bias=bias,
            activation=activation,
            backend=backend,
        ).to(device=device, dtype=dtype)
    return _build(dq), _build(dk), _build(dv)


@pytest.mark.parametrize(
    ('B', 'T', 'dq', 'dk', 'dv', 'bias', 'activation', 'use_mask', 'dtype'),
    [
        pytest.param(*case, id="B{}-T{}-q{}k{}v{}-bias{}-{}-mask{}-{}".format(*case))
        for case in [
            # Equal q/k/v channel counts (DeltaNet / GDN / KDA / Comba style).
            (2, 128, 128, 128, 256, False, 'silu', False, torch.bfloat16),
            (2, 100, 128, 128, 256, True, 'silu', False, torch.float16),
            # Grouped (GQA) channel counts (GLA / GSA / multiscale retention style).
            (2, 128, 128, 64, 256, False, 'silu', False, torch.bfloat16),
            (2, 96, 128, 64, 256, True, 'silu', False, torch.float16),
            # No in-conv activation (LightNet style, silu is applied post-conv).
            (2, 128, 128, 128, 256, False, None, False, torch.bfloat16),
            # Masked dense path (ABC / SimpleGLA / LightNet style).
            (2, 128, 128, 128, 256, False, 'silu', True, torch.bfloat16),
            (1, 64, 128, 64, 256, True, None, True, torch.float16),
        ]
    ],
)
def test_fused_qkv_short_conv_matches_separate(
    B: int,
    T: int,
    dq: int,
    dk: int,
    dv: int,
    bias: bool,
    activation: str | None,
    use_mask: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q_conv, k_conv, v_conv = _make_convs(dq, dk, dv, 4, bias, activation, dtype)
    q_conv.train()
    k_conv.train()
    v_conv.train()

    q_base = torch.randn(B, T, dq, device=device, dtype=dtype)
    k_base = torch.randn(B, T, dk, device=device, dtype=dtype)
    v_base = torch.randn(B, T, dv, device=device, dtype=dtype)
    mask = None
    if use_mask:
        mask = torch.randint(0, 2, (B, T), device=device, dtype=dtype)
    do = (
        torch.randn(B, T, dq, device=device, dtype=dtype),
        torch.randn(B, T, dk, device=device, dtype=dtype),
        torch.randn(B, T, dv, device=device, dtype=dtype),
    )

    # Count calls into the ShortConvolution modules: the fused path bypasses
    # them and issues a single `causal_conv1d`, the separate path calls all three.
    conv_calls = {'n': 0}

    def bump(*_):
        conv_calls['n'] += 1
    handles = [m.register_forward_hook(bump) for m in (q_conv, k_conv, v_conv)]
    try:
        # Reference: three separate short convolutions.
        q, k, v = _as_input(q_base), _as_input(k_base), _as_input(v_base)
        conv_calls['n'] = 0
        q_ref, _ = q_conv(q, mask=mask)
        k_ref, _ = k_conv(k, mask=mask)
        v_ref, _ = v_conv(v, mask=mask)
        sep_calls = conv_calls['n']
        ((q_ref * do[0]).sum() + (k_ref * do[1]).sum() + (v_ref * do[2]).sum()).backward()
        g_sep = dict(_named_params(q_conv, k_conv, v_conv))
        dq_sep = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
        for m in (q_conv, k_conv, v_conv):
            m.zero_grad(set_to_none=True)

        # Fused: a single concatenated short convolution.
        q, k, v = _as_input(q_base), _as_input(k_base), _as_input(v_base)
        conv_calls['n'] = 0
        q_fused, k_fused, v_fused = fused_qkv_short_conv(q, k, v, q_conv, k_conv, v_conv, mask=mask)
        fused_calls = conv_calls['n']
        ((q_fused * do[0]).sum() + (k_fused * do[1]).sum() + (v_fused * do[2]).sum()).backward()
        g_fused = dict(_named_params(q_conv, k_conv, v_conv))
        dq_fused = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
    finally:
        for h in handles:
            h.remove()

    assert fused_calls == 0, f"fused path should bypass the conv modules, saw {fused_calls} call(s)"
    assert sep_calls == 3, f"separate path should call all three conv modules, saw {sep_calls} call(s)"

    assert_close('q', q_ref, q_fused, 0.0)
    assert_close('k', k_ref, k_fused, 0.0)
    assert_close('v', v_ref, v_fused, 0.0)
    for name, ref, tri in zip(('dq', 'dk', 'dv'), dq_sep, dq_fused, strict=False):
        assert_close(name, ref, tri, 1e-3)
    # The fused path concatenates the q/k/v conv weights into one conv, which
    # regroups the backward accumulation for those weights, so the conv-weight
    # grads can differ by tiny accumulation-order noise.
    for name in g_fused:
        assert_close(f'd{name}', g_sep[name], g_fused[name], 5e-3)


def _named_params(q_conv, k_conv, v_conv):
    for prefix, module in (('q', q_conv), ('k', k_conv), ('v', v_conv)):
        for name, param in module.named_parameters():
            yield f'{prefix}.{name}', param.grad.detach().clone()


def test_can_fuse_qkv_short_conv_predicate():
    q_conv, k_conv, v_conv = _make_convs(128, 128, 256, 4, False, 'silu', torch.float32)
    assert can_fuse_qkv_short_conv(q_conv, k_conv, v_conv)

    # Mismatched activation blocks fusion (e.g. DeltaNet with non-silu q/k).
    _, k_conv_mismatch, _ = _make_convs(128, 128, 256, 4, False, None, torch.float32)
    assert not can_fuse_qkv_short_conv(q_conv, k_conv_mismatch, v_conv)

    # Mismatched kernel size blocks fusion.
    _, k_conv_mismatch, _ = _make_convs(128, 128, 256, 3, False, 'silu', torch.float32)
    assert not can_fuse_qkv_short_conv(q_conv, k_conv_mismatch, v_conv)

    # Mismatched backend blocks fusion. Set the attribute directly so the test
    # does not require the optional CUDA `causal_conv1d` package.
    k_conv_mismatch = mock.Mock(wraps=k_conv)
    k_conv_mismatch.backend = 'cuda'
    k_conv_mismatch.activation = k_conv.activation
    k_conv_mismatch.kernel_size = k_conv.kernel_size
    assert not can_fuse_qkv_short_conv(q_conv, k_conv_mismatch, v_conv)
