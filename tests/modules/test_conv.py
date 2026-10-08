# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.modules.convolution import (
    ShortConvolution,
    causal_conv1d,
    causal_conv1d_bwd,
    causal_conv1d_fwd,
    causal_conv1d_update,
)
from fla.ops.utils import prepare_chunk_indices
from fla.utils import IS_NVIDIA, assert_close, device

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


_CONV_REF_FP32_DTYPES = (torch.float16, torch.bfloat16)


def _conv_ref_compute_dtype(*tensors: torch.Tensor | None) -> torch.dtype:
    if any(t is not None and t.dtype in _CONV_REF_FP32_DTYPES for t in tensors):
        return torch.float32
    return tensors[0].dtype


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """PyTorch convolution with channel-first input and a W-1-token initial state."""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    compute_dtype = _conv_ref_compute_dtype(x, weight, bias, initial_state)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    weight_conv = weight.to(compute_dtype)
    bias_conv = bias.to(compute_dtype) if bias is not None else None
    if initial_state is None:
        x_full = x
        out = F.conv1d(x_full.to(compute_dtype), weight_conv.unsqueeze(1), bias_conv, padding=width - 1, groups=dim)
    else:
        x_full = torch.cat([initial_state, x], dim=-1)
        out = F.conv1d(x_full.to(compute_dtype), weight_conv.unsqueeze(1), bias_conv, padding=0, groups=dim)
    out = out[..., :seqlen]
    if output_final_state:
        final_state = F.pad(x_full, (width - 1 - x_full.shape[-1], 0)).to(dtype_in)
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not output_final_state else (out, final_state)


def causal_conv1d_update_ref(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """PyTorch decoding reference that updates the W-token cache in place."""
    dtype = x.dtype
    compute_dtype = _conv_ref_compute_dtype(x, weight, bias, cache)
    squeeze = x.ndim == 2
    if squeeze:
        x = x.unsqueeze(-1)
    B, D, T = x.shape
    W = weight.shape[-1]
    assert cache.shape == (B, D, W)
    assert weight.shape[0] == D
    x_full = torch.cat([cache, x], dim=-1).to(compute_dtype)
    cache.copy_(x_full[:, :, -W:].to(dtype))
    out = F.conv1d(
        x_full,
        weight.to(compute_dtype).unsqueeze(1),
        bias.to(compute_dtype) if bias is not None else None,
        padding=0,
        groups=D,
    )[:, :, -T:]
    if squeeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype)


@pytest.fixture
def conv_backend_calls(monkeypatch: pytest.MonkeyPatch) -> list[str] | None:
    from fla.modules.backends.gluon import GluonBackend

    if not GluonBackend.is_available():
        return None
    from fla.modules.backends.gluon import causal_conv1d

    calls = []
    fwd, bwd = causal_conv1d.causal_conv1d_fwd, causal_conv1d.causal_conv1d_bwd

    def forward(*args, **kwargs):
        calls.append('fwd')
        return fwd(*args, **kwargs)

    def backward(*args, **kwargs):
        calls.append('bwd')
        return bwd(*args, **kwargs)

    monkeypatch.setattr(causal_conv1d, 'causal_conv1d_fwd', forward)
    monkeypatch.setattr(causal_conv1d, 'causal_conv1d_bwd', backward)
    return calls


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype', 'backend'),
    [
        pytest.param(*test, id="B{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}_{8}".format(*test))
        for test in [
            (2, 64, 128, 3, "swish", True, True, torch.float32, 'triton'),
            (2, 128, 128, 4, "swish", False, True, torch.float32, 'triton'),
            (2, 64, 128, 3, "swish", True, False, torch.float32, 'triton'),
            (2, 128, 128, 4, "swish", False, False, torch.float32, 'triton'),
            (2, 500, 1024, 3, None, True, True, torch.float32, 'cuda'),
            (2, 1024, 1024, 4, None, False, True, torch.float32, 'triton'),
            (2, 64, 128, 3, None, True, False, torch.float16, 'triton'),
            (2, 128, 128, 4, None, False, False, torch.float16, 'triton'),
            (2, 64, 128, 3, "swish", True, True, torch.float32, 'cuda'),
            (2, 128, 128, 4, "swish", False, True, torch.float32, 'cuda'),
            (2, 64, 128, 3, "swish", True, False, torch.float32, 'cuda'),
            (2, 128, 128, 4, "swish", False, False, torch.float32, 'cuda'),
        ]
    ],
)
def test_conv(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if backend == 'cuda':
        if causal_conv1d_fn is None:
            pytest.skip("causal_conv1d is not installed for CUDA backend")
        if not IS_NVIDIA:
            pytest.skip("CUDA backend requires an NVIDIA GPU")
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn(B, T, D).to(device, dtype)

    ref = causal_conv1d_ref(
        x=rearrange(x, "b t d -> b d t"),
        weight=weight,
        bias=bias,
        activation=activation,
    )
    ref = rearrange(ref, "b d t -> b t d")
    if has_residual:
        ref += residual
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri, _ = causal_conv1d(x=x, weight=weight, bias=bias, residual=residual, activation=activation, backend=backend)
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None
    if has_residual:
        tri_dr, residual.grad = residual.grad, None

    assert_close(" y", ref, tri, 1e-3)
    assert_close("dx", ref_dx, tri_dx, 1e-3)
    assert_close("dw", ref_dw, tri_dw, 1e-3)
    if has_bias:
        assert_close("db", ref_db, tri_db, 1e-3)
    if has_residual:
        assert_close("dr", ref_dr, tri_dr, 1e-3)


@pytest.mark.parametrize(
    ('N', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype', 'backend'),
    [
        pytest.param(*test, id="N{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}_{8}".format(*test))
        for test in [
            (4, 500, 128, 3, "swish", True, True, torch.float32, 'triton'),
            (4, 1024, 200, 4, "swish", False, True, torch.float32, 'triton'),
            (4, 500, 128, 3, None, True, False, torch.float16, 'triton'),
            (4, 1024, 1024, 4, None, False, False, torch.float16, 'triton'),
            (4, 500, 128, 3, "swish", True, True, torch.float32, 'cuda'),
            (4, 1024, 200, 4, "swish", False, True, torch.float32, 'cuda'),
            (4, 500, 128, 3, None, True, False, torch.float16, 'cuda'),
            (4, 1024, 1024, 4, None, False, False, torch.float16, 'cuda'),
        ]
    ],
)
def test_conv_varlen(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if backend == 'cuda':
        if causal_conv1d_fn is None:
            pytest.skip("causal_conv1d is not installed for CUDA backend")
        if not IS_NVIDIA:
            pytest.skip("CUDA backend requires an NVIDIA GPU")
    torch.manual_seed(42)
    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long),
    ], 0).to(device).sort()[0]

    x = torch.randn(1, T, D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn(1, T, D).to(device, dtype)

    ref = torch.cat([
        rearrange(
            causal_conv1d_ref(
                x=rearrange(x[:, bos:eos].contiguous(), "b t d -> b d t"),
                weight=weight,
                bias=bias,
                activation=activation,
            ),
            "b t d -> b d t",
        ) + (residual[:, bos:eos] if has_residual else torch.zeros_like(x[:, bos:eos]))
        for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
    ], 1)
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri, _ = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        activation=activation,
        backend=backend,
        cu_seqlens=cu_seqlens,
    )
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None
    if has_residual:
        tri_dr, residual.grad = residual.grad, None

    assert_close(" y", ref, tri, 1e-3)
    assert_close("dx", ref_dx, tri_dx, 1e-3)
    assert_close("dw", ref_dw, tri_dw, 1e-3)
    if has_bias:
        assert_close("db", ref_db, tri_db, 1e-3)
    if has_residual:
        assert_close("dr", ref_dr, tri_dr, 1e-3)


@pytest.mark.parametrize(
    ('N', 'T', 'D', 'W', 'activation', 'has_bias', 'dtype'),
    [
        pytest.param(*test, id="N{}_T{}_D{}_W{}_activation{}_has_bias{}_{}".format(*test))
        for test in [
            (4, 1024, 4096, 3, "swish", True, torch.float32),
            (4, 1024, 4096, 4, "swish", False, torch.float32),
            (4, 1024, 4096, 3, None, True, torch.float16),
            (4, 1024, 4096, 4, None, False, torch.float16),
        ]
    ],
)
def test_fast_conv_varlen(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if causal_conv1d_fn is None:
        pytest.skip("causal_conv1d is not installed for CUDA backend")
    if not IS_NVIDIA:
        pytest.skip("fast_causal_conv1d requires an NVIDIA GPU")
    from fla.modules.convolution import fast_causal_conv1d_fn

    cu_seqlens = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 16)[:N-1]],
        torch.tensor([T], dtype=torch.long),
    ], 0).to(device).sort()[0]

    x = torch.randn(1, T, D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    dy = torch.randn(1, T, D).to(device, dtype)

    ref = torch.cat([
        rearrange(
            causal_conv1d_ref(
                x=rearrange(x[:, bos:eos].contiguous(), "b t d -> b d t"),
                weight=weight,
                bias=bias,
                activation=activation,
            ),
            "b t d -> b d t",
        )
        for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
    ], 1)
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None

    tri, _ = fast_causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.cpu(),
    )
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None

    assert_close(" y", ref, tri, 1e-3)
    assert_close("dx", ref_dx, tri_dx, 1e-3)
    assert_close("dw", ref_dw, tri_dw, 1e-3)
    if has_bias:
        assert_close("db", ref_db, tri_db, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'output_final_state', 'dtype'),
    [
        pytest.param(*test, id="B{}_T{}_D{}_W{}_{}_bias{}_residual{}_state{}_{}".format(*test))
        for test in [
            (2, 64, 100, 3, 'swish', True, True, True, torch.float32),
            (2, 128, 128, 4, 'swish', True, True, True, torch.float32),
            (3, 128, 128, 4, 'swish', True, True, True, torch.float32),
            (3, 128, 256, 4, 'swish', True, True, True, torch.float32),
            (3, 128, 512, 4, 'swish', True, True, True, torch.float32),
            (2, 128, 1024, 4, 'swish', True, True, True, torch.float32),
            (2, 128, 2048, 3, 'swish', True, True, True, torch.float32),
            (2, 128, 4096, 4, 'swish', True, True, True, torch.float32),
            (2, 128, 8192, 4, 'swish', True, True, True, torch.float32),
            (1, 1, 128, 4, 'swish', True, True, True, torch.float32),
            (2, 2, 128, 4, 'swish', True, True, True, torch.float32),
            (1, 1, 64, 3, 'swish', True, True, True, torch.float32),
            (2, 64, 100, 3, 'swish', True, False, False, torch.float32),
            (2, 128, 128, 4, 'swish', True, False, False, torch.float32),
            (3, 128, 128, 4, 'swish', False, False, False, torch.float32),
            (2, 64, 256, 4, 'swish', True, True, False, torch.float32),
            (2, 128, 512, 4, None, True, False, False, torch.float32),
            (2, 64, 128, 3, 'swish', True, False, False, torch.float16),
            (1, 8192, 4096, 4, 'swish', True, False, False, torch.float32),
            (1, 8192, 8192, 4, 'swish', True, False, False, torch.float32),
        ]
    ],
)
def test_conv_initial_state(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    output_final_state: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(D, W, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=dtype, requires_grad=True) if has_bias else None
    residual = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True) if has_residual else None
    h0 = F.pad(torch.randn(B, D, W - 1, device=device, dtype=dtype), (1, 0)).requires_grad_(True)
    dy = torch.randn_like(x)
    dht = torch.randn_like(h0[..., 1:]) if output_final_state else None

    ref, ref_ht = causal_conv1d_ref(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        initial_state=h0[..., 1:],
        output_final_state=True,
        activation=activation,
    )
    ref = ref.transpose(1, 2)
    if has_residual:
        ref = ref + residual
    ref_loss = (ref * dy).sum()
    if output_final_state:
        ref_loss = ref_loss + (ref_ht * dht).sum()
    ref_loss.backward()
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    ref_dh0, h0.grad = h0.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None

    tri, tri_ht = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=h0,
        output_final_state=output_final_state,
        activation=activation,
    )
    tri_loss = (tri * dy).sum()
    if output_final_state:
        tri_loss = tri_loss + (tri_ht[..., 1:] * dht).sum()
    tri_loss.backward()
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    tri_dh0, h0.grad = h0.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None
    if has_residual:
        tri_dr, residual.grad = residual.grad, None

    assert_close('y', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
    assert_close('dw', ref_dw, tri_dw, 1e-3)
    assert_close('dh0', ref_dh0, tri_dh0, 1e-3)
    if has_bias:
        assert_close('db', ref_db, tri_db, 1e-3)
    if has_residual:
        assert_close('dr', ref_dr, tri_dr, 1e-3)
    if output_final_state:
        assert_close('ht', ref_ht, tri_ht[..., 1:], 1e-3)


@pytest.mark.parametrize(
    ('T', 'D', 'lengths'),
    [
        pytest.param(256, 128, None, id='random_split_T256'),
        pytest.param(None, 4096, [32] * 128 + [8192], id='packed_128x32_D4096'),
        pytest.param(None, 8192, [32] * 128 + [8192], id='packed_128x32_D8192'),
    ],
)
def test_conv_varlen_initial_state(T: int | None, D: int, lengths: list[int] | None):
    torch.manual_seed(42)
    W = 4
    dtype = torch.float32
    if lengths is None:
        split = int(torch.randint(low=W, high=T - W, size=(1,)).item())
        cu_seqlens = torch.tensor([0, split, T], device=device, dtype=torch.int32)
    else:
        T = sum(lengths)
        cu_seqlens = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()], device=device, dtype=torch.int32)
    N = cu_seqlens.numel() - 1
    x = torch.randn(1, T, D, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(D, W, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=dtype, requires_grad=True)
    h0 = F.pad(torch.randn(N, D, W - 1, device=device, dtype=dtype), (1, 0)).requires_grad_(True)
    dy = torch.randn_like(x)

    ref = torch.cat([
        causal_conv1d_ref(
            x=x[:, bos:eos].transpose(1, 2),
            weight=weight,
            bias=bias,
            initial_state=h0[i:i+1, :, 1:].contiguous(),
            activation='swish',
        ).transpose(1, 2)
        for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:]))
    ], dim=1)
    ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    ref_db, bias.grad = bias.grad, None
    ref_dh0, h0.grad = h0.grad, None

    tri, _ = causal_conv1d(x=x, weight=weight, bias=bias, initial_state=h0, activation='swish', cu_seqlens=cu_seqlens)
    tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    tri_db, bias.grad = bias.grad, None
    tri_dh0, h0.grad = h0.grad, None

    assert_close('y', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
    assert_close('dw', ref_dw, tri_dw, 1e-3)
    assert_close('db', ref_db, tri_db, 1e-3)
    assert_close('dh0', ref_dh0, tri_dh0, 1e-3)


@pytest.mark.parametrize(
    ('B', 'N', 'T', 'D', 'W', 'is_varlen', 'activation', 'has_bias', 'dtype'),
    [
        pytest.param(*test, id="B{}_N{}_T{}_D{}_W{}_varlen{}_activation{}_has_bias{}_{}".format(*test))
        for test in [
            (2, 2, 64, 128, 3, False, "swish", True, torch.float32),
            (2, 2, 128, 128, 4, False, "swish", False, torch.float32),
            (2, 2, 64, 128, 3, False, None, True, torch.float16),
            (1, 4, 128, 64, 3, True, "swish", True, torch.float32),
            (1, 4, 256, 128, 4, True, "swish", False, torch.float32),
            (1, 2, 64, 128, 3, True, None, True, torch.float16),
        ]
    ],
)
@pytest.mark.parametrize(
    ('index', 'has_residual', 'has_initial_state'),
    [(0, False, False), (1, False, False), (2, False, False), (1, True, False), (1, False, True)],
    ids=['q', 'k', 'v', 'residual', 'state'],
)
def test_conv_non_contiguous_qkv(
    B: int,
    N: int,
    T: int,
    D: int,
    W: int,
    is_varlen: bool,
    activation: str | None,
    has_bias: bool,
    dtype: torch.dtype,
    index: int,
    has_residual: bool,
    has_initial_state: bool,
):
    torch.manual_seed(42)
    cu_seqlens = None
    if is_varlen:
        lengths = [T // N] * N
        lengths[-1] += T % N
        cu_seqlens = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()], device=device, dtype=torch.int32)

    qkv = torch.randn(B, T, 3 * D).to(device, dtype)
    x = qkv[..., index * D:(index + 1) * D]
    assert not x.is_contiguous()
    ref_x = x.contiguous().requires_grad_(True)
    tri_x = x.detach().requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    residual = x.clone().requires_grad_(True) if has_residual else None
    h0 = torch.randn(N, D, W).to(device, dtype).requires_grad_(True) if has_initial_state else None
    dy = torch.randn_like(ref_x)

    ref, ref_ht = causal_conv1d(
        x=ref_x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=h0,
        output_final_state=has_initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    ref.backward(dy)
    ref_dx, ref_x.grad = ref_x.grad, None
    ref_dw, weight.grad = weight.grad, None
    if has_bias:
        ref_db, bias.grad = bias.grad, None
    if has_residual:
        ref_dr, residual.grad = residual.grad, None
    if has_initial_state:
        ref_dh0, h0.grad = h0.grad, None

    tri, tri_ht = causal_conv1d(
        x=tri_x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=h0,
        output_final_state=has_initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    tri.backward(dy)
    tri_dx, tri_x.grad = tri_x.grad, None
    tri_dw, weight.grad = weight.grad, None
    if has_bias:
        tri_db, bias.grad = bias.grad, None
    if has_residual:
        tri_dr, residual.grad = residual.grad, None
    if has_initial_state:
        tri_dh0, h0.grad = h0.grad, None

    assert_close("y", ref, tri, 1e-3)
    assert_close("dx", ref_dx, tri_dx, 1e-3)
    assert_close("dw", ref_dw, tri_dw, 1e-3)
    if has_bias:
        assert_close("db", ref_db, tri_db, 1e-3)
    if has_residual:
        assert_close("dr", ref_dr, tri_dr, 1e-3)
    if has_initial_state:
        assert_close("ht", ref_ht, tri_ht, 1e-3)
        assert_close("dh0", ref_dh0, tri_dh0, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'activation', 'dtype'),
    [
        pytest.param(*test, id="B{0}_T{1}_D{2}_W{3}_activation{4}_{5}".format(*test))
        for test in [
            (2, 64, 128, 4, None, torch.float32),
            (2, 128, 128, 3, "silu", torch.float32),
            (1, 15, 64, 2, None, torch.bfloat16),
            (4, 300, 32, 4, "silu", torch.bfloat16),
        ]
    ],
)
@pytest.mark.parametrize('layout', ['strided', 'broadcast'])
def test_conv_non_contiguous_dy(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    dtype: torch.dtype,
    layout: str,
):
    torch.manual_seed(42)
    weight = torch.randn(D, W, device=device, dtype=dtype).requires_grad_(True)
    h0 = torch.randn(B, D, W, device=device, dtype=dtype).requires_grad_(True)
    ref_x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    tri_x = ref_x.detach().clone().requires_grad_(True)

    ref, _ = causal_conv1d(x=ref_x, weight=weight, initial_state=h0, activation=activation)
    dy = torch.randn_like(ref) if layout == 'strided' else torch.ones_like(ref)
    ref.backward(dy)
    ref_dx, ref_x.grad = ref_x.grad, None
    ref_dw, weight.grad = weight.grad, None
    ref_dh0, h0.grad = h0.grad, None

    tri, _ = causal_conv1d(x=tri_x, weight=weight, initial_state=h0, activation=activation)
    if layout == 'strided':
        dy = torch.cat([dy, torch.zeros_like(dy), torch.zeros_like(dy)], dim=-1)[..., :D]
        assert not dy.is_contiguous()
        tri.backward(dy)
    else:
        tri.sum().backward()
    tri_dx, tri_x.grad = tri_x.grad, None
    tri_dw, weight.grad = weight.grad, None
    tri_dh0, h0.grad = h0.grad, None

    assert_close("dx", ref_dx, tri_dx, 1e-3)
    assert_close("dw", ref_dw, tri_dw, 1e-3)
    assert_close("dh0", ref_dh0, tri_dh0, 1e-3)


@pytest.mark.skipif(not IS_NVIDIA, reason='Gluon convolution requires NVIDIA')
@pytest.mark.parametrize(
    ('dtype', 'weight_dtype'),
    [
        pytest.param(torch.float32, torch.float32, id='fp32-fp32'),
        pytest.param(torch.float16, torch.float32, id='fp16-fp32'),
        pytest.param(torch.bfloat16, torch.float32, id='bf16-fp32'),
        pytest.param(torch.float16, torch.float16, id='fp16-fp16'),
        pytest.param(torch.bfloat16, torch.bfloat16, id='bf16-bf16'),
    ],
)
@pytest.mark.parametrize('activation', [None, 'silu'], ids=['linear', 'silu'])
@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'is_varlen', 'has_initial_state', 'non_contiguous'),
    [
        pytest.param(2, 1, 33, 4, False, False, False, id='one-token'),
        pytest.param(2, 63, 65, 3, False, False, True, id='channel-tail'),
        pytest.param(1, 129, 127, 2, False, False, False, id='time-tail'),
        pytest.param(1, 257, 256, 4, True, False, True, id='packed-qkv'),
        pytest.param(1, 129, 65, 4, True, True, False, id='packed-state'),
        pytest.param(2, 3, 65, 4, False, True, True, id='short-state'),
        pytest.param(2, 32, 65, 4, False, True, True, id='split-boundary'),
        pytest.param(2, 33, 65, 4, False, True, True, id='split-tail'),
        pytest.param(1, 1024, 1024, 4, False, False, False, id='small-tile-boundary'),
        pytest.param(1, 1025, 1024, 4, False, False, False, id='large-tile-boundary'),
        pytest.param(1, 8193, 65, 3, False, False, True, id='reduction-tail'),
        pytest.param(1, 8193, 65, 4, True, False, True, id='packed-reduction-tail'),
    ],
)
def test_conv_backend_parity(
    monkeypatch: pytest.MonkeyPatch,
    B: int,
    T: int,
    D: int,
    W: int,
    is_varlen: bool,
    has_initial_state: bool,
    non_contiguous: bool,
    activation: str | None,
    dtype: torch.dtype,
    weight_dtype: torch.dtype,
):
    pytest.importorskip('fla.modules.backends.gluon.causal_conv1d')
    torch.manual_seed(42)
    x = torch.randn(B, T, D * (3 if non_contiguous else 1), device=device, dtype=dtype)
    x = x[..., D:2 * D] if non_contiguous else x
    x.requires_grad_(True)
    weight = torch.randn(D, W, device=device, dtype=weight_dtype, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=weight_dtype, requires_grad=True)
    residual = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    cu_seqlens = torch.tensor([0, 0, 1, 3, T], device=device) if is_varlen else None
    N = 4 if is_varlen else B
    h0 = torch.randn(N, D, W, device=device, dtype=dtype, requires_grad=True) if has_initial_state else None
    dy = torch.randn(B, T, D * 2, device=device, dtype=dtype)[..., ::2]
    dht = torch.randn_like(h0) if has_initial_state else None

    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_CONV_GLUON', '0')
    ref, ref_ht = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=h0,
        output_final_state=has_initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    if has_initial_state:
        torch.autograd.backward((ref, ref_ht), (dy, dht))
    else:
        ref.backward(dy)
    ref_dx, x.grad = x.grad, None
    ref_dw, weight.grad = weight.grad, None
    ref_db, bias.grad = bias.grad, None
    ref_dr, residual.grad = residual.grad, None
    if has_initial_state:
        ref_dh0, h0.grad = h0.grad, None

    monkeypatch.setenv('FLA_GLUON', '1')
    tri, tri_ht = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=h0,
        output_final_state=has_initial_state,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    if has_initial_state:
        torch.autograd.backward((tri, tri_ht), (dy, dht))
    else:
        tri.backward(dy)
    tri_dx, x.grad = x.grad, None
    tri_dw, weight.grad = weight.grad, None
    tri_db, bias.grad = bias.grad, None
    tri_dr, residual.grad = residual.grad, None
    if has_initial_state:
        tri_dh0, h0.grad = h0.grad, None

    assert_close('y', ref, tri, 1e-3)
    assert_close('dx', ref_dx, tri_dx, 1e-3)
    assert_close('dw', ref_dw, tri_dw, 1e-3)
    assert_close('db', ref_db, tri_db, 1e-3)
    assert_close('dr', ref_dr, tri_dr, 1e-3)
    if has_initial_state:
        assert_close('ht', ref_ht, tri_ht, 1e-3)
        assert_close('dh0', ref_dh0, tri_dh0, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype'),
    [
        pytest.param(*test, id="B{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(*test))
        for test in [
            (2, 64, 128, 3, "swish", True, True, torch.float32),
            (2, 128, 128, 4, "swish", False, True, torch.float32),
            (2, 64, 128, 3, "swish", True, False, torch.float32),
            (2, 128, 128, 4, "swish", False, False, torch.float32),
            (2, 500, 1024, 3, None, True, True, torch.float32),
            (2, 1024, 1024, 4, None, False, True, torch.float32),
            (2, 64, 128, 3, None, True, False, torch.float16),
            (2, 128, 128, 4, None, False, False, torch.float16),
        ]
    ],
)
@torch.no_grad
def test_conv_update(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype)
    weight = torch.randn(D, W).to(device, dtype)
    bias = torch.randn(D).to(device, dtype) if has_bias else None
    residual = x.clone() if has_residual else None

    ref = causal_conv1d_ref(
        x=rearrange(x, "b t d -> b d t"),
        weight=weight,
        bias=bias,
        activation=activation,
    )
    ref = rearrange(ref, "b d t -> b t d")
    if has_residual:
        ref += residual
    ref_cache = x.new_zeros(B, D, W)
    ref_cache[:, :, -min(W, T):].copy_(rearrange(x[..., -min(W, T):, :], 'n w d -> n d w'))

    tri = torch.zeros_like(x)
    tri_cache = x.new_zeros(B, D, W)
    for i in range(T):
        y, tri_cache = causal_conv1d_update(
            x=x[:, i:i+1, :],
            cache=tri_cache,
            residual=residual[:, i:i+1, :] if has_residual else None,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        tri[:, i:i+1, :] = y

    assert_close("    y", ref, tri, 1e-3)
    assert_close("cache", ref_cache, tri_cache, 1e-3)


@pytest.mark.parametrize(
    ('N', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype'),
    [
        pytest.param(*test, id="N{0}_D{1}_W{2}_activation{3}_has_bias{4}_has_residual{5}_{6}".format(*test))
        for test in [
            (4, 128, 3, "swish", True, True, torch.float32),
            (4, 128, 4, "swish", False, True, torch.float32),
            (4, 128, 3, "swish", True, False, torch.float32),
            (2, 128, 3, None, True, True, torch.float16),
        ]
    ],
)
@torch.no_grad
def test_conv_update_varlen(
    N: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    T = 64
    min_len_each = max(1, T // N)
    lengths = [min_len_each] * N
    lengths[-1] += T % N
    xs = [torch.randn(1, length, D).to(device, dtype) for length in lengths]
    weight = torch.randn(D, W).to(device, dtype)
    bias = torch.randn(D).to(device, dtype) if has_bias else None

    refs, tris, ref_caches, tri_caches = [], [], [], []
    for x in xs:
        length = x.shape[1]
        residual = x.clone() if has_residual else None
        cache = x.new_zeros(1, D, W)
        cache[:, :, -min(W, length):].copy_(rearrange(x[:, -min(W, length):, :], 'b w d -> b d w'))
        ref_cache, tri_cache = cache.clone(), cache.clone()
        ref, tri = torch.zeros_like(x), torch.zeros_like(x)
        for t in range(length):
            ref_y = causal_conv1d_update_ref(
                x=x[:, t, :],
                cache=ref_cache,
                weight=weight,
                bias=bias,
                activation=activation,
            ).unsqueeze(1)
            if has_residual:
                ref_y += residual[:, t:t+1, :]
            ref[:, t:t+1, :] = ref_y
            tri_y, tri_cache = causal_conv1d_update(
                x=x[:, t:t+1, :],
                cache=tri_cache,
                residual=residual[:, t:t+1, :] if has_residual else None,
                weight=weight,
                bias=bias,
                activation=activation,
            )
            tri[:, t:t+1, :] = tri_y
        refs.append(ref)
        tris.append(tri)
        ref_caches.append(ref_cache)
        tri_caches.append(tri_cache)

    assert_close("varlen decode y", torch.cat(refs, dim=1), torch.cat(tris, dim=1), 1e-3)
    assert_close("varlen decode cache", torch.cat(ref_caches, dim=0), torch.cat(tri_caches, dim=0), 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype'),
    [
        pytest.param(*test, id="B{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(*test))
        for test in [
            (2, 64, 128, 3, "swish", True, True, torch.float32),
            (2, 128, 128, 4, "swish", False, True, torch.float32),
            (2, 64, 128, 3, "swish", True, False, torch.float32),
            (2, 128, 128, 4, None, False, False, torch.float16),
        ]
    ],
)
@torch.no_grad
def test_conv_update_non_contiguous(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    x_full = torch.randn(B, T * 2, D, device=device, dtype=dtype)
    x = x_full[:, ::2, :]
    residual = x_full[:, ::2, :] if has_residual else None
    assert not x.is_contiguous()
    if has_residual:
        assert not residual.is_contiguous()
    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype) if has_bias else None

    cache = x.new_zeros(B, D, W)
    cache[:, :, -min(W, T):].copy_(rearrange(x[..., -min(W, T):, :], 'b w d -> b d w'))
    ref_x = x.contiguous()
    ref_residual = residual.contiguous() if has_residual else None
    ref_cache, tri_cache = cache.clone(), cache.clone()
    ref, tri = torch.zeros_like(x), torch.zeros_like(x)
    for i in range(T):
        ref_y, ref_cache = causal_conv1d_update(
            x=ref_x[:, i:i+1, :],
            cache=ref_cache,
            residual=ref_residual[:, i:i+1, :] if has_residual else None,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        ref[:, i:i+1, :] = ref_y
        tri_y, tri_cache = causal_conv1d_update(
            x=x[:, i:i+1, :],
            cache=tri_cache,
            residual=residual[:, i:i+1, :] if has_residual else None,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        tri[:, i:i+1, :] = tri_y

    assert_close("decode y with non-contiguous x", ref, tri, 1e-3)
    assert_close("decode cache with non-contiguous x", ref_cache, tri_cache, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype', 'backend'),
    [
        pytest.param(
            *test, id="B{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}_{8}".format(*test))
        for test in [
            (2, 64, 128, 3, "swish", True, True, torch.float32, 'triton'),
            (2, 128, 128, 4, "swish", False, True, torch.float32, 'triton'),
            (2, 64, 128, 3, "swish", True, False, torch.float32, 'triton'),
            (2, 128, 128, 4, "swish", False, False, torch.float32, 'triton'),
            (2, 500, 1024, 3, None, True, True, torch.float32, 'triton'),
            (2, 1024, 1024, 4, None, False, True, torch.float32, 'triton'),
            (2, 64, 128, 3, None, True, False, torch.float16, 'triton'),
            (2, 128, 128, 4, None, False, False, torch.float16, 'triton'),
            (2, 64, 128, 3, "swish", True, True, torch.float32, 'cuda'),
            (2, 128, 128, 4, "swish", False, True, torch.float32, 'cuda'),
            (2, 64, 128, 3, "swish", True, False, torch.float32, 'cuda'),
            (2, 128, 128, 4, "swish", False, False, torch.float32, 'cuda'),
            (2, 2, 128, 4, "swish", True, True, torch.float32, 'cuda'),
            (2, 2, 128, 4, "swish", True, True, torch.float32, 'triton'),
            (2, 3, 128, 4, "swish", True, True, torch.float32, 'triton'),
            (2, 4, 128, 4, "swish", True, True, torch.float32, 'triton'),
            (2, 2, 128, 3, "swish", True, True, torch.float32, 'triton'),
        ]
    ],
)
@torch.no_grad
def test_conv_prefill(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if backend == 'cuda':
        if causal_conv1d_fn is None:
            pytest.skip("causal_conv1d is not installed for CUDA backend")
        if not IS_NVIDIA:
            pytest.skip("CUDA backend requires an NVIDIA GPU")
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype)
    residual = torch.randn(B, T, D).to(device, dtype) if has_residual else None

    conv = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=has_bias,
        activation=activation,
        backend=backend,
        device=device,
        dtype=dtype,
    )

    cache = torch.randn(B, D, W - 1).to(device, dtype)

    ref = causal_conv1d_ref(
        x=x.transpose(1, 2),
        weight=rearrange(conv.weight, "d 1 w -> d w"),
        bias=conv.bias,
        initial_state=cache,
        activation=activation,
    ).transpose(1, 2)
    if has_residual:
        ref += residual

    zero_padding = torch.zeros(B, D, 1).to(device, dtype)
    tri_cache = torch.cat([zero_padding, cache], dim=-1)
    tri, cache_out = conv(x=x, residual=residual, cache=tri_cache.clone(), output_final_state=True)

    assert_close("y", ref, tri, 1e-3)
    for p in range(1, W):
        if p <= T:
            expected = x[:, -p, :]
        else:
            expected = tri_cache[:, :, -(p - T)]
        torch.testing.assert_close(
            cache_out[:, :, -p],
            expected,
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.parametrize(
    ('N', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype', 'backend'),
    [
        pytest.param(
            *test,
            id="N{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}_{8}".format(*test),
        )
        for test in [
            (3, 128, 64, 4, "swish", True, True, torch.float32, 'triton'),
            (4, 256, 128, 3, None,  False, True, torch.float32, 'triton'),
            (2,  64, 128, 4, "swish", True, False, torch.float16, 'cuda'),
            (3, 200,  64, 3, None,  False, False, torch.float16, 'cuda'),
            (2,   3,  64, 4, "swish", True, True, torch.float32, 'triton'),
            (2,   3,  64, 3, None,  False, True, torch.float32, 'cuda'),
        ]
    ],
)
@torch.no_grad
def test_conv_varlen_prefill(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if backend == 'cuda':
        if causal_conv1d_fn is None:
            pytest.skip("causal_conv1d is not installed for CUDA backend")
        if not IS_NVIDIA:
            pytest.skip("CUDA backend requires an NVIDIA GPU")
    torch.manual_seed(42)

    min_len_each = max(1, T // N)
    lengths = [min_len_each] * N
    lengths[-1] += T % N
    cu_seqlens = torch.tensor([0] + torch.cumsum(torch.tensor(lengths), 0).tolist(), device=device, dtype=torch.int32)

    x = torch.randn(1, T, D).to(device, dtype)
    residual = torch.randn(1, T, D).to(device, dtype) if has_residual else None

    conv = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=has_bias,
        activation=activation,
        backend=backend,
        device=device,
        dtype=dtype,
    )

    cache = torch.randn(N, D, W - 1).to(device, dtype)
    refs = []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)):
        ref_seq = causal_conv1d_ref(
            x=x[:, bos:eos].transpose(1, 2),
            weight=rearrange(conv.weight, "d 1 w -> d w"),
            bias=conv.bias,
            initial_state=cache[i:i+1],
            activation=activation,
        ).transpose(1, 2)
        if has_residual:
            ref_seq += residual[:, bos:eos]
        refs.append(ref_seq)
    ref = torch.cat(refs, dim=1)

    zero_pad = torch.zeros(N, D, 1, device=device, dtype=dtype)
    tri_cache = torch.cat([zero_pad, cache], dim=-1)
    tri, cache_out = conv(
        x=x,
        residual=residual,
        cache=tri_cache.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    assert_close("varlen y", ref, tri, 1e-3)

    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)):
        length = eos - bos
        for p in range(1, W):
            if p <= length:
                expected = x[0, eos - p, :]
            else:
                expected = tri_cache[i, :, -(p - length)]
            torch.testing.assert_close(
                cache_out[i, :, -p],
                expected,
                atol=1e-3,
                rtol=1e-3,
            )


@pytest.mark.parametrize(
    ('B', 'D', 'W', 'has_bias', 'has_residual', 'activation', 'dtype', 'backend'),
    [
        pytest.param(*test, id="B{0}_D{1}_W{2}_has_bias{3}_has_residual{4}_activation{5}_{6}_{7}".format(*test))
        for test in [
            (2, 128, 3, True, True, "swish", torch.float32, 'triton'),
            (2, 128, 4, False, True, "swish", torch.float32, 'triton'),
            (2, 128, 3, True, False, "swish", torch.float32, 'triton'),
            (2, 128, 4, False, False, "swish", torch.float32, 'triton'),
            (2, 128, 3, True, True, "swish", torch.float32, 'cuda'),
            (2, 128, 4, False, True, "swish", torch.float32, 'cuda'),
            (2, 128, 3, True, False, "swish", torch.float32, 'cuda'),
            (2, 128, 4, False, False, "swish", torch.float32, 'cuda'),
            (2, 128, 4, False, False, None, torch.float32, 'cuda'),
            (2, 128, 4, False, False, None, torch.float32, 'triton'),
        ]
    ],
)
@torch.no_grad
def test_conv_step(
    B: int,
    D: int,
    W: int,
    activation: str | None,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if backend == 'cuda':
        if causal_conv1d_fn is None:
            pytest.skip("causal_conv1d is not installed for CUDA backend")
        if not IS_NVIDIA:
            pytest.skip("CUDA backend requires an NVIDIA GPU")
    torch.manual_seed(42)

    x = torch.randn(B, 1, D).to(device, dtype)
    residual = x.clone() if has_residual else None

    conv = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=has_bias,
        activation=activation,
        backend=backend,
        device=device,
        dtype=dtype,
    )

    cache = torch.randn(B, D, W).to(device, dtype)

    ref = causal_conv1d_update_ref(
        x=x.squeeze(1),
        cache=cache.clone(),
        weight=rearrange(conv.weight, "d 1 w -> d w"),
        bias=conv.bias,
        activation=activation,
    ).unsqueeze(1)
    if has_residual:
        ref += residual

    tri, _ = conv.step(x=x, residual=residual, cache=cache.clone())

    assert_close("y", ref, tri, 1e-3)


def test_conv_varlen_empty_sequence():
    """A packed batch with a zero-length sequence must not be misdetected as a decode step."""
    torch.manual_seed(42)
    D, W = 16, 4
    dtype = torch.float32
    # an empty sequence makes B*T == N insufficient to identify a decode step.
    cu_seqlens = torch.tensor([0, 0, 2], device=device, dtype=torch.int32)
    N, T = 2, 2
    x = torch.randn(1, T, D).to(device, dtype)

    conv = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=False,
        activation='silu',
        device=device,
        dtype=dtype,
    )

    cache = torch.randn(N, D, W - 1).to(device, dtype)
    # reference: only the real sequence (index 1) is processed
    ref = causal_conv1d_ref(
        x=x.transpose(1, 2),
        weight=rearrange(conv.weight, "d 1 w -> d w"),
        bias=conv.bias,
        initial_state=cache[1:2],
        activation='silu',
    ).transpose(1, 2)

    zero_pad = torch.zeros(N, D, 1, device=device, dtype=dtype)
    tri, _ = conv(
        x=x,
        cache=torch.cat([zero_pad, cache], dim=-1).clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    assert_close("y", ref, tri, 1e-3)


def test_conv_backend_override(monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(42)
    monkeypatch.setenv('FLA_CONV_BACKEND', 'bogus')
    with pytest.raises(ValueError, match='Invalid backend'):
        ShortConvolution(hidden_size=8, kernel_size=3)

    monkeypatch.setenv('FLA_CONV_BACKEND', 'cuda')
    monkeypatch.setattr('fla.modules.conv.short_conv.causal_conv1d_fn_cuda', None)
    with pytest.warns(UserWarning, match='Switching to the Triton implementation'):
        conv = ShortConvolution(hidden_size=8, kernel_size=3, backend='triton')
    assert conv.backend == 'triton'


@pytest.mark.parametrize(
    'case',
    ['rank', 'channels', 'width', 'weight', 'packed-batch', 'chunk', 'state', 'dtype', 'distributed'],
)
def test_conv_backend_verifier(monkeypatch: pytest.MonkeyPatch, case: str):
    from fla.modules.backends.gluon import GluonBackend

    backend = GluonBackend()
    x = torch.empty(2, 64, 32)
    weight = torch.empty(32, 4)
    kwargs = {}
    if case == 'dtype':
        x = x.double()
    elif case == 'distributed':
        monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    elif case == 'rank':
        x = x.unsqueeze(-1)
    elif case == 'channels':
        x = x[..., ::2]
    elif case == 'width':
        weight = torch.empty(32, 5)
    elif case == 'weight':
        weight = None
    elif case == 'packed-batch':
        kwargs['cu_seqlens'] = torch.tensor([0, 128])
    elif case == 'chunk':
        kwargs['chunk_size'] = 32
    if case == 'state':
        accepted, reason = backend.causal_conv1d_bwd_verifier(
            x=x,
            dy=x,
            dht=None,
            weight=weight,
            initial_state=torch.empty(2, 32, 4),
        )
    else:
        accepted, reason = backend.causal_conv1d_fwd_verifier(x=x, weight=weight, **kwargs)
    assert not accepted and reason


@pytest.mark.skipif(not IS_NVIDIA, reason='Gluon convolution requires NVIDIA')
@pytest.mark.parametrize(('global_enable', 'local_enable'), [('0', '0'), ('0', '1'), ('1', '0')], ids=['disabled', 'local', 'global'])
@pytest.mark.parametrize('W', [4, 5], ids=['W4', 'W5'])
def test_conv_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    conv_backend_calls: list[str] | None,
    global_enable: str,
    local_enable: str,
    W: int,
):
    from fla.ops.backends import _DISPATCH_DISABLED

    if conv_backend_calls is None:
        pytest.skip('Gluon convolution is unavailable')
    torch.manual_seed(42)
    monkeypatch.setenv('FLA_GLUON', global_enable)
    monkeypatch.setenv('FLA_CONV_GLUON', local_enable)
    x = torch.randn(1, 65, 64, device=device, requires_grad=True)
    weight = torch.randn(64, W, device=device, requires_grad=True)
    y, _ = causal_conv1d(x=x, weight=weight, activation='silu')
    y.sum().backward()
    enabled = (global_enable == '1' or local_enable == '1') and W == 4 and not _DISPATCH_DISABLED
    assert conv_backend_calls == (['fwd', 'bwd'] if enabled else [])

    monkeypatch.setenv('FLA_GLUON', '0')
    monkeypatch.setenv('FLA_CONV_GLUON', '0')
    conv_backend_calls.clear()
    y, _ = causal_conv1d(x=x, weight=weight, activation='silu')
    y.sum().backward()
    assert not conv_backend_calls


@pytest.mark.parametrize('chunk_size', [32, 64], ids=['BT32', 'BT64'])
def test_conv_deprecated_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
    conv_backend_calls: list[str] | None,
    chunk_size: int,
):
    from fla.ops.backends import _DISPATCH_DISABLED

    monkeypatch.setenv('FLA_GLUON', '1')
    monkeypatch.setenv('FLA_CONV_GLUON', '0')
    enabled = conv_backend_calls is not None and chunk_size == 64 and not _DISPATCH_DISABLED
    calls = conv_backend_calls if conv_backend_calls is not None else []
    torch.manual_seed(42)
    x = torch.randn(1, 129, 64, device=device)
    weight = torch.randn(64, 4, device=device)
    bias = torch.randn(64, device=device)
    dy = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 1, 34, 129], device=device)
    kwargs = dict(
        weight=weight,
        bias=bias,
        residual=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=prepare_chunk_indices(cu_seqlens, chunk_size),
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always', FutureWarning)
        y, _ = causal_conv1d_fwd(x=x, **kwargs, chunk_size=chunk_size)
        grads = causal_conv1d_bwd(x=x, dy=dy, dht=None, **kwargs, chunk_size=chunk_size)
    assert not any(issubclass(record.category, FutureWarning) for record in records)
    assert calls == (['fwd', 'bwd'] if enabled else [])
    calls.clear()
    with pytest.warns(FutureWarning, match='`BT` is deprecated.*Use `chunk_size` instead'):
        y_old, _ = causal_conv1d_fwd(x=x, **kwargs, BT=chunk_size)
    with pytest.warns(FutureWarning, match='`BT` is deprecated.*Use `chunk_size` instead'):
        grads_old = causal_conv1d_bwd(x=x, dy=dy, dht=None, **kwargs, BT=chunk_size)
    assert calls == (['fwd', 'bwd'] if enabled else [])
    assert_close('y', y, y_old, 1e-3)
    for name, expected, actual in zip(('dx', 'dw', 'db'), grads, grads_old):
        assert_close(name, expected, actual, 1e-3)
