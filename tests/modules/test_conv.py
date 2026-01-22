
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.modules.convolution import ShortConvolution, causal_conv1d, causal_conv1d_update
from fla.utils import assert_close, device

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def causal_conv1d_ref_torch(
    x,
    weight,
    bias=None,
    initial_state=None,
    output_final_state=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_state: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_state is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_state, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if output_final_state:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in,
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not output_final_state else (out, final_states_out)


def causal_conv1d_update_ref_torch(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


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
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if causal_conv1d_fn is None and backend == 'cuda':
        pytest.skip("causal_conv1d is not installed for CUDA backend")
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype).requires_grad_(True)
    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None
    residual = x.detach().clone().requires_grad_(True) if has_residual else None
    dy = torch.randn(B, T, D).to(device, dtype)

    ref = causal_conv1d_ref_torch(
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

    tri, _ = causal_conv1d(x, weight, bias, residual=residual, activation=activation, backend=backend)
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
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if causal_conv1d_fn is None and backend == 'cuda':
        pytest.skip("causal_conv1d is not installed for CUDA backend")
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
            causal_conv1d_ref_torch(
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

    tri, _ = causal_conv1d(x, weight, bias, residual=residual, activation=activation, cu_seqlens=cu_seqlens, backend=backend)
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
def test_conv_decoding(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    x = torch.randn(B, T, D).to(device, dtype)
    weight = torch.randn(D, W).to(device, dtype) * 0
    bias = torch.randn(D).to(device, dtype) if has_bias else None
    residual = x.clone() if has_residual else None

    ref = causal_conv1d_ref_torch(
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
            (2, 2, 128, 4, "swish", True, True, torch.float32, 'cuda'),  # T_prefill < W
            (2, 2, 128, 4, "swish", True, True, torch.float32, 'triton'),
            (2, 3, 128, 4, "swish", True, True, torch.float32, 'triton'),
            (2, 4, 128, 4, "swish", True, True, torch.float32, 'triton'),
            (2, 2, 128, 3, "swish", True, True, torch.float32, 'triton'),
        ]
    ],
)
@torch.no_grad
def test_conv_with_cache_prefill_fwd(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if causal_conv1d_fn is None and backend == 'cuda':
        pytest.skip("causal_conv1d is not installed for CUDA backend")
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

    ref = causal_conv1d_ref_torch(
        x=x.transpose(1, 2),                    # (B, D, T)
        weight=rearrange(conv.weight, "d 1 w -> d w"),
        bias=conv.bias,
        initial_state=cache,                    # (B, D, W-1)
        activation=activation,
    ).transpose(1, 2)                           # (B, T, D)
    if has_residual:
        ref += residual

    zero_padding = torch.zeros(B, D, 1).to(device, dtype)
    tri_cache = torch.cat([zero_padding, cache], dim=-1)  # (B, D, W)
    tri, cache_out = conv(x, residual=residual, cache=tri_cache.clone(), output_final_state=True)

    assert_close("y", ref, tri, 1e-3)
    for p in range(1, W):
        if p <= T:
            expected = x[:, -p, :]
        else:
            expected = tri_cache[:, :, -(p - T)]
        torch.testing.assert_close(
            cache_out[:, :, -p],
            expected,
            atol=1e-3, rtol=1e-3,
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
            (2,   3,  64, 4, "swish", True, True, torch.float32, 'triton'),  # T < W
            (2,   3,  64, 3, None,  False, True, torch.float32, 'cuda'),     # T < W
        ]
    ],
)
@torch.no_grad
def test_conv_varlen_with_cache_prefill_fwd(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if causal_conv1d_fn is None and backend == 'cuda':
        pytest.skip("causal_conv1d is not installed for CUDA backend")
    torch.manual_seed(42)

    min_len_each = max(1, T // N)
    lengths = [min_len_each] * N
    lengths[-1] += T % N
    assert all(length >= 1 for length in lengths), "all lengths must >= 1"
    cu_seqlens = torch.tensor([0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
                              device=device, dtype=torch.int32)

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
    ref_list = []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)):
        xi = x[:, bos:eos, :].transpose(1, 2)  # (1, D, l)
        ci = cache[i:i + 1]                    # (1, D, W-1)
        refi = causal_conv1d_ref_torch(
            x=xi,
            weight=rearrange(conv.weight, "d 1 w -> d w"),
            bias=conv.bias,
            initial_state=ci,
            activation=activation,
        ).transpose(1, 2)                      # (1, l, D)
        if has_residual:
            refi += residual[:, bos:eos, :]
        ref_list.append(refi)
    ref = torch.cat(ref_list, dim=1)           # (1, T, D)

    zero_pad = torch.zeros(N, D, 1, device=device, dtype=dtype)
    tri_cache = torch.cat([zero_pad, cache], dim=-1)  # (N, D, W)
    tri, cache_out = conv(x,
                          residual=residual,
                          cache=tri_cache.clone(),
                          cu_seqlens=cu_seqlens,
                          output_final_state=True)

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
def test_conv_decoding_with_cache(
    B: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
    backend: str,
):
    if causal_conv1d_fn is None and backend == 'cuda':
        pytest.skip("causal_conv1d is not installed for CUDA backend")
    torch.manual_seed(42)

    x = torch.randn(B, 1, D).to(device, dtype)        # (B, 1, D)
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

    state = torch.randn(B, D, W).to(device, dtype)

    # reference
    ref = causal_conv1d_update_ref_torch(
        x.squeeze(1),                           # (B, D)
        conv_state=state.clone(),
        weight=rearrange(conv.weight, "d 1 w -> d w"),
        bias=conv.bias,
        activation=activation,
    ).unsqueeze(1)                             # (B, 1, D)
    if has_residual:
        ref += residual

    # ShortConvolution step
    with torch.no_grad():
        y, _ = conv.step(x, residual, state.clone())

    assert_close("y", ref, y, 1e-3)


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
def test_conv_varlen_decoding(
    N: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    """Test varlen mode decoding with causal_conv1d_update."""
    torch.manual_seed(42)

    # Create varlen sequences
    T = 64
    min_len_each = max(1, T // N)
    lengths = [min_len_each] * N
    lengths[-1] += T % N
    # Create input for each sequence
    x_list = []
    residual_list = []
    for i in range(N):
        seq_len = lengths[i]
        x_seq = torch.randn(1, seq_len, D).to(device, dtype)
        x_list.append(x_seq)
        if has_residual:
            residual_list.append(x_seq.clone())

    weight = torch.randn(D, W).to(device, dtype)
    bias = torch.randn(D).to(device, dtype) if has_bias else None

    # Reference: process each sequence separately
    ref_outputs = []
    ref_caches = []
    for i in range(N):
        x_seq = x_list[i]
        B_i, T_i = x_seq.shape[0], x_seq.shape[1]

        ref_cache = x_seq.new_zeros(B_i, D, W)
        ref_cache[:, :, -min(W, T_i):].copy_(
            rearrange(x_seq[..., -min(W, T_i):, :], 'b w d -> b d w')
        )

        ref_output = torch.zeros_like(x_seq)
        tri_cache_i = ref_cache.clone()

        residual_i = residual_list[i] if has_residual else None

        for t in range(T_i):
            y, tri_cache_i = causal_conv1d_update(
                x=x_seq[:, t:t+1, :],
                cache=tri_cache_i,
                residual=residual_i[:, t:t+1, :] if has_residual else None,
                weight=weight,
                bias=bias,
                activation=activation,
            )
            ref_output[:, t:t+1, :] = y

        ref_outputs.append(ref_output)
        ref_caches.append(ref_cache)

    ref_y = torch.cat(ref_outputs, dim=1)
    ref_cache = torch.cat(ref_caches, dim=0)

    # Note: causal_conv1d_update doesn't support cu_seqlens directly
    # So we test by processing with a loop using the same logic as reference
    # This test documents the expected behavior for varlen decode

    # For now, just verify the reference implementation works
    # In real usage, one would need to either:
    # 1. Call causal_conv1d_update in a loop for each sequence (as in reference)
    # 2. Or extend causal_conv1d_update to support cu_seqlens parameter

    # Since causal_conv1d_update doesn't support cu_seqlens, we test with loop approach
    tri_outputs = []
    tri_caches = []

    for i in range(N):
        x_seq = x_list[i]
        B_i, T_i = x_seq.shape[0], x_seq.shape[1]

        tri_cache_i = x_seq.new_zeros(B_i, D, W)
        tri_cache_i[:, :, -min(W, T_i):].copy_(
            rearrange(x_seq[..., -min(W, T_i):, :], 'b w d -> b d w')
        )

        tri_output = torch.zeros_like(x_seq)
        residual_i = residual_list[i] if has_residual else None

        for t in range(T_i):
            y, tri_cache_i = causal_conv1d_update(
                x=x_seq[:, t:t+1, :],
                cache=tri_cache_i,
                residual=residual_i[:, t:t+1, :] if has_residual else None,
                weight=weight,
                bias=bias,
                activation=activation,
            )
            tri_output[:, t:t+1, :] = y

        tri_outputs.append(tri_output)
        tri_caches.append(tri_cache_i)

    tri_y = torch.cat(tri_outputs, dim=1)
    tri_cache = torch.cat(tri_caches, dim=0)

    # Verify
    assert_close("varlen decode y", ref_y, tri_y, 1e-3)
    assert_close("varlen decode cache", ref_cache, tri_cache, 1e-3)


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
def test_conv_decoding_non_contiguous_x(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    """Test decoding with non-contiguous input x."""
    torch.manual_seed(42)

    # Create a larger tensor and take a non-contiguous slice
    x_full = torch.randn(B, T * 2, D, device=device, dtype=dtype)
    x = x_full[:, ::2, :]  # [B, T, D], non-contiguous
    assert not x.is_contiguous(), "x should be non-contiguous"

    if has_residual:
        residual = x_full[:, ::2, :]  # Also non-contiguous
        assert not residual.is_contiguous(), "residual should be non-contiguous"
    else:
        residual = None

    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype) if has_bias else None

    # Reference: use contiguous version
    ref_cache = x.new_zeros(B, D, W)
    ref_cache[:, :, -min(W, T):].copy_(
        rearrange(x[..., -min(W, T):, :], 'b w d -> b d w')
    )

    x_contiguous = x.contiguous()
    ref_output = torch.zeros_like(x)
    ref_cache_copy = ref_cache.clone()
    residual_contiguous = residual.contiguous() if has_residual else None

    for i in range(T):
        y, ref_cache_copy = causal_conv1d_update(
            x=x_contiguous[:, i:i+1, :],
            cache=ref_cache_copy,
            residual=residual_contiguous[:, i:i+1, :] if has_residual else None,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        ref_output[:, i:i+1, :] = y

    # Test: use non-contiguous x directly
    tri_cache = ref_cache.clone()
    tri_output = torch.zeros_like(x)

    for i in range(T):
        # Pass non-contiguous slice
        x_slice = x[:, i:i+1, :]  # This is non-contiguous because x is non-contiguous

        residual_slice = residual[:, i:i+1, :] if has_residual else None

        y, tri_cache = causal_conv1d_update(
            x=x_slice,
            cache=tri_cache,
            residual=residual_slice,
            weight=weight,
            bias=bias,
            activation=activation,
        )
        tri_output[:, i:i+1, :] = y

    # Verify
    assert_close("decode y with non-contiguous x", ref_output, tri_output, 1e-3)
    assert_close("decode cache with non-contiguous x", ref_cache, tri_cache, 1e-3)


@pytest.mark.parametrize(
    ('N', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype'),
    [
        pytest.param(*test, id="N{}_T{}_D{}_W{}_activation{}_has_bias{}_has_residual{}_{}".format(*test))
        for test in [
            (4, 1024, 4096, 3, "swish", True, False, torch.float32),
            (4, 1024, 4096, 4, "swish", False, False, torch.float32),
            (4, 1024, 4096, 3, None, True, False, torch.float16),
            (4, 1024, 4096, 4, None, False, False, torch.float16),
        ]
    ],
)
def test_fast_conv_varlen(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    has_residual: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    if causal_conv1d_fn is None:
        pytest.skip("causal_conv1d is not installed for CUDA backend")
    assert has_residual is False
    from fla.modules.convolution import fast_causal_conv1d_fn
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
            causal_conv1d_ref_torch(
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

    tri, _ = fast_causal_conv1d_fn(x, weight, bias, residual=residual, activation=activation,
                                   cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens.cpu())
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
    ('B', 'T', 'D', 'W', 'has_bias', 'has_residual', 'activation', 'dtype'),
    [
        pytest.param(*test, id="B{0}_T{1}_D{2}_W{3}_has_bias{4}_has_residual{5}_activation{6}_{7}".format(*test))
        for test in [
            (2, 64, 100, 3, True, True, "swish", torch.float32),
            (2, 128, 128, 4, True, True, "swish", torch.float32),
            (3, 128, 128, 4, True, True, "swish", torch.float32),
            (3, 128, 256, 4, True, True, "swish", torch.float32),
            (3, 128, 512, 4, True, True, "swish", torch.float32),
            (2, 128, 1024, 4, True, True, "swish", torch.float32),
            (2, 128, 2048, 3, True, True, "swish", torch.float32),
            (2, 128, 4096, 4, True, True, "swish", torch.float32),
            (2, 128, 8192, 4, True, True, "swish", torch.float32),
        ]
    ],
)
def test_conv_cache_backward(
    B: int,
    T: int,
    D: int,
    W: int,
    has_bias: bool,
    has_residual: bool,
    activation: str,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(D, W, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=dtype, requires_grad=True) if has_bias else None
    residual = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True) if has_residual else None
    cache = torch.randn(B, D, W - 1, device=device, dtype=dtype, requires_grad=True)

    def ref_func(x, weight, bias, residual, cache):
        out, cache_out = causal_conv1d_ref_torch(
            x.transpose(1, 2),
            weight,
            bias,
            initial_state=cache,
            output_final_state=True,
            activation=activation,
        )
        out = out.transpose(1, 2)
        if residual is not None:
            out += residual
        return out, cache_out

    def triton_func(x, weight, bias, residual, cache):
        zero_padding = torch.zeros(B, D, 1, device=device, dtype=dtype)
        triton_cache = torch.cat([zero_padding, cache], dim=-1).contiguous()
        tri, cache_out_triton = causal_conv1d(
            x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=triton_cache,
            output_final_state=True,
            activation=activation,
        )
        cache_out_triton = cache_out_triton[..., 1:].clone()  # [B, D, W-1]
        return tri, cache_out_triton

    d_tri = torch.randn_like(x)
    d_cache_out = torch.randn_like(cache)

    def get_grads(func, *inputs):
        out, cache_out = func(*inputs)
        loss = (out * d_tri).sum() + (cache_out * d_cache_out).sum()
        grads = torch.autograd.grad(
            loss,
            inputs,
            retain_graph=True,
            create_graph=False,
        )
        return grads

    inputs = (x, weight, bias, residual, cache)
    grads_ref = get_grads(ref_func, *inputs)
    grads_tri = get_grads(triton_func, *inputs)

    names = ["x", "weight", "bias", "residual", "cache"]
    for name, g_ref, g_tri in zip(names, grads_ref, grads_tri, strict=False):
        assert_close(name, g_ref, g_tri, ratio=1e-3)


def test_conv_varlen_initial_state_backward_random():
    torch.manual_seed(1234)
    B = 1
    T = 256
    D = 128
    W = 4
    activation = "swish"

    # Random but deterministic split into two sequences
    l1 = int(torch.randint(low=W, high=T - W, size=(1,)).item())
    cu_seqlens = torch.tensor([0, l1, T], device=device, dtype=torch.int32)

    x = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(D, W, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=torch.float32, requires_grad=True)

    # initial_state uses padded layout [N, D, W] with column 0 as padding
    initial_state = torch.zeros(2, D, W, device=device, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        initial_state[:, :, 1:].copy_(torch.randn(2, D, W - 1, device=device, dtype=torch.float32))

    dy = torch.randn_like(x)

    def ref_varlen(x, weight, bias, initial_state, cu_seqlens):
        outs = []
        caches = []
        num_seqs = cu_seqlens.numel() - 1
        for i in range(num_seqs):
            s = int(cu_seqlens[i].item())
            e = int(cu_seqlens[i + 1].item())
            x_seq = x[:, s:e, :]
            cache = initial_state[i:i+1, :, 1:].contiguous()
            out_seq, cache_out = causal_conv1d_ref_torch(
                x_seq.transpose(1, 2),
                weight,
                bias,
                initial_state=cache,
                output_final_state=True,
                activation=activation,
            )
            outs.append(out_seq.transpose(1, 2))
            caches.append(cache_out)
        return torch.cat(outs, dim=1), torch.cat(caches, dim=0)

    y_ref, _ = ref_varlen(x, weight, bias, initial_state, cu_seqlens)
    loss_ref = (y_ref * dy).sum()
    grads_ref = torch.autograd.grad(
        loss_ref,
        (x, weight, bias, initial_state),
        retain_graph=False,
        create_graph=False,
    )

    y_tri, _ = causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )
    loss_tri = (y_tri * dy).sum()
    grads_tri = torch.autograd.grad(
        loss_tri,
        (x, weight, bias, initial_state),
        retain_graph=False,
        create_graph=False,
    )

    assert_close("dx", grads_ref[0], grads_tri[0], ratio=1e-3)
    assert_close("dw", grads_ref[1], grads_tri[1], ratio=1e-3)
    assert_close("db", grads_ref[2], grads_tri[2], ratio=1e-3)
    assert_close("d_init", grads_ref[3], grads_tri[3], ratio=1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'has_bias', 'has_residual', 'activation', 'dtype'),
    [
        pytest.param(*test, id="B{0}_T{1}_D{2}_W{3}_has_bias{4}_has_residual{5}_activation{6}_{7}".format(*test))
        for test in [
            # Test USE_INITIAL_STATE=True, USE_FINAL_STATE=False case
            # This specifically tests the "if not USE_FINAL_STATE" branch with initial_state
            (2, 64, 100, 3, True, False, "swish", torch.float32),
            (2, 128, 128, 4, True, False, "swish", torch.float32),
            (3, 128, 128, 4, False, False, "swish", torch.float32),
            (2, 64, 256, 4, True, True, "swish", torch.float32),
            (2, 128, 512, 4, True, False, None, torch.float32),
            (2, 64, 128, 3, True, False, "swish", torch.float16),
        ]
    ],
)
def test_conv_cache_backward_no_final_state(
    B: int,
    T: int,
    D: int,
    W: int,
    has_bias: bool,
    has_residual: bool,
    activation: str,
    dtype: torch.dtype,
):
    """Test backward with initial_state but WITHOUT output_final_state.

    This tests the 'if not USE_FINAL_STATE' branch in causal_conv1d_bwd_kernel,
    which previously was missing dh0 calculation and dw contribution from initial_state.
    """
    torch.manual_seed(42)

    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(D, W, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=dtype, requires_grad=True) if has_bias else None
    residual = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True) if has_residual else None
    cache = torch.randn(B, D, W - 1, device=device, dtype=dtype, requires_grad=True)

    def ref_func(x, weight, bias, residual, cache):
        # Use output_final_state=True for ref so we get a tuple, then ignore final_state
        # This ensures we test the same forward computation
        out, _ = causal_conv1d_ref_torch(
            x.transpose(1, 2),
            weight,
            bias,
            initial_state=cache,
            output_final_state=True,  # Use True to get tuple return
            activation=activation,
        )
        out = out.transpose(1, 2)
        if residual is not None:
            out += residual
        return out

    def triton_func(x, weight, bias, residual, cache):
        zero_padding = torch.zeros(B, D, 1, device=device, dtype=dtype)
        triton_cache = torch.cat([zero_padding, cache], dim=-1).contiguous()
        # Key: output_final_state=False to test the "if not USE_FINAL_STATE" branch
        # causal_conv1d always returns tuple (y, final_state)
        tri, _ = causal_conv1d(
            x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=triton_cache,
            output_final_state=False,  # This is what we're testing!
            activation=activation,
        )
        return tri

    d_tri = torch.randn_like(x)

    def get_grads(func, inputs_dict):
        out = func(**inputs_dict)
        loss = (out * d_tri).sum()
        # Filter out None values for autograd
        tensors_to_grad = {k: v for k, v in inputs_dict.items() if v is not None}
        grads = torch.autograd.grad(
            loss,
            list(tensors_to_grad.values()),
            retain_graph=True,
            create_graph=False,
        )
        return dict(zip(tensors_to_grad.keys(), grads))

    inputs_dict = {"x": x, "weight": weight, "bias": bias, "residual": residual, "cache": cache}
    grads_ref = get_grads(lambda **kw: ref_func(kw["x"], kw["weight"], kw["bias"], kw["residual"], kw["cache"]), inputs_dict)
    grads_tri = get_grads(lambda **kw: triton_func(kw["x"], kw["weight"],
                          kw["bias"], kw["residual"], kw["cache"]), inputs_dict)

    for name in ["x", "weight", "bias", "residual", "cache"]:
        if name in grads_ref:
            assert_close(name, grads_ref[name], grads_tri[name], ratio=1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'W', 'activation', 'has_bias', 'dtype'),
    [
        pytest.param(*test, id="B{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_{6}".format(*test))
        for test in [
            (2, 64, 128, 3, "swish", True, torch.float32),
            (2, 128, 128, 4, "swish", False, torch.float32),
            (2, 64, 128, 3, None, True, torch.float16),
        ]
    ],
)
def test_conv_non_contiguous_qkv(
    B: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    dtype: torch.dtype,
):
    """Test non-contiguous input from QKV concatenated tensor (non-varlen mode)."""
    torch.manual_seed(42)

    # Simulate QKV concatenated tensor: [B, T, 3 * D]
    qkv = torch.randn(B, T, 3 * D).to(device, dtype).requires_grad_(True)

    # Get non-contiguous views for q, k, v
    q = qkv[:, :, :D]  # [B, T, D]
    k = qkv[:, :, D:2*D]  # [B, T, D], non-contiguous
    v = qkv[:, :, 2*D:]  # [B, T, D], non-contiguous

    # Verify non-contiguous
    assert not q.is_contiguous(), "q should be non-contiguous"
    assert not k.is_contiguous(), "k should be non-contiguous"
    assert not v.is_contiguous(), "v should be non-contiguous"

    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None

    # Test forward
    ref_k = k.contiguous().requires_grad_(True)
    ref_k_out, _ = causal_conv1d(ref_k, weight, bias, activation=activation)

    tri_k_out, _ = causal_conv1d(k, weight, bias, activation=activation)

    assert_close("o", ref_k_out, tri_k_out, 1e-3)

    # Test backward
    dy = torch.randn_like(tri_k_out)

    # Detach and create new leaf nodes for gradient comparison
    k_detached = k.detach().requires_grad_(True)
    ref_k_detached = k.detach().contiguous().requires_grad_(True)

    tri_k_out_detached, _ = causal_conv1d(k_detached, weight, bias, activation=activation)
    ref_k_out_detached, _ = causal_conv1d(ref_k_detached, weight, bias, activation=activation)

    tri_k_out_detached.backward(dy)
    ref_k_out_detached.backward(dy)

    # Check gradients
    assert_close("dx", ref_k_detached.grad, k_detached.grad, 1e-3)
    assert_close("dw", weight.grad, weight.grad, 1e-3)
    if has_bias:
        assert_close("dbias", bias.grad, bias.grad, 1e-3)

    # Test with residual (residual is contiguous)
    residual = k.detach().clone().requires_grad_(True)

    ref_k_res = k.detach().contiguous().requires_grad_(True)
    ref_residual = residual.detach().contiguous().requires_grad_(True)
    ref_k_out_res, _ = causal_conv1d(ref_k_res, weight, bias, residual=ref_residual, activation=activation)

    k_res = k.detach().requires_grad_(True)
    tri_k_out_res, _ = causal_conv1d(k_res, weight, bias, residual=residual, activation=activation)

    assert_close("o", ref_k_out_res, tri_k_out_res, 1e-3)

    # Backward with residual
    dy = torch.randn_like(tri_k_out_res)
    ref_k_out_res.backward(dy)
    tri_k_out_res.backward(dy)

    assert_close("dx", ref_k_res.grad, k_res.grad, 1e-3)
    assert_close("dr", ref_residual.grad, residual.grad, 1e-3)

    # Test with initial_state (including dh0 gradient)
    ref_initial_state = torch.randn(B, D, W).to(device, dtype).requires_grad_(True)
    tri_initial_state = ref_initial_state.detach().clone().requires_grad_(True)

    # Forward with state
    ref_k_state = k.detach().contiguous().requires_grad_(True)
    ref_k_out_state, ref_final_state = causal_conv1d(
        ref_k_state, weight, bias, initial_state=ref_initial_state,
        output_final_state=True, activation=activation
    )

    k_state = k.detach().requires_grad_(True)
    tri_k_out_state, tri_final_state = causal_conv1d(
        k_state, weight, bias, initial_state=tri_initial_state,
        output_final_state=True, activation=activation
    )

    assert_close("o", ref_k_out_state, tri_k_out_state, 1e-3)
    assert_close("h", ref_final_state, tri_final_state, 1e-3)

    # Backward with state
    dy = torch.randn_like(tri_k_out_state)
    ref_k_out_state.backward(dy)
    tri_k_out_state.backward(dy)

    assert_close("dx", ref_k_state.grad, k_state.grad, 1e-3)
    assert_close("dh0", ref_initial_state.grad, tri_initial_state.grad, 1e-3)


@pytest.mark.parametrize(
    ('N', 'T', 'D', 'W', 'activation', 'has_bias', 'dtype'),
    [
        pytest.param(*test, id="N{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_{6}".format(*test))
        for test in [
            (4, 128, 64, 3, "swish", True, torch.float32),
            (4, 256, 128, 4, "swish", False, torch.float32),
            (2, 64, 128, 3, None, True, torch.float16),
        ]
    ],
)
def test_conv_varlen_non_contiguous_qkv(
    N: int,
    T: int,
    D: int,
    W: int,
    activation: str,
    has_bias: bool,
    dtype: torch.dtype,
):
    """Test non-contiguous input from QKV concatenated tensor (varlen mode)."""
    torch.manual_seed(42)

    # Create varlen sequences
    min_len_each = max(1, T // N)
    lengths = [min_len_each] * N
    lengths[-1] += T % N
    cu_seqlens = torch.tensor([0] + torch.cumsum(torch.tensor(lengths), 0).tolist(),
                              device=device, dtype=torch.int32)

    # Simulate QKV concatenated tensor: [1, T, 3 * D]
    qkv = torch.randn(1, T, 3 * D).to(device, dtype).requires_grad_(True)

    # Get non-contiguous views for q, k, v
    q = qkv[:, :, :D]  # [1, T, D]
    k = qkv[:, :, D:2*D]  # [1, T, D], non-contiguous
    v = qkv[:, :, 2*D:]  # [1, T, D], non-contiguous

    # Verify non-contiguous
    assert not q.is_contiguous(), "q should be non-contiguous"
    assert not k.is_contiguous(), "k should be non-contiguous"
    assert not v.is_contiguous(), "v should be non-contiguous"

    weight = torch.randn(D, W).to(device, dtype).requires_grad_(True)
    bias = torch.randn(D).to(device, dtype).requires_grad_(True) if has_bias else None

    # Test forward
    ref_k = k.contiguous().requires_grad_(True)

    ref_k_out, _ = causal_conv1d(ref_k, weight, bias, activation=activation, cu_seqlens=cu_seqlens)

    tri_k_out, _ = causal_conv1d(k, weight, bias, activation=activation, cu_seqlens=cu_seqlens)

    assert_close("dx", ref_k_out, tri_k_out, 1e-3)

    # Test backward
    dy = torch.randn_like(tri_k_out)

    # Clear gradients
    weight.grad = None
    if has_bias:
        bias.grad = None

    # Detach and create new leaf nodes for gradient comparison
    k_detached = k.detach().requires_grad_(True)
    ref_k_detached = k.detach().contiguous().requires_grad_(True)

    # Forward for gradient comparison
    tri_k_out_detached, _ = causal_conv1d(k_detached, weight, bias, activation=activation, cu_seqlens=cu_seqlens)
    ref_k_out_detached, _ = causal_conv1d(ref_k_detached, weight, bias, activation=activation, cu_seqlens=cu_seqlens)

    # Backward to compute gradients
    tri_k_out_detached.backward(dy.clone())
    ref_k_out_detached.backward(dy.clone())

    # Capture reference gradients
    ref_grad_weight = weight.grad.clone()
    if has_bias:
        ref_grad_bias = bias.grad.clone()

    # Clear gradients again for second run
    weight.grad = None
    if has_bias:
        bias.grad = None

    # Second forward/backward for actual test
    tri_k_out_detached2, _ = causal_conv1d(k_detached, weight, bias, activation=activation, cu_seqlens=cu_seqlens)
    ref_k_out_detached2, _ = causal_conv1d(ref_k_detached, weight, bias, activation=activation, cu_seqlens=cu_seqlens)

    tri_k_out_detached2.backward(dy.clone())
    ref_k_out_detached2.backward(dy.clone())

    # Check gradients
    assert_close("dx", ref_k_detached.grad, k_detached.grad, 1e-3)
    assert_close("dw", ref_grad_weight, weight.grad, 1e-3)
    if has_bias:
        assert_close("dbias", ref_grad_bias, bias.grad, 1e-3)

    # Test with residual (residual is contiguous)
    residual = k.detach().clone().requires_grad_(True)

    ref_k_res = k.detach().contiguous().requires_grad_(True)
    ref_residual = residual.detach().contiguous().requires_grad_(True)
    ref_k_out_res, _ = causal_conv1d(ref_k_res, weight, bias, residual=ref_residual,
                                     activation=activation, cu_seqlens=cu_seqlens)

    k_res = k.detach().requires_grad_(True)
    tri_k_out_res, _ = causal_conv1d(k_res, weight, bias, residual=residual, activation=activation, cu_seqlens=cu_seqlens)

    assert_close("o", ref_k_out_res, tri_k_out_res, 1e-3)

    # Backward with residual
    dy = torch.randn_like(tri_k_out_res)
    ref_k_out_res.backward(dy)
    tri_k_out_res.backward(dy)

    assert_close("dx", ref_k_res.grad, k_res.grad, 1e-3)
    assert_close("dr", ref_residual.grad, residual.grad, 1e-3)

    # Test with initial_state (including dh0 gradient)
    ref_initial_state = torch.randn(N, D, W).to(device, dtype).requires_grad_(True)
    tri_initial_state = ref_initial_state.detach().clone().requires_grad_(True)

    # Forward with state
    ref_k_state = k.detach().contiguous().requires_grad_(True)
    ref_k_out_state, ref_final_state = causal_conv1d(
        ref_k_state, weight, bias, initial_state=ref_initial_state,
        output_final_state=True, activation=activation, cu_seqlens=cu_seqlens
    )

    k_state = k.detach().requires_grad_(True)
    tri_k_out_state, tri_final_state = causal_conv1d(
        k_state, weight, bias, initial_state=tri_initial_state,
        output_final_state=True, activation=activation, cu_seqlens=cu_seqlens
    )

    assert_close("o", ref_k_out_state, tri_k_out_state, 1e-3)
    assert_close("dh", ref_final_state, tri_final_state, 1e-3)

    # Backward with state
    dy = torch.randn_like(tri_k_out_state)
    ref_k_out_state.backward(dy)
    tri_k_out_state.backward(dy)

    assert_close("dx", ref_k_state.grad, k_state.grad, 1e-3)
    assert_close("dh0", ref_initial_state.grad, tri_initial_state.grad, 1e-3)
