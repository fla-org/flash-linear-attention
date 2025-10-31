
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
def test_conv(
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

    tri, _ = causal_conv1d(x, weight, bias, residual=residual, activation=activation)
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
    ('N', 'T', 'D', 'W', 'activation', 'has_bias', 'has_residual', 'dtype'),
    [
        pytest.param(*test, id="N{0}_T{1}_D{2}_W{3}_activation{4}_has_bias{5}_has_residual{6}_{7}".format(*test))
        for test in [
            (4, 500, 128, 3, "swish", True, True, torch.float32),
            (4, 1024, 200, 4, "swish", False, True, torch.float32),
            (4, 500, 128, 3, None, True, False, torch.float16),
            (4, 1024, 1024, 4, None, False, False, torch.float16),
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
):
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

    tri, _ = causal_conv1d(x, weight, bias, residual=residual, activation=activation, cu_seqlens=cu_seqlens)
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
    ('B', 'T', 'D', 'W', 'has_bias', 'has_residual', 'activation', 'dtype'),
    [
        pytest.param(*test, id="B{0}_T{1}_D{2}_W{3}_has_bias{4}_has_residual{5}_activation{6}_{7}".format(*test))
        for test in [
            (2, 64, 128, 3, True, True, "swish", torch.float32),
            (2, 128, 128, 4, False, True, "swish", torch.float32),
            (2, 64, 128, 3, True, False, "swish", torch.float32),
            (2, 128, 128, 4, False, False, "swish", torch.float32),
        ]
    ],
)
@torch.no_grad
def test_mixed_backend(
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
    T_decode = 1
    x = torch.randn(B, T + T_decode, D, device=device, dtype=dtype)
    residual = torch.randn_like(x) if has_residual else None

    conv = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=has_bias,
        activation=activation,
        backend="cuda",
        device=device,
        dtype=dtype,
    )

    cache = torch.randn(B, D, W-1, device=device, dtype=dtype)
    y_cuda_prefill, final_state = conv(
        x[:, :T],
        residual=residual[:, :T] if has_residual else None,
        cache=cache,
        output_final_state=True,
    )

    conv.backend = "triton"
    y_triton_decode, _ = conv(
        x[:, T:],
        residual=residual[:, T:] if has_residual else None,
        cache=final_state,
        output_final_state=True,
    )

    conv.backend = "triton"
    cache = torch.cat((torch.zeros_like(cache[..., :1]), cache), -1)
    y_triton_full, _ = conv(x, residual=residual, cache=cache)

    y_mixed = torch.cat([y_cuda_prefill, y_triton_decode], dim=1)
    assert_close("cuda→triton vs triton", y_mixed, y_triton_full, 1e-3)

    conv.backend = "triton"
    y_triton_prefill, final_state = conv(
        x[:, :T],
        residual=residual[:, :T] if has_residual else None,
        cache=cache,
        output_final_state=True,
    )

    conv.backend = "cuda"
    y_cuda_decode, _ = conv(
        x[:, T:],
        residual=residual[:, T:] if has_residual else None,
        cache=final_state,
        output_final_state=True,
    )

    y_mixed2 = torch.cat([y_triton_prefill, y_cuda_decode], dim=1)
    assert_close("triton→cuda vs triton", y_mixed2, y_triton_full,  1e-3)


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
