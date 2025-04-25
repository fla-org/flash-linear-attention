# -*- coding: utf-8 -*-

# from https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/convolution.py

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

from fla.modules.activations import ACT2FN
from fla.ops.utils import prepare_sequence_ids
from fla.utils import checkpoint, input_guard

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None


def fft_conv(u, k, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


@checkpoint
def proj_then_conv1d(
    x: torch.Tensor,
    proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: Optional[torch.Tensor] = None,
    cache: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # We do matmul and transpose BLH -> HBL at the same time
    x = rearrange(proj_weight @ rearrange(x, "b t d -> d (b t)"), "d (b t) -> b d t", t=x.shape[-2])

    if causal_conv1d_fn is None:
        raise ImportError("`causal_conv1d_fn` is not available. Please install `causal-conv1d` first.")
    if cache is None:
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(conv1d_weight, "d 1 w -> d w"),
            bias=conv1d_bias,
            activation="silu",
        ).transpose(1, 2)
    else:
        assert x.shape[-1] == 1, "Only support decoding with 1 token at a time for now"
        x = x.squeeze(-1)
        x = causal_conv1d_update(
            x=x,
            weight=rearrange(conv1d_weight, "d 1 w -> d w"),
            bias=conv1d_bias,
            cache=cache,
            activation="silu",
        )
    return x


@triton.jit
def causal_conv1d_varlen_states_fwd_kernel(
    x,
    cache,
    offsets,
    D,
    W,
    BD: tl.constexpr,
    BW: tl.constexpr
):
    i_d, i_w, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    eos = tl.load(offsets + i_n + 1)
    bos = tl.maximum(tl.load(offsets + i_n), eos - W)
    o_t = eos - (i_w + 1) * BW + tl.arange(0, BW)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = W - (i_w + 1) * BW + tl.arange(0, BW)

    b_x = tl.load(x + o_t * D + o_d[:, None], mask=(o_t >= bos) & (o_d[:, None] < D), other=0)
    tl.store(cache + i_n * D*W + o_d[:, None] * W + o_w, b_x, mask=(o_d[:, None] < D) & (o_w >= 0))


@input_guard
def causal_conv1d_varlen_states_fwd(
    x: torch.Tensor,
    cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_len: int
) -> torch.Tensor:
    N, D, W = len(cu_seqlens) - 1, x.shape[-1], state_len
    cache = torch.empty(N, D, W, dtype=x.dtype, device=x.device) if cache is None else cache
    BD = min(triton.next_power_of_2(D), 256)
    BW = min(triton.next_power_of_2(state_len), 16)
    grid = (triton.cdiv(D, BD), triton.cdiv(W, BW), N)
    with torch.cuda.device(x.device.index):
        causal_conv1d_varlen_states_fwd_kernel[grid](
            x=x,
            cache=cache,
            offsets=cu_seqlens,
            D=D,
            W=W,
            BW=BW,
            BD=BD
        )
    return cache


class ShortConvolution(nn.Conv1d):
    """
    Simple wrapper around `nn.Conv1d` that accepts dimension last.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: Optional[str] = 'silu',
        use_fast_conv1d: Optional[bool] = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if causal_conv1d_fn is None:
            if use_fast_conv1d:
                raise RuntimeError(
                    "Please either install `causal-conv1d>=1.4.0` to enable fast causal short convolution CUDA kernel "
                    "or set `use_fast_conv1d` to False"
                )
            else:
                warnings.warn(
                    "The naive Pytorch verison is very slow in practice, "
                    "please run `pip install causal-conv1d>=1.4.0` to install fast causal short convolution CUDA kernel",
                    category=ImportWarning
                )
        self.use_fast_conv1d = use_fast_conv1d

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        if not self.use_fast_conv1d:
            s += ', use_fast_conv1d={use_fast_conv1d}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`.
                If `seq_idx` is provided, `B` must be 1.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        B, T, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))
        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)
        # during the decoding phase, we assume the batch is composed of sequences of length 1
        if cache is not None and B * T == N:
            return self.step(x, cache, cu_seqlens)

        if cache is not None:
            if cu_seqlens is not None:
                cache = causal_conv1d_varlen_states_fwd(x, cache, cu_seqlens, W)
            else:
                cache[:, :, -min(W, T):].copy_(rearrange(x[..., -min(W, T):, :], 'n w d -> n d w'))

        x = rearrange(x, 'b t d -> b d t')
        if self.use_fast_conv1d:
            # Sequence index for each token. Used for varlen.
            # Suppose a batch consists of two sequences with lengths 3 and 4,
            # seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
            # NOTE: No need to provide this arg if `cu_seqlens` is passed.
            # This arg is just for BC, and will be removed in the future.
            # [B, T]
            seq_idx = kwargs.get('seq_idx', None)
            if cu_seqlens is not None and seq_idx is None:
                seq_idx = prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0)
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
        else:
            if cu_seqlens is not None:
                raise ValueError("`cu_seqlens` is not supported for the naive Pytorch version")
            x = self._conv_forward(x, self.weight, self.bias)[..., :x.shape[-1]]
            if self.activation is not None:
                x = ACT2FN[self.activation](x)
        return rearrange(x, "b d t -> b t d"), cache

    def step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None
    ):
        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        if self.use_fast_conv1d:
            x = causal_conv1d_update(
                x=x,
                conv_state=cache,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )
        else:
            dtype = x.dtype
            # we follow the fast mode that updates the cache in-place
            cache.copy_(cache.roll(shifts=-1, dims=-1))
            cache[:, :, -1] = x
            x = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
            if self.bias is not None:
                x = x + self.bias
            if self.activation is not None:
                x = ACT2FN[self.activation](x).to(dtype=dtype)
        return x.view(shape), cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size


class LongConvolution(nn.Module):
    """
    LongConvolution applies a convolution operation on the input tensor using a fixed
    filter of length max_len.
    The filter is learned during training and is applied using FFT convolution.
    Args:
        hidden_size (int): The number of expected features in the input and output.
        max_len (int): The maximum sequence length.
    Returns:
        y: [batch_size, seq_len, hidden_size] tensor
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        **kwargs,
    ):
        """
        Initializes the LongConvolution module.
        Args:
            hidden_size (int): The number of expected features in the input and output.
            max_len (int): The maximum sequence length.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.filter = nn.Parameter(torch.randn(self.hidden_size, max_len), requires_grad=True)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Applies the LongConvolution operation on the input tensor.
        Args:
            x: [batch_size, seq_len, hidden_size] tensor
        Returns:
            y: [batch_size, seq_len, hidden_size] tensor
        """
        x = x.transpose(1, 2)
        y = fft_conv(x, self.filter, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for implicit long convolution filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L]


class ImplicitLongConvolution(nn.Module):
    """
    Long convolution with implicit filter parameterized by an MLP.

    Args:
        hidden_size (int):
            The number of expected features in the input and output.
        max_len (int):
            The maximum sequence length.
        d_emb (Optional[int]):
            The dimension of the positional embeddings. Must be odd and greater or equal to 3 (time, sine and cosine).
            Defaults to 3.
        d_hidden (Optional[int]):
            The number of features in the hidden layer of the MLP. Defaults to 16.

    Attributes:
        pos_emb (`PositionalEmbedding`): The positional embedding layer.
        mlp (`nn.Sequential`): The MLP that parameterizes the implicit filter.

    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        d_emb: int = 3,
        d_hidden: int = 16,
        **kwargs,
    ):
        """
        Long convolution with implicit filter parameterized by an MLP.


        """
        super().__init__()
        self.hidden_size = hidden_size
        self.d_emb = d_emb

        assert (
            d_emb % 2 != 0 and d_emb >= 3
        ), "d_emb must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(d_emb, max_len)

        # final linear layer
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            torch.nn.ReLU(),
            nn.Linear(d_hidden, hidden_size),
        )

    def filter(self, seq_len: int, *args, **kwargs):
        k = self.mlp(self.pos_emb(seq_len))

        return k.transpose(1, 2)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [batch_size, seq_len, hidden_size] tensor
        Returns:
            y: [batch_size, seq_len, hidden_size] tensor
        """
        x = x.transpose(1, 2)
        k = self.filter(x.shape[-1])
        y = fft_conv(x, k, dropout_mask=None, gelu=False)

        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)
