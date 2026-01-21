import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def fft_conv(u, k, dropout_mask, gelu=True, k_rev=None):
    """
    Apply a convolution in the frequency domain between input `u` and time-domain kernel `k`, add the input as a residual connection, then optionally apply GELU activation and scale by a dropout mask.
    
    Parameters:
        u (torch.Tensor): Input tensor whose last dimension is the sequence length.
        k (torch.Tensor): Time-domain convolution kernel compatible with `u`'s channel layout.
        dropout_mask (torch.Tensor or None): Optional mask with shape broadcastable to `[batch, H, 1]`; when provided the output is multiplied by this mask and cast back to `u`'s dtype.
        gelu (bool): If True, apply GELU to the residual-added result before dropout masking.
        k_rev (torch.Tensor or None): Optional second kernel whose frequency-domain conjugate is added to `k`'s frequency response (used to include a reversed or symmetric component).
    
    Returns:
        torch.Tensor: The convolved tensor with the same shape as `u`, cast to `u`'s dtype.
    """
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
        Create a LongConvolution module and initialize its learnable convolution filter.
        
        Initializes a trainable parameter `filter` of shape (hidden_size, max_len) with a standard normal distribution and stores `hidden_size`.
        
        Parameters:
            hidden_size (int): Number of input/output feature channels; determines first dimension of `filter`.
            max_len (int): Maximum convolution length; determines second dimension of `filter`.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.filter = nn.Parameter(torch.randn(self.hidden_size, max_len), requires_grad=True)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Apply the module's learned FFT-based long convolution to an input sequence and return the convolved output.
        
        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size] with the same dtype as `x`.
        """
        x = x.transpose(1, 2)
        y = fft_conv(x, self.filter, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """
        Create fixed complex-exponential positional embeddings used to parameterize implicit convolution filters.
        
        Parameters:
            emb_dim (int): Embedding dimension. The first channel is a normalized time embedding; the remaining channels are paired real/imag components of complex exponentials (so emb_dim should typically be 1 + 2 * bands).
            seq_len (int): Maximum sequence length; embeddings are created for positions [0, seq_len-1].
        
        Notes:
            - Produces self.z with shape (1, seq_len, emb_dim), stored as a non-trainable parameter.
            - The first channel is t normalized to [0, 1]. Remaining channels are real and imaginary parts of exp(-i * f * w) for a set of frequencies f.
        """
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
        """
        Return the first L positional embeddings.
        
        Parameters:
            L (int): Number of positions to return.
        
        Returns:
            torch.Tensor: Complex-valued positional embeddings for positions [0, L), shape (embedding_dim, L).
        """
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
        Construct an implicit long-range convolution module whose filter is produced by an MLP conditioned on positional embeddings.
        
        Parameters:
        	hidden_size (int): Number of input/output channels (the MLP output dimension).
        	max_len (int): Maximum sequence length supported by the internal positional embeddings.
        	d_emb (int, optional): Dimensionality of positional embeddings; must be odd and >= 3 (default: 3).
        	d_hidden (int, optional): Hidden dimension of the MLP (default: 16).
        	**kwargs: Ignored; present for API compatibility.
        
        Details:
        	Creates a PositionalEmbedding of dimension `d_emb` for sequences up to `max_len` and a two-layer MLP
        	that maps per-position embeddings of size `d_emb` to a filter of size `hidden_size`. The MLP architecture is
        	Linear(d_emb -> d_hidden) -> ReLU -> Linear(d_hidden -> hidden_size).
        
        Raises:
        	AssertionError: If `d_emb` is not odd or is less than 3.
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
        """
        Generate a convolution kernel for a given sequence length by applying the positional embeddings to the module MLP.
        
        Parameters:
            seq_len (int): Desired sequence length for the generated kernel.
        
        Returns:
            torch.Tensor: Kernel tensor with shape (1, hidden_size, seq_len), where the middle dimension indexes output channels/features and the last dimension indexes time positions.
        """
        return self.mlp(self.pos_emb(seq_len)).transpose(1, 2)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Apply the module's implicit long-range convolution filter to the input via FFT.
        
        Parameters:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size] with the same dtype as the input.
        """
        x = x.transpose(1, 2)
        k = self.filter(x.shape[-1])
        y = fft_conv(x, k, dropout_mask=None, gelu=False)

        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)