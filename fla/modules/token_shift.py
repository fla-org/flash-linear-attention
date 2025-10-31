

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard, is_amd, tensor_cache

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [2, 4, 8, 16, 32]


def token_shift_ref(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if cu_seqlens is not None:
        # Variable length mode with cu_seqlens
        assert x.dim() == 3, "Input must be [B, T, D]"
        B, T, D = x.shape
        assert B == 1, "Batch size must be 1 when using cu_seqlens"

        result = torch.zeros_like(x)
        N = cu_seqlens.shape[0] - 1

        for i in range(N):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i+1].item()
            seq_len = end - start

            if seq_len <= 1:
                # For sequences of length 1 or 0, delta is simply -x
                result[0, start:end] = -x[0, start:end]
            else:
                # For longer sequences, handle padding manually
                shifted = torch.zeros_like(x[0, start:end])
                shifted[1:] = x[0, start:end-1]
                delta = shifted - x[0, start:end]
                result[0, start:end] = delta

        return result
    else:
        time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
        shifted = time_shift(x)
        delta = shifted - x
        return delta


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['cache'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=['BD'],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_fwd_kernel_short(
    x,
    y,
    cu_seqlens,
    cache,
    cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        g_t = i_t + bos

        if g_t >= eos:
            return

        is_first_pos = (i_t == 0)
        is_last_pos = (g_t == eos - 1)
    else:
        g_t = i_t
        is_first_pos = (g_t == 0)
        is_last_pos = (g_t == T - 1)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        base_offset = g_t * D + o_d
    else:
        base_offset = i_b * T*D + g_t * D + o_d

    b_x = tl.load(x + base_offset, mask=m_d)
    if IS_VARLEN:
        cache_offset = i_n * D + o_d  # i_n is seq index
    else:
        cache_offset = i_b * D + o_d  # i_b is batch index

    if IS_DECODE and USE_INITIAL_STATE:
        b_cache = tl.load(cache + cache_offset, mask=m_d)
        delta = b_cache - b_x
        tl.store(y + base_offset, delta, mask=m_d)
        if STORE_FINAL_STATE:
            tl.store(cache_out + cache_offset, b_x, mask=m_d)
        return

    if is_first_pos:
        # First position in sequence: delta = -hidden_states
        if USE_INITIAL_STATE:
            # cache shape: [N, D]
            b_cache = tl.load(cache + cache_offset, mask=m_d)
            delta = b_cache - b_x
            tl.store(y + base_offset, delta, mask=m_d)
        else:
            tl.store(y + base_offset, -b_x, mask=m_d)
        return

    # Other positions: delta = prev - curr
    if IS_VARLEN:
        prev_offset = (g_t-1) * D + o_d
    else:
        prev_offset = i_b * T*D + (g_t-1) * D + o_d

    prev_values = tl.load(x + prev_offset, mask=m_d)
    delta = prev_values - b_x
    tl.store(y + base_offset, delta, mask=m_d)
    if STORE_FINAL_STATE:
        if is_last_pos:
            tl.store(cache_out + cache_offset, b_x, mask=m_d)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['cache'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=['BD', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_fwd_kernel_long(
    x,
    y,
    cu_seqlens,
    chunk_indices,
    cache,
    cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), \
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        t_start = i_t * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        i_n = i_b
        bos, eos = i_b * T, (i_b + 1) * T
        t_start = i_t * BT
        t_end = tl.minimum(t_start + BT, T)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    for t in range(t_start, t_end):
        global_t = bos + t
        offset = global_t * D + o_d
        b_x = tl.load(x + offset, mask=m_d)
        is_first = (global_t == bos)
        if is_first:
            if USE_INITIAL_STATE:
                # cache shape: [N, D]
                cache_off = i_n * D + o_d if IS_VARLEN else i_b * D + o_d
                b_cache = tl.load(cache + cache_off, mask=m_d)
                delta = b_cache - b_x
            else:
                delta = -b_x
        else:
            prev_off = offset - D
            b_prev = tl.load(x + prev_off, mask=m_d)
            delta = b_prev - b_x

        tl.store(y + offset, delta, mask=m_d)

        if STORE_FINAL_STATE:
            if global_t == eos - 1:
                cache_out_off = i_n * D + o_d if IS_VARLEN else i_b * D + o_d
                tl.store(cache_out + cache_out_off, b_x, mask=m_d)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['grad_cache_out'] is not None,
    'HAS_DCACHE': lambda args: args['grad_cache_in'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=['BD'],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_bwd_kernel_short(
    dx,
    dy,
    cu_seqlens,
    grad_cache_in,
    grad_cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_DCACHE: tl.constexpr,
):
    i_b, i_t = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        g_t = i_t + bos
        if g_t >= eos:
            return
        is_first_pos = (g_t == bos)
        is_last_pos = (g_t == eos - 1)
    else:
        g_t = i_t
        is_first_pos = (g_t == 0)
        is_last_pos = (g_t == T - 1)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        base_offset = g_t * D + o_d
        # This should not be used for varlen
        cache_off = i_n * D + o_d
    else:
        base_offset = i_b * T * D + g_t * D + o_d
        cache_off = i_b * D + o_d

    b_dy = tl.load(dy + base_offset, mask=m_d)

    if is_last_pos:
        # grad = -grad_delta[t] + grad_cache_in（from next rank）
        if HAS_DCACHE:
            b_dy_cache = tl.load(grad_cache_in + cache_off, mask=m_d)
            b_dx = -b_dy + b_dy_cache
        else:
            b_dx = -b_dy
    else:
        # grad = -grad_delta[t] + grad_delta[t+1]
        if IS_VARLEN:
            next_offset = (g_t + 1) * D + o_d
        else:
            next_offset = i_b * T * D + (g_t + 1) * D + o_d
        b_dx = -b_dy + tl.load(dy + next_offset, mask=m_d)

    tl.store(dx + base_offset, b_dx, mask=m_d)

    if USE_INITIAL_STATE:
        if is_first_pos:
            tl.store(grad_cache_out + cache_off, b_dy, mask=m_d)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_INITIAL_STATE': lambda args: args['grad_cache_out'] is not None,
    'HAS_DCACHE': lambda args: args['grad_cache_in'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [1, 2, 3]
    ],
    key=['BD', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit
def token_shift_bwd_kernel_long(
    dx,
    dy,
    cu_seqlens,
    chunk_indices,
    grad_cache_in,
    grad_cache_out,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    NB: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_DCACHE: tl.constexpr,
):
    i_d, i_t_blk, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t_blk = tl.load(chunk_indices + i_t_blk * 2).to(tl.int32), \
            tl.load(chunk_indices + i_t_blk * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        bos, eos = i_b * T, (i_b + 1) * T
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, T)

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    cache_off = i_n * D + o_d if IS_VARLEN else i_b * D + o_d

    for t in range(t_start, t_end):
        global_t = bos + t
        offset = global_t * D + o_d
        b_dy = tl.load(dy + offset, mask=m_d)

        if global_t == eos - 1:
            if HAS_DCACHE:
                b_dy_cache = tl.load(grad_cache_in + cache_off, mask=m_d)
                b_dx = -b_dy + b_dy_cache
            else:
                b_dx = -b_dy
        else:
            next_off = offset + D
            b_dx = -b_dy + tl.load(dy + next_off, mask=m_d)

        tl.store(dx + offset, b_dx, mask=m_d)

        if USE_INITIAL_STATE:
            if global_t == bos:
                tl.store(grad_cache_out + cache_off, b_dy, mask=m_d)


@tensor_cache
def prepare_maxlens(cu_seqlens: torch.LongTensor) -> int:
    return torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item()


def token_shift_fwd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
    output_cache: bool = False,
) -> torch.Tensor:
    B, T, D = x.shape
    y = torch.empty_like(x)
    use_short_kernel = T <= 4096

    if cu_seqlens is not None:
        T = prepare_maxlens(cu_seqlens)
        N = len(cu_seqlens) - 1
    else:
        N = B

    if output_cache:
        cache_out = torch.empty((N, D), device=x.device, dtype=x.dtype)
    else:
        cache_out = None

    if use_short_kernel:
        if cu_seqlens is not None:
            N = len(cu_seqlens) - 1
        else:
            N = B
        BD = triton.next_power_of_2(D)
        grid = (N, T)
        IS_DECODE = T == 1 or (B == 1 and T == N)
        token_shift_fwd_kernel_short[grid](
            x=x,
            y=y,
            cu_seqlens=cu_seqlens,
            cache=cache,
            cache_out=cache_out,
            T=T,
            D=D,
            BD=BD,
            STORE_FINAL_STATE=output_cache,
            IS_DECODE=IS_DECODE,
        )
    else:
        BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B*T), get_multiprocessor_count(x.device.index))))
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

        BD = triton.next_power_of_2(D)
        NB = triton.cdiv(B*T, 1024)

        def grid(meta): return (triton.cdiv(D, meta['BD']), NT, N)
        token_shift_fwd_kernel_long[grid](
            x,
            y,
            cu_seqlens,
            chunk_indices,
            cache,
            cache_out,
            T,
            D=D,
            BD=BD,
            BT=BT,
            NB=NB,
            STORE_FINAL_STATE=output_cache,
        )

    return y, N, T, use_short_kernel, cache_out


def token_shift_bwd(
    dy: torch.Tensor,
    N: int,
    T: int,
    dcache: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    use_short_kernel: bool = True,
    has_init_cache: bool = False,
) -> torch.Tensor:
    D = dy.shape[2]
    BD = triton.next_power_of_2(D)
    dx = torch.empty_like(dy)
    if has_init_cache:
        grad_cache_out = torch.empty((N, D), device=dy.device, dtype=dy.dtype)
    else:
        grad_cache_out = None
    if use_short_kernel:
        grid = (N, T)
        token_shift_bwd_kernel_short[grid](
            dy=dy,
            dx=dx,
            cu_seqlens=cu_seqlens,
            grad_cache_in=dcache,
            grad_cache_out=grad_cache_out,
            T=T,
            D=D,
            BD=BD,
        )
    else:
        BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, dy.numel() // D),
                                                        get_multiprocessor_count(dy.device.index))))
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
        NB = triton.cdiv(N * dy.shape[1], 1024)
        BD = triton.next_power_of_2(D)

        def grid(meta): return (triton.cdiv(D, meta['BD']), NT, N)
        token_shift_bwd_kernel_long[grid](
            dx,
            dy,
            cu_seqlens,
            chunk_indices,
            dcache,
            grad_cache_out,
            T,
            D=D,
            BD=BD,
            BT=BT,
            NB=NB,
        )
    return dx, grad_cache_out


class TokenShift(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
                cache: torch.Tensor | None = None, output_cache: bool = False):
        output, N, T, use_short_kernel, cache_out = token_shift_fwd(x, cu_seqlens, cache, output_cache)
        ctx.cu_seqlens = cu_seqlens
        ctx.N = N
        ctx.T = T
        ctx.use_short_kernel = use_short_kernel
        ctx.has_cache = cache is not None
        return output, cache_out

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor, dcache: torch.Tensor | None = None):
        dx, grad_cache = token_shift_bwd(dy, ctx.N, ctx.T, dcache, ctx.cu_seqlens,
                                         ctx.use_short_kernel, ctx.has_cache)
        return dx, None, grad_cache, None


def token_shift(
    x: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    cache: torch.Tensor | None = None,
    output_cache: bool = False,
):
    """
    Token-shift operation implemented with Triton kernels.

    Args:
        x: Input tensor of shape [B, T, D] (or [1, T, D] when `cu_seqlens` is supplied).
        cu_seqlens: Optional cumulative sequence lengths of shape [B + 1].
                    When supplied, `x.shape[0]` must be 1 and `x.dim()` must be 3.
        cache: Optional cache tensor of shape [N, D] that holds the last token
               from the previous call.
        output_cache: Whether to return the updated cache alongside the output.
                      In previous versions this parameter did not exist and the
                      cache was always dropped; to preserve backward compatibility
                      the default is False.

    Returns:
        output: Tensor of shape [B, T, D] after applying the token-shift.

        cache_out: Tensor of shape [B, 1, D] containing the last token that
                   should be fed as `cache` in the next call.  Only returned
                   when `output_cache=True`.
    """
    if cu_seqlens is not None:
        assert x.dim() == 3, "Input must be [B, T, D]"
        assert x.shape[0] == 1, "Batch size must be 1 when using cu_seqlens"

    output, cache_out = TokenShift.apply(x, cu_seqlens, cache, output_cache)
    if output_cache:
        return output, cache_out
    else:
        return output
