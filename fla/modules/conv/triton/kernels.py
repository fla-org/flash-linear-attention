import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.utils import IS_AMD, autotune_cache_kwargs, input_guard

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]
STATIC_WARPS = 32 if not IS_AMD else 16


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'W', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Compute a block-wise forward pass of a 1D causal convolution into the output tensor `y`.
    
    This Triton kernel reads a block of input sequence frames (with optional variable-length handling and an initial state),
    applies a causal convolution over a window of width `W` with optional `weight` and `bias`, applies an optional
    activation ("swish"/"silu"), adds an optional `residual`, and writes the resulting block into `y`.
    
    Parameters:
        x: Input tensor buffer (flattened pointer used by Triton) containing sequence data.
        y: Output tensor buffer where computed blocks are stored.
        weight: Optional convolution weight buffer with layout [D, W] accessed when HAS_WEIGHT is true.
        bias: Optional bias buffer of length D accessed when HAS_BIAS is true.
        residual: Optional residual tensor buffer added to the result when HAS_RESIDUAL is true.
        cu_seqlens: Cumulative sequence lengths used when IS_VARLEN is true to locate per-sequence ranges.
        initial_state: Optional initial-state buffer of shape (N, D * W) used for pre-history when USE_INITIAL_STATE is true.
        chunk_indices: Index buffer used to map program grid indices to sequence indices when IS_VARLEN is true.
        B: Number of batches (or batch-related extent) in the launch grid.
        T: Sequence length (per-chunk length when varlen handling is disabled).
        stride_x_n, stride_x_t, stride_x_d: Strides for indexing `x` along batch/sequence/channel dimensions.
        D (tl.constexpr): Channel dimension size.
        W (tl.constexpr): Convolution window width.
        BT, BW, BD, NB (tl.constexpr): Block-size and launch-tuning constants used to partition time and channel dimensions.
        ACTIVATION (tl.constexpr): Activation identifier; supports 'swish'/'silu' to apply x * sigmoid(x).
        HAS_WEIGHT, HAS_BIAS, HAS_RESIDUAL, USE_INITIAL_STATE, IS_VARLEN (tl.constexpr): Compile-time flags that enable corresponding features and code paths.
    
    Notes:
        - The kernel writes directly into `y` and does not return a value.
        - Behavior differs for variable-length sequences (IS_VARLEN) vs fixed-length, and for positions early in the sequence
          where initial-state values may be read when USE_INITIAL_STATE is true.
    """
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
        p_x = x + i_b * stride_x_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(p_x, (T, D), (stride_x_t, stride_x_d), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        # to make Triton compiler happy, we need to copy codes
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(p_x, (T, D), (stride_x_t, stride_x_d), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(
                p_x + o_x[:, None] * stride_x_t + o_d * stride_x_d,
                mask=m_x,
                other=0
            ).to(tl.float32)

            b_yi += tl.load(initial_state + i_n * D*W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(tl.float32)

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [4, 8, 16, 32]
    ],
    key=['D', 'W', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_x_n,   # x batch stride
    stride_x_t,   # x time stride
    stride_x_d,   # x dim stride
    stride_dx_n,  # dx batch stride
    stride_dx_t,  # dx time stride
    stride_dx_d,  # dx dim stride
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Compute gradients for a 1D causal convolution and store them into dx, dw, db, and optionally dht.
    
    This Triton kernel computes the gradient of the causal_conv1d forward pass with respect to the input (dx), weight (dw), bias (db), and — when enabled — the final/initial state gradient buffer (dht). It supports variable-length batches, optional weight/bias, optional initial/final state handling, and activation-aware gradient modulation for swish/silu.
    
    Parameters:
        x: Input tensor pointer for forward-pass values.
        y: Output tensor pointer from the forward pass.
        weight: Pointer to convolution weights (may be None if HAS_WEIGHT is false).
        initial_state: Pointer to initial-state cache used when USE_INITIAL_STATE is true.
        dht: Pointer to final-state gradient buffer (used when USE_FINAL_STATE is true).
        dy: Pointer to gradients of the output y.
        dx: Pointer where computed input gradients will be written.
        dw: Pointer where computed weight gradients will be accumulated (when HAS_WEIGHT is true).
        db: Pointer where computed bias gradients will be accumulated (when HAS_BIAS is true).
        cu_seqlens: Cumulative sequence lengths (used when IS_VARLEN is true) to derive per-sample sequence boundaries.
        chunk_indices: Index pairs mapping program grid indices to (sample, time-offset) for varlen execution.
        B: Number of batches (program-grid related).
        T: Time dimension length (per-chunk or full sequence length depending on IS_VARLEN).
        stride_x_n, stride_x_t, stride_x_d: x tensor strides for batch, time, and channel dimensions.
        stride_dx_n, stride_dx_t, stride_dx_d: dx tensor strides for batch, time, and channel dimensions.
        D (constexpr): Channel dimension size.
        W (constexpr): Convolution kernel width (receptive field).
        BT, BW, BD, NB (constexpr): Block configuration constants used by the kernel grid.
        ACTIVATION (constexpr): Activation mode; supports 'swish'/'silu' to apply activation-aware gradient correction.
        HAS_WEIGHT (constexpr): If true, weight gradients (dw) are computed/updated.
        HAS_BIAS (constexpr): If true, bias gradients (db) are computed/updated.
        USE_INITIAL_STATE (constexpr): If true, gradients include contributions from the provided initial_state cache.
        USE_FINAL_STATE (constexpr): If true, the kernel reads dht contributions into dx for tail timesteps.
        IS_VARLEN (constexpr): If true, per-sample variable-length sequences are used and cu_seqlens/chunk_indices are consulted.
    
    Notes:
        - The kernel writes results directly into the provided dx, dw, db, and reads/writes dht as controlled by the flags.
        - Behavior differs across branches to support varlen/fixed-length inputs and initial/final state options; activation correction is applied only when ACTIVATION is 'swish' or 'silu'.
    """
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        p_x = x + bos * stride_x_t
    else:
        i_tg = i_b * tl.num_programs(1) + i_t
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
        p_x = x + i_b * stride_x_n

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        p_x = tl.make_block_ptr(p_x, (T, D), (stride_x_t, stride_x_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1))
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    if not USE_FINAL_STATE and not USE_INITIAL_STATE:
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                # [BT, BD]
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                # [BD]
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    elif i_t * BT >= W:
        # to make Triton compiler happy, we need to copy codes
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                # [BT, BD]
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                # [BD]
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    else:
        # which may use initial state
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y_shift = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y_shift)
                b_dy_shift = b_dy_shift * b_ys * (1 + b_y_shift * (1 - b_ys))
            if HAS_WEIGHT:
                # gradient comes from x：sum_t dy[t+i_w] * x[t]
                b_dw = tl.sum(b_dy_shift * b_x, 0)
                # index of cache：c = W - i_w + t
                if USE_INITIAL_STATE:
                    mask_head_rows = (o_t < i_w)
                    # dy_head = dy[t]
                    b_dy_head = tl.load(dy + bos * D + o_t[:, None] * D + o_d, mask=(mask_head_rows[:, None] & m_d[None, :]),
                                        other=0.0).to(tl.float32)
                    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                        # use y[t] （not y[t+i_w]）
                        b_y_head = tl.load(y + bos * D + o_t[:, None] * D + o_d,
                                           mask=(mask_head_rows[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                        b_ys_head = tl.sigmoid(b_y_head)
                        b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                    o_c = W - i_w + o_t
                    # index 0 is padding 0
                    mask_c = (mask_head_rows & (o_c >= 1) & (o_c < W))
                    b_xc = tl.load(initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                                   mask=(mask_c[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                    # add the gradient comes from initial_state
                    b_dw += tl.sum(b_dy_head * b_xc, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy_shift, 0)
            b_wdy = b_dy_shift if not HAS_WEIGHT else (b_dy_shift * tl.sum(b_w * (o_w == (W - i_w - 1)), 1))
            b_dx += b_wdy

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    if USE_FINAL_STATE:
        if i_t * BT + BT >= T-W:
            start_tok = max(0, T - (W - 1))
            offset = i_t * BT + tl.arange(0, BT)
            tok_idx = offset - start_tok
            mask = (offset >= start_tok) & (offset < T)
            w_idx = 1 + tok_idx
            dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.).to(tl.float32)
            b_dx += b_dht

    if IS_VARLEN:
        p_dx = dx + bos * stride_dx_t
    else:
        p_dx = dx + i_b * stride_dx_n

    p_dx = tl.make_block_ptr(p_dx, (T, D), (stride_dx_t, stride_dx_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['cache'] is not None,
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
})
@triton.jit
def causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    stride_x_n,  # batch stride
    stride_x_d,  # dim stride
    stride_y_n,  # batch stride
    stride_y_d,  # dim stride
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """
    Compute an update step for a 1D causal convolution: produce output y for a single (channel block, batch)
    tile and optionally read/write the per-batch input cache.
    
    Given input slice x and a sliding cache of the previous W frames, this kernel:
    - loads a BD-sized channel block from x for batch index i_n;
    - optionally shifts and updates the per-batch cache (when USE_INITIAL_STATE) by inserting the current x at the last cache column;
    - computes the convolution output over the W window using optional per-channel `weight` and optional `bias`;
    - applies the `swish`/`silu` activation if requested;
    - adds an optional `residual` contribution;
    - writes the computed output into y and, when USE_INITIAL_STATE is true, writes back the updated cache.
    
    Parameters with non-obvious meanings:
    - stride_x_n, stride_x_d: strides to index x by batch and channel respectively.
    - stride_y_n, stride_y_d: strides to index y by batch and channel respectively.
    - D, W, BD, BW: compile-time block dimensions: total channels (D), kernel window (W),
      channel block size (BD), and window block size (BW).
    - ACTIVATION: compile-time string; supports 'swish' / 'silu' to enable activation.
    - USE_INITIAL_STATE: compile-time flag to enable reading/updating `cache`.
    - HAS_WEIGHT, HAS_BIAS, HAS_RESIDUAL: compile-time flags to enable use of `weight`, `bias`, and `residual`.
    
    Behavioral notes:
    - All out-of-bounds channel or window lanes are masked.
    - Writes to `cache` are performed with boundary checks to preserve valid entries.
    """
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    m_w = o_w < W

    # [BD]
    b_x = tl.load(x + i_n * stride_x_n + o_d * stride_x_d, mask=m_d, other=0).to(tl.float32)

    b_cache = tl.zeros((BD, BW), dtype=tl.float32)

    if USE_INITIAL_STATE:
        # 2. Shift Cache (Read [1:])
        p_cache_read = tl.make_block_ptr(
            cache + i_n * D*W,
            shape=(D, W),
            strides=(W, 1),
            offsets=(i_d * BD, 1),
            block_shape=(BD, BW),
            order=(1, 0)
        )
        b_cache = tl.load(p_cache_read, boundary_check=(0, 1)).to(tl.float32)

        # 3. Fill x to the last position
        m_update = o_w == (W - 1)
        b_cache = tl.where(m_update[None, :], b_x[:, None], b_cache)

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)
        b_y = tl.sum(b_cache * b_w, 1)
    else:
        b_y = tl.sum(b_cache, 1)

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        b_y += tl.load(residual + i_n * D + o_d, mask=m_d, other=0)

    tl.store(y + i_n * stride_y_n + o_d * stride_y_d, tl.cast(b_y,
             dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_d)

    if USE_INITIAL_STATE:
        p_cache_write = tl.make_block_ptr(
            cache + i_n * D*W,
            shape=(D, W),
            strides=(W, 1),
            offsets=(i_d * BD, 0),
            block_shape=(BD, BW),
            order=(1, 0)
        )
        tl.store(p_cache_write, tl.cast(b_cache, dtype=cache.dtype.element_ty,
                 fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'USE_ACTIVATION': lambda args: args['y'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Compute the gradient with respect to the initial state (dh0) for each batch and channel block across the convolution window.
    
    This kernel accumulates contributions for dh0[:, :, i_w] for i_w in [1, W) by summing dy over time steps t in [0, min(i_w, seq_len) - 1], multiplying by the corresponding weight column for offset (i_w - 1 - t), and applying an activation-aware correction when enabled. For variable-length inputs (IS_VARLEN), sequence boundaries are read from cu_seqlens; otherwise the full length T is used. Results are written in-place into dh0.
    
    Parameters:
        dy: Pointer to output gradients over time (dy). For non-varlen: layout [B, T, D]; for varlen: flattened [1, total_T, D] and offset by cu_seqlens.
        y: Pointer to forward outputs used when USE_ACTIVATION is true to compute activation correction (same layout/offset semantics as dy).
        weight: Pointer to convolution weights of shape [D, W] (stored column-major per channel block in this kernel).
        dh0: Pointer to output buffer where computed initial-state gradients are stored; layout expected [N, D, W] (this kernel writes dh0[i_n, :, i_w]).
        cu_seqlens: Pointer to cumulative sequence lengths used when IS_VARLEN is true; used to compute bos/eos offsets per sequence.
        stride_dy_n: Stride (in elements) to advance dy between batches (used for non-varlen layout).
        stride_dy_t: Stride (in elements) to advance dy between time steps.
        T: Full sequence length (used when IS_VARLEN is false).
        D: Channel dimension size (constexpr).
        W: Convolution window size (constexpr).
        BD: Channel block size processed by the kernel (constexpr).
        USE_ACTIVATION: Constexpr flag; when true applies activation-aware gradient modulation using y.
        IS_VARLEN: Constexpr flag; when true uses cu_seqlens to determine per-sequence start/end and offsets.
    
    Note:
        - The kernel iterates i_w from 1 to W-1; dh0 for i_w == 0 is not produced here.
        - This function writes dh0 in-place and does not return a value.
    """
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    # Get sequence boundaries
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        # For varlen, dy is [1, total_T, D], offset by bos
        dy_base = dy + bos * stride_dy_t
    else:
        seq_len = T
        # For non-varlen, dy is [B, T, D], offset by i_n * stride_dy_n
        dy_base = dy + i_n * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    # For each i_w in [1, W), compute dh0[i_n, :, i_w]
    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        # Accumulate contributions from t = 0 to min(i_w, seq_len) - 1
        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t

                # Load dy[t, :] relative to dy_base
                p_dy = dy_base + t * stride_dy_t + o_d
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = y + bos * stride_dy_t + t * stride_dy_t + o_d
                    else:
                        p_y = y + i_n * stride_dy_n + t * stride_dy_t + o_d
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                # Get weight[:, w_idx]
                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)

                # Accumulate
                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        # Store dh0[i_n, :, i_w]
        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Compute and store the final per-sample channel-major state window for a causal 1D convolution.
    
    This Triton kernel reads a BWxBD tile of the input sequence for a specific channel block and sample, optionally adds a corresponding initial state slice when USE_INITIAL_STATE is true and the sequence is shorter than BW, and writes the transposed tile into final_state as channel-major windows of length W. Supports variable-length sequences when IS_VARLEN is set by using cu_seqlens to derive per-sample start/end offsets.
    
    Parameters:
        x: Pointer or tensor base for input sequence frames laid out with time and channel strides given by stride_x_t and stride_x_d.
        initial_state: Base pointer for initial states stored per-sample as contiguous windows of length W (used only when USE_INITIAL_STATE is true).
        final_state: Destination buffer to store final state windows with layout [N, D, W] (channel-major within each sample).
        cu_seqlens: Per-sample start indices (length N+1) used when IS_VARLEN is true; ignored for fixed-length sequences.
        T: Maximum sequence length (used when IS_VARLEN is false).
        D: Total number of channels.
        W: Window length for the causal convolution (final_state width).
        stride_x_n: Stride (in elements) to advance x between samples (used when IS_VARLEN is false).
        stride_x_t: Stride (in elements) to advance x between time steps.
        stride_x_d: Stride (in elements) to advance x between channels within a time step.
        BD: Block depth (channels per block) as a compile-time constant.
        BW: Block width (time positions per block) as a compile-time constant.
        USE_INITIAL_STATE: Compile-time flag that enables adding initial_state for short sequences.
        IS_VARLEN: Compile-time flag that enables variable-length sequence handling via cu_seqlens.
    
    Notes:
        - final_state is written in channel-major order per sample: final_state[n, d, :] holds the last W positions for channel d of sample n.
        - For sequences shorter than BW when USE_INITIAL_STATE is true, the kernel fetches and adds the appropriate tail of initial_state before storing final_state.
    """
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    # o_d Shape: [BD]
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        p_x = x + bos * stride_x_t
    else:
        seq_len = T
        p_x = x + i_n * stride_x_n

    p_x = tl.make_block_ptr(p_x, (seq_len, D), (stride_x_t, stride_x_d), (seq_len - BW, i_d * BD), (BW, BD), (1, 0))

    # b_x Shape: [BW, BD]
    b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    if USE_INITIAL_STATE:
        if seq_len < BW:
            o_c = W - (BW - seq_len) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)

            p_init = initial_state + i_n * D*W + o_d[None, :] * W + o_c[:, None]
            mask_init = m_d[None, :] & m_c[:, None]

            b_cache = tl.load(p_init, mask=mask_init, other=0)
            b_x += b_cache

    # final_state: [N, D, W] (Channel Major inside sample)
    # o_w Shape: [BW]
    o_w = W - BW + tl.arange(0, BW)

    # o_d[:, None] -> [BD, 1]
    # o_w[None, :] -> [1, BW]
    # p_final Shape -> [BD, BW]
    p_final = final_state + i_n * D*W + o_d[:, None] * W + o_w[None, :]

    # m_final Shape -> [BD, BW]
    m_final = m_d[:, None] & (o_w[None, :] >= 0)

    tl.store(p_final, tl.trans(b_x).to(final_state.dtype.element_ty), mask=m_final)


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the final per-sequence cache (state) of length `state_len` for a causal 1D convolution.
    
    Parameters:
        x (torch.Tensor): Input activations. Expected shapes:
            - (B, T, D) for batched fixed-length sequences,
            - (T, D) or (T, D) with implicit batch when `cu_seqlens` is provided,
            or (N variable-length packed) when using `cu_seqlens`. The last dimension is features D.
        state_len (int): Number of past time steps to include in the state (W).
        initial_state (torch.Tensor | None): Optional initial state tensor of shape (N, D, W) to be prepended to each sequence before extracting the final state.
        cu_seqlens (torch.Tensor | None): Optional 1D cumulative sequence lengths of length N+1 for variable-length packed inputs; if provided the function treats `x` as packed along time and derives per-sequence boundaries from `cu_seqlens`.
    
    Returns:
        torch.Tensor: `final_state` tensor of shape (N, D, W) containing, for each sequence, the last `state_len` frames (features ordered along D) from the sequence after applying the optional `initial_state`. The returned tensor has the same dtype and device as `x`.
    """
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if x.dim() == 2:
            stride_x_n = 0
            stride_x_t, stride_x_d = x.stride()
            T = x.shape[0]
        else:
            stride_x_n = x.stride(0)
            stride_x_t, stride_x_d = x.stride(1), x.stride(2)
            T = x.shape[1]
        D = x.shape[-1]
    else:
        B, T, D = x.shape
        N = B
        stride_x_n, stride_x_t, stride_x_d = x.stride()

    W = state_len
    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)

    BD = min(triton.next_power_of_2(D), 256)
    BW = triton.next_power_of_2(W)

    grid = (triton.cdiv(D, BD), N)

    causal_conv1d_states_fwd_kernel[grid](
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        BW=BW,
        BD=BD,
    )
    return final_state


@input_guard(no_guard_contiguous=["x"])
def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """
    Compute one update step of a stateful 1D causal convolution over a single time step and return the output and updated cache.
    
    Parameters:
        x (torch.Tensor): Input for the current step. Supported shapes:
            - (N, D)
            - (1, N, D)  (time=1, batch=N, dim=D)
            - (N, 1, D)  (batch=N, time=1, dim=D)
            If `weight` is provided and the trailing dimension does not match `weight.shape[0]`,
            `x` will be reshaped to combine trailing dimensions into a feature dimension.
        cache (torch.Tensor): Rolling cache storing past input frames required by the causal window.
        residual (torch.Tensor | None): Optional residual tensor added to the convolution output (must be broadcastable to x).
        weight (torch.Tensor | None): Optional convolution weight of shape (D, W) where W is the causal window width.
        bias (torch.Tensor | None): Optional bias added per output channel (shape compatible with D).
        activation (str | None): Optional activation identifier applied after convolution (e.g., "silu" / "swish"); None means no activation.
    
    Returns:
        tuple:
            y (torch.Tensor): Output tensor with the same shape as the (possibly original) input `x`.
            cache (torch.Tensor): The (potentially updated) cache tensor after consuming the current input step.
    """
    shape = x.shape
    if weight is not None and x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')

    D = x.shape[-1]
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else None
    BD = 8
    BW = triton.next_power_of_2(W)

    if x.dim() == 2:
        # Case: (N, D)
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(1)
    elif x.dim() == 3 and x.shape[0] == 1:
        # Case: (1, N, D) -> Time=1, Batch=N, Dim=D
        # Batch 在 dim 1
        stride_x_n = x.stride(1)
        stride_x_d = x.stride(2)
    elif x.dim() == 3:
        # Case: (N, 1, D) -> Batch=N, Time=1, Dim=D
        # Batch 在 dim 0
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(2)
    else:
        # Fallback / Error case
        raise ValueError(f"Unsupported input shape: {x.shape}")

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    if y.dim() == 2:
        stride_y_n, stride_y_d = y.stride(0), y.stride(1)
    elif y.dim() == 3 and y.shape[0] == 1:
        stride_y_n, stride_y_d = y.stride(1), y.stride(2)
    elif y.dim() == 3:
        stride_y_n, stride_y_d = y.stride(0), y.stride(2)

    def grid(meta): """
Compute the Triton kernel launch grid for the given metadata.

Parameters:
    meta (dict): Kernel metadata containing 'BD' (block depth).

Returns:
    tuple: (number of D-blocks, N) where number of D-blocks is ceil(D / meta['BD']) and N is the batch/sequence dimension.
"""
return (triton.cdiv(D, meta['BD']), N)

    causal_conv1d_update_kernel[grid](
        x=x,
        cache=cache,
        residual=residual,
        y=y,
        weight=weight,
        bias=bias,
        stride_x_n=stride_x_n,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_d=stride_y_d,
        D=D,
        W=W,
        BD=BD,
        BW=BW,
        ACTIVATION=activation,
        num_warps=STATIC_WARPS,
    )
    return y.view(shape), cache