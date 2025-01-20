from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.generalized_delta_rule.iplr.wy_fast import fwd_prepare_wy_repr
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_generalized_iplr_delta_rule_fwd_kernel_h(
    k,
    v,
    d,
    b,
    u,
    v_new,
    h,
    h0,
    ht,
    offsets,
    chunk_offsets,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        if HEAD_FIRST:
            p_h = tl.make_block_ptr(h + (i_nh * NT + i_t) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        else:
            p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        # since we need to make all DK in the SRAM. we face serve SRAM memory burden. By subchunking we allievate such burden
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            if HEAD_FIRST:
                p_k = tl.make_block_ptr(k + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_b = tl.make_block_ptr(b + i_nh * T*K, (K, T), (1, K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d + i_nh * T*K, (T, K), (K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_u = tl.make_block_ptr(u + i_nh * T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+i_nh*T*V, (T, V), (V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            else:
                p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_b = tl.make_block_ptr(b+(bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
                p_d = tl.make_block_ptr(d+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
                p_v = tl.make_block_ptr(v+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_u = tl.make_block_ptr(u+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
                p_v_new = tl.make_block_ptr(v_new+(bos*H+i_h)*V, (T, V), (H*V, 1), (i_t*BT+i_c*BC, i_v * BV), (BC, BV), (1, 0))
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_b = tl.load(p_b, boundary_check=(0, 1))
            b_v2 = tl.dot(b_d, b_h.to(b_d.dtype)) + tl.load(p_u, boundary_check=(0, 1))
            b_hc += tl.dot(b_k, b_v)
            b_hc += tl.dot(b_b, b_v2.to(b_k.dtype))
            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))
        b_h += b_hc

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64, 128]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=["BT"],
)
@triton.jit
def chunk_generalized_iplr_delta_rule_fwd_kernel_o(
    q,
    k,
    v,
    u,
    b,
    h,
    o,
    offsets,
    indices,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_tg = i_t
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (i_bh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    k += (i_bh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    b += (i_bh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    v += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    u += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    o += (i_bh * T * V) if HEAD_FIRST else ((bos * H + i_h) * V)
    h += ((i_bh * NT + i_t) * K * V) if HEAD_FIRST else ((i_tg * H + i_h) * K * V)
    stride_qk = K if HEAD_FIRST else H*K
    stride_vo = V if HEAD_FIRST else H*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aqb = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_b = tl.make_block_ptr(b, (K, T), (1, stride_qk), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_Aqk += tl.dot(b_q, b_k)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_Aqb += tl.dot(b_q, b_b)

    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_Aqk = tl.where(m_A, b_Aqk, 0)
    b_Aqb = tl.where(m_A, b_Aqb, 0)

    p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_u = tl.make_block_ptr(u, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_u = tl.load(p_u, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_Aqk.to(b_v.dtype), b_v) + tl.dot(b_Aqb.to(b_u.dtype), b_u)) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_generalized_iplr_delta_rule_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    b: torch.Tensor,
    h: torch.Tensor,
    scale: Optional[float] = None,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> torch.Tensor:
    if head_first:
        B, H, T, K, V = *q.shape, v.shape[-1]
    else:
        B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = k.shape[-1] ** -0.5
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)
        NT = len(indices)

    o = torch.empty_like(v)

    def grid(meta): return (
        triton.cdiv(V, meta['BV']),
        NT,
        B * H
    )
    chunk_generalized_iplr_delta_rule_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        u=v_new,
        b=b,
        h=h,
        o=o,
        offsets=offsets,
        indices=indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        HEAD_FIRST=head_first
    )
    return o


def chunk_generalized_iplr_delta_rule_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    b: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, u.shape[-1]
    else:
        B, T, H, K, V = *k.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        if chunk_offsets is None:
            chunk_offsets = torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], BT)]).cumsum(-1)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64 if K <= 128 else 32
    else:
        BV = 32
        BC = 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    if head_first:
        h = k.new_empty(B, H, NT, K, V)
    else:
        h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)

    chunk_generalized_iplr_delta_rule_fwd_kernel_h[grid](
        k=k,
        v=v,
        d=w,
        b=b,
        u=u,
        v_new=v_new,
        h=h,
        h0=initial_state,
        ht=final_state,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first
    )
    return h, v_new, final_state


def chunk_generalized_iplr_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u, _ = fwd_prepare_wy_repr(
        a=a,
        b=b,
        k=k,
        v=v,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )

    h, v_new, final_state = chunk_generalized_iplr_delta_rule_fwd_h(
        k=k,
        v=v,
        b=b,
        w=w,
        u=u,
        initial_state=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT
    )
    o = chunk_generalized_iplr_delta_rule_fwd_o(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        b=b,
        h=h,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    return o, final_state


class ChunkGeneralizedIPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True
    ):
        chunk_size = 64

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = None
        if offsets is not None:
            indices = torch.cat([torch.arange(n) for n in triton.cdiv(offsets[1:] - offsets[:-1], chunk_size).tolist()])
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(offsets)

        o, final_state = chunk_generalized_iplr_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        raise NotImplementedError(
            "Backward pass for ChunkGeneralizedIPLRDeltaRuleFunction is not implemented yet. "
            "Stay tuned!"
        )


def chunk_iplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            activations of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            betas of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkDeltaRuleFunction does not support float32. Please use bfloat16."

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkGeneralizedIPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first
    )
    return o, final_state
