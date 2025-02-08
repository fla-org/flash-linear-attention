# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


def naive_recurrent_rwkv7(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,  # Dynamic learning rate modulator
    b: torch.Tensor,  # State update modulator
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    state_ckpt: bool = False,
    state_ckpt_interval: int = 16
):
    """
    Naive recurrent implementation of RWKV-7 (Goose) attention mechanism.

    Args:
        q, k, v: Query, Key, and Value tensors
        w: Time decay weights
        a: Dynamic learning rate modulator, influences the in-context learning rate
        b: State update modulator, directly participates in state update calculation
        scale: Scaling factor for attention scores
        initial_state: Initial state for the recurrent computation
        output_final_state: Whether to output the final state

    Returns:
        Attention output and optionally the final state
    """
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    orig_dtype = q.dtype
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    # q, k, v, a, b, w,
    # shape: (B, H, L, D), (B, H, L, D), (B, H, T, V), (B, H, L, D), (B, H, L, D), (B, H, L, D)
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    o = torch.zeros_like(v)

    if scale == -1.0:
        scale = N ** -0.5

    if initial_state is not None:
        state += initial_state.to(dtype=torch_dtype)

    state_cache = []

    for t in range(L):
        q_t = q[:, :, t] * scale
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        a_t = a[:, :, t]
        b_t = b[:, :, t]
        if t % state_ckpt_interval == 0 and state_ckpt:
            state_cache.append(state.detach())

        # from bo's code
        sab = torch.einsum('bhik,bhk,bhj->bhij', state, a_t, b_t)
        state = state * torch.exp(-torch.exp(w[:, :, t, None, :])) + sab + torch.einsum('bhj,bhi->bhij', k_t, v_t)
        o[:, :, t] = torch.einsum('bhj,bhij->bhi', q_t, state)

    if not output_final_state:
        ht = None
    elif initial_state is not None:
        ht = state.to(initial_state.dtype)
    else:
        ht = state.to(orig_dtype)

    state_cache = torch.stack(state_cache) if state_ckpt else None
    return o.to(orig_dtype), ht, state_cache if state_ckpt else None


def naive_recurrent_rwkv7_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,  # Dynamic learning rate modulator
    b: torch.Tensor,  # State update modulator
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
):
    """
    Naive recurrent implementation of RWKV-7 (Goose) attention mechanism.

    Args:
        q, k, v: Query, Key, and Value tensors
        w: Time decay weights
        a: Dynamic learning rate modulator, influences the in-context learning rate
        b: State update modulator, directly participates in state update calculation
        scale: Scaling factor for attention scores
        initial_state: Initial state for the recurrent computation
        output_final_state: Whether to output the final state

    Returns:
        Attention output and optionally the final state
    """
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    orig_dtype = q.dtype
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    # q, k, v, a, b, w,
    # shape: (B, H, L, D), (B, H, L, D), (B, H, T, V), (B, H, L, D), (B, H, L, D), (B, H, L, D)
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    o = torch.zeros_like(v)

    if scale == -1.0:
        scale = N ** -0.5

    if initial_state is not None:
        state += initial_state.to(dtype=torch_dtype)

    for t in range(L):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t] * scale
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                w_t = torch.exp(-torch.exp(w[bi, hi, t]))

                sa = (a_t[:, None] * state[bi, hi]).sum(dim=0)

                state[bi, hi] = (state[bi, hi] * w_t[:, None] +  # [N,V] * [N,1]
                                 k_t[:, None] * v_t[None, :] +     # [N,1] * [1,V]
                                 sa * b_t[:, None])                # [V] * [N,1]

                y = (state[bi, hi] * q_t[:, None]).sum(dim=0)

                o[bi, hi, t] = y

    ht = state if output_final_state else None
    return o.to(orig_dtype), ht,  None


@torch.no_grad()
def naive_recurrent_rwkv7_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    doutput: torch.Tensor,
    dh_t: Optional[torch.Tensor] = None,
    state_cache: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    state_ckpt_interval: int = 16
):
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    q, k, v, w, a, b, doutput, state_cache = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b, doutput, state_cache))
    if dh_t is not None:
        dh_t = dh_t.to(dtype=torch_dtype)
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    # q, k, v, a, b, w,
    # shape: (B, H, L, D), (B, H, L, D), (B, H, T, V), (B, H, L, D), (B, H, L, D), (B, H, L, D)
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dw = torch.zeros_like(w)
    da = torch.zeros_like(a)
    db = torch.zeros_like(b)
    dstate = torch.zeros_like(state)

    if dh_t is not None:
        dstate += dh_t

    if scale == -1.0:
        scale = N ** -0.5

    state += state_cache[0]

    def reconstruct_state_cache(state_t, last_ckpt, length):
        states = [state_t]
        for t in range(last_ckpt, length):
            k_t, v_t, a_t, b_t, w_t = k[:, :, t], v[:, :, t], a[:, :, t], b[:, :, t], torch.exp(-torch.exp(w[:, :, t]))
            sa = torch.matmul(states[-1], a_t.unsqueeze(-1))
            sab = torch.matmul(sa, b_t.unsqueeze(-2))
            kv = torch.matmul(v_t.unsqueeze(-1), k_t.unsqueeze(-2))
            new_state = states[-1] * w_t[:, :, None, :] + sab + kv
            states.append(new_state)
        return states

    # Rebuild all states for the last chunk
    last_ckpt = (L - 1) // state_ckpt_interval * state_ckpt_interval
    states = reconstruct_state_cache(state_cache[-1], last_ckpt, L)

    for t in range(L-1, -1, -1):
        chunk_index = t // state_ckpt_interval
        local_index = t % state_ckpt_interval

        if local_index == state_ckpt_interval - 1 and t != L-1:
            # Rebuild states for the next chunk
            states = reconstruct_state_cache(state_cache[chunk_index], chunk_index, chunk_index + state_ckpt_interval)

        state = states[local_index + 1]
        prev_state = states[local_index]

        q_t = q[:, :, t] * scale
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        a_t = a[:, :, t]
        b_t = b[:, :, t]
        w_t_temp = -torch.exp(w[:, :, t])
        w_t = torch.exp(w_t_temp)

        # Gradient of output
        dq[:, :, t] += torch.matmul(doutput[:, :, t].unsqueeze(-2), state).squeeze(-2) * scale
        # torch.einsum('bhi,bhj->bhij', doutput[:, :, t], q_t)
        dstate += torch.mul(doutput[:, :, t].unsqueeze(3), q_t.unsqueeze(2))

        # Gradient of state update
        dw[:, :, t] += torch.sum(dstate * prev_state, dim=(-2)) * w_t * w_t_temp

        # Gradient of sab
        # torch.einsum('bhij,bhik,bhj->bhk', dstate, prev_state, b_t)
        # temp = torch.bmm(dstate.view(B*H, V, V).permute(0, 2, 1), prev_state.view(B*H, V, V)).view(B*H, V, V)
        temp = torch.matmul(dstate.permute(0, 1, 3, 2), prev_state)
        da[:, :, t] += torch.matmul(temp.permute(0, 1, 3, 2), b_t.unsqueeze(-1)).squeeze(-1)

        # torch.einsum('bhij,bhik,bhk->bhj', dstate, prev_state, a_t)
        db[:, :, t] += torch.matmul(temp, a_t.unsqueeze(-1)).squeeze(-1)

        # Gradient of k_t * v_t
        # torch.einsum('bhij,bhi->bhj', dstate, v_t)
        # dk[:, :, t] += torch.bmm(dstate.view(B*H, V, V).permute(0, 2, 1), v_t.view(B*H, V, 1)).view(B, H, V)
        dk[:, :, t] += torch.matmul(dstate.permute(0, 1, 3, 2), v_t.unsqueeze(-1)).squeeze(-1)
        # torch.einsum('bhij,bhj->bhi', dstate, k_t)
        # dv[:, :, t] += torch.bmm(dstate.view(B*H, V, V), k_t.view(B*H, V, 1)).view(B, H, V)
        dv[:, :, t] += torch.matmul(dstate, k_t.unsqueeze(-1)).squeeze(-1)

        # Gradient for previous state
        # torch.einsum('bhij,bhk,bhj->bhik', dstate, a_t, b_t)
        # [B, H, V, 1, V] * [B, H, 1, V, 1] = [B, H, V, 1, V]
        mul_result = (dstate.unsqueeze(3) * a_t.unsqueeze(2).unsqueeze(-1)).view(B, H, V*V, V)
        # dprev_state = torch.bmm(mul_result.view(B*H, V*V, V), b_t.view(B*H, V, 1)).view(B, H, V, V)
        dprev_state = torch.matmul(mul_result, b_t.unsqueeze(-1)).view(B, H, V, V)

        dprev_state += dstate * w_t[..., None, :]

        # Update dstate for next iteration
        dstate = dprev_state

    return dq, dk, dv, dw, da, db, dstate


class NativeRecurrentRWKV7Function(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, a, b, scale, initial_state,
                dtype: Optional[torch.dtype] = None,
                state_ckpt_interval: int = 16):
        o, ht, state_cache = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=scale, initial_state=initial_state,
                                                   state_ckpt=True, state_ckpt_interval=state_ckpt_interval)

        ctx.save_for_backward(q, k, v, w, a, b, state_cache)
        ctx.scale = scale
        ctx.dtype = dtype
        ctx.ckpt_interval = state_ckpt_interval
        ctx.use_initial_state = initial_state is not None
        return o, ht

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, w, a, b, state_cache = ctx.saved_tensors
        dq, dk, dv, dw, da, db, dh = naive_recurrent_rwkv7_bwd(
            q, k, v, w, a, b, do, dht, state_cache, ctx.scale, dtype=ctx.dtype)
        dh = dh if ctx.use_initial_state else None
        return dq, dk, dv, dw, da, db, None, dh, None, None


def native_recurrent_rwkv7(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor = None,
    log_w: torch.Tensor = None,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,

) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        w (torch.Tensor):
            decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`, kernel
            will apply log_w = -torch.exp(w)
        log_w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """
    assert cu_seqlens is None
    assert head_first is True
    assert log_w is None
    assert w is not None
    if scale == -1.0:
        scale = r.shape[-1] ** -0.5
    o, h_t = NativeRecurrentRWKV7Function.apply(r, k, v, w, a, b, scale, initial_state)
    final_state = h_t if output_final_state else None
    return o, final_state
