# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Adapted from https://github.com/a1600012888/LaCT/blob/main/lact_llm/lact_model/ttt_operation.py

import torch
import torch.nn.functional as F

NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
    """
    Approximately orthogonalize the trailing two dims of `G` with the quintic Newton-Schulz iteration used by Muon.
    """
    X = G
    transpose = G.shape[-2] > G.shape[-1]
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in NS_COEFFS:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if transpose:
        X = X.mT
    return X


def renorm(w: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    return w / (w.norm(dim=-1, keepdim=True) + 1e-5) * norm


def swiglu_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    # [N, L, K] -> [N, L, V]
    return (F.silu(x @ w0.mT) * (x @ w2.mT)) @ w1.mT


def lact_update(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Negative gradients of `sum_t lr[t, i] * <f(k_t), v_t>` w.r.t. the fast weights `w_i` of `f(x) = w1 (silu(w0 x) * (w2 x))`.
    The products run in the dtype of `k`; the results are returned in fp32.
    """
    dtype = k.dtype
    w0, w1, w2 = w0.to(dtype), w1.to(dtype), w2.to(dtype)
    # [N, C, D]
    gate = k @ w0.mT
    up = k @ w2.mT
    act = F.silu(gate)
    dh = v @ w1
    sig = torch.sigmoid(gate)
    dgate = dh * up * sig * (1 + gate * (1 - sig))
    dup = dh * act
    dw0 = dgate.mT @ (k * lr[..., 0:1]).to(dtype)
    dw1 = v.mT @ (act * up * lr[..., 1:2]).to(dtype)
    dw2 = dup.mT @ (k * lr[..., 2:3]).to(dtype)
    return dw0.float(), dw1.float(), dw2.float()


def chunk_lact_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    lr: torch.Tensor,
    momentum: torch.Tensor | None,
    chunk_size: int,
    use_muon: bool,
    prenorm: bool,
    initial_state: tuple[torch.Tensor, ...] | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    # q/k: [N, T, K], v: [N, T, V], w0/w2: [N, D, K], w1: [N, V, D], lr: [N, T, 3], momentum: [N, T]
    T = q.shape[1]
    norms = [w.float().norm(dim=-1, keepdim=True) for w in (w0, w1, w2)]
    if initial_state is None:
        main = [w.float() for w in (w0, w1, w2)]
        fast = main
        bufs = [torch.zeros_like(w) for w in main] if momentum is not None else None
    else:
        main = [s.float() for s in initial_state[:3]]
        fast = [renorm(w, n) for w, n in zip(main, norms)] if prenorm else main
        bufs = [s.float() for s in initial_state[3:]] if momentum is not None else None

    o = []
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        o.append(swiglu_apply(*(w.to(q.dtype) for w in fast), q[:, s:e]))
        # a trailing complete chunk only changes the final state
        if e - s < chunk_size or (e == T and not output_final_state):
            continue
        dws = lact_update(*fast, k[:, s:e], v[:, s:e], lr[:, s:e].float())
        if momentum is not None:
            m = momentum[:, s:e].float().mean(1)[:, None, None]
            dws = [dw + buf * m for dw, buf in zip(dws, bufs)]
            bufs = dws
        if use_muon:
            dws = [zeropower_via_newtonschulz5(dw.to(q.dtype)).float() for dw in dws]
        main = [w + dw for w, dw in zip(main, dws)]
        fast = [renorm(w, n) for w, n in zip(main, norms)]
        if not prenorm:
            main = fast
    o = torch.cat(o, 1)

    final_state = None
    if output_final_state:
        final_state = tuple(main) + (tuple(bufs) if momentum is not None else ())
    return o, final_state


def chunk_lact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    lr: torch.Tensor,
    momentum: torch.Tensor | None = None,
    chunk_size: int = 2048,
    use_muon: bool = False,
    prenorm: bool = False,
    initial_state: tuple[torch.Tensor, ...] | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    r"""
    Large-chunk test-time training (LaCT) with a SwiGLU fast-weight network, from https://arxiv.org/abs/2505.23884.

    The fast weights define `f(x) = w1 (silu(w0 x) * (w2 x))` per head. Each chunk of `chunk_size` tokens first reads
    `o_t = f(q_t)` with the current fast weights, then takes one gradient step on `-sum_t lr_t <f(k_t), v_t>` over the
    chunk, so the outputs of a chunk only depend on earlier chunks. After every step, each row of the fast weights is
    rescaled to the row norm of the given `w0`, `w1` and `w2`.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        w0 (torch.Tensor):
            Initial gate projection of shape `[H, D, K]`, where `D` is the hidden size of the fast-weight network.
        w1 (torch.Tensor):
            Initial down projection of shape `[H, V, D]`.
        w2 (torch.Tensor):
            Initial up projection of shape `[H, D, K]`.
        lr (torch.Tensor):
            Per-token learning rates for `w0`, `w1` and `w2`, of shape `[B, T, H, 3]`.
        momentum (torch.Tensor, Optional):
            Per-token momentum coefficients of shape `[B, T, H]`, averaged within each chunk. Default: `None`.
        chunk_size (int, Optional):
            Number of tokens per test-time training step. Default: 2048.
        use_muon (bool, Optional):
            Whether to orthogonalize each update with Newton-Schulz iterations. Default: `False`.
        prenorm (bool, Optional):
            Whether to accumulate updates on unnormalized weights and only renormalize the weights used for reading.
            Default: `False`.
        initial_state (tuple[torch.Tensor, ...], Optional):
            Fast weights `(w0, w1, w2)` of shape `[N, H, ...]` to start from, followed by their momentum buffers when
            `momentum` is given, where `N` is the number of sequences. The row norms are still taken from `w0`, `w1`
            and `w2`. Default: `None`.
        output_final_state (bool, Optional):
            Whether to return the state after the last complete chunk, in the format of `initial_state`. Tokens of a
            trailing incomplete chunk are read but do not update the state. Default: `False`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length inputs with `B = 1`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (tuple[torch.Tensor, ...]):
            Final state in fp32 if `output_final_state=True` else `None`.
    """
    B, T, H, _ = q.shape
    if cu_seqlens is not None and B != 1:
        raise ValueError(f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`.")
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if initial_state is not None and initial_state[0].shape[0] != N:
        raise ValueError(f"The number of initial states is expected to be {N} rather than {initial_state[0].shape[0]}.")
    expected = 3 if momentum is None else 6
    if initial_state is not None and len(initial_state) != expected:
        raise ValueError(f"`initial_state` is expected to hold {expected} tensors rather than {len(initial_state)}.")

    def heads_first(x):
        return x.transpose(1, 2).flatten(0, 1) if x is not None else None

    def run(q, k, v, lr, momentum, n, state):
        # [n, T, H, ...] -> [n * H, T, ...]
        state = tuple(s.flatten(0, 1) for s in state) if state is not None else None
        o, ht = chunk_lact_fwd(
            *map(heads_first, (q, k, v)),
            *(w.repeat(n, 1, 1) for w in (w0, w1, w2)),
            lr=heads_first(lr),
            momentum=heads_first(momentum),
            chunk_size=chunk_size,
            use_muon=use_muon,
            prenorm=prenorm,
            initial_state=state,
            output_final_state=output_final_state,
        )
        o = o.unflatten(0, (n, H)).transpose(1, 2)
        ht = tuple(s.unflatten(0, (n, H)) for s in ht) if ht is not None else None
        return o, ht

    if cu_seqlens is None:
        return run(q, k, v, lr, momentum, B, initial_state)

    o, states = [], []
    for i, (bos, eos) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())):
        o_i, ht_i = run(
            q[:, bos:eos],
            k[:, bos:eos],
            v[:, bos:eos],
            lr[:, bos:eos],
            momentum[:, bos:eos] if momentum is not None else None,
            1,
            tuple(s[i:i+1] for s in initial_state) if initial_state is not None else None,
        )
        o.append(o_i)
        states.append(ht_i)
    o = torch.cat(o, 1)
    final_state = tuple(torch.cat(s, 0) for s in zip(*states)) if output_final_state else None
    return o, final_state
