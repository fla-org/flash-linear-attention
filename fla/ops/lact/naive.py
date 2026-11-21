# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import torch.nn.functional as F

from fla.ops.lact.chunk import zeropower_via_newtonschulz5


def naive_lact(
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
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    """
    Reference for `chunk_lact` that loops over sequences, heads and chunks in fp32 and obtains each test-time training
    step from autograd on the inner loss.
    """
    dtype = q.dtype
    B, T, H, _ = q.shape
    q, k, v, lr = (x.float() for x in (q, k, v, lr))

    def f(ws, x):
        return F.linear(F.silu(F.linear(x, ws[0])) * F.linear(x, ws[2]), ws[1])

    o, states = [], []
    for b in range(B):
        o_b, states_b = [], []
        for h in range(H):
            base = [w[h].float() for w in (w0, w1, w2)]
            norms = [w.norm(dim=-1, keepdim=True) for w in base]
            if initial_state is None:
                main = base
                bufs = [torch.zeros_like(w) for w in base]
            else:
                main = [s[b, h].float() for s in initial_state[:3]]
                bufs = [s[b, h].float() for s in initial_state[3:]]
            fast = main if initial_state is None or not prenorm else [
                w / (w.norm(dim=-1, keepdim=True) + 1e-5) * n for w, n in zip(main, norms)
            ]
            o_h = []
            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                o_h.append(f(fast, q[b, s:e, h]))
                if e - s < chunk_size:
                    continue
                with torch.enable_grad():
                    ws = [w if w.requires_grad else w.detach().requires_grad_() for w in fast]
                    score = (f(ws, k[b, s:e, h]) * v[b, s:e, h]).sum(-1)
                    dws = [
                        torch.autograd.grad((lr[b, s:e, h, i] * score).sum(), ws[i], create_graph=True)[0]
                        for i in range(3)
                    ]
                if momentum is not None:
                    m = momentum[b, s:e, h].float().mean()
                    dws = [dw + buf * m for dw, buf in zip(dws, bufs)]
                    bufs = dws
                if use_muon:
                    dws = [zeropower_via_newtonschulz5(dw[None])[0] for dw in dws]
                main = [w + dw for w, dw in zip(main, dws)]
                fast = [w / (w.norm(dim=-1, keepdim=True) + 1e-5) * n for w, n in zip(main, norms)]
                if not prenorm:
                    main = fast
            o_b.append(torch.cat(o_h, 0))
            states_b.append(main + (bufs if momentum is not None else []))
        o.append(torch.stack(o_b, 1))
        states.append([torch.stack(s, 0) for s in zip(*states_b)])
    final_state = tuple(torch.stack(s, 0) for s in zip(*states)) if output_final_state else None
    return torch.stack(o, 0).to(dtype), final_state
