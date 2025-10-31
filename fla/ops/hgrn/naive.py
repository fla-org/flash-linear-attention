

import torch


def naive_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
) -> torch.Tensor:
    dtype = x.dtype
    x, g = map(lambda i: i.float(), (x, g))
    B, T, D = x.shape

    h = torch.zeros(B, D, dtype=torch.float, device=x.device)
    o = torch.zeros_like(x)

    final_state = None
    if initial_state is not None:
        h += initial_state

    for i in range(T):
        h = g[:, i].exp() * h + x[:, i]
        o[:, i] = h

    if output_final_state:
        final_state = h
    return o.to(dtype), final_state


def naive_chunk_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    chunk_size: int = 64,
) -> torch.Tensor:
    dtype = x.dtype
    x, g = map(lambda i: i.float(), (x, g))
    B, T, D = x.shape

    gc = g.view(B, chunk_size, D).cumsum(-2).view_as(g)
    h = torch.zeros(B, D, dtype=torch.float, device=x.device)
    o = torch.zeros_like(x)

    final_state = None
    if initial_state is not None:
        h += initial_state

    for i in range(0, T, chunk_size):
        hp = h
        h = torch.zeros(B, D, dtype=torch.float, device=x.device)
        for j in range(i, i + chunk_size):
            h = g[:, j].exp() * h + x[:, j]
            o[:, j] = hp * gc[:, j].exp() + h
        h = o[:, j].clone()

    if output_final_state:
        final_state = h
    return o.to(dtype), final_state
