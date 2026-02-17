# Naive (reference) implementation of LinOSS, directly mirroring the original JAX code.
# https://github.com/tk-rusch/linoss/blob/main/models/LinOSS.py

import torch


def naive_recurrent_linoss(
    x: torch.Tensor,
    B_re: torch.Tensor,
    B_im: torch.Tensor,
    C_re: torch.Tensor,
    C_im: torch.Tensor,
    a_diag: torch.Tensor,
    dt: torch.Tensor,
    d_skip: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    discretization: str = 'IM',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Naive recurrent implementation of LinOSS.

    Args:
        x (torch.Tensor): input sequence `[B, T, H]`.
        B_re (torch.Tensor): real part of B `[P, H]`.
        B_im (torch.Tensor): imag part of B `[P, H]`.
        C_re (torch.Tensor): real part of C `[H, P]`.
        C_im (torch.Tensor): imag part of C `[H, P]`.
        a_diag (torch.Tensor): diagonal A (pre-relu) `[P]`.
        dt (torch.Tensor): discretization steps (pre-sigmoid) `[P]`.
        d_skip (torch.Tensor): skip connection `[H]`.
        initial_state (torch.Tensor | None): initial `[h1, h2]` state `[B, 2, P]`.
        output_final_state (bool): whether to return final state.
        discretization (str): 'IM' or 'IMEX'.

    Returns:
        o (torch.Tensor): output `[B, T, H]`.
        final_state (torch.Tensor | None): `[B, 2, P]` if requested.
    """
    dtype = x.dtype
    x = x.float()
    B_re, B_im = B_re.float(), B_im.float()
    C_re, C_im = C_re.float(), C_im.float()

    Bat, T, H = x.shape
    P = a_diag.shape[0]

    A = torch.relu(a_diag.float())          # relu(A_diag)
    step = torch.sigmoid(dt.float())        # sigmoid(steps)

    if discretization == 'IMEX':
        M11 = torch.ones_like(A)
        M12 = -step * A
        M21 = step.clone()
        M22 = 1.0 - step * step * A
    elif discretization == 'IM':
        schur = 1.0 / (1.0 + step * step * A)
        M11 = 1.0 - step * step * A * schur
        M12 = -step * A * schur
        M21 = step * schur
        M22 = schur
    else:
        raise ValueError(f"Unknown discretization: {discretization}")

    Bu = torch.complex(x @ B_re.t(), x @ B_im.t())

    if discretization == 'IMEX':
        F1 = Bu * step.unsqueeze(0).unsqueeze(0)
        F2 = Bu * (step * step).unsqueeze(0).unsqueeze(0)
    else:
        F1 = M11.unsqueeze(0).unsqueeze(0) * Bu * step.unsqueeze(0).unsqueeze(0)
        F2 = M21.unsqueeze(0).unsqueeze(0) * Bu * step.unsqueeze(0).unsqueeze(0)

    h1 = torch.zeros(Bat, P, dtype=torch.cfloat, device=x.device)
    h2 = torch.zeros(Bat, P, dtype=torch.cfloat, device=x.device)

    if initial_state is not None:
        h1 = initial_state[:, 0].to(torch.cfloat)
        h2 = initial_state[:, 1].to(torch.cfloat)

    C_complex = torch.complex(C_re, C_im)

    o = torch.zeros(Bat, T, H, dtype=dtype, device=x.device)
    for t in range(T):
        f1_t = F1[:, t]
        f2_t = F2[:, t]

        h1_new = M11.unsqueeze(0) * h1 + M12.unsqueeze(0) * h2 + f1_t
        h2_new = M21.unsqueeze(0) * h1 + M22.unsqueeze(0) * h2 + f2_t

        h1 = h1_new
        h2 = h2_new

        y_t = (h2 @ C_complex.t()).real + d_skip.unsqueeze(0) * x[:, t]
        o[:, t] = y_t.to(dtype)

    final_state = None
    if output_final_state:
        final_state = torch.stack([h1.real.to(dtype), h2.real.to(dtype)], dim=1)

    return o, final_state
