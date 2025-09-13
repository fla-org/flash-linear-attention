# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch


def solve_unit_lower_triangular_system(rhs: torch.Tensor, lower_with_unit_diag: torch.Tensor) -> torch.Tensor:
    """Solve L x = rhs where L is unit lower-triangular (diagonal assumed 1)."""
    return torch.linalg.solve_triangular(
        lower_with_unit_diag.float(),
        rhs.float(),
        upper=False,
        unitriangular=True
    ).to(rhs.dtype)


def solve_unit_lower_triangular_system_inplace(rhs_out: torch.Tensor, lower_with_unit_diag: torch.Tensor) -> None:
    rhs_out.copy_(solve_unit_lower_triangular_system(rhs_out, lower_with_unit_diag))


def solve_unit_upper_triangular_system(rhs: torch.Tensor, lower_with_unit_diag: torch.Tensor) -> torch.Tensor:
    """Solve U x = rhs where U = tril(lower_with_unit_diag, -1).H is unit upper-triangular."""
    return torch.linalg.solve_triangular(
        lower_with_unit_diag.tril(-1).mH.float(),
        rhs.float(),
        upper=True,
        unitriangular=True
    ).to(rhs.dtype)


def triangular_solve_backward(rhs_grad: torch.Tensor, lower_with_unit_diag: torch.Tensor, solution: torch.Tensor):
    """Backward pass helper for unit lower-triangular solve.

    Returns gradients wrt rhs (du) and strictly lower part of L (dw_lower_strict).
    """
    du = torch.linalg.solve_triangular(
        lower_with_unit_diag.tril(-1).mH.float(),
        rhs_grad.float(),
        upper=True,
        unitriangular=True
    ).to(rhs_grad.dtype)
    dw = torch.bmm(-du, solution.mH)
    dw = dw.tril(-1)
    return du, dw
