# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def forward(u, w):
    return torch.linalg.solve_triangular(
        w.float(),
        u.float(),
        upper=False,
        unitriangular=True,
    ).to(u.dtype)


def forward_inplace(u, w):
    u.copy_(forward(u, w))


def backward_x(do, w):
    return torch.linalg.solve_triangular(
        w.tril(-1).mH.float(),
        do.float(),
        upper=True,
        unitriangular=True,
    ).to(do.dtype)


def backward(do, w, x):
    du = torch.linalg.solve_triangular(
        w.tril(-1).mH.float(),
        do.float(),
        upper=True,
        unitriangular=True,
    ).to(do.dtype)
    dw = torch.bmm(-du, x.mH)
    dw = dw.tril(-1)
    return du, dw
