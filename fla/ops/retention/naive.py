# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def naive_retention(q, k, v):
    orig_type = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    _, n_heads, seq_len, d_head = q.shape
    s = (1 - q.new_tensor(2., dtype=torch.float).pow(-5. - q.new_tensor(range(n_heads), dtype=torch.float))).log2()
    n = q.new_tensor(range(seq_len), dtype=torch.float)
    n = torch.exp2((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)
    s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
    o = torch.einsum('bhqk,bhkd->bhqd', s, v)
    return o.to(orig_type)
