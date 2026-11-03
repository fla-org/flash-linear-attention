# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl


@triton.jit
def gka_dot(SMALL_SMEM: tl.constexpr, a, b):
    # smaller output tiles bound the shared-memory staging for MMA operands
    if SMALL_SMEM:
        M: tl.constexpr = a.shape[0]
        K: tl.constexpr = a.shape[1]
        N: tl.constexpr = b.shape[1]
        if M >= 64:
            a0, a1 = tl.split(a.reshape(2, M // 2, K).permute(1, 2, 0))
            if N >= 64:
                b0, b1 = tl.split(b.reshape(K, 2, N // 2).permute(0, 2, 1))
                c0 = tl.join(tl.dot(a0, b0), tl.dot(a0, b1)).permute(0, 2, 1).reshape(M // 2, N)
                c1 = tl.join(tl.dot(a1, b0), tl.dot(a1, b1)).permute(0, 2, 1).reshape(M // 2, N)
            else:
                c0, c1 = tl.dot(a0, b), tl.dot(a1, b)
            c = tl.join(c0, c1).permute(2, 0, 1).reshape(M, N)
        elif N >= 64:
            b0, b1 = tl.split(b.reshape(K, 2, N // 2).permute(0, 2, 1))
            c = tl.join(tl.dot(a, b0), tl.dot(a, b1)).permute(0, 2, 1).reshape(M, N)
        else:
            c = tl.dot(a, b)
    else:
        c = tl.dot(a, b)
    return c
