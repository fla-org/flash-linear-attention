# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# REVISED FROM
# https://github.com/shawntan/stickbreaking-attention/blob/main/stickbreaking_attention/sb_varlen/softplus.py

import triton
from triton import language as tl

from fla.utils import IS_NVIDIA


def _generate_softplus(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 20.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p mul.f32            ${out_reg}, ${in_reg}, 1.4426950408889634;
        @!p ex2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p mul.f32            ${out_reg}, ${out_reg}, 0.6931471805599453;
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_softplus2(num_pack):
    # 1 + exp2(x) rounds to 1 for x < -24, so use exp2(x) * log2(e) directly in the tail.
    template = """
        .reg .f32 t, e;
        .reg .pred p, q;
        ex2.approx.ftz.f32 e, ${in_reg};
        mov.f32 t, e;
        add.f32 t, t, 1.0;
        lg2.approx.ftz.f32 t, t;
        mul.f32 e, e, 1.4426950408889634;
        setp.lt.f32 q, ${in_reg}, -16.;
        selp.f32 t, e, t, q;
        setp.gt.f32 p, ${in_reg}, 15.;
        selp.f32 ${out_reg}, ${in_reg}, t, p;
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_constraints(num_pack):
    return ",".join("=r" for i in range(num_pack)) + "," + ",".join("r" for i in range(num_pack))


_NUM_REG = 1
s_softplus: tl.constexpr = tl.constexpr(_generate_softplus(_NUM_REG))
s_softplus2: tl.constexpr = tl.constexpr(_generate_softplus2(_NUM_REG))
s_constraints: tl.constexpr = tl.constexpr(_generate_constraints(_NUM_REG))
NUM_REG: tl.constexpr = tl.constexpr(_NUM_REG)


@triton.jit
def softplus_nv(x):
    # equivalent to:
    # return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)
    return tl.inline_asm_elementwise(
        asm=s_softplus,
        constraints=s_constraints,
        pack=NUM_REG,
        args=[
            x,
        ],
        dtype=tl.float32,
        is_pure=True,
    )


@triton.jit
def softplus_triton(x):
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)


@triton.jit
def softplus2_nv(x):
    # equivalent to:
    # return tl.where(x < -16.0, tl.math.exp2(x) * 1.4426950408889634,
    #                 tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x))
    return tl.inline_asm_elementwise(
        asm=s_softplus2,
        constraints=s_constraints,
        pack=NUM_REG,
        args=[
            x,
        ],
        dtype=tl.float32,
        is_pure=True,
    )


@triton.jit
def softplus2_triton(x):
    # adding exp2(x) to 1 loses precision in the negative tail; use exp2(x)
    # scaled by log2(e) directly there, where log2(1 + exp2(x)) is close to it
    return tl.where(
        x < -16.0,
        tl.math.exp2(x) * 1.4426950408889634,
        tl.where(x < 15.0, tl.math.log2(1.0 + tl.math.exp2(x)), x),
    )


# named wrappers preserve the symbols referenced by kernels in Inductor's generated source
if IS_NVIDIA:
    @triton.jit
    def softplus(x):
        return softplus_nv(x)

    @triton.jit
    def softplus2(x):
        return softplus2_nv(x)
else:
    @triton.jit
    def softplus(x):
        return softplus_triton(x)

    @triton.jit
    def softplus2(x):
        return softplus2_triton(x)
