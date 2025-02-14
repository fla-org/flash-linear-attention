# -*- coding: utf-8 -*-


import torch
import triton
import triton.language as tl
from fla.utils import contiguous


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit
def swiglu_fwd_kernel(
    x,
    y,
    o,
    T: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr
):
    # float(x) * float(y) / (1.0f + ::exp(-float(x)))
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_y = tl.load(y + o_i, mask=m_i, other=0.).to(tl.float32)
    b_o = b_x * b_y / (1. + tl.exp(-b_x))
    tl.store(o + o_i, b_o.to(o.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit
def swiglu_bwd_kernel(
    x,
    y,
    g,
    dx,
    dy,
    T: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr
):
    # float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    # dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    # dy = float(x) * x_sigmoid * float(g);
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_y = tl.load(y + o_i, mask=m_i, other=0.).to(tl.float32)
    b_g = tl.load(g + o_i, mask=m_i, other=0.).to(tl.float32)
    b_x_sigmoid = 1. / (1. + tl.exp(-b_x))
    b_dx = b_x_sigmoid * (1. + b_x * (1. - b_x_sigmoid)) * b_g * b_y
    b_dy = b_x * b_x_sigmoid * b_g
    tl.store(dx + o_i, b_dx.to(dx.dtype.element_ty), mask=m_i)
    tl.store(dy + o_i, b_dy.to(dy.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit
def swiglu_fwdbwd_kernel(
    x,
    y,
    g,
    z,
    dx,
    dy,
    T: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr
):
    # float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    # float x_swish = float(x) * x_sigmoid;
    # dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    # dy = x_swish * float(g);
    # z = x_swish * float(y);
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_y = tl.load(y + o_i, mask=m_i, other=0.).to(tl.float32)
    b_g = tl.load(g + o_i, mask=m_i, other=0.).to(tl.float32)
    b_x_sigmoid = 1. / (1. + tl.exp(-b_x))
    b_x_swish = b_x * b_x_sigmoid
    b_dx = b_x_sigmoid * (1. + b_x * (1. - b_x_sigmoid)) * b_g * b_y
    b_dy = b_x_swish * b_g
    b_z = b_x_swish * b_y
    tl.store(dx + o_i, b_dx.to(dx.dtype.element_ty), mask=m_i)
    tl.store(dy + o_i, b_dy.to(dy.dtype.element_ty), mask=m_i)
    tl.store(z + o_i, b_z.to(z.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit
def logsigmoid_fwd_kernel(
    x,
    y,
    temperature,
    T: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_m = tl.minimum(0., b_x)
    b_z = 1. + tl.exp(-tl.abs(b_x))
    b_y = (b_m - tl.log(b_z)) / temperature
    tl.store(y + o_i, b_y.to(y.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit
def logsigmoid_bwd_kernel(
    x,
    dx,
    dy,
    temperature,
    T: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_dy = tl.load(dy + o_i, mask=m_i, other=0.).to(tl.float32)
    b_dx = b_dy * (1. - tl.sigmoid(b_x)) / temperature
    tl.store(dx + o_i, b_dx.to(dx.dtype.element_ty), mask=m_i)


@contiguous
def swiglu_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(
        T, triton.runtime.driver.active.utils.get_device_properties(x.device.index)['multiprocessor_count']))
    o = torch.empty_like(x)
    swiglu_fwd_kernel[(triton.cdiv(T, B),)](
        x=x,
        y=y,
        o=o,
        T=T,
        D=D,
        B=B
    )
    return o


@contiguous
def swiglu_bwd(x: torch.Tensor, y: torch.Tensor, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(
        T, triton.runtime.driver.active.utils.get_device_properties(x.device.index)['multiprocessor_count']))
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    swiglu_bwd_kernel[(triton.cdiv(T, B),)](
        x=x,
        y=y,
        g=dout,
        dx=dx,
        dy=dy,
        T=T,
        D=D,
        B=B
    )
    return dx, dy


@contiguous
def swiglu_fwdbwd(
    x: torch.Tensor,
    y: torch.Tensor,
    dout: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(
        T, triton.runtime.driver.active.utils.get_device_properties(x.device.index)['multiprocessor_count']))
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    z = torch.empty_like(x)
    swiglu_fwdbwd_kernel[(triton.cdiv(T, B),)](
        x=x,
        y=y,
        g=dout,
        z=z,
        dx=dx,
        dy=dy,
        T=T,
        D=D,
        B=B
    )
    return dx, dy, z


def logsigmoid_fwd(x: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(
        T, triton.runtime.driver.active.utils.get_device_properties(x.device.index)['multiprocessor_count']))
    y = torch.empty_like(x)
    logsigmoid_fwd_kernel[(triton.cdiv(T, B),)](
        x=x,
        y=y,
        temperature=temperature,
        T=T,
        D=D,
        B=B
    )
    return y


def logsigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    B = triton.next_power_of_2(triton.cdiv(
        T, triton.runtime.driver.active.utils.get_device_properties(x.device.index)['multiprocessor_count']))
    dx = torch.empty_like(x)
    logsigmoid_bwd_kernel[(triton.cdiv(T, B),)](
        x=x,
        dx=dx,
        dy=dy,
        temperature=temperature,
        T=T,
        D=D,
        B=B
    )
    return dx
