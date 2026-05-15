"""Synthetic B200 repros for large-offset FLA kernels.

These repros use random tensors only.  They exercise shapes where flattened
offsets such as ``batch * sequence_length * heads * dim`` exceed int32 range.

Run one kernel at a time on a large Blackwell GPU, for example:

    python scripts/repro_b200_int64_offsets.py --kernel conv
    python scripts/repro_b200_int64_offsets.py --kernel wy
    python scripts/repro_b200_int64_offsets.py --kernel norm
    python scripts/repro_b200_int64_offsets.py --kernel chunk-o

On unfixed kernels, the true B=256 path can disagree with the reference/split
path or hit an illegal memory access.  With the int64-offset fixes, the reported
max difference should be zero or bf16-rounding scale, depending on the kernel.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def _check_blackwell() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device={name} capability={capability}")
    if capability[0] < 10:
        print("warning: this repro is intended for Blackwell/B200-class GPUs")


def _randn(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype)


def _rand(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.rand(shape, device="cuda", dtype=dtype)


def _report(name: str, got: torch.Tensor, ref: torch.Tensor, threshold: float = 1e-2) -> None:
    diff = (got.float() - ref.float()).abs()
    max_flat = int(diff.argmax().item())
    idx = []
    rem = max_flat
    for size in reversed(diff.shape):
        idx.append(rem % size)
        rem //= size
    idx = tuple(reversed(idx))
    print(
        f"{name}: shape={tuple(got.shape)} max_abs={diff.max().item():.6g} "
        f"mean_abs={diff.mean().item():.6g} bad_gt_{threshold:g}={(diff > threshold).sum().item()}"
    )
    print(f"  worst idx={idx} got={got.float()[idx].item():.9g} ref={ref.float()[idx].item():.9g}")


def repro_conv() -> None:
    from fla.modules.conv.causal_conv1d import causal_conv1d

    batch, seq_len, dim, width = 256, 6144, 1536, 4
    x = _randn((batch, seq_len, dim))
    weight = _randn((dim, width))

    ref = F.conv1d(
        x.transpose(1, 2).float(),
        weight[:, None, :].float(),
        padding=width - 1,
        groups=dim,
    )[..., :seq_len].transpose(1, 2)
    ref = F.silu(ref)

    got, _ = causal_conv1d(x=x, weight=weight, activation="silu", backend="triton")
    torch.cuda.synchronize()
    _report("causal_conv1d_fwd_kernel", got, ref.to(got.dtype))


def _wy_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    block_t = a.shape[-1]
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    for start in range(0, seq_len, block_t):
        end = min(start + block_t, seq_len)
        length = end - start
        a_block = a[:, start:end].float()
        b_block = beta[:, start:end].float()
        vb = torch.zeros((batch, block_t, heads, value_dim), device="cuda", dtype=torch.float32)
        vb[:, :length] = v[:, start:end].float() * b_block[..., None]
        u[:, start:end] = torch.einsum("blhm,bmhv->blhv", a_block, vb).to(u.dtype)

        kb = torch.zeros((batch, block_t, heads, key_dim), device="cuda", dtype=torch.float32)
        kb[:, :length] = k[:, start:end].float() * (b_block * torch.exp2(g[:, start:end].float()))[..., None]
        w[:, start:end] = torch.einsum("blhm,bmhk->blhk", a_block, kb).to(w.dtype)
    return w, u


def repro_wy() -> None:
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd

    batch, seq_len, heads, key_dim, value_dim, block_t = 256, 6144, 12, 64, 128, 64
    k = _randn((batch, seq_len, heads, key_dim))
    v = _randn((batch, seq_len, heads, value_dim))
    beta = _rand((batch, seq_len, heads))
    a = _randn((batch, seq_len, heads, block_t))
    g = _randn((batch, seq_len, heads))
    offsets = torch.arange(seq_len, device="cuda") % block_t
    cols = torch.arange(block_t, device="cuda")
    a.masked_fill_(offsets[None, :, None, None] < cols[None, None, None, :], 0)

    ref_w, ref_u = _wy_reference(k, v, beta, a, g)
    got_w, got_u = recompute_w_u_fwd(k, v, beta, a, g)
    torch.cuda.synchronize()
    _report("recompute_w_u_fwd_kernel/w", got_w, ref_w)
    _report("recompute_w_u_fwd_kernel/u", got_u, ref_u)


def repro_norm() -> None:
    from fla.modules.fused_norm_gate import rms_norm_gated

    batch, seq_len, heads, dim = 256, 6144, 12, 128
    x = _randn((batch, seq_len, heads, dim))
    gate = _randn((batch, seq_len, heads, dim))
    weight = torch.ones((dim,), device="cuda", dtype=torch.bfloat16)

    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref = ref * weight.float() * F.silu(gate.float())
    got = rms_norm_gated(x, gate, weight, None, eps=1e-6)
    torch.cuda.synchronize()
    _report("layer_norm_gated_fwd_kernel", got, ref.to(got.dtype))


def repro_chunk_o() -> None:
    from fla.ops.common.chunk_o import chunk_fwd_o

    batch, seq_len, heads, key_dim, value_dim, block_t = 256, 6144, 12, 64, 128, 64
    num_chunks = (seq_len + block_t - 1) // block_t
    q = _randn((batch, seq_len, heads, key_dim))
    k = _randn((batch, seq_len, heads, key_dim))
    v = _randn((batch, seq_len, heads, value_dim))
    h = _randn((batch, num_chunks, heads, key_dim, value_dim))
    g = _randn((batch, seq_len, heads))

    got = chunk_fwd_o(q, k, v, h, g=g, chunk_size=block_t)
    ref = torch.cat(
        [
            chunk_fwd_o(
                q[start : start + 128],
                k[start : start + 128],
                v[start : start + 128],
                h[start : start + 128],
                g=g[start : start + 128],
                chunk_size=block_t,
            )
            for start in range(0, batch, 128)
        ],
        dim=0,
    )
    torch.cuda.synchronize()
    _report("chunk_fwd_kernel_o true-B vs split-B", got, ref)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["conv", "wy", "norm", "chunk-o"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    _check_blackwell()
    if args.kernel == "conv":
        repro_conv()
    elif args.kernel == "wy":
        repro_wy()
    elif args.kernel == "norm":
        repro_norm()
    elif args.kernel == "chunk-o":
        repro_chunk_o()


if __name__ == "__main__":
    main()
