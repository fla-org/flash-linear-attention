import torch
from einops import rearrange

from fla.modules.convolution import ShortConvolution
from fla.modules.l2norm import l2norm
from fla.utils import device

def separate_conv_l2(x, conv, head_dim):
    """Separate Conv + L2 Norm"""
    y, _ = conv(x)
    y = rearrange(y, 'b t (h d) -> b t h d', d=head_dim)
    y = l2norm(y, eps=1e-5)
    y = rearrange(y, 'b t h d -> b t (h d)')
    return y

def fused_conv_l2(x, conv_fused, head_dim):
    """Fused Conv + L2 Norm"""
    y, _ = conv_fused(x, head_dim=head_dim)
    return y

if __name__ == "__main__":
    import torch.utils.benchmark as benchmark
    
    # Test configurations
    B, T, D, W = 4, 2048, 2048, 4
    H = 16
    head_dim = D // H
    
    print("="*80)
    print(f"Benchmarking Conv + L2 Norm: B={B}, T={T}, D={D}, W={W}, H={H}, head_dim={head_dim}")
    print("="*80)
    
    dtype = torch.bfloat16
    
    # Create input
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    
    # Separate Conv (no norm)
    conv_separate = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=False,
        activation='silu',
        norm=None,
        device=device,
        dtype=dtype,
    )
    
    # Fused Conv + L2 Norm
    conv_fused = ShortConvolution(
        hidden_size=D,
        kernel_size=W,
        bias=False,
        activation='silu',
        norm='l2',
        norm_eps=1e-5,
        device=device,
        dtype=dtype,
    )
    
    # Copy weights
    conv_fused.weight.data.copy_(conv_separate.weight.data)
    
    # Benchmark Forward
    print("\n" + "="*80)
    print("Forward Pass")
    print("="*80)
    
    t_sep_fwd = benchmark.Timer(
        stmt="separate_conv_l2(x, conv, head_dim)",
        globals={"separate_conv_l2": separate_conv_l2, "x": x, "conv": conv_separate, "head_dim": head_dim},
    )
    m_sep_fwd = t_sep_fwd.timeit(100)
    print(f"Separate: {m_sep_fwd}")
    
    t_fused_fwd = benchmark.Timer(
        stmt="fused_conv_l2(x, conv, head_dim)",
        globals={"fused_conv_l2": fused_conv_l2, "x": x, "conv": conv_fused, "head_dim": head_dim},
    )
    m_fused_fwd = t_fused_fwd.timeit(100)
    print(f"Fused:    {m_fused_fwd}")
    
    # Benchmark Backward
    print("\n" + "="*80)
    print("Backward Pass")
    print("="*80)
    
    # Pre-compute forward for backward benchmark
    y_sep = separate_conv_l2(x, conv_separate, head_dim)
    grad_sep = torch.randn_like(y_sep)
    
    def backward_sep():
        for xi in [x]:
            if isinstance(xi, torch.Tensor):
                xi.grad = None
        y_sep.backward(grad_sep, retain_graph=True)
    
    t_sep_bwd = benchmark.Timer(
        stmt="backward_sep()",
        globals={"backward_sep": backward_sep},
    )
    m_sep_bwd = t_sep_bwd.timeit(100)
    print(f"Separate: {m_sep_bwd}")
    
    y_fused = fused_conv_l2(x, conv_fused, head_dim)
    grad_fused = torch.randn_like(y_fused)
    
    def backward_fused():
        for xi in [x]:
            if isinstance(xi, torch.Tensor):
                xi.grad = None
        y_fused.backward(grad_fused, retain_graph=True)
    
    t_fused_bwd = benchmark.Timer(
        stmt="backward_fused()",
        globals={"backward_fused": backward_fused},
    )
    m_fused_bwd = t_fused_bwd.timeit(100)
    print(f"Fused:    {m_fused_bwd}")
    
    # Benchmark Combined
    print("\n" + "="*80)
    print("Forward + Backward Pass")
    print("="*80)
    
    def combined_sep():
        for xi in [x]:
            if isinstance(xi, torch.Tensor):
                xi.grad = None
        y = separate_conv_l2(x, conv_separate, head_dim)
        y.backward(grad_sep, retain_graph=True)
    
    t_sep_combined = benchmark.Timer(
        stmt="combined_sep()",
        globals={"combined_sep": combined_sep},
    )
    m_sep_combined = t_sep_combined.timeit(100)
    print(f"Separate: {m_sep_combined}")
    
    def combined_fused():
        for xi in [x]:
            if isinstance(xi, torch.Tensor):
                xi.grad = None
        y = fused_conv_l2(x, conv_fused, head_dim)
        y.backward(grad_fused, retain_graph=True)
    
    t_fused_combined = benchmark.Timer(
        stmt="combined_fused()",
        globals={"combined_fused": combined_fused},
    )
    m_fused_combined = t_fused_combined.timeit(100)
    print(f"Fused:    {m_fused_combined}")
    
    # Summary
    time_sep_fwd = m_sep_fwd.median * 1000
    time_sep_bwd = m_sep_bwd.median * 1000
    time_sep_combined = m_sep_combined.median * 1000
    
    time_fused_fwd = m_fused_fwd.median * 1000
    time_fused_bwd = m_fused_bwd.median * 1000
    time_fused_combined = m_fused_combined.median * 1000
    
    print(f"\n{'='*80}")
    print(f"{'Method':<35} {'Forward':<12} {'Backward':<12} {'Combined':<12} {'Speedup':<10}")
    print("-"*80)
    print(f"{'Separate (FLA)':<35} {time_sep_fwd:>10.3f}ms {time_sep_bwd:>10.3f}ms {time_sep_combined:>10.3f}ms {'1.00x':<10}")
    print(f"{'Fused (Recompute)':<35} {time_fused_fwd:>10.3f}ms {time_fused_bwd:>10.3f}ms {time_fused_combined:>10.3f}ms {time_sep_combined/time_fused_combined:<10.2f}x")
    
    speedup_fwd = (time_sep_fwd / time_fused_fwd - 1) * 100
    speedup_bwd = (time_sep_bwd / time_fused_bwd - 1) * 100
    speedup_combined = (time_sep_combined / time_fused_combined - 1) * 100
    
    print(f"\n{'='*80}")
    print(f"Forward Speedup:   {speedup_fwd:>+8.2f}%")
    print(f"Backward Speedup:  {speedup_bwd:>+8.2f}%")
    print(f"Combined Speedup:  {speedup_combined:>+8.2f}%")
    print(f"\nMemory Saved: {B*T*D*2/1024/1024:.2f} MB per Conv layer (Y_act not stored)")
    print(f"{'='*80}")
