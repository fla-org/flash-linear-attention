
import torch
import triton
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra

def benchmark_intra_chunk(B=8, T=4096, H=16, K=128, chunk_size=64):
    dtype = torch.bfloat16
    device = 'cuda'
    
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    g = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    beta = torch.randn(B, T, H, device=device, dtype=dtype)
    
    scale = 1.0
    
    quantiles = [0.5, 0.2, 0.8]
    
    # Warmup
    for _ in range(10):
        chunk_kda_fwd_intra(q, k, g, beta, scale=scale, chunk_size=chunk_size, impl_type="token")
        chunk_kda_fwd_intra(q, k, g, beta, scale=scale, chunk_size=chunk_size, impl_type="recursive")
        chunk_kda_fwd_intra(q, k, g, beta, scale=scale, chunk_size=chunk_size, impl_type="recurrent")
        
    ms_token = triton.testing.do_bench(
        lambda: chunk_kda_fwd_intra(q, k, g, beta, scale=scale, chunk_size=chunk_size, impl_type="token"),
        quantiles=quantiles
    )
    
    ms_recursive = triton.testing.do_bench(
        lambda: chunk_kda_fwd_intra(q, k, g, beta, scale=scale, chunk_size=chunk_size, impl_type="recursive"),
        quantiles=quantiles
    )

    try:
        ms_recurrent = triton.testing.do_bench(
            lambda: chunk_kda_fwd_intra(q, k, g, beta, scale=scale, chunk_size=chunk_size, impl_type="recurrent"),
            quantiles=quantiles
        )
        t_recurrent = ms_recurrent[0]
    except Exception as e:
        t_recurrent = float('nan')
    
    # Format for table row
    # Shape | Token | Recursive | Recurrent | Rec vs Token
    row_str = f"B={B}, T={T}, H={H}, K={K}"
    print(f"{row_str:<30} | {ms_token[0]:.3f} ms            | {ms_recursive[0]:.3f} ms            | {t_recurrent:.3f} ms        | {ms_token[0]/ms_recursive[0]:.2f}x       ")

if __name__ == "__main__":
    configs = [
        (8, 4096, 16, 128),
        (1, 8192, 16, 128),
        (8, 4096, 32, 64),
        (1, 8192, 32, 64),
        # Large Batch
        (32, 512, 12, 64),
        # High Head Dim
        (2, 4096, 8, 256),
    ]
    
    print(f"{'Shape':<30} | {'Token (Original)':<20} | {'Recursive (New)':<20} | {'Recurrent':<15} | {'Speedup (Rec/Tok)':<15}")
    print("-" * 110)
    
    for B, T, H, K in configs:
        try:
            benchmark_intra_chunk(B=B, T=T, H=H, K=K, chunk_size=64)
        except Exception as e:
            print(f"Failed for shape B={B}, T={T}, H={H}, K={K}: {e}")
