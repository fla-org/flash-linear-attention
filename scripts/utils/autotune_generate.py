import os
from pathlib import Path


def get_triton_cache_dir(path: str | None = None) -> Path:
    """Get Triton's cache directory via Triton's internal API."""
    if path is not None:
        return Path(path)
    from triton.runtime.cache import knobs
    return Path(knobs.cache.dir)


class FLACacheGenerator:
    """Runs FLA kernels to populate a Triton autotune cache for a given head_dim."""

    def __init__(self, head_dim: int, *, device):
        self.head_dim = head_dim
        self.device = device
        self._tensors: tuple | None = None

    @property
    def tensors(self) -> tuple:
        if self._tensors is None:
            self._tensors = self._prepare_tensors()
        return self._tensors

    def _prepare_tensors(self) -> tuple:
        import torch
        torch.manual_seed(42)
        dtype = torch.bfloat16
        B, T, H, D = 2, 1500, 4, self.head_dim
        print(f"Generating cache with head_dim={D}")

        q = torch.rand(B, T, H, D, dtype=dtype)
        k = torch.rand(B, T, H, D, dtype=dtype)
        v = torch.rand(B, T, H, D, dtype=dtype)
        g = torch.randn(B, T, H, D, dtype=torch.float32)
        A_log = torch.randn(H, dtype=torch.float)
        dt_bias = torch.randn(H * D, dtype=torch.float)
        beta = torch.randn(B, T, H, dtype=dtype).sigmoid()
        h0 = torch.randn(B, H, D, D, dtype=torch.float32)
        A_log, dt_bias = map(lambda x: x.to(self.device).requires_grad_(True), (A_log, dt_bias))
        q, k, v, beta, h0 = map(lambda x: x.to(self.device).requires_grad_(True), (q, k, v, beta, h0))
        g = g.to(self.device).requires_grad_(True)

        do = torch.randn_like(v)
        dht = torch.randn_like(h0)
        return q, k, v, g, beta, h0, do, dht, A_log, dt_bias

    def generate_kda(self) -> None:
        import torch
        import torch.nn.functional as F

        from fla.ops.kda import chunk_kda, fused_recurrent_kda
        from fla.ops.kda.gate import fused_kda_gate

        q, k, v, g, beta, h0, do, dht, A_log, dt_bias = self.tensors
        g_nonsafe = F.logsigmoid(g.clone().float())
        g_safe = -5 * torch.sigmoid(g.clone().float())

        for use_gate, safe_gate, g_in, lower_bound in [
            (False, False, g_nonsafe, None),
            (True,  False, g.clone(), -5),
            (False, True,  g_safe,    -5),
            (True,  True,  g.clone(), -5),
        ]:
            kw = dict(use_gate_in_kernel=use_gate, safe_gate=safe_gate)
            if lower_bound is not None:
                kw["lower_bound"] = lower_bound
            tri, tri_ht = chunk_kda(
                q=q.clone(), k=k.clone(), v=v.clone(), g=g_in,
                beta=beta.clone(), A_log=A_log.clone(), dt_bias=dt_bias.clone(),
                scale=None, initial_state=h0.clone(), output_final_state=True,
                use_qk_l2norm_in_kernel=True, **kw,
            )
            ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)

        g_gated = fused_kda_gate(g=g.clone(), A_log=A_log.clone(), dt_bias=dt_bias.clone())
        fused_recurrent_kda(
            q=q.clone(), k=k.clone(), v=v.clone(), g=g_gated,
            beta=beta.clone(), initial_state=h0.clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )

    def generate_gdn(self) -> None:
        import torch.nn.functional as F

        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        q, k, v, g, beta, h0, do, dht, _, _ = self.tensors
        g = g[..., 0].float().detach().requires_grad_(True)

        for use_qk_l2norm_in_kernel in (False, True):
            tri, tri_ht = chunk_gated_delta_rule(
                q=(F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone()),
                k=(F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone()),
                v=v.clone(), g=g.clone(), beta=beta.clone(),
                scale=None, initial_state=h0.clone(), output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=not use_qk_l2norm_in_kernel)
            q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    def generate_conv(self) -> None:
        import torch
        torch.manual_seed(42)
        dtype = torch.bfloat16
        B, T, H, D = 1, 8192, 32, self.head_dim
        W = 4
        x = torch.randn(B, T, H * D).to(self.device, dtype).requires_grad_(True)
        weight = torch.randn(H * D, W).to(self.device, dtype).requires_grad_(True)
        dy = torch.randn(B, T, H * D).to(self.device, dtype)

        from fla.modules.convolution import causal_conv1d
        tri, _ = causal_conv1d(x, weight, None, residual=None, activation="silu")
        tri.backward(dy)


def generate_fla_cache(
    op: str = 'kda',
    head_dims: int | list[int] | tuple[int, ...] = 128,
    triton_cache_dir: str | None = None,
) -> str:
    """Generate Triton cache with custom directory."""
    from fla.utils import device

    fla_triton_cache = get_triton_cache_dir(triton_cache_dir) / "fla_triton_cache"
    fla_triton_cache.mkdir(parents=True, exist_ok=True)
    for autotune_file in fla_triton_cache.rglob("*.autotune.json"):
        autotune_file.unlink()
    os.environ["TRITON_CACHE_DIR"] = str(fla_triton_cache)

    print(f"Using FLA Triton cache directory: {fla_triton_cache}")

    dims_to_generate = [head_dims] if isinstance(head_dims, int) else list(dict.fromkeys(head_dims))
    print(f"Generating Triton cache for head_dim values: {dims_to_generate}")

    for head_dim in dims_to_generate:
        gen = FLACacheGenerator(head_dim, device=device)
        if op in ('kda', 'both'):
            gen.generate_kda()
        if op in ('gdn', 'both'):
            gen.generate_gdn()
        elif op != 'kda':
            raise ValueError(f"Unsupported op: {op}")
        gen.generate_conv()

    return str(fla_triton_cache)
