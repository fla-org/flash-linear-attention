# SwiLA recurrent operators

Switching Linear Attention (SwiLA) maintains a bank of linear memories and routes each key/value update and query among those memories. These operators implement plain, gated, temporal, and temporal-gated recurrence. The PyTorch reference is in `naive.py`; the Triton entry points own their forward/backward implementations and share operation-local primitives.

```python
import torch

from fla.ops.swila import fused_recurrent_swila

torch.manual_seed(0)
q = torch.randn(1, 16, 2, 32, device="cuda", requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
beta = torch.full((1, 16, 2, 32, 4), 0.5, device="cuda", requires_grad=True)
pi_q = torch.randn_like(beta, requires_grad=True)
pi_k = torch.randn_like(beta, requires_grad=True)
output, final_state, responsibilities, query_weights = fused_recurrent_swila(
    q=q,
    k=k,
    v=v,
    beta=beta,
    pi_q=pi_q,
    pi_k=pi_k,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
output.square().mean().backward()
```

## Inputs and variants

Queries and keys have shape `[B, T, H, Dk]`, values have shape `[B, T, H, Dv]`, and beta and router logits have shape `[B, T, H, Dv, E]`. The recurrent memory has shape `[N, H, Dv, E, Dk]`. Beta is the already transformed learning rate; `pi_q` and `pi_k` are logits. `scale=None` scales queries by `Dk**-0.5`.

| Entry point                             | Additional inputs                                     | Return values                                                          |
| --------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| `fused_recurrent_swila`                 | Optional `log_decay` delegates to the gated operator. | `output, S_final, r_seq, rho_seq`                                      |
| `fused_recurrent_gated_swila`           | Required key-wise `log_decay`.                        | `output, S_final, r_seq, rho_seq`                                      |
| `fused_recurrent_swila_temporal`        | Reset logits `gate_kv`, `gate_q`; optional carries.   | `output, S_final, r_final, rho_final, r_seq, rho_seq, None, u, pi_bar` |
| `fused_recurrent_gated_swila_temporal`  | Temporal inputs and required key-wise `log_decay`.    | Same as the temporal operator.                                         |

Gated recurrence applies `exp(log_decay)` to the previous memory before prediction, responsibility calculation and the residual update. `log_decay` has shape `[B, T, H, Dk]` and is shared over values and experts. Temporal recurrence uses an exact identity carry mixed with the reset prior. The fused temporal reset-logit shapes are `[B, T, H, Dv]`; the PyTorch reference expects a trailing singleton expert axis. Initial temporal carries `r_prev_init` and `rho_prev_init` must be supplied together and have shape `[N, H, Dv, E]`.

Fused recurrence supports `1 <= E <= 8`. With dense inputs, `N=B`. Packed inputs require `B=1` and contiguous on-device int32/int64 `cu_seqlens`, starting at zero and ending at T, with strictly increasing boundaries; `N` is the number of sequences. Empty packed sequences are unsupported. The PyTorch reference has no packed-boundary argument: evaluate each sequence independently when constructing a packed reference.

## Outputs, gradients and precision

The output follows the query dtype. Recurrent memories and key-side responsibilities are FP32; temporal query weights, residuals and mixed priors are FP32. Non-temporal query weights are `softmax(pi_q)` and follow PyTorch’s softmax autocast policy: ordinarily the input dtype, but FP32 under CUDA autocast. Final state/carry outputs are `None` unless `output_final_state=True`. Non-temporal calls can suppress `rho_seq` using `return_rho_seq=False`.

Backward supports output, memory, responsibility and temporal-carry objectives, including temporal residual/mixed-prior auxiliaries. The checkpoint phase context in `checkpoint_context.py` suppresses temporal auxiliary materialization during the prepass and restores the ordinary behavior during recomputation. `swila_recurrence` provides the differentiable sequential PyTorch implementation and returns `(output, final_state, responsibilities)`; its accumulation is FP32.

`fused_swila_log_decay_gate` computes FP32 `-exp(a_log) * softplus(raw + dt_bias)` with a stable negative-tail softplus. `compute_swila_mi_loss` supplies the graph-connected mutual-information objective, with optional detached statistics. Packed MI averages tokens within each sequence, then averages sequences equally. E=1 yields a connected zero. These helpers remain independent of model logging and training frameworks.

General and split backward paths are distinct execution routes, not different recurrence equations. Fused recurrence currently requires NVIDIA CUDA; other backends fail before kernel execution. Gate and MI helpers retain eager fallbacks on other devices. Eager-capable custom autograd helpers preserve the actual input device’s autocast state, including CPU, while fused recurrence uses FLA’s accelerator AMP decorators without implicit input casts.

See the [SwiLA model guide](../../models/swila/README.md) for layer, language-model and MI usage, and the [paper](https://arxiv.org/abs/2609.39034) for the recurrence derivation. The original implementation's MIT attribution is retained in [LICENSE](LICENSE).
