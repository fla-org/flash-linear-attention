# SwiLA layer support

`fla.layers.SwitchingLinearAttention` owns its parameters, projections and caches.
The helpers here initialize or inspect that layer without adding a parameter-owning wrapper.

```python
from fla.layers import SwitchingLinearAttention
from fla.modules.swila import check_swila_initialization, capture_swila_diagnostics

layer = SwitchingLinearAttention(d_model=256, num_heads=4, num_experts=4)
check_swila_initialization(layer)
```

Initialization checks are opt-in and intended for freshly initialized values, not
trained checkpoints. They preserve parameters and RNG state; distributed checks
must be called on every rank.

`capture_swila_diagnostics(layer, grad_norms=True)` is a context manager. Keep it
open through backward when collecting gradient norms. Captured statistics are
detached; the layer's MI auxiliary loss remains differentiable regardless of
whether diagnostics are enabled. The full layer return is
`(output, attentions, cache, aux_loss)` when `return_output_only=False`; otherwise
the layer returns the output tensor and retains the connected loss in `_aux_loss`.

`swila_checkpoint_context_fn` supplies the two contexts accepted by non-reentrant
PyTorch checkpointing. It marks prepass and recomputation for the same operator
phase state used by diagnostic capture; it does not install checkpointing.

`initialization.py` restores specialized initialization, loading guards and
optimizer metadata. `router.py` computes logits and optional noise in their
original order. `packing.py` validates packed sequences and repads results.
`projections.py` provides the low-rank linear and SwiGLU projections. These are
SwiLA-specific helpers; they do not replace FLA's shared modules.
