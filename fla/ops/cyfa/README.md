# Cyclic Flow Attention

CyFA maintains aligned key and value memories organized by relative time.
A learned clock determines cyclic transport; a scalar gate controls decay and a learned readout retrieves from the memory.
In absolute-clock coordinates the computation is two scalar-gated linear attention passes connected by a token-wise readout.

- [Paper](https://arxiv.org/abs/2609.36259)
- [Official implementation and checkpoints](https://github.com/Chyxx/CyclicFlowAttention)

`chunk_cyfa` supports training and prefill, including packed variable-length sequences and differentiable initial and final states.
`checkpoint_level=0` saves the intermediate readout activations; `1` recomputes them during backward.
`fused_recurrent_cyfa` supports inference, including single-token decoding, and rejects inputs requiring gradients.
The current optimized implementation targets NVIDIA GPUs with BF16 support (Ampere or newer).

Inputs use sequence-first layouts: `q` and `k` have shape `[B, T, H, Dk]`, `v` has shape `[B, T, H, Dv]`, and `g`, `delta`, and `beta` have shape `[B, T, H]`.
`g` is the log decay, while `delta` and `beta` are the activated clock increment and write strength.
The learned `readout` has shape `[H, M - 1, M - 1]`; the released models use `M=128` storage coordinates for 127 active coordinates.
Q/K RMSNorm weights have shape `[Dk]` and are applied inside the operator.

Both operators return `(output, final_state)`, with `final_state=None` unless requested.
The state tuple contains FP32 tensors with shapes `[N, H, Dk, M]`, `[N, H, M, Dv]`, and `[N, H]` for the key memory, value memory, and cumulative clock.
For dense inputs `N=B`; for packed inputs `B=1` and `N=len(cu_seqlens)-1`.
The inference-only single-token path can update the supplied memory tensors in place when final states are requested.

```python
import torch

from fla.layers import CyclicFlowAttention

layer = CyclicFlowAttention(
    hidden_size=1024,
    num_heads=4,
    head_dim=256,
    num_slots=128,
    layer_idx=0,
).cuda().bfloat16()
x = torch.randn(1, 2048, 1024, device='cuda', dtype=torch.bfloat16)
y, _, _ = layer(x)
```

Importing `fla` registers `CyclicFlowAttentionConfig`, `CyclicFlowAttentionModel`, and `CyclicFlowAttentionForCausalLM` with Transformers.
The model type remains `cyclic_flow_attention`, so the official checkpoints use the same config and parameter names.

```python
import fla
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'cyxxxxxxxxxx/cyfa-400M-15B',
    dtype=torch.bfloat16,
).cuda().eval()
```
