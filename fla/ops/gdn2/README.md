# Symmetric erase inputs

The GDN2 operator accepts a general channel-wise erase gate.
Callers can represent the symmetric transition `I - u u.T` with `k=u` and an all-ones erase gate.
For this transition, non-expansiveness requires `||u||² <= 2` after dtype conversion.

For example, casting `sqrt(2) * normalize([1, 2])` to BF16 gives `[0.6328125, 1.265625]`.
Its squared norm is `2.00225830078125`, which makes the erase step expansive.

Call `bound_symmetric_key` after constructing and casting a symmetric key:

```python
import torch

from fla.ops.gdn2 import chunk_gdn2
from fla.ops.gdn2.bound import bound_symmetric_key

# q is already normalized; u is the already-transformed symmetric key.
u = bound_symmetric_key(u)
output, state = chunk_gdn2(
    q=q,
    k=u,
    v=v,
    g=g,
    b=torch.ones_like(u),
    w=w,
    use_qk_l2norm_in_kernel=False,
)
```

Do not normalize the transformed key again. That would change its erase and write strength.
Do not pass the original channel-wise erase gate again. It is already part of the symmetric key construction.

The helper targets a delivered squared norm of 1.99 and reserves final-cast rounding headroom.
The 1.99 target is a conservative margin below the mathematical limit of two.
Accumulation uses at least FP32 and preserves FP64 inputs.
Inputs must be finite, with squared norms representable in the accumulation dtype.
Regression coverage includes key widths 64, 128, and 256 in BF16, FP16, FP32, and FP64.

Keys below the internal ceiling retain their values. The cap changes erase and write dynamics near the boundary.
Autograd differentiates through the radial cap, including its scale. The decay tensor is unchanged.
This utility does not make arbitrary channel-wise erase transitions non-expansive.
It also does not guarantee kernel accuracy or bounded model outputs.
