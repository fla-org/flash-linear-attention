# Dynamic short convolution

`DynamicShortConvolution` mixes nearby tokens with filters generated from the current input. Each channel combines the most recent `kernel_size` tokens, and the current token determines how much each position contributes. The operation is causal: it never reads future tokens.

Use it to add input-dependent local mixing to hidden states or Q/K/V projections before attention. It supports full-sequence training and cached decoding, with the same input, output, and cache conventions as `ShortConvolution`.

```python
import torch

from fla.modules import DynamicShortConvolution

conv = DynamicShortConvolution(hidden_size=512, kernel_size=4, rank=16).cuda()
x = torch.randn(2, 128, 512, device="cuda")
y, state = conv(x, output_final_state=True)

x_next = torch.randn(2, 1, 512, device="cuda")
y_next, state = conv(x_next, cache=state, output_final_state=True)
```

The kernel generator uses a low-rank projection controlled by `rank`. Its intermediate activation defaults to `None`; set `generator_activation="swish"` for a nonlinear generator. The separate `activation` argument controls the convolution output activation and defaults to `"silu"`.

By default, filters are generated from `x`. To generate them from hidden states while convolving a projected representation, set `generator_input_size` and pass those hidden states as `generator_input`.

Reference: [Dynamic Short Convolutions Improve Transformers](https://arxiv.org/abs/2606.03825v1).
