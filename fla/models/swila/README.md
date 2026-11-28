# SwiLA: Switching Linear Attention

[SwiLA](https://arxiv.org/abs/2609.39034) routes each token among a fixed bank of linear memories, combining nonlinear retrieval with recurrent inference state that does not grow with sequence length at fixed model dimensions. FLA provides a standalone attention layer and Hugging Face models with four recurrent variants.

The examples use NVIDIA CUDA and BF16. Follow the [CUDA installation instructions](../../../INSTALL.md) before running them. SwiLA uses `mode="fused_recurrent"` and supports `1 <= num_experts <= 8`; its layer and model have no CPU recurrence fallback. The differentiable PyTorch reference is available separately as `fla.ops.swila.swila_recurrence`.

## Use the layer

```python
import torch

from fla.layers import SwitchingLinearAttention

torch.manual_seed(0)
layer = SwitchingLinearAttention(
    d_model=256,
    num_heads=4,
    num_experts=4,
    temporal=True,
    use_forget_gate=True,
).to(device="cuda", dtype=torch.bfloat16).eval()
hidden_states = torch.randn(2, 64, 256, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    output = layer(hidden_states)  # [2, 64, 256]
```

Select a variant through the same two flags on the layer or `SwiLAConfig`:

| Variant              | `temporal` | `use_forget_gate` |
| -------------------- | ---------- | ----------------- |
| Plain SwiLA          | `False`    | `False`           |
| Gated SwiLA          | `False`    | `True`            |
| Temporal SwiLA       | `True`     | `False`           |
| Temporal-gated SwiLA | `True`     | `True`            |

`use_gate` is a separate output gate. Temporal filtering uses exact identity carry with learned reset gates. With the default `architecture_profile="current"`, beta is shared over experts; gated recurrence applies key-wise decay before prediction and the memory update. See the [operator contracts](../../ops/swila/README.md) for tensor shapes and return values.

## Compute a language-model loss with MI

```python
import torch

from fla.models.swila import SwiLAConfig, SwiLAForCausalLM

torch.manual_seed(0)
config = SwiLAConfig(
    vocab_size=1024,
    hidden_size=256,
    num_hidden_layers=2,
    num_heads=4,
    num_experts=4,
    temporal=True,
    use_forget_gate=True,
    mi_loss_weight=5e-4,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
model = SwiLAForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).train()
tokens = torch.randint(3, config.vocab_size, (2, 64), device="cuda")
model.zero_grad(set_to_none=True)
result = model(input_ids=tokens, labels=tokens, use_cache=False, return_dict=True)
result.loss.backward()
```

The reported `result.loss` value is cross entropy; its backward includes cross entropy and the sum of the weighted layer mutual-information (MI) objectives. Do not add MI again. MI is enabled in training mode when `mi_loss_weight > 0`; calls without `labels` or `shift_labels` return no language-model loss. The random tokens demonstrate the API; real text needs a tokenizer whose vocabulary matches the model.

For a standalone layer, request the four-item return and add its already weighted auxiliary loss to your task objective:

```python
import torch

from fla.layers import SwitchingLinearAttention

torch.manual_seed(0)
layer = SwitchingLinearAttention(
    d_model=256,
    num_heads=4,
    num_experts=4,
    temporal=True,
    use_forget_gate=True,
    return_output_only=False,
    mi_loss_weight=5e-4,
).to(device="cuda", dtype=torch.bfloat16).train()
hidden_states = torch.randn(2, 64, 256, device="cuda", dtype=torch.bfloat16)
output, attentions, cache, auxiliary_loss = layer(hidden_states)
loss = output.float().square().mean()
if auxiliary_loss is not None:
    loss = loss + auxiliary_loss
loss.backward()
```

The default tensor-only layer return does not attach MI to an external task loss. Diagnostic capture and external logging are not required for either MI path.

Default Hugging Face `Trainer`, without label smoothing or a custom loss function, averages the complete per-call CE+MI objective across gradient-accumulation microbatches. Equal sequence counts and equal valid-label counts per microbatch match the corresponding full-batch objective. Unequal microbatches retain a mean of microbatch means. Label smoothing or custom loss code that computes CE outside the model must include MI explicitly. Passing `num_items_in_batch` directly normalizes CE only; custom training loops must choose their MI accumulation scaling separately.

## Save, load, and generate

This block continues the language-model example above and uses native Transformers 5 loading; no remote model code is needed:

```python
checkpoint_dir = "./swila-example"
model.save_pretrained(checkpoint_dir)
loaded = SwiLAForCausalLM.from_pretrained(
    checkpoint_dir,
    dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda").eval()
with torch.no_grad():
    generated = loaded.generate(inputs=tokens[:1, :8], max_new_tokens=16, do_sample=False, use_cache=True)
```

Importing `fla.models.swila` also registers `SwiLAConfig`, `SwiLAModel` and `SwiLAForCausalLM` with the corresponding Transformers auto classes. For gated models, checkpoint-stored FP32 `A_log` and `dt_bias` values remain FP32 through `from_pretrained(dtype=...)` and ordinary `.to(...)` placement. This preserves the stored precision; it cannot recover values already rounded in a checkpoint.

Greedy decoding uses FLA's recurrent cache. Generation modes requiring recurrent-state rollback are unsupported.

## Cache, padding, and packed inputs

- For inference at model level, `use_cache=True` creates or accepts an FLA cache and returns it in `past_key_values`. A standalone layer needs an explicit `fla.models.cache_utils.Cache()` and a nonnegative `layer_idx`. It advances a supplied cache even with `use_cache=False`; the flag alone does not create a standalone cache. Use `return_output_only=False` to receive its four-item return.
- Dense batches support a 2-D `attention_mask`, including left-padded prefill and decoding. A supplied mask takes precedence over explicit packed boundaries.
- Packed inputs use batch size one and contiguous on-device int32/int64 `cu_seqlens`. Boundaries start at zero, end at the total sequence length and increase strictly; empty sequences are unsupported. Cached states need one entry per sequence in consistent order.
- Packed boundaries isolate recurrence, but ordinary causal-label shifting does not mask targets across sequence boundaries. For packed language-model training, set labels at the starts of subsequent sequences to `-100`, or supply correctly constructed `shift_labels`. Padding labels also need `-100` masking.
- Packed MI averages tokens within each sequence, then weights sequences equally. This is separate from CE token normalization. Packed forward/cache support does not make `.generate()` decode packed sequences as independent requests.

## Configuration compatibility

The configuration retains SwiLA's projection, initialization and optional hybrid-attention settings. Unsupported top-level options fail explicitly; `downstream_config` is an inert metadata namespace for external tools. Historical `model_type="sla"` configurations load as `SwiLAConfig`, and newly saved configurations use `model_type="swila"`. Equivalent retired selector values are validated and discarded; incompatible values and retired execution modes are rejected. Loading does not convert incompatible weights or infer the opt-in historical plain architecture profile.
