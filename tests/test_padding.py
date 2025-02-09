# Credit: https://github.com/state-spaces/mamba/pull/70
# I made some modifications to the original code to make it work with the current version of the library.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fla.utils import device

model = AutoModelForCausalLM.from_pretrained('fla-hub/gla-340M-15B').to(device).to(torch.float32)
tokenizer = AutoTokenizer.from_pretrained('fla-hub/gla-340M-15B')
print("The original tokenizer padding side:", tokenizer.padding_side)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
print("The afterward tokenizer padding side:", tokenizer.padding_side)

pad_count = 24
seq_len = 1024
input_ids = torch.randint(1, 1000, (1, seq_len)).to(device)
# Check prefill logits
if tokenizer.padding_side == 'left':
    input_ids_padded = torch.cat([torch.zeros_like(input_ids[:, [0] * pad_count]), input_ids], dim=1)
    attention_mask = torch.cat([torch.zeros_like(input_ids[:, [0] * pad_count]), torch.ones_like(input_ids)], dim=1)
else:
    input_ids_padded = torch.cat([input_ids, torch.zeros_like(input_ids[:, [0] * pad_count])], dim=1)
    attention_mask = torch.cat([torch.ones_like(input_ids), torch.zeros_like(input_ids[:, [0] * pad_count])], dim=1)

out = model(input_ids_padded).logits.detach().cpu()
out_padded = model(input_ids_padded, attention_mask).logits.detach().cpu()
out_true = model(input_ids).logits.detach().cpu()

if tokenizer.padding_side == 'left':
    print("max L2 error (unpadded):", (out_true - out[:, pad_count:]).norm(dim=-1).max())
    print("max L2 errors (padded):", (out_true - out_padded[:, pad_count:]).norm(dim=-1).max())
else:
    print("max L2 error (unpadded):", (out_true - out[:, :seq_len]).norm(dim=-1).max())
    print("max L2 errors (padded):", (out_true - out_padded[:, :seq_len]).norm(dim=-1).max())

# Check decoding outputs
text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.'

print("\n\nNo CUDA graph:")
inputs = tokenizer([text], return_tensors='pt').to(device)
x = model.generate(inputs.input_ids, max_length=100, temperature=0)
print("\nNo pad, no mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

inputs = tokenizer(['<|endoftext|>' * pad_count + text], return_tensors='pt').to(device)
x = model.generate(inputs.input_ids, max_length=100 + pad_count, temperature=0)
print("\nPad, no mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

inputs = tokenizer(['<|endoftext|>' * pad_count + text], return_tensors='pt').to(device)
inputs.attention_mask[:, :pad_count] = 0
x = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=100 + pad_count, temperature=0)
print("\nPad, mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

print("\n\nCUDA graph:")
inputs = tokenizer([text], return_tensors='pt').to(device)
x = model.generate(inputs.input_ids, max_length=100, temperature=0)
print("\nNo pad, no mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

inputs = tokenizer(['<|endoftext|>' * pad_count + text], return_tensors='pt').to(device)
x = model.generate(inputs.input_ids, max_length=100 + pad_count, temperature=0)
print("\nPad, no mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))

inputs = tokenizer(['<|endoftext|>' * pad_count + text], return_tensors='pt').to(device)
inputs.attention_mask[:, :pad_count] = 0
x = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=100 + pad_count, temperature=0)
print("\nPad, mask:")
print(tokenizer.decode(x[0], skip_special_tokens=True))
