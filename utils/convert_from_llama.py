
# scripts for converting pretrained hf model weights to fla style
# calling the code to make conversions for mistralai/Mistral-7B-v0.1 would achieve the following results:
# |    Tasks     |Version|Filter|n-shot|  Metric  |Value |   |Stderr|
# |--------------|------:|------|-----:|----------|-----:|---|-----:|
# |arc_challenge |      1|none  |     0|acc       |0.5043|±  |0.0146|
# |              |       |none  |     0|acc_norm  |0.5392|±  |0.0146|
# |arc_easy      |      1|none  |     0|acc       |0.8081|±  |0.0081|
# |              |       |none  |     0|acc_norm  |0.7946|±  |0.0083|
# |boolq         |      2|none  |     0|acc       |0.8373|±  |0.0065|
# |copa          |      1|none  |     0|acc       |0.9300|±  |0.0256|
# |hellaswag     |      1|none  |     0|acc       |0.6127|±  |0.0049|
# |              |       |none  |     0|acc_norm  |0.8100|±  |0.0039|
# |lambada_openai|      1|none  |     0|perplexity|3.1810|±  |0.0583|
# |              |       |none  |     0|acc       |0.7563|±  |0.0060|
# |openbookqa    |      1|none  |     0|acc       |0.3260|±  |0.0210|
# |              |       |none  |     0|acc_norm  |0.4380|±  |0.0222|
# |piqa          |      1|none  |     0|acc       |0.8069|±  |0.0092|
# |              |       |none  |     0|acc_norm  |0.8215|±  |0.0089|
# |sciq          |      1|none  |     0|acc       |0.9580|±  |0.0063|
# |              |       |none  |     0|acc_norm  |0.9390|±  |0.0076|
# |winogrande    |      1|none  |     0|acc       |0.7395|±  |0.0123|


import argparse
import warnings

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'


def convert(
    llama: str,
    config: str,
    output: str,
    precision: str = 'float32',
):
    AutoTokenizer.from_pretrained(llama).save_pretrained(output)
    llama = AutoModelForCausalLM.from_pretrained(llama, torch_dtype=precision)

    config = AutoConfig.from_pretrained(config)
    config.torch_dtype = precision
    model = AutoModelForCausalLM.from_config(config)
    if precision in ['float16', 'fp16']:
        model = model.to(torch.float16)
    elif precision in ['bfloat16', 'bf16']:
        model = model.to(torch.bfloat16)
    model.num_parameters()

    vocab_size = llama.model.embed_tokens.weight.shape[0]
    if model.model.embeddings.weight.shape[0] != vocab_size:
        warnings.warn(f"Llama and the model have different embedding sizes "
                      f"({vocab_size} vs {model.model.embeddings.weight.shape[0]}), "
                      f"the model embeddings will be extended with randomly initialized values or truncated", stacklevel=2)
        vocab_size = min(model.model.embeddings.weight.shape[0], vocab_size)
    model.model.embeddings.weight.data[:vocab_size].copy_(llama.model.embed_tokens.weight[:vocab_size])
    torch.testing.assert_close(model.model.embeddings.weight[:vocab_size], llama.model.embed_tokens.weight[:vocab_size])
    for i in range(config.num_hidden_layers):
        if hasattr(model.model.layers[i], 'attn_norm'):
            if model.model.layers[i].attn_norm.weight is not None:
                model.model.layers[i].attn_norm.weight.data.copy_(llama.model.layers[i].input_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].attn_norm.weight,
                                           llama.model.layers[i].input_layernorm.weight)
            if model.model.layers[i].attn_norm.bias is not None:
                model.model.layers[i].attn_norm.bias.data.copy_(llama.model.layers[i].input_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].attn_norm.bias,
                                           llama.model.layers[i].input_layernorm.bias)
            model.model.layers[i].attn_norm.eps = llama.model.layers[i].input_layernorm.variance_epsilon
        if hasattr(model.model.layers[i].attn, 'norm'):
            if model.model.layers[i].attn.norm.weight is not None:
                model.model.layers[i].attn.norm.weight.data.copy_(llama.model.layers[i].input_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].attn.norm.weight,
                                           llama.model.layers[i].input_layernorm.weight)
            if model.model.layers[i].attn.norm.bias is not None:
                model.model.layers[i].attn.norm.bias.data.copy_(llama.model.layers[i].input_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].attn.norm.bias,
                                           llama.model.layers[i].input_layernorm.bias)
            model.model.layers[i].attn.norm.eps = llama.model.layers[i].input_layernorm.variance_epsilon

        model.model.layers[i].attn.q_proj.weight.data.copy_(llama.model.layers[i].self_attn.q_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.q_proj.weight, llama.model.layers[i].self_attn.q_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.q_proj, 'bias') and hasattr(model.model.layers[i].attn.q_proj, 'bias'):
            model.model.layers[i].attn.q_proj.bias.data.copy_(llama.model.layers[i].self_attn.q_proj.bias)
            torch.testing.assert_close(model.model.layers[i].attn.q_proj.bias, llama.model.layers[i].self_attn.q_proj.bias)
        model.model.layers[i].attn.k_proj.weight.data.copy_(llama.model.layers[i].self_attn.k_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.k_proj.weight, llama.model.layers[i].self_attn.k_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.k_proj, 'bias') and hasattr(model.model.layers[i].attn.k_proj, 'bias'):
            model.model.layers[i].attn.k_proj.bias.data.copy_(llama.model.layers[i].self_attn.k_proj.bias)
            torch.testing.assert_close(model.model.layers[i].attn.k_proj.bias, llama.model.layers[i].self_attn.k_proj.bias)
        model.model.layers[i].attn.v_proj.weight.data.copy_(llama.model.layers[i].self_attn.v_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.v_proj.weight, llama.model.layers[i].self_attn.v_proj.weight)
        if hasattr(llama.model.layers[i].self_attn.v_proj, 'bias') and hasattr(model.model.layers[i].attn.v_proj, 'bias'):
            model.model.layers[i].attn.v_proj.bias.data.copy_(llama.model.layers[i].self_attn.v_proj.bias)
            torch.testing.assert_close(model.model.layers[i].attn.v_proj.bias, llama.model.layers[i].self_attn.v_proj.bias)

        model.model.layers[i].attn.o_proj.weight.data.copy_(llama.model.layers[i].self_attn.o_proj.weight)
        torch.testing.assert_close(model.model.layers[i].attn.o_proj.weight, llama.model.layers[i].self_attn.o_proj.weight)

        if hasattr(model.model.layers[i], 'mlp_norm'):
            if model.model.layers[i].mlp_norm.weight is not None:
                model.model.layers[i].mlp_norm.weight.data.copy_(llama.model.layers[i].post_attention_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].mlp_norm.weight,
                                           llama.model.layers[i].post_attention_layernorm.weight)
            if model.model.layers[i].mlp_norm.bias is not None:
                model.model.layers[i].mlp_norm.bias.data.copy_(llama.model.layers[i].post_attention_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].mlp_norm.bias,
                                           llama.model.layers[i].post_attention_layernorm.bias)
            model.model.layers[i].mlp_norm.eps = llama.model.layers[i].post_attention_layernorm.variance_epsilon
        if hasattr(model.model.layers[i].mlp, 'norm'):
            if model.model.layers[i].mlp.norm.weight is not None:
                model.model.layers[i].mlp.norm.weight.data.copy_(llama.model.layers[i].post_attention_layernorm.weight)
                torch.testing.assert_close(model.model.layers[i].mlp.norm.weight,
                                           llama.model.layers[i].post_attention_layernorm.weight)
            if model.model.layers[i].mlp.norm.bias is not None:
                model.model.layers[i].mlp.norm.bias.data.copy_(llama.model.layers[i].post_attention_layernorm.bias)
                torch.testing.assert_close(model.model.layers[i].mlp.norm.bias,
                                           llama.model.layers[i].post_attention_layernorm.bias)
            model.model.layers[i].mlp.norm.eps = llama.model.layers[i].post_attention_layernorm.variance_epsilon

        model.model.layers[i].mlp.gate_proj.weight.data.copy_(llama.model.layers[i].mlp.gate_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.gate_proj.weight, llama.model.layers[i].mlp.gate_proj.weight)
        model.model.layers[i].mlp.up_proj.weight.data.copy_(llama.model.layers[i].mlp.up_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.up_proj.weight, llama.model.layers[i].mlp.up_proj.weight)

        model.model.layers[i].mlp.down_proj.weight.data.copy_(llama.model.layers[i].mlp.down_proj.weight)
        torch.testing.assert_close(model.model.layers[i].mlp.down_proj.weight,
                                   llama.model.layers[i].mlp.down_proj.weight)

    if model.model.norm.weight is not None:
        model.model.norm.weight.data.copy_(llama.model.norm.weight)
        torch.testing.assert_close(model.model.norm.weight, llama.model.norm.weight)
    if model.model.norm.bias is not None:
        model.model.norm.bias.data.copy_(llama.model.norm.bias)
        torch.testing.assert_close(model.model.norm.bias, llama.model.norm.bias)
    model.model.norm.eps = llama.model.norm.variance_epsilon

    if not model.config.tie_word_embeddings:
        model.lm_head.weight.data[:vocab_size].copy_(llama.lm_head.weight[:vocab_size])
        torch.testing.assert_close(model.lm_head.weight[:vocab_size], llama.lm_head.weight[:vocab_size])
    model.config.rope_theta = llama.config.rope_theta

    model.save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='mistralai/Mistral-7B-v0.1')
    parser.add_argument("--config", default='configs/transformer_7B.json')
    parser.add_argument("--output", default='converted/transformer-7B')
    parser.add_argument('--precision', type=str, default='float32')
    args = parser.parse_args()
    convert(args.model, args.config, args.output, precision=args.precision)
