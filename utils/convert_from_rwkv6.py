# -*- coding: utf-8 -*-

# scripts for converting pretrained hf model weights to fla style
# calling the code to make conversions for RWKV/rwkv-6-world-7b would achieve the following results:
# |    Tasks     |Version|Filter|n-shot|    Metric     | Value |   |Stderr|
# |--------------|------:|------|-----:|---------------|------:|---|------|
# |arc_challenge |      1|none  |     0|acc            | 0.4130|±  |0.0144|
# |              |       |none  |     0|acc_norm       | 0.4403|±  |0.0145|
# |arc_easy      |      1|none  |     0|acc            | 0.7382|±  |0.0090|
# |              |       |none  |     0|acc_norm       | 0.7079|±  |0.0093|
# |boolq         |      2|none  |     0|acc            | 0.6823|±  |0.0081|
# |copa          |      1|none  |     0|acc            | 0.8700|±  |0.0338|
# |hellaswag     |      1|none  |     0|acc            | 0.5508|±  |0.0050|
# |              |       |none  |     0|acc_norm       | 0.7171|±  |0.0045|
# |lambada_openai|      1|none  |     0|perplexity     | 3.2989|±  |0.0634|
# |              |       |none  |     0|acc            | 0.7493|±  |0.0060|
# |openbookqa    |      1|none  |     0|acc            | 0.3200|±  |0.0209|
# |              |       |none  |     0|acc_norm       | 0.4440|±  |0.0222|
# |piqa          |      1|none  |     0|acc            | 0.7753|±  |0.0097|
# |              |       |none  |     0|acc_norm       | 0.7894|±  |0.0095|
# |sciq          |      1|none  |     0|acc            | 0.9370|±  |0.0077|
# |              |       |none  |     0|acc_norm       | 0.8860|±  |0.0101|
# |winogrande    |      1|none  |     0|acc            | 0.6867|±  |0.0130|

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from fla.utils import device


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'


def convert(
    rwkv6: str,
    config: str,
    output: str
):
    torch.manual_seed(1)
    AutoTokenizer.from_pretrained(rwkv6, trust_remote_code=True).save_pretrained(output)
    rwkv6 = AutoModelForCausalLM.from_pretrained(rwkv6, trust_remote_code=True).to(device)
    print(f"Loading rwkv6 ...\n{rwkv6}")

    config = AutoConfig.from_pretrained(config)
    model = AutoModelForCausalLM.from_config(config).to(device)
    num_parameters = model.num_parameters()
    print(f"Initializing the model from the config:\n{config}\n{model}")
    print(f"Number of parameters in total: {num_parameters} ({sizeof_fmt(num_parameters)})")

    print("Copying the weights from rwkv6 to the model ...")
    print("rwkv6.rwkv.embeddings                        -> model.model.embeddings")
    model.model.embeddings.weight.data.copy_(rwkv6.rwkv.embeddings.weight)
    torch.testing.assert_close(model.model.embeddings.weight, rwkv6.rwkv.embeddings.weight)
    for i in range(config.num_hidden_layers):
        if hasattr(model.model.layers[i], 'pre_norm'):
            if model.model.layers[i].pre_norm.weight is not None:
                print(f"rwkv6.rwkv.blocks{i}.pre_ln.weight -> model.model.layers{i}.pre_norm.weight")
                model.model.layers[i].pre_norm.weight.data.copy_(rwkv6.rwkv.blocks[i].pre_ln.weight)
                torch.testing.assert_close(model.model.layers[i].pre_norm.weight, rwkv6.rwkv.blocks[i].pre_ln.weight)
            if model.model.layers[i].pre_norm.bias is not None:
                print(f"rwkv6.rwkv.blocks{i}.pre_ln.bias -> model.model.layers{i}.pre_norm.bias")
                model.model.layers[i].pre_norm.bias.data.copy_(rwkv6.rwkv.blocks[i].pre_ln.bias)
                torch.testing.assert_close(model.model.layers[i].pre_norm.bias, rwkv6.rwkv.blocks[i].pre_ln.bias)
            model.model.layers[i].pre_norm.eps = rwkv6.rwkv.blocks[i].pre_ln.eps
        if model.model.layers[i].attn_norm.weight is not None:
            print(f"rwkv6.rwkv.blocks{i}.ln1.weight -> model.model.layers{i}.attn_norm.weight")
            model.model.layers[i].attn_norm.weight.data.copy_(rwkv6.rwkv.blocks[i].ln1.weight)
            torch.testing.assert_close(model.model.layers[i].attn_norm.weight, rwkv6.rwkv.blocks[i].ln1.weight)
        if model.model.layers[i].attn_norm.bias is not None:
            print(f"rwkv6.rwkv.blocks{i}.ln1.bias -> model.model.layers{i}.attn_norm.bias")
            model.model.layers[i].attn_norm.bias.data.copy_(rwkv6.rwkv.blocks[i].ln1.bias)
            torch.testing.assert_close(model.model.layers[i].attn_norm.bias, rwkv6.rwkv.blocks[i].ln1.bias)
        model.model.layers[i].attn_norm.eps = rwkv6.rwkv.blocks[i].ln1.eps

        print(f"rwkv6.rwkv.blocks{i}.attention.time_maa_x -> model.model.layers.{i}.attn.x_proj0.mu")
        model.model.layers[i].attn.x_proj[0].mu.data.copy_(rwkv6.rwkv.blocks[i].attention.time_maa_x.view(-1))
        torch.testing.assert_close(model.model.layers[i].attn.x_proj[0].mu,
                                   rwkv6.rwkv.blocks[i].attention.time_maa_x.view(-1))
        print(f"rwkv6.rwkv.blocks{i}.attention.time_maa_w1.weight  -> model.model.layers{i}.attn.x_proj0.linear.weight")
        ww, wk, wv, wr, wg = rwkv6.rwkv.blocks[i].attention.time_maa_w1.view(config.hidden_size, 5, -1).unbind(-2)
        w = torch.cat((wr, ww, wk, wv, wg), -1).t()
        model.model.layers[i].attn.x_proj[0].linear.weight.data.copy_(w)
        torch.testing.assert_close(model.model.layers[i].attn.x_proj[0].linear.weight, w)

        print(f"rwkv6.rwkv.blocks{i}.attention.time_maa_w2.weight  -> model.model.layers{i}.attn.x_proj2.weight")
        ww, wk, wv, wr, wg = rwkv6.rwkv.blocks[i].attention.time_maa_w2.unbind(0)
        w = torch.cat((wr, ww, wk, wv, wg), 0).t()
        model.model.layers[i].attn.x_proj[2].weight.data.copy_(w)
        torch.testing.assert_close(model.model.layers[i].attn.x_proj[2].weight, w)

        print(f"rwkv6.rwkv.blocks{i}.attention.time_maa_wkvrg  -> model.model.layers{i}.attn.x_bias")
        bias = torch.stack((rwkv6.rwkv.blocks[i].attention.time_maa_r.view(-1),
                            rwkv6.rwkv.blocks[i].attention.time_maa_w.view(-1),
                            rwkv6.rwkv.blocks[i].attention.time_maa_k.view(-1),
                            rwkv6.rwkv.blocks[i].attention.time_maa_v.view(-1),
                            rwkv6.rwkv.blocks[i].attention.time_maa_g.view(-1)))
        model.model.layers[i].attn.x_bias.data.copy_(bias)
        torch.testing.assert_close(model.model.layers[i].attn.x_bias, bias)

        print(f"rwkv6.rwkv.blocks{i}.attention.receptance.weight  -> model.model.layers{i}.attn.r_proj.linear.weight")
        model.model.layers[i].attn.r_proj.linear.weight.data.copy_(rwkv6.rwkv.blocks[i].attention.receptance.weight)
        torch.testing.assert_close(model.model.layers[i].attn.r_proj.linear.weight,
                                   rwkv6.rwkv.blocks[i].attention.receptance.weight)
        print(f"rwkv6.rwkv.blocks{i}.attention.time_decay_w1  -> model.model.layers{i}.attn.w_proj.linear.lora0.weight")
        model.model.layers[i].attn.w_proj.linear.lora[0].weight.data.copy_(rwkv6.rwkv.blocks[i].attention.time_decay_w1.t())
        torch.testing.assert_close(model.model.layers[i].attn.w_proj.linear.lora[0].weight,
                                   rwkv6.rwkv.blocks[i].attention.time_decay_w1.t())
        print(f"rwkv6.rwkv.blocks{i}.attention.time_decay_w2  -> model.model.layers{i}.attn.w_proj.linear.lora2.weight")
        model.model.layers[i].attn.w_proj.linear.lora[2].weight.data.copy_(rwkv6.rwkv.blocks[i].attention.time_decay_w2.t())
        torch.testing.assert_close(model.model.layers[i].attn.w_proj.linear.lora[2].weight,
                                   rwkv6.rwkv.blocks[i].attention.time_decay_w2.t())
        print(f"rwkv6.rwkv.blocks{i}.attention.time_decay  -> model.model.layers{i}.attn.w_proj.linear.lora2.bias")
        model.model.layers[i].attn.w_proj.linear.lora[2].bias.data.copy_(rwkv6.rwkv.blocks[i].attention.time_decay.view(-1))
        torch.testing.assert_close(model.model.layers[i].attn.w_proj.linear.lora[2].bias,
                                   rwkv6.rwkv.blocks[i].attention.time_decay.view(-1))

        print(f"rwkv6.rwkv.blocks{i}.attention.key.weight -> model.model.layers.{i}.attn.k_proj.linear.weight")
        model.model.layers[i].attn.k_proj.linear.weight.data.copy_(rwkv6.rwkv.blocks[i].attention.key.weight)
        torch.testing.assert_close(model.model.layers[i].attn.k_proj.linear.weight,
                                   rwkv6.rwkv.blocks[i].attention.key.weight)
        print(f"rwkv6.rwkv.blocks{i}.attention.value.weight -> model.model.layers.{i}.attn.v_proj.linear.weight")
        model.model.layers[i].attn.v_proj.linear.weight.data.copy_(rwkv6.rwkv.blocks[i].attention.value.weight)
        torch.testing.assert_close(model.model.layers[i].attn.v_proj.linear.weight,
                                   rwkv6.rwkv.blocks[i].attention.value.weight)
        print(f"rwkv6.rwkv.blocks{i}.attention.gate.weight -> model.model.layers.{i}.attn.g_proj.linear.weight")
        model.model.layers[i].attn.g_proj.linear.weight.data.copy_(rwkv6.rwkv.blocks[i].attention.gate.weight)
        torch.testing.assert_close(model.model.layers[i].attn.g_proj.linear.weight,
                                   rwkv6.rwkv.blocks[i].attention.gate.weight)
        print(f"rwkv6.rwkv.blocks{i}.attention.time_faaaa -> model.model.layers.{i}.attn.bonus")
        bonus = rwkv6.rwkv.blocks[i].attention.time_faaaa.view(config.num_heads, -1)
        model.model.layers[i].attn.bonus.data.copy_(bonus)
        torch.testing.assert_close(model.model.layers[i].attn.bonus, bonus)

        if model.model.layers[i].attn.g_norm.weight is not None:
            print(f"rwkv6.rwkv.blocks{i}.attention.ln_x.weight -> model.model.layers[i].attn.g_norm.weight")
            model.model.layers[i].attn.g_norm.weight.data.copy_(rwkv6.rwkv.blocks[i].attention.ln_x.weight)
            torch.testing.assert_close(model.model.layers[i].attn.g_norm.weight, rwkv6.rwkv.blocks[i].attention.ln_x.weight)
        if model.model.layers[i].attn.g_norm.bias is not None:
            print(f"rwkv6.rwkv.blocks{i}.attention.ln_x.bias -> model.model.layers[i].attn.g_norm.bias")
            model.model.layers[i].attn.g_norm.bias.data.copy_(rwkv6.rwkv.blocks[i].attention.ln_x.bias)
            torch.testing.assert_close(model.model.layers[i].attn.g_norm.bias, rwkv6.rwkv.blocks[i].attention.ln_x.bias)
        model.model.layers[i].attn.g_norm.eps = rwkv6.rwkv.blocks[i].attention.ln_x.eps

        print(f"rwkv6.rwkv.blocks{i}.attention.output.weight -> model.model.layers.{i}.attn.o_proj.weight")
        model.model.layers[i].attn.o_proj.weight.data.copy_(rwkv6.rwkv.blocks[i].attention.output.weight)
        torch.testing.assert_close(model.model.layers[i].attn.o_proj.weight, rwkv6.rwkv.blocks[i].attention.output.weight)

        if model.model.layers[i].ffn_norm.weight is not None:
            print(f"rwkv6.rwkv.blocks{i}.ln2.weight -> model.model.layers{i}.ffn_norm.weight")
            model.model.layers[i].ffn_norm.weight.data.copy_(rwkv6.rwkv.blocks[i].ln2.weight)
            torch.testing.assert_close(model.model.layers[i].ffn_norm.weight, rwkv6.rwkv.blocks[i].ln2.weight)
        if model.model.layers[i].ffn_norm.bias is not None:
            print(f"rwkv6.rwkv.blocks{i}.ln2.bias -> model.model.layers{i}.ffn_norm.bias")
            model.model.layers[i].ffn_norm.bias.data.copy_(rwkv6.rwkv.blocks[i].ln2.bias)
            torch.testing.assert_close(model.model.layers[i].ffn_norm.bias, rwkv6.rwkv.blocks[i].ln2.bias)
        model.model.layers[i].ffn_norm.eps = rwkv6.rwkv.blocks[i].ln2.eps

        print(f"rwkv6.rwkv.blocks{i}.feed_forward.key.weight -> model.model.layers.{i}.ffn.key.linear.weight")
        model.model.layers[i].ffn.key.linear.weight.data.copy_(rwkv6.rwkv.blocks[i].feed_forward.key.weight)
        torch.testing.assert_close(model.model.layers[i].ffn.key.linear.weight,
                                   rwkv6.rwkv.blocks[i].feed_forward.key.weight)
        print(f"rwkv6.rwkv.blocks{i}.feed_forward.time_maa_k -> model.model.layers.{i}.ffn.key.mu")
        model.model.layers[i].ffn.key.mu.data.copy_(rwkv6.rwkv.blocks[i].feed_forward.time_maa_k.view(-1))
        torch.testing.assert_close(model.model.layers[i].ffn.key.mu,
                                   rwkv6.rwkv.blocks[i].feed_forward.time_maa_k.view(-1))

        print(f"rwkv6.rwkv.blocks{i}.feed_forward.value.weight -> model.model.layers.{i}.ffn.value.weight")
        model.model.layers[i].ffn.value.weight.data.copy_(rwkv6.rwkv.blocks[i].feed_forward.value.weight)
        torch.testing.assert_close(model.model.layers[i].ffn.value.weight,
                                   rwkv6.rwkv.blocks[i].feed_forward.value.weight)

        print(f"rwkv6.rwkv.blocks{i}.feed_forward.receptance.weight -> model.model.layers.{i}.ffn.receptance.linear.weight")
        model.model.layers[i].ffn.receptance.linear.weight.data.copy_(rwkv6.rwkv.blocks[i].feed_forward.receptance.weight)
        torch.testing.assert_close(model.model.layers[i].ffn.receptance.linear.weight,
                                   rwkv6.rwkv.blocks[i].feed_forward.receptance.weight)
        print(f"rwkv6.rwkv.blocks{i}.feed_forward.time_maa_r -> model.model.layers.{i}.ffn.receptance.mu")
        model.model.layers[i].ffn.receptance.mu.data.copy_(rwkv6.rwkv.blocks[i].feed_forward.time_maa_r.view(-1))
        torch.testing.assert_close(model.model.layers[i].ffn.receptance.mu,
                                   rwkv6.rwkv.blocks[i].feed_forward.time_maa_r.view(-1))

    if model.model.norm.weight is not None:
        print("rwkv6.rwkv.ln_out.weight -> model.model.norm.weight")
        model.model.norm.weight.data.copy_(rwkv6.rwkv.ln_out.weight)
        torch.testing.assert_close(model.model.norm.weight, rwkv6.rwkv.ln_out.weight)
    if model.model.norm.bias is not None:
        print("rwkv6.rwkv.ln_out.bias -> model.model.norm.bias")
        model.model.norm.bias.data.copy_(rwkv6.rwkv.ln_out.bias)
        torch.testing.assert_close(model.model.norm.bias, rwkv6.rwkv.ln_out.bias)
    model.model.norm.eps = rwkv6.rwkv.ln_out.eps

    if not model.config.tie_word_embeddings:
        print("rwkv6.rwkv.head.weight -> model.lm_head.weight")
        model.lm_head.weight.data.copy_(rwkv6.head.weight)
        torch.testing.assert_close(model.lm_head.weight, rwkv6.head.weight)

    print(f"Saving converted model \n{model}\n to {output} ...")
    model.save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='RWKV/rwkv-6-world-7b')
    parser.add_argument("--config", default='configs/rwkv6_7B.json')
    parser.add_argument("--output", default='converted/rwkv6-7B')
    args = parser.parse_args()
    convert(args.model, args.config, args.output)
