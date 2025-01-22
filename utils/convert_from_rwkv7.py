# -*- coding: utf-8 -*-

# scripts for converting pretrained hf model weights to fla style

import argparse
import re

import torch
from transformers import AutoModelForCausalLM

from fla.models.rwkv7 import RWKV7Config


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'


def convert(
    rwkv7: str,
    output: str
):
    weights = torch.load(rwkv7, weights_only=True)
    config = RWKV7Config()
    config.vocab_size = weights['emb.weight'].shape[0]  # 50304
    config.hidden_size = weights['blocks.0.ffn.key.weight'].shape[1]  # 768
    config.num_hidden_layers = 0
    while f'blocks.{config.num_hidden_layers}.ffn.key.weight' in weights:
        config.num_hidden_layers += 1
    config.decay_low_rank_dim = weights['blocks.0.att.w1'].shape[1]  # 64
    config.gate_low_rank_dim = weights['blocks.0.att.g1'].shape[1]  # 128
    config.a_low_rank_dim = weights['blocks.0.att.a1'].shape[1]  # 64
    try:
        config.v_low_rank_dim = weights['blocks.1.att.v1'].shape[1]  # 32
    except Exception:
        config.v_low_rank_dim = 32

    print(f"Creating model with config:\n{config}")
    model = AutoModelForCausalLM.from_config(config)
    print(model)
    model_dict = model.state_dict()
    model_names = [n for n in model_dict]

    unused_weights = [
        'model.layers.0.attn.v_lora.lora.0.weight',
        'model.layers.0.attn.v_lora.lora.2.weight',
        'model.layers.0.attn.v_lora.lora.2.bias'
    ]
    possible_absent_weights = ['model.layers.0.pre_norm.weight', 'model.layers.0.pre_norm.bias']

    def translate_into_fla(name):
        transposed = False
        emb_head = {
            'emb.weight': 'model.embeddings.weight',
            'ln_out.weight': 'model.norm.weight',
            'ln_out.bias': 'model.norm.bias',
            'head.weight': 'lm_head.weight'
        }
        proj = {
            'receptance': 'r_proj',
            'key': 'k_proj',
            'value': 'v_proj',
            'ln_x': 'g_norm',
            'output': 'o_proj',
        }
        if name in emb_head:
            return emb_head[name], False
        name_compo = name.split('.')
        assert name_compo[0] == 'blocks'
        name_compo[0] = 'model.layers'
        assert int(name_compo[1]) in range(config.num_hidden_layers)
        name_compo[2] = {
            'att': 'attn',
            'ffn': 'ffn',
            'ln0': 'pre_norm',
            'ln1': 'attn_norm',
            'ln2': 'ffn_norm'
        }[name_compo[2]]
        if re.match("[wvag][012]", name_compo[3]):
            typ, num = name_compo[3]
            name_compo[3] = f'{typ}_lora.lora.' + {
                '0': '2.bias',
                '1': '0.weight',
                '2': '2.weight',
            }[num]
            transposed |= (num in ['1', '2'])
        elif name_compo[2] == 'attn' and name_compo[3] in proj:
            name_compo[3] = proj[name_compo[3]]
        elif name_compo[2] == 'attn' and re.match('x_[rwkvag]', name_compo[3]):
            name_compo[3] = 'x_x'
        return '.'.join(name_compo), transposed

    x_x = {}
    for name in weights:
        fla_name, transposed = translate_into_fla(name)
        print(f"{name:32} -> {fla_name:50}, {transposed}")
        if re.match('.*att.x_[rwkvag]', name):
            x_x[name] = weights[name]
            if len(x_x) == 6:
                weight = torch.stack(list(x_x.values())).squeeze_()
                x_x = {}
            else:
                continue
        else:
            weight = weights[name] if not transposed else weights[name].T
        if re.match('.*[wva]0', name):
            weight.squeeze_()
        if re.match('.*att.[kr]_[k_a]', name):
            weight.squeeze_()
        if re.match('.*ffn.x_[xk]', name):
            weight.squeeze_()
        assert model_dict[fla_name].shape == weight.shape
        model_dict[fla_name].data.copy_(weight)
        model_names.remove(fla_name)

    print("unused parameters: ", model_names)
    for n in model_names:
        if not (n in unused_weights or n in possible_absent_weights):
            raise KeyError(n)

    print(f"Saving model to {output}")
    model.save_pretrained(output)
    model.config.save_pretrained(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--output")
    args = parser.parse_args()
    convert(args.model, args.output)
