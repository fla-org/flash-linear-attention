# -*- coding: utf-8 -*-

import argparse
import json

try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError:
    evaluator = None
    HFLM = None
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from fla.models.rwkv7 import RWKV7ForCausalLM


if __name__ == '__main__':
    def test_rwkv7_lm_eval(model, tokenizer, task_names=["lambada_openai"]):
        tokenizer1 = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<|endoftext|>",
            pad_token="<|padding|>"
        )
        hf_model = HFLM(pretrained=model, tokenizer=tokenizer1)
        results = evaluator.simple_evaluate(
            model=hf_model,
            tasks=task_names,
            batch_size=1,
        )
        # {
        # "lambada_openai": {
        #     "perplexity,none": 14.457888475382047,
        #     "perplexity_stderr,none": 0.4455143803996477,
        #     "acc,none": 0.4585678245682127,
        #     "acc_stderr,none": 0.006942020515885241,
        #     "alias": "lambada_openai"
        # }
        # }
        print(json.dumps(results['results'], indent=2))

    # official results:
    # pile 168M: lambada_openai ppl 14.2 acc 45.6%
    # pile 421M: lambada_openai ppl 8.14 acc 55.6%
    # pile 1.47B: lambada_openai ppl 5.04 acc 64.9%
    parser = argparse.ArgumentParser(description='Convert RWKV7')
    parser.add_argument('model', type=str, help='path to model')
    parser.add_argument('tokenizer', type=str, help='path to tokenizer')
    parser.add_argument('--tasks', type=str, nargs='*',
                        default=['lambada_openai'])
    args = parser.parse_args()

    model = RWKV7ForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="cuda",
    ).half().eval()
    tokenizer = Tokenizer.from_file(args.tokenizer)

    test_rwkv7_lm_eval(model, tokenizer, task_names=["lambada_openai"])
