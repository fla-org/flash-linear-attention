# Evaluation

We provide evaluation scripts for `fla` models with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), as well as a standalone perplexity evaluator.
Install `fla` following the [installation guide](../INSTALL.md), and run the commands below from the repository root.

* [lm-evaluation-harness](#lm-evaluation-harness)
  * [Single-GPU Evaluation](#single-gpu-evaluation)
  * [Multi-GPU Evaluation](#multi-gpu-evaluation)
  * [RULER Benchmarks](#ruler-benchmarks)
* [Perplexity](#perplexity)
  * [Quick Start](#quick-start)
  * [Evaluation Settings](#evaluation-settings)
  * [Scoring](#scoring)
  * [Results](#results)

## lm-evaluation-harness

The [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library allows you to easily perform (zero-shot) model evaluations.
Install `lm_eval` following [their instructions](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md).

### Single-GPU Evaluation

Run evaluation with:

```sh
MODEL='fla-hub/gla-1.3B-100B'
python -m evals.harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config
```

We've made `fla` compatible with hf-style evaluations, you can call [evals.harness](harness.py) to finish the evaluations.
The command above evaluates the GLA checkpoint on the tasks used in the GLA paper.

### Multi-GPU Evaluation

Use the Hugging Face Accelerate launcher for data-parallel evaluation, where each GPU loads a full copy of the model:

```sh
MODEL='fla-hub/gla-1.3B-100B'
accelerate launch -m evals.harness --model hf \
    --model_args pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa \
    --batch_size 64 \
    --num_fewshot 0 \
    --device cuda \
    --show_config \
    --trust_remote_code
```

### RULER Benchmarks

The RULER benchmarks are commonly used for evaluating model performance on long-context tasks.
You can evaluate `fla` models on RULER directly using `lm-evaluation-harness`. RULER is only available in a relatively recent version of `lm-evaluation-harness`, so make sure you have the latest version installed.

```sh
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
pip install -e './lm-evaluation-harness[ruler]'
```

For example, to evaluate contexts up to 32K tokens:

```sh
MODEL='fla-hub/gla-1.3B-100B'
OUTPUT='results/ruler'
accelerate launch -m evals.harness --model hf \
    --output_path $OUTPUT \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_hotpot,ruler_qa_squad \
    --model_args pretrained=$MODEL,dtype=bfloat16,max_length=32768,trust_remote_code=True \
    --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
    --batch_size 2 \
    --show_config \
    --trust_remote_code
```

If a GPU can't load a full copy of the model, please refer to [this link](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate) for FSDP settings.

> [!TIP]
> If you are using `lm-evaluation-harness` as an external library and can't find (almost) any tasks available, before calling `lm_eval.evaluate()` or `lm_eval.simple_evaluate()`, simply run the following to load the library's stock tasks:
> ```py
> from lm_eval.tasks import TaskManager
> TaskManager().initialize_tasks()
> ```

## Perplexity

[evals.ppl](ppl.py) evaluates next-token perplexity over fixed-length contexts and reports perplexity by position within each context.

### Quick Start

Install the dataset dependency:

```sh
pip install datasets
```

For example, to evaluate the GLA checkpoint on the PG19 test split and save the results:

```sh
python -m evals.ppl \
    --path fla-hub/gla-1.3B-100B \
    --data fla-hub/pg19 --split test \
    --block_size 28672 --bucket_size 2048 --batch_size 1 \
    --dtype bfloat16 --num_proc 4 \
    --output results/ppl.json
```

### Evaluation Settings

| Argument          | Default                 | Description                                                 |
|-------------------|-------------------------|-------------------------------------------------------------|
| `--path`          | `fla-hub/gla-1.3B-100B` | Hugging Face model ID or local checkpoint directory.        |
| `--data`          | `fla-hub/pg19`          | Dataset to evaluate.                                        |
| `--split`         | `train`                 | Dataset split; set this explicitly for held-out evaluation. |
| `--column_name`   | `text`                  | Dataset column containing text.                             |
| `--block_size`    | `28672`                 | Context length in tokens; must be at least 2.               |
| `--bucket_size`   | `2048`                  | Number of prediction positions per reported bucket.         |
| `--batch_size`    | `1`                     | Number of full contexts evaluated together.                 |
| `--device`        | Auto-detected           | Device on which to load and evaluate the model.             |
| `--dtype`         | `bfloat16`              | Model dtype: `float32`, `float16`, or `bfloat16`.           |
| `--num_proc`      | `32`                    | Number of dataset tokenization processes.                   |
| `--revision`      | `main`                  | Model and tokenizer revision.                               |
| `--data_revision` | `main`                  | Dataset revision.                                           |
| `--output`        | None                    | Optional path for JSON results and evaluation settings.     |

Use immutable commit hashes for `--revision` and `--data_revision` to reproduce a run.
For long contexts, reduce `--batch_size` to lower model memory use and `--bucket_size` to reduce the FP32 softmax workspace.
The model still materializes its full logits.

### Scoring

Documents are tokenized with the tokenizer's default special-token policy, then concatenated in dataset order without extra separators.
The token stream is divided into independent contexts of `block_size` tokens, and the final partial context is evaluated separately without padding.
Context is reset at every block boundary; there is no sliding window or recurrent state carried between blocks.

Each context of length `L` contributes `L-1` next-token predictions.
The first token is unscored, and no synthetic EOS target is appended after the last token.
EOS tokens present in the input are scored normally.
A one-token tail contributes no predictions; a dataset with fewer than two tokens raises `ValueError`.

Overall PPL and positional PPL use the same FP32 negative log-likelihoods, weighted by the number of scored predictions.
Buckets group predictions by their zero-based logit position within each context: bucket 0 covers positions `0` through `bucket_size-1`.
Models configured with fused linear cross entropy are supported, and evaluation runs with caching disabled.

### Results

The script prints overall PPL, positional PPL, and token counts.
With `--output`, it also saves a JSON file containing:

* `perplexity` and `block_perplexities`: overall and positional PPL; buckets with no predictions have `null` PPL.
* `total_nll`, `block_nlls`, and `block_tokens`: summed NLLs and per-bucket prediction counts.
* `total_tokens` and `total_input_tokens`: scored predictions and input tokens, respectively.
* `total_sentences`: the number of evaluated contexts, including any one-token tail.
* `config`, `model_commit`, `dataset_fingerprint`, and `versions`: evaluation settings, the resolved model commit when available, dataset fingerprint, and package versions.

> [!NOTE]
> Earlier versions appended a synthetic EOS only to positional statistics, used BF16 softmax for BF16 logits, and dropped the final partial context. Scores and token counts from those versions may differ from the corrected results.
