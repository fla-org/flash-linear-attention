<div align="center">

# 🔥 Flame: Flash Linear Attention Made Easy

</div>

> [!IMPORTANT]
> The `flame` project has been migrated to a new project built on torchtitan.  
> Please visit the [new repository](https://github.com/fla-org/flame) for details and updates.
>
> The code here is now **archived as legacy**, and no future updates will be synchronized here.

A minimal framework for training FLA models, whether from scratch or through finetuning.

Built on the robust infrastructure of 🤗, `flame` enables you to train large language models with just a few lines of code:
we use `datasets` for data processing, `transformers` for model definitions, and `accelerate`[^1] for seamless distributed training.

In this README, we will guide you through the process of using `flame` to train GLA models.

## Setup

To get started, you'll need to install the required packages.
Both `fla` and `flame` have minimal dependencies.
Clone the `fla` repository and install the necessary packages as follows:

```bash
git clone https://github.com/sustcsonglin/flash-linear-attention.git
pip install .
pip install accelerate
```

> [!CAUTION]
> The 🤗 `tokenizers` have some [memory leak issues](https://github.com/huggingface/tokenizers/issues/1539) when processing very long documents.
> To address this, please ensure you install `tokenizers>=0.20.4`.

## Preprocessing

Before training, you need to download and pre-tokenize your dataset.
We provide a straightforward script for this.
For instance, to tokenize a 10B sample of the `fineweb-edu` dataset, run:

```bash
python preprocess.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --name sample-10BT \
  --split train \
  --context_length 2048
```

This will cache the processed dataset at `data/HuggingFaceFW/fineweb-edu/sample-10BT/train`.

GLA utilizes a subset of Slimpajama for pretraining [in the paper](https://proceedings.mlr.press/v235/yang24ab.html).
Given the size of the dataset, the fastest way to download it is using `git lfs` (refer to [this issue](https://huggingface.co/datasets/cerebras/SlimPajama-627B/discussions/2)).
```bash
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B --depth 1
python preprocess.py \
  --dataset SlimPajama-627B \
  --split train \
  --context_length 2048
```

## Training from scratch

To train your 340M model from scratch, execute the following command:

```bash
bash train.sh \
  type=gla \
  lr=3e-4 \
  scheduler=cosine_with_min_lr \
  batch=32 \
  update=1 \
  warmup=1024 \
  steps=20480 \
  context=2048 \
  gpus=8 \  
  nodes=1 \
  path=exp/gla-340M-10B \
  project=fla \
  model=configs/gla_340M.json \
  data=HuggingFaceFW/fineweb-edu \
  name=sample-10BT \
  cache=data/HuggingFaceFW/fineweb-edu/sample-10BT/train
```

Key parameters:

|           | Description                   | Default              |
| :-------- | :---------------------------- | -------------------- |
| lr        | `learning_rate`               | `3e-4`               |
| scheduler | `lr_scheduler_type`           | `cosine_with_min_lr` |
| batch     | `batch_size`                  | `32`                 |
| update    | `gradient_accumulation_steps` | `1`                  |
| context   | `context_length`              | `2048`               |
| gpus      | `num_gpus_per_node`           | `8`                  |
| nodes     | `num_nodes`                   | `1`                  |
| warmup    | `warmup_steps`                | `1024`               |
| steps     | `max_steps`                   | `20480`              |

The learning rate is set to `3e-4` by default, equipped with a cosine scheduler.
Other scheduler types like WSD (`warmup_stable_decay`)[^2] are also supported.

The total number of tokens processed per batch, referred to as `global_batch_size`, is calculated as
`batch_size × gradient_accumulation_steps × context_length × num_gpus_per_node × num_nodes`.
For instance, in the 340M model example, the `global_batch_size` calculates to $32 \times 1 \times 2048 \times 8 \times 1 = 524,288$ (0.5M tokens).

The `warmup_steps` parameter indicates the number of steps for the learning rate warmup phase, while `max_steps` represents the maximum number of training steps.
Each step processes `global_batch_size` tokens.
Consequently, `512` and `20480` correspond to processing 0.5B and 10B tokens, respectively.

:warning: Monitor the value of `global_batch_size`, `warmup_steps`, and `max_steps` carefully when modifying any of the hyperparameters!!

`flame` also supports resuming interrupted training by specifying the checkpoint path.
Simply use the following command:

```bash
bash train.sh \
  type=gla \
  lr=3e-4 \
  steps=20480 \
  batch=32 \
  update=1 \
  warmup=1024 \
  context=2048 \
  gpus=8 \  
  nodes=1 \
  path=exp/gla-340M-10B \
  project=fla \
  model=configs/gla_340M.json \
  data=HuggingFaceFW/fineweb-edu \
  name=sample-10BT \
  cache=data/HuggingFaceFW/fineweb-edu/sample-10BT/train \
  checkpoint=exp/gla-340M-10B/checkpoint-8192
```

You can also use `wandb` to monitor your training process effectively.

![wandb](https://github.com/user-attachments/assets/05ca031c-1cae-41c9-bfcb-5b6b6d0df729)

## Continual Pretraining

`flame` supports continual training from a pretrained checkpoint.
Below, we provide an example of how to finetune Mistral-7B to GLA.
You can follow similar steps to reproduce the results in the [GSA paper](https://arxiv.org/abs/2409.07146):

1. Initialize a brand-new GLA-7B model from the config and copy the mathced pretrained weights from Mistral-7B:
```bash
cd ../utils
python convert_from_llama.py \
  --model mistralai/Mistral-7B-v0.1 \
  --config ../training/configs/gla_7B.json \
  --output ../training/converted/gla-7B
cd -
```

2. Directly launch training from the converted checkpoint:
```bash
bash train.sh \
  type=gla \
  lr=3e-5 \
  steps=10240 \
  batch=4 \
  update=8 \
  warmup=512 \
  context=2048 \
  path=exp/gla-7B-20B \
  project=fla \
  model=converted/gla-7B \
  data=SlimPajama-627B \
  cache=data/SlimPajama-627B/train
```

Please be aware that finetuning on a single node may not be the most efficient approach.
If available, consider leveraging multi-node GPUs for optimal performance.
You can find guidance on how to launch a multi-node job in the [accelerate tutorial](https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh).

[^1]: The `accelerate` library supports various distributed frameworks, like `deepspeed` and `megatron` for large-scale training. We use `deepspeed` in our case.
[^2]: https://arxiv.org/abs/2404.06395
