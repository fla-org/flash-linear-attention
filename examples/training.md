<div align="center">

# 🔥 Flame: Flash Linear Attention Made Easy

</div>

Welcome to 🔥 `flame`, a minimal and efficient framework built on `torchtitan` for training Flash Linear Attention (FLA) models with blazing efficiency.

This guide will walk you through training GLA models while demonstrating `flame`'s flexibility to extend to other FLA architectures.

## Setup

To get started, clone the `flame` repository and install the required dependencies:

```bash
git clone https://github.com/fla-org/flame.git
cd flame
pip install .
```

`flame` includes `fla` and `torchtitan` as submodules. After installation, initialize and update the submodules using:
```sh
git submodule update --init --recursive
```

## Preparing the dataset

Unlike the [legacy codebase](legacy/training), which required extensive pre-processing,
`flame` streamlines dataset handling with smart on-the-fly processing.

For most datasets:
```py
from datasets import load_dataset

# Load fineweb-edu with parallel processing
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=64)
```

For SlimPajama-627B (used in [GLA paper](https://proceedings.mlr.press/v235/yang24ab.html)):
```bash
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B --depth 1
```

## Training from scratch

To train your 340M model from scratch, execute the following command:

```sh
bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/gla-340M-10B/batch32.seqlen2048.warmup1024.update1.steps20480.lr3e-4 \
  --model.config configs/gla_340M.json \
  --model.tokenizer_path fla-hub/gla-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 32 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 1
```

We provide several [config files](https://github.com/fla-org/flame/tree/main/configs) in the `flame` repository for different models.
By default, the learning rate is set to `3e-4` with a cosine scheduler.
Other schedulers, such as WSD (wsd), are also supported. For a detailed explanation of all parameters, run:
```sh
bash train.sh -h
```

`flame` supports resuming interrupted training from the last checkpoint.
If a checkpoint exists, the training process will automatically resume from it. Alternatively, you can resume from a specific step by specifying `--checkpoint.load_step <step_number>`.

The training progress is logged using `wandb` for easy monitoring.

## Continual Pretraining

`flame` supports continual training from a pretrained checkpoint.
Below, we provide an example of how to finetune Mistral-7B to GLA.
You can follow similar steps to reproduce the results in the [GSA paper](https://arxiv.org/abs/2409.07146):

1. Initialize a brand-new GLA-7B model from the config and copy the mathced pretrained weights from Mistral-7B:
```bash
cd ../utils
python convert_from_llama.py \
  --model mistralai/Mistral-7B-v0.1 \
  --config <path-to-gsa-config> \
  --output <path-to-output-folder>
cd -
```

2. Convert the 🤗 format model back into DCP format.
```bash
python -m flame.utils.convert_hf_to_dcp --model <path-to-output-folder> --checkpoint <path-to-output-folder/checkpoint/step-0>
```
Here, <path-to-output-folder> is the directory where your distributed checkpoints will be stored. The checkpoint is intentionally saved at <step-0> within the checkpoint folder to ensure it is loadable by flame during the initial training step, similar to how a seed checkpoint is handled.

3. Directly launch training from the converted checkpoint:
```sh
bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder <path-to-output-folder> \
  --model.config <path-to-gsa-config> \
  --model.tokenizer_path fla-hub/gla-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-5 \
  --lr_scheduler.warmup_steps 512 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 4 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 10240 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --checkpoint.interval 1024 \
  --checkpoint.load_step 0 \
  --metrics.log_freq 1
```

Finetuning on a single node may not be the most efficient approach.
If you have access to multi-node GPUs, consider leveraging them for optimal performance.
This process is straightforward and well-documented in the PyTorch [docs](https://pytorch.org/docs/stable/elastic/run.html).

Simply set the environment variables `MASTER_ADDR=<ip>` and `MASTER_PORT=<port>` before running the training script across all nodes. If you're using a job scheduler like Slurm, it will handle these variables for you.
`torchtitan` provides a [Slurm script](https://github.com/pytorch/torchtitan/blob/main/multinode_trainer.slurm) for multi-node training.
